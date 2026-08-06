"""Localhost HTTP server for browsing/deleting run reports under ``runs_dir``.

Browsers cannot delete folders from a ``file://`` page, so the hub opens the
reports index via this loopback server and the page POSTs to ``/api/runs/<id>/delete``.
Bug reports POST to ``/api/runs/<id>/bug`` to zip a run folder onto a network share.
Recording landmark edits POST to ``/api/runs/<id>/events/<n>/landmarks``.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.common.io_utils import read_json, write_json
from src.common.nearby_side import NearbyHint, apply_nearby_landmarks, normalize_nearby_hints
from src.common.session_html import write_recording_html_from_run, write_runs_index_html

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_DELETE_PATH_RE = re.compile(r"^/api/runs/([^/]+)/delete/?$")
_BUG_PATH_RE = re.compile(r"^/api/runs/([^/]+)/bug/?$")
_LANDMARKS_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/landmarks/?$"
)

# Default destination for "report bug" zip copies (Windows UNC share).
BUG_REPORT_SHARE_DIR = Path(r"\\192.168.0.9\Joseph\CUA-BUG")

_server_lock = threading.Lock()
_active_server: RunsReportServer | None = None


class RunsReportServer:
    """Serve ``runs_root`` on ``127.0.0.1`` and accept report delete / bug-zip POSTs."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("runs report server is not running")
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def is_running(self) -> bool:
        return (
            self._httpd is not None
            and self._thread is not None
            and self._thread.is_alive()
        )

    def start(self) -> str:
        if self.is_running():
            return self.base_url

        handler = _make_handler(self.runs_root)
        # Bind loopback only; port 0 picks a free ephemeral port.
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        httpd.daemon_threads = True
        self._httpd = httpd

        thread = threading.Thread(
            target=httpd.serve_forever,
            name="runs-report-server",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return self.base_url

    def stop(self) -> None:
        httpd = self._httpd
        self._httpd = None
        thread = self._thread
        self._thread = None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:
                pass
            try:
                httpd.server_close()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


def ensure_runs_report_server(runs_root: Path) -> RunsReportServer:
    """Return a running server for ``runs_root``, replacing any other active server."""
    global _active_server
    runs_root = Path(runs_root).resolve()
    with _server_lock:
        if (
            _active_server is not None
            and _active_server.runs_root == runs_root
            and _active_server.is_running()
        ):
            return _active_server
        if _active_server is not None:
            _active_server.stop()
            _active_server = None
        server = RunsReportServer(runs_root)
        server.start()
        _active_server = server
        return server


def stop_runs_report_server() -> None:
    global _active_server
    with _server_lock:
        if _active_server is not None:
            _active_server.stop()
            _active_server = None


def delete_run_report_folder(runs_root: Path, run_id: str) -> Path:
    """Delete one direct child run folder under ``runs_root`` and rebuild the index.

    Raises ``ValueError`` for invalid ids / path traversal / missing folders.
    """
    target = resolve_deletable_run_folder(runs_root, run_id)
    shutil.rmtree(target)
    write_runs_index_html(Path(runs_root).resolve())
    return target


def resolve_deletable_run_folder(runs_root: Path, run_id: str) -> Path:
    """Validate ``run_id`` and return the absolute path of a deletable run folder."""
    runs_root = Path(runs_root).resolve()
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run id")
    if Path(run_id).name != run_id:
        raise ValueError("invalid run id")

    target = (runs_root / run_id).resolve()
    try:
        target.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("run folder is outside runs root") from exc
    if target.parent != runs_root:
        raise ValueError("run folder must be a direct child of runs root")
    if not target.is_dir():
        raise ValueError("run folder not found")
    return target


def _hints_from_selected_payload(raw: Any) -> list[NearbyHint]:
    if not isinstance(raw, list):
        return []
    return normalize_nearby_hints(raw)


def _load_recording_event_kind(run_dir: Path, event_index: int) -> str:
    event_path = run_dir / "events" / f"event_{event_index:03d}.json"
    payload = read_json(event_path, {})
    if isinstance(payload, dict):
        kind = payload.get("kind")
        if isinstance(kind, str) and kind.strip():
            return kind.strip()
    return ""


def _rebuild_report_instructions(run_dir: Path, report: dict[str, Any]) -> list[str]:
    """Rebuild ``instructions`` from analysis files (preserves wait lines)."""
    events_dir = run_dir / "events"
    analysis_dir = run_dir / "analysis"
    event_paths = sorted(events_dir.glob("event_*.json")) if events_dir.is_dir() else []
    instructions: list[str] = []
    for event_path in event_paths:
        event = read_json(event_path, {})
        if not isinstance(event, dict):
            continue
        raw_index = event.get("index")
        if not isinstance(raw_index, int):
            continue
        analysis = read_json(analysis_dir / f"event_{raw_index:03d}.json", {})
        if not isinstance(analysis, dict):
            continue
        wait = analysis.get("wait_instruction")
        if isinstance(wait, str) and wait.strip():
            instructions.append(wait.strip())
        instruction = analysis.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            instructions.append(instruction.strip())
    report["instructions"] = instructions
    return instructions


def apply_recording_event_landmarks(
    runs_root: Path,
    run_id: str,
    event_index: int,
    *,
    selected: Any,
    selected_end: Any = None,
) -> dict[str, Any]:
    """Reformat one event instruction from selected landmarks and persist files.

    Returns ``{"instruction": ...}``. Raises ``ValueError`` for invalid input.
    """
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    if not isinstance(event_index, int) or event_index < 1:
        raise ValueError("invalid event index")

    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    analysis = read_json(analysis_path, None)
    if not isinstance(analysis, dict):
        raise ValueError("analysis not found")
    instruction = analysis.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction not found")

    kind = _load_recording_event_kind(run_dir, event_index)
    if kind not in {
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "scroll",
        "drag",
        "hold",
    }:
        raise ValueError("event does not support landmarks")

    start_hints = _hints_from_selected_payload(selected)
    end_hints = _hints_from_selected_payload(selected_end) if kind == "drag" else []
    new_instruction = apply_nearby_landmarks(
        instruction.strip(),
        start_hints,
        kind=kind,
        end_hints=end_hints if kind == "drag" else None,
    )

    landmarks_payload: dict[str, Any] = {
        "selected": [
            {"label": hint.label, "side": hint.side.value if hint.side else None}
            for hint in start_hints
        ]
    }
    if kind == "drag":
        landmarks_payload["selected_end"] = [
            {"label": hint.label, "side": hint.side.value if hint.side else None}
            for hint in end_hints
        ]

    analysis["instruction"] = new_instruction
    analysis["landmarks"] = landmarks_payload
    write_json(analysis_path, analysis)

    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    _rebuild_report_instructions(run_dir, report)
    write_json(report_path, report)

    write_recording_html_from_run(run_dir, update_index=False)
    return {"instruction": new_instruction}


def zip_run_report_to_bug_share(
    runs_root: Path,
    run_id: str,
    dest_dir: Path | None = None,
) -> Path:
    """Zip one run folder and copy the archive to the bug-report share.

    Returns the destination ``.zip`` path. Raises ``ValueError`` for invalid ids
    and ``OSError`` when the share/path is unreachable or not writable.
    """
    target = resolve_deletable_run_folder(runs_root, run_id)
    dest_root = Path(dest_dir) if dest_dir is not None else BUG_REPORT_SHARE_DIR
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"bug share not reachable: {dest_root} ({exc})") from exc

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_name = f"{run_id}_{stamp}.zip"
    dest_path = dest_root / dest_name

    with tempfile.TemporaryDirectory(prefix="cua-bug-") as tmp:
        archive_base = Path(tmp) / run_id
        zip_path = Path(
            shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=str(target.parent),
                base_dir=target.name,
            )
        )
        try:
            shutil.copy2(zip_path, dest_path)
        except OSError as exc:
            raise OSError(f"failed to copy zip to bug share: {dest_path} ({exc})") from exc

    return dest_path


def _make_handler(runs_root: Path) -> type[SimpleHTTPRequestHandler]:
    root = runs_root.resolve()

    class RunsReportHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            # Keep hub console quiet during normal browsing.
            return

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            delete_match = _DELETE_PATH_RE.fullmatch(path)
            bug_match = _BUG_PATH_RE.fullmatch(path)
            landmarks_match = _LANDMARKS_PATH_RE.fullmatch(path)

            if delete_match is not None:
                run_id = delete_match.group(1)
                try:
                    deleted = delete_run_report_folder(root, run_id)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "deleted": deleted.name})
                return

            if bug_match is not None:
                run_id = bug_match.group(1)
                try:
                    dest = zip_run_report_to_bug_share(root, run_id)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(
                    200,
                    {"ok": True, "run_id": run_id, "copied_to": str(dest)},
                )
                return

            if landmarks_match is not None:
                run_id = landmarks_match.group(1)
                event_index_raw = landmarks_match.group(2)
                try:
                    event_index = int(event_index_raw)
                    body = self._read_json_body()
                    result = apply_recording_event_landmarks(
                        root,
                        run_id,
                        event_index,
                        selected=body.get("selected"),
                        selected_end=body.get("selected_end"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            self._send_json(404, {"ok": False, "error": "not found"})

        def _read_json_body(self) -> dict[str, Any]:
            length_raw = self.headers.get("Content-Length", "0")
            try:
                length = int(length_raw)
            except ValueError as exc:
                raise ValueError("invalid content length") from exc
            if length < 0 or length > 1_000_000:
                raise ValueError("invalid content length")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid json body") from exc
            if not isinstance(payload, dict):
                raise ValueError("json body must be an object")
            return payload

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return RunsReportHandler
