"""Localhost HTTP server for browsing/deleting run reports under ``runs_dir``.

Browsers cannot delete folders from a ``file://`` page, so the hub opens the
reports index via this loopback server and the page POSTs to ``/api/runs/<id>/delete``.
Bug reports POST to ``/api/runs/<id>/bug`` to zip a run folder onto a network share.
Recording landmark edits POST to ``/api/runs/<id>/events/<n>/landmarks``
(also accepts optional primary-target index swaps).
Recording typed-text edits POST to ``/api/runs/<id>/events/<n>/text``.
Recording event deletes POST to ``/api/runs/<id>/events/<n>/delete``.
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
from src.recorder.analyze import instruction_for_text_input, rebuild_pointer_instruction
from src.recorder.models import (
    RecordedEvent,
    event_json_path,
    screenshot_path_for_event,
    screenshot_path_for_event_end,
)
from src.recorder.vision_context import (
    drag_end_vision,
    drag_end_yolo_suffix,
    format_drag_candidate_anchor,
    reorder_yolo_ocr_primary,
    vision_from_yolo_ocr,
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_DELETE_PATH_RE = re.compile(r"^/api/runs/([^/]+)/delete/?$")
_BUG_PATH_RE = re.compile(r"^/api/runs/([^/]+)/bug/?$")
_LANDMARKS_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/landmarks/?$"
)
_EVENT_DELETE_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/delete/?$"
)
_EVENT_TEXT_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/text/?$"
)
_TYPED_TEXT_MAX_LEN = 8192

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


def _drop_primary_from_hints(
    hints: list[NearbyHint],
    vision: dict[str, Any],
) -> list[NearbyHint]:
    """Omit landmarks that duplicate the primary target label."""
    candidates = vision.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return hints
    primary_label = format_drag_candidate_anchor(candidates[0])
    if not primary_label:
        return hints
    return [hint for hint in hints if hint.label != primary_label]


def _load_recording_event_kind(run_dir: Path, event_index: int) -> str:
    event_path = run_dir / "events" / f"event_{event_index:03d}.json"
    payload = read_json(event_path, {})
    if isinstance(payload, dict):
        kind = payload.get("kind")
        if isinstance(kind, str) and kind.strip():
            return kind.strip()
    return ""


def _rebuild_report_instructions(run_dir: Path, report: dict[str, Any]) -> list[str]:
    """Rebuild ``instructions`` / ``expected_outcomes`` from analysis files (preserves wait lines)."""
    events_dir = run_dir / "events"
    analysis_dir = run_dir / "analysis"
    event_paths = sorted(events_dir.glob("event_*.json")) if events_dir.is_dir() else []
    instructions: list[str] = []
    expected_outcomes: list[str | None] = []
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
            expected_outcomes.append(None)
        instruction = analysis.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            instructions.append(instruction.strip())
            outcome = analysis.get("expected_outcome")
            if isinstance(outcome, str) and outcome.strip():
                expected_outcomes.append(outcome.strip())
            else:
                expected_outcomes.append(None)
    report["instructions"] = instructions
    report["expected_outcomes"] = expected_outcomes
    return instructions


def _unlink_if_under_run(run_dir: Path, path: Path) -> None:
    try:
        resolved = path.resolve()
        resolved.relative_to(run_dir.resolve())
    except (OSError, ValueError):
        return
    if resolved.is_file():
        try:
            resolved.unlink()
        except OSError:
            pass


def _resolve_event_media_path(run_dir: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate


def _delete_recording_event_files(run_dir: Path, event_index: int, event: dict[str, Any]) -> None:
    for key in ("screenshot_path", "end_screenshot_path"):
        media = _resolve_event_media_path(run_dir, event.get(key))
        if media is not None:
            _unlink_if_under_run(run_dir, media)

    for path in (
        event_json_path(run_dir, event_index),
        run_dir / "analysis" / f"event_{event_index:03d}.json",
        screenshot_path_for_event(run_dir, event_index),
        screenshot_path_for_event_end(run_dir, event_index),
    ):
        _unlink_if_under_run(run_dir, path)

    yolo_dir = run_dir / "yolo_ocr"
    if yolo_dir.is_dir():
        prefix = f"event_{event_index:03d}"
        for path in yolo_dir.glob(f"{prefix}*"):
            if path.is_file():
                _unlink_if_under_run(run_dir, path)


def _remaining_recording_event_paths(run_dir: Path) -> list[Path]:
    events_dir = run_dir / "events"
    if not events_dir.is_dir():
        return []
    paths = [path for path in events_dir.glob("event_*.json") if path.is_file()]
    paths.sort(key=lambda path: path.name)
    return paths


def _rewrite_session_event_list(run_dir: Path) -> int:
    remaining_paths = _remaining_recording_event_paths(run_dir)
    remaining = len(remaining_paths)
    session_path = run_dir / "session.json"
    session = read_json(session_path, {})
    if not isinstance(session, dict):
        session = {}
    session["event_count"] = remaining
    session["events"] = [
        path.relative_to(run_dir).as_posix() for path in remaining_paths
    ]
    write_json(session_path, session)
    return remaining


def purge_recording_event_from_session(run_dir: Path, event_index: int) -> int:
    """Delete one event's files and update ``session.json``.

    Does not rebuild ``report.json`` or HTML. Returns the remaining event count.
    Raises ``ValueError`` when the event is missing or the index is invalid.
    """
    run_dir = Path(run_dir)
    if not isinstance(event_index, int) or event_index < 1:
        raise ValueError("invalid event index")

    event_path = event_json_path(run_dir, event_index)
    event = read_json(event_path, None)
    if not isinstance(event, dict):
        raise ValueError("event not found")

    _delete_recording_event_files(run_dir, event_index, event)
    return _rewrite_session_event_list(run_dir)


def sync_recording_events(run_dir: Path, events: list[RecordedEvent]) -> dict[str, Any]:
    """Make on-disk events/session match ``events`` (survivors after coalesce).

    Writes each survivor event JSON, deletes absorbed event files, and rewrites
    ``session.json``. Returns ``{"kept": ..., "purged": [...], "remaining": ...}``.
    """
    run_dir = Path(run_dir)
    (run_dir / "events").mkdir(parents=True, exist_ok=True)

    keep_indices = {int(event.index) for event in events}
    purged: list[int] = []
    events_dir = run_dir / "events"
    for path in list(events_dir.glob("event_*.json")):
        if not path.is_file():
            continue
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index", 0))
        except (TypeError, ValueError):
            continue
        if index not in keep_indices:
            _delete_recording_event_files(run_dir, index, raw)
            purged.append(index)

    for event in events:
        write_json(event_json_path(run_dir, event.index), event.to_dict())

    remaining = _rewrite_session_event_list(run_dir)
    purged.sort()
    return {"kept": len(events), "purged": purged, "remaining": remaining}


def delete_recording_event(
    runs_root: Path,
    run_id: str,
    event_index: int,
) -> dict[str, Any]:
    """Delete one recorded event and rebuild report/HTML artifacts.

    Returns ``{"event_index": ..., "remaining": ...}``. Raises ``ValueError``
    for invalid input / missing events.
    """
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    remaining = purge_recording_event_from_session(run_dir, event_index)

    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    report["recorded"] = remaining
    if "processed" in report:
        report["processed"] = remaining
    if "cached" in report:
        analysis_dir = run_dir / "analysis"
        cached = (
            len(list(analysis_dir.glob("event_*.json")))
            if analysis_dir.is_dir()
            else 0
        )
        report["cached"] = cached
    _rebuild_report_instructions(run_dir, report)
    write_json(report_path, report)

    write_recording_html_from_run(run_dir, update_index=True)
    return {"event_index": event_index, "remaining": remaining}


def _optional_int_index(value: Any, *, field_name: str) -> int | None:
    """Parse an optional primary-target index from a JSON body field."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid {field_name}")
    if value < 0:
        raise ValueError(f"invalid {field_name}")
    return value


def apply_recording_event_landmarks(
    runs_root: Path,
    run_id: str,
    event_index: int,
    *,
    selected: Any,
    selected_end: Any = None,
    primary_index: Any = None,
    primary_end_index: Any = None,
) -> dict[str, Any]:
    """Reformat one event instruction from selected landmarks/targets and persist.

    When ``primary_index`` / ``primary_end_index`` move a non-zero candidate to
    index 0, reorders the matching ``yolo_ocr`` file and rebuilds the base
    instruction from vision helpers before applying landmark checkboxes.

    Returns ``{"instruction": ..., "rebuilt": bool}``. Raises ``ValueError`` for
    invalid input.
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
        "triple_click",
        "right_click",
        "middle_click",
        "scroll",
        "drag",
        "hold",
    }:
        raise ValueError("event does not support landmarks")

    start_primary = _optional_int_index(primary_index, field_name="primary_index")
    end_primary = (
        _optional_int_index(primary_end_index, field_name="primary_end_index")
        if kind == "drag"
        else None
    )

    rebuilt = False
    if start_primary is not None and start_primary != 0:
        reorder_yolo_ocr_primary(run_dir, event_index, start_primary, suffix="")
        rebuilt = True

    if kind == "drag" and end_primary is not None and end_primary != 0:
        reorder_yolo_ocr_primary(
            run_dir,
            event_index,
            end_primary,
            suffix=drag_end_yolo_suffix(run_dir, event_index),
        )
        rebuilt = True

    start_hints = _hints_from_selected_payload(selected)
    end_hints = _hints_from_selected_payload(selected_end) if kind == "drag" else []

    if rebuilt:
        event_path = event_json_path(run_dir, event_index)
        event_payload = read_json(event_path, None)
        if not isinstance(event_payload, dict):
            raise ValueError("event not found")
        event = RecordedEvent.from_dict(event_payload)
        vision = vision_from_yolo_ocr(run_dir, event_index, suffix="")
        destination = drag_end_vision(run_dir, event_index) if kind == "drag" else {}
        rebuilt_instruction = rebuild_pointer_instruction(
            event,
            vision,
            destination,
            include_nearby=False,
        )
        if not rebuilt_instruction:
            raise ValueError("unable to rebuild instruction for selected target")
        instruction = rebuilt_instruction
        start_hints = _drop_primary_from_hints(start_hints, vision)
        if kind == "drag":
            end_hints = _drop_primary_from_hints(end_hints, destination)
    else:
        instruction = instruction.strip()

    new_instruction = apply_nearby_landmarks(
        instruction,
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
    if start_primary is not None:
        landmarks_payload["primary_index"] = 0 if start_primary != 0 else start_primary
    if kind == "drag" and end_primary is not None:
        landmarks_payload["primary_end_index"] = (
            0 if end_primary != 0 else end_primary
        )

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
    return {"instruction": new_instruction, "rebuilt": rebuilt}


def apply_recording_event_text(
    runs_root: Path,
    run_id: str,
    event_index: int,
    *,
    text: Any,
) -> dict[str, Any]:
    """Replace typed text for one ``text_input`` event and persist instruction artifacts.

    Returns ``{"text": ..., "instruction": ...}``. Raises ``ValueError`` for invalid input.
    """
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    if not isinstance(event_index, int) or event_index < 1:
        raise ValueError("invalid event index")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if len(text) > _TYPED_TEXT_MAX_LEN:
        raise ValueError("text is too long")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text is empty")

    event_path = event_json_path(run_dir, event_index)
    event_payload = read_json(event_path, None)
    if not isinstance(event_payload, dict):
        raise ValueError("event not found")
    kind = event_payload.get("kind")
    if kind != "text_input":
        raise ValueError("event does not support typed-text edits")

    previous_text = event_payload.get("text")
    event_payload["text"] = cleaned
    write_json(event_path, event_payload)

    instruction = instruction_for_text_input(cleaned)
    if instruction is None:
        raise ValueError("text is empty")

    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    analysis = read_json(analysis_path, None)
    if isinstance(analysis, dict):
        analysis["instruction"] = instruction
        resolution = analysis.get("text_resolution")
        if not isinstance(resolution, dict):
            resolution = {}
            analysis["text_resolution"] = resolution
        if "recorded_text" not in resolution:
            recorded = previous_text if isinstance(previous_text, str) else cleaned
            resolution["recorded_text"] = recorded
        resolution["resolved_text"] = cleaned
        resolution["source"] = "user"
        resolution["reason"] = "edited in recording_steps.html"
        write_json(analysis_path, analysis)

        report_path = run_dir / "report.json"
        report = read_json(report_path, {})
        if not isinstance(report, dict):
            report = {}
        _rebuild_report_instructions(run_dir, report)
        write_json(report_path, report)

    write_recording_html_from_run(run_dir, update_index=False)
    return {"text": cleaned, "instruction": instruction}


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
            event_text_match = _EVENT_TEXT_PATH_RE.fullmatch(path)
            event_delete_match = _EVENT_DELETE_PATH_RE.fullmatch(path)

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
                        primary_index=body.get("primary_index"),
                        primary_end_index=body.get("primary_end_index"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if event_text_match is not None:
                run_id = event_text_match.group(1)
                event_index_raw = event_text_match.group(2)
                try:
                    event_index = int(event_index_raw)
                    body = self._read_json_body()
                    result = apply_recording_event_text(
                        root,
                        run_id,
                        event_index,
                        text=body.get("text"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if event_delete_match is not None:
                run_id = event_delete_match.group(1)
                event_index_raw = event_delete_match.group(2)
                try:
                    event_index = int(event_index_raw)
                    result = delete_recording_event(root, run_id, event_index)
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
