"""Localhost HTTP server for browsing/deleting run reports under ``runs_dir``.

Browsers cannot delete folders from a ``file://`` page, so the hub opens the
reports index via this loopback server and the page POSTs to ``/api/runs/<id>/delete``.
Bug reports POST to ``/api/runs/<id>/bug`` to zip a run folder onto a network share.
Recording landmark edits POST to ``/api/runs/<id>/events/<n>/landmarks``
(also accepts optional primary-target index swaps).
Recording typed-text edits POST to ``/api/runs/<id>/events/<n>/text``.
Recording expected-outcome edits POST to ``/api/runs/<id>/events/<n>/expected_outcome``.
Recording event deletes POST to ``/api/runs/<id>/events/<n>/delete``.
Recording bulk event deletes POST to ``/api/runs/<id>/events/delete`` with
``{"event_indices": [1, 2, ...]}``.
Recording event inserts POST to ``/api/runs/<id>/events/add``.
Recording instruction edits POST to ``/api/runs/<id>/events/<n>/instruction``.
Recording character-target edits POST to ``/api/runs/<id>/events/<n>/char_target``.
Recording YOLO/OCR retry POST to ``/api/runs/<id>/events/<n>/yolo_ocr``.
Recording folder rename POST to ``/api/runs/<id>/rename``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cua_mcp.char_target import parse_char_target_instruction
from src.common.io_utils import read_json, write_json
from src.common.nearby_side import (
    NearbyHint,
    apply_nearby_landmarks,
    extract_nearby_hints_from_instruction,
    normalize_nearby_hints,
)
from src.common.script_helper import collect_recording_instructions
from src.common.session_html import (
    recording_event_json_paths,
    write_recording_html_from_run,
    write_runs_index_html,
)
from src.recorder.analyze import (
    instruction_for_key,
    instruction_for_scroll,
    instruction_for_text_input,
    rebuild_pointer_instruction,
    use_char_target_enabled,
)
from src.recorder.models import (
    POINTER_EVENT_KINDS,
    RecordedEvent,
    event_json_path,
    next_recording_event_index,
    screenshot_path_for_event,
    screenshot_path_for_event_end,
    utc_now_iso,
)
from src.recorder.vision_context import (
    drag_end_vision,
    drag_end_yolo_suffix,
    format_drag_candidate_anchor,
    primary_candidate_char_target,
    reorder_yolo_ocr_primary,
    run_pointer_event_yolo_ocr,
    vision_from_yolo_ocr,
)

# Allow Unicode folder names (CJK, spaces); reject path separators and Windows-illegal chars.
_RUN_ID_ILLEGAL_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RUN_ID_MAX_LEN = 191
_DELETE_PATH_RE = re.compile(r"^/api/runs/([^/]+)/delete/?$")
_BUG_PATH_RE = re.compile(r"^/api/runs/([^/]+)/bug/?$")
_RENAME_PATH_RE = re.compile(r"^/api/runs/([^/]+)/rename/?$")
_LANDMARKS_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/landmarks/?$"
)
_EVENT_DELETE_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/delete/?$"
)
_EVENTS_BULK_DELETE_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/delete/?$"
)
_EVENT_TEXT_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/text/?$"
)
_EVENT_EXPECTED_OUTCOME_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/expected_outcome/?$"
)
_EVENT_INSTRUCTION_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/instruction/?$"
)
_EVENT_CHAR_TARGET_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/char_target/?$"
)
_EVENT_YOLO_OCR_PATH_RE = re.compile(
    r"^/api/runs/([^/]+)/events/(\d+)/yolo_ocr/?$"
)
_EVENT_ADD_PATH_RE = re.compile(r"^/api/runs/([^/]+)/events/add/?$")
_TYPED_TEXT_MAX_LEN = 8192
_EXPECTED_OUTCOME_MAX_LEN = 8192
_INSTRUCTION_MAX_LEN = 8192
_ADDABLE_EVENT_KINDS = frozenset(
    {
        "click",
        "double_click",
        "right_click",
        "text_input",
        "key_press",
        "hotkey",
        "scroll",
        "wait",
        "condition",
        "manual",
    }
)
_CHAR_TARGET_EVENT_KINDS = frozenset(
    {
        "click",
        "double_click",
        "triple_click",
        "right_click",
        "middle_click",
        "hold",
    }
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


def _is_valid_run_id(run_id: str) -> bool:
    """Validate a direct-child folder name under the runs root."""
    if not isinstance(run_id, str):
        return False
    cleaned = run_id.strip()
    if not cleaned or cleaned in {".", ".."}:
        return False
    if len(cleaned) > _RUN_ID_MAX_LEN:
        return False
    if _RUN_ID_ILLEGAL_RE.search(cleaned):
        return False
    if Path(cleaned).name != cleaned:
        return False
    return True


def resolve_deletable_run_folder(runs_root: Path, run_id: str) -> Path:
    """Validate ``run_id`` and return the absolute path of a deletable run folder."""
    runs_root = Path(runs_root).resolve()
    if not _is_valid_run_id(run_id):
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


def rename_recording_folder(
    runs_root: Path,
    run_id: str,
    new_name: Any,
) -> dict[str, Any]:
    """Rename a recording folder under ``runs_root`` and rebuild HTML/index.

    Returns ``{"old_id": ..., "new_id": ...}``.
    """
    if not isinstance(new_name, str):
        raise ValueError("name must be a string")
    cleaned = new_name.strip()
    if not _is_valid_run_id(cleaned):
        raise ValueError("invalid run name")

    runs_root = Path(runs_root).resolve()
    source = resolve_deletable_run_folder(runs_root, run_id)
    if cleaned == source.name:
        return {"old_id": run_id, "new_id": cleaned}

    dest = (runs_root / cleaned).resolve()
    try:
        dest.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("run folder is outside runs root") from exc
    if dest.parent != runs_root:
        raise ValueError("run folder must be a direct child of runs root")
    if dest.exists():
        raise ValueError("a folder with that name already exists")

    source.rename(dest)
    write_recording_html_from_run(dest, update_index=True)
    return {"old_id": run_id, "new_id": cleaned}


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
    instructions, expected_outcomes = collect_recording_instructions(run_dir)
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


def _rewrite_session_event_list(
    run_dir: Path,
    event_paths: list[Path] | None = None,
) -> int:
    if event_paths is None:
        event_paths = recording_event_json_paths(run_dir)
    remaining = len(event_paths)
    session_path = run_dir / "session.json"
    session = read_json(session_path, {})
    if not isinstance(session, dict):
        session = {}
    session["event_count"] = remaining
    session["events"] = [
        path.relative_to(run_dir).as_posix() for path in event_paths
    ]
    write_json(session_path, session)
    return remaining


def purge_recording_events_from_session(
    run_dir: Path,
    event_indices: list[int],
) -> int:
    """Delete one or more events' files and update ``session.json``.

    Does not rebuild ``report.json`` or HTML. Returns the remaining event count.
    Raises ``ValueError`` when indices are empty/invalid or any event is missing.
    """
    run_dir = Path(run_dir)
    if not event_indices:
        raise ValueError("event_indices is empty")

    unique: list[int] = []
    seen: set[int] = set()
    for raw in event_indices:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
            raise ValueError("invalid event index")
        if raw in seen:
            continue
        seen.add(raw)
        unique.append(raw)

    pending: list[tuple[int, dict[str, Any]]] = []
    for event_index in unique:
        event_path = event_json_path(run_dir, event_index)
        event = read_json(event_path, None)
        if not isinstance(event, dict):
            raise ValueError(f"event not found: {event_index}")
        pending.append((event_index, event))

    for event_index, event in pending:
        _delete_recording_event_files(run_dir, event_index, event)
    return _rewrite_session_event_list(run_dir)


def purge_recording_event_from_session(run_dir: Path, event_index: int) -> int:
    """Delete one event's files and update ``session.json``.

    Does not rebuild ``report.json`` or HTML. Returns the remaining event count.
    Raises ``ValueError`` when the event is missing or the index is invalid.
    """
    return purge_recording_events_from_session(run_dir, [event_index])


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

    remaining = _rewrite_session_event_list(
        run_dir,
        [event_json_path(run_dir, event.index) for event in events],
    )
    purged.sort()
    return {"kept": len(events), "purged": purged, "remaining": remaining}


def _rebuild_recording_after_event_purge(run_dir: Path, remaining: int) -> None:
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


def delete_recording_events(
    runs_root: Path,
    run_id: str,
    event_indices: list[int],
) -> dict[str, Any]:
    """Delete one or more recorded events and rebuild report/HTML once.

    Returns ``{"event_indices": [...], "deleted": n, "remaining": ...}``.
    Raises ``ValueError`` for invalid input / missing events.
    """
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    unique: list[int] = []
    seen: set[int] = set()
    for raw in event_indices:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
            raise ValueError("invalid event index")
        if raw in seen:
            continue
        seen.add(raw)
        unique.append(raw)
    remaining = purge_recording_events_from_session(run_dir, unique)
    _rebuild_recording_after_event_purge(run_dir, remaining)
    return {
        "event_indices": unique,
        "deleted": len(unique),
        "remaining": remaining,
    }


def delete_recording_event(
    runs_root: Path,
    run_id: str,
    event_index: int,
) -> dict[str, Any]:
    """Delete one recorded event and rebuild report/HTML artifacts.

    Returns ``{"event_index": ..., "remaining": ...}``. Raises ``ValueError``
    for invalid input / missing events.
    """
    result = delete_recording_events(runs_root, run_id, [event_index])
    return {"event_index": event_index, "remaining": result["remaining"]}


def _optional_stripped_str(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    return cleaned or None


def _parse_after_event_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid after_event_index")
    if value < 0:
        raise ValueError("invalid after_event_index")
    return value


def _parse_hotkey_keys(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(",", "+").split("+") if item.strip()]
    elif isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("keys must be a string or list")
    return parts or None


def _parse_scroll_delta(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid scroll_delta")
    if value == 0:
        raise ValueError("scroll_delta must be non-zero")
    return value


def _parse_duration_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid duration_seconds")
    if value <= 0:
        raise ValueError("duration_seconds must be positive")
    return float(value)


def _parse_condition_presence(value: Any) -> str:
    if value is None:
        return "has"
    if not isinstance(value, str):
        raise ValueError("invalid presence")
    cleaned = value.strip().lower()
    if cleaned in {"has", "有", "visible", "exists"}:
        return "has"
    if cleaned in {"missing", "沒有", "not_visible", "absent", "none"}:
        return "missing"
    raise ValueError("presence must be has or missing")


def _instruction_for_condition(
    *,
    presence: str,
    target: str,
    then_action: str | None,
) -> str:
    prefix = "如果畫面上沒有" if presence == "missing" else "如果畫面上有"
    if then_action:
        return f"{prefix}{target}，則{then_action}"
    return f"{prefix}{target}"


def _next_recording_event_index(run_dir: Path) -> int:
    return next_recording_event_index(run_dir)


def _copy_previous_screenshot(
    run_dir: Path,
    source_event: dict[str, Any] | None,
    dest_index: int,
) -> str:
    if not isinstance(source_event, dict):
        return ""
    src = _resolve_event_media_path(run_dir, source_event.get("screenshot_path"))
    if src is None or not src.is_file():
        raw_index = source_event.get("index")
        if isinstance(raw_index, int) and raw_index > 0:
            fallback = screenshot_path_for_event(run_dir, raw_index)
            if fallback.is_file():
                src = fallback
    if src is None or not src.is_file():
        return ""
    dest = screenshot_path_for_event(run_dir, dest_index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    try:
        return dest.relative_to(run_dir).as_posix()
    except ValueError:
        return str(dest)


def _clear_analysis_wait_instruction(run_dir: Path, event_index: int) -> None:
    """Drop a virtual ``wait_instruction`` so a real wait event is not duplicated."""
    if event_index < 1:
        return
    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    analysis = read_json(analysis_path, None)
    if not isinstance(analysis, dict) or "wait_instruction" not in analysis:
        return
    analysis.pop("wait_instruction", None)
    write_json(analysis_path, analysis)


def _instruction_for_added_event(
    *,
    kind: str,
    instruction: str | None,
    text: str | None,
    key: str | None,
    keys: list[str] | None,
    scroll_delta: int | None,
    duration_seconds: float | None,
    presence: str | None = None,
    then_action: str | None = None,
) -> str:
    if instruction:
        return instruction
    placeholder = RecordedEvent(index=1, timestamp_utc=utc_now_iso(), kind=kind)
    if kind == "text_input":
        generated = instruction_for_text_input(text or "")
    elif kind in {"key_press", "hotkey"}:
        placeholder.key = key
        placeholder.keys = keys
        generated = instruction_for_key(placeholder)
    elif kind == "scroll":
        placeholder.scroll_delta = scroll_delta
        generated = instruction_for_scroll(placeholder, {})
    elif kind == "wait" and duration_seconds is not None:
        generated = f"等待 {math.ceil(duration_seconds)} 秒"
    elif kind == "condition" and text:
        generated = _instruction_for_condition(
            presence=presence or "has",
            target=text,
            then_action=then_action,
        )
    else:
        generated = None
    if not generated:
        raise ValueError("instruction is empty")
    return generated


def add_recording_event(
    runs_root: Path,
    run_id: str,
    *,
    kind: Any,
    after_event_index: Any = None,
    instruction: Any = None,
    expected_outcome: Any = None,
    text: Any = None,
    key: Any = None,
    keys: Any = None,
    scroll_delta: Any = None,
    duration_seconds: Any = None,
    presence: Any = None,
    then_action: Any = None,
) -> dict[str, Any]:
    """Insert a user-authored recording event and rebuild report/HTML artifacts."""
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("kind is required")
    kind = kind.strip()
    if kind not in _ADDABLE_EVENT_KINDS:
        raise ValueError("unknown kind")

    after = _parse_after_event_index(after_event_index)
    ordered_paths = recording_event_json_paths(run_dir)
    ordered_rels = [path.relative_to(run_dir).as_posix() for path in ordered_paths]
    if after is not None and after > 0:
        after_rel = event_json_path(run_dir, after).relative_to(run_dir).as_posix()
        if after_rel not in ordered_rels:
            raise ValueError("after_event_index not found")

    typed_text = _optional_stripped_str(text, field_name="text")
    key_name = _optional_stripped_str(key, field_name="key")
    hotkey_keys = _parse_hotkey_keys(keys)
    parsed_scroll = _parse_scroll_delta(scroll_delta)
    parsed_duration = _parse_duration_seconds(duration_seconds)
    parsed_instruction = _optional_stripped_str(instruction, field_name="instruction")
    parsed_then = _optional_stripped_str(then_action, field_name="then_action")
    parsed_presence: str | None = None
    if parsed_instruction is not None and len(parsed_instruction) > _INSTRUCTION_MAX_LEN:
        raise ValueError("instruction is too long")
    if not isinstance(expected_outcome, (str, type(None))):
        raise ValueError("expected_outcome must be a string")
    if isinstance(expected_outcome, str) and len(expected_outcome) > _EXPECTED_OUTCOME_MAX_LEN:
        raise ValueError("expected_outcome is too long")
    cleaned_outcome = expected_outcome.strip() if isinstance(expected_outcome, str) else None
    cleaned_outcome = cleaned_outcome or None

    click_count: int | None = None
    button: str | None = None
    if kind == "click":
        click_count = 1
        button = "left"
        if not parsed_instruction:
            raise ValueError("instruction is empty")
    elif kind == "double_click":
        click_count = 2
        button = "left"
        if not parsed_instruction:
            raise ValueError("instruction is empty")
    elif kind == "right_click":
        button = "right"
        if not parsed_instruction:
            raise ValueError("instruction is empty")
    elif kind == "text_input":
        if not typed_text:
            raise ValueError("text is empty")
    elif kind == "key_press":
        if not key_name:
            raise ValueError("key is empty")
    elif kind == "hotkey":
        if not hotkey_keys:
            raise ValueError("keys is empty")
    elif kind == "scroll":
        if parsed_scroll is None:
            raise ValueError("scroll_delta is required")
    elif kind == "wait":
        if parsed_duration is None:
            raise ValueError("duration_seconds is required")
    elif kind == "condition":
        if not typed_text:
            raise ValueError("text is empty")
        parsed_presence = _parse_condition_presence(presence)
    elif kind == "manual":
        if not parsed_instruction:
            raise ValueError("instruction is empty")

    resolved_instruction = _instruction_for_added_event(
        kind=kind,
        instruction=parsed_instruction,
        text=typed_text,
        key=key_name,
        keys=hotkey_keys,
        scroll_delta=parsed_scroll,
        duration_seconds=parsed_duration,
        presence=parsed_presence,
        then_action=parsed_then,
    )
    if len(resolved_instruction) > _INSTRUCTION_MAX_LEN:
        raise ValueError("instruction is too long")

    new_index = _next_recording_event_index(run_dir)
    previous_event: dict[str, Any] | None = None
    if after is None and ordered_paths:
        previous_event = read_json(ordered_paths[-1], None)
        previous_event = previous_event if isinstance(previous_event, dict) else None
    elif after is not None and after > 0:
        previous_event = read_json(event_json_path(run_dir, after), None)
        previous_event = previous_event if isinstance(previous_event, dict) else None
    screenshot_rel = _copy_previous_screenshot(run_dir, previous_event, new_index)

    if kind == "wait":
        # Materializing a wait before the next step should replace any virtual wait line.
        follow_index: int | None = None
        if after is None:
            follow_index = None
        elif after == 0:
            if ordered_paths:
                first = read_json(ordered_paths[0], None)
                if isinstance(first, dict) and isinstance(first.get("index"), int):
                    follow_index = first["index"]
        else:
            after_rel = event_json_path(run_dir, after).relative_to(run_dir).as_posix()
            try:
                after_pos = ordered_rels.index(after_rel)
            except ValueError:
                after_pos = -1
            if 0 <= after_pos < len(ordered_rels) - 1:
                following = read_json(run_dir / ordered_rels[after_pos + 1], None)
                if isinstance(following, dict) and isinstance(following.get("index"), int):
                    follow_index = following["index"]
        if follow_index is not None:
            _clear_analysis_wait_instruction(run_dir, follow_index)

    event = RecordedEvent(
        index=new_index,
        timestamp_utc=utc_now_iso(),
        kind=kind,
        button=button,
        key=key_name,
        keys=hotkey_keys,
        text=typed_text,
        scroll_delta=parsed_scroll,
        duration_seconds=parsed_duration,
        click_count=click_count,
        screenshot_path=screenshot_rel,
    )
    event_payload = event.to_dict()
    if kind == "condition":
        event_payload["presence"] = parsed_presence or "has"
        if parsed_then:
            event_payload["then_action"] = parsed_then
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    write_json(event_json_path(run_dir, new_index), event_payload)
    analysis_payload: dict[str, Any] = {
        "event_index": new_index,
        "instruction": resolved_instruction,
        "expected_outcome": cleaned_outcome,
    }
    write_json(run_dir / "analysis" / f"event_{new_index:03d}.json", analysis_payload)

    new_rel = event_json_path(run_dir, new_index).relative_to(run_dir).as_posix()
    if after is None:
        ordered_rels.append(new_rel)
    elif after == 0:
        ordered_rels.insert(0, new_rel)
    else:
        after_rel = event_json_path(run_dir, after).relative_to(run_dir).as_posix()
        ordered_rels.insert(ordered_rels.index(after_rel) + 1, new_rel)
    remaining = _rewrite_session_event_list(
        run_dir,
        [run_dir / rel for rel in ordered_rels],
    )

    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    report["recorded"] = remaining
    if "processed" in report:
        report["processed"] = remaining
    if "cached" in report:
        analysis_dir = run_dir / "analysis"
        report["cached"] = (
            len(list(analysis_dir.glob("event_*.json"))) if analysis_dir.is_dir() else 0
        )
    _rebuild_report_instructions(run_dir, report)
    write_json(report_path, report)
    write_recording_html_from_run(run_dir, update_index=True)
    return {
        "event_index": new_index,
        "remaining": remaining,
        "instruction": resolved_instruction,
    }


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
            use_char_target=use_char_target_enabled(analysis),
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


def _replace_ocr_choice_text(
    resolution: dict[str, Any],
    cleaned: str,
    *,
    choice_text: str | None,
) -> None:
    """Update ``ocr_text`` / ``ocr_options`` so the selected OCR choice shows ``cleaned``."""
    previous_resolved = str(resolution.get("resolved_text") or "").strip()
    old_ocr = str(resolution.get("ocr_text") or "").strip()
    target = (choice_text or "").strip() or previous_resolved or old_ocr
    resolution["ocr_text"] = cleaned
    options = resolution.get("ocr_options")
    if not isinstance(options, list):
        resolution["ocr_options"] = [cleaned]
        return
    new_options: list[str] = []
    replaced = False
    seen: set[str] = set()
    for item in options:
        text = str(item or "").strip()
        if not text:
            continue
        if not replaced and target and text == target:
            text = cleaned
            replaced = True
        if text in seen:
            continue
        seen.add(text)
        new_options.append(text)
    if not replaced and cleaned and cleaned not in seen:
        new_options.insert(0, cleaned)
    resolution["ocr_options"] = new_options


def apply_recording_event_text(
    runs_root: Path,
    run_id: str,
    event_index: int,
    *,
    text: Any,
    source: Any = None,
    choice_text: Any = None,
) -> dict[str, Any]:
    """Replace typed text for one ``text_input`` event and persist instruction artifacts.

    When ``source`` is ``\"ocr\"`` or ``\"recorded\"``, also update that choice so the
    HTML OCR / 鍵盤 buttons stay in sync with the applied value.

    Returns ``{"text": ..., "instruction": ..., "source": ...}``. Raises ``ValueError``
    for invalid input.
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

    selected_source = ""
    if isinstance(source, str):
        selected_source = source.strip().lower()
    if selected_source and selected_source not in {"ocr", "recorded"}:
        raise ValueError("source must be ocr or recorded")
    selected_choice = choice_text.strip() if isinstance(choice_text, str) else None
    if selected_choice == "":
        selected_choice = None

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
        if selected_source == "recorded":
            resolution["recorded_text"] = cleaned
        elif selected_source == "ocr":
            _replace_ocr_choice_text(
                resolution,
                cleaned,
                choice_text=selected_choice,
            )
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
    return {
        "text": cleaned,
        "instruction": instruction,
        "source": selected_source or None,
    }


def apply_recording_event_expected_outcome(
    runs_root: Path,
    run_id: str,
    event_index: int,
    *,
    expected_outcome: Any,
) -> dict[str, Any]:
    """Replace verification text for one recorded event and persist artifacts.

    Empty / whitespace-only values clear ``expected_outcome`` (stored as ``null``).
    Returns ``{"expected_outcome": str|None}``. Raises ``ValueError`` for invalid input.
    """
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    if not isinstance(event_index, int) or event_index < 1:
        raise ValueError("invalid event index")
    if not isinstance(expected_outcome, str):
        raise ValueError("expected_outcome must be a string")
    if len(expected_outcome) > _EXPECTED_OUTCOME_MAX_LEN:
        raise ValueError("expected_outcome is too long")
    cleaned = expected_outcome.strip() or None

    event_path = event_json_path(run_dir, event_index)
    event_payload = read_json(event_path, None)
    if not isinstance(event_payload, dict):
        raise ValueError("event not found")

    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    analysis = read_json(analysis_path, None)
    if not isinstance(analysis, dict):
        raise ValueError("analysis not found")

    analysis["expected_outcome"] = cleaned
    write_json(analysis_path, analysis)

    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    _rebuild_report_instructions(run_dir, report)
    write_json(report_path, report)

    write_recording_html_from_run(run_dir, update_index=False)
    return {"expected_outcome": cleaned}


def apply_recording_event_instruction(
    runs_root: Path,
    run_id: str,
    event_index: int,
    *,
    instruction: Any,
) -> dict[str, Any]:
    """Replace the hub-script instruction for one recorded event and persist artifacts."""
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    if not isinstance(event_index, int) or event_index < 1:
        raise ValueError("invalid event index")
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string")
    if len(instruction) > _INSTRUCTION_MAX_LEN:
        raise ValueError("instruction is too long")
    cleaned = instruction.strip()
    if not cleaned:
        raise ValueError("instruction is empty")

    event_path = event_json_path(run_dir, event_index)
    event_payload = read_json(event_path, None)
    if not isinstance(event_payload, dict):
        raise ValueError("event not found")

    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    analysis = read_json(analysis_path, None)
    if not isinstance(analysis, dict):
        analysis = {"event_index": event_index}

    analysis["instruction"] = cleaned
    analysis["use_char_target"] = parse_char_target_instruction(cleaned) is not None
    write_json(analysis_path, analysis)

    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    _rebuild_report_instructions(run_dir, report)
    write_json(report_path, report)

    write_recording_html_from_run(run_dir, update_index=False)
    return {
        "instruction": cleaned,
        "use_char_target": analysis["use_char_target"],
    }


def apply_recording_event_char_target(
    runs_root: Path,
    run_id: str,
    event_index: int,
    *,
    use_char_target: Any,
) -> dict[str, Any]:
    """Rebuild one click instruction with or without the recorded character.

    Returns ``{"instruction": ..., "use_char_target": bool}``.
    Raises ``ValueError`` for invalid input.
    """
    if not isinstance(use_char_target, bool):
        raise ValueError("use_char_target must be a boolean")

    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    if not isinstance(event_index, int) or event_index < 1:
        raise ValueError("invalid event index")

    kind = _load_recording_event_kind(run_dir, event_index)
    if kind not in _CHAR_TARGET_EVENT_KINDS:
        raise ValueError("event does not support character targeting")

    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    analysis = read_json(analysis_path, None)
    if not isinstance(analysis, dict):
        raise ValueError("analysis not found")

    event_path = event_json_path(run_dir, event_index)
    event_payload = read_json(event_path, None)
    if not isinstance(event_payload, dict):
        raise ValueError("event not found")
    event = RecordedEvent.from_dict(event_payload)

    vision = vision_from_yolo_ocr(run_dir, event_index, suffix="")
    if primary_candidate_char_target(vision) is None:
        raise ValueError("clicked character is not available")

    rebuilt = rebuild_pointer_instruction(
        event,
        vision,
        None,
        include_nearby=False,
        use_char_target=use_char_target,
    )
    if not rebuilt:
        raise ValueError("unable to rebuild instruction for character target")

    landmarks_payload = analysis.get("landmarks")
    if isinstance(landmarks_payload, dict):
        start_hints = _hints_from_selected_payload(landmarks_payload.get("selected"))
    else:
        current = analysis.get("instruction")
        start_hints = extract_nearby_hints_from_instruction(
            current if isinstance(current, str) else ""
        )

    new_instruction = apply_nearby_landmarks(rebuilt, start_hints, kind=kind)
    analysis["instruction"] = new_instruction
    analysis["use_char_target"] = use_char_target
    write_json(analysis_path, analysis)

    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    _rebuild_report_instructions(run_dir, report)
    write_json(report_path, report)

    write_recording_html_from_run(run_dir, update_index=False)
    return {"instruction": new_instruction, "use_char_target": use_char_target}


def rerun_recording_event_yolo_ocr(
    runs_root: Path,
    run_id: str,
    event_index: int,
) -> dict[str, Any]:
    """Re-run YOLO/OCR for one pointer event, rebuild the instruction, and persist.

    Returns ``instruction``, ``detection_count``, and ``candidate_count``.
    Raises ``ValueError`` for invalid input and ``RuntimeError`` when inference
    fails or yields no usable targets.
    """
    run_dir = resolve_deletable_run_folder(runs_root, run_id)
    if not isinstance(event_index, int) or event_index < 1:
        raise ValueError("invalid event index")

    event_path = event_json_path(run_dir, event_index)
    event_payload = read_json(event_path, None)
    if not isinstance(event_payload, dict):
        raise ValueError("event not found")
    event = RecordedEvent.from_dict(event_payload)
    if event.kind not in POINTER_EVENT_KINDS:
        raise ValueError("event does not support YOLO/OCR")

    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    analysis = read_json(analysis_path, None)
    if not isinstance(analysis, dict):
        analysis = {"event_index": event_index}
    use_char_target = use_char_target_enabled(analysis)

    vision = run_pointer_event_yolo_ocr(event, run_dir=run_dir, persist_debug=True)
    yolo_error = vision.get("yolo_error")
    if yolo_error:
        raise RuntimeError(str(yolo_error))

    destination = vision.get("destination") if isinstance(vision.get("destination"), dict) else {}
    start_candidates = vision.get("candidates") if isinstance(vision.get("candidates"), list) else []
    end_candidates = (
        destination.get("candidates") if isinstance(destination.get("candidates"), list) else []
    )
    if not start_candidates and (event.kind != "drag" or not end_candidates):
        raise RuntimeError("YOLO/OCR 沒有偵測到目標。請確認 Triton 可用後再試。")

    rebuilt = rebuild_pointer_instruction(
        event,
        vision,
        destination,
        include_nearby=True,
        use_char_target=use_char_target,
    )
    if not rebuilt:
        raise RuntimeError("已寫入 YOLO/OCR，但無法自動重建指令。")

    analysis["instruction"] = rebuilt
    analysis["vision"] = {
        "used_vision": vision.get("used_vision"),
        "candidate_text": vision.get("candidate_text"),
    }
    analysis.pop("landmarks", None)
    if primary_candidate_char_target(vision) is not None:
        analysis["use_char_target"] = use_char_target
    else:
        analysis.pop("use_char_target", None)
    write_json(analysis_path, analysis)

    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    _rebuild_report_instructions(run_dir, report)
    write_json(report_path, report)

    write_recording_html_from_run(run_dir, update_index=False)
    return {
        "instruction": rebuilt,
        "detection_count": int(vision.get("detection_count") or len(start_candidates)),
        "candidate_count": len(start_candidates),
    }


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
            event_outcome_match = _EVENT_EXPECTED_OUTCOME_PATH_RE.fullmatch(path)
            event_instruction_match = _EVENT_INSTRUCTION_PATH_RE.fullmatch(path)
            event_char_target_match = _EVENT_CHAR_TARGET_PATH_RE.fullmatch(path)
            event_yolo_ocr_match = _EVENT_YOLO_OCR_PATH_RE.fullmatch(path)
            event_delete_match = _EVENT_DELETE_PATH_RE.fullmatch(path)
            events_bulk_delete_match = _EVENTS_BULK_DELETE_PATH_RE.fullmatch(path)
            event_add_match = _EVENT_ADD_PATH_RE.fullmatch(path)

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

            rename_match = _RENAME_PATH_RE.fullmatch(path)
            if rename_match is not None:
                run_id = rename_match.group(1)
                try:
                    body = self._read_json_body()
                    result = rename_recording_folder(
                        root, run_id, body.get("name")
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
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
                        source=body.get("source"),
                        choice_text=body.get("choice_text"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if event_outcome_match is not None:
                run_id = event_outcome_match.group(1)
                event_index_raw = event_outcome_match.group(2)
                try:
                    event_index = int(event_index_raw)
                    body = self._read_json_body()
                    result = apply_recording_event_expected_outcome(
                        root,
                        run_id,
                        event_index,
                        expected_outcome=body.get("expected_outcome"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if event_instruction_match is not None:
                run_id = event_instruction_match.group(1)
                event_index_raw = event_instruction_match.group(2)
                try:
                    event_index = int(event_index_raw)
                    body = self._read_json_body()
                    result = apply_recording_event_instruction(
                        root,
                        run_id,
                        event_index,
                        instruction=body.get("instruction"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if event_char_target_match is not None:
                run_id = event_char_target_match.group(1)
                event_index_raw = event_char_target_match.group(2)
                try:
                    event_index = int(event_index_raw)
                    body = self._read_json_body()
                    result = apply_recording_event_char_target(
                        root,
                        run_id,
                        event_index,
                        use_char_target=body.get("use_char_target"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if event_yolo_ocr_match is not None:
                run_id = event_yolo_ocr_match.group(1)
                event_index_raw = event_yolo_ocr_match.group(2)
                try:
                    event_index = int(event_index_raw)
                    result = rerun_recording_event_yolo_ocr(root, run_id, event_index)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except RuntimeError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if event_add_match is not None:
                run_id = event_add_match.group(1)
                try:
                    body = self._read_json_body()
                    result = add_recording_event(
                        root,
                        run_id,
                        kind=body.get("kind"),
                        after_event_index=body.get("after_event_index"),
                        instruction=body.get("instruction"),
                        expected_outcome=body.get("expected_outcome"),
                        text=body.get("text"),
                        key=body.get("key"),
                        keys=body.get("keys"),
                        scroll_delta=body.get("scroll_delta"),
                        duration_seconds=body.get("duration_seconds"),
                        presence=body.get("presence"),
                        then_action=body.get("then_action"),
                    )
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, **result})
                return

            if events_bulk_delete_match is not None:
                run_id = events_bulk_delete_match.group(1)
                try:
                    body = self._read_json_body()
                    raw_indices = body.get("event_indices")
                    if not isinstance(raw_indices, list):
                        raise ValueError("event_indices must be a list")
                    event_indices: list[int] = []
                    for item in raw_indices:
                        if isinstance(item, bool) or not isinstance(item, int):
                            raise ValueError("invalid event index")
                        event_indices.append(item)
                    result = delete_recording_events(root, run_id, event_indices)
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
