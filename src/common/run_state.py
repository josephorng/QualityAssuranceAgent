from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.common.io_utils import append_text, write_json

_RUN_FOLDER_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_RUN_FOLDER_ILLEGAL_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RUN_FOLDER_MAX_LEN = 191


def ts_name() -> str:
    """UTC timestamp string for unique run folder suffixes: ``YYYYMMDD_HHMMSS_microseconds``."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def unique_run_folder_name(name: str = "task") -> str:
    """Unique folder name under a runs root: ``{prefix}_`` + UTC timestamp.

    Simple identifier prefixes (e.g. ``task``, ``recording``) are kept. Free-form
    task text or LLM prose falls back to ``task`` so folders are never named from it.
    """
    prefix = str(name).strip().lower()
    if not _RUN_FOLDER_PREFIX_RE.fullmatch(prefix):
        prefix = "task"
    return f"{prefix}_{ts_name()}"


def unique_named_run_folder(base_name: str) -> str:
    """Unique folder name that preserves a human label: ``{base_name}_{timestamp}``.

    Used when a task report should mirror a recording folder name (including
    non-ASCII). Illegal path characters are replaced; length is capped to match
    run-id validation. Empty/invalid labels fall back to ``unique_run_folder_name``.
    """
    stamp = ts_name()
    cleaned = _RUN_FOLDER_ILLEGAL_RE.sub("_", str(base_name).strip()).strip(" ._")
    if not cleaned or cleaned in {".", ".."}:
        return unique_run_folder_name()
    max_base = _RUN_FOLDER_MAX_LEN - 1 - len(stamp)
    if max_base < 1:
        return unique_run_folder_name()
    if len(cleaned) > max_base:
        cleaned = cleaned[:max_base].rstrip(" ._")
    if not cleaned:
        return unique_run_folder_name()
    return f"{cleaned}_{stamp}"


_B64_BODY_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/= \t\r\n"
)


def sanitize_log_text(text: str, *, min_redact_chars: int = 64, preview_chars: int = 32) -> str:
    """
    Redact long ``data:*;base64,...`` segments so logs stay small and readable.

    Callers often pass ``repr``/f-string payloads that embed screenshots; this keeps
    the prefix and a short preview plus an omitted-byte count.

    Args:
        text: Raw log line or fragment that may contain data URLs.
        min_redact_chars: Minimum base64 payload length (after whitespace strip) to redact.
        preview_chars: How many decoded base64 characters to keep at the start of the payload.

    Returns:
        The same string with long base64 segments shortened; unchanged if no data URLs match.
    """
    lower = text.casefold()
    if "data:" not in lower or "base64," not in lower:
        return text
    parts: list[str] = []
    i = 0
    while i < len(text):
        d = lower.find("data:", i)
        if d < 0:
            parts.append(text[i:])
            break
        if d > 0 and text[d - 1].isalnum():
            parts.append(text[i : d + 1])
            i = d + 1
            continue
        parts.append(text[i:d])
        sep = ";base64,"
        b = lower.find(sep, d)
        if b < 0:
            parts.append(text[d:])
            break
        start_payload = b + len(sep)
        e = start_payload
        while e < len(text) and text[e] in _B64_BODY_CHARS:
            e += 1
        raw = text[start_payload:e]
        compact = "".join(raw.split())
        prefix = text[d:start_payload]
        if len(compact) >= min_redact_chars:
            preview = compact[:preview_chars]
            parts.append(f"{prefix}{preview}…<omitted {len(compact)} base64 chars>")
        else:
            parts.append(text[d:e])
        i = e
    return "".join(parts)


@dataclass
class RunPaths:
    """Paths under a single run directory (captures, logs, hand trace, storage)."""

    root: Path
    eye_dir: Path
    yolo_ocr_dir: Path
    storage_dir: Path
    hand_csv: Path
    storage_json: Path
    info_log: Path


class RunStateManager:
    """Creates run folders, appends to ``run.log``, and mirrors some lines to per-step logs."""

    def __init__(self, runs_root: Path) -> None:
        """Use ``runs_root`` as the parent directory for each new run subfolder."""
        self.runs_root = runs_root
        self.paths: RunPaths | None = None
        self._step_log: tuple[int, int] | None = None
        self.session_end_reason: str | None = None

    def set_session_end_reason(self, reason: str) -> None:
        """Record why the current coordinator session ended (used for ``report.json``)."""
        self.session_end_reason = reason

    def init_run(self, task_input: str, run_folder_name: str | None = None) -> RunPaths:
        """
        Allocate a new run directory, create standard subfolders and empty artifacts, and log startup.

        Args:
            task_input: Task description (logged only; not used for folder naming).
            run_folder_name: If set, use this exact folder name under ``runs_root`` instead of auto-naming.

        Returns:
            ``RunPaths`` for the new run; also stored on ``self.paths``.
        """
        self.runs_root.mkdir(parents=True, exist_ok=True)
        folder_name = run_folder_name or unique_run_folder_name()
        root = self.runs_root / folder_name
        eye_dir = root / "eye"
        yolo_ocr_dir = root / "yolo_ocr"
        storage_dir = root / "storage"
        hand_csv = root / "hand.csv"
        storage_json = root / "storage.json"
        info_log = root / "run.log"

        eye_dir.mkdir(parents=True, exist_ok=True)
        yolo_ocr_dir.mkdir(parents=True, exist_ok=True)
        storage_dir.mkdir(parents=True, exist_ok=True)
        if not hand_csv.exists():
            hand_csv.write_text("", encoding="utf-8")
        if not storage_json.exists():
            write_json(storage_json, [])
        if not info_log.exists():
            info_log.write_text("", encoding="utf-8")

        self._step_log = None
        self.session_end_reason = None
        self.paths = RunPaths(
            root=root,
            eye_dir=eye_dir,
            yolo_ocr_dir=yolo_ocr_dir,
            storage_dir=storage_dir,
            hand_csv=hand_csv,
            storage_json=storage_json,
            info_log=info_log,
        )
        self.log_info(f"Run initialized for task: {task_input}")
        return self.paths

    def require_paths(self) -> RunPaths:
        """Return the active ``RunPaths`` or raise if ``init_run`` has not been called."""
        if self.paths is None:
            raise RuntimeError("Run state not initialized")
        return self.paths

    def set_step_log_context(self, transcript_counter: int, script_step_index: int) -> None:
        """Also append subsequent ``log_info`` / ``log_error`` lines to ``steps/<tc>_<si>.log`` beside ``run.log``."""
        self._step_log = (transcript_counter, script_step_index)

    def clear_step_log_context(self) -> None:
        """Stop duplicating log lines to the per-step file (see ``set_step_log_context``)."""
        self._step_log = None

    def _caller_label(self) -> str:
        """Short module path (under ``src/``) or filename of the direct log caller for log prefixes."""
        caller_label = "unknown"
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None
        if caller_frame is not None:
            caller_file = Path(caller_frame.f_code.co_filename)
            caller_no_suffix = caller_file.with_suffix("")
            parts = caller_no_suffix.parts
            if "src" in parts:
                src_index = parts.index("src")
                caller_label = "/".join(parts[src_index + 1 :]) or caller_no_suffix.name
            else:
                caller_label = caller_no_suffix.name
        return caller_label

    def _emit_log_line(self, line: str) -> None:
        """Print ``line``, append it to ``run.log``, and optionally to the active step log."""
        paths = self.require_paths()
        print(line)
        append_text(paths.info_log, line + "\n")
        if self._step_log is not None:
            tc, si = self._step_log
            step_log = paths.root / "steps" / f"{tc}_{si}.log"
            append_text(step_log, line + "\n")

    def log_info(self, text: str) -> None:
        """Write an INFO line with UTC time and caller label; long base64 in ``text`` is sanitized."""
        ts = datetime.now(timezone.utc).isoformat()
        caller_label = self._caller_label()
        line = f"[{ts}] [{caller_label}] {sanitize_log_text(text)}"
        self._emit_log_line(line)

    def log_error(self, text: str) -> None:
        """Write an ERROR line with UTC time and caller label; long base64 in ``text`` is sanitized."""
        ts = datetime.now(timezone.utc).isoformat()
        caller_label = self._caller_label()
        line = f"[{ts}] [ERROR] [{caller_label}] {sanitize_log_text(text)}"
        self._emit_log_line(line)

_manager: RunStateManager | None = None


def reset_run_state_manager() -> None:
    """Clear the cached manager so the next ``get_run_state_manager`` rebuilds from ``CUA_RUN_ROOT``."""
    global _manager
    _manager = None


def get_run_state_manager() -> RunStateManager:
    """
    Return the process-wide ``RunStateManager``, constructing it once from ``get_runtime_env``.

    Used by brain, eye, hand, and LLM clients for run-scoped logging.
    """
    global _manager
    if _manager is None:
        from src.common.runtime_context import get_runtime_env

        run_root, _ = get_runtime_env()
        _manager = RunStateManager(run_root.parent)
    return _manager
