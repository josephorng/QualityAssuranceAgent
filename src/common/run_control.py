"""Thread-safe pause/resume gate for agent runs (hub UI ↔ coordinator / queue worker)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

_PAUSE_POLL_S = 0.05

# Set = paused; clear = running.
_paused = threading.Event()
_logged_pause = False
_log_lock = threading.Lock()

# Optional hub callback: (script_step_index, "ok"|"fail") from the coordinator thread.
_StepStatusCallback = Callable[[int, str], None]
_step_status_callback: _StepStatusCallback | None = None
_step_status_lock = threading.Lock()


def pause_run() -> None:
    global _logged_pause
    _paused.set()
    with _log_lock:
        _logged_pause = False


def resume_run() -> None:
    global _logged_pause
    _paused.clear()
    with _log_lock:
        _logged_pause = False


def set_step_status_callback(callback: _StepStatusCallback | None) -> None:
    """Register or clear the per-step status callback used by the single-script hub UI."""
    global _step_status_callback
    with _step_status_lock:
        _step_status_callback = callback


def clear_step_status_callback() -> None:
    """Drop the per-step status callback (also called from ``reset_run_control``)."""
    set_step_status_callback(None)


def notify_step_status(step_index: int, status: str) -> None:
    """Notify the hub that script step ``step_index`` finished with ``ok`` or ``fail``."""
    with _step_status_lock:
        callback = _step_status_callback
    if callback is not None:
        callback(step_index, status)


def reset_run_control() -> None:
    """Clear pause state and step-status callback at run start/end."""
    resume_run()
    clear_step_status_callback()


def is_paused() -> bool:
    return _paused.is_set()


def take_pause_log() -> bool:
    """Return True once per pause episode (for coordinator audit logging)."""
    global _logged_pause
    if not _paused.is_set():
        return False
    with _log_lock:
        if _logged_pause:
            return False
        _logged_pause = True
        return True


async def wait_while_paused() -> None:
    """Block the event loop cooperatively while paused; cancel still interrupts sleep."""
    while _paused.is_set():
        await asyncio.sleep(_PAUSE_POLL_S)


def wait_while_paused_blocking() -> None:
    """Block the calling thread while paused (queue worker between scripts)."""
    while _paused.is_set():
        time.sleep(_PAUSE_POLL_S)
