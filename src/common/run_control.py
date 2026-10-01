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
_step_errors: dict[int, str] = {}

# In-progress script step (0-based). None before the first step or after reset.
_ActiveStepCallback = Callable[[int | None], None]
_active_step_index: int | None = None
_active_step_instruction: str | None = None
_active_step_event_index: int | None = None
_active_step_callback: _ActiveStepCallback | None = None
_pending_step_jump: int | None = None
_script_step_lines: list[str] = []
_script_step_event_indices: list[int | None] = []
_ScriptStepsCallback = Callable[[list[str], list[int | None]], None]
_script_steps_callback: _ScriptStepsCallback | None = None
_active_step_lock = threading.Lock()


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
    global _step_errors
    set_step_status_callback(None)
    with _step_status_lock:
        _step_errors = {}


def notify_step_status(
    step_index: int,
    status: str,
    reason: str | None = None,
    *,
    retry: bool = False,
) -> None:
    """Notify the hub that script step ``step_index`` finished with ``ok`` or ``fail``.

    ``retry=True`` keeps ``reason`` on the overlay while the same step runs again.
    A later ``ok`` without ``retry`` clears it.
    """
    text = reason.strip() if isinstance(reason, str) else ""
    with _step_status_lock:
        if text and (status == "fail" or retry):
            _step_errors[step_index] = text
        elif status == "ok":
            _step_errors.pop(step_index, None)
        callback = _step_status_callback
    if callback is not None:
        callback(step_index, status)


def step_error(step_index: int) -> str | None:
    """Return the failure text for ``step_index``, if that step failed."""
    with _step_status_lock:
        return _step_errors.get(step_index)


def step_errors() -> dict[int, str]:
    """Return failure text for every step that has failed in this run."""
    with _step_status_lock:
        return dict(_step_errors)


def set_active_step_callback(callback: _ActiveStepCallback | None) -> None:
    """Register or clear the in-progress step callback used by the replay overlay."""
    global _active_step_callback
    with _active_step_lock:
        _active_step_callback = callback


def clear_active_step() -> None:
    """Drop the in-progress step index and its callback."""
    global _active_step_index, _active_step_instruction, _active_step_event_index
    global _active_step_callback, _pending_step_jump, _script_steps_callback
    global _script_step_lines, _script_step_event_indices
    with _active_step_lock:
        _active_step_index = None
        _active_step_instruction = None
        _active_step_event_index = None
        _active_step_callback = None
        _pending_step_jump = None
        _script_steps_callback = None
        _script_step_lines = []
        _script_step_event_indices = []


def notify_active_step(
    step_index: int | None,
    *,
    instruction: str | None = None,
    event_index: int | None = None,
) -> None:
    """Publish the script step the agent is currently running."""
    global _active_step_index, _active_step_instruction, _active_step_event_index
    with _active_step_lock:
        _active_step_index = step_index
        _active_step_instruction = instruction
        _active_step_event_index = event_index if isinstance(event_index, int) else None
        callback = _active_step_callback
    if callback is not None:
        callback(step_index)


def active_step_index() -> int | None:
    """Return the in-progress script step, or None when no step has started."""
    with _active_step_lock:
        return _active_step_index


def active_step_instruction() -> str | None:
    """Return the in-progress step text after the latest recording reload."""
    with _active_step_lock:
        return _active_step_instruction


def active_step_event_index() -> int | None:
    """Return the recording event id for the in-progress step, when known."""
    with _active_step_lock:
        return _active_step_event_index


def request_step_jump(index: int) -> None:
    """Ask the coordinator to run ``index`` on the next step boundary."""
    global _pending_step_jump
    with _active_step_lock:
        _pending_step_jump = index


def take_pending_step_jump() -> int | None:
    """Return and clear a queued step jump, if the user chose one."""
    global _pending_step_jump
    with _active_step_lock:
        index = _pending_step_jump
        _pending_step_jump = None
        return index


def peek_pending_step_jump() -> int | None:
    """Return a queued step jump without clearing it."""
    with _active_step_lock:
        return _pending_step_jump


def set_script_steps_callback(callback: _ScriptStepsCallback | None) -> None:
    """Register or clear the callback that receives the live script step list."""
    global _script_steps_callback
    with _active_step_lock:
        _script_steps_callback = callback


def publish_script_steps(
    lines: list[str],
    event_indices: list[int | None] | None = None,
) -> None:
    """Publish the script the run will execute, including recording edits."""
    global _script_step_lines, _script_step_event_indices
    copied_lines = [line for line in lines if isinstance(line, str)]
    copied_events: list[int | None] = []
    raw_events = event_indices or []
    for index in range(len(copied_lines)):
        event_index = raw_events[index] if index < len(raw_events) else None
        copied_events.append(event_index if isinstance(event_index, int) else None)
    with _active_step_lock:
        _script_step_lines = copied_lines
        _script_step_event_indices = copied_events
        callback = _script_steps_callback
    if callback is not None:
        callback(list(copied_lines), list(copied_events))


def script_step_lines() -> list[str]:
    """Return the latest published script lines."""
    with _active_step_lock:
        return list(_script_step_lines)


def script_step_event_indices() -> list[int | None]:
    """Return recording event ids aligned with ``script_step_lines``."""
    with _active_step_lock:
        return list(_script_step_event_indices)


def reset_run_control() -> None:
    """Clear pause state, step-status callback, and the in-progress step at run start/end."""
    resume_run()
    clear_step_status_callback()
    clear_active_step()


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
