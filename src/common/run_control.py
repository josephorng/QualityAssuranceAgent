"""Thread-safe pause/resume gate for agent runs (hub UI ↔ coordinator / queue worker)."""

from __future__ import annotations

import asyncio
import threading
import time

_PAUSE_POLL_S = 0.05

# Set = paused; clear = running.
_paused = threading.Event()
_logged_pause = False
_log_lock = threading.Lock()


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


def reset_run_control() -> None:
    """Clear pause state at run start/end."""
    resume_run()


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
