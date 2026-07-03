from __future__ import annotations

import threading
from typing import Callable

from pynput import keyboard

_RECORDING_HOTKEY = "<ctrl>+<shift>+r"


class RecordingHotkeyManager:
    """Global hotkey to toggle screen recording."""

    def __init__(self) -> None:
        self._listener: keyboard.GlobalHotKeys | None = None
        self._toggle_callback: Callable[[], None] | None = None
        self._lock = threading.Lock()

    def register(self, toggle_callback: Callable[[], None]) -> None:
        with self._lock:
            self._toggle_callback = toggle_callback
            if self._listener is not None:
                return

            def _on_hotkey() -> None:
                with self._lock:
                    cb = self._toggle_callback
                if cb is not None:
                    cb()

            self._listener = keyboard.GlobalHotKeys({_RECORDING_HOTKEY: _on_hotkey})
            self._listener.start()

    def unregister(self) -> None:
        with self._lock:
            listener = self._listener
            self._listener = None
            self._toggle_callback = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
