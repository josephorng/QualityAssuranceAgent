from __future__ import annotations

from src.recorder.hotkey import (
    RECORDING_HOTKEY_DISPLAY,
    is_recording_toggle_hotkey,
)


def test_recording_hotkey_display() -> None:
    assert RECORDING_HOTKEY_DISPLAY == "Ctrl+Shift+R"


def test_is_recording_toggle_hotkey() -> None:
    assert is_recording_toggle_hotkey(["ctrl", "shift", "r"])
    assert is_recording_toggle_hotkey(["shift", "ctrl", "R"])
    assert not is_recording_toggle_hotkey(["ctrl", "shift", "s"])
    assert not is_recording_toggle_hotkey(["ctrl", "r"])
    assert not is_recording_toggle_hotkey(None)
    assert not is_recording_toggle_hotkey([])
