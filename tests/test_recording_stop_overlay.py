from __future__ import annotations

from src.recorder.stop_overlay import RecordingStopOverlay, _EDGE_TRIGGER_PX


def test_cursor_at_top_edge_uses_monitor_top(monkeypatch) -> None:
    overlay = RecordingStopOverlay.__new__(RecordingStopOverlay)
    overlay._monitors_cache = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": -200, "width": 1920, "height": 1080},
    ]
    overlay._monitors_cache_at = 1e18

    assert overlay._cursor_at_top_edge(100, 0) is True
    assert overlay._cursor_at_top_edge(100, _EDGE_TRIGGER_PX) is True
    assert overlay._cursor_at_top_edge(100, _EDGE_TRIGGER_PX + 1) is False
    # Second monitor is shifted up; its own top edge should trigger.
    assert overlay._cursor_at_top_edge(2000, -200) is True
    assert overlay._cursor_at_top_edge(2000, -200 + _EDGE_TRIGGER_PX + 1) is False
