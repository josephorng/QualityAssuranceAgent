from __future__ import annotations

from src.recorder.stop_overlay import (
    RecordingStopOverlay,
    ReplayControlOverlay,
    _EDGE_TRIGGER_PX,
    _ERROR_SHOW_MS,
    clip_step_snippet,
)


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


def test_replay_overlay_uses_the_same_top_edge() -> None:
    overlay = ReplayControlOverlay.__new__(ReplayControlOverlay)
    overlay._monitors_cache = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": -200, "width": 1920, "height": 1080},
    ]
    overlay._monitors_cache_at = 1e18

    assert overlay._cursor_at_top_edge(100, _EDGE_TRIGGER_PX) is True
    assert overlay._cursor_at_top_edge(100, _EDGE_TRIGGER_PX + 1) is False
    assert overlay._cursor_at_top_edge(2000, -200) is True
    assert overlay._cursor_at_top_edge(2000, -200 + _EDGE_TRIGGER_PX + 1) is False


def test_clip_step_snippet_collapses_and_truncates() -> None:
    assert clip_step_snippet("  click   the button  ") == "click the button"
    clipped = clip_step_snippet("x" * 80)
    assert len(clipped) == 42
    assert clipped.endswith("…")


class _ErrorLabel:
    def __init__(self) -> None:
        self.text = ""
        self.mapped = False

    def configure(self, **kwargs: object) -> None:
        if "text" in kwargs:
            self.text = str(kwargs["text"])

    def winfo_ismapped(self) -> bool:
        return self.mapped

    def pack(self, **kwargs: object) -> None:
        del kwargs
        self.mapped = True

    def pack_forget(self) -> None:
        self.mapped = False


def test_error_overlay_stays_up_for_three_seconds(monkeypatch) -> None:
    overlay = ReplayControlOverlay.__new__(ReplayControlOverlay)
    overlay._destroyed = False
    overlay._pinned = False
    overlay._visible = False
    overlay._shown_error = ""
    overlay._error_show_after_id = None
    overlay._countdown_active = False
    overlay._error_label = _ErrorLabel()
    overlay._nav_frame = object()
    overlay._cancel_hide = lambda: None
    scheduled: list[tuple[int, object]] = []

    def after(ms: int, callback: object) -> str:
        scheduled.append((ms, callback))
        return "timer"

    overlay._master = type("Master", (), {"after": staticmethod(after)})()
    overlay._apply_error("window missing")
    overlay._apply_error("window missing")

    assert overlay._pinned is True
    assert scheduled == [(_ERROR_SHOW_MS, overlay._end_error_show)]
    assert _ERROR_SHOW_MS == 3000

    hidden: list[bool] = []
    overlay._hide = lambda: hidden.append(True)
    monkeypatch.setattr(
        "src.recorder.stop_overlay.pyautogui.position",
        lambda: type("Pos", (), {"x": 100, "y": 400})(),
    )
    overlay._cursor_at_top_edge = lambda x, y: False
    overlay._cursor_over_overlay = lambda x, y: False
    overlay._end_error_show()

    assert overlay._pinned is False
    assert hidden == [True]


def test_retry_countdown_pins_until_cleared(monkeypatch) -> None:
    overlay = ReplayControlOverlay.__new__(ReplayControlOverlay)
    overlay._destroyed = False
    overlay._pinned = False
    overlay._visible = True
    overlay._countdown_active = False
    overlay._error_show_after_id = "error-timer"
    overlay._countdown_label = _ErrorLabel()
    overlay._hold_pause_btn = _ErrorLabel()
    overlay._stop_btn = None
    overlay._nav_frame = object()
    cancelled: list[str] = []
    overlay._master = type(
        "Master",
        (),
        {"after_cancel": staticmethod(lambda after_id: cancelled.append(after_id))},
    )()
    overlay._measure_and_hide_if_hidden = lambda: None
    overlay.reveal = lambda: None
    hidden: list[bool] = []
    overlay._hide = lambda: hidden.append(True)
    monkeypatch.setattr(
        "src.recorder.stop_overlay.pyautogui.position",
        lambda: type("Pos", (), {"x": 100, "y": 400})(),
    )
    overlay._cursor_at_top_edge = lambda x, y: False
    overlay._cursor_over_overlay = lambda x, y: False

    overlay.set_countdown(30)

    assert overlay._pinned is True
    assert overlay._countdown_active is True
    assert overlay._countdown_label.text == "30 秒後繼續"
    assert overlay._hold_pause_btn.mapped is True
    assert cancelled == ["error-timer"]
    assert hidden == []

    overlay.set_countdown(None)

    assert overlay._pinned is False
    assert overlay._countdown_active is False
    assert overlay._hold_pause_btn.mapped is False
    assert hidden == [True]


def test_editing_keeps_overlay_pinned(monkeypatch) -> None:
    overlay = ReplayControlOverlay.__new__(ReplayControlOverlay)
    overlay._destroyed = False
    overlay._pinned = True
    overlay._visible = True
    overlay._editing = True
    overlay._countdown_active = False
    overlay._error_show_after_id = None
    overlay._countdown_label = _ErrorLabel()
    overlay._hold_pause_btn = _ErrorLabel()
    overlay._hold_pause_btn.mapped = True
    overlay._stop_btn = None
    hidden: list[bool] = []
    overlay._hide = lambda: hidden.append(True)
    overlay._measure_and_hide_if_hidden = lambda: None
    monkeypatch.setattr(
        "src.recorder.stop_overlay.pyautogui.position",
        lambda: type("Pos", (), {"x": 100, "y": 400})(),
    )
    overlay._cursor_at_top_edge = lambda x, y: False
    overlay._cursor_over_overlay = lambda x, y: False

    overlay._end_error_show()
    overlay.set_countdown(None)

    assert overlay._pinned is True
    assert hidden == []


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
