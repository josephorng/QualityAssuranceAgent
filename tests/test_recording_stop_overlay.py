from __future__ import annotations

from src.recorder.stop_overlay import (
    RecordingStopOverlay,
    ReplayControlOverlay,
    ReplayMonitorFrameOverlay,
    _EDGE_TRIGGER_PX,
    _ERROR_SHOW_MS,
    _REPLAY_FRAME_THICKNESS,
    clip_step_snippet,
    replay_frame_badge_origin,
    replay_frame_strip_geometries,
    resolve_replay_frame_monitors,
)



def test_replay_overlay_uses_the_same_top_edge(monkeypatch) -> None:
    monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": -200, "width": 1920, "height": 1080},
    ]
    monkeypatch.setattr(
        "src.recorder.stop_overlay._cached_monitors",
        lambda force=False: monitors,
    )
    overlay = ReplayControlOverlay.__new__(ReplayControlOverlay)

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
    monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": -200, "width": 1920, "height": 1080},
    ]
    monkeypatch.setattr(
        "src.recorder.stop_overlay._cached_monitors",
        lambda force=False: monitors,
    )
    overlay = RecordingStopOverlay.__new__(RecordingStopOverlay)

    assert overlay._cursor_at_top_edge(100, 0) is True
    assert overlay._cursor_at_top_edge(100, _EDGE_TRIGGER_PX) is True
    assert overlay._cursor_at_top_edge(100, _EDGE_TRIGGER_PX + 1) is False
    # Second monitor is shifted up; its own top edge should trigger.
    assert overlay._cursor_at_top_edge(2000, -200) is True
    assert overlay._cursor_at_top_edge(2000, -200 + _EDGE_TRIGGER_PX + 1) is False


def test_resolve_replay_frame_monitors_selected_and_fallback() -> None:
    monitors = [
        {"left": 0, "top": 0, "width": 800, "height": 600},
        {"left": 800, "top": 0, "width": 1024, "height": 768},
    ]
    assert resolve_replay_frame_monitors(None, monitors=monitors) == monitors
    assert resolve_replay_frame_monitors([], monitors=monitors) == monitors
    assert resolve_replay_frame_monitors([2], monitors=monitors) == [monitors[1]]
    assert resolve_replay_frame_monitors([1, 2, 2, 99], monitors=monitors) == monitors
    # Invalid-only selection falls back to all physical monitors.
    assert resolve_replay_frame_monitors([99], monitors=monitors) == monitors
    assert resolve_replay_frame_monitors([1], monitors=[]) == []


def test_replay_frame_strip_geometries_four_edges() -> None:
    mon = {"left": 100, "top": 50, "width": 200, "height": 100}
    strips = replay_frame_strip_geometries(mon, thickness=_REPLAY_FRAME_THICKNESS)
    assert strips == [
        (200, 4, 100, 50),
        (200, 4, 100, 146),
        (4, 92, 100, 54),
        (4, 92, 296, 54),
    ]
    assert replay_frame_badge_origin(mon) == (112, 62)


def test_replay_monitor_frame_overlay_destroy_is_idempotent() -> None:
    class _FakeWin:
        def __init__(self) -> None:
            self.destroyed = False

        def destroy(self) -> None:
            self.destroyed = True

    overlay = ReplayMonitorFrameOverlay.__new__(ReplayMonitorFrameOverlay)
    wins = [_FakeWin(), _FakeWin()]
    overlay._destroyed = False
    overlay._windows = wins
    overlay._badges = [{"win": wins[0]}]

    overlay.destroy()
    overlay.destroy()

    assert overlay._destroyed is True
    assert overlay._windows == []
    assert overlay._badges == []
    assert all(win.destroyed for win in wins)


def test_replay_monitor_frame_overlay_set_progress_updates_labels() -> None:
    class _Label:
        def __init__(self) -> None:
            self.text = ""
            self.mapped = False
            self.after_widget = None

        def configure(self, **kwargs: object) -> None:
            if "text" in kwargs:
                self.text = str(kwargs["text"])

        def winfo_ismapped(self) -> bool:
            return self.mapped

        def pack(self, **kwargs: object) -> None:
            self.mapped = True
            self.after_widget = kwargs.get("after")

        def pack_forget(self) -> None:
            self.mapped = False
            self.after_widget = None

    class _FakeWin:
        def __init__(self) -> None:
            self.geometry_calls: list[str] = []

        def update_idletasks(self) -> None:
            return None

        def winfo_reqwidth(self) -> int:
            return 140

        def winfo_reqheight(self) -> int:
            return 64

        def geometry(self, value: str) -> None:
            self.geometry_calls.append(value)

        def winfo_exists(self) -> bool:
            return True

        def attributes(self, *_args: object, **_kwargs: object) -> None:
            return None

        def lift(self) -> None:
            return None

    overlay = ReplayMonitorFrameOverlay.__new__(ReplayMonitorFrameOverlay)
    overlay._destroyed = False
    win = _FakeWin()
    title = _Label()
    phase = _Label()
    elapsed = _Label()
    overlay._windows = [win]
    overlay._badges = [
        {
            "win": win,
            "mon": {"left": 100, "top": 50, "width": 200, "height": 100},
            "title": title,
            "phase": phase,
            "elapsed": elapsed,
        }
    ]
    overlay._step_text = "重播中"
    overlay._phase_text = ""
    overlay._elapsed_text = ""
    overlay._harden_all = lambda: None  # type: ignore[method-assign]

    overlay.set_progress(
        step_text="重播中 · 步驟 4/12",
        phase="選取 · FindAll",
        elapsed_text="00:07",
    )
    assert title.text == "重播中 · 步驟 4/12"
    assert phase.text == "選取 · FindAll"
    assert phase.mapped is True
    assert elapsed.text == "00:07"
    assert elapsed.mapped is True
    assert win.geometry_calls
    assert win.geometry_calls[-1].endswith("+112+62")

    overlay.clear_progress()
    assert title.text == "重播中"
    assert phase.text == ""
    assert phase.mapped is False
    assert elapsed.text == ""
    assert elapsed.mapped is False


def test_fill_verify_checks_does_not_stack_scroll_shells() -> None:
    """Rebuilding the checklist must destroy CTkScrollableFrame's packed parent."""
    import customtkinter as ctk

    root = ctk.CTk()
    root.withdraw()
    try:
        overlay = ReplayControlOverlay.__new__(ReplayControlOverlay)
        overlay._destroyed = False
        overlay._window = root
        overlay._edit_frame = ctk.CTkFrame(root, fg_color="transparent")
        overlay._edit_frame.pack()
        overlay._checks_empty = ctk.CTkLabel(overlay._edit_frame, text="")
        overlay._edit_message = ctk.CTkLabel(overlay._edit_frame, text="")
        overlay._edit_message.pack(anchor="w")
        overlay._checks_scroll = None
        overlay._check_vars = []

        checks = [("title", None, "視窗標題", "Demo", True)]
        overlay._fill_verify_checks(checks)
        overlay._fill_verify_checks(checks)

        parents = [
            child
            for child in overlay._edit_frame.winfo_children()
            if isinstance(child, ctk.CTkFrame)
        ]
        assert len(parents) == 1
        assert overlay._checks_scroll is not None
        assert overlay._checks_scroll._parent_frame is parents[0]
    finally:
        root.destroy()
