"""Settle-UIA LED status emissions and idle ElementFromPoint label."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import (
    RecordingSession,
    _DeferredCaptureJob,
    _IDLE_EFP_DEBOUNCE_S,
    _QueuedEvent,
)
from src.recorder.window_snapshot import ClickWindowInfo


def test_collect_and_publish_settle_uia_emits_yellow_then_red_without_cursor() -> None:
    """Settle publish no longer auto-greens; no idle cursor → red."""
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)

    def _snap(hwnd: int) -> dict:
        return {
            "window": [0, 0, 10, 10],
            "inputs": [],
            "scrollbars": [],
            "texts": [{"bbox": [0, 0, 8, 8], "text": "A"}],
        }

    with (
        patch("src.recorder.uia_input_boxes.snapshot_uia_screen", side_effect=_snap),
        patch("src.recorder.capture._window_class_name", return_value="Chrome_WidgetWin_1"),
    ):
        out = session._collect_and_publish_settle_uia(42, after_event_index=3)

    assert out is not None
    assert [row["state"] for row in statuses] == ["yellow", "red"]
    assert statuses[-1]["after_index"] == 3
    assert statuses[-1]["efp_name"] == ""
    assert session._last_settle_uia_by_hwnd == {}


def test_collect_and_publish_settle_uia_emits_green_when_cursor_ready() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._idle_efp_cursor_xy = (4, 4)
    session._idle_efp_probed_name = "關閉"
    session._idle_efp_probed_value = ""

    def _snap(hwnd: int) -> dict:
        return {
            "window": [0, 0, 10, 10],
            "inputs": [],
            "scrollbars": [],
            "texts": [{"bbox": [0, 0, 8, 8], "text": "關閉"}],
        }

    click_win = ClickWindowInfo(
        hwnd=42,
        title="T",
        process_name="a.exe",
        left=0,
        top=0,
        width=10,
        height=10,
        is_maximized=False,
        class_name="Chrome_WidgetWin_1",
    )
    with (
        patch("src.recorder.uia_input_boxes.snapshot_uia_screen", side_effect=_snap),
        patch("src.recorder.capture._window_class_name", return_value="Chrome_WidgetWin_1"),
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=click_win,
        ),
    ):
        # after_index=0 matches empty-event previous_index so the cache attaches.
        out = session._collect_and_publish_settle_uia(42, after_event_index=0)

    assert out is not None
    assert [row["state"] for row in statuses] == ["yellow", "green"]
    assert statuses[-1]["efp_name"] == "關閉"
    assert statuses[-1]["after_index"] == 0


def test_collect_and_publish_settle_uia_skips_empty_tree_keeps_prior() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    prior = {
        "hwnd": 10,
        "class_name": "Chrome_WidgetWin_1",
        "window": [0, 0, 10, 10],
        "inputs": [],
        "scrollbars": [],
        "texts": [{"bbox": [0, 0, 4, 4], "text": "先前"}],
    }
    session._last_settle_uia = prior
    session._last_settle_uia_by_hwnd = {10: dict(prior)}
    session._last_settle_uia_after_index = 1

    with (
        patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            return_value={
                "window": [0, 0, 10, 10],
                "inputs": [],
                "scrollbars": [],
                "texts": [],
            },
        ),
        patch(
            "src.recorder.capture._window_class_name",
            return_value="Chrome_WidgetWin_1",
        ),
    ):
        out = session._collect_and_publish_settle_uia(42, after_event_index=2)

    assert out is None
    assert session._last_settle_uia is not None
    assert session._last_settle_uia["hwnd"] == 10
    assert session._last_settle_uia["texts"][0]["text"] == "先前"
    assert 42 not in session._last_settle_uia_by_hwnd
    assert session._last_settle_uia_after_index == 2
    assert [row["state"] for row in statuses] == ["yellow", "red"]


def test_collect_and_publish_settle_uia_clears_same_hwnd_when_empty() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._last_settle_uia = {
        "hwnd": 42,
        "window": [0, 0, 10, 10],
        "inputs": [],
        "scrollbars": [],
        "texts": [{"bbox": [0, 0, 4, 4], "text": "舊"}],
    }
    with (
        patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            return_value={
                "window": [0, 0, 10, 10],
                "inputs": [],
                "scrollbars": [],
                "texts": [],
            },
        ),
        patch(
            "src.recorder.capture._window_class_name",
            return_value="Chrome_WidgetWin_1",
        ),
    ):
        assert session._collect_and_publish_settle_uia(42, after_event_index=3) is None
    assert session._last_settle_uia is None


def test_collect_and_publish_settle_uia_emits_red_on_fail() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._last_settle_uia_by_hwnd = {
        10: {
            "hwnd": 10,
            "class_name": "Chrome_WidgetWin_1",
            "window": [0, 0, 1, 1],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
        20: {
            "hwnd": 20,
            "class_name": "Shell_TrayWnd",
            "window": [0, 0, 1, 1],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    }

    with patch("src.recorder.uia_input_boxes.snapshot_uia_screen", return_value=None):
        out = session._collect_and_publish_settle_uia(42, after_event_index=1)

    assert out is None
    assert [row["state"] for row in statuses] == ["yellow", "red"]
    assert set(session._last_settle_uia_by_hwnd) == {20}


def test_collect_and_publish_settle_uia_red_for_bad_hwnd() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._last_settle_uia_by_hwnd = {
        20: {
            "hwnd": 20,
            "class_name": "Shell_TrayWnd",
            "window": [0, 0, 1, 1],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    }
    assert session._collect_and_publish_settle_uia(0, after_event_index=2) is None
    assert statuses[-1]["state"] == "red"
    assert statuses[-1]["after_index"] == 2
    assert set(session._last_settle_uia_by_hwnd) == {20}


def test_persist_pointer_pulses_attach_then_red(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    (tmp_path / "events").mkdir(parents=True, exist_ok=True)
    session.set_on_uia_status(statuses.append)
    session._start_window_step = lambda _item: None  # type: ignore[method-assign]
    session._notify_event = lambda _event: None  # type: ignore[method-assign]

    item = _QueuedEvent(
        kind="click",
        cursor_xy=(1, 2),
        event_index=1,
        timestamp_utc="t",
        screenshot_path="a.jpeg",
        monitor_index=1,
        monitor_offset=(0, 0),
        uia_snapshot={
            "hwnd": 9,
            "window": [0, 0, 1, 1],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    )
    session._persist_queued_event(item)
    assert any(row.get("pulse") == "attach" for row in statuses)
    assert statuses[-1]["state"] == "red"


def test_persist_pointer_pulses_miss_without_snapshot(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    (tmp_path / "events").mkdir(parents=True, exist_ok=True)
    session.set_on_uia_status(statuses.append)
    session._start_window_step = lambda _item: None  # type: ignore[method-assign]
    session._notify_event = lambda _event: None  # type: ignore[method-assign]

    item = _QueuedEvent(
        kind="click",
        cursor_xy=(1, 2),
        event_index=1,
        timestamp_utc="t",
        screenshot_path="a.jpeg",
        monitor_index=1,
        monitor_offset=(0, 0),
        uia_snapshot=None,
    )
    session._persist_queued_event(item)
    assert any(row.get("pulse") == "miss" for row in statuses)


def test_window_step_sealed_emits_red() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._resolve_window_change = lambda _item: (None, None, {"ok": True})  # type: ignore[method-assign]
    session._notify_pending_text_input = lambda _i: None  # type: ignore[method-assign]
    session._release_drag_uia_notify = lambda _item: None  # type: ignore[method-assign]
    session._patch_window_change = lambda *_a, **_k: None  # type: ignore[method-assign]

    item = _QueuedEvent(
        kind="click",
        cursor_xy=(1, 1),
        event_index=5,
        timestamp_utc="t",
        windows_before=(
            ClickWindowInfo(
                hwnd=1,
                title="A",
                process_name="a.exe",
                left=0,
                top=0,
                width=10,
                height=10,
                is_maximized=False,
            ),
        ),
        window_after_sealed=True,
    )
    session._window_step_body(item)
    assert statuses[-1]["state"] == "red"
    assert statuses[-1]["after_index"] == 5


def test_idle_efp_timer_cancel_on_move_drops_stale(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._accepting_input = True
    session.set_on_uia_status(statuses.append)
    session._last_settle_uia_after_index = 0
    session._last_settle_uia = {
        "hwnd": 7,
        "class_name": "Chrome_WidgetWin_1",
        "window": [0, 0, 100, 100],
        "inputs": [],
        "scrollbars": [],
        "texts": [{"bbox": [0, 0, 80, 80], "text": "NameA"}],
    }
    click_win = ClickWindowInfo(
        hwnd=7,
        title="T",
        process_name="a.exe",
        left=0,
        top=0,
        width=100,
        height=100,
        is_maximized=False,
        class_name="Chrome_WidgetWin_1",
    )

    calls: list[tuple[int, int]] = []

    def _probe(x: int, y: int):
        calls.append((x, y))
        return ("label", (0, 0, 1, 1), "text", "NameA", "ValA")

    with (
        patch(
            "src.recorder.uia_input_boxes.uia_text_element_at_point",
            side_effect=_probe,
        ),
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=click_win,
        ),
    ):
        session._schedule_idle_efp(10, 20)
        first_gen = session._idle_efp_generation
        # Move again before debounce — bumps generation and cancels timer.
        session._schedule_idle_efp(30, 40)
        assert session._idle_efp_generation == first_gen + 1
        time.sleep(_IDLE_EFP_DEBOUNCE_S + 0.15)
        # Only the second schedule should enqueue; drain worker manually.
        job = None
        try:
            while True:
                item = session._event_queue.get_nowait()
                if isinstance(item, _DeferredCaptureJob) and item.action == "idle_efp":
                    job = item
        except Exception:
            pass
        assert job is not None
        assert job.cursor_xy == (30, 40)
        session._worker_idle_efp(job)

    assert calls == [(30, 40)]
    assert statuses[-1]["state"] == "green"
    assert statuses[-1]["efp_name"] == "NameA"
    assert statuses[-1]["efp_value"] == "ValA"
    assert statuses[-1]["efp_pending"] is False


def test_idle_efp_clears_label_when_findall_empty(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session._accepting_input = True
    session.set_on_uia_status(statuses.append)
    session._idle_efp_generation = 1
    session._last_settle_uia_after_index = 0
    session._last_settle_uia = {
        "hwnd": 7,
        "class_name": "ComboLBox",
        "window": [0, 0, 100, 100],
        "inputs": [],
        "scrollbars": [],
        "texts": [],
    }
    click_win = ClickWindowInfo(
        hwnd=7,
        title="",
        process_name="a.exe",
        left=0,
        top=0,
        width=100,
        height=100,
        is_maximized=False,
        class_name="ComboLBox",
    )
    with (
        patch(
            "src.recorder.uia_input_boxes.uia_text_element_at_point",
            return_value=("x", (0, 0, 1, 1), "text", "Leaf", "V"),
        ),
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=click_win,
        ),
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(30, 40),
                meta={"generation": 1},
            )
        )
    assert statuses[-1]["state"] == "red"
    assert statuses[-1]["efp_name"] == ""
    assert statuses[-1]["efp_value"] == ""


def test_worker_idle_efp_ignores_stale_generation() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._accepting_input = True
    session.set_on_uia_status(statuses.append)
    session._idle_efp_generation = 9

    with patch(
        "src.recorder.uia_input_boxes.uia_text_element_at_point",
        return_value=("x", (0, 0, 1, 1), "text", "Stale", "X"),
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(1, 1),
                meta={"generation": 3},
            )
        )
    assert statuses == []


def test_recording_uia_status_overlay_apply_status() -> None:
    import customtkinter as ctk

    from src.recorder.stop_overlay import RecordingUiaStatusOverlay

    root = ctk.CTk()
    root.withdraw()
    try:
        overlay = RecordingUiaStatusOverlay(root)
        overlay.apply_status(
            {
                "state": "green",
                "pulse": None,
                "efp_name": "關閉",
                "efp_value": "",
                "efp_pending": False,
            }
        )
        assert overlay._state == "green"
        assert overlay._label is not None
        assert "關閉" in str(overlay._label.cget("text"))
        overlay.destroy()
    finally:
        root.destroy()


def test_monitor_for_point_picks_secondary() -> None:
    from src.recorder.stop_overlay import _monitor_for_point, _top_center_geometry

    monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": -100, "width": 2560, "height": 1440},
    ]
    mon = _monitor_for_point(2000, 50, monitors=monitors)
    assert mon is not None
    assert mon["left"] == 1920
    width, height, ox, oy = _top_center_geometry(mon, 300, 40)
    assert width == 300
    assert height == 40
    assert ox == 1920 + (2560 - 300) // 2
    assert oy == -100 + 16


def test_top_edge_geometry_slots() -> None:
    from src.recorder.stop_overlay import _top_edge_geometry

    mon = {"left": 100, "top": 50, "width": 1920, "height": 1080}
    _w, _h, cx, cy = _top_edge_geometry(mon, 300, 40, "center")
    assert (cx, cy) == (100 + (1920 - 300) // 2, 50 + 16)
    _w, _h, lx, ly = _top_edge_geometry(mon, 300, 40, "left")
    assert (lx, ly) == (100 + 16, 50 + 16)
    _w, _h, rx, ry = _top_edge_geometry(mon, 300, 40, "right")
    assert (rx, ry) == (100 + 1920 - 300 - 16, 50 + 16)


def test_uia_status_slot_for_cursor_dodges_center_overlap() -> None:
    from src.recorder.stop_overlay import (
        _top_edge_geometry,
        _uia_status_slot_for_cursor,
    )

    mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    _w, _h, ox, oy = _top_edge_geometry(mon, 300, 40, "center")
    # Cursor in the right half of the center plate → dodge left.
    assert (
        _uia_status_slot_for_cursor(mon, 300, 40, ox + 250, oy + 10) == "left"
    )
    # Cursor in the left half of the center plate → dodge right.
    assert (
        _uia_status_slot_for_cursor(mon, 300, 40, ox + 20, oy + 10) == "right"
    )
    # Clear of the plate → stay center.
    assert _uia_status_slot_for_cursor(mon, 300, 40, 100, 200) == "center"


def test_uia_status_overlay_places_on_cursor_monitor(monkeypatch) -> None:
    from src.recorder import stop_overlay as so

    monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": -100, "width": 2560, "height": 1440},
    ]
    geometries: list[str] = []

    class _FakeWin:
        def update_idletasks(self) -> None:
            return None

        def winfo_reqwidth(self) -> int:
            return 300

        def winfo_reqheight(self) -> int:
            return 40

        def geometry(self, value: str) -> None:
            geometries.append(value)

        def deiconify(self) -> None:
            return None

        def lift(self) -> None:
            return None

        def attributes(self, *_args: object, **_kwargs: object) -> None:
            return None

    overlay = so.RecordingUiaStatusOverlay.__new__(so.RecordingUiaStatusOverlay)
    overlay._destroyed = False
    overlay._window = _FakeWin()
    overlay._placed_monitor_key = None
    overlay._placed_slot = "center"
    overlay._last_width = 240
    overlay._last_height = 32
    # y=200 keeps the pointer clear of the top strip so center stays preferred.
    overlay._cursor_xy = lambda: (2100, 200)  # type: ignore[method-assign]

    monkeypatch.setattr(so, "_cached_monitors", lambda force=False: monitors)

    overlay.ensure_visible()
    assert geometries
    assert geometries[-1].endswith("+3050+-84")
    assert overlay._placed_monitor_key == (1920, -100, 2560, 1440)
    assert overlay._placed_slot == "center"

    before = len(geometries)
    overlay.ensure_visible()
    assert len(geometries) == before

    overlay._cursor_xy = lambda: (100, 200)  # type: ignore[method-assign]
    overlay.ensure_visible()
    assert len(geometries) == before + 1
    assert geometries[-1].endswith("+810+16")
    assert overlay._placed_monitor_key == (0, 0, 1920, 1080)
    assert overlay._placed_slot == "center"


def test_uia_status_overlay_place_same_monitor_noop() -> None:
    from src.recorder import stop_overlay as so

    geometries: list[str] = []

    class _FakeWin:
        def update_idletasks(self) -> None:
            return None

        def winfo_reqwidth(self) -> int:
            return 240

        def winfo_reqheight(self) -> int:
            return 32

        def geometry(self, value: str) -> None:
            geometries.append(value)

    overlay = so.RecordingUiaStatusOverlay.__new__(so.RecordingUiaStatusOverlay)
    overlay._destroyed = False
    overlay._window = _FakeWin()
    overlay._placed_monitor_key = None
    overlay._placed_slot = "center"
    overlay._last_width = 240
    overlay._last_height = 32
    mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}

    assert overlay._place_on_monitor(mon) is True
    assert len(geometries) == 1
    assert overlay._placed_slot == "center"
    assert overlay._place_on_monitor(mon) is False
    assert len(geometries) == 1
    assert overlay._place_on_monitor(mon, force=True) is True
    assert len(geometries) == 2


def test_uia_status_overlay_dodges_cursor_then_restores() -> None:
    from src.recorder import stop_overlay as so

    geometries: list[str] = []

    class _FakeWin:
        def update_idletasks(self) -> None:
            return None

        def winfo_reqwidth(self) -> int:
            return 300

        def winfo_reqheight(self) -> int:
            return 40

        def geometry(self, value: str) -> None:
            geometries.append(value)

    overlay = so.RecordingUiaStatusOverlay.__new__(so.RecordingUiaStatusOverlay)
    overlay._destroyed = False
    overlay._window = _FakeWin()
    overlay._placed_monitor_key = None
    overlay._placed_slot = "center"
    overlay._last_width = 300
    overlay._last_height = 40
    mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}

    # Center plate spans ~810..1110 at y=16; cursor on its right half → left.
    assert overlay._place_on_monitor(mon, cursor=(1000, 20)) is True
    assert overlay._placed_slot == "left"
    assert geometries[-1].endswith("+16+16")

    # Same overlap still no-ops.
    assert overlay._place_on_monitor(mon, cursor=(1000, 20)) is False
    assert len(geometries) == 1

    # Clear of the plate → restore center.
    assert overlay._place_on_monitor(mon, cursor=(100, 200)) is True
    assert overlay._placed_slot == "center"
    assert geometries[-1].endswith("+810+16")


def test_should_exclude_uia_status_skips_remote_and_vm(monkeypatch) -> None:
    from src.recorder import stop_overlay as so

    monkeypatch.delenv(so._UIA_STATUS_EXCLUDE_FROM_CAPTURE_ENV, raising=False)
    monkeypatch.setattr(so, "_is_remote_display_session", lambda: False)
    monkeypatch.setattr(so, "_is_virtual_machine_guest", lambda: False)
    assert so._should_exclude_uia_status_from_capture() is True

    monkeypatch.setattr(so, "_is_remote_display_session", lambda: True)
    assert so._should_exclude_uia_status_from_capture() is False

    monkeypatch.setattr(so, "_is_remote_display_session", lambda: False)
    monkeypatch.setattr(so, "_is_virtual_machine_guest", lambda: True)
    assert so._should_exclude_uia_status_from_capture() is False

    monkeypatch.setattr(so, "_is_virtual_machine_guest", lambda: False)
    monkeypatch.setenv(so._UIA_STATUS_EXCLUDE_FROM_CAPTURE_ENV, "0")
    assert so._should_exclude_uia_status_from_capture() is False


def test_apply_exclude_from_capture_noop_when_gated(monkeypatch) -> None:
    from src.recorder import stop_overlay as so

    calls: list[tuple[int, int]] = []

    class _User32:
        @staticmethod
        def SetWindowDisplayAffinity(hwnd: int, affinity: int) -> int:
            calls.append((hwnd, affinity))
            return 1

    monkeypatch.setattr(so, "_should_exclude_uia_status_from_capture", lambda: False)
    monkeypatch.setattr(so, "_toplevel_hwnd", lambda _win: 42)
    monkeypatch.setattr(so.ctypes, "windll", type("W", (), {"user32": _User32()})())
    so._apply_exclude_from_capture(object())
    assert calls == []

    monkeypatch.setattr(so, "_should_exclude_uia_status_from_capture", lambda: True)
    so._apply_exclude_from_capture(object())
    assert calls == [(42, so._WDA_EXCLUDEFROMCAPTURE)]


def test_apply_click_through_sets_opaque_layered_alpha(monkeypatch) -> None:
    from src.recorder import stop_overlay as so

    styles: list[int] = []
    alphas: list[tuple[int, int, int, int]] = []

    class _User32:
        @staticmethod
        def GetWindowLongW(_hwnd: int, _index: int) -> int:
            return 0

        @staticmethod
        def SetWindowLongW(_hwnd: int, _index: int, style: int) -> int:
            styles.append(int(style))
            return style

        @staticmethod
        def SetLayeredWindowAttributes(
            hwnd: int, color: int, alpha: int, flags: int
        ) -> int:
            alphas.append((hwnd, color, alpha, flags))
            return 1

    monkeypatch.setattr(so, "_toplevel_hwnd", lambda _win: 7)
    monkeypatch.setattr(so.ctypes, "windll", type("W", (), {"user32": _User32()})())
    so._apply_click_through(object())
    assert styles and (styles[-1] & so._WS_EX_LAYERED)
    assert styles[-1] & so._WS_EX_TRANSPARENT
    assert alphas == [(7, 0, 255, so._LWA_ALPHA)]
