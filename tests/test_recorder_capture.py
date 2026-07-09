from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import (
    RecordingSession,
    _DOUBLE_CLICK_INTERVAL_S,
    _finalize_drag_end_screenshot,
)
from src.recorder.window_snapshot import WindowInfo


@contextmanager
def _default_capture_window_patches():
    class DummyListener:
        running = True

        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            return None

    with patch("src.recorder.capture.mouse.Listener", DummyListener), patch(
        "src.recorder.capture.keyboard.Listener",
        DummyListener,
    ), patch("src.recorder.capture.snapshot_top_level_windows", return_value=[]), patch(
        "src.recorder.capture.settle_delay_for_click",
        return_value=0.0,
    ):
        yield


def _mock_screenshot(*_args, **_kwargs) -> tuple[str, int, tuple[int, int]]:
    dest = _args[2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"fake")
    return str(dest), 1, (0, 0)


def _left_click(session: RecordingSession, x: int, y: int) -> None:
    from pynput.mouse import Button

    session._on_mouse_click(x, y, Button.left, True)
    session._on_mouse_click(x, y, Button.left, False)
    time.sleep(_DOUBLE_CLICK_INTERVAL_S + 0.05)


def test_left_drag_records_single_drag_event(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(100, 100, Button.left, True)
            session._on_mouse_move(150, 150)
            session._on_mouse_click(200, 200, Button.left, False)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "drag"
    assert raw["cursor_xy"] == [100, 100]
    assert raw["end_xy"] == [200, 200]
    assert raw["button"] == "left"
    assert raw["end_screenshot_path"].endswith("event_001_end.jpeg")


def test_small_move_still_records_click(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(400, 400, Button.left, True)
            session._on_mouse_move(402, 401)
            session._on_mouse_click(403, 402, Button.left, False)
            time.sleep(_DOUBLE_CLICK_INTERVAL_S + 0.05)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "click"
    assert raw["cursor_xy"] == [400, 400]


def test_keyboard_events_not_filtered_by_ignore_rect(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    hub_rect = (0, 0, 2000, 1200)

    with _default_capture_window_patches(), patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start(ignore_rect=hub_rect)
        try:
            from pynput.keyboard import KeyCode

            session._on_key_press(KeyCode.from_char("a"))
        finally:
            session.stop()

    assert session.event_count() == 1
    assert (run_dir / "events" / "event_001.json").is_file()


def test_typing_burst_coalesced_into_one_event(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    captures: list[tuple[int, tuple[int, int]]] = []

    def track_capture(run_dir, index, cursor_xy):
        captures.append((index, cursor_xy))
        return str(run_dir / "screenshots" / f"event_{index:03d}.jpeg"), 1, (0, 0)

    with _default_capture_window_patches(), patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch.object(
        RecordingSession,
        "_capture_immediate_screenshot",
        side_effect=track_capture,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import KeyCode

            for ch in "chrome":
                session._on_key_press(KeyCode.from_char(ch))
            assert captures == []
            session.stop()
        finally:
            if session.is_active():
                session.stop()

    assert session.event_count() == 1
    assert len(captures) == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "chrome"


def test_slow_typing_stays_one_event_until_other_operation(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import KeyCode

            session._on_key_press(KeyCode.from_char("c"))
            time.sleep(0.7)
            session._on_key_press(KeyCode.from_char("h"))
            time.sleep(0.7)
            session._on_key_press(KeyCode.from_char("r"))
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["text"] == "chr"


def test_functional_key_flushes_text_then_records_key(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            for ch in "go":
                session._on_key_press(KeyCode.from_char(ch))
            session._on_key_press(Key.enter)
        finally:
            session.stop()

    assert session.event_count() == 2
    text_event = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    enter_event = json.loads((run_dir / "events" / "event_002.json").read_text(encoding="utf-8"))
    assert text_event["kind"] == "text_input"
    assert text_event["text"] == "go"
    assert enter_event["kind"] == "key_press"
    assert enter_event["key"] == "enter"


def test_mouse_click_inside_ignore_rect_is_skipped(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    hub_rect = (0, 0, 500, 500)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start(ignore_rect=hub_rect)
        try:
            from pynput.mouse import Button

            session._on_mouse_click(100, 100, Button.left, True)
            time.sleep(0.5)
            _left_click(session, 900, 900)
        finally:
            session.stop()

    assert session.event_count() == 1
    assert (run_dir / "events" / "event_001.json").is_file()


def test_left_click_uses_press_time_screenshot(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    captures: list[str] = []

    def _track_capture(x: int, y: int, dest) -> tuple[str, int, tuple[int, int]]:
        captures.append(str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")
        return str(dest), 1, (0, 0)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_track_capture,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            _left_click(session, 400, 400)
        finally:
            session.stop()

    assert session.event_count() == 1
    assert any(name.endswith("_pending_capture.jpeg") for name in captures)
    assert (run_dir / "screenshots" / "event_001.jpeg").is_file()


def test_text_input_stores_anchor_click_xy_after_pointer_event(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode
            from pynput.mouse import Button

            _left_click(session, 400, 300)
            session._on_key_press(KeyCode.from_char("a"))
            session._on_key_press(Key.enter)
        finally:
            session.stop()

    text_event = json.loads((run_dir / "events" / "event_002.json").read_text(encoding="utf-8"))
    assert text_event["kind"] == "text_input"
    assert text_event["anchor_click_xy"] == [400, 300]


def test_pointer_click_persists_window_change(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    before = (
        WindowInfo(
            hwnd=100,
            title="Google Chrome",
            pid=1,
            left=100,
            top=100,
            width=800,
            height=600,
            is_minimized=False,
            is_maximized=False,
        ),
    )
    after = [
        WindowInfo(
            hwnd=100,
            title="Google Chrome",
            pid=1,
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=True,
            is_maximized=False,
        )
    ]

    snapshot_calls = {"count": 0}

    def _snapshot_side_effect() -> list[WindowInfo]:
        snapshot_calls["count"] += 1
        if snapshot_calls["count"] == 1:
            return list(before)
        return after

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ), patch(
        "src.recorder.capture.snapshot_top_level_windows",
        side_effect=_snapshot_side_effect,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            _left_click(session, 400, 120)
        finally:
            session.stop()

    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["window_change"] == {
        "action": "minimize",
        "title": "Google Chrome",
        "confidence": "high",
    }
    assert raw["target_window_title"] == "Google Chrome"
    assert raw["window_snapshot_debug"]["windows_before_count"] == 1
    assert raw["window_snapshot_debug"]["target_hwnd"] == 100
    assert raw["window_snapshot_debug"]["detection_path"] == "target"
    assert raw["window_snapshot_debug"]["windows_before"][0]["title"] == "Google Chrome"
    assert raw["window_snapshot_debug"]["windows_after"][0]["is_minimized"] is True


def test_finalize_drag_end_screenshot_selects_release_monitor(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "screenshots").mkdir()
    pending = {
        1: (str(run_dir / "screenshots" / "_pending_drag_end_mon1.jpeg"), 1, (0, 0)),
        2: (str(run_dir / "screenshots" / "_pending_drag_end_mon2.jpeg"), 2, (1920, 0)),
    }
    for mon, (path, _, _) in pending.items():
        Path(path).write_bytes(f"monitor-{mon}".encode())

    with patch(
        "src.recorder.capture._monitor_at_point",
        return_value=(2, 1920, 0, 1920, 1080),
    ):
        end_path, end_mon_idx, end_offset = _finalize_drag_end_screenshot(
            run_dir,
            2,
            (2100, 500),
            pending,
            fallback_mon_idx=1,
            fallback_mon_offset=(0, 0),
        )

    assert end_path.endswith("event_002_end.jpeg")
    assert end_mon_idx == 2
    assert end_offset == (1920, 0)
    assert (run_dir / "screenshots" / "event_002_end.jpeg").read_bytes() == b"monitor-2"
    assert not Path(pending[1][0]).exists()
    assert not Path(pending[2][0]).exists()


def test_drag_prefers_pre_captured_monitor_on_release(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    capture_calls = {"all": 0, "end_point": 0}

    def _mock_capture_all(run_dir):
        capture_calls["all"] += 1
        pending = {
            1: (str(run_dir / "screenshots" / "_pending_drag_end_mon1.jpeg"), 1, (0, 0)),
            2: (str(run_dir / "screenshots" / "_pending_drag_end_mon2.jpeg"), 2, (1920, 0)),
        }
        for mon, (path, _, _) in pending.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(f"monitor-{mon}".encode())
        return pending

    def _mock_capture_point(x, y, dest):
        if str(dest).endswith("_end.jpeg"):
            capture_calls["end_point"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fallback")
        return str(dest), 1, (0, 0)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_all_monitors_to_pending",
        side_effect=_mock_capture_all,
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_capture_point,
    ), patch(
        "src.recorder.capture._monitor_at_point",
        return_value=(2, 1920, 0, 1920, 1080),
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(100, 100, Button.left, True)
            session._on_mouse_move(150, 150)
            session._on_mouse_click(2100, 500, Button.left, False)
        finally:
            session.stop()

    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "drag"
    assert capture_calls["all"] >= 1
    assert capture_calls["end_point"] == 0
    assert (run_dir / "screenshots" / "event_001_end.jpeg").read_bytes() == b"monitor-2"
    assert raw["end_monitor_index"] == 2
    assert raw["end_monitor_offset"] == [1920, 0]


def test_drag_end_capture_runs_once_at_drag_start(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    capture_calls = {"all": 0}

    def _mock_capture_all(run_dir):
        capture_calls["all"] += 1
        pending = {
            1: (str(run_dir / "screenshots" / "_pending_drag_end_mon1.jpeg"), 1, (0, 0)),
        }
        Path(pending[1][0]).parent.mkdir(parents=True, exist_ok=True)
        Path(pending[1][0]).write_bytes(b"mon1")
        return pending

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_all_monitors_to_pending",
        side_effect=_mock_capture_all,
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(100, 100, Button.left, True)
            session._on_mouse_move(150, 150)
            session._on_mouse_move(300, 300)
            session._on_mouse_move(450, 450)
            session._on_mouse_click(500, 500, Button.left, False)
        finally:
            session.stop()

    assert capture_calls["all"] == 1
    assert (run_dir / "screenshots" / "event_001_end.jpeg").is_file()

