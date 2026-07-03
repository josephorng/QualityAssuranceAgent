from __future__ import annotations

import json
import time
from unittest.mock import patch

from src.recorder.capture import RecordingSession, _DOUBLE_CLICK_INTERVAL_S


def _mock_screenshot(*_args, **_kwargs) -> tuple[str, int, tuple[int, int]]:
    dest = _args[2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"fake")
    return str(dest), 1, (0, 0)


def test_keyboard_events_not_filtered_by_ignore_rect(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    hub_rect = (0, 0, 2000, 1200)

    with patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
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

    with patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import KeyCode

            for ch in "chrome":
                session._on_key_press(KeyCode.from_char(ch))
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "chrome"


def test_slow_typing_stays_one_event_until_other_operation(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
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

    with patch("src.recorder.capture.pyautogui.position", return_value=type("P", (), {"x": 100, "y": 100})()), patch(
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

    with patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start(ignore_rect=hub_rect)
        try:
            from pynput.mouse import Button

            session._on_mouse_click(100, 100, Button.left, True)
            time.sleep(0.5)
            session._on_mouse_click(900, 900, Button.left, True)
            time.sleep(0.5)
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

    with patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_track_capture,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(400, 400, Button.left, True)
            time.sleep(_DOUBLE_CLICK_INTERVAL_S + 0.05)
        finally:
            session.stop()

    assert session.event_count() == 1
    assert any(name.endswith("_pending_capture.jpeg") for name in captures)
    assert (run_dir / "screenshots" / "event_001.jpeg").is_file()
