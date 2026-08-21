from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import (
    RecordingSession,
    _DOUBLE_CLICK_INTERVAL_S,
    _HOLD_THRESHOLD_S,
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

    def _fake_resolve_typing_screen_xy(
        *,
        last_click_xy=None,
        mouse_xy=None,
        **_kwargs,
    ):
        return last_click_xy or mouse_xy

    with patch("src.recorder.capture.mouse.Listener", DummyListener), patch(
        "src.recorder.capture.keyboard.Listener",
        DummyListener,
    ), patch("src.recorder.capture.snapshot_top_level_windows", return_value=[]), patch(
        "src.recorder.capture.settle_delay_for_click",
        return_value=0.0,
    ), patch(
        "src.recorder.capture.resolve_typing_screen_xy",
        side_effect=_fake_resolve_typing_screen_xy,
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


def test_event_timestamp_is_captured_before_worker_persistence(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    captured_items = []

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        session.start()
        try:
            with patch.object(session, "_queue_event", side_effect=captured_items.append), patch(
                "src.recorder.capture.utc_now_iso",
                return_value="2026-07-30T03:00:03.125000+00:00",
            ):
                from pynput.mouse import Button

                session._on_mouse_click(900, 700, Button.right, True)
                session._on_mouse_click(900, 700, Button.right, False)
        finally:
            session.stop()

    assert len(captured_items) == 1
    assert captured_items[0].timestamp_utc == "2026-07-30T03:00:03.125000+00:00"
    assert captured_items[0].kind == "right_click"


def test_typing_burst_coalesced_into_one_event(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    captures: list[tuple[int, tuple[int, int]]] = []

    def track_capture(run_dir, index, cursor_xy, dest=None):
        captures.append((index, cursor_xy))
        path = dest if dest is not None else run_dir / "screenshots" / f"event_{index:03d}.jpeg"
        return str(path), 1, (0, 0)

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
            assert len(captures) == 1
            assert captures[0][0] == 1
            session.stop()
        finally:
            if session.is_active():
                session.stop()

    assert session.event_count() == 1
    assert len(captures) == 2
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "chrome"
    assert raw["screenshot_path"].endswith("event_001.jpeg")
    assert raw["end_screenshot_path"].endswith("event_001_end.jpeg")


def test_numpad_vk_keys_recorded_as_text_input(tmp_path) -> None:
    """Num Lock digits often arrive as vk-only KeyCodes with char=None."""
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import KeyCode

            # VK_NUMPAD1..3 and VK_DECIMAL
            for vk in (97, 98, 99, 110):
                session._on_key_press(KeyCode.from_vk(vk))
            session.stop()
        finally:
            if session.is_active():
                session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "123."


def test_ctrl_letter_hotkey_uses_letter_not_control_char(tmp_path) -> None:
    """Windows pynput reports Ctrl+A as char='\\x01'; store 'a' instead."""
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.ctrl_l)
            # Real Windows listener shape: vk=A with control-char payload.
            session._on_key_press(KeyCode(vk=65, char="\x01"))
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hotkey"
    assert raw["keys"] == ["ctrl", "a"]


def test_ctrl_c_hotkey_from_control_char_only(tmp_path) -> None:
    """Even without vk, ASCII control chars map back to the letter."""
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.ctrl_l)
            session._on_key_press(KeyCode.from_char("\x03"))
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hotkey"
    assert raw["keys"] == ["ctrl", "c"]


def test_shift_letter_records_as_text_input(tmp_path) -> None:
    """Shift+letter is uppercase typing, not a Shift+A hotkey."""
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.shift_l)
            session._on_key_press(KeyCode(vk=65, char="A"))
            session._on_key_press(KeyCode(vk=66, char="B"))
            session._on_key_release(Key.shift_l)
            session._on_key_press(KeyCode.from_char("c"))
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "ABc"


def test_shift_symbol_records_as_text_input(tmp_path) -> None:
    """Shift+1 → '!' should coalesce into text_input."""
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.shift)
            session._on_key_press(KeyCode(vk=49, char="!"))
            session._on_key_release(Key.shift)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "!"


def test_shift_tab_still_hotkey(tmp_path) -> None:
    """Shift+Tab is a chord, not typing."""
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key

            session._on_key_press(Key.shift_r)
            session._on_key_press(Key.tab)
            session._on_key_release(Key.shift_r)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hotkey"
    assert raw["keys"] == ["shift", "tab"]


def test_ctrl_shift_letter_still_hotkey(tmp_path) -> None:
    """Ctrl+Shift+S remains a hotkey even though Shift is held."""
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.ctrl_l)
            session._on_key_press(Key.shift_l)
            session._on_key_press(KeyCode(vk=83, char="S"))
            session._on_key_release(Key.shift_l)
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hotkey"
    assert raw["keys"] == ["ctrl", "shift", "S"]


def test_ctrl_v_with_text_records_as_text_input(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ), patch(
        "src.recorder.capture.pyperclip.paste",
        return_value="hello world",
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.ctrl_l)
            session._on_key_press(KeyCode(vk=86, char="\x16"))
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "hello world"


def test_ctrl_v_empty_clipboard_stays_hotkey(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ), patch(
        "src.recorder.capture.pyperclip.paste",
        return_value="",
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.ctrl_l)
            session._on_key_press(KeyCode(vk=86, char="\x16"))
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hotkey"
    assert raw["keys"] == ["ctrl", "v"]


def test_ctrl_v_clipboard_read_failure_stays_hotkey(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ), patch(
        "src.recorder.capture.pyperclip.paste",
        side_effect=RuntimeError("clipboard locked"),
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.ctrl_l)
            session._on_key_press(KeyCode(vk=86, char="\x16"))
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hotkey"
    assert raw["keys"] == ["ctrl", "v"]


def test_ctrl_shift_v_with_text_records_as_text_input(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ), patch(
        "src.recorder.capture.pyperclip.paste",
        return_value="plain text",
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(Key.ctrl_l)
            session._on_key_press(Key.shift_l)
            session._on_key_press(KeyCode(vk=86, char="V"))
            session._on_key_release(Key.shift_l)
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "plain text"


def test_paste_coalesces_with_prior_typing(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ), patch(
        "src.recorder.capture.pyperclip.paste",
        return_value=" world",
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(KeyCode.from_char("h"))
            session._on_key_press(KeyCode.from_char("i"))
            session._on_key_press(Key.ctrl_l)
            session._on_key_press(KeyCode(vk=86, char="\x16"))
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "hi world"


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


def test_backspace_edits_pending_text_input(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(KeyCode.from_char("a"))
            session._on_key_press(KeyCode.from_char("b"))
            session._on_key_press(Key.backspace)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "a"


def test_left_arrow_and_delete_edit_pending_text_input(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            for ch in "abc":
                session._on_key_press(KeyCode.from_char(ch))
            session._on_key_press(Key.left)
            session._on_key_press(Key.left)
            session._on_key_press(Key.delete)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "ac"


def test_left_arrow_then_type_inserts_at_caret(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(KeyCode.from_char("a"))
            session._on_key_press(KeyCode.from_char("b"))
            session._on_key_press(Key.left)
            session._on_key_press(KeyCode.from_char("X"))
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "aXb"


def test_home_and_end_edit_pending_text_input(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            for ch in "abc":
                session._on_key_press(KeyCode.from_char(ch))
            session._on_key_press(Key.home)
            session._on_key_press(KeyCode.from_char("X"))
            session._on_key_press(Key.end)
            session._on_key_press(KeyCode.from_char("Y"))
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "text_input"
    assert raw["text"] == "XabcY"


def test_backspace_without_typing_records_key_press(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key

            session._on_key_press(Key.backspace)
            session._on_key_press(Key.left)
            session._on_key_press(Key.home)
            session._on_key_press(Key.end)
        finally:
            session.stop()

    assert session.event_count() == 4
    keys = []
    for i in range(1, 5):
        raw = json.loads((run_dir / "events" / f"event_{i:03d}.json").read_text(encoding="utf-8"))
        assert raw["kind"] == "key_press"
        keys.append(raw["key"])
    assert keys == ["backspace", "left", "home", "end"]


def test_delete_all_pending_chars_emits_no_text_input(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            session._on_key_press(KeyCode.from_char("a"))
            session._on_key_press(KeyCode.from_char("b"))
            session._on_key_press(Key.backspace)
            session._on_key_press(Key.backspace)
        finally:
            session.stop()

    assert session.event_count() == 0
    assert not (run_dir / "events" / "event_001.json").is_file()


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


def test_text_input_end_uses_typing_focus_end_shot(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    ocr_points: list[tuple[int, int]] = []

    def _track(x: int, y: int, dest):
        name = Path(dest).name
        if name.endswith("_end.jpeg"):
            ocr_points.append((x, y))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")
        return str(dest), 1, (1920, -1)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture.resolve_typing_screen_xy",
        return_value=(2416, 240),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_track,
    ), patch(
        "src.recorder.capture._monitor_at_point",
        return_value=(1, 1920, -1, 1920, 1080),
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import KeyCode

            session._on_key_press(KeyCode.from_char("a"))
            _left_click(session, 500, 500)
        finally:
            session.stop()

    assert session.event_count() == 2
    text_raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    click_raw = json.loads((run_dir / "events" / "event_002.json").read_text(encoding="utf-8"))
    assert text_raw["kind"] == "text_input"
    assert click_raw["kind"] == "click"
    assert text_raw["end_screenshot_path"].endswith("event_001_end.jpeg")
    assert text_raw["end_monitor_offset"] == [1920, -1]
    assert text_raw["cursor_xy"] == [2416, 240]
    assert (run_dir / "screenshots" / "event_001_end.jpeg").is_file()
    assert ocr_points
    assert ocr_points[0] == (2416, 240)


def test_text_input_stores_before_on_first_key_and_after_on_flush(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    dests: list[str] = []

    def _track(x: int, y: int, dest):
        dests.append(Path(dest).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")
        return str(dest), 1, (0, 0)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_track,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import KeyCode

            session._on_key_press(KeyCode.from_char("a"))
            session._on_key_press(KeyCode.from_char("b"))
            session.stop()
        finally:
            if session.is_active():
                session.stop()

    assert dests[0] == "event_001.jpeg"
    assert "event_001_end.jpeg" in dests
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["text"] == "ab"
    assert raw["screenshot_path"].endswith("event_001.jpeg")
    assert raw["end_screenshot_path"].endswith("event_001_end.jpeg")


def test_text_input_uses_focus_point_not_anchor_click(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture.pyautogui.position",
        return_value=type("P", (), {"x": 100, "y": 100})(),
    ), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ), patch(
        "src.recorder.capture.resolve_typing_screen_xy",
        return_value=(555, 666),
    ) as resolve_mock:
        run_dir = session.start()
        try:
            from pynput.keyboard import Key, KeyCode

            _left_click(session, 400, 300)
            session._on_key_press(KeyCode.from_char("a"))
            session._on_key_press(Key.enter)
        finally:
            session.stop()

    text_event = json.loads((run_dir / "events" / "event_002.json").read_text(encoding="utf-8"))
    assert text_event["kind"] == "text_input"
    assert text_event["cursor_xy"] == [555, 666]
    assert text_event["anchor_click_xy"] is None
    resolve_mock.assert_called()
    assert resolve_mock.call_args.kwargs["last_click_xy"] == (400, 300)


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


def test_ctrl_click_records_modifiers(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key
            from pynput.mouse import Button

            session._on_key_press(Key.ctrl_l)
            _left_click(session, 120, 240)
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "click"
    assert raw["modifiers"] == ["ctrl"]


def test_shift_double_click_records_modifiers(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key
            from pynput.mouse import Button

            session._on_key_press(Key.shift_l)
            session._on_mouse_click(50, 60, Button.left, True)
            session._on_mouse_click(50, 60, Button.left, False)
            session._on_mouse_click(50, 60, Button.left, True)
            session._on_mouse_click(50, 60, Button.left, False)
            session._on_key_release(Key.shift_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "double_click"
    assert raw["modifiers"] == ["shift"]


def test_left_hold_records_hold_event(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(120, 240, Button.left, True)
            time.sleep(_HOLD_THRESHOLD_S + 0.05)
            session._on_mouse_click(120, 240, Button.left, False)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hold"
    assert raw["button"] == "left"
    assert raw["duration_seconds"] >= _HOLD_THRESHOLD_S


def test_short_left_press_still_records_click(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(120, 240, Button.left, True)
            time.sleep(0.05)
            session._on_mouse_click(120, 240, Button.left, False)
            time.sleep(_DOUBLE_CLICK_INTERVAL_S + 0.05)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "click"
    assert raw.get("duration_seconds") is None


def test_left_hold_with_drag_still_records_drag(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(100, 100, Button.left, True)
            time.sleep(_HOLD_THRESHOLD_S + 0.05)
            session._on_mouse_move(150, 150)
            session._on_mouse_click(200, 200, Button.left, False)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "drag"


def test_right_hold_records_hold_event(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.mouse import Button

            session._on_mouse_click(300, 400, Button.right, True)
            time.sleep(_HOLD_THRESHOLD_S + 0.05)
            session._on_mouse_click(300, 400, Button.right, False)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hold"
    assert raw["button"] == "right"
    assert raw["duration_seconds"] >= _HOLD_THRESHOLD_S


def test_ctrl_hold_records_modifiers(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import Key
            from pynput.mouse import Button

            session._on_key_press(Key.ctrl_l)
            session._on_mouse_click(120, 240, Button.left, True)
            time.sleep(_HOLD_THRESHOLD_S + 0.05)
            session._on_mouse_click(120, 240, Button.left, False)
            session._on_key_release(Key.ctrl_l)
        finally:
            session.stop()

    assert session.event_count() == 1
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["kind"] == "hold"
    assert raw["modifiers"] == ["ctrl"]


def test_begin_stop_and_finalize_stop_write_session(tmp_path) -> None:
    session = RecordingSession(runs_root=tmp_path)

    with _default_capture_window_patches(), patch(
        "src.recorder.capture._capture_screenshot_at_point",
        side_effect=_mock_screenshot,
    ):
        run_dir = session.start()
        try:
            from pynput.keyboard import KeyCode

            session._on_key_press(KeyCode.from_char("x"))
        finally:
            assert session.is_active()
            hinted = session.begin_stop()
            assert hinted == run_dir
            assert not session.is_active()
            assert session.is_finalizing()
            finalized = session.finalize_stop()

    assert finalized == run_dir
    assert not session.is_finalizing()
    assert (run_dir / "session.json").is_file()
    assert session.event_count() == 1

