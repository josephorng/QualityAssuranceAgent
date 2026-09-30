from __future__ import annotations

from src.recorder.models import RecordedEvent


def test_recorded_event_round_trip() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="click",
        cursor_xy=(100, 200),
        button="left",
        screenshot_path="runs/test/screenshots/event_001.jpeg",
        monitor_index=1,
        monitor_offset=(0, 0),
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.index == 1
    assert restored.kind == "click"
    assert restored.cursor_xy == (100, 200)
    assert restored.button == "left"
    assert restored.monitor_offset == (0, 0)


def test_recorded_event_round_trip_anchor_click_xy() -> None:
    event = RecordedEvent(
        index=2,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="text_input",
        text="你好",
        anchor_click_xy=(400, 300),
        screenshot_path="runs/test/screenshots/event_002.jpeg",
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.anchor_click_xy == (400, 300)
    assert restored.text == "你好"


def test_recorded_event_round_trip_window_change() -> None:
    event = RecordedEvent(
        index=3,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="click",
        cursor_xy=(100, 100),
        window_change={"action": "minimize", "title": "Chrome", "confidence": "high"},
        target_window_title="Chrome",
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.window_change == {
        "action": "minimize",
        "title": "Chrome",
        "confidence": "high",
    }
    assert restored.target_window_title == "Chrome"


def test_recorded_event_round_trip_drag() -> None:
    event = RecordedEvent(
        index=4,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="drag",
        cursor_xy=(100, 200),
        end_xy=(300, 400),
        button="left",
        screenshot_path="runs/test/screenshots/event_004.jpeg",
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.kind == "drag"
    assert restored.cursor_xy == (100, 200)
    assert restored.end_xy == (300, 400)
    assert restored.button == "left"


def test_recorded_event_round_trip_modifiers() -> None:
    event = RecordedEvent(
        index=6,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="click",
        cursor_xy=(100, 200),
        button="left",
        modifiers=["ctrl", "shift"],
        screenshot_path="runs/test/screenshots/event_006.jpeg",
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.modifiers == ["ctrl", "shift"]
    assert restored.button == "left"


def test_recorded_event_round_trip_drag_end_screenshot() -> None:
    event = RecordedEvent(
        index=5,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="drag",
        cursor_xy=(100, 200),
        end_xy=(300, 400),
        screenshot_path="runs/test/screenshots/event_005.jpeg",
        end_screenshot_path="runs/test/screenshots/event_005_end.jpeg",
        monitor_offset=(0, 0),
        end_monitor_offset=(0, 0),
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.end_screenshot_path == "runs/test/screenshots/event_005_end.jpeg"
    assert restored.end_monitor_offset == (0, 0)


def test_recorded_event_round_trip_drag_click_windows() -> None:
    start_window = {"hwnd": 1, "title": "圖片", "rect": [0, 0, 400, 300]}
    end_window = {"hwnd": 2, "title": "下載", "rect": [10, 20, 500, 400]}
    event = RecordedEvent(
        index=7,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="drag",
        cursor_xy=(100, 200),
        end_xy=(300, 400),
        click_window=start_window,
        end_click_window=end_window,
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.click_window == start_window
    assert restored.end_click_window == end_window
    assert "end_click_window" not in RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
    ).to_dict()


def test_recorded_event_round_trip_hold() -> None:
    event = RecordedEvent(
        index=6,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="hold",
        cursor_xy=(100, 200),
        button="right",
        duration_seconds=1.25,
        screenshot_path="runs/test/screenshots/event_006.jpeg",
        monitor_offset=(0, 0),
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.kind == "hold"
    assert restored.button == "right"
    assert restored.duration_seconds == 1.25


def test_recorded_event_round_trip_click_count() -> None:
    event = RecordedEvent(
        index=7,
        timestamp_utc="2026-07-02T00:00:00+00:00",
        kind="click",
        cursor_xy=(100, 200),
        button="left",
        click_count=4,
        screenshot_path="runs/test/screenshots/event_007.jpeg",
    )
    restored = RecordedEvent.from_dict(event.to_dict())
    assert restored.kind == "click"
    assert restored.click_count == 4
