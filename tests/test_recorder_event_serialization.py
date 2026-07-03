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
