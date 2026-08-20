from __future__ import annotations

from src.recorder.coalesce import (
    coalesce_consecutive_same_location_clicks,
    coalesce_consecutive_text_inputs,
)
from src.recorder.models import RecordedEvent


def _text_event(index: int, text: str) -> RecordedEvent:
    return RecordedEvent(
        index=index,
        timestamp_utc="t",
        kind="text_input",
        text=text,
    )


def _click_event(
    index: int,
    *,
    timestamp_utc: str,
    cursor_xy: tuple[int, int] = (100, 200),
    kind: str = "click",
    button: str = "left",
    modifiers: list[str] | None = None,
    screenshot_path: str = "",
) -> RecordedEvent:
    return RecordedEvent(
        index=index,
        timestamp_utc=timestamp_utc,
        kind=kind,
        cursor_xy=cursor_xy,
        button=button,
        modifiers=modifiers,
        screenshot_path=screenshot_path,
    )


def test_coalesce_consecutive_text_inputs_merges_chars() -> None:
    events = [
        _text_event(1, "c"),
        _text_event(2, "h"),
        _text_event(3, "r"),
        _text_event(4, "o"),
        _text_event(5, "m"),
        _text_event(6, "e"),
    ]
    merged = coalesce_consecutive_text_inputs(events)
    assert len(merged) == 1
    assert merged[0].text == "chrome"
    assert merged[0].index == 1


def test_coalesce_consecutive_text_inputs_keeps_breaks() -> None:
    events = [
        _text_event(1, "a"),
        _text_event(2, "b"),
        RecordedEvent(index=3, timestamp_utc="t", kind="click", button="left"),
        _text_event(4, "c"),
        _text_event(5, "d"),
    ]
    merged = coalesce_consecutive_text_inputs(events)
    assert len(merged) == 3
    assert merged[0].text == "ab"
    assert merged[1].kind == "click"
    assert merged[2].text == "cd"


def test_coalesce_consecutive_text_inputs_keeps_first_before_and_last_after() -> None:
    events = [
        RecordedEvent(
            index=1,
            timestamp_utc="t",
            kind="text_input",
            text="a",
            anchor_click_xy=(10, 20),
            screenshot_path="first.jpeg",
            monitor_index=1,
            monitor_offset=(0, 0),
            end_screenshot_path="first_end.jpeg",
            end_monitor_index=1,
            end_monitor_offset=(0, 0),
        ),
        RecordedEvent(
            index=2,
            timestamp_utc="t2",
            kind="text_input",
            text="b",
            anchor_click_xy=(99, 99),
            screenshot_path="last.jpeg",
            monitor_index=2,
            monitor_offset=(100, 0),
            end_screenshot_path="last_end.jpeg",
            end_monitor_index=2,
            end_monitor_offset=(100, 0),
        ),
    ]
    merged = coalesce_consecutive_text_inputs(events)
    assert len(merged) == 1
    assert merged[0].text == "ab"
    assert merged[0].anchor_click_xy == (10, 20)
    assert merged[0].screenshot_path == "first.jpeg"
    assert merged[0].monitor_index == 1
    assert merged[0].monitor_offset == (0, 0)
    assert merged[0].end_screenshot_path == "last_end.jpeg"
    assert merged[0].end_monitor_index == 2
    assert merged[0].end_monitor_offset == (100, 0)


def test_coalesce_same_location_clicks_three_singles_to_triple() -> None:
    events = [
        _click_event(1, timestamp_utc="2026-08-12T00:00:00+00:00", screenshot_path="a.jpeg"),
        _click_event(2, timestamp_utc="2026-08-12T00:00:00.400000+00:00"),
        _click_event(3, timestamp_utc="2026-08-12T00:00:00.800000+00:00"),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 1
    assert merged[0].kind == "triple_click"
    assert merged[0].index == 1
    assert merged[0].click_count is None
    assert merged[0].screenshot_path == "a.jpeg"


def test_coalesce_same_location_clicks_double_plus_click_to_triple() -> None:
    events = [
        _click_event(
            1,
            timestamp_utc="2026-08-12T00:00:00+00:00",
            kind="double_click",
        ),
        _click_event(2, timestamp_utc="2026-08-12T00:00:00.300000+00:00"),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 1
    assert merged[0].kind == "triple_click"


def test_coalesce_same_location_clicks_four_uses_click_count() -> None:
    events = [
        _click_event(1, timestamp_utc="2026-08-12T00:00:00+00:00"),
        _click_event(2, timestamp_utc="2026-08-12T00:00:00.200000+00:00"),
        _click_event(3, timestamp_utc="2026-08-12T00:00:00.400000+00:00"),
        _click_event(4, timestamp_utc="2026-08-12T00:00:00.600000+00:00"),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 1
    assert merged[0].kind == "click"
    assert merged[0].click_count == 4


def test_coalesce_same_location_clicks_keeps_far_apart() -> None:
    events = [
        _click_event(1, timestamp_utc="2026-08-12T00:00:00+00:00"),
        _click_event(2, timestamp_utc="2026-08-12T00:00:02+00:00"),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 2
    assert all(event.kind == "click" for event in merged)


def test_coalesce_same_location_clicks_keeps_distant_coords() -> None:
    events = [
        _click_event(1, timestamp_utc="2026-08-12T00:00:00+00:00", cursor_xy=(100, 100)),
        _click_event(2, timestamp_utc="2026-08-12T00:00:00.200000+00:00", cursor_xy=(200, 100)),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 2


def test_coalesce_same_location_clicks_keeps_modifier_mismatch() -> None:
    events = [
        _click_event(
            1,
            timestamp_utc="2026-08-12T00:00:00+00:00",
            modifiers=["ctrl"],
        ),
        _click_event(2, timestamp_utc="2026-08-12T00:00:00.200000+00:00"),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 2


def test_coalesce_same_location_clicks_preserves_shared_modifiers() -> None:
    events = [
        _click_event(
            1,
            timestamp_utc="2026-08-12T00:00:00+00:00",
            modifiers=["shift"],
        ),
        _click_event(
            2,
            timestamp_utc="2026-08-12T00:00:00.200000+00:00",
            modifiers=["shift"],
        ),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 1
    assert merged[0].kind == "double_click"
    assert merged[0].modifiers == ["shift"]


def test_coalesce_same_location_clicks_ignores_right_click() -> None:
    events = [
        _click_event(1, timestamp_utc="2026-08-12T00:00:00+00:00", button="right", kind="right_click"),
        _click_event(2, timestamp_utc="2026-08-12T00:00:00.200000+00:00", button="right", kind="right_click"),
    ]
    merged = coalesce_consecutive_same_location_clicks(events)
    assert len(merged) == 2
