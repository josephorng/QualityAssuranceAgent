from __future__ import annotations

from src.recorder.coalesce import coalesce_consecutive_text_inputs
from src.recorder.models import RecordedEvent


def _text_event(index: int, text: str) -> RecordedEvent:
    return RecordedEvent(
        index=index,
        timestamp_utc="t",
        kind="text_input",
        text=text,
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


def test_coalesce_consecutive_text_inputs_keeps_first_anchor_and_last_screenshot() -> None:
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
        ),
    ]
    merged = coalesce_consecutive_text_inputs(events)
    assert len(merged) == 1
    assert merged[0].text == "ab"
    assert merged[0].anchor_click_xy == (10, 20)
    assert merged[0].screenshot_path == "last.jpeg"
    assert merged[0].monitor_index == 2
    assert merged[0].monitor_offset == (100, 0)
