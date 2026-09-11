from __future__ import annotations

from src.recorder.coalesce import (
    coalesce_chinese_ime_candidate_keys,
    coalesce_consecutive_same_location_clicks,
    coalesce_consecutive_text_inputs,
    reclassify_negligible_drags_as_clicks,
    retarget_ime_candidate_end_screenshots,
    text_contains_cjk,
)
from src.recorder.models import RecordedEvent


def _text_event(index: int, text: str) -> RecordedEvent:
    return RecordedEvent(
        index=index,
        timestamp_utc="t",
        kind="text_input",
        text=text,
    )


def _key_press_event(
    index: int,
    key: str,
    *,
    screenshot_path: str = "",
    end_screenshot_path: str = "",
    monitor_index: int | None = None,
    modifiers: list[str] | None = None,
) -> RecordedEvent:
    return RecordedEvent(
        index=index,
        timestamp_utc="t",
        kind="key_press",
        key=key,
        screenshot_path=screenshot_path,
        end_screenshot_path=end_screenshot_path,
        monitor_index=monitor_index,
        modifiers=modifiers,
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


def _drag_event(
    index: int,
    *,
    cursor_xy: tuple[int, int],
    end_xy: tuple[int, int],
    timestamp_utc: str = "2026-08-12T00:00:00+00:00",
    screenshot_path: str = "drag.jpeg",
) -> RecordedEvent:
    return RecordedEvent(
        index=index,
        timestamp_utc=timestamp_utc,
        kind="drag",
        cursor_xy=cursor_xy,
        end_xy=end_xy,
        button="left",
        screenshot_path=screenshot_path,
        end_screenshot_path="drag_end.jpeg",
        end_monitor_index=1,
        end_monitor_offset=(0, 0),
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


def test_text_contains_cjk() -> None:
    assert text_contains_cjk("比特幣的價值")
    assert text_contains_cjk("hello世界")
    assert not text_contains_cjk("1u3wk41u42k7ru8456")
    assert not text_contains_cjk("hello")
    assert not text_contains_cjk("")
    assert not text_contains_cjk(None)


def test_retarget_ime_candidate_end_screenshots_points_at_last_key() -> None:
    events = [
        RecordedEvent(
            index=7,
            timestamp_utc="t",
            kind="text_input",
            text="1u3wk41u42k7ru8456",
            screenshot_path="before.jpeg",
            end_screenshot_path="mid.jpeg",
        ),
        _key_press_event(8, "down", screenshot_path="down.jpeg"),
        _key_press_event(9, "enter", screenshot_path="enter.jpeg", monitor_index=2),
        RecordedEvent(index=10, timestamp_utc="t", kind="click", button="left"),
    ]
    out = retarget_ime_candidate_end_screenshots(events)
    assert len(out) == 4
    assert out[0].end_screenshot_path == "enter.jpeg"
    assert out[0].end_monitor_index == 2
    assert out[1].key == "down"
    assert out[2].key == "enter"


def test_coalesce_chinese_ime_candidate_keys_merges_when_chinese() -> None:
    events = [
        RecordedEvent(
            index=7,
            timestamp_utc="t",
            kind="text_input",
            text="1u3wk41u42k7ru8456",
            screenshot_path="text_before.jpeg",
            end_screenshot_path="text_mid.jpeg",
        ),
        _key_press_event(8, "down", screenshot_path="down1.jpeg"),
        _key_press_event(9, "right", screenshot_path="right.jpeg"),
        _key_press_event(10, "enter", screenshot_path="enter_last.jpeg", monitor_index=2),
        RecordedEvent(index=11, timestamp_utc="t", kind="click", button="left"),
    ]
    merged = coalesce_chinese_ime_candidate_keys(events, {7})
    assert len(merged) == 2
    assert merged[0].kind == "text_input"
    assert merged[0].end_screenshot_path == "enter_last.jpeg"
    assert merged[0].end_monitor_index == 2
    assert merged[1].kind == "click"


def test_coalesce_chinese_ime_keeps_keys_when_not_chinese() -> None:
    events = [
        _text_event(1, "hello"),
        _key_press_event(2, "down", screenshot_path="down.jpeg"),
        _key_press_event(3, "enter", screenshot_path="enter.jpeg"),
    ]
    merged = coalesce_chinese_ime_candidate_keys(events, set())
    assert len(merged) == 3
    assert merged[1].key == "down"
    assert merged[2].key == "enter"


def test_coalesce_chinese_ime_keeps_lone_enter_even_when_chinese() -> None:
    events = [
        RecordedEvent(
            index=1,
            timestamp_utc="t",
            kind="text_input",
            text="搜尋",
            end_screenshot_path="typed.jpeg",
        ),
        _key_press_event(2, "enter", screenshot_path="enter.jpeg"),
    ]
    merged = coalesce_chinese_ime_candidate_keys(events, {1})
    assert len(merged) == 2
    assert merged[1].key == "enter"


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


def test_reclassify_negligible_drag_as_click() -> None:
    # event_007-like: 1px release offset after interim drag arming
    events = [
        _drag_event(7, cursor_xy=(2698, 278), end_xy=(2697, 278)),
    ]
    result = reclassify_negligible_drags_as_clicks(events)
    assert len(result) == 1
    assert result[0].kind == "click"
    assert result[0].cursor_xy == (2698, 278)
    assert result[0].end_xy is None
    assert result[0].end_screenshot_path == ""
    assert result[0].screenshot_path == "drag.jpeg"


def test_reclassify_keeps_real_drag() -> None:
    events = [
        _drag_event(1, cursor_xy=(100, 100), end_xy=(200, 200)),
    ]
    result = reclassify_negligible_drags_as_clicks(events)
    assert result[0].kind == "drag"
    assert result[0].end_xy == (200, 200)


def test_reclassify_boundary_at_threshold_becomes_click() -> None:
    events = [
        _drag_event(1, cursor_xy=(100, 100), end_xy=(108, 100)),
    ]
    result = reclassify_negligible_drags_as_clicks(events)
    assert result[0].kind == "click"


def test_reclassify_just_beyond_threshold_stays_drag() -> None:
    events = [
        _drag_event(1, cursor_xy=(100, 100), end_xy=(109, 100)),
    ]
    result = reclassify_negligible_drags_as_clicks(events)
    assert result[0].kind == "drag"


def test_reclassify_then_coalesce_with_adjacent_click() -> None:
    events = [
        _drag_event(
            1,
            cursor_xy=(100, 200),
            end_xy=(101, 200),
            timestamp_utc="2026-08-12T00:00:00+00:00",
            screenshot_path="a.jpeg",
        ),
        _click_event(2, timestamp_utc="2026-08-12T00:00:00.300000+00:00"),
    ]
    result = coalesce_consecutive_same_location_clicks(
        reclassify_negligible_drags_as_clicks(events)
    )
    assert len(result) == 1
    assert result[0].kind == "double_click"
    assert result[0].screenshot_path == "a.jpeg"
