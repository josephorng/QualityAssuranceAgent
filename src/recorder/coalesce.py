from __future__ import annotations

from datetime import datetime

from src.recorder.models import RecordedEvent

_MULTI_CLICK_MAX_GAP_S = 1.0
_MULTI_CLICK_MAX_DIST_PX = 8
# Match capture._DRAG_THRESHOLD_PX: start/end within this → treat drag as click.
_NEGLIGIBLE_DRAG_DIST_PX = 8
_COALESCABLE_CLICK_KINDS = frozenset({"click", "double_click", "triple_click"})
_IME_CANDIDATE_KEYS = frozenset({"up", "down", "left", "right", "enter"})


def text_contains_cjk(text: str | None) -> bool:
    """True when ``text`` includes CJK ideographs (common Chinese/Japanese/Korean Han)."""
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # Extension A
            or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
        ):
            return True
    return False


def coalesce_consecutive_text_inputs(events: list[RecordedEvent]) -> list[RecordedEvent]:
    """Merge adjacent ``text_input`` events into one event with concatenated text.

    Keep the first burst's before-screenshot and the last burst's after-screenshot.
    """
    if not events:
        return []

    merged: list[RecordedEvent] = []
    for event in events:
        if (
            merged
            and event.kind == "text_input"
            and merged[-1].kind == "text_input"
            and event.text
        ):
            prev = merged[-1]
            merged[-1] = RecordedEvent(
                index=prev.index,
                timestamp_utc=prev.timestamp_utc,
                kind="text_input",
                cursor_xy=prev.cursor_xy,
                text=(prev.text or "") + event.text,
                screenshot_path=prev.screenshot_path or event.screenshot_path,
                monitor_index=(
                    prev.monitor_index if prev.monitor_index is not None else event.monitor_index
                ),
                monitor_offset=(
                    prev.monitor_offset if prev.monitor_offset is not None else event.monitor_offset
                ),
                end_screenshot_path=event.end_screenshot_path or prev.end_screenshot_path,
                end_monitor_index=(
                    event.end_monitor_index
                    if event.end_monitor_index is not None
                    else prev.end_monitor_index
                ),
                end_monitor_offset=(
                    event.end_monitor_offset
                    if event.end_monitor_offset is not None
                    else prev.end_monitor_offset
                ),
                anchor_click_xy=prev.anchor_click_xy or event.anchor_click_xy,
                focus_rect=prev.focus_rect or event.focus_rect,
            )
            continue
        merged.append(event)
    return merged


def _ime_candidate_run_end(events: list[RecordedEvent], text_pos: int) -> int:
    """Return exclusive end index of an IME candidate key run after ``text_pos``."""
    j = text_pos + 1
    while j < len(events):
        nxt = events[j]
        if nxt.kind != "key_press":
            break
        key = str(nxt.key or "").strip().lower()
        if key not in _IME_CANDIDATE_KEYS or nxt.modifiers:
            break
        j += 1
    return j


def _run_has_vertical_nav(events: list[RecordedEvent], start: int, end: int) -> bool:
    return any(
        str(item.key or "").strip().lower() in ("up", "down") for item in events[start:end]
    )


def _end_shot_fields_from_event(
    event: RecordedEvent,
) -> tuple[str, int | None, tuple[int, int] | None]:
    if event.end_screenshot_path:
        return (
            event.end_screenshot_path,
            event.end_monitor_index,
            event.end_monitor_offset,
        )
    if event.screenshot_path:
        return event.screenshot_path, event.monitor_index, event.monitor_offset
    return "", None, None


def retarget_ime_candidate_end_screenshots(
    events: list[RecordedEvent],
) -> list[RecordedEvent]:
    """Point ``text_input`` end shots at the last following IME-nav key frame.

    Does not drop keys. Used before OCR so Chinese commit is visible in the
    typing end frame when a vertical candidate-nav run follows typing.
    """
    if not events:
        return []

    out: list[RecordedEvent] = []
    i = 0
    while i < len(events):
        event = events[i]
        if event.kind != "text_input":
            out.append(event)
            i += 1
            continue
        run_end = _ime_candidate_run_end(events, i)
        if run_end > i + 1 and _run_has_vertical_nav(events, i + 1, run_end):
            end_path, end_mon_idx, end_mon_off = _end_shot_fields_from_event(
                events[run_end - 1]
            )
            if end_path:
                out.append(
                    RecordedEvent(
                        index=event.index,
                        timestamp_utc=event.timestamp_utc,
                        kind="text_input",
                        cursor_xy=event.cursor_xy,
                        text=event.text,
                        screenshot_path=event.screenshot_path,
                        monitor_index=event.monitor_index,
                        monitor_offset=event.monitor_offset,
                        end_screenshot_path=end_path,
                        end_monitor_index=end_mon_idx,
                        end_monitor_offset=end_mon_off,
                        anchor_click_xy=event.anchor_click_xy,
                        focus_rect=event.focus_rect,
                        window_change=event.window_change,
                        target_window_title=event.target_window_title,
                        window_snapshot_debug=event.window_snapshot_debug,
                    )
                )
            else:
                out.append(event)
            out.extend(events[i + 1 : run_end])
            i = run_end
            continue
        out.append(event)
        i += 1
    return out


def coalesce_chinese_ime_candidate_keys(
    events: list[RecordedEvent],
    chinese_event_indexes: set[int],
) -> list[RecordedEvent]:
    """Drop IME candidate nav keys after Chinese ``text_input`` events.

    Merges a following Up/Down/Left/Right/Enter run into the typing step only when
    that step's index is in ``chinese_event_indexes`` and the run includes ≥1
    Up/Down (so English type+Enter submit stays intact).
    """
    if not events:
        return []

    merged: list[RecordedEvent] = []
    i = 0
    while i < len(events):
        event = events[i]
        if event.kind != "text_input" or event.index not in chinese_event_indexes:
            merged.append(event)
            i += 1
            continue

        run_end = _ime_candidate_run_end(events, i)
        run = events[i + 1 : run_end]
        if run and _run_has_vertical_nav(events, i + 1, run_end):
            end_path, end_mon_idx, end_mon_off = _end_shot_fields_from_event(run[-1])
            if not end_path:
                end_path = event.end_screenshot_path
                end_mon_idx = event.end_monitor_index
                end_mon_off = event.end_monitor_offset
            merged.append(
                RecordedEvent(
                    index=event.index,
                    timestamp_utc=event.timestamp_utc,
                    kind="text_input",
                    cursor_xy=event.cursor_xy,
                    text=event.text,
                    screenshot_path=event.screenshot_path,
                    monitor_index=event.monitor_index,
                    monitor_offset=event.monitor_offset,
                    end_screenshot_path=end_path,
                    end_monitor_index=end_mon_idx,
                    end_monitor_offset=end_mon_off,
                    anchor_click_xy=event.anchor_click_xy,
                    focus_rect=event.focus_rect,
                    window_change=event.window_change,
                    target_window_title=event.target_window_title,
                    window_snapshot_debug=event.window_snapshot_debug,
                )
            )
            i = run_end
            continue

        merged.append(event)
        i += 1

    return merged


def _click_weight(event: RecordedEvent) -> int:
    if event.click_count is not None and int(event.click_count) > 0:
        return int(event.click_count)
    if event.kind == "double_click":
        return 2
    if event.kind == "triple_click":
        return 3
    return 1


def _modifiers_key(modifiers: list[str] | None) -> tuple[str, ...]:
    if not modifiers:
        return ()
    return tuple(sorted(str(m).strip().lower() for m in modifiers if str(m).strip()))


def _elapsed_seconds(previous_timestamp_utc: str, current_timestamp_utc: str) -> float | None:
    try:
        previous = datetime.fromisoformat(previous_timestamp_utc.replace("Z", "+00:00"))
        current = datetime.fromisoformat(current_timestamp_utc.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if previous.tzinfo is None or current.tzinfo is None:
        return None
    elapsed = (current - previous).total_seconds()
    return elapsed if elapsed >= 0 else None


def _is_left_button(button: str | None) -> bool:
    return button in (None, "left")


def _can_merge_same_location_clicks(prev: RecordedEvent, event: RecordedEvent) -> bool:
    if prev.kind not in _COALESCABLE_CLICK_KINDS or event.kind not in _COALESCABLE_CLICK_KINDS:
        return False
    if not _is_left_button(prev.button) or not _is_left_button(event.button):
        return False
    if _modifiers_key(prev.modifiers) != _modifiers_key(event.modifiers):
        return False
    if prev.cursor_xy is None or event.cursor_xy is None:
        return False
    px, py = prev.cursor_xy
    cx, cy = event.cursor_xy
    if abs(px - cx) > _MULTI_CLICK_MAX_DIST_PX or abs(py - cy) > _MULTI_CLICK_MAX_DIST_PX:
        return False
    elapsed = _elapsed_seconds(prev.timestamp_utc, event.timestamp_utc)
    if elapsed is None or elapsed > _MULTI_CLICK_MAX_GAP_S:
        return False
    return True


def _kind_and_count_for_total(total: int) -> tuple[str, int | None]:
    if total <= 1:
        return "click", None
    if total == 2:
        return "double_click", None
    if total == 3:
        return "triple_click", None
    return "click", total


def _merge_click_group(group: list[RecordedEvent]) -> RecordedEvent:
    first = group[0]
    total = sum(_click_weight(event) for event in group)
    kind, click_count = _kind_and_count_for_total(total)
    return RecordedEvent(
        index=first.index,
        timestamp_utc=first.timestamp_utc,
        kind=kind,
        cursor_xy=first.cursor_xy,
        button=first.button or "left",
        modifiers=list(first.modifiers) if first.modifiers else None,
        click_count=click_count,
        screenshot_path=first.screenshot_path,
        monitor_index=first.monitor_index,
        monitor_offset=first.monitor_offset,
        anchor_click_xy=first.anchor_click_xy,
        window_change=first.window_change,
        target_window_title=first.target_window_title,
        window_snapshot_debug=first.window_snapshot_debug,
    )


def _drag_start_end_too_close(event: RecordedEvent) -> bool:
    """True when a drag's release is within the negligible-distance threshold of press."""
    if event.kind != "drag":
        return False
    if event.cursor_xy is None or event.end_xy is None:
        return False
    sx, sy = event.cursor_xy
    ex, ey = event.end_xy
    return (
        abs(sx - ex) <= _NEGLIGIBLE_DRAG_DIST_PX
        and abs(sy - ey) <= _NEGLIGIBLE_DRAG_DIST_PX
    )


def _drag_as_click(event: RecordedEvent) -> RecordedEvent:
    """Convert a negligible drag into a left click at the press point."""
    return RecordedEvent(
        index=event.index,
        timestamp_utc=event.timestamp_utc,
        kind="click",
        cursor_xy=event.cursor_xy,
        button=event.button or "left",
        modifiers=list(event.modifiers) if event.modifiers else None,
        screenshot_path=event.screenshot_path,
        monitor_index=event.monitor_index,
        monitor_offset=event.monitor_offset,
        anchor_click_xy=event.anchor_click_xy,
        window_change=event.window_change,
        target_window_title=event.target_window_title,
        window_snapshot_debug=event.window_snapshot_debug,
    )


def reclassify_negligible_drags_as_clicks(
    events: list[RecordedEvent],
) -> list[RecordedEvent]:
    """Turn drag events whose start and end are too close into clicks.

    Capture may mark a press as a drag after interim movement, then release near
    the press point. During analyze those should be treated as clicks.
    """
    if not events:
        return []
    return [
        _drag_as_click(event) if _drag_start_end_too_close(event) else event
        for event in events
    ]


def coalesce_consecutive_same_location_clicks(
    events: list[RecordedEvent],
) -> list[RecordedEvent]:
    """Merge nearby left-clicks at the same spot into one multi-click event.

    Consecutive ``click`` / ``double_click`` / ``triple_click`` events that share
    modifiers, fall within a short time gap, and stay within a few pixels are
    combined (e.g. double_click + click → triple_click, or three clicks →
    triple_click). Counts above three become ``click`` with ``click_count`` set.
    """
    if not events:
        return []

    merged: list[RecordedEvent] = []
    group: list[RecordedEvent] = []

    def flush() -> None:
        nonlocal group
        if not group:
            return
        merged.append(group[0] if len(group) == 1 else _merge_click_group(group))
        group = []

    for event in events:
        if group and _can_merge_same_location_clicks(group[-1], event):
            group.append(event)
            continue
        flush()
        if event.kind in _COALESCABLE_CLICK_KINDS and _is_left_button(event.button):
            group = [event]
        else:
            merged.append(event)
    flush()
    return merged
