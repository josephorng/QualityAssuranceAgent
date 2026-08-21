from __future__ import annotations

from datetime import datetime

from src.recorder.models import RecordedEvent

_MULTI_CLICK_MAX_GAP_S = 1.0
_MULTI_CLICK_MAX_DIST_PX = 8
_COALESCABLE_CLICK_KINDS = frozenset({"click", "double_click", "triple_click"})


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
