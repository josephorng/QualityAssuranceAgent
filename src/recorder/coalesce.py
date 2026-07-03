from __future__ import annotations

from src.recorder.models import RecordedEvent


def coalesce_consecutive_text_inputs(events: list[RecordedEvent]) -> list[RecordedEvent]:
    """Merge adjacent ``text_input`` events into one event with concatenated text."""
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
                screenshot_path=event.screenshot_path or prev.screenshot_path,
                monitor_index=event.monitor_index if event.monitor_index is not None else prev.monitor_index,
                monitor_offset=event.monitor_offset if event.monitor_offset is not None else prev.monitor_offset,
                anchor_click_xy=prev.anchor_click_xy or event.anchor_click_xy,
            )
            continue
        merged.append(event)
    return merged
