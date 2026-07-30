from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.recorder.models import RecordedEvent
from src.recorder.vision_context import (
    _global_to_local,
    build_vision_context_at_point,
    extract_nearest_text,
)

def _vision_for_llm(vision: dict[str, Any]) -> dict[str, Any]:
    return {
        "used_vision": vision.get("used_vision"),
        "candidate_text": vision.get("candidate_text"),
        "local_cursor": vision.get("local_cursor"),
        "candidates": vision.get("candidates"),
        "detection_count": vision.get("detection_count"),
    }


async def resolve_text_input_text(
    event: RecordedEvent,
    *,
    run_dir: Path,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resolve typing text with vision first, falling back to the recorded keystrokes."""
    recorded_text = event.text or ""
    anchor = event.anchor_click_xy or event.cursor_xy
    if anchor is None:
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "source": "recorded",
            "meaningful": None,
            "reason": "vision unavailable: no anchor or cursor coordinates",
            "vision": None,
        }

    local = _global_to_local(event, anchor)
    vision = build_vision_context_at_point(
        event,
        local_x=local[0],
        local_y=local[1],
        run_dir=run_dir,
        persist_debug=True,
        reference_xy=anchor,
    )
    bgr = vision.pop("bgr", None)
    all_detections = vision.pop("all_detections", [])
    ocr_text: str | None = None
    if bgr is not None and all_detections:
        ocr_text = extract_nearest_text(bgr, all_detections, local[0], local[1])

    if ocr_text:
        if log_info is not None:
            log_info(
                f"resolve_text_input_text event={event.index} "
                f"vision_first recorded={recorded_text!r} resolved={ocr_text!r}"
            )
        return {
            "text": ocr_text,
            "recorded_text": recorded_text,
            "source": "ocr",
            "meaningful": None,
            "reason": "vision-first OCR",
            "vision": _vision_for_llm(vision),
        }

    return {
        "text": recorded_text,
        "recorded_text": recorded_text,
        "source": "recorded",
        "meaningful": None,
        "reason": "vision produced no text; kept recorded text",
        "vision": _vision_for_llm(vision),
    }


def event_with_resolved_text(event: RecordedEvent, resolved: dict[str, Any]) -> RecordedEvent:
    """Return a copy of ``event`` with ``text`` replaced by the resolved value."""
    return replace(event, text=resolved["text"])
