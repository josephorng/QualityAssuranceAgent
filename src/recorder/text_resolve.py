from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.recorder.models import RecordedEvent
from src.recorder.vision_context import (
    _global_to_local,
    _global_to_local_end,
    build_vision_context_at_point,
    extract_nearest_text,
    resolve_event_screenshot_path,
)

def _vision_for_llm(vision: dict[str, Any]) -> dict[str, Any]:
    return {
        "used_vision": vision.get("used_vision"),
        "candidate_text": vision.get("candidate_text"),
        "local_cursor": vision.get("local_cursor"),
        "candidates": vision.get("candidates"),
        "detection_count": vision.get("detection_count"),
    }


def _strip_ocr_caret(text: str) -> str:
    """Remove a trailing ``|`` that OCR often reads from the text caret."""
    cleaned = text.rstrip()
    if cleaned.endswith("|"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


async def resolve_text_input_text(
    event: RecordedEvent,
    *,
    run_dir: Path,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resolve typing text with OCR on the after-screenshot when available."""
    recorded_text = event.text or ""
    anchor = event.anchor_click_xy or event.cursor_xy
    if anchor is None:
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "ocr_text": None,
            "source": "recorded",
            "meaningful": None,
            "reason": "vision unavailable: no anchor or cursor coordinates",
            "vision": None,
        }

    use_after = resolve_event_screenshot_path(event, run_dir, debug_name="_end") is not None
    if use_after:
        local = _global_to_local_end(event, anchor)
        debug_name = "_end"
        reason = "after-screenshot OCR"
    else:
        local = _global_to_local(event, anchor)
        debug_name = None
        reason = "vision-first OCR"

    vision = build_vision_context_at_point(
        event,
        local_x=local[0],
        local_y=local[1],
        run_dir=run_dir,
        persist_debug=True,
        reference_xy=anchor,
        debug_name=debug_name,
    )
    bgr = vision.pop("bgr", None)
    all_detections = vision.pop("all_detections", [])
    ocr_text: str | None = None
    if bgr is not None and all_detections:
        raw_ocr = extract_nearest_text(bgr, all_detections, local[0], local[1])
        if raw_ocr:
            cleaned = _strip_ocr_caret(raw_ocr)
            ocr_text = cleaned or None

    if ocr_text:
        if log_info is not None:
            shot_label = "after" if use_after else "before"
            log_info(
                f"resolve_text_input_text event={event.index} "
                f"shot={shot_label} recorded={recorded_text!r} resolved={ocr_text!r}"
            )
        return {
            "text": ocr_text,
            "recorded_text": recorded_text,
            "ocr_text": ocr_text,
            "source": "ocr",
            "meaningful": None,
            "reason": reason,
            "vision": _vision_for_llm(vision),
        }

    return {
        "text": recorded_text,
        "recorded_text": recorded_text,
        "ocr_text": None,
        "source": "recorded",
        "meaningful": None,
        "reason": "vision produced no text; kept recorded text",
        "vision": _vision_for_llm(vision),
    }


def event_with_resolved_text(event: RecordedEvent, resolved: dict[str, Any]) -> RecordedEvent:
    """Return a copy of ``event`` with ``text`` replaced by the resolved value."""
    return replace(event, text=resolved["text"])
