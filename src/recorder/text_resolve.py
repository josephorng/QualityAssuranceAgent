from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.recorder.models import RecordedEvent, screenshot_path_for_event_end
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
    """Resolve typing text, preferring recorded keystrokes; OCR kept as alternate."""
    recorded_text = event.text or ""
    anchor = event.cursor_xy or event.anchor_click_xy
    if anchor is None:
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "ocr_text": None,
            "source": "recorded",
            "meaningful": None,
            "reason": "vision unavailable: no cursor or anchor coordinates",
            "vision": None,
        }

    # Prefer the dedicated typing end frame (captured at focus on the typing monitor).
    # Shared next-action screenshots can be on another monitor and break OCR anchoring.
    typing_end = screenshot_path_for_event_end(run_dir, event.index)
    if typing_end.is_file():
        local = _global_to_local(event, anchor)
        debug_name = "_end"
        ocr_reason = "after-screenshot OCR"
        ocr_image_path = str(typing_end)
    else:
        use_after = resolve_event_screenshot_path(event, run_dir, debug_name="_end") is not None
        if use_after:
            local = _global_to_local_end(event, anchor)
            debug_name = "_end"
            ocr_reason = "after-screenshot OCR"
            ocr_image_path = None
        else:
            local = _global_to_local(event, anchor)
            debug_name = None
            ocr_reason = "vision-first OCR"
            ocr_image_path = None

    vision = build_vision_context_at_point(
        event,
        local_x=local[0],
        local_y=local[1],
        run_dir=run_dir,
        persist_debug=True,
        reference_xy=anchor,
        debug_name=debug_name,
        image_path=ocr_image_path,
    )
    bgr = vision.pop("bgr", None)
    all_detections = vision.pop("all_detections", [])
    ocr_text: str | None = None
    if bgr is not None and all_detections:
        raw_ocr = extract_nearest_text(bgr, all_detections, local[0], local[1])
        if raw_ocr:
            cleaned = _strip_ocr_caret(raw_ocr)
            ocr_text = cleaned or None

    vision_for_llm = _vision_for_llm(vision)
    used_after = debug_name == "_end"
    if recorded_text:
        if log_info is not None:
            shot_label = "after" if used_after else "before"
            log_info(
                f"resolve_text_input_text event={event.index} "
                f"shot={shot_label} recorded={recorded_text!r} ocr={ocr_text!r} "
                f"resolved={recorded_text!r} source=recorded"
            )
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "ocr_text": ocr_text,
            "source": "recorded",
            "meaningful": None,
            "reason": (
                f"prefer recorded text; {ocr_reason} available as alternate"
                if ocr_text
                else "prefer recorded text; vision produced no OCR alternate"
            ),
            "vision": vision_for_llm,
        }

    if ocr_text:
        if log_info is not None:
            shot_label = "after" if used_after else "before"
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
            "reason": f"recorded text empty; used {ocr_reason}",
            "vision": vision_for_llm,
        }

    return {
        "text": recorded_text,
        "recorded_text": recorded_text,
        "ocr_text": None,
        "source": "recorded",
        "meaningful": None,
        "reason": "vision produced no text; kept recorded text",
        "vision": vision_for_llm,
    }


def event_with_resolved_text(event: RecordedEvent, resolved: dict[str, Any]) -> RecordedEvent:
    """Return a copy of ``event`` with ``text`` replaced by the resolved value."""
    return replace(event, text=resolved["text"])
