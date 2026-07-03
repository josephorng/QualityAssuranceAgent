from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from cua_mcp.selection_engine import request_json_with_retry
from src.common.prompting import get_prompt
from src.recorder.models import RecordedEvent
from src.recorder.vision_context import (
    _global_to_local,
    build_vision_context_at_point,
    extract_nearest_text,
)

_MEANINGFUL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "meaningful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["meaningful", "reason"],
}


def _parse_meaningful_reply(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    meaningful = data.get("meaningful")
    reason = data.get("reason")
    if not isinstance(meaningful, bool):
        raise ValueError("meaningful must be a boolean")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")
    return {"meaningful": meaningful, "reason": reason.strip()}


async def _check_text_meaningful(
    event: RecordedEvent,
    *,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    prompt = get_prompt("recording_text_meaningful_check").format(
        recorded_text=event.text or "",
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    shot = event.screenshot_path
    if shot and Path(shot).is_file():
        messages[0]["images"] = [shot]

    return await request_json_with_retry(
        messages=messages,
        response_schema=_MEANINGFUL_RESPONSE_SCHEMA,
        parse_reply=_parse_meaningful_reply,
        retry_instruction=get_prompt("recording_text_meaningful_check_retry"),
        log_info=log_info,
        append_image_sizes=True,
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
    """Resolve final typing text for analysis, with OCR fallback when capture is unreliable."""
    recorded_text = event.text or ""
    if not recorded_text.strip():
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "source": "recorded",
            "meaningful": True,
            "reason": "empty recorded text",
            "vision": None,
        }

    try:
        check = await _check_text_meaningful(event, log_info=log_info)
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"resolve_text_input_text meaningful check failed event={event.index}: {exc}")
        check = {"meaningful": True, "reason": "meaningful check failed; keeping recorded text"}

    if check["meaningful"]:
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "source": "recorded",
            "meaningful": True,
            "reason": check.get("reason", ""),
            "vision": None,
        }

    anchor = event.anchor_click_xy
    if anchor is None:
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "source": "recorded",
            "meaningful": False,
            "reason": check.get("reason", ""),
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
                f"ocr_fallback recorded={recorded_text!r} resolved={ocr_text!r}"
            )
        return {
            "text": ocr_text,
            "recorded_text": recorded_text,
            "source": "ocr",
            "meaningful": False,
            "reason": check.get("reason", ""),
            "vision": _vision_for_llm(vision),
        }

    return {
        "text": recorded_text,
        "recorded_text": recorded_text,
        "source": "recorded",
        "meaningful": False,
        "reason": check.get("reason", ""),
        "vision": _vision_for_llm(vision),
    }


def event_with_resolved_text(event: RecordedEvent, resolved: dict[str, Any]) -> RecordedEvent:
    """Return a copy of ``event`` with ``text`` replaced by the resolved value."""
    return replace(event, text=resolved["text"])
