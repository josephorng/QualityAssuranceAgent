from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cua_mcp.selection_engine import request_json_with_retry
from src.common.prompting import get_prompt
from src.recorder.models import RecordedEvent
from src.recorder.to_cache import event_summary_for_llm
from src.recorder.vision_context import (
    build_vision_context,
    candidate_anchor_name,
    candidate_offset_for_instruction,
    format_drag_candidate_anchor,
    format_drag_destination_offset_hints,
    format_field_context_hint,
)
from src.recorder.window_snapshot import format_window_change_hint, instruction_for_window_change

_INSTRUCTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
    },
    "required": ["instruction"],
}

_DRAG_ANCHOR_RE = re.compile(r"拖到「([^」]+)」")
_DRAG_SOURCE_RE = re.compile(r"^從「[^」]+」(?:文字|圖示|檔案|資料夾|按鈕|元素)*(?=拖到)")
_DRAG_DESTINATION_SUFFIX_RE = re.compile(r"(文字|圖示|檔案|資料夾|按鈕|元素)*")
_DRAG_OFFSET_PHRASE_RE = re.compile(
    r"(?:(?:左方|右方|上方|下方)\d+個像素)(?:、(?:(?:左方|右方|上方|下方)\d+個像素))*"
)


def _parse_instruction_reply(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    instruction = data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction missing or empty")
    return {"instruction": instruction.strip()}


def enrich_drag_instruction_source(
    instruction: str,
    vision: dict[str, Any],
) -> str:
    """Replace the drag source with the nearest OCR/YOLO candidate."""
    if "拖到" not in instruction:
        return instruction

    candidates = vision.get("candidates") or []
    if not candidates:
        return instruction

    anchor = format_drag_candidate_anchor(candidates[0])
    if not anchor:
        return instruction

    match = _DRAG_SOURCE_RE.match(instruction)
    if not match:
        return instruction

    return f"從{anchor}{instruction[match.end():]}"


def enrich_drag_instruction_destination(
    instruction: str,
    destination: dict[str, Any],
) -> str:
    """Replace the drag destination anchor with the nearest OCR/YOLO candidate."""
    if "拖到" not in instruction:
        return instruction

    candidates = destination.get("candidates") or []
    if not candidates:
        return instruction

    anchor = format_drag_candidate_anchor(candidates[0])
    if not anchor:
        return instruction

    match = _DRAG_ANCHOR_RE.search(instruction)
    if not match:
        return instruction

    anchor_end = match.end()
    suffix_match = _DRAG_DESTINATION_SUFFIX_RE.match(instruction[anchor_end:])
    insert_at = anchor_end + (suffix_match.end() if suffix_match else 0)
    remainder = instruction[insert_at:]
    remainder = _DRAG_OFFSET_PHRASE_RE.sub("", remainder, count=1)
    if remainder.startswith("的位置"):
        remainder = remainder[len("的位置") :]
    return instruction[: match.start()] + f"拖到{anchor}" + remainder


def enrich_drag_instruction_offset(
    instruction: str,
    destination: dict[str, Any],
) -> str:
    """Normalize drag instructions to the exact OCR-derived relative pixel offset."""
    if "拖到" not in instruction:
        return instruction

    match = _DRAG_ANCHOR_RE.search(instruction)
    if not match:
        return instruction

    offset_phrase = candidate_offset_for_instruction(destination, match.group(1))
    if not offset_phrase:
        return instruction

    anchor_end = match.end()
    suffix_match = _DRAG_DESTINATION_SUFFIX_RE.match(instruction[anchor_end:])
    insert_at = anchor_end + (suffix_match.end() if suffix_match else 0)
    remainder = instruction[insert_at:]
    remainder = _DRAG_OFFSET_PHRASE_RE.sub("", remainder, count=1)
    if remainder.startswith("的位置"):
        remainder = remainder[len("的位置") :]
    return instruction[:insert_at] + offset_phrase + "的位置" + remainder


def enrich_drag_instruction(
    instruction: str,
    *,
    vision: dict[str, Any],
    destination: dict[str, Any],
) -> str:
    """Normalize drag source, destination, and relative pixel offset from vision."""
    instruction = enrich_drag_instruction_source(instruction, vision)
    instruction = enrich_drag_instruction_destination(instruction, destination)
    return enrich_drag_instruction_offset(instruction, destination)


def instruction_for_text_input(text: str) -> str | None:
    """Build a hub-script line for a typing-only recorded event."""
    cleaned = text.strip()
    if not cleaned:
        return None
    return f"輸入「{cleaned}」"


def instruction_for_drag(
    vision: dict[str, Any],
    destination: dict[str, Any],
) -> str | None:
    """Build a hub-script drag line from nearest OCR/YOLO candidates."""
    source_candidates = vision.get("candidates") or []
    dest_candidates = destination.get("candidates") or []
    if not source_candidates or not dest_candidates:
        return None

    source_anchor = format_drag_candidate_anchor(source_candidates[0])
    dest_anchor = format_drag_candidate_anchor(dest_candidates[0])
    if not source_anchor or not dest_anchor:
        return None

    dest_name = candidate_anchor_name(dest_candidates[0])
    offset_phrase = (
        candidate_offset_for_instruction(destination, dest_name)
        if dest_name
        else None
    )
    if offset_phrase:
        return f"從{source_anchor}拖到{dest_anchor}{offset_phrase}的位置"
    return f"從{source_anchor}拖到{dest_anchor}"


async def analyze_event_to_cache(
    event: RecordedEvent,
    *,
    run_dir: Path,
    vision: dict[str, Any] | None = None,
    log_info: Any = None,
) -> dict[str, Any] | None:
    """Return a hub-script instruction for one recorded event, or None on failure."""
    if event.kind == "text_input":
        instruction = instruction_for_text_input(event.text or "")
        if instruction is not None:
            return {"instruction": instruction}

    if event.window_change:
        deterministic = instruction_for_window_change(event.window_change)
        if deterministic is not None:
            return {"instruction": deterministic}

    if vision is None:
        vision = await build_vision_context(event, run_dir=run_dir, log_info=log_info)

    destination = vision.get("destination") if isinstance(vision.get("destination"), dict) else {}
    if event.kind == "drag":
        drag_instruction = instruction_for_drag(vision, destination)
        if drag_instruction is not None:
            return {"instruction": drag_instruction}

    local = vision.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        cursor_x, cursor_y = local[0], local[1]
    else:
        cursor_x, cursor_y = "", ""

    dest_local = destination.get("local_cursor")
    if isinstance(dest_local, (list, tuple)) and len(dest_local) == 2:
        destination_x, destination_y = dest_local[0], dest_local[1]
    else:
        destination_x, destination_y = "", ""

    field_context = format_field_context_hint(
        vision,
        typed_text=event.text if event.kind == "text_input" else None,
    )
    destination_field_context = destination.get("field_context") or "(none)"
    destination_candidate_text = destination.get("candidate_text") or "(none)"
    if event.kind == "drag":
        destination_offset_hints = (
            destination.get("destination_offset_hints")
            or format_drag_destination_offset_hints(destination)
        )
    else:
        destination_offset_hints = "(not applicable)"

    prompt = get_prompt("recording_action_to_cache").format(
        event_json=event_summary_for_llm(event),
        cursor_x=cursor_x,
        cursor_y=cursor_y,
        candidate_text=vision.get("candidate_text") or "(none)",
        field_context=field_context,
        destination_x=destination_x,
        destination_y=destination_y,
        destination_candidate_text=destination_candidate_text,
        destination_field_context=destination_field_context,
        destination_offset_hints=destination_offset_hints,
        window_change_hint=format_window_change_hint(event.window_change),
    )

    user_content = prompt
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    shot = event.screenshot_path
    if shot and Path(shot).is_file() and event.kind in {
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "scroll",
        "drag",
    }:
        images = [shot]
        if event.kind == "drag":
            end_shot = event.end_screenshot_path
            if end_shot and Path(end_shot).is_file() and end_shot != shot:
                images.append(end_shot)
        messages[0]["images"] = images

    try:
        result = await request_json_with_retry(
            messages=messages,
            response_schema=_INSTRUCTION_RESPONSE_SCHEMA,
            parse_reply=_parse_instruction_reply,
            retry_instruction=get_prompt("recording_action_to_cache_retry"),
            log_info=log_info,
            append_image_sizes=True,
        )
        if event.kind == "drag":
            instruction = enrich_drag_instruction(
                result["instruction"],
                vision=vision,
                destination=destination,
            )
            return {"instruction": instruction}
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"analyze_event_to_cache failed event={event.index}: {exc}")
        return None
