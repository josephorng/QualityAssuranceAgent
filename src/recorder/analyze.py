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
    candidate_offset_for_instruction,
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

_DRAG_DESTINATION_SUFFIX_RE = re.compile(r"(文字|圖示|檔案|資料夾|按鈕)*")
_OFFSET_ALREADY_PRESENT_RE = re.compile(r"個像素")


def _parse_instruction_reply(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    instruction = data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction missing or empty")
    return {"instruction": instruction.strip()}


def enrich_drag_instruction_offset(
    instruction: str,
    destination: dict[str, Any],
) -> str:
    """Append exact relative pixel offset to a drag instruction when missing."""
    if "拖到" not in instruction or _OFFSET_ALREADY_PRESENT_RE.search(instruction):
        return instruction

    match = re.search(r"拖到「([^」]+)」", instruction)
    if not match:
        return instruction

    offset_phrase = candidate_offset_for_instruction(destination, match.group(1))
    if not offset_phrase:
        return instruction

    anchor_end = match.end()
    suffix_match = _DRAG_DESTINATION_SUFFIX_RE.match(instruction[anchor_end:])
    insert_at = anchor_end + (suffix_match.end() if suffix_match else 0)
    if instruction[insert_at : insert_at + 3] == "的位置":
        return instruction[:insert_at] + offset_phrase + instruction[insert_at:]
    return instruction[:insert_at] + f"{offset_phrase}的位置" + instruction[insert_at:]


def instruction_for_text_input(text: str) -> str | None:
    """Build a hub-script line for a typing-only recorded event."""
    cleaned = text.strip()
    if not cleaned:
        return None
    return f"輸入「{cleaned}」"


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

    local = vision.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        cursor_x, cursor_y = local[0], local[1]
    else:
        cursor_x, cursor_y = "", ""

    destination = vision.get("destination") if isinstance(vision.get("destination"), dict) else {}
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
        if event.kind == "drag" and destination:
            instruction = enrich_drag_instruction_offset(
                result["instruction"],
                destination,
            )
            return {"instruction": instruction}
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"analyze_event_to_cache failed event={event.index}: {exc}")
        return None
