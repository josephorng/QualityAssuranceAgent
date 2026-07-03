from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.tools import TOOL_FUNCTIONS
from src.common.prompting import get_prompt
from src.recorder.models import RecordedEvent
from src.recorder.to_cache import event_summary_for_llm, validate_tool_calls
from src.recorder.vision_context import build_vision_context

_CACHE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["instruction", "tool_calls"],
}

_VALID_TOOL_NAMES = sorted({fn.__name__ for fn in TOOL_FUNCTIONS})


def _parse_cache_reply(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    instruction = data.get("instruction")
    tool_calls = data.get("tool_calls")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction missing or empty")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("tool_calls missing or empty")
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("tool call is not an object")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool name missing")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            arguments = {}
        normalized.append({"name": name.strip(), "arguments": dict(arguments)})
    err = validate_tool_calls(normalized)
    if err:
        raise ValueError(err)
    return {"instruction": instruction.strip(), "tool_calls": normalized}


async def analyze_event_to_cache(
    event: RecordedEvent,
    *,
    run_dir: Path,
    vision: dict[str, Any] | None = None,
    log_info: Any = None,
) -> dict[str, Any] | None:
    """Return instruction + tool_calls for one recorded event, or None on failure."""
    if vision is None:
        vision = build_vision_context(event, run_dir=run_dir)

    local = vision.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        cursor_x, cursor_y = local[0], local[1]
    else:
        cursor_x, cursor_y = "", ""

    prompt = get_prompt("recording_action_to_cache").format(
        event_json=event_summary_for_llm(event),
        cursor_x=cursor_x,
        cursor_y=cursor_y,
        candidate_text=vision.get("candidate_text") or "(none)",
        valid_tools=", ".join(_VALID_TOOL_NAMES),
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
    }:
        messages[0]["images"] = [shot]

    try:
        result = await request_json_with_retry(
            messages=messages,
            response_schema=_CACHE_RESPONSE_SCHEMA,
            parse_reply=_parse_cache_reply,
            retry_instruction=get_prompt("recording_action_to_cache_retry"),
            log_info=log_info,
            append_image_sizes=True,
        )
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"analyze_event_to_cache failed event={event.index}: {exc}")
        return None
