from __future__ import annotations

import inspect
import json
from typing import Any

from cua_mcp.tools import TOOL_FUNCTIONS
from src.recorder.models import RecordedEvent


def _tool_names() -> set[str]:
    return {fn.__name__ for fn in TOOL_FUNCTIONS}


def _required_params(tool_name: str) -> set[str]:
    for fn in TOOL_FUNCTIONS:
        if fn.__name__ == tool_name:
            sig = inspect.signature(fn)
            required: set[str] = set()
            for name, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty and name != "instruction":
                    required.add(name)
            return required
    return set()


def validate_tool_calls(tool_calls: list[dict[str, Any]]) -> str | None:
    """Return an error message when invalid; None when valid."""
    if not tool_calls:
        return "tool_calls is empty"
    valid_names = _tool_names()
    for call in tool_calls:
        if not isinstance(call, dict):
            return "tool call is not an object"
        name = call.get("name")
        if not isinstance(name, str) or name not in valid_names:
            return f"invalid tool name: {name!r}"
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            return f"arguments for {name} must be an object"
        missing = _required_params(name) - set(arguments.keys())
        if missing:
            return f"tool {name} missing required args: {sorted(missing)}"
    return None


def event_summary_for_llm(event: RecordedEvent) -> str:
    payload: dict[str, Any] = {
        "kind": event.kind,
        "button": event.button,
        "key": event.key,
        "keys": event.keys,
        "text": event.text,
        "scroll_delta": event.scroll_delta,
        "cursor_xy": list(event.cursor_xy) if event.cursor_xy else None,
        "end_xy": list(event.end_xy) if event.end_xy else None,
        "window_change": event.window_change,
        "target_window_title": event.target_window_title,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
