from __future__ import annotations

import json
from typing import Any


def extract_json_object_string(raw: str) -> str:
    """Extract the first JSON object from possibly fenced/free-form model output."""
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_json_object(raw: str, *, empty_error: str, decode_error_prefix: str) -> dict[str, Any]:
    json_text = extract_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(empty_error)
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{decode_error_prefix} ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict):
        raise ValueError(f"JSON must be an object, got {type(out).__name__}; preview={preview!r}")
    return out
