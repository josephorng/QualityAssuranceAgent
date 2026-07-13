"""Parse relative pixel offsets from Traditional Chinese mouse instructions."""

from __future__ import annotations

import json
import re
from typing import Any

from cua_mcp.selection_engine import request_json_with_retry
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager

_OFFSET_TOKEN_RE = re.compile(r"(右方|左方|上方|下方)(\d+)個像素")
_TRAILING_CONTEXT_COMMENT_RE = re.compile(r"(?:（[^）]*）)+$")
_INLINE_START_CONTEXT_COMMENT_RE = re.compile(r"（起點附近[^）]*）")

_OFFSET_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anchor": {"type": "string"},
        "dx": {"type": "integer"},
        "dy": {"type": "integer"},
        "nearby": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["anchor", "dx", "dy", "nearby"],
}


def _log_info(text: str) -> None:
    try:
        get_run_state_manager().log_info(text)
    except RuntimeError:
        pass


def _parse_offset_reply(raw: str) -> tuple[str, int, int, list[str]]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    anchor = data.get("anchor")
    dx = data.get("dx")
    dy = data.get("dy")
    nearby = data.get("nearby")
    if not isinstance(anchor, str):
        raise ValueError("anchor must be a string")
    if not isinstance(dx, int) or isinstance(dx, bool):
        raise ValueError("dx must be an integer")
    if not isinstance(dy, int) or isinstance(dy, bool):
        raise ValueError("dy must be an integer")
    if not isinstance(nearby, list) or not all(isinstance(item, str) for item in nearby):
        raise ValueError("nearby must be a list of strings")
    nearby_labels = [item.strip() for item in nearby if item.strip()]
    return anchor.strip(), dx, dy, nearby_labels


def _strip_trailing_context_comment(text: str) -> str:
    """Remove trailing full-width parenthetical nearby-context comments."""
    return _TRAILING_CONTEXT_COMMENT_RE.sub("", text).strip()


def _strip_context_comments(text: str) -> str:
    """Remove inline start and trailing nearby-context comments."""
    text = _strip_trailing_context_comment(text)
    return _INLINE_START_CONTEXT_COMMENT_RE.sub("", text).strip()


def _parse_relative_pixel_offset_regex(instruction: str) -> tuple[str, int, int]:
    """Regex fallback when LLM extraction fails."""
    text = _strip_context_comments((instruction or "").strip())
    if not text:
        return "", 0, 0

    dx = 0
    dy = 0
    for direction, raw_n in _OFFSET_TOKEN_RE.findall(text):
        n = int(raw_n)
        if direction == "右方":
            dx += n
        elif direction == "左方":
            dx -= n
        elif direction == "下方":
            dy += n
        elif direction == "上方":
            dy -= n

    anchor = text
    if anchor.endswith("的位置"):
        anchor = anchor[:-3]
    anchor = _OFFSET_TOKEN_RE.sub("", anchor)
    anchor = re.sub(r"、+$", "", anchor).strip()
    return anchor, dx, dy


async def parse_relative_pixel_offset(
    instruction: str,
) -> tuple[str, int, int, list[str]]:
    """
    Split an instruction into anchor text, relative pixel offset, and nearby labels.

    Uses an LLM to extract anchor/dx/dy/nearby, with a regex fallback for
    anchor/dx/dy on failure (nearby falls back to an empty list).
    Positive dx is right; positive dy is down.

    The raw instruction is passed to the LLM unchanged. Nearby labels come from
    contextual comments such as （附近有…） / （起點附近有…） / （終點附近有…）.
    """
    text = (instruction or "").strip()
    if not text:
        return "", 0, 0, []

    prompt = get_prompt("instruction_relative_offset").format(instruction=text)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        anchor, dx, dy, nearby = await request_json_with_retry(
            messages=messages,
            response_schema=_OFFSET_RESPONSE_SCHEMA,
            parse_reply=_parse_offset_reply,
            retry_instruction=get_prompt("instruction_relative_offset_retry"),
            log_info=lambda message: _log_info(f"parse_relative_pixel_offset: {message}"),
            append_image_sizes=False,
        )
        if not anchor:
            raise ValueError("anchor is empty")
        _log_info(
            "parse_relative_pixel_offset LLM "
            f"anchor={anchor!r} dx={dx} dy={dy} nearby={nearby!r}"
        )
        return anchor, dx, dy, nearby
    except (ValueError, json.JSONDecodeError) as exc:
        anchor, dx, dy = _parse_relative_pixel_offset_regex(text)
        _log_info(
            "parse_relative_pixel_offset regex fallback "
            f"({exc}) anchor={anchor!r} dx={dx} dy={dy} nearby=[]"
        )
        return anchor, dx, dy, []
