"""Parse relative pixel offsets from Traditional Chinese mouse instructions."""

from __future__ import annotations

import json
import re
from typing import Any

from cua_mcp.selection_engine import request_json_with_retry
from src.common.nearby_side import (
    NearbyHint,
    extract_nearby_hints_from_instruction,
    normalize_nearby_hints,
    parse_side_schema_value,
)
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager

_OFFSET_TOKEN_RE = re.compile(r"(右方|左方|上方|下方)(\d+)個像素")
_TRAILING_CONTEXT_COMMENT_RE = re.compile(r"(?:（[^）]*）)+$")
_INLINE_START_CONTEXT_COMMENT_RE = re.compile(r"（起點(?:附近|在)[^）]*）")

_OFFSET_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anchor": {"type": "string"},
        "dx": {"type": "integer"},
        "dy": {"type": "integer"},
        "nearby": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "side": {
                        "type": ["string", "null"],
                    },
                },
                "required": ["label", "side"],
            },
        },
    },
    "required": ["anchor", "dx", "dy", "nearby"],
}


def _log_info(text: str) -> None:
    try:
        get_run_state_manager().log_info(text)
    except RuntimeError:
        pass


def _parse_nearby_items(nearby: Any) -> list[NearbyHint]:
    """Accept structured nearby objects or legacy string labels."""
    if not isinstance(nearby, list):
        raise ValueError("nearby must be a list")
    items: list[Any] = []
    for item in nearby:
        if isinstance(item, str):
            items.append(item)
            continue
        if isinstance(item, dict):
            label = item.get("label")
            if not isinstance(label, str):
                raise ValueError("nearby item label must be a string")
            side = parse_side_schema_value(item.get("side"))
            items.append({"label": label, "side": side.value if side else None})
            continue
        raise ValueError("nearby items must be objects or strings")
    return normalize_nearby_hints(items)


def _parse_offset_reply(raw: str) -> tuple[str, int, int, list[NearbyHint]]:
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
    nearby_hints = _parse_nearby_items(nearby)
    return anchor.strip(), dx, dy, nearby_hints


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
) -> tuple[str, int, int, list[NearbyHint]]:
    """
    Split an instruction into anchor text, relative pixel offset, and nearby hints.

    Uses an LLM to extract anchor/dx/dy/nearby, with a regex fallback for
    anchor/dx/dy on failure. Nearby hints are also recovered from parenthetical
    comments on the regex path when possible.
    Positive dx is right; positive dy is down.
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
        nearby = extract_nearby_hints_from_instruction(text)
        _log_info(
            "parse_relative_pixel_offset regex fallback "
            f"({exc}) anchor={anchor!r} dx={dx} dy={dy} nearby={nearby!r}"
        )
        return anchor, dx, dy, nearby
