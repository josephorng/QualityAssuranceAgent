"""Parse relative pixel offsets from Traditional Chinese mouse instructions."""

from __future__ import annotations

import json
import re
from typing import Any

from cua_mcp.char_target import parse_char_target_instruction, text_anchor_from_full_text
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
_TRACK_PERCENT_RE = re.compile(r"的(\d+)%處")
_TRAILING_CONTEXT_COMMENT_RE = re.compile(r"(?:（[^）]*）)+$")
_INLINE_START_CONTEXT_COMMENT_RE = re.compile(r"（起點(?:附近|在)[^）]*）")

_OFFSET_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anchor": {"type": "string"},
        "dx": {"type": "integer"},
        "dy": {"type": "integer"},
        "track_percent": {"type": ["integer", "null"]},
        "char": {"type": ["string", "null"]},
        "char_occurrence": {"type": "integer"},
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


def _parse_char_occurrence(raw: Any) -> int:
    if raw is None:
        return 0
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError("char_occurrence must be an integer")
    return max(0, raw)


def _parse_track_percent(raw: Any) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError("track_percent must be an integer or null")
    return max(0, min(100, raw))


def _extract_track_percent_regex(text: str) -> tuple[str, int | None]:
    """Pull ``的N%處`` from text; prefer the percent after ``拖到`` when present."""
    if "拖到" in text:
        drag_at = text.index("拖到")
        dest_part = text[drag_at:]
        dest_matches = list(_TRACK_PERCENT_RE.finditer(dest_part))
        if dest_matches:
            percent = int(dest_matches[-1].group(1))
            cleaned = _TRACK_PERCENT_RE.sub("", text)
            return cleaned, max(0, min(100, percent))
    matches = list(_TRACK_PERCENT_RE.finditer(text))
    if not matches:
        return text, None
    percent = int(matches[-1].group(1))
    cleaned = _TRACK_PERCENT_RE.sub("", text)
    return cleaned, max(0, min(100, percent))


def _parse_offset_reply(
    raw: str,
) -> tuple[str, int, int, list[NearbyHint], str | None, int, int | None]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    anchor = data.get("anchor")
    dx = data.get("dx")
    dy = data.get("dy")
    nearby = data.get("nearby")
    char = data.get("char")
    char_occurrence = _parse_char_occurrence(data.get("char_occurrence"))
    track_percent = _parse_track_percent(data.get("track_percent"))
    if not isinstance(anchor, str):
        raise ValueError("anchor must be a string")
    if not isinstance(dx, int) or isinstance(dx, bool):
        raise ValueError("dx must be an integer")
    if not isinstance(dy, int) or isinstance(dy, bool):
        raise ValueError("dy must be an integer")
    if char is not None and (not isinstance(char, str) or not char):
        raise ValueError("char must be a non-empty string or null")
    nearby_hints = _parse_nearby_items(nearby)
    if track_percent is None:
        _, regex_percent = _extract_track_percent_regex(anchor)
        if regex_percent is not None:
            track_percent = regex_percent
            anchor = _TRACK_PERCENT_RE.sub("", anchor).strip()
    if track_percent is not None:
        dx, dy = 0, 0
    return anchor.strip(), dx, dy, nearby_hints, char, char_occurrence, track_percent


def _strip_trailing_context_comment(text: str) -> str:
    """Remove trailing full-width parenthetical nearby-context comments."""
    return _TRAILING_CONTEXT_COMMENT_RE.sub("", text).strip()


def _strip_context_comments(text: str) -> str:
    """Remove inline start and trailing nearby-context comments."""
    text = _strip_trailing_context_comment(text)
    return _INLINE_START_CONTEXT_COMMENT_RE.sub("", text).strip()


def _normalize_char_target_fields(
    anchor: str,
    dx: int,
    dy: int,
    char: str | None,
    char_occurrence: int,
) -> tuple[str, int, int, str | None, int]:
    """Normalize char-target instructions to YOLO anchor + char metadata."""
    parsed = parse_char_target_instruction(anchor)
    if parsed is not None:
        full_text, parsed_char, parsed_occurrence = parsed
        resolved_char = char or parsed_char
        resolved_occurrence = char_occurrence if char is not None else parsed_occurrence
        return (
            text_anchor_from_full_text(full_text),
            0,
            0,
            resolved_char,
            resolved_occurrence,
        )

    if char:
        return anchor, 0, 0, char, char_occurrence
    return anchor, dx, dy, None, 0


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


def _parse_mouse_target_regex(
    instruction: str,
) -> tuple[str, int, int, str | None, int, int | None]:
    """Regex-only parse including char-target phrases and scrollbar track %."""
    text = _strip_context_comments((instruction or "").strip())
    if not text:
        return "", 0, 0, None, 0, None

    parsed = parse_char_target_instruction(text)
    if parsed is not None:
        full_text, char, occurrence = parsed
        return text_anchor_from_full_text(full_text), 0, 0, char, occurrence, None

    text_wo_percent, track_percent = _extract_track_percent_regex(text)
    anchor, dx, dy = _parse_relative_pixel_offset_regex(text_wo_percent)
    if track_percent is not None:
        dx, dy = 0, 0
    anchor, dx, dy, char, occurrence = _normalize_char_target_fields(
        anchor, dx, dy, None, 0
    )
    return anchor, dx, dy, char, occurrence, track_percent


async def parse_mouse_target_instruction(
    instruction: str,
) -> tuple[str, int, int, list[NearbyHint], str | None, int, int | None]:
    """
    Split an instruction into anchor text, relative pixel offset, nearby hints,
    optional char target metadata, and optional scrollbar track percent.

    Uses an LLM to extract anchor/dx/dy/nearby/char/track_percent, with a regex
    fallback on failure. Positive dx is right; positive dy is down.
    ``track_percent`` is 0–100 along a scrollbar main axis when the instruction
    contains ``的N%處``.
    """
    text = (instruction or "").strip()
    if not text:
        return "", 0, 0, [], None, 0, None

    prompt = get_prompt("mouse_target_instruction").format(instruction=text)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        (
            anchor,
            dx,
            dy,
            nearby,
            char,
            char_occurrence,
            track_percent,
        ) = await request_json_with_retry(
            messages=messages,
            response_schema=_OFFSET_RESPONSE_SCHEMA,
            parse_reply=_parse_offset_reply,
            retry_instruction=get_prompt("mouse_target_instruction_retry"),
            log_info=lambda message: _log_info(f"parse_mouse_target_instruction: {message}"),
            append_image_sizes=False,
        )
        if not anchor:
            raise ValueError("anchor is empty")
        anchor, dx, dy, char, char_occurrence = _normalize_char_target_fields(
            anchor, dx, dy, char, char_occurrence
        )
        if track_percent is None:
            _, regex_percent = _extract_track_percent_regex(text)
            track_percent = regex_percent
        if track_percent is not None:
            dx, dy = 0, 0
            anchor = _TRACK_PERCENT_RE.sub("", anchor).strip()
        _log_info(
            "parse_mouse_target_instruction LLM "
            f"anchor={anchor!r} dx={dx} dy={dy} char={char!r} "
            f"char_occurrence={char_occurrence} track_percent={track_percent!r} "
            f"nearby={nearby!r}"
        )
        return anchor, dx, dy, nearby, char, char_occurrence, track_percent
    except (ValueError, json.JSONDecodeError) as exc:
        anchor, dx, dy, char, char_occurrence, track_percent = _parse_mouse_target_regex(
            text
        )
        nearby = extract_nearby_hints_from_instruction(text)
        _log_info(
            "parse_mouse_target_instruction regex fallback "
            f"({exc}) anchor={anchor!r} dx={dx} dy={dy} char={char!r} "
            f"char_occurrence={char_occurrence} track_percent={track_percent!r} "
            f"nearby={nearby!r}"
        )
        return anchor, dx, dy, nearby, char, char_occurrence, track_percent


def extract_track_percent(instruction: str) -> int | None:
    """Return 0–100 when ``instruction`` contains ``的N%處`` (e.g. 滾動條的60%處)."""
    _, track_percent = _extract_track_percent_regex((instruction or "").strip())
    return track_percent
