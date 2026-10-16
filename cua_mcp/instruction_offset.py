"""Parse relative pixel offsets from Traditional Chinese mouse instructions."""

from __future__ import annotations

import re

from cua_mcp.char_target import parse_char_target_instruction, text_anchor_from_full_text
from src.common.nearby_side import (
    NearbyHint,
    extract_nearby_hints_from_instruction,
)
from src.common.run_state import get_run_state_manager

_OFFSET_TOKEN_RE = re.compile(r"(右方|左方|上方|下方)(\d+)個像素")
# ``的N%處`` (start) and bare ``N%處`` (drag destination shorthand).
_TRACK_PERCENT_RE = re.compile(r"(?:的)?(\d+)%處")
_TRAILING_CONTEXT_COMMENT_RE = re.compile(r"(?:（[^）]*）)+$")
_INLINE_START_CONTEXT_COMMENT_RE = re.compile(r"（起點(?:附近|在)[^）]*）")


def _log_info(text: str) -> None:
    try:
        get_run_state_manager().log_info(text)
    except RuntimeError:
        pass


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


def _strip_trailing_context_comment(text: str) -> str:
    """Remove trailing full-width parenthetical nearby-context comments."""
    return _TRAILING_CONTEXT_COMMENT_RE.sub("", text).strip()


def _strip_context_comments(text: str) -> str:
    """Remove inline start and trailing nearby-context comments."""
    text = _strip_trailing_context_comment(text)
    return _INLINE_START_CONTEXT_COMMENT_RE.sub("", text).strip()


def _destination_after_drag(text: str) -> str:
    """Keep only the destination after the last ``拖到`` (LLM-parity for drag sentences)."""
    if "拖到" not in text:
        return text
    return text.rsplit("拖到", 1)[-1].strip()


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
    """Extract anchor and relative pixel offset from Traditional Chinese phrases."""
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

    text = _destination_after_drag(text)
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


def parse_mouse_target_instruction(
    instruction: str,
) -> tuple[str, int, int, list[NearbyHint], str | None, int, int | None]:
    """
    Split an instruction into anchor text, relative pixel offset, nearby hints,
    optional char target metadata, and optional scrollbar track percent.

    Uses deterministic regex / string extractors only. Positive dx is right;
    positive dy is down. ``track_percent`` is 0–100 along a scrollbar main axis
    when the instruction contains ``的N%處``. Drag sentences keep only the
    destination after the last ``拖到``.
    """
    text = (instruction or "").strip()
    if not text:
        return "", 0, 0, [], None, 0, None

    nearby = extract_nearby_hints_from_instruction(text)
    anchor, dx, dy, char, char_occurrence, track_percent = _parse_mouse_target_regex(
        text
    )
    _log_info(
        "parse_mouse_target_instruction regex "
        f"anchor={anchor!r} dx={dx} dy={dy} char={char!r} "
        f"char_occurrence={char_occurrence} track_percent={track_percent!r} "
        f"nearby={nearby!r}"
    )
    return anchor, dx, dy, nearby, char, char_occurrence, track_percent


def extract_track_percent(instruction: str) -> int | None:
    """Return 0–100 when ``instruction`` contains ``的N%處`` or bare ``N%處``."""
    _, track_percent = _extract_track_percent_regex((instruction or "").strip())
    return track_percent
