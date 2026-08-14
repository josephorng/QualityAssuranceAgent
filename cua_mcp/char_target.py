"""Character-level mouse targeting within OCR text detections."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cua_mcp.geometry import clip_box
from cua_mcp.read_screen_text.constrained_decode import CharSpan, DecodeMode
from cua_mcp.read_screen_text.ocr_image import ocr_box_with_spans

if TYPE_CHECKING:
    import numpy as np

OCR_BOX_MARGIN = 2
_DEFAULT_LINE_HEIGHT = 32

_CHAR_TARGET_RE = re.compile(
    r"「([^」]+)」的(?:第(\d+)個)?「([^」]+)」字上"
)


def _line_image_width(crop_w: int, crop_h: int, *, line_height: int = _DEFAULT_LINE_HEIGHT) -> int:
    effective_h = _DEFAULT_LINE_HEIGHT if line_height < 2 else line_height
    return max(1, int((crop_w / max(1, crop_h)) * effective_h))


def expand_ocr_box(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    *,
    margin: int = OCR_BOX_MARGIN,
) -> tuple[int, int, int, int]:
    """Expand a box by ``margin`` pixels on all sides, clamped to image bounds."""
    x, y, w, h = bbox
    return clip_box(x - margin, y - margin, w + 2 * margin, h + 2 * margin, img_w, img_h)


def format_char_target_anchor(text: str, char: str, *, occurrence: int = 0) -> str:
    """Return a hub-style char target phrase like ``「搜尋」的「搜」字上``."""
    if occurrence > 0:
        return f"「{text}」的第{occurrence + 1}個「{char}」字上"
    return f"「{text}」的「{char}」字上"


def text_anchor_from_full_text(text: str) -> str:
    """Return YOLO anchor phrase ``「{text}」文字`` for similarity matching."""
    return f"「{text}」文字"


def parse_char_target_instruction(text: str) -> tuple[str, str, int] | None:
    """Parse char-target phrase; return ``(full_text, char, occurrence_0based)`` or ``None``."""
    cleaned = (text or "").strip()
    match = _CHAR_TARGET_RE.search(cleaned)
    if not match:
        return None
    full_text = match.group(1).strip()
    raw_occurrence = match.group(2)
    char = match.group(3)
    if not full_text or not char:
        return None
    occurrence = int(raw_occurrence) - 1 if raw_occurrence else 0
    if occurrence < 0:
        occurrence = 0
    return full_text, char, occurrence


def span_at_local_x(spans: list[CharSpan], rel_x: float) -> CharSpan | None:
    """Find the span whose line-image x-range contains ``rel_x``."""
    for span in spans:
        if span.x_start <= rel_x < span.x_end:
            return span
    return None


def occurrence_index(spans: list[CharSpan], target_span: CharSpan) -> int:
    """Return 0-based occurrence index of ``target_span.char`` at ``target_span``."""
    count = 0
    for span in spans:
        if span is target_span:
            return count
        if span.char == target_span.char:
            count += 1
    return 0


def _screen_x_from_span(
    expanded_x: int,
    crop_w: int,
    crop_h: int,
    span: CharSpan,
    *,
    line_height: int = _DEFAULT_LINE_HEIGHT,
) -> float:
    line_w = _line_image_width(crop_w, crop_h, line_height=line_height)
    span_center_line = (span.x_start + span.x_end) / 2.0
    return expanded_x + span_center_line / line_w * crop_w


def _line_x_from_screen_x(
    expanded_x: int,
    crop_w: int,
    crop_h: int,
    screen_x: int,
    *,
    line_height: int = _DEFAULT_LINE_HEIGHT,
) -> float:
    rel_raw = screen_x - expanded_x
    line_w = _line_image_width(crop_w, crop_h, line_height=line_height)
    return rel_raw / max(1, crop_w) * line_w


def resolve_char_screen_point(
    bbox: tuple[int, int, int, int],
    spans: list[CharSpan],
    char: str,
    *,
    occurrence: int = 0,
    margin: int = OCR_BOX_MARGIN,
    img_w: int,
    img_h: int,
    line_height: int = _DEFAULT_LINE_HEIGHT,
) -> tuple[int, int] | None:
    """Map a char span to screenshot-local ``(x, y)``, or ``None`` when not found."""
    matches = [span for span in spans if span.char == char]
    if not matches or occurrence >= len(matches):
        return None

    span = matches[occurrence]
    expanded_x, expanded_y, crop_w, crop_h = expand_ocr_box(
        bbox, img_w, img_h, margin=margin
    )
    char_x = _screen_x_from_span(
        expanded_x, crop_w, crop_h, span, line_height=line_height
    )
    char_y = expanded_y + crop_h // 2
    return int(round(char_x)), int(round(char_y))


def detect_clicked_char(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    click_x: int,
    text: str,
    *,
    mode: DecodeMode = "text",
    margin: int = OCR_BOX_MARGIN,
    line_height: int = _DEFAULT_LINE_HEIGHT,
) -> tuple[str, int] | None:
    """Run OCR with spans and return ``(char, occurrence_0based)`` at ``click_x``, or ``None``."""
    visible = (text or "").strip()
    if len(visible) <= 1:
        return None

    decoded, spans = ocr_box_with_spans(bgr, bbox, mode=mode, margin=margin, line_height=line_height)
    if not decoded or not spans:
        return None

    img_h, img_w = bgr.shape[:2]
    expanded_x, expanded_y, crop_w, crop_h = expand_ocr_box(
        bbox, img_w, img_h, margin=margin
    )
    rel_x = _line_x_from_screen_x(
        expanded_x, crop_w, crop_h, click_x, line_height=line_height
    )
    span = span_at_local_x(spans, rel_x)
    if span is None:
        return None
    return span.char, occurrence_index(spans, span)
