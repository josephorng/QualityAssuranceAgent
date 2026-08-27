from __future__ import annotations

from cua_mcp.char_target import (
    format_char_target_anchor,
    occurrence_index,
    parse_char_target_instruction,
    resolve_char_screen_point,
    screen_bbox_from_span,
    span_at_local_x,
    text_anchor_from_full_text,
)
from cua_mcp.read_screen_text.constrained_decode import CharSpan


def _spans_for_google() -> list[CharSpan]:
    return [
        CharSpan(char="G", t_start=0, t_end=0, x_start=0.0, x_end=10.0),
        CharSpan(char="o", t_start=1, t_end=1, x_start=10.0, x_end=20.0),
        CharSpan(char="o", t_start=2, t_end=2, x_start=20.0, x_end=30.0),
        CharSpan(char="g", t_start=3, t_end=3, x_start=30.0, x_end=40.0),
        CharSpan(char="l", t_start=4, t_end=4, x_start=40.0, x_end=50.0),
        CharSpan(char="e", t_start=5, t_end=5, x_start=50.0, x_end=60.0),
    ]


def test_format_char_target_anchor_unique_char() -> None:
    assert format_char_target_anchor("搜尋", "搜") == "「搜尋」的「搜」字上"


def test_format_char_target_anchor_duplicate_char() -> None:
    assert format_char_target_anchor("Google", "o", occurrence=1) == "「Google」的第2個「o」字上"


def test_parse_char_target_instruction_unique() -> None:
    parsed = parse_char_target_instruction("將滑鼠移到「搜尋」的「搜」字上")
    assert parsed == ("搜尋", "搜", 0)


def test_parse_char_target_instruction_duplicate() -> None:
    parsed = parse_char_target_instruction("「Google」的第2個「o」字上")
    assert parsed == ("Google", "o", 1)


def test_text_anchor_from_full_text() -> None:
    assert text_anchor_from_full_text("搜尋") == "「搜尋」文字"


def test_span_at_local_x() -> None:
    spans = _spans_for_google()
    assert span_at_local_x(spans, 15.0) is spans[1]
    assert span_at_local_x(spans, 999.0) is None


def test_occurrence_index() -> None:
    spans = _spans_for_google()
    assert occurrence_index(spans, spans[1]) == 0
    assert occurrence_index(spans, spans[2]) == 1


def test_resolve_char_screen_point() -> None:
    bbox = (100, 200, 60, 20)
    spans = _spans_for_google()
    point = resolve_char_screen_point(
        bbox,
        spans,
        "o",
        occurrence=1,
        img_w=1000,
        img_h=1000,
    )
    assert point is not None
    x, y = point
    assert 98 <= x <= 100 + 60 + 4
    assert y == 198 + (20 + 4) // 2


def test_resolve_char_screen_point_missing_char() -> None:
    assert (
        resolve_char_screen_point(
            (0, 0, 10, 10),
            _spans_for_google(),
            "z",
            img_w=100,
            img_h=100,
        )
        is None
    )


def test_screen_bbox_from_span_left_and_right_halves() -> None:
    # margin=0 so expanded crop == bbox; line_w scales with crop (60x20 → line_w=96).
    bbox = (100, 200, 60, 20)
    left = CharSpan(char="A", t_start=0, t_end=0, x_start=0.0, x_end=48.0)
    right = CharSpan(char="B", t_start=1, t_end=1, x_start=48.0, x_end=96.0)
    left_box = screen_bbox_from_span(bbox, left, img_w=1000, img_h=1000, margin=0)
    right_box = screen_bbox_from_span(bbox, right, img_w=1000, img_h=1000, margin=0)
    assert left_box == (100, 200, 30, 20)
    assert right_box == (130, 200, 30, 20)
    assert left_box[0] + left_box[2] == right_box[0]


def test_screen_bbox_from_span_clips_to_image() -> None:
    bbox = (0, 0, 10, 10)
    # Span extends past crop in line coords; result must stay inside image.
    span = CharSpan(char="X", t_start=0, t_end=0, x_start=0.0, x_end=1000.0)
    box = screen_bbox_from_span(bbox, span, img_w=20, img_h=20, margin=2)
    x, y, w, h = box
    assert x >= 0 and y >= 0
    assert x + w <= 20 and y + h <= 20
    assert w >= 1 and h >= 1
