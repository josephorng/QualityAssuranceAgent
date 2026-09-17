"""Tests for line-aware text box merging (cua_mcp.geometry.merge_same_line_boxes)."""

from __future__ import annotations

from cua_mcp.geometry import (
    merge_overlapping_boxes,
    merge_same_line_boxes,
    vertical_overlap_frac,
)


def test_vertical_overlap_frac_uses_shorter_box():
    tall = (0, 0, 50, 20)
    short = (0, 5, 50, 5)
    assert vertical_overlap_frac(tall, short) == 1.0
    assert vertical_overlap_frac(short, tall) == 1.0


def test_vertical_overlap_frac_for_stacked_lines():
    # 12px lines overlapping by 1px.
    assert vertical_overlap_frac((0, 587, 50, 12), (0, 598, 50, 13)) < 0.1


def test_keeps_stacked_dropdown_lines_separate():
    """
    Regression: a 12px-pitch dropdown whose line boxes overlap by 1-3px.

    Plain overlap merging chains these into one 3-line box (observed centered at
    757,604 in the 資產設備 runs); line-aware merging must keep them apart.
    """
    lines = [
        (734, 587, 49, 12),
        (734, 598, 49, 13),
        (734, 610, 49, 12),
    ]
    assert len(merge_overlapping_boxes(lines)) == 1  # the old behaviour
    assert sorted(merge_same_line_boxes(lines)) == sorted(lines)


def test_merges_fragments_of_one_line():
    left = (100, 50, 40, 13)
    right = (135, 50, 40, 13)  # overlaps in x, same y range
    merged = merge_same_line_boxes([left, right])
    assert merged == [(100, 50, 75, 13)]


def test_merges_short_fragment_inside_a_line():
    line = (100, 50, 40, 13)
    punctuation = (138, 55, 6, 4)
    merged = merge_same_line_boxes([line, punctuation])
    assert len(merged) == 1
    assert merged[0] == (100, 50, 44, 13)


def test_merges_transitively_within_a_line():
    boxes = [(0, 10, 20, 12), (18, 10, 20, 12), (36, 10, 20, 12)]
    assert merge_same_line_boxes(boxes) == [(0, 10, 56, 12)]


def test_non_overlapping_boxes_are_untouched():
    boxes = [(0, 0, 10, 10), (100, 100, 10, 10)]
    assert sorted(merge_same_line_boxes(boxes)) == sorted(boxes)


def test_single_box_passthrough():
    assert merge_same_line_boxes([(1, 2, 3, 4)]) == [(1, 2, 3, 4)]
    assert merge_same_line_boxes([]) == []
