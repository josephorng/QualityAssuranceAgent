"""Unit tests for UIA input/scrollbar/text refine geometry and prefer-UIA policy."""

from __future__ import annotations

from src.recorder.uia_input_boxes import (
    filter_uia_input_rects,
    is_oversized_uia_rect,
    merge_nested_element_boxes,
    uia_label_prefer_value,
)


def test_merge_nested_element_boxes_keeps_larger() -> None:
    outer = (612, 874, 21, 18)
    inner = (615, 876, 14, 14)
    assert merge_nested_element_boxes([inner, outer]) == [outer]


def test_merge_nested_element_boxes_keeps_side_by_side() -> None:
    left = (10, 10, 16, 16)
    right = (40, 10, 16, 16)
    assert sorted(merge_nested_element_boxes([left, right])) == sorted([left, right])


def test_is_oversized_uia_rect_near_full_window() -> None:
    window = (0, 0, 1000, 800)
    assert is_oversized_uia_rect((10, 10, 950, 750), window) is True
    assert is_oversized_uia_rect((100, 100, 200, 40), window) is False
    assert is_oversized_uia_rect((0, 0, 0, 10), window) is True


def test_filter_uia_input_rects_drops_oversized_and_out_of_roi() -> None:
    window = (100, 50, 400, 300)  # screen xywh
    screen_boxes = [
        (120, 80, 300, 120),  # good field
        (100, 50, 500, 350),  # near-window oversized
        (600, 80, 700, 120),  # outside ocr_roi after convert
    ]
    boxes = filter_uia_input_rects(
        screen_boxes,
        window_screen_xywh=window,
        monitor_offset=(100, 50),
        image_w=500,
        image_h=400,
        ocr_roi=(0, 0, 250, 200),
        ocr_roi_pad=0,
    )
    assert boxes == [(20, 30, 180, 40)]


def test_filter_uia_input_rects_drops_edge_sliver_from_wrong_monitor() -> None:
    """Off-monitor boxes that clip to a 1px primary edge must not become targets."""
    window = (1912, -9, 1936, 1048)  # maximized on secondary, left overlaps primary
    # Real control on secondary; clipping onto primary (0..1920) leaves a 1px strip.
    screen_boxes = [
        (2100, 300, 2250, 330),
        (100, 200, 220, 240),  # fully on primary — keep
    ]
    boxes = filter_uia_input_rects(
        screen_boxes,
        window_screen_xywh=window,
        monitor_offset=(0, 0),
        image_w=1920,
        image_h=1080,
    )
    assert boxes == [(100, 200, 120, 40)]


def test_is_clipped_uia_sliver_keeps_uncropped_small_box() -> None:
    from src.recorder.uia_input_boxes import _is_clipped_uia_sliver

    assert (
        _is_clipped_uia_sliver(orig_w=2, orig_h=16, clipped_w=2, clipped_h=16) is False
    )
    assert (
        _is_clipped_uia_sliver(orig_w=100, orig_h=20, clipped_w=1, clipped_h=20) is True
    )
    assert (
        _is_clipped_uia_sliver(orig_w=100, orig_h=20, clipped_w=40, clipped_h=20) is True
    )


def test_hough_path_still_drops_inputs_overlapping_h_scrollbar() -> None:
    """Hough/YOLO merge still drops boxes that overlap horizontal scrollbars."""
    from cua_mcp.input_box_rectangles import (
        _drop_xywh_boxes_overlapping_horizontal_scrollbars,
    )

    inputs = [(10, 70, 180, 25), (10, 10, 80, 30)]
    h_scrollbars = [(0, 80, 200, 15)]
    kept = _drop_xywh_boxes_overlapping_horizontal_scrollbars(inputs, h_scrollbars)
    assert kept == [(10, 10, 80, 30)]


def test_uia_label_prefer_value_over_name() -> None:
    assert uia_label_prefer_value("  val  ", "name") == "val"
    assert uia_label_prefer_value("", "  name  ") == "name"
    assert uia_label_prefer_value("   ", "name") == "name"
    assert uia_label_prefer_value("", "") == ""


def test_scrollbar_thumb_unions_same_line_arrow_buttons() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    thumb = (900, 400, 916, 520)
    up = (900, 160, 916, 176)
    down = (898, 980, 918, 996)
    other_column = (200, 160, 216, 176)
    rects, used = compose_scrollbar_screen_rects([], [thumb], [up, down, other_column])
    assert rects == [(898, 160, 918, 996)]
    assert used == [0, 1]


def test_scrollbar_thumb_alone_is_a_scrollbar() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    thumb = (900, 400, 916, 520)
    rects, used = compose_scrollbar_screen_rects([], [thumb], [])
    assert rects == [thumb]
    assert used == []


def test_scrollbar_thumb_ignores_button_that_only_clips_the_edge() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    thumb = (900, 400, 916, 520)
    clipped = (910, 160, 940, 176)
    rects, used = compose_scrollbar_screen_rects([], [thumb], [clipped])
    assert rects == [thumb]
    assert used == []


def test_scrollbar_thumbs_keep_separate_columns() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    left_thumb = (100, 400, 116, 520)
    right_thumb = (400, 400, 416, 520)
    left_up = (100, 160, 116, 176)
    right_down = (400, 800, 416, 816)
    rects, used = compose_scrollbar_screen_rects(
        [],
        [left_thumb, right_thumb],
        [left_up, right_down],
    )
    assert rects == [(100, 160, 116, 520), (400, 400, 416, 816)]
    assert used == [0, 1]


def test_scrollbar_thumb_inside_control_is_not_added_again() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    control = (890, 150, 920, 1000)
    thumb = (900, 400, 916, 520)
    up = (900, 160, 916, 176)
    down = (900, 980, 916, 996)
    rects, used = compose_scrollbar_screen_rects([control], [thumb], [up, down])
    assert rects == [control]
    assert used == [0, 1]


def test_scrollbar_thumb_inside_huge_control_is_kept() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    window_sized = (0, 0, 800, 900)
    thumb = (780, 400, 796, 520)
    rects, used = compose_scrollbar_screen_rects([window_sized], [thumb], [])
    assert rects == [window_sized, thumb]
    assert used == []


def test_labeled_text_control_types_include_interactive_items() -> None:
    from src.recorder import uia_input_boxes as uia

    assert uia._UIA_BUTTON_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_MENUITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_TREEITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_DATAITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_LISTITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_TEXT_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_HYPERLINK_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
