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


def test_labeled_text_control_types_include_interactive_items() -> None:
    from src.recorder import uia_input_boxes as uia

    assert uia._UIA_BUTTON_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_MENUITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_TREEITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_DATAITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_LISTITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_TEXT_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_HYPERLINK_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
