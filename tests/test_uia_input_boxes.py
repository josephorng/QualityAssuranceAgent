"""Unit tests for UIA input/scrollbar/text refine geometry and prefer-UIA policy."""

from __future__ import annotations

from src.recorder.uia_input_boxes import (
    associate_uia_labels_with_yolo_text,
    filter_uia_input_rects,
    is_oversized_uia_rect,
    try_uia_refine_inputs_and_scrollbars,
    try_uia_replace_yolo_inputs,
    uia_covers_yolo_inputs,
    uia_label_prefer_value,
)


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


def test_uia_covers_yolo_inputs_empty_and_partial() -> None:
    uia = [(10, 10, 100, 30), (200, 10, 80, 25)]
    assert uia_covers_yolo_inputs([], [(10, 10, 100, 30)]) is False
    assert uia_covers_yolo_inputs(uia, []) is True
    assert uia_covers_yolo_inputs(uia, [(12, 12, 90, 26)]) is True
    assert uia_covers_yolo_inputs(uia, [(12, 12, 90, 26), (400, 10, 50, 20)]) is False


def test_try_uia_replace_yolo_inputs_nonempty_skips_hough_path() -> None:
    # Screen Edit at (110,60)-(210,100); monitor origin (100,50) → local (10,10,100,40)
    def fake_enum(_hwnd: int):
        return [(110, 60, 210, 100)]

    replaced, meta = try_uia_replace_yolo_inputs(
        [(12, 12, 90, 30)],
        hwnd=1,
        monitor_offset=(100, 50),
        image_w=400,
        image_h=300,
        enumerate_fn=fake_enum,
        window_screen_xywh=(100, 50, 400, 300),
    )
    assert replaced == [(10, 10, 100, 40)]
    assert meta["covered"] is True
    assert meta["yolo_covered"] is True
    assert meta["used"] == "uia"


def test_try_uia_replace_prefers_uia_even_when_yolo_not_fully_covered() -> None:
    def fake_enum(_hwnd: int):
        return [(110, 60, 210, 100)]

    replaced, meta = try_uia_replace_yolo_inputs(
        [(12, 12, 90, 30), (250, 10, 40, 20)],
        hwnd=1,
        monitor_offset=(100, 50),
        image_w=400,
        image_h=300,
        enumerate_fn=fake_enum,
        window_screen_xywh=(100, 50, 400, 300),
    )
    assert replaced == [(10, 10, 100, 40)]
    assert meta["covered"] is True
    assert meta["yolo_covered"] is False
    assert meta["used"] == "uia"


def test_try_uia_replace_empty_falls_back_to_hough() -> None:
    replaced, meta = try_uia_replace_yolo_inputs(
        [(12, 12, 90, 30)],
        hwnd=1,
        monitor_offset=(100, 50),
        image_w=400,
        image_h=300,
        enumerate_fn=lambda _hwnd: [],
        window_screen_xywh=(100, 50, 400, 300),
    )
    assert replaced is None
    assert meta["covered"] is False
    assert meta["used"] == "hough"


def test_try_uia_refine_inputs_and_scrollbars_independent_prefer() -> None:
    # Input screen (110,60)-(210,100) → local (10,10,100,40)
    # Scrollbar screen (480,50)-(500,350) → local (380,0,20,300)
    # Text screen (150,100)-(200,120) → local (50,50,50,20)
    def fake_combined(_hwnd: int):
        return (
            [(110, 60, 210, 100)],
            [(480, 50, 500, 350)],
            [((150, 100, 200, 120), "Item A")],
        )

    inputs, scrollbars, texts, meta = try_uia_refine_inputs_and_scrollbars(
        [(12, 12, 90, 30)],
        [(370, 10, 18, 280)],
        hwnd=1,
        monitor_offset=(100, 50),
        image_w=400,
        image_h=300,
        enumerate_fn=fake_combined,
        window_screen_xywh=(100, 50, 400, 300),
    )
    assert inputs == [(10, 10, 100, 40)]
    assert scrollbars == [(380, 0, 20, 300)]
    assert texts == [((50, 50, 50, 20), "Item A")]
    assert meta["used"] == "uia"
    assert meta["scrollbar_used"] == "uia"
    assert meta["uia_inputs"] == 1
    assert meta["uia_scrollbars"] == 1
    assert meta["uia_texts"] == 1


def test_try_uia_refine_empty_both_falls_back() -> None:
    inputs, scrollbars, texts, meta = try_uia_refine_inputs_and_scrollbars(
        [(12, 12, 90, 30)],
        [(370, 10, 18, 280)],
        hwnd=1,
        monitor_offset=(100, 50),
        image_w=400,
        image_h=300,
        enumerate_fn=lambda _hwnd: ([], [], []),
        window_screen_xywh=(100, 50, 400, 300),
    )
    assert inputs is None
    assert scrollbars is None
    assert texts == []
    assert meta["used"] == "hough"
    assert meta["scrollbar_used"] == "yolo"


def test_try_uia_refine_scrollbar_only() -> None:
    def fake_combined(_hwnd: int):
        return ([], [(480, 50, 500, 350)], [])

    inputs, scrollbars, texts, meta = try_uia_refine_inputs_and_scrollbars(
        [(12, 12, 90, 30)],
        [],
        hwnd=1,
        monitor_offset=(100, 50),
        image_w=400,
        image_h=300,
        enumerate_fn=fake_combined,
        window_screen_xywh=(100, 50, 400, 300),
    )
    assert inputs is None
    assert scrollbars == [(380, 0, 20, 300)]
    assert texts == []
    assert meta["used"] == "hough"
    assert meta["scrollbar_used"] == "uia"


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


def test_associate_split_blob_emits_uia_children() -> None:
    blob = (0, 0, 100, 90)
    uia = [
        ((10, 5, 80, 20), "One"),
        ((10, 35, 80, 20), "Two"),
        ((10, 65, 80, 20), "Three"),
    ]
    sourced, for_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [blob], uia
    )
    assert for_ocr == []
    assert element_ocr == []
    assert meta["split_blobs"] == 1
    assert {label for _box, label in sourced} == {"One", "Two", "Three"}
    assert len(sourced) == 3


def test_associate_iou_fill_keeps_yolo_bbox() -> None:
    yolo = (10, 10, 80, 20)
    uia = [((12, 12, 70, 16), "Label")]
    sourced, for_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [yolo], uia
    )
    assert for_ocr == []
    assert element_ocr == []
    assert meta["iou_filled"] == 1
    assert sourced == [(yolo, "Label")]


def test_associate_orphan_in_roi() -> None:
    sourced, for_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [],
        [((50, 50, 40, 15), "Orphan")],
        ocr_roi=(0, 0, 200, 200),
        ocr_roi_pad=0,
    )
    assert for_ocr == []
    assert element_ocr == []
    assert meta["orphans"] == 1
    assert sourced == [((50, 50, 40, 15), "Orphan")]


def test_associate_unmatched_yolo_goes_to_ocr() -> None:
    yolo = (10, 10, 80, 20)
    sourced, for_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [yolo], []
    )
    assert sourced == []
    assert for_ocr == [yolo]
    assert element_ocr == []
    assert meta["yolo_text_for_ocr"] == 1


def test_associate_element_iou_fill_skips_element_ocr() -> None:
    element = (10, 10, 80, 20)
    uia = [((12, 12, 70, 16), "Save")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [],
        uia,
        yolo_element_boxes=[element],
    )
    assert sourced == [(element, "Save")]
    assert text_ocr == []
    assert element_ocr == []
    assert meta["iou_filled_element"] == 1
    assert meta["element_claimed"] == 1


def test_associate_overlapped_text_element_uia_collapses_both() -> None:
    text = (10, 10, 80, 24)
    element = (12, 12, 76, 20)  # high IoU with text
    uia = [((14, 14, 70, 16), "Favorites")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [text],
        uia,
        yolo_element_boxes=[element],
    )
    assert text_ocr == []
    assert element_ocr == []
    # Cluster collapse and/or IoU+sibling / containment may claim the pair.
    assert meta["element_claimed"] == 1
    assert meta["text_ocr_skipped"] == 1
    assert (
        meta["cluster_collapsed"] >= 1
        or meta["sibling_filled"] >= 1
        or meta["containment_filled"] >= 1
    )
    assert len(sourced) == 1
    assert sourced[0][1] == "Favorites"


def test_associate_overlapped_text_element_without_uia_keeps_both_for_ocr() -> None:
    text = (10, 10, 80, 24)
    element = (12, 12, 76, 20)
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [text],
        [],
        yolo_element_boxes=[element],
    )
    assert sourced == []
    assert text_ocr == [text]
    assert element_ocr == [element]
    assert meta["cluster_collapsed"] == 0
    assert meta["element_claimed"] == 0


def test_associate_containment_small_yolo_inside_large_uia() -> None:
    # Small YOLO fully inside large UIA; IoU = 400/10000 = 0.04 < 0.3.
    yolo = (40, 40, 20, 20)
    uia = [((0, 0, 100, 100), "Start")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [yolo], uia
    )
    assert sourced == [(yolo, "Start")]
    assert text_ocr == []
    assert element_ocr == []
    assert meta["iou_filled"] == 0
    assert meta["containment_filled"] == 1
    assert meta["orphans"] == 0


def test_associate_containment_rejects_two_nonoverlapping_text_occupants() -> None:
    left = (10, 10, 20, 20)
    right = (70, 10, 20, 20)
    uia = [((0, 0, 100, 100), "Busy")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [left, right],
        uia,
        ocr_roi=(0, 0, 100, 100),
    )
    assert meta["containment_filled"] == 0
    # Orphan still emits the UIA box; YOLO boxes remain for OCR.
    assert meta["orphans"] == 1
    assert (uia[0][0], "Busy") in sourced
    assert left in text_ocr and right in text_ocr
    assert element_ocr == []


def test_associate_containment_side_by_side_text_element_pair() -> None:
    """List-row icon + label: one UIA may claim both even without mutual IoU."""
    element = (10, 10, 24, 24)
    text = (50, 12, 40, 20)
    uia = [((0, 0, 100, 40), "Notepad")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [text],
        uia,
        yolo_element_boxes=[element],
    )
    assert text_ocr == []
    assert element_ocr == []
    assert meta["containment_filled"] == 1
    assert meta["containment_filled_element"] == 1
    assert meta["element_claimed"] == 1
    assert meta["text_ocr_skipped"] == 1
    assert len(sourced) == 1
    assert sourced[0][1] == "Notepad"
    # Primary emit is the larger YOLO box (text).
    assert sourced[0][0] == text


def test_associate_containment_rejects_text_plus_two_elements() -> None:
    text = (40, 10, 40, 20)
    icon_a = (5, 10, 16, 16)
    icon_b = (24, 10, 16, 16)
    uia = [((0, 0, 100, 40), "Ambiguous")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [text],
        uia,
        yolo_element_boxes=[icon_a, icon_b],
        ocr_roi=(0, 0, 100, 40),
    )
    assert meta["containment_filled"] == 0
    assert meta["sibling_filled"] == 0
    assert text in text_ocr
    assert icon_a in element_ocr and icon_b in element_ocr


def test_associate_iou_text_completes_sibling_element() -> None:
    """IoU claims the text; one remaining icon inside the same UIA is claimed too."""
    text = (30, 10, 60, 20)
    element = (8, 12, 16, 16)
    # List-item UIA: high IoU with text only; icon still mostly inside.
    uia = [((5, 5, 90, 30), "Notepad")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [text],
        uia,
        yolo_element_boxes=[element],
    )
    assert text_ocr == []
    assert element_ocr == []
    assert meta["iou_filled"] == 1
    assert meta["sibling_filled"] == 1
    assert meta["element_claimed"] == 1
    assert len(sourced) == 1
    assert sourced[0] == (text, "Notepad")


def test_associate_containment_overlapped_text_element_pair() -> None:
    text = (40, 40, 24, 24)
    element = (42, 42, 20, 20)  # high IoU with text; both inside large UIA
    uia = [((0, 0, 100, 100), "Favorites")]
    sourced, text_ocr, element_ocr, meta = associate_uia_labels_with_yolo_text(
        [text],
        uia,
        yolo_element_boxes=[element],
    )
    assert text_ocr == []
    assert element_ocr == []
    assert meta["containment_filled"] == 1
    assert meta["containment_filled_element"] == 1
    assert len(sourced) == 1
    assert sourced[0][1] == "Favorites"
    # Primary emit is the larger YOLO box (text).
    assert sourced[0][0] == text
