"""Unit tests for UIA input/scrollbar refine geometry and prefer-UIA policy."""

from __future__ import annotations

from src.recorder.uia_input_boxes import (
    filter_uia_input_rects,
    is_oversized_uia_rect,
    try_uia_refine_inputs_and_scrollbars,
    try_uia_replace_yolo_inputs,
    uia_covers_yolo_inputs,
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
    def fake_combined(_hwnd: int):
        return (
            [(110, 60, 210, 100)],
            [(480, 50, 500, 350)],
        )

    inputs, scrollbars, meta = try_uia_refine_inputs_and_scrollbars(
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
    assert meta["used"] == "uia"
    assert meta["scrollbar_used"] == "uia"
    assert meta["uia_inputs"] == 1
    assert meta["uia_scrollbars"] == 1


def test_try_uia_refine_empty_both_falls_back() -> None:
    inputs, scrollbars, meta = try_uia_refine_inputs_and_scrollbars(
        [(12, 12, 90, 30)],
        [(370, 10, 18, 280)],
        hwnd=1,
        monitor_offset=(100, 50),
        image_w=400,
        image_h=300,
        enumerate_fn=lambda _hwnd: ([], []),
        window_screen_xywh=(100, 50, 400, 300),
    )
    assert inputs is None
    assert scrollbars is None
    assert meta["used"] == "hough"
    assert meta["scrollbar_used"] == "yolo"


def test_try_uia_refine_scrollbar_only() -> None:
    def fake_combined(_hwnd: int):
        return ([], [(480, 50, 500, 350)])

    inputs, scrollbars, meta = try_uia_refine_inputs_and_scrollbars(
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
