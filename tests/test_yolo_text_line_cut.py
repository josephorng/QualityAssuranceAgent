"""Tests for Stage 4 text line cut (text_line_cut + yolo_onnx hook)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cua_mcp import yolo_onnx as yolo_mod
from cua_mcp.text_line_cut import (
    cut_multiline_text_boxes,
    cut_text_box_into_lines,
)
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT, run_yolo_onnx_end2end

# Geometry shared by the crop-level tests: 60 rows tall, two 14-row bands of "glyphs"
# separated by a 12-row valley, with 10 rows of padding above and below.
CROP_H = 60
CROP_W = 120
BAND_1 = (10, 24)
BAND_2 = (36, 50)
INK_X0 = 15
INK_X1 = 100
# Median *box* height for the frame; YOLO text boxes are taller than their glyph ink.
MEDIAN_LINE_H = 25.0


def _paint_glyph_rows(
    img: np.ndarray,
    y0: int,
    y1: int,
    *,
    fg: int,
    x0: int = INK_X0,
    x1: int = INK_X1,
    off_x: int = 0,
    off_y: int = 0,
) -> None:
    """Dashed columns stand in for glyph strokes (sparse, like real text)."""
    img[off_y + y0 : off_y + y1, off_x + x0 : off_x + x1 : 3] = fg


def _two_line_crop(*, bg: int, fg: int) -> np.ndarray:
    img = np.full((CROP_H, CROP_W, 3), bg, dtype=np.uint8)
    _paint_glyph_rows(img, *BAND_1, fg=fg)
    _paint_glyph_rows(img, *BAND_2, fg=fg)
    return img


def test_cut_splits_white_text_on_black_background():
    lines = cut_text_box_into_lines(
        _two_line_crop(bg=0, fg=255),
        median_line_h=MEDIAN_LINE_H,
    )
    assert lines is not None
    assert len(lines) == 2
    assert [(y0, y1) for _x0, y0, _x1, y1 in lines] == [BAND_1, BAND_2]


def test_cut_splits_black_text_on_white_background_identically():
    """Polarity must not matter: the inverted image yields the same bands."""
    white_on_black = cut_text_box_into_lines(
        _two_line_crop(bg=0, fg=255),
        median_line_h=MEDIAN_LINE_H,
    )
    black_on_white = cut_text_box_into_lines(
        _two_line_crop(bg=255, fg=0),
        median_line_h=MEDIAN_LINE_H,
    )
    assert white_on_black is not None
    assert black_on_white == white_on_black


def test_cut_handles_per_row_background_changes():
    """Alternating row striping / a selected row must not defeat the profile."""
    img = np.full((CROP_H, CROP_W, 3), 30, dtype=np.uint8)
    # Second half of the box sits on a light background (e.g. a highlighted row).
    img[30:, :] = 230
    _paint_glyph_rows(img, *BAND_1, fg=220)
    _paint_glyph_rows(img, *BAND_2, fg=20)

    lines = cut_text_box_into_lines(img, median_line_h=MEDIAN_LINE_H)
    assert lines is not None
    assert [(y0, y1) for _x0, y0, _x1, y1 in lines] == [BAND_1, BAND_2]


def test_cut_tightens_band_x_extent():
    lines = cut_text_box_into_lines(
        _two_line_crop(bg=0, fg=255),
        median_line_h=MEDIAN_LINE_H,
    )
    assert lines is not None
    for x0, _y0, x1, _y1 in lines:
        assert x0 == INK_X0
        assert INK_X0 < x1 <= INK_X1
        assert x1 < CROP_W


def test_cut_bands_are_disjoint_in_y():
    """Padded / overlapping bands would be re-merged by merge_overlapping_boxes."""
    lines = cut_text_box_into_lines(
        _two_line_crop(bg=0, fg=255),
        median_line_h=MEDIAN_LINE_H,
    )
    assert lines is not None
    for (_ax0, _ay0, _ax1, a_y1), (_bx0, b_y0, _bx1, _by1) in zip(
        lines, lines[1:], strict=False
    ):
        assert a_y1 <= b_y0


def test_cut_splits_tightly_leaded_lines():
    """A 3px valley is the dense-dropdown case Stage 3 cannot fix; it must still split."""
    img = np.full((CROP_H, CROP_W, 3), 0, dtype=np.uint8)
    _paint_glyph_rows(img, 10, 24, fg=255)
    _paint_glyph_rows(img, 27, 41, fg=255)

    lines = cut_text_box_into_lines(img, median_line_h=MEDIAN_LINE_H)
    assert lines is not None
    assert [(y0, y1) for _x0, y0, _x1, y1 in lines] == [(10, 24), (27, 41)]


def _rendered_two_line_crop(*, bg: int, fg: int, pitch: int = 26) -> np.ndarray:
    """Anti-aliased glyphs, closer to a real screenshot than solid dashes."""
    img = np.full((70, 260, 3), bg, dtype=np.uint8)
    for i, label in enumerate(("Asset List", "Device Name")):
        cv2.putText(
            img,
            label,
            (8, 26 + i * pitch),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (fg, fg, fg),
            2,
            cv2.LINE_AA,
        )
    return img


@pytest.mark.parametrize(
    ("bg", "fg"),
    [(0, 255), (255, 0), (90, 245)],
    ids=["white-on-black", "black-on-white", "selected-row"],
)
def test_cut_splits_rendered_antialiased_text(bg: int, fg: int):
    lines = cut_text_box_into_lines(
        _rendered_two_line_crop(bg=bg, fg=fg),
        median_line_h=MEDIAN_LINE_H,
    )
    assert lines is not None
    assert len(lines) == 2
    assert lines[0][3] <= lines[1][1]
    # Each band is trimmed to its own text width, so the shorter label is narrower.
    assert (lines[0][2] - lines[0][0]) < (lines[1][2] - lines[1][0])


def test_cut_rejects_single_line_with_padding():
    img = np.full((CROP_H, CROP_W, 3), 0, dtype=np.uint8)
    _paint_glyph_rows(img, 22, 36, fg=255)
    assert cut_text_box_into_lines(img, median_line_h=MEDIAN_LINE_H) is None


def test_cut_rejects_touching_lines_without_a_valley():
    img = np.full((CROP_H, CROP_W, 3), 0, dtype=np.uint8)
    _paint_glyph_rows(img, 10, 24, fg=255)
    _paint_glyph_rows(img, 24, 38, fg=255)  # no blank row at all between the lines
    assert cut_text_box_into_lines(img, median_line_h=MEDIAN_LINE_H) is None


def test_cut_splits_on_a_single_blank_row():
    """A 1px valley is unambiguous when the row is empty, so it must split."""
    img = np.full((CROP_H, CROP_W, 3), 0, dtype=np.uint8)
    _paint_glyph_rows(img, 10, 24, fg=255)
    _paint_glyph_rows(img, 25, 39, fg=255)

    lines = cut_text_box_into_lines(img, median_line_h=MEDIAN_LINE_H)
    assert lines is not None
    assert [(y0, y1) for _x0, y0, _x1, y1 in lines] == [(10, 24), (25, 39)]


def _dense_list_crop(
    *,
    items: int = 20,
    pitch: int = 12,
    band_h: int = 11,
    width: int = 47,
    bg: int = 255,
    fg: int = 0,
) -> np.ndarray:
    """A compact dropdown: many short lines, each separated by exactly one blank row."""
    img = np.full((items * pitch, width, 3), bg, dtype=np.uint8)
    for i in range(items):
        top = i * pitch
        img[top : top + band_h, 2 : width - 4 : 3] = fg
    return img


@pytest.mark.parametrize("frame_median_h", [12.0, 16.0, 20.0, 25.0])
def test_cut_splits_dense_dropdown_regardless_of_frame_median(frame_median_h: float):
    """
    Regression: a 20-item, 12px-pitch dropdown (observed in 資產設備 runs).

    Its lines are shorter than the frame's median text box, so validation must judge the
    bands against each other rather than against the frame median.
    """
    lines = cut_text_box_into_lines(
        _dense_list_crop(),
        median_line_h=frame_median_h,
    )
    assert lines is not None
    assert len(lines) == 20
    assert all((y1 - y0) == 11 for _x0, y0, _x1, y1 in lines)
    # Still disjoint, so the OCR-side union cannot put them back together.
    ys = [(y0, y1) for _x0, y0, _x1, y1 in lines]
    assert all(a[1] <= b[0] for a, b in zip(ys, ys[1:], strict=False))


def test_cut_drops_sliver_bands_from_highlighted_rows():
    """A 1px run (the edge of a selected-row background) must not fail the whole box."""
    img = np.full((60, CROP_W, 3), 255, dtype=np.uint8)
    _paint_glyph_rows(img, 10, 21, fg=0)
    img[26, 2 : CROP_W - 4] = 0  # lone 1px band
    _paint_glyph_rows(img, 32, 43, fg=0)

    lines = cut_text_box_into_lines(img, median_line_h=MEDIAN_LINE_H)
    assert lines is not None
    assert [(y0, y1) for _x0, y0, _x1, y1 in lines] == [(10, 21), (32, 43)]


def test_cut_rejects_glyph_internal_blank_rows():
    """Blank rows *inside* glyphs (二, 三) must not be mistaken for line separators."""
    img = np.full((48, CROP_W, 3), 255, dtype=np.uint8)
    for top in (8, 28):
        _paint_glyph_rows(img, top, top + 3, fg=0)
        _paint_glyph_rows(img, top + 4, top + 7, fg=0)
    assert cut_text_box_into_lines(img, median_line_h=MEDIAN_LINE_H) is None


def test_cut_rejects_blank_crop():
    blank = np.full((CROP_H, CROP_W, 3), 127, dtype=np.uint8)
    assert cut_text_box_into_lines(blank, median_line_h=MEDIAN_LINE_H) is None


def _frame_with_tall_text_box(*, bg: int = 0, fg: int = 255) -> np.ndarray:
    """200×200 frame holding one two-line block at (10, 10)."""
    img = np.full((200, 200, 3), bg, dtype=np.uint8)
    _paint_glyph_rows(img, *BAND_1, fg=fg, off_x=10, off_y=10)
    _paint_glyph_rows(img, *BAND_2, fg=fg, off_x=10, off_y=10)
    return img


def _tall_box_detections() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One tall text box plus two normal-height text boxes (median height 25)."""
    xyxy = np.asarray(
        [
            [10.0, 10.0, 10.0 + CROP_W, 10.0 + CROP_H],
            [10.0, 120.0, 90.0, 145.0],
            [10.0, 150.0, 90.0, 175.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    cls = np.asarray([YOLO_CLASS_TEXT] * 3, dtype=np.int32)
    return xyxy, scores, cls


def test_cut_multiline_replaces_tall_box_and_keeps_the_rest():
    bgr = _frame_with_tall_text_box()
    xyxy, scores, cls = _tall_box_detections()

    out_xy, out_sc, out_cls, cut = cut_multiline_text_boxes(bgr, xyxy, scores, cls)

    assert cut == 1
    # Tall box replaced by 2 bands; the two normal boxes pass through.
    assert len(out_xy) == 4
    assert all(int(c) == YOLO_CLASS_TEXT for c in out_cls)
    bands = [b for b in out_xy if b[1] < 100.0]
    assert len(bands) == 2
    for band in bands:
        # Inside the parent box, and tighter than it.
        assert band[0] >= 10.0 and band[2] <= 10.0 + CROP_W
        assert band[1] >= 10.0 and band[3] <= 10.0 + CROP_H
        assert (band[3] - band[1]) < CROP_H
    # New boxes inherit the parent score.
    assert all(abs(float(s) - 0.9) < 1e-6 for s, b in zip(out_sc, out_xy, strict=True) if b[1] < 100.0)


def test_cut_multiline_skips_boxes_that_are_not_tall():
    bgr = _frame_with_tall_text_box()
    xyxy, scores, cls = _tall_box_detections()
    # Make every box the same height so nothing is tall vs the median.
    xyxy[0, 3] = xyxy[0, 1] + 25.0

    out_xy, _out_sc, _out_cls, cut = cut_multiline_text_boxes(bgr, xyxy, scores, cls)

    assert cut == 0
    assert np.array_equal(out_xy, xyxy)


def test_cut_multiline_ignores_non_text_classes():
    bgr = _frame_with_tall_text_box()
    xyxy, scores, cls = _tall_box_detections()
    cls[0] = YOLO_CLASS_ELEMENT

    out_xy, _out_sc, out_cls, cut = cut_multiline_text_boxes(bgr, xyxy, scores, cls)

    assert cut == 0
    assert np.array_equal(out_xy, xyxy)
    assert int(out_cls[0]) == YOLO_CLASS_ELEMENT


def test_cut_multiline_handles_empty_input():
    bgr = _frame_with_tall_text_box()
    out_xy, out_sc, out_cls, cut = cut_multiline_text_boxes(
        bgr,
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
    )
    assert cut == 0
    assert len(out_xy) == 0 and len(out_sc) == 0 and len(out_cls) == 0


def _fake_end2end(
    n_valid: int,
    *,
    slot_count: int = 300,
    score: float = 0.9,
    cls_id: int = YOLO_CLASS_TEXT,
    box_xyxy: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    out = np.zeros((1, slot_count, 6), dtype=np.float32)
    if box_xyxy is None:
        box_xyxy = (10.0, 10.0, 30.0, 30.0)
    n = min(n_valid, slot_count)
    for i in range(n):
        out[0, i, 0:4] = box_xyxy
        out[0, i, 4] = score
        out[0, i, 5] = float(cls_id)
    return out


def _spy_on_line_cut(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"n": 0}
    real = yolo_mod.cut_multiline_text_boxes

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(yolo_mod, "cut_multiline_text_boxes", spy)
    return calls


def test_run_yolo_invokes_line_cut_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", lambda _img: _fake_end2end(3))
    calls = _spy_on_line_cut(monkeypatch)

    run_yolo_onnx_end2end(
        np.zeros((640, 640, 3), dtype=np.uint8),
        class_ids={YOLO_CLASS_TEXT},
        line_cut=True,
    )
    assert calls["n"] == 1


def test_run_yolo_skips_line_cut_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", lambda _img: _fake_end2end(3))
    calls = _spy_on_line_cut(monkeypatch)

    run_yolo_onnx_end2end(
        np.zeros((640, 640, 3), dtype=np.uint8),
        class_ids={YOLO_CLASS_TEXT},
        line_cut=False,
    )
    assert calls["n"] == 0


def test_run_yolo_skips_line_cut_without_text_class(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        yolo_mod,
        "_run_yolo_raw_output",
        lambda _img: _fake_end2end(3, cls_id=YOLO_CLASS_ELEMENT),
    )
    calls = _spy_on_line_cut(monkeypatch)

    run_yolo_onnx_end2end(
        np.zeros((640, 640, 3), dtype=np.uint8),
        class_ids={YOLO_CLASS_ELEMENT},
        line_cut=True,
    )
    assert calls["n"] == 0


def test_run_yolo_runs_line_cut_when_overlap_refine_accepts_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Stage 4 must not be skipped by Stage 3's "nothing changed" path."""
    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", lambda _img: _fake_end2end(3))
    monkeypatch.setattr(
        yolo_mod,
        "refine_overlapping_text_with_crops",
        lambda _bgr, xyxy, scores, cls, **_kw: (xyxy, scores, cls, 0),
    )
    calls = _spy_on_line_cut(monkeypatch)

    run_yolo_onnx_end2end(
        np.zeros((640, 640, 3), dtype=np.uint8),
        class_ids={YOLO_CLASS_TEXT},
        overlap_refine=True,
        line_cut=True,
    )
    assert calls["n"] == 1
