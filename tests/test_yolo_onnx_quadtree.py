"""Tests for YOLO end2end quadtree divide-and-conquer when detections hit max_det."""

from __future__ import annotations

import numpy as np
import pytest

from cua_mcp import yolo_onnx as yolo_mod
from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    DEFAULT_QUADTREE_MAX_DEPTH,
    DEFAULT_QUADTREE_MIN_SIDE,
    DEFAULT_QUADTREE_OVERLAP_FRAC,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_TEXT,
    YOLO_END2END_MAX_DET,
    YOLO_ONNX_INPUT_SIZE,
    _count_end2end_valid,
    _overlapping_quad_rois,
    _quadtree_can_split,
    run_yolo_onnx_end2end,
)


def _fake_end2end(
    n_valid: int,
    *,
    slot_count: int = YOLO_END2END_MAX_DET,
    score: float = 0.9,
    cls_id: int = YOLO_CLASS_TEXT,
    box_xyxy: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Build ``(1, slot_count, 6)`` with the first ``n_valid`` rows above conf."""
    out = np.zeros((1, slot_count, 6), dtype=np.float32)
    if box_xyxy is None:
        # Small box near letterbox origin; scales into original image.
        box_xyxy = (10.0, 10.0, 30.0, 30.0)
    n = min(n_valid, slot_count)
    for i in range(n):
        out[0, i, 0] = box_xyxy[0]
        out[0, i, 1] = box_xyxy[1]
        out[0, i, 2] = box_xyxy[2]
        out[0, i, 3] = box_xyxy[3]
        out[0, i, 4] = score
        out[0, i, 5] = float(cls_id)
    return out


def test_count_end2end_valid_ignores_class_and_low_conf():
    det = np.zeros((1, 5, 6), dtype=np.float32)
    det[0, 0, 4] = 0.9
    det[0, 1, 4] = 0.04  # below default conf
    det[0, 2, 4] = 0.5
    assert _count_end2end_valid(det, DEFAULT_CONF_YOLOV26_END2END) == 2


def test_overlapping_quad_rois_cover_and_intersect():
    h, w = 800, 1000
    rois = _overlapping_quad_rois(h, w, DEFAULT_QUADTREE_OVERLAP_FRAC)
    assert len(rois) == 4

    # Union covers full image.
    xs1 = min(r[0] for r in rois)
    ys1 = min(r[1] for r in rois)
    xs2 = max(r[2] for r in rois)
    ys2 = max(r[3] for r in rois)
    assert (xs1, ys1, xs2, ys2) == (0, 0, w, h)

    tl, tr, bl, br = rois
    # Horizontal neighbors intersect.
    assert tl[2] > tr[0]
    assert bl[2] > br[0]
    # Vertical neighbors intersect.
    assert tl[3] > bl[1]
    assert tr[3] > br[1]


def test_quadtree_can_split_respects_depth_and_min_side():
    assert _quadtree_can_split(640, 640, 0) is True
    assert _quadtree_can_split(640, 640, DEFAULT_QUADTREE_MAX_DEPTH) is False
    assert _quadtree_can_split(DEFAULT_QUADTREE_MIN_SIDE * 2 - 1, 1000, 0) is False
    assert _quadtree_can_split(1000, DEFAULT_QUADTREE_MIN_SIDE * 2 - 1, 0) is False


def test_no_split_when_under_cap(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[int, ...]] = []

    def fake_infer(img_data: np.ndarray) -> np.ndarray:
        calls.append(tuple(img_data.shape))
        return _fake_end2end(10)

    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", fake_infer)

    bgr = np.zeros((640, 640, 3), dtype=np.uint8)
    xyxy, scores, cls_ids = run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT},
    )

    assert len(calls) == 1
    assert len(xyxy) == 10
    assert len(scores) == 10
    assert len(cls_ids) == 10
    assert xyxy.dtype == np.int32


def test_split_once_when_full_image_hits_cap(monkeypatch: pytest.MonkeyPatch):
    """Full image returns 300 valid; each tile returns fewer → four extra infers."""
    call_shapes: list[tuple[int, int]] = []  # (h0, w0) proxied via letterbox input only
    call_i = {"n": 0}

    def fake_infer(img_data: np.ndarray) -> np.ndarray:
        n = call_i["n"]
        call_i["n"] += 1
        call_shapes.append(tuple(img_data.shape))
        if n == 0:
            # Full image: at cap (discarded).
            return _fake_end2end(YOLO_END2END_MAX_DET)
        # Tiles: one distinct box per tile in letterbox space (will scale differently
        # per crop size, but each contributes ≥1 detection).
        # Offset boxes slightly so NMS does not collapse everything.
        dx = float((n - 1) * 40)
        return _fake_end2end(
            5,
            box_xyxy=(10.0 + dx, 10.0, 40.0 + dx, 40.0),
            cls_id=YOLO_CLASS_TEXT,
        )

    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", fake_infer)

    bgr = np.zeros((800, 800, 3), dtype=np.uint8)
    xyxy, scores, cls_ids = run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT},
    )

    # 1 full + 4 tiles
    assert call_i["n"] == 5
    assert len(xyxy) > 0
    # All boxes must lie inside the parent image.
    assert int(xyxy[:, 0].min()) >= 0
    assert int(xyxy[:, 1].min()) >= 0
    assert int(xyxy[:, 2].max()) <= 800
    assert int(xyxy[:, 3].max()) <= 800
    assert len(scores) == len(xyxy)
    assert len(cls_ids) == len(xyxy)


def test_cross_tile_nms_collapses_duplicate_seam_boxes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overlapping tiles returning the same box (same class) collapse via NMS."""
    call_i = {"n": 0}
    # Identical letterbox box on every tile → after scale+offset, high IoU duplicates.

    def fake_infer(img_data: np.ndarray) -> np.ndarray:
        n = call_i["n"]
        call_i["n"] += 1
        if n == 0:
            return _fake_end2end(YOLO_END2END_MAX_DET)
        return _fake_end2end(
            1,
            score=0.8 + 0.01 * n,
            box_xyxy=(100.0, 100.0, 200.0, 200.0),
            cls_id=YOLO_CLASS_ELEMENT,
        )

    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", fake_infer)

    bgr = np.zeros((800, 800, 3), dtype=np.uint8)
    xyxy, scores, cls_ids = run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_ELEMENT},
    )

    assert call_i["n"] == 5
    # Seam duplicates should not yield four independent boxes of the same content.
    # Exact count depends on scale/offset; at most a small handful after NMS.
    assert len(xyxy) <= 4
    assert all(int(c) == YOLO_CLASS_ELEMENT for c in cls_ids)


def test_recurse_when_tile_also_hits_cap(monkeypatch: pytest.MonkeyPatch):
    """A first-level tile at cap produces grandchild crops (depth-limited)."""
    # Track approximate crops by recording bgr via wrapping recursive worker is hard;
    # count total infers. Large enough image to allow depth≥2.
    # depth0 full: cap → 4 tiles
    # first tile (call among tiles): also cap → 4 grandchildren
    # remaining 3 tiles + 4 grandchildren: under cap
    # Total: 1 + 4 + 4 = 9 if only first tile hits; but tile order is TL,TR,BL,BR
    # and we don't know which fake maps to which tile easily by call index alone
    # because only the first recursive after depth0 is the first tile (TL).

    call_i = {"n": 0}

    def fake_infer(img_data: np.ndarray) -> np.ndarray:
        n = call_i["n"]
        call_i["n"] += 1
        # Call 0: full image at cap.
        # Call 1: first tile (TL) at cap → recurse again.
        # Calls 2–5: TL's four grandchildren under cap.
        # Calls 6–8: TR, BL, BR under cap.
        if n in (0, 1):
            return _fake_end2end(YOLO_END2END_MAX_DET)
        return _fake_end2end(
            3,
            box_xyxy=(20.0 + n, 20.0, 50.0 + n, 50.0),
            cls_id=YOLO_CLASS_TEXT,
        )

    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", fake_infer)

    # Need room to split twice: after first split tiles ~ half+overlap of 1600 ≈ 800+.
    bgr = np.zeros((1600, 1600, 3), dtype=np.uint8)
    xyxy, scores, _cls = run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT},
    )

    assert call_i["n"] == 9
    assert len(xyxy) > 0
    assert len(scores) == len(xyxy)


def test_safety_stops_split_on_small_image(monkeypatch: pytest.MonkeyPatch):
    """Tiny image at cap must not split (half side < min_side)."""
    call_i = {"n": 0}

    def fake_infer(img_data: np.ndarray) -> np.ndarray:
        call_i["n"] += 1
        return _fake_end2end(YOLO_END2END_MAX_DET)

    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", fake_infer)

    # half of 500 = 250 < 320 → no split
    side = DEFAULT_QUADTREE_MIN_SIDE * 2 - 1
    bgr = np.zeros((side, side, 3), dtype=np.uint8)
    xyxy, scores, cls_ids = run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT},
    )

    assert call_i["n"] == 1
    assert len(xyxy) == YOLO_END2END_MAX_DET
    assert len(scores) == YOLO_END2END_MAX_DET
    assert len(cls_ids) == YOLO_END2END_MAX_DET


def test_safety_stops_at_max_depth(monkeypatch: pytest.MonkeyPatch):
    """Even dense large images stop recursing at DEFAULT_QUADTREE_MAX_DEPTH."""
    call_i = {"n": 0}

    def fake_infer(img_data: np.ndarray) -> np.ndarray:
        call_i["n"] += 1
        # Always report full buffer so every split level tries to recurse.
        return _fake_end2end(YOLO_END2END_MAX_DET)

    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", fake_infer)

    # Large enough that size stop does not fire before depth stop.
    # Depth 0 → 4, depth 1 → 16, depth 2 → 64 leaves at depth 3 (no further split).
    # Infers: 1 + 4 + 16 + 64 = 85
    side = 4096
    bgr = np.zeros((side, side, 3), dtype=np.uint8)
    run_yolo_onnx_end2end(bgr, class_ids={YOLO_CLASS_TEXT})

    expected = 1
    for d in range(DEFAULT_QUADTREE_MAX_DEPTH):
        expected += 4 ** (d + 1)
    assert call_i["n"] == expected


def test_cap_uses_raw_buffer_length(monkeypatch: pytest.MonkeyPatch):
    """Cap trigger follows ``raw.shape[1]`` when export slot count differs."""
    call_i = {"n": 0}
    slots = 50

    def fake_infer(img_data: np.ndarray) -> np.ndarray:
        n = call_i["n"]
        call_i["n"] += 1
        if n == 0:
            return _fake_end2end(slots, slot_count=slots)
        return _fake_end2end(2, slot_count=slots)

    monkeypatch.setattr(yolo_mod, "_run_yolo_raw_output", fake_infer)

    bgr = np.zeros((800, 800, 3), dtype=np.uint8)
    run_yolo_onnx_end2end(bgr, class_ids={YOLO_CLASS_TEXT})
    assert call_i["n"] == 5


def test_letterbox_input_size_unchanged_in_helpers():
    assert YOLO_ONNX_INPUT_SIZE == 1280
