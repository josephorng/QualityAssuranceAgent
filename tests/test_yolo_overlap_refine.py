"""Tests for Stage 3 text overlap refine (yolo_divide_conquer + yolo_onnx hook)."""

from __future__ import annotations

import numpy as np
import pytest

from cua_mcp import yolo_onnx as yolo_mod
from cua_mcp.yolo_divide_conquer import (
    find_text_overlap_refine_clusters,
    refine_overlapping_text_with_crops,
)
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT, run_yolo_onnx_end2end


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


def test_find_clusters_links_iou_overlap():
    xyxy = np.asarray(
        [
            [10.0, 10.0, 80.0, 30.0],
            [20.0, 15.0, 90.0, 35.0],  # overlaps first
            [200.0, 10.0, 260.0, 28.0],  # isolated
        ],
        dtype=np.float32,
    )
    cls = np.asarray([0, 0, 0], dtype=np.int32)
    clusters = find_text_overlap_refine_clusters(xyxy, cls, text_class_id=0)
    assert any(set(c) == {0, 1} for c in clusters)
    assert not any(2 in c and len(c) > 1 for c in clusters)


def test_refine_replaces_crossing_cluster_with_cleaner_crop():
    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    # Two crossing/overlapping text lines.
    xyxy = np.asarray(
        [
            [40.0, 40.0, 160.0, 70.0],
            [50.0, 55.0, 170.0, 90.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.85], dtype=np.float32)
    cls = np.asarray([0, 0], dtype=np.int32)

    def predict_crop(_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Cleaner non-overlapping lines inside the crop (local coords).
        # Caller offsets strips; one-shot crop uses crop-local pixels.
        return (
            np.asarray(
                [
                    [10.0, 8.0, 110.0, 28.0],
                    [10.0, 36.0, 110.0, 56.0],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.95, 0.94], dtype=np.float32),
            np.asarray([0, 0], dtype=np.int32),
        )

    out_xy, out_sc, out_cls, accepted = refine_overlapping_text_with_crops(
        bgr,
        xyxy,
        scores,
        cls,
        predict_crop_fn=predict_crop,
        text_class_id=0,
    )
    assert accepted == 1
    assert len(out_xy) == 2
    assert all(int(c) == 0 for c in out_cls)
    assert len(out_sc) == 2


def test_run_yolo_invokes_overlap_refine_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        yolo_mod,
        "_run_yolo_raw_output",
        lambda _img: _fake_end2end(3),
    )
    calls = {"n": 0}
    real = yolo_mod.refine_overlapping_text_with_crops

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(yolo_mod, "refine_overlapping_text_with_crops", spy)

    bgr = np.zeros((640, 640, 3), dtype=np.uint8)
    run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT},
        overlap_refine=True,
    )
    assert calls["n"] == 1


def test_run_yolo_skips_overlap_refine_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        yolo_mod,
        "_run_yolo_raw_output",
        lambda _img: _fake_end2end(3),
    )
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            0,
        )

    monkeypatch.setattr(yolo_mod, "refine_overlapping_text_with_crops", spy)

    bgr = np.zeros((640, 640, 3), dtype=np.uint8)
    run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT},
        overlap_refine=False,
    )
    assert calls["n"] == 0


def test_run_yolo_skips_overlap_refine_without_text_class(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        yolo_mod,
        "_run_yolo_raw_output",
        lambda _img: _fake_end2end(3, cls_id=YOLO_CLASS_ELEMENT),
    )
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("refine should not run without text class")

    monkeypatch.setattr(yolo_mod, "refine_overlapping_text_with_crops", spy)

    bgr = np.zeros((640, 640, 3), dtype=np.uint8)
    run_yolo_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_ELEMENT},
        overlap_refine=True,
    )
    assert calls["n"] == 0
