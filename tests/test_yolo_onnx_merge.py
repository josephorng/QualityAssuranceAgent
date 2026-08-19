import numpy as np
import pytest

from cua_mcp.yolo_onnx import (
    DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
    expand_input_boxes_with_overlapping_text_xyxy,
    merge_touching_same_class_xyxy,
)


def test_merge_input_boxes_above_iou_threshold():
    xyxy = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.6, 0.9], dtype=np.float32)
    cls_ids = np.array([YOLO_CLASS_INPUT, YOLO_CLASS_INPUT], dtype=np.int64)

    out_xy, out_sc, out_cls = merge_touching_same_class_xyxy(
        xyxy,
        scores,
        cls_ids,
        min_iou=DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
        merge_class_ids={YOLO_CLASS_INPUT},
    )

    assert out_xy.shape == (1, 4)
    assert np.allclose(out_xy[0], [0.0, 0.0, 11.0, 11.0])
    assert out_sc[0] == pytest.approx(0.9)
    assert out_cls[0] == YOLO_CLASS_INPUT


def test_merge_scrollbar_only_leaves_other_classes():
    xyxy = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [100.0, 100.0, 110.0, 110.0],
            [105.0, 105.0, 115.0, 115.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.6, 0.9, 0.4, 0.8], dtype=np.float32)
    cls_ids = np.array(
        [
            YOLO_CLASS_INPUT,
            YOLO_CLASS_INPUT,
            YOLO_CLASS_ELEMENT,
            YOLO_CLASS_ELEMENT,
        ],
        dtype=np.int64,
    )

    out_xy, out_sc, out_cls = merge_touching_same_class_xyxy(
        xyxy,
        scores,
        cls_ids,
        min_iou=DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
        merge_class_ids={YOLO_CLASS_INPUT, YOLO_CLASS_SCROLLBAR},
    )

    assert out_xy.shape == (3, 4)
    assert np.sum(out_cls == YOLO_CLASS_INPUT) == 1
    assert np.sum(out_cls == YOLO_CLASS_ELEMENT) == 2


def test_expand_input_unions_overlapping_text_boxes():
    xyxy = np.array(
        [
            [10.0, 10.0, 50.0, 30.0],
            [40.0, 12.0, 80.0, 28.0],
            [42.0, 8.0, 70.0, 32.0],
            [200.0, 200.0, 240.0, 220.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    cls_ids = np.array(
        [
            YOLO_CLASS_INPUT,
            YOLO_CLASS_TEXT,
            YOLO_CLASS_TEXT,
            YOLO_CLASS_TEXT,
        ],
        dtype=np.int64,
    )

    out_xy, out_sc, out_cls = expand_input_boxes_with_overlapping_text_xyxy(
        xyxy, scores, cls_ids
    )

    assert np.allclose(out_xy[0], [10.0, 8.0, 80.0, 32.0])
    assert np.allclose(out_xy[1], xyxy[1])
    assert np.allclose(out_xy[2], xyxy[2])
    assert np.allclose(out_xy[3], xyxy[3])
    assert np.allclose(out_sc, scores)
    assert np.array_equal(out_cls, cls_ids)


def test_expand_input_leaves_non_overlapping_text_and_other_classes():
    xyxy = np.array(
        [
            [0.0, 0.0, 20.0, 10.0],
            [50.0, 0.0, 80.0, 10.0],
            [5.0, 2.0, 15.0, 8.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    cls_ids = np.array(
        [YOLO_CLASS_INPUT, YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT],
        dtype=np.int64,
    )

    out_xy, out_sc, out_cls = expand_input_boxes_with_overlapping_text_xyxy(
        xyxy, scores, cls_ids
    )

    assert np.allclose(out_xy, xyxy)
    assert np.allclose(out_sc, scores)
    assert np.array_equal(out_cls, cls_ids)


def test_expand_input_does_not_chain_through_grown_union():
    # Nearby label overlaps the expanded union, but not the original input.
    xyxy = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [8.0, 0.0, 20.0, 10.0],
            [18.0, 0.0, 30.0, 10.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    cls_ids = np.array(
        [YOLO_CLASS_INPUT, YOLO_CLASS_TEXT, YOLO_CLASS_TEXT],
        dtype=np.int64,
    )

    out_xy, _out_sc, _out_cls = expand_input_boxes_with_overlapping_text_xyxy(
        xyxy, scores, cls_ids
    )

    assert np.allclose(out_xy[0], [0.0, 0.0, 20.0, 10.0])
    assert np.allclose(out_xy[2], xyxy[2])
