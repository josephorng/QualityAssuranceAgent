import numpy as np
import pytest

from cua_mcp.yolo_onnx import (
    DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
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
