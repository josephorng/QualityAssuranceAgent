"""
YOLOv26 ONNX helpers — ONNX Runtime on CPU.

Ultralytics-style exports use a fixed square input (default 640): BGR → RGB, ``NCHW``,
``float32 / 255``, then ``InferenceSession.run``.

Two post-process paths:

* **Raw head** ``(1, 4+nc, num_anchors)`` — decode ``cx,cy,w,h`` + class scores, optional
  Python NMS (:func:`nms_indices_xyxy`). Used by e.g. ``get_UI/model.onnx``.
* **End-to-end** ``(1, N, 6+)`` with ``end2end=True`` — ``x1,y1,x2,y2,score,cls``; NMS is
  in the graph. Used by e.g. ``read_screen_text/yolo_best.onnx``.

Tune defaults via ``DEFAULT_CONF_*`` / ``DEFAULT_IOU_*``, or pass keyword args per call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

YOLO_ONNX_INPUT_SIZE: int = 640
DEFAULT_PROVIDERS: Sequence[str] = ("CPUExecutionProvider",)

# Raw-head decode (NMS in Python)
DEFAULT_CONF_YOLOV26_RAW: float = 0.05
DEFAULT_IOU_YOLOV26_RAW: float = 0.7

# End-to-end decode (NMS in ONNX graph)
DEFAULT_CONF_YOLOV26_END2END: float = 0.25


def create_cpu_session(model_path: str | Path) -> tuple[ort.InferenceSession, str]:
    """Build an ONNX Runtime session on CPU and return ``(session, input_tensor_name)``."""
    path = Path(model_path)
    session = ort.InferenceSession(str(path), providers=list(DEFAULT_PROVIDERS))
    return session, session.get_inputs()[0].name


def bgr_to_nchw_normalized(
    bgr: np.ndarray, size: int = YOLO_ONNX_INPUT_SIZE
) -> tuple[np.ndarray, int, int]:
    """
    Preprocess a BGR image for YOLOv26 ONNX: resize to ``size``×``size``, RGB, CHW, /255, batch 1.

    Returns ``(input_nchw, orig_h, orig_w)`` where ``input_nchw`` has shape ``(1, 3, size, size)``.
    """
    h0, w0 = bgr.shape[:2]
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    batch = np.expand_dims(chw, axis=0)
    return batch, h0, w0


def nms_indices_xyxy(
    xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    """Return indices kept after NMS; ``xyxy`` is ``N×4`` float, ``scores`` length ``N``."""
    if len(xyxy) == 0:
        return []
    boxes_wh: list[list[float]] = []
    for row in xyxy:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        boxes_wh.append([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)])
    idx = cv2.dnn.NMSBoxes(
        boxes_wh,
        scores.astype(np.float32).tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )
    if idx is None or len(idx) == 0:
        return []
    flat = np.asarray(idx).reshape(-1)
    return [int(i) for i in flat]


def decode_yolov26_raw_output(
    pred: np.ndarray,
    orig_h: int,
    orig_w: int,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_RAW,
    iou_threshold: float = DEFAULT_IOU_YOLOV26_RAW,
    input_size: int = YOLO_ONNX_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    YOLOv26 **raw** ONNX head: ``(1, 4+nc, num_anchors)``.

    Each anchor is ``cx, cy, w, h`` (+ class scores) in ``input_size`` pixel space.
    Returns ``(xyxy, scores)`` in original image pixels after score filter and NMS.
    """
    if pred.ndim != 3 or pred.shape[1] < 5:
        raise RuntimeError(f"unexpected YOLOv26 raw ONNX output shape: {pred.shape}")
    pred_t = pred[0].T
    boxes = pred_t[:, :4].astype(np.float32)
    scores = (
        pred_t[:, 4:].max(axis=1).astype(np.float32)
        if pred_t.shape[1] > 5
        else pred_t[:, 4].astype(np.float32)
    )
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    sx = orig_w / float(input_size)
    sy = orig_h / float(input_size)
    x1 = (cx - bw / 2.0) * sx
    y1 = (cy - bh / 2.0) * sy
    x2 = (cx + bw / 2.0) * sx
    y2 = (cy + bh / 2.0) * sy
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms_indices_xyxy(xyxy, scores, iou_threshold)
    if not keep:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    idx = np.asarray(keep, dtype=np.int64)
    return xyxy[idx], scores[idx]


def decode_yolov26_end2end(
    det: np.ndarray,
    orig_h: int,
    orig_w: int,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    input_size: int = YOLO_ONNX_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    YOLOv26 **end2end** ONNX: ``(1, N, 6+)`` with ``x1, y1, x2, y2, score, class_id, ...``.

    Coordinates are in ``input_size`` space; NMS is assumed baked into the graph.
    Returns ``(xyxy, scores)`` in original image pixels (scores aligned with rows).
    """
    if det.ndim != 3 or det.shape[-1] < 6:
        raise RuntimeError(f"unexpected YOLOv26 end2end ONNX output shape: {det.shape}")
    det = det[0]
    scores = det[:, 4].astype(np.float32)
    mask = scores >= conf_threshold
    det = det[mask]
    scores = scores[mask]
    if len(det) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    sx = orig_w / float(input_size)
    sy = orig_h / float(input_size)
    xyxy = det[:, :4].copy().astype(np.float32)
    xyxy[:, [0, 2]] *= sx
    xyxy[:, [1, 3]] *= sy
    return xyxy, scores


# Back-compat alias (older name)
decode_yolov8_raw_output = decode_yolov26_raw_output
