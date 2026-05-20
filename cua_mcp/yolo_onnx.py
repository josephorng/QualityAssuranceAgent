"""
YOLOv26 ONNX helpers — ONNX Runtime on CPU.

Ultralytics-style exports use a fixed square input (default 640): BGR → RGB, ``NCHW``,
``float32 / 255``, then ``InferenceSession.run``.

Two post-process paths:

* **Raw head** ``(1, 4+nc, num_anchors)`` — decode ``cx,cy,w,h`` + class scores, optional
  Python NMS (:func:`nms_indices_xyxy`).
* **End-to-end** ``(1, N, 6+)`` with ``end2end=True`` — ``x1,y1,x2,y2,score,cls``; NMS is
  in the graph. Used by :data:`DEFAULT_YOLO_ONNX_PATH` (``cua_mcp/best.onnx``), classes
  ``text`` (:data:`YOLO_CLASS_TEXT`) and ``element`` (:data:`YOLO_CLASS_ELEMENT`).

Tune defaults via ``DEFAULT_CONF_*`` / ``DEFAULT_IOU_*``, or pass keyword args per call.

Shared :func:`run_best_onnx_end2end` / :func:`get_cached_cpu_session` keep one ONNX Runtime
session per resolved model path (OCR text + UI element both use ``best.onnx``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import onnxruntime as ort

YOLO_ONNX_INPUT_SIZE: int = 640
DEFAULT_PROVIDERS: Sequence[str] = ("CPUExecutionProvider",)

# Raw-head decode (NMS in Python)
DEFAULT_CONF_YOLOV26_RAW: float = 0.05
DEFAULT_IOU_YOLOV26_RAW: float = 0.7

# End-to-end decode (NMS in ONNX graph)
DEFAULT_CONF_YOLOV26_END2END: float = 0.05

# ``cua_mcp/best.onnx`` classes (Ultralytics metadata: Text=0, Element=1)
YOLO_CLASS_TEXT: int = 0
YOLO_CLASS_ELEMENT: int = 1
YOLO_CLASS_NAMES: dict[int, str] = {
    YOLO_CLASS_TEXT: "text",
    YOLO_CLASS_ELEMENT: "element",
}

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_ONNX_PATH = _PACKAGE_DIR / "best.onnx"


def create_cpu_session(model_path: str | Path) -> tuple[ort.InferenceSession, str]:
    """Build an ONNX Runtime session on CPU and return ``(session, input_tensor_name)``."""
    path = Path(model_path)
    session = ort.InferenceSession(str(path), providers=list(DEFAULT_PROVIDERS))
    return session, session.get_inputs()[0].name


_SESSION_BY_RESOLVED_PATH: dict[str, tuple[ort.InferenceSession, str]] = {}


def get_cached_cpu_session(
    model_path: str | Path,
    *,
    on_created: Callable[[Path], None] | None = None,
) -> tuple[ort.InferenceSession, str]:
    """
    Return a cached CPU ``InferenceSession`` for ``model_path`` (one session per resolved path).

    ``on_created`` runs at most once per path, immediately after the session is constructed.
    """
    path = Path(model_path).expanduser().resolve()
    key = str(path)
    if key not in _SESSION_BY_RESOLVED_PATH:
        if not path.is_file():
            raise FileNotFoundError(f"YOLO ONNX model not found: {path}")
        _SESSION_BY_RESOLVED_PATH[key] = create_cpu_session(path)
        if on_created is not None:
            on_created(path)
    return _SESSION_BY_RESOLVED_PATH[key]


def run_best_onnx_end2end(
    bgr: np.ndarray,
    *,
    class_ids: set[int],
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    model_path: str | Path | None = None,
    on_session_created: Callable[[Path], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess ``bgr``, run the packaged multi-class YOLOv26 end2end ONNX (default
    :data:`DEFAULT_YOLO_ONNX_PATH`), and return :func:`decode_yolov26_end2end` outputs
    ``(xyxy, scores, class_ids)`` filtered to ``class_ids``.

    ``on_session_created`` runs only when a new cached session is built for the model path
    (first use in the process, or first use of a new ``model_path``); later calls reuse the
    session and do not invoke it again.
    """
    path = DEFAULT_YOLO_ONNX_PATH if model_path is None else Path(model_path)
    session, input_name = get_cached_cpu_session(
        path, on_created=on_session_created
    )
    img_data, h0, w0 = bgr_to_nchw_normalized(bgr)
    outputs = session.run(None, {input_name: img_data})
    return decode_yolov26_end2end(
        outputs[0],
        h0,
        w0,
        conf_threshold=conf_threshold,
        class_ids=class_ids,
    )


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


# def decode_yolov26_raw_output(
#     pred: np.ndarray,
#     orig_h: int,
#     orig_w: int,
#     *,
#     conf_threshold: float = DEFAULT_CONF_YOLOV26_RAW,
#     iou_threshold: float = DEFAULT_IOU_YOLOV26_RAW,
#     input_size: int = YOLO_ONNX_INPUT_SIZE,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     YOLOv26 **raw** ONNX head: ``(1, 4+nc, num_anchors)``.

#     Each anchor is ``cx, cy, w, h`` (+ class scores) in ``input_size`` pixel space.
#     Returns ``(xyxy, scores)`` in original image pixels after score filter and NMS.
#     """
#     if pred.ndim != 3 or pred.shape[1] < 5:
#         raise RuntimeError(f"unexpected YOLOv26 raw ONNX output shape: {pred.shape}")
#     pred_t = pred[0].T
#     boxes = pred_t[:, :4].astype(np.float32)
#     scores = (
#         pred_t[:, 4:].max(axis=1).astype(np.float32)
#         if pred_t.shape[1] > 5
#         else pred_t[:, 4].astype(np.float32)
#     )
#     mask = scores >= conf_threshold
#     boxes = boxes[mask]
#     scores = scores[mask]
#     if len(boxes) == 0:
#         return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
#     cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     sx = orig_w / float(input_size)
#     sy = orig_h / float(input_size)
#     x1 = (cx - bw / 2.0) * sx
#     y1 = (cy - bh / 2.0) * sy
#     x2 = (cx + bw / 2.0) * sx
#     y2 = (cy + bh / 2.0) * sy
#     xyxy = np.stack([x1, y1, x2, y2], axis=1)
#     keep = nms_indices_xyxy(xyxy, scores, iou_threshold)
#     if not keep:
#         return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
#     idx = np.asarray(keep, dtype=np.int64)
#     return xyxy[idx], scores[idx]

def decode_yolov26_end2end(
    det: np.ndarray,
    orig_h: int,
    orig_w: int,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    input_size: int = YOLO_ONNX_INPUT_SIZE,
    class_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes YOLOv26 end-to-end ONNX output of shape (1, N, 6+).
    
    Returns:
        xyxy: (M, 4) np.ndarray of type int32 (ready for drawing/cropping)
        scores: (M,) np.ndarray of type float32
        class_ids: (M,) np.ndarray of type int64
    """
    if det.ndim != 3 or det.shape[-1] < 6:
        raise RuntimeError(f"Unexpected YOLOv26 end2end ONNX output shape: {det.shape}")
        
    # Remove batch dimension -> shape (N, 6+)
    det = det[0]
    
    # 1. Filter by confidence threshold first (highly memory efficient)
    scores = det[:, 4].astype(np.float32)
    mask = scores >= conf_threshold
    
    # 2. Filter by targeted class IDs if provided
    cls = det[:, 5].astype(np.int64)
    if class_ids is not None:
        mask &= np.isin(cls, list(class_ids))
        
    # Apply master mask
    det = det[mask]
    scores = scores[mask]
    cls = cls[mask]
    
    # Handle empty detection graph cleanly
    if len(det) == 0:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
        
    # 3. Calculate scale factors
    sx = orig_w / float(input_size)
    sy = orig_h / float(input_size)
    
    # 4. Extract and scale coordinates
    xyxy = det[:, :4].copy().astype(np.float32)
    xyxy[:, [0, 2]] *= sx
    xyxy[:, [1, 3]] *= sy
    
    # 5. Boundary Protection: Clip coordinates to actual image boundaries
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)
    
    # 6. Convert to integer so it plays nice with cv2.rectangle / image slicers
    xyxy = np.round(xyxy).astype(np.int32)
    
    return xyxy, scores, cls
