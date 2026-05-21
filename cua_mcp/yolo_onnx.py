"""
YOLOv26 ONNX helpers — ONNX Runtime on CPU.

Ultralytics-compatible preprocessing (default 640): ``LetterBox``–style resize–pad to a
square tensor, BGR → RGB, ``NCHW``, ``float32 / 255``, then ``InferenceSession.run``. Box
coordinates are mapped back with the same ``scale_boxes`` math as ``ultralytics``
(``padding=True``, ``ratio_pad=None``).

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
# Matches ``ultralytics.data.augment.LetterBox`` default ``padding_value``.
YOLO_LETTERBOX_PAD_BGR: tuple[int, int, int] = (114, 114, 114)
DEFAULT_PROVIDERS: Sequence[str] = ("CPUExecutionProvider",)

# Raw-head decode (NMS in Python)
DEFAULT_CONF_YOLOV26_RAW: float = 0.05
DEFAULT_IOU_YOLOV26_RAW: float = 0.7

# End-to-end decode (NMS in ONNX graph)
DEFAULT_CONF_YOLOV26_END2END: float = 0.05

# After decode, optionally merge same-class detections when pairwise IoU exceeds
# :data:`DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD` (intersection/union); each merged group is the
# axis-aligned union with max score. Default off: set ``DEFAULT_MERGE_TOUCHING_SAME_CLASS`` True
# or pass ``merge_touching_same_class=True`` to enable.
DEFAULT_MERGE_TOUCHING_SAME_CLASS: bool = False
# Pairs of same-class boxes are linked (and merged transitively) when ``IoU >`` this value.
DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD: float = 0.5

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
    merge_touching_same_class: bool = DEFAULT_MERGE_TOUCHING_SAME_CLASS,
    merge_same_class_iou_threshold: float = DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess ``bgr``, run the packaged multi-class YOLOv26 end2end ONNX (default
    :data:`DEFAULT_YOLO_ONNX_PATH`), and return :func:`decode_yolov26_end2end` outputs
    ``(xyxy, scores, class_ids)`` filtered to ``class_ids``.

    ``on_session_created`` runs only when a new cached session is built for the model path
    (first use in the process, or first use of a new ``model_path``); later calls reuse the
    session and do not invoke it again.

    Pass ``merge_touching_same_class=True`` to fuse same-class boxes whose pairwise IoU exceeds
    ``merge_same_class_iou_threshold`` after decode (see :data:`DEFAULT_MERGE_TOUCHING_SAME_CLASS`
    and :data:`DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD`).
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
        merge_touching_same_class=merge_touching_same_class,
        merge_same_class_iou_threshold=merge_same_class_iou_threshold,
    )


def bgr_to_nchw_normalized(
    bgr: np.ndarray, size: int = YOLO_ONNX_INPUT_SIZE
) -> tuple[np.ndarray, int, int]:
    """
    Preprocess a BGR image like ``ultralytics.data.augment.LetterBox`` (``auto=False``,
    ``scaleup=True``, ``center=True``): fit inside ``size``×``size`` with aspect ratio
    preserved, pad with :data:`YOLO_LETTERBOX_PAD_BGR`, then RGB CHW ``/255``, batch 1.

    Returns ``(input_nchw, orig_h, orig_w)`` where ``input_nchw`` has shape ``(1, 3, size, size)``.
    """
    letter_bgr = letterbox_bgr_ultralytics(bgr, size=size)
    rgb = cv2.cvtColor(letter_bgr, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    batch = np.expand_dims(chw, axis=0)
    h0, w0 = bgr.shape[:2]
    return batch, h0, w0


def letterbox_bgr_ultralytics(bgr: np.ndarray, size: int = YOLO_ONNX_INPUT_SIZE) -> np.ndarray:
    """
    Match ``LetterBox(new_shape=(size, size), auto=False, scaleup=True, center=True)`` on BGR uint8.
    """
    h0, w0 = bgr.shape[:2]
    new_shape = (size, size)
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (round(w0 * r), round(h0 * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0
    if (w0, h0) != new_unpad:
        bgr = cv2.resize(bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(
        bgr,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=YOLO_LETTERBOX_PAD_BGR,
    )


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

def scale_xyxy_letterboxed_to_original(
    xyxy: np.ndarray,
    orig_h: int,
    orig_w: int,
    *,
    input_h: int = YOLO_ONNX_INPUT_SIZE,
    input_w: int = YOLO_ONNX_INPUT_SIZE,
) -> np.ndarray:
    """
    Map ``xyxy`` from letterboxed ``input_h``×``input_w`` inference space to original
    ``(orig_h, orig_w)``. Matches ``ultralytics.utils.ops.scale_boxes(..., padding=True)`` when
    ``ratio_pad=None``.
    """
    if len(xyxy) == 0:
        return xyxy.astype(np.float32)
    img1_h, img1_w = input_h, input_w
    gain = min(img1_h / orig_h, img1_w / orig_w)
    pad_x = round((img1_w - orig_w * gain) / 2 - 0.1)
    pad_y = round((img1_h - orig_h * gain) / 2 - 0.1)
    out = xyxy.astype(np.float32).copy()
    out[:, 0] -= pad_x
    out[:, 1] -= pad_y
    out[:, 2] -= pad_x
    out[:, 3] -= pad_y
    out /= gain
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out


def _iou_xyxy(xy0: np.ndarray, xy1: np.ndarray) -> float:
    """Intersection-over-union for axis-aligned ``xyxy`` boxes (returns ``0.0`` if disjoint or degenerate union)."""
    ax1, ay1, ax2, ay2 = (float(xy0[i]) for i in range(4))
    bx1, by1, bx2, by2 = (float(xy1[i]) for i in range(4))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aw = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aw + bw - inter
    if union <= 0:
        return 0.0
    return inter / union


class _DisjointSet:
    __slots__ = ("_p",)

    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        p = self._p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self._p[pa] = pb


def merge_touching_same_class_xyxy(
    xyxy: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    *,
    min_iou: float = DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge detections **per class** when pairwise IoU is **strictly greater** than ``min_iou``
    (intersection over union); groups are connected transitively. Each output row is the union
    bbox of its group with the max score in that group.
    """
    if len(xyxy) == 0:
        return xyxy, scores, cls_ids

    xyxy = np.asarray(xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    cls_ids = np.asarray(cls_ids, dtype=np.int64).reshape(-1)

    merged_xy: list[np.ndarray] = []
    merged_sc: list[float] = []
    merged_cls: list[int] = []

    for c in sorted(int(x) for x in np.unique(cls_ids)):
        idx = np.flatnonzero(cls_ids == c)
        sub_xy = xyxy[idx]
        sub_sc = scores[idx]
        n = sub_xy.shape[0]
        dsu = _DisjointSet(n)
        for i in range(n):
            for j in range(i + 1, n):
                if _iou_xyxy(sub_xy[i], sub_xy[j]) > min_iou:
                    dsu.union(i, j)

        roots: dict[int, list[int]] = {}
        for i in range(n):
            roots.setdefault(dsu.find(i), []).append(i)

        for members in roots.values():
            gxy = sub_xy[members]
            gsc = sub_sc[members]
            union_box = np.array(
                [
                    float(gxy[:, 0].min()),
                    float(gxy[:, 1].min()),
                    float(gxy[:, 2].max()),
                    float(gxy[:, 3].max()),
                ],
                dtype=np.float32,
            )
            merged_xy.append(union_box)
            merged_sc.append(float(np.max(gsc)))
            merged_cls.append(c)

    return (
        np.stack(merged_xy, axis=0),
        np.asarray(merged_sc, dtype=np.float32),
        np.asarray(merged_cls, dtype=np.int64),
    )


def decode_yolov26_end2end(
    det: np.ndarray,
    orig_h: int,
    orig_w: int,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    input_size: int = YOLO_ONNX_INPUT_SIZE,
    class_ids: set[int] | None = None,
    merge_touching_same_class: bool = DEFAULT_MERGE_TOUCHING_SAME_CLASS,
    merge_same_class_iou_threshold: float = DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes YOLOv26 end-to-end ONNX output of shape (1, N, 6+).

    When ``merge_touching_same_class`` is True, same-class pairs with ``IoU > merge_same_class_iou_threshold``
    are merged transitively via :func:`merge_touching_same_class_xyxy`.

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
        
    # Map from letterboxed ``input_size`` space to original image (Ultralytics ``scale_boxes``).
    xyxy = scale_xyxy_letterboxed_to_original(
        det[:, :4],
        orig_h,
        orig_w,
        input_h=input_size,
        input_w=input_size,
    )

    if merge_touching_same_class:
        xyxy, scores, cls = merge_touching_same_class_xyxy(
            xyxy,
            scores,
            cls,
            min_iou=merge_same_class_iou_threshold,
        )

    # Convert to integer so it plays nice with cv2.rectangle / image slicers
    xyxy = np.round(xyxy).astype(np.int32)
    
    return xyxy, scores, cls
