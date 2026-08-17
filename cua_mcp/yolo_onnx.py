"""
YOLOv26 ONNX helpers — Triton inference + local preprocess/decode.

Ultralytics-compatible preprocessing (default 1280): ``LetterBox``–style resize–pad to a
square tensor, BGR → RGB, ``NCHW``, ``float32 / 255``, then Triton ``infer_yolo``. Box
coordinates are mapped back with the same ``scale_boxes`` math as ``ultralytics``
(``padding=True``, ``ratio_pad=None``).

End-to-end output ``(1, N, 6+)`` — ``x1,y1,x2,y2,score,cls``; NMS is in the graph
(Ultralytics default ``max_det`` ⇒ ``N`` ≈ :data:`YOLO_END2END_MAX_DET`). When a pass
fills that buffer, :func:`run_yolo_onnx_end2end` recursively splits the image into an
overlapping 2×2 quadtree, re-infers each tile, and merges results.

Classes: ``text`` (:data:`YOLO_CLASS_TEXT`), ``element`` (:data:`YOLO_CLASS_ELEMENT`),
``input`` (:data:`YOLO_CLASS_INPUT`), and ``scrollbar`` (:data:`YOLO_CLASS_SCROLLBAR`).

Tune defaults via ``DEFAULT_CONF_*``, or pass keyword args per call.
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

YOLO_ONNX_INPUT_SIZE: int = 1280
# Matches ``ultralytics.data.augment.LetterBox`` default ``padding_value``.
YOLO_LETTERBOX_PAD_BGR: tuple[int, int, int] = (114, 114, 114)

# End-to-end decode (NMS in ONNX graph)
DEFAULT_CONF_YOLOV26_END2END: float = 0.05
# Ultralytics end2end export default ``max_det`` / output slot count ``(1, N, 6+)``.
YOLO_END2END_MAX_DET: int = 300

# When valid conf detections fill :data:`YOLO_END2END_MAX_DET`, split into overlapping
# 2×2 tiles and re-run (quadtree). Only tiles that still hit the cap recurse further.
DEFAULT_QUADTREE_OVERLAP_FRAC: float = 0.125
DEFAULT_QUADTREE_MAX_DEPTH: int = 3
DEFAULT_QUADTREE_MIN_SIDE: int = 320
DEFAULT_CROSS_TILE_NMS_IOU: float = 0.5

# After decode, optionally merge same-class detections when pairwise IoU exceeds
# :data:`DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD` (intersection/union); each merged group is the
# axis-aligned union with max score.
#
# ``input`` and ``scrollbar`` are always merged when IoU exceeds the threshold.
# Set ``DEFAULT_MERGE_TOUCHING_SAME_CLASS`` True or pass ``merge_touching_same_class=True``
# to also merge ``text`` and ``element``.
DEFAULT_MERGE_TOUCHING_SAME_CLASS: bool = False
# Pairs of same-class boxes are linked (and merged transitively) when ``IoU >`` this value.
DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD: float = 0.2

# ``best.onnx`` classes (Ultralytics metadata: Text=0, Element=1, Input=2, Scrollbar=3)
YOLO_CLASS_TEXT: int = 0
YOLO_CLASS_ELEMENT: int = 1
YOLO_CLASS_INPUT: int = 2
YOLO_CLASS_SCROLLBAR: int = 3
DEFAULT_MERGE_TOUCHING_CLASS_IDS = frozenset({
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
})
# Synthetic picker IDs for OCR-derived UI candidates (not YOLO model class IDs).
PICKER_CLASS_TEXT = 100
PICKER_CLASS_OCR_ICON = 101
# YOLO ``element`` with empty OCR, plain-text OCR, or unmapped PUA — ambiguous / unknown.
PICKER_CLASS_UNKNOWN = 102

YOLO_CLASS_NAMES: dict[int, str] = {
    YOLO_CLASS_TEXT: "text",
    YOLO_CLASS_ELEMENT: "element",
    YOLO_CLASS_INPUT: "input",
    YOLO_CLASS_SCROLLBAR: "scrollbar",
    PICKER_CLASS_UNKNOWN: "unknown",
}

OCR_DETECTION_CLASS_IDS = frozenset({
    YOLO_CLASS_TEXT,
    YOLO_CLASS_ELEMENT,
})
UI_DETECTION_CLASS_IDS = frozenset({
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
})
MOUSE_TARGET_CLASS_IDS = frozenset({
    YOLO_CLASS_TEXT,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
})

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_ONNX_PATH = _PACKAGE_DIR / "best.onnx"


def _log_yolo_profile(message: str) -> None:
    try:
        from src.common.run_state import get_run_state_manager

        get_run_state_manager().log_info(f"[vision/yolo] {message}")
    except RuntimeError:
        pass


def _run_yolo_raw_output(img_data: np.ndarray) -> np.ndarray:
    from cua_mcp.vision_triton import infer_yolo

    started = time.perf_counter()
    out = infer_yolo(img_data)
    elapsed = time.perf_counter() - started
    _log_yolo_profile(
        f"infer backend=triton shape={list(img_data.shape)} "
        f"elapsed_s={elapsed:.3f}"
    )
    return out


def _count_end2end_valid(det: np.ndarray, conf_threshold: float) -> int:
    """Count raw end2end rows with ``score >= conf_threshold`` (no class filter)."""
    if det.ndim != 3 or det.shape[-1] < 6:
        return 0
    scores = det[0, :, 4].astype(np.float32)
    return int(np.count_nonzero(scores >= conf_threshold))


def _overlapping_quad_rois(
    h: int,
    w: int,
    overlap_frac: float = DEFAULT_QUADTREE_OVERLAP_FRAC,
) -> list[tuple[int, int, int, int]]:
    """
    Four ``(x1, y1, x2, y2)`` ROIs from a mid split with overlap expanded toward edges.

    ``overlap_frac`` is a fraction of each half-size (e.g. ``0.125`` ⇒ each tile grows by
    12.5% of half-width / half-height into the opposite half). Clipped to image bounds.
    """
    mid_x = w // 2
    mid_y = h // 2
    ox = int(round(mid_x * float(overlap_frac)))
    oy = int(round(mid_y * float(overlap_frac)))
    # TL, TR, BL, BR
    return [
        (0, 0, min(w, mid_x + ox), min(h, mid_y + oy)),
        (max(0, mid_x - ox), 0, w, min(h, mid_y + oy)),
        (0, max(0, mid_y - oy), min(w, mid_x + ox), h),
        (max(0, mid_x - ox), max(0, mid_y - oy), w, h),
    ]


def _offset_detections(xyxy: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """Translate ``xyxy`` by crop origin ``(ox, oy)``."""
    if len(xyxy) == 0:
        return xyxy
    out = np.asarray(xyxy, dtype=np.int32).copy()
    out[:, 0] += int(ox)
    out[:, 1] += int(oy)
    out[:, 2] += int(ox)
    out[:, 3] += int(oy)
    return out


def _dedupe_cross_tile(
    xyxy: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    *,
    iou_threshold: float = DEFAULT_CROSS_TILE_NMS_IOU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class OpenCV NMS to drop duplicate boxes across overlapping tiles."""
    if len(xyxy) == 0:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    xyxy = np.asarray(xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    cls_ids = np.asarray(cls_ids, dtype=np.int64).reshape(-1)

    kept_xy: list[np.ndarray] = []
    kept_sc: list[float] = []
    kept_cls: list[int] = []

    for c in sorted(int(x) for x in np.unique(cls_ids)):
        idx = np.flatnonzero(cls_ids == c)
        sub_xy = xyxy[idx]
        sub_sc = scores[idx]
        keep_local = nms_indices_xyxy(sub_xy, sub_sc, iou_threshold)
        for li in keep_local:
            kept_xy.append(sub_xy[li])
            kept_sc.append(float(sub_sc[li]))
            kept_cls.append(c)

    if not kept_xy:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.round(np.stack(kept_xy, axis=0)).astype(np.int32),
        np.asarray(kept_sc, dtype=np.float32),
        np.asarray(kept_cls, dtype=np.int64),
    )


def _quadtree_can_split(
    h: int,
    w: int,
    depth: int,
    *,
    max_depth: int = DEFAULT_QUADTREE_MAX_DEPTH,
    min_side: int = DEFAULT_QUADTREE_MIN_SIDE,
) -> bool:
    """True if recursive 2×2 split is allowed at this depth/size."""
    if depth >= max_depth:
        return False
    # After a mid split, each half (before overlap) must remain at least ``min_side``.
    if h // 2 < min_side or w // 2 < min_side:
        return False
    return True


def run_yolo_onnx_end2end(
    bgr: np.ndarray,
    *,
    class_ids: set[int],
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    merge_touching_same_class: bool = DEFAULT_MERGE_TOUCHING_SAME_CLASS,
    merge_same_class_iou_threshold: float = DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess ``bgr``, run YOLOv26 end2end via Triton, and return
    :func:`decode_yolov26_end2end` outputs ``(xyxy, scores, class_ids)`` filtered to
    ``class_ids``.

    When raw end2end detections with ``score >= conf_threshold`` fill the model buffer
    (:data:`YOLO_END2END_MAX_DET`), the image is recursively split into overlapping 2×2
    tiles; the truncated full-image pass is discarded and tile results are merged.

    Pass ``merge_touching_same_class=True`` to also fuse ``text`` and ``element`` boxes whose
    pairwise IoU exceeds ``merge_same_class_iou_threshold``. ``input`` and ``scrollbar`` are
    always merged at that threshold (see :data:`DEFAULT_MERGE_TOUCHING_CLASS_IDS`).
    """
    return _run_yolo_onnx_end2end_recursive(
        bgr,
        depth=0,
        class_ids=class_ids,
        conf_threshold=conf_threshold,
        merge_touching_same_class=merge_touching_same_class,
        merge_same_class_iou_threshold=merge_same_class_iou_threshold,
    )


def _run_yolo_onnx_end2end_recursive(
    bgr: np.ndarray,
    *,
    depth: int,
    class_ids: set[int],
    conf_threshold: float,
    merge_touching_same_class: bool,
    merge_same_class_iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_data, h0, w0 = bgr_to_nchw_normalized(bgr)
    raw = _run_yolo_raw_output(img_data)

    max_det = YOLO_END2END_MAX_DET
    if raw.ndim == 3 and raw.shape[1] > 0:
        max_det = int(raw.shape[1])
    valid = _count_end2end_valid(raw, conf_threshold)
    hit_cap = valid >= max_det

    if not hit_cap or not _quadtree_can_split(h0, w0, depth):
        return decode_yolov26_end2end(
            raw,
            h0,
            w0,
            conf_threshold=conf_threshold,
            class_ids=class_ids,
            merge_touching_same_class=merge_touching_same_class,
            merge_same_class_iou_threshold=merge_same_class_iou_threshold,
        )

    # Cap hit: discard truncated full-image decode; results come only from tiles.
    rois = _overlapping_quad_rois(h0, w0, DEFAULT_QUADTREE_OVERLAP_FRAC)
    _log_yolo_profile(
        f"quadtree split depth={depth} size={w0}x{h0} "
        f"valid={valid} max_det={max_det} tiles={len(rois)}"
    )

    all_xy: list[np.ndarray] = []
    all_sc: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []
    for x1, y1, x2, y2 in rois:
        tile = bgr[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        t_xy, t_sc, t_cls = _run_yolo_onnx_end2end_recursive(
            tile,
            depth=depth + 1,
            class_ids=class_ids,
            conf_threshold=conf_threshold,
            merge_touching_same_class=merge_touching_same_class,
            merge_same_class_iou_threshold=merge_same_class_iou_threshold,
        )
        if len(t_xy) == 0:
            continue
        all_xy.append(_offset_detections(t_xy, x1, y1))
        all_sc.append(t_sc)
        all_cls.append(t_cls)

    if not all_xy:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    xyxy = np.concatenate(all_xy, axis=0)
    scores = np.concatenate(all_sc, axis=0)
    cls_arr = np.concatenate(all_cls, axis=0)

    xyxy, scores, cls_arr = _dedupe_cross_tile(
        xyxy,
        scores,
        cls_arr,
        iou_threshold=DEFAULT_CROSS_TILE_NMS_IOU,
    )

    merge_ids = (
        None
        if merge_touching_same_class
        else DEFAULT_MERGE_TOUCHING_CLASS_IDS
    )
    xyxy_f, scores, cls_arr = merge_touching_same_class_xyxy(
        xyxy.astype(np.float32),
        scores,
        cls_arr,
        min_iou=merge_same_class_iou_threshold,
        merge_class_ids=merge_ids,
    )
    xyxy = np.round(xyxy_f).astype(np.int32)
    return xyxy, scores, cls_arr


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
    merge_class_ids: set[int] | frozenset[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge detections **per class** when pairwise IoU is **strictly greater** than ``min_iou``
    (intersection over union); groups are connected transitively. Each output row is the union
    bbox of its group with the max score in that group.

    When ``merge_class_ids`` is set, only those classes are merged; other classes pass through
    unchanged.
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
        if merge_class_ids is not None and c not in merge_class_ids:
            for i in range(n):
                merged_xy.append(sub_xy[i])
                merged_sc.append(float(sub_sc[i]))
                merged_cls.append(c)
            continue

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

    ``input`` and ``scrollbar`` are always merged when IoU exceeds ``merge_same_class_iou_threshold``
    (see :data:`DEFAULT_MERGE_TOUCHING_CLASS_IDS`).

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

    merge_ids = (
        None
        if merge_touching_same_class
        else DEFAULT_MERGE_TOUCHING_CLASS_IDS
    )
    xyxy, scores, cls = merge_touching_same_class_xyxy(
        xyxy,
        scores,
        cls,
        min_iou=merge_same_class_iou_threshold,
        merge_class_ids=merge_ids,
    )

    # Convert to integer so it plays nice with cv2.rectangle / image slicers
    xyxy = np.round(xyxy).astype(np.int32)
    
    return xyxy, scores, cls
