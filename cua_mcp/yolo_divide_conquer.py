"""
Divide-and-conquer YOLO inference: if a full-frame pass hits ``max_det``,
split the image into overlapping 2×2 tiles, predict each, merge with NMS.

Also optional overlap refine: when text boxes overlap / cross / look multi-line,
crop each hard cluster and re-run YOLO at higher effective resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

# Ultralytics default ``max_det``; end2end ONNX exports often share this cap.
YOLO_MAX_DET_DEFAULT: int = 300
TILE_OVERLAP_FRAC_DEFAULT: float = 0.15
TILE_MERGE_IOU_DEFAULT: float = 0.20

# Overlap / dense-text refine (crop hard clusters and re-predict).
OVERLAP_REFINE_DEFAULT: bool = True
OVERLAP_REFINE_TEXT_CLASS_ID: int = 0
# Link two text boxes into a refine cluster when pairwise IoU exceeds this.
OVERLAP_REFINE_LINK_IOU: float = 0.08
# Also link when boxes cross (intersect without full containment).
OVERLAP_REFINE_LINK_CROSSING: bool = True
# Dense vertical stack linking (optional). Off by default: gap-only links pull in
# already-clean neighbors and make crops too large / merge-prone. IoU/crossing
# still catch mixed boxes.
OVERLAP_REFINE_X_OVERLAP_FRAC: float = 0.45
OVERLAP_REFINE_MAX_V_GAP_FRAC: float = -1.0
# Single box taller than this × median text height is treated as a cluster alone.
OVERLAP_REFINE_TALL_HEIGHT_FACTOR: float = 1.85
# Pad crop around cluster union. Text lines need extra X pad so short/fragment
# unions do not starve re-detection (narrow crops miss most dropdown lines).
OVERLAP_REFINE_PAD_FRAC: float = 0.15
OVERLAP_REFINE_PAD_LINE_FRAC: float = 0.5
OVERLAP_REFINE_MIN_PAD_X_PX: float = 36.0
OVERLAP_REFINE_MIN_PAD_Y_PX: float = 8.0
OVERLAP_REFINE_MIN_CROP_SIDE: int = 24
OVERLAP_REFINE_MAX_CROPS: int = 24
# When a refine crop is taller than this × median line height, predict overlapping
# horizontal strips instead of one shot (avoids re-merging dense dropdowns).
OVERLAP_REFINE_STRIP_HEIGHT_FACTOR: float = 2.6
OVERLAP_REFINE_STRIP_OVERLAP_FRAC: float = 0.45
OVERLAP_REFINE_OVERLAP_PAIR_IOU: float = 0.08
OVERLAP_REFINE_UNIQUE_LINE_IOU: float = 0.35


@dataclass(frozen=True)
class DetectArrays:
    """Detections in source-image pixel coordinates (xyxy)."""

    xyxy: np.ndarray  # (N, 4) float
    scores: np.ndarray  # (N,) float
    cls: np.ndarray  # (N,) int / float
    used_tiles: bool
    full_count: int
    used_overlap_refine: bool = False
    refine_crop_count: int = 0


def empty_detect_arrays(
    *,
    used_tiles: bool = False,
    full_count: int = 0,
    used_overlap_refine: bool = False,
    refine_crop_count: int = 0,
) -> DetectArrays:
    return DetectArrays(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        cls=np.zeros((0,), dtype=np.int32),
        used_tiles=used_tiles,
        full_count=full_count,
        used_overlap_refine=used_overlap_refine,
        refine_crop_count=refine_crop_count,
    )


def load_bgr(source: str | Path | np.ndarray) -> np.ndarray:
    """Return a BGR uint8 image from a path or existing ndarray."""
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] < 3:
            raise ValueError(f"Expected HxWxC image, got shape {source.shape}")
        return source
    path = Path(source)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def split_bgr_2x2(
    bgr: np.ndarray,
    *,
    overlap_frac: float = TILE_OVERLAP_FRAC_DEFAULT,
) -> list[tuple[np.ndarray, int, int]]:
    """
    Split ``bgr`` into four overlapping tiles: TL, TR, BL, BR.

    ``overlap_frac`` is relative to half-width / half-height (e.g. 0.15 → each
    tile extends 15% past the center cut into the neighboring half).

    Returns ``(tile_bgr, origin_x, origin_y)`` in full-image coordinates.
    """
    h, w = bgr.shape[:2]
    if h < 2 or w < 2:
        return [(bgr, 0, 0)]

    overlap_frac = max(0.0, float(overlap_frac))
    half_w = w // 2
    half_h = h // 2
    ow = int(round(half_w * overlap_frac))
    oh = int(round(half_h * overlap_frac))

    regions = (
        (0, 0, min(w, half_w + ow), min(h, half_h + oh)),
        (max(0, half_w - ow), 0, w, min(h, half_h + oh)),
        (0, max(0, half_h - oh), min(w, half_w + ow), h),
        (max(0, half_w - ow), max(0, half_h - oh), w, h),
    )
    out: list[tuple[np.ndarray, int, int]] = []
    for x0, y0, x1, y1 in regions:
        if x1 <= x0 or y1 <= y0:
            continue
        out.append((bgr[y0:y1, x0:x1], int(x0), int(y0)))
    return out or [(bgr, 0, 0)]


def ultralytics_results_to_arrays(results: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten Ultralytics ``predict`` results into ``(xyxy, scores, cls)``."""
    xyxy_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    cls_parts: list[np.ndarray] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls = boxes.cls
        if xyxy is None or conf is None or cls is None:
            continue
        arr_xyxy = xyxy.cpu().numpy().astype(np.float32, copy=False)
        if arr_xyxy.size == 0:
            continue
        xyxy_parts.append(arr_xyxy)
        score_parts.append(conf.cpu().numpy().astype(np.float32, copy=False))
        cls_parts.append(cls.cpu().numpy().astype(np.float32, copy=False))
    if not xyxy_parts:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.concatenate(xyxy_parts, axis=0),
        np.concatenate(score_parts, axis=0),
        np.concatenate(cls_parts, axis=0).astype(np.int32, copy=False),
    )


def offset_xyxy(xyxy: np.ndarray, ox: float, oy: float) -> np.ndarray:
    if xyxy.size == 0:
        return xyxy
    out = xyxy.astype(np.float32, copy=True)
    out[:, [0, 2]] += float(ox)
    out[:, [1, 3]] += float(oy)
    return out


def merge_detections_nms(
    xyxy: np.ndarray,
    scores: np.ndarray,
    cls: np.ndarray,
    *,
    iou_threshold: float = TILE_MERGE_IOU_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Class-aware NMS: suppress lower-score boxes of the same class when IoU exceeds
    ``iou_threshold``. Used to dedupe detections from overlapping tiles.
    """
    if xyxy.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    xyxy = np.asarray(xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    cls = np.asarray(cls).reshape(-1)
    n = min(len(xyxy), len(scores), len(cls))
    xyxy, scores, cls = xyxy[:n], scores[:n], cls[:n]

    keep_indices: list[int] = []
    for class_id in sorted(set(int(c) for c in cls)):
        idxs = [i for i in range(n) if int(cls[i]) == class_id]
        if not idxs:
            continue
        boxes_wh = []
        for i in idxs:
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            boxes_wh.append([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)])
        nms_idx = cv2.dnn.NMSBoxes(
            boxes_wh,
            scores[idxs].tolist(),
            score_threshold=0.0,
            nms_threshold=float(iou_threshold),
        )
        if nms_idx is None or len(nms_idx) == 0:
            continue
        flat = np.array(nms_idx).reshape(-1)
        keep_indices.extend(int(idxs[int(j)]) for j in flat)

    keep_indices.sort(key=lambda i: float(scores[i]), reverse=True)
    if not keep_indices:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    keep = np.asarray(keep_indices, dtype=np.int64)
    return xyxy[keep], scores[keep], cls[keep].astype(np.int32, copy=False)


def _concat_parts(
    parts_xyxy: list[np.ndarray],
    parts_scores: list[np.ndarray],
    parts_cls: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not parts_xyxy:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.concatenate(parts_xyxy, axis=0),
        np.concatenate(parts_scores, axis=0),
        np.concatenate(parts_cls, axis=0).astype(np.int32, copy=False),
    )


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _boxes_cross_xyxy(a: np.ndarray, b: np.ndarray) -> bool:
    """True when boxes intersect but neither fully contains the other."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    a_in_b = ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2
    b_in_a = bx1 >= ax1 and by1 >= ay1 and bx2 <= ax2 and by2 <= ay2
    return not (a_in_b or b_in_a)


def _x_overlap_frac(a: np.ndarray, b: np.ndarray) -> float:
    ax1, _ay1, ax2, _ay2 = (float(v) for v in a)
    bx1, _by1, bx2, _by2 = (float(v) for v in b)
    aw = max(1e-6, ax2 - ax1)
    bw = max(1e-6, bx2 - bx1)
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    inter = max(0.0, ix2 - ix1)
    return float(inter / min(aw, bw))


def _vertical_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Signed gap between vertical extents; negative means Y-overlap."""
    _ax1, ay1, _ax2, ay2 = (float(v) for v in a)
    _bx1, by1, _bx2, by2 = (float(v) for v in b)
    if ay2 <= by1:
        return float(by1 - ay2)
    if by2 <= ay1:
        return float(ay1 - by2)
    return float(-(min(ay2, by2) - max(ay1, by1)))


def _box_height(box: np.ndarray) -> float:
    return max(0.0, float(box[3]) - float(box[1]))


def _union_find_clusters(n: int, links: Sequence[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in links:
        if 0 <= i < n and 0 <= j < n:
            union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _text_boxes_should_link(
    a: np.ndarray,
    b: np.ndarray,
    *,
    link_iou: float,
    link_crossing: bool,
    x_overlap_frac: float,
    max_v_gap_frac: float,
) -> bool:
    if _iou_xyxy(a, b) > link_iou:
        return True
    if link_crossing and _boxes_cross_xyxy(a, b):
        return True
    if max_v_gap_frac < 0.0:
        return False
    ha = _box_height(a)
    hb = _box_height(b)
    min_h = max(1.0, min(ha, hb))
    if _x_overlap_frac(a, b) >= x_overlap_frac and _vertical_gap(a, b) <= max_v_gap_frac * min_h:
        return True
    return False


def _drop_tall_text_boxes(
    xyxy: np.ndarray,
    scores: np.ndarray,
    cls: np.ndarray,
    *,
    text_class_id: int,
    max_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xyxy.size == 0:
        return xyxy, scores, cls
    keep: list[int] = []
    for i in range(len(xyxy)):
        if int(cls[i]) == int(text_class_id) and _box_height(xyxy[i]) > max_height:
            continue
        keep.append(i)
    if len(keep) == len(xyxy):
        return xyxy, scores, cls
    if not keep:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    idx = np.asarray(keep, dtype=np.int64)
    return xyxy[idx], scores[idx], cls[idx].astype(np.int32, copy=False)


def _predict_roi_with_optional_strips(
    bgr: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    predict_crop_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    median_h: float,
    strip_height_factor: float,
    strip_overlap_frac: float,
    merge_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict on ``bgr[y0:y1, x0:x1]``. If the ROI is tall vs median line height,
    split into overlapping horizontal strips, predict each, and NMS-merge.
    """
    crop_h = max(1, y1 - y0)
    strip_h = max(48, int(round(strip_height_factor * max(1.0, median_h))))
    if crop_h <= int(strip_h * 1.35):
        c_xyxy, c_scores, c_cls = predict_crop_fn(bgr[y0:y1, x0:x1])
        c_xyxy = np.asarray(c_xyxy, dtype=np.float32)
        c_scores = np.asarray(c_scores, dtype=np.float32).reshape(-1)
        c_cls = np.asarray(c_cls).reshape(-1).astype(np.int32, copy=False)
        if c_xyxy.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        if c_xyxy.ndim != 2:
            c_xyxy = c_xyxy.reshape(-1, 4)
        n = min(len(c_xyxy), len(c_scores), len(c_cls))
        return offset_xyxy(c_xyxy[:n], x0, y0), c_scores[:n], c_cls[:n]

    overlap = max(8, int(round(strip_overlap_frac * strip_h)))
    parts_xyxy: list[np.ndarray] = []
    parts_scores: list[np.ndarray] = []
    parts_cls: list[np.ndarray] = []
    y = y0
    while y < y1:
        ys = int(y)
        ye = min(y1, ys + strip_h)
        if ye <= ys:
            break
        s_xyxy, s_scores, s_cls = predict_crop_fn(bgr[ys:ye, x0:x1])
        s_xyxy = np.asarray(s_xyxy, dtype=np.float32)
        s_scores = np.asarray(s_scores, dtype=np.float32).reshape(-1)
        s_cls = np.asarray(s_cls).reshape(-1).astype(np.int32, copy=False)
        if s_xyxy.size:
            if s_xyxy.ndim != 2:
                s_xyxy = s_xyxy.reshape(-1, 4)
            n = min(len(s_xyxy), len(s_scores), len(s_cls))
            parts_xyxy.append(offset_xyxy(s_xyxy[:n], x0, ys))
            parts_scores.append(s_scores[:n])
            parts_cls.append(s_cls[:n])
        if ye >= y1:
            break
        y = ye - overlap
        if y <= ys:
            y = ys + 1

    merged_xyxy, merged_scores, merged_cls = _concat_parts(
        parts_xyxy, parts_scores, parts_cls
    )
    return merge_detections_nms(
        merged_xyxy,
        merged_scores,
        merged_cls,
        iou_threshold=merge_iou,
    )


def _cluster_union_xyxy(xyxy: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    boxes = xyxy[list(indices)]
    return np.asarray(
        [
            float(boxes[:, 0].min()),
            float(boxes[:, 1].min()),
            float(boxes[:, 2].max()),
            float(boxes[:, 3].max()),
        ],
        dtype=np.float32,
    )


def _padded_crop_xyxy(
    union: np.ndarray,
    *,
    img_w: int,
    img_h: int,
    pad_frac: float,
    pad_x_px: float,
    pad_y_px: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in union)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    pad_x = max(float(pad_x_px), w * pad_frac)
    pad_y = max(float(pad_y_px), h * pad_frac)
    ix1 = int(max(0, np.floor(x1 - pad_x)))
    iy1 = int(max(0, np.floor(y1 - pad_y)))
    ix2 = int(min(img_w, np.ceil(x2 + pad_x)))
    iy2 = int(min(img_h, np.ceil(y2 + pad_y)))
    if ix2 <= ix1:
        ix2 = min(img_w, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(img_h, iy1 + 1)
    return ix1, iy1, ix2, iy2


def _filter_crop_dets_to_roi(
    xyxy: np.ndarray,
    scores: np.ndarray,
    cls: np.ndarray,
    *,
    roi_xyxy: np.ndarray,
    text_class_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep detections whose center lies inside the (slightly inset) cluster ROI."""
    if xyxy.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    rx1, ry1, rx2, ry2 = (float(v) for v in roi_xyxy)
    rw = max(1.0, rx2 - rx1)
    rh = max(1.0, ry2 - ry1)
    # Small inset so edge noise from neighboring UI is less likely to stick.
    inset_x = 0.02 * rw
    inset_y = 0.02 * rh
    keep: list[int] = []
    for i in range(len(xyxy)):
        if int(cls[i]) != int(text_class_id):
            continue
        cx = 0.5 * (float(xyxy[i, 0]) + float(xyxy[i, 2]))
        cy = 0.5 * (float(xyxy[i, 1]) + float(xyxy[i, 3]))
        if (rx1 + inset_x) <= cx <= (rx2 - inset_x) and (ry1 + inset_y) <= cy <= (ry2 - inset_y):
            keep.append(i)
    if not keep:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    idx = np.asarray(keep, dtype=np.int64)
    return xyxy[idx], scores[idx], cls[idx].astype(np.int32, copy=False)


def _overlap_pair_count(xyxy: np.ndarray, *, iou_thr: float = OVERLAP_REFINE_OVERLAP_PAIR_IOU) -> int:
    n = len(xyxy)
    if n < 2:
        return 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _iou_xyxy(xyxy[i], xyxy[j]) > iou_thr:
                count += 1
    return count


def _approx_unique_box_count(
    xyxy: np.ndarray,
    *,
    iou_thr: float = OVERLAP_REFINE_UNIQUE_LINE_IOU,
) -> int:
    """NMS with equal scores to estimate how many distinct line boxes are present."""
    if xyxy.size == 0:
        return 0
    n = len(xyxy)
    scores = np.ones((n,), dtype=np.float32)
    cls = np.zeros((n,), dtype=np.int32)
    kept, _, _ = merge_detections_nms(xyxy, scores, cls, iou_threshold=iou_thr)
    return int(len(kept))


def _cluster_candidate_key(xyxy: np.ndarray) -> tuple[int, int, int]:
    """Sort key for cluster replacements: fewer overlaps, then more unique lines."""
    if xyxy.size == 0:
        return (10**9, 10**9, 10**9)
    ov = _overlap_pair_count(xyxy)
    uniq = _approx_unique_box_count(xyxy)
    # Minimize overlaps, then maximize unique lines, then prefer fewer raw dups.
    return (ov, -uniq, len(xyxy) - uniq)


def _pick_best_cluster_boxes(
    old_xyxy: np.ndarray,
    old_scores: np.ndarray,
    old_cls: np.ndarray,
    crop_xyxy: np.ndarray,
    crop_scores: np.ndarray,
    crop_cls: np.ndarray,
    *,
    dedupe_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Choose among original / NMS-deduped / crop-refined cluster boxes.
    Returns ``None`` when the original is best (no change).
    """
    old_uniq = _approx_unique_box_count(old_xyxy)
    dedup_xyxy, dedup_scores, dedup_cls = merge_detections_nms(
        old_xyxy,
        old_scores,
        old_cls,
        iou_threshold=dedupe_iou,
    )
    candidates: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("old", old_xyxy, old_scores, old_cls),
        ("dedup", dedup_xyxy, dedup_scores, dedup_cls),
    ]
    # Crop must preserve unique line coverage; otherwise prefer dedupe.
    if crop_xyxy.size and _approx_unique_box_count(crop_xyxy) >= old_uniq:
        candidates.append(("crop", crop_xyxy, crop_scores, crop_cls))

    best_name, best_xyxy, best_scores, best_cls = min(
        candidates,
        key=lambda item: _cluster_candidate_key(item[1]),
    )
    if best_name == "old":
        return None
    if _cluster_candidate_key(best_xyxy) >= _cluster_candidate_key(old_xyxy):
        return None
    return best_xyxy, best_scores, best_cls


def find_text_overlap_refine_clusters(
    xyxy: np.ndarray,
    cls: np.ndarray,
    *,
    text_class_id: int = OVERLAP_REFINE_TEXT_CLASS_ID,
    link_iou: float = OVERLAP_REFINE_LINK_IOU,
    link_crossing: bool = OVERLAP_REFINE_LINK_CROSSING,
    x_overlap_frac: float = OVERLAP_REFINE_X_OVERLAP_FRAC,
    max_v_gap_frac: float = OVERLAP_REFINE_MAX_V_GAP_FRAC,
    tall_height_factor: float = OVERLAP_REFINE_TALL_HEIGHT_FACTOR,
) -> list[list[int]]:
    """
    Return clusters of global detection indices that should be crop-refined.

    Includes overlapping / crossing / densely stacked text boxes, and singleton
    text boxes that are unusually tall vs the median text height.
    """
    xyxy = np.asarray(xyxy, dtype=np.float32)
    cls = np.asarray(cls).reshape(-1)
    n = min(len(xyxy), len(cls))
    if n == 0:
        return []
    xyxy, cls = xyxy[:n], cls[:n]
    text_idxs = [i for i in range(n) if int(cls[i]) == int(text_class_id)]
    if not text_idxs:
        return []

    heights = np.asarray([_box_height(xyxy[i]) for i in text_idxs], dtype=np.float32)
    median_h = float(np.median(heights)) if len(heights) else 0.0
    median_h = max(1.0, median_h)

    local_n = len(text_idxs)
    links: list[tuple[int, int]] = []
    for a in range(local_n):
        ia = text_idxs[a]
        for b in range(a + 1, local_n):
            ib = text_idxs[b]
            if _text_boxes_should_link(
                xyxy[ia],
                xyxy[ib],
                link_iou=link_iou,
                link_crossing=link_crossing,
                x_overlap_frac=x_overlap_frac,
                max_v_gap_frac=max_v_gap_frac,
            ):
                links.append((a, b))

    local_clusters = _union_find_clusters(local_n, links)
    out: list[list[int]] = []
    seen_singletons: set[int] = set()
    for cluster in local_clusters:
        global_ids = [text_idxs[i] for i in cluster]
        if len(global_ids) >= 2:
            out.append(sorted(global_ids))
            continue
        # Singleton: only refine if suspiciously tall (likely multi-line merge).
        gi = global_ids[0]
        if _box_height(xyxy[gi]) >= tall_height_factor * median_h:
            out.append([gi])
            seen_singletons.add(gi)

    # Tall boxes that never linked still need a cluster entry.
    for i in text_idxs:
        if i in seen_singletons:
            continue
        if any(i in c for c in out):
            continue
        if _box_height(xyxy[i]) >= tall_height_factor * median_h:
            out.append([i])

    # Prefer larger clusters first; cap applied by caller.
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


def refine_overlapping_text_with_crops(
    bgr: np.ndarray,
    xyxy: np.ndarray,
    scores: np.ndarray,
    cls: np.ndarray,
    *,
    predict_crop_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    text_class_id: int = OVERLAP_REFINE_TEXT_CLASS_ID,
    link_iou: float = OVERLAP_REFINE_LINK_IOU,
    link_crossing: bool = OVERLAP_REFINE_LINK_CROSSING,
    x_overlap_frac: float = OVERLAP_REFINE_X_OVERLAP_FRAC,
    max_v_gap_frac: float = OVERLAP_REFINE_MAX_V_GAP_FRAC,
    tall_height_factor: float = OVERLAP_REFINE_TALL_HEIGHT_FACTOR,
    pad_frac: float = OVERLAP_REFINE_PAD_FRAC,
    pad_line_frac: float = OVERLAP_REFINE_PAD_LINE_FRAC,
    min_pad_x_px: float = OVERLAP_REFINE_MIN_PAD_X_PX,
    min_pad_y_px: float = OVERLAP_REFINE_MIN_PAD_Y_PX,
    min_crop_side: int = OVERLAP_REFINE_MIN_CROP_SIDE,
    max_crops: int = OVERLAP_REFINE_MAX_CROPS,
    strip_height_factor: float = OVERLAP_REFINE_STRIP_HEIGHT_FACTOR,
    strip_overlap_frac: float = OVERLAP_REFINE_STRIP_OVERLAP_FRAC,
    merge_iou: float = TILE_MERGE_IOU_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Crop hard text clusters and re-run ``predict_crop_fn``; replace a cluster when
    the crop yields clearer / more line boxes.

    ``predict_crop_fn(crop_bgr) -> (xyxy, scores, cls)`` in crop-local pixels.
    Returns ``(xyxy, scores, cls, num_crops_accepted)``.
    """
    xyxy = np.asarray(xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    cls = np.asarray(cls).reshape(-1).astype(np.int32, copy=False)
    n = min(len(xyxy), len(scores), len(cls))
    if n == 0 or bgr.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            0,
        )
    xyxy, scores, cls = xyxy[:n].copy(), scores[:n].copy(), cls[:n].copy()
    img_h, img_w = bgr.shape[:2]

    clusters = find_text_overlap_refine_clusters(
        xyxy,
        cls,
        text_class_id=text_class_id,
        link_iou=link_iou,
        link_crossing=link_crossing,
        x_overlap_frac=x_overlap_frac,
        max_v_gap_frac=max_v_gap_frac,
        tall_height_factor=tall_height_factor,
    )
    if not clusters:
        return xyxy, scores, cls, 0

    text_heights = [
        _box_height(xyxy[i]) for i in range(n) if int(cls[i]) == int(text_class_id)
    ]
    text_widths = [
        max(0.0, float(xyxy[i, 2]) - float(xyxy[i, 0]))
        for i in range(n)
        if int(cls[i]) == int(text_class_id)
    ]
    median_h = float(np.median(np.asarray(text_heights, dtype=np.float32))) if text_heights else 12.0
    median_w = float(np.median(np.asarray(text_widths, dtype=np.float32))) if text_widths else 40.0
    pad_y_px = max(float(min_pad_y_px), pad_line_frac * median_h)
    pad_x_px = max(float(min_pad_x_px), 0.75 * median_w)
    max_line_h = tall_height_factor * max(1.0, median_h)

    replaced: set[int] = set()
    new_parts_xyxy: list[np.ndarray] = []
    new_parts_scores: list[np.ndarray] = []
    new_parts_cls: list[np.ndarray] = []
    accepted = 0

    for cluster in clusters:
        if accepted >= int(max_crops):
            break
        live = [i for i in cluster if i not in replaced]
        if not live:
            continue
        # Need a multi-box conflict or a single tall box.
        if len(live) == 1 and _box_height(xyxy[live[0]]) < max_line_h:
            continue

        union = _cluster_union_xyxy(xyxy, live)
        x0, y0, x1, y1 = _padded_crop_xyxy(
            union,
            img_w=img_w,
            img_h=img_h,
            pad_frac=pad_frac,
            pad_x_px=pad_x_px,
            pad_y_px=pad_y_px,
        )
        if (x1 - x0) < int(min_crop_side) or (y1 - y0) < int(min_crop_side):
            continue

        c_xyxy, c_scores, c_cls = _predict_roi_with_optional_strips(
            bgr,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            predict_crop_fn=predict_crop_fn,
            median_h=median_h,
            strip_height_factor=strip_height_factor,
            strip_overlap_frac=strip_overlap_frac,
            merge_iou=merge_iou,
        )
        crop_roi = np.asarray([x0, y0, x1, y1], dtype=np.float32)
        if c_xyxy.size:
            # Prefer centers near the original cluster union (not the full padded crop),
            # so neighboring labels pulled in by X-pad do not replace the cluster.
            ux1, uy1, ux2, uy2 = (float(v) for v in union)
            uw = max(1.0, ux2 - ux1)
            uh = max(1.0, uy2 - uy1)
            soft_union = np.asarray(
                [
                    max(float(x0), ux1 - 0.15 * uw),
                    max(float(y0), uy1 - 0.15 * uh),
                    min(float(x1), ux2 + 0.15 * uw),
                    min(float(y1), uy2 + 0.15 * uh),
                ],
                dtype=np.float32,
            )
            c_xyxy, c_scores, c_cls = _filter_crop_dets_to_roi(
                c_xyxy,
                c_scores,
                c_cls,
                roi_xyxy=soft_union,
                text_class_id=text_class_id,
            )
            # Drop multi-line merges from crop output; keep single-line candidates only.
            c_xyxy, c_scores, c_cls = _drop_tall_text_boxes(
                c_xyxy,
                c_scores,
                c_cls,
                text_class_id=text_class_id,
                max_height=max_line_h,
            )
        else:
            c_xyxy = np.zeros((0, 4), dtype=np.float32)
            c_scores = np.zeros((0,), dtype=np.float32)
            c_cls = np.zeros((0,), dtype=np.int32)

        old_xyxy = xyxy[live]
        old_scores = scores[live]
        old_cls = cls[live]
        picked = _pick_best_cluster_boxes(
            old_xyxy,
            old_scores,
            old_cls,
            c_xyxy,
            c_scores,
            c_cls,
            dedupe_iou=max(merge_iou, OVERLAP_REFINE_UNIQUE_LINE_IOU),
        )
        if picked is None:
            # Singleton tall box: accept crop only when it clearly splits into lines.
            if (
                len(live) == 1
                and len(c_xyxy) >= 2
                and _approx_unique_box_count(c_xyxy) >= 2
                and _overlap_pair_count(c_xyxy) == 0
            ):
                repl_xyxy, repl_scores, repl_cls = c_xyxy, c_scores, c_cls
            else:
                continue
        else:
            repl_xyxy, repl_scores, repl_cls = picked

        for i in live:
            replaced.add(i)
        new_parts_xyxy.append(repl_xyxy)
        new_parts_scores.append(repl_scores)
        new_parts_cls.append(repl_cls)
        accepted += 1

    if accepted == 0:
        return xyxy, scores, cls, 0

    keep = [i for i in range(n) if i not in replaced]
    parts_xyxy = ([xyxy[keep]] if keep else []) + new_parts_xyxy
    parts_scores = ([scores[keep]] if keep else []) + new_parts_scores
    parts_cls = ([cls[keep]] if keep else []) + new_parts_cls
    merged_xyxy, merged_scores, merged_cls = _concat_parts(
        parts_xyxy, parts_scores, parts_cls
    )
    merged_xyxy, merged_scores, merged_cls = merge_detections_nms(
        merged_xyxy,
        merged_scores,
        merged_cls,
        iou_threshold=merge_iou,
    )
    return merged_xyxy, merged_scores, merged_cls, accepted


def predict_ultralytics_with_2x2_fallback(
    model: Any,
    source: str | Path | np.ndarray,
    *,
    conf: float,
    max_det: int = YOLO_MAX_DET_DEFAULT,
    overlap_frac: float = TILE_OVERLAP_FRAC_DEFAULT,
    merge_iou: float = TILE_MERGE_IOU_DEFAULT,
    overlap_refine: bool = OVERLAP_REFINE_DEFAULT,
    text_class_id: int = OVERLAP_REFINE_TEXT_CLASS_ID,
    **predict_kwargs: Any,
) -> DetectArrays:
    """
    Run Ultralytics ``model.predict`` on the full image; if the count reaches
    ``max_det``, re-run on overlapping 2×2 tiles and merge with NMS.

    When ``overlap_refine`` is True, crop overlapping / tall text clusters and
    re-predict each crop to split mixed line boxes.
    """
    bgr = load_bgr(source)
    predict_kwargs = dict(predict_kwargs)
    predict_kwargs.setdefault("verbose", False)
    predict_kwargs.setdefault("save", False)

    results = model.predict(
        source=bgr,
        conf=conf,
        max_det=max_det,
        **predict_kwargs,
    )
    xyxy, scores, cls = ultralytics_results_to_arrays(results)
    full_count = int(len(xyxy))
    used_tiles = False
    if full_count >= int(max_det):
        parts_xyxy: list[np.ndarray] = []
        parts_scores: list[np.ndarray] = []
        parts_cls: list[np.ndarray] = []
        for tile, ox, oy in split_bgr_2x2(bgr, overlap_frac=overlap_frac):
            tile_results = model.predict(
                source=tile,
                conf=conf,
                max_det=max_det,
                **predict_kwargs,
            )
            t_xyxy, t_scores, t_cls = ultralytics_results_to_arrays(tile_results)
            if t_xyxy.size == 0:
                continue
            parts_xyxy.append(offset_xyxy(t_xyxy, ox, oy))
            parts_scores.append(t_scores)
            parts_cls.append(t_cls)

        xyxy, scores, cls = _concat_parts(parts_xyxy, parts_scores, parts_cls)
        xyxy, scores, cls = merge_detections_nms(
            xyxy,
            scores,
            cls,
            iou_threshold=merge_iou,
        )
        used_tiles = True

    refine_crop_count = 0
    used_overlap_refine = False
    if overlap_refine:

        def _predict_crop(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            crop_results = model.predict(
                source=crop_bgr,
                conf=conf,
                max_det=max_det,
                **predict_kwargs,
            )
            return ultralytics_results_to_arrays(crop_results)

        xyxy, scores, cls, refine_crop_count = refine_overlapping_text_with_crops(
            bgr,
            xyxy,
            scores,
            cls,
            predict_crop_fn=_predict_crop,
            text_class_id=text_class_id,
            merge_iou=merge_iou,
        )
        used_overlap_refine = refine_crop_count > 0

    return DetectArrays(
        xyxy=xyxy,
        scores=scores,
        cls=cls,
        used_tiles=used_tiles,
        full_count=full_count,
        used_overlap_refine=used_overlap_refine,
        refine_crop_count=refine_crop_count,
    )


def predict_onnx_with_2x2_fallback(
    bgr: np.ndarray,
    *,
    predict_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_det: int = YOLO_MAX_DET_DEFAULT,
    overlap_frac: float = TILE_OVERLAP_FRAC_DEFAULT,
    merge_iou: float = TILE_MERGE_IOU_DEFAULT,
    overlap_refine: bool = OVERLAP_REFINE_DEFAULT,
    text_class_id: int = OVERLAP_REFINE_TEXT_CLASS_ID,
    **predict_kwargs: Any,
) -> DetectArrays:
    """
    Run ``predict_fn(bgr, **kwargs) -> (xyxy, scores, cls)``; if the count reaches
    ``max_det``, re-run on overlapping 2×2 tiles and merge with NMS.

    When ``overlap_refine`` is True, crop overlapping / tall text clusters and
    re-predict each crop to split mixed line boxes.

    ``predict_fn`` is typically a *leaf* detector (e.g. Ultralytics or a Triton
    decode without its own tiling). Production ONNX already runs recursive
    quadtree + Stage 3 inside :func:`yolo_onnx.run_yolo_onnx_end2end`; do not
    wrap that entry point here or you will nest flat 2×2 / refine on top of it.
    If you must wrap it, pass ``overlap_refine=False`` on this helper and on
    ``predict_fn`` (and prefer a ``max_det`` high enough that Stage 2 never runs).
    """
    xyxy, scores, cls = predict_fn(bgr, **predict_kwargs)
    xyxy = np.asarray(xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    cls = np.asarray(cls).reshape(-1).astype(np.int32, copy=False)
    if xyxy.ndim != 2:
        xyxy = xyxy.reshape(-1, 4) if xyxy.size else np.zeros((0, 4), dtype=np.float32)
    full_count = int(min(len(xyxy), len(scores), len(cls)))
    xyxy, scores, cls = xyxy[:full_count], scores[:full_count], cls[:full_count]

    used_tiles = False
    if full_count >= int(max_det):
        parts_xyxy: list[np.ndarray] = []
        parts_scores: list[np.ndarray] = []
        parts_cls: list[np.ndarray] = []
        for tile, ox, oy in split_bgr_2x2(bgr, overlap_frac=overlap_frac):
            t_xyxy, t_scores, t_cls = predict_fn(tile, **predict_kwargs)
            t_xyxy = np.asarray(t_xyxy, dtype=np.float32)
            t_scores = np.asarray(t_scores, dtype=np.float32).reshape(-1)
            t_cls = np.asarray(t_cls).reshape(-1).astype(np.int32, copy=False)
            if t_xyxy.size == 0:
                continue
            if t_xyxy.ndim != 2:
                t_xyxy = t_xyxy.reshape(-1, 4)
            n = min(len(t_xyxy), len(t_scores), len(t_cls))
            parts_xyxy.append(offset_xyxy(t_xyxy[:n], ox, oy))
            parts_scores.append(t_scores[:n])
            parts_cls.append(t_cls[:n])

        xyxy, scores, cls = _concat_parts(parts_xyxy, parts_scores, parts_cls)
        xyxy, scores, cls = merge_detections_nms(
            xyxy,
            scores,
            cls,
            iou_threshold=merge_iou,
        )
        used_tiles = True

    refine_crop_count = 0
    used_overlap_refine = False
    if overlap_refine:

        def _predict_crop(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return predict_fn(crop_bgr, **predict_kwargs)

        xyxy, scores, cls, refine_crop_count = refine_overlapping_text_with_crops(
            bgr,
            xyxy,
            scores,
            cls,
            predict_crop_fn=_predict_crop,
            text_class_id=text_class_id,
            merge_iou=merge_iou,
        )
        used_overlap_refine = refine_crop_count > 0

    return DetectArrays(
        xyxy=xyxy,
        scores=scores,
        cls=cls,
        used_tiles=used_tiles,
        full_count=full_count,
        used_overlap_refine=used_overlap_refine,
        refine_crop_count=refine_crop_count,
    )
