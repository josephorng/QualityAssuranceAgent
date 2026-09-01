"""Detect flat axis-aligned rectangles from horizontal Hough line pairs.

Used for input-box borders: keep near-horizontal segments, merge collinear
pieces, then pair parallels with high overlap and large width/height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class LineSegmentParams:
    blur_ksize: int = 5
    canny_low: int = 5
    canny_high: int = 35
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 5
    min_line_length: int = 15
    max_line_gap: int = 0
    min_width_over_height: float = 5.0
    min_overlap_frac: float = 0.9
    min_height: float = 10.0
    max_height: float = 60.0
    vertical_merge_gap: float = 60.0


@dataclass(frozen=True)
class HorizontalRectangleResult:
    """Pipeline stages from Hough segments to completed input-box rectangles."""

    raw_segments: list[tuple[int, int, int, int]]
    merged_segments: list[tuple[int, int, int, int]]
    candidate_segments: list[tuple[int, int, int, int]]
    rectangles: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1)
    vertical_merged_segments: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class _AxisSeg:
    x1: int
    y1: int
    x2: int
    y2: int
    orient: str  # "h" | "v"

    @property
    def xmin(self) -> int:
        return min(self.x1, self.x2)

    @property
    def xmax(self) -> int:
        return max(self.x1, self.x2)

    @property
    def ymin(self) -> int:
        return min(self.y1, self.y2)

    @property
    def ymax(self) -> int:
        return max(self.y1, self.y2)

    @property
    def mid_x(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def mid_y(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def length(self) -> float:
        return float(((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def _classify_axis_segment(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    angle_tol_deg: float = 12.0,
) -> _AxisSeg | None:
    """Keep nearly-horizontal / nearly-vertical segments; drop diagonals."""
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None
    angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
    if angle <= angle_tol_deg or angle >= 180.0 - angle_tol_deg:
        mid_y = int(round(0.5 * (y1 + y2)))
        xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
        return _AxisSeg(xa, mid_y, xb, mid_y, "h")
    if abs(angle - 90.0) <= angle_tol_deg:
        mid_x = int(round(0.5 * (x1 + x2)))
        ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
        return _AxisSeg(mid_x, ya, mid_x, yb, "v")
    return None


def _interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Gap between two 1D intervals; 0 if they touch or overlap."""
    if a1 < b0:
        return float(b0 - a1)
    if b1 < a0:
        return float(a0 - b1)
    return 0.0


def _can_merge_axis_segments(
    a: _AxisSeg,
    b: _AxisSeg,
    *,
    pos_tol: float,
    gap_tol: float,
) -> bool:
    if a.orient != b.orient:
        return False
    if a.orient == "h":
        if abs(a.mid_y - b.mid_y) > pos_tol:
            return False
        return _interval_gap(a.xmin, a.xmax, b.xmin, b.xmax) <= gap_tol
    if abs(a.mid_x - b.mid_x) > pos_tol:
        return False
    return _interval_gap(a.ymin, a.ymax, b.ymin, b.ymax) <= gap_tol


def _merge_two_axis_segments(a: _AxisSeg, b: _AxisSeg) -> _AxisSeg:
    if a.orient == "h":
        y = int(round(0.5 * (a.mid_y + b.mid_y)))
        return _AxisSeg(min(a.xmin, b.xmin), y, max(a.xmax, b.xmax), y, "h")
    x = int(round(0.5 * (a.mid_x + b.mid_x)))
    return _AxisSeg(x, min(a.ymin, b.ymin), x, max(a.ymax, b.ymax), "v")


def _merge_collinear_axis_segments(
    segments: list[_AxisSeg],
    *,
    pos_tol: float,
    gap_tol: float,
) -> list[_AxisSeg]:
    """Merge nearly overlapping / collinear H or V segments into longer ones."""
    merged: list[_AxisSeg] = []
    for orient in ("h", "v"):
        group = [s for s in segments if s.orient == orient]
        if not group:
            continue
        if orient == "h":
            group = sorted(group, key=lambda s: (s.mid_y, s.xmin, -s.length))
        else:
            group = sorted(group, key=lambda s: (s.mid_x, s.ymin, -s.length))
        changed = True
        while changed:
            changed = False
            nxt: list[_AxisSeg] = []
            used = [False] * len(group)
            for i, a in enumerate(group):
                if used[i]:
                    continue
                cur = a
                used[i] = True
                grew = True
                while grew:
                    grew = False
                    for j, b in enumerate(group):
                        if used[j]:
                            continue
                        if _can_merge_axis_segments(
                            cur, b, pos_tol=pos_tol, gap_tol=gap_tol
                        ):
                            cur = _merge_two_axis_segments(cur, b)
                            used[j] = True
                            grew = True
                            changed = True
                nxt.append(cur)
            group = nxt
        merged.extend(group)
    return merged


def _extract_merged_axis_segments(
    segments: list[tuple[int, int, int, int]],
    orient: str,
    *,
    pos_tol: float,
    gap_tol: float,
) -> list[tuple[int, int, int, int]]:
    """Classify raw Hough segments and merge collinear ones for ``orient`` (``h`` or ``v``)."""
    axis: list[_AxisSeg] = []
    for x1, y1, x2, y2 in segments:
        seg = _classify_axis_segment(int(x1), int(y1), int(x2), int(y2))
        if seg is not None and seg.orient == orient:
            axis.append(seg)
    if not axis:
        return []
    merged = _merge_collinear_axis_segments(axis, pos_tol=pos_tol, gap_tol=gap_tol)
    return [seg.as_tuple() for seg in merged]


def _detect_vertical_merged_segments(
    edges: np.ndarray,
    params: LineSegmentParams,
    *,
    pos_tol: float,
) -> list[tuple[int, int, int, int]]:
    """Detect and merge near-vertical segments, bridging table grid breaks."""
    merge_gap = max(
        float(params.vertical_merge_gap),
        float(max(8, int(params.max_line_gap) + 6)),
        10.0,
    )
    close_h = max(5, min(int(round(merge_gap / 2)), 80))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h))
    v_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    hough_gap = max(int(params.max_line_gap), int(round(merge_gap / 2)))
    lines = cv2.HoughLinesP(
        v_edges,
        rho=max(0.1, float(params.rho)),
        theta=np.deg2rad(max(0.1, float(params.theta_deg))),
        threshold=max(1, int(params.threshold)),
        minLineLength=max(0, int(params.min_line_length)),
        maxLineGap=max(0, hough_gap),
    )
    if lines is None:
        return []
    raw = [
        (int(x1), int(y1), int(x2), int(y2))
        for x1, y1, x2, y2 in lines[:, 0]
    ]
    return _extract_merged_axis_segments(
        raw, "v", pos_tol=pos_tol, gap_tol=merge_gap
    )


def _horizontal_overlap_passes(
    h1: _AxisSeg,
    h2: _AxisSeg,
    *,
    min_overlap_frac: float,
) -> bool:
    x_lo = max(h1.xmin, h2.xmin)
    x_hi = min(h1.xmax, h2.xmax)
    overlap = float(x_hi - x_lo)
    if overlap <= 0:
        return False
    return (
        overlap >= min_overlap_frac * h1.length
        and overlap >= min_overlap_frac * h2.length
    )


def _horizontal_pair_rectangle(
    h1: _AxisSeg,
    h2: _AxisSeg,
    *,
    min_width_over_height: float,
    min_height: float,
    max_height: float,
) -> tuple[int, int, int, int] | None:
    height = abs(h1.mid_y - h2.mid_y)
    if height <= min_height or height > max_height:
        return None
    x_lo = min(h1.xmin, h2.xmin)
    x_hi = max(h1.xmax, h2.xmax)
    width = float(x_hi - x_lo)
    if width <= 0:
        return None
    if width / height < min_width_over_height:
        return None
    y_lo = min(h1.mid_y, h2.mid_y)
    y_hi = max(h1.mid_y, h2.mid_y)
    x0 = int(round(x_lo))
    y0 = int(round(y_lo))
    x1 = int(round(x_hi))
    y1 = int(round(y_hi))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _nearest_overlap_neighbor(
    axis: list[_AxisSeg],
    idx: int,
    *,
    direction: str,
    min_overlap_frac: float,
) -> int | None:
    """Return closest line above (``direction='above'``) or below with enough overlap."""
    line = axis[idx]
    best: int | None = None
    best_dist = float("inf")
    for j, other in enumerate(axis):
        if j == idx:
            continue
        if direction == "above":
            if other.mid_y >= line.mid_y:
                continue
            dist = line.mid_y - other.mid_y
        else:
            if other.mid_y <= line.mid_y:
                continue
            dist = other.mid_y - line.mid_y
        if not _horizontal_overlap_passes(
            line, other, min_overlap_frac=min_overlap_frac
        ):
            continue
        if dist < best_dist:
            best_dist = dist
            best = j
    return best


def pair_horizontal_rectangles(
    segments: list[tuple[int, int, int, int]],
    *,
    pos_tol: float,
    min_width_over_height: float = 5.0,
    min_overlap_frac: float = 0.95,
    min_height: float = 10.0,
    max_height: float = 60.0,
    horizontal_scrollbar_boxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """Build rectangles from parallel horizontal pairs only (ignore verticals).

    Returns ``(merged, candidates, rectangles)`` where:
    - ``merged``: after near-collinear merge of horizontals
    - ``candidates``: horizontal sides of accepted pairs
    - ``rectangles``: axis-aligned boxes ``(x0, y0, x1, y1)`` for each pair

    Pairing strategy:
    1. For each horizontal, collect overlap-valid neighbors and keep the nearest
       line above and the nearest line below.
    2. Drop lines with no such neighbor still in play.
    3. Repeatedly pair any line that has exactly one remaining neighbor until no
       more forced pairs exist.

    Pairs shorter than ``min_height`` (default 10px) or taller than
    ``max_height`` (default 60px) are rejected.
    Rectangles overlapping a horizontal scrollbar are removed last.
    """
    axis: list[_AxisSeg] = []
    for x1, y1, x2, y2 in segments:
        seg = _classify_axis_segment(int(x1), int(y1), int(x2), int(y2))
        if seg is not None and seg.orient == "h":
            axis.append(seg)
    if not axis:
        return [], [], []

    gap_tol = max(pos_tol, 10.0)
    axis = _merge_collinear_axis_segments(axis, pos_tol=pos_tol, gap_tol=gap_tol)
    merged = [seg.as_tuple() for seg in axis]
    if len(axis) < 2:
        return merged, [], []

    axis = sorted(axis, key=lambda s: (s.mid_y, s.xmin))

    above: list[int | None] = [
        _nearest_overlap_neighbor(
            axis, i, direction="above", min_overlap_frac=min_overlap_frac
        )
        for i in range(len(axis))
    ]
    below: list[int | None] = [
        _nearest_overlap_neighbor(
            axis, i, direction="below", min_overlap_frac=min_overlap_frac
        )
        for i in range(len(axis))
    ]

    active = set(range(len(axis)))

    def _active_candidates(idx: int) -> set[int]:
        cands: set[int] = set()
        up = above[idx]
        down = below[idx]
        if up is not None and up in active:
            cands.add(up)
        if down is not None and down in active:
            cands.add(down)
        return cands

    def _prune_lines_without_candidates() -> bool:
        removed = False
        while True:
            to_remove = {i for i in active if not _active_candidates(i)}
            if not to_remove:
                return removed
            active.difference_update(to_remove)
            removed = True

    candidate_segments: set[_AxisSeg] = set()
    completed: set[tuple[int, int, int, int]] = set()

    _prune_lines_without_candidates()

    while True:
        singles = [i for i in active if len(_active_candidates(i)) == 1]
        if not singles:
            break
        progress = False
        for i in singles:
            neighbors = _active_candidates(i)
            if len(neighbors) != 1:
                continue
            j = next(iter(neighbors))
            if j not in active:
                continue
            rect = _horizontal_pair_rectangle(
                axis[i],
                axis[j],
                min_width_over_height=min_width_over_height,
                min_height=min_height,
                max_height=max_height,
            )
            if rect is None:
                active.discard(i)
                progress = True
                break
            candidate_segments.add(axis[i])
            candidate_segments.add(axis[j])
            completed.add(rect)
            active.difference_update((i, j))
            progress = True
            break
        if not progress:
            break
        _prune_lines_without_candidates()

    rectangles = _drop_rectangles_containing_others(sorted(completed))
    if horizontal_scrollbar_boxes:
        rectangles = _drop_rectangles_overlapping_horizontal_scrollbars(
            rectangles,
            horizontal_scrollbar_boxes,
        )
    return (
        merged,
        [seg.as_tuple() for seg in candidate_segments],
        rectangles,
    )


def _xyxy_to_xywh_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return x0, y0, x1 - x0, y1 - y0


def _drop_rectangles_overlapping_horizontal_scrollbars(
    rectangles: list[tuple[int, int, int, int]],
    scrollbar_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Remove rectangles that overlap any horizontal scrollbar ``(x, y, w, h)``."""
    from cua_mcp.geometry import boxes_overlap
    from cua_mcp.scrollbar_arrows import scrollbar_orientation

    horizontal = [
        sb
        for sb in scrollbar_boxes
        if scrollbar_orientation(sb) == "horizontal"
    ]
    if not horizontal or not rectangles:
        return list(rectangles)
    keep: list[tuple[int, int, int, int]] = []
    for rect in rectangles:
        rect_xywh = _xyxy_to_xywh_box(rect)
        if any(boxes_overlap(rect_xywh, sb) for sb in horizontal):
            continue
        keep.append(rect)
    return keep


def _drop_xywh_boxes_overlapping_horizontal_scrollbars(
    boxes: list[tuple[int, int, int, int]],
    scrollbar_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Remove ``(x, y, w, h)`` boxes that overlap any horizontal scrollbar."""
    from cua_mcp.geometry import boxes_overlap
    from cua_mcp.scrollbar_arrows import scrollbar_orientation

    horizontal = [
        sb
        for sb in scrollbar_boxes
        if scrollbar_orientation(sb) == "horizontal"
    ]
    if not horizontal or not boxes:
        return list(boxes)
    return [
        box
        for box in boxes
        if not any(boxes_overlap(box, sb) for sb in horizontal)
    ]


def _xyxy_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    """True if ``outer`` fully contains ``inner`` and is strictly larger."""
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    if not (ox0 <= ix0 and oy0 <= iy0 and ox1 >= ix1 and oy1 >= iy1):
        return False
    return (ox1 - ox0) * (oy1 - oy0) > (ix1 - ix0) * (iy1 - iy0)


def _drop_rectangles_containing_others(
    rectangles: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Remove any rectangle that fully contains another rectangle."""
    if len(rectangles) < 2:
        return list(rectangles)
    keep: list[tuple[int, int, int, int]] = []
    for i, rect in enumerate(rectangles):
        if any(
            j != i and _xyxy_contains(rect, other)
            for j, other in enumerate(rectangles)
        ):
            continue
        keep.append(rect)
    return keep


def _image_to_gray(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert PIL RGB/RGBA or OpenCV BGR/BGRA/gray array to a single-channel gray image."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise TypeError(f"unsupported image array shape: {arr.shape!r}")
    channels = arr.shape[2]
    if channels == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    if channels == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    raise TypeError(f"unsupported image channel count: {channels}")


def detect_horizontal_rectangles(
    image: Image.Image | np.ndarray,
    params: LineSegmentParams | None = None,
    *,
    horizontal_scrollbar_boxes: list[tuple[int, int, int, int]] | None = None,
) -> HorizontalRectangleResult:
    """Detect Hough segments and pair them into flat horizontal rectangles.

    ``image`` may be a PIL image (RGB) or a NumPy array (BGR / BGRA / gray),
    matching typical ``cua_mcp`` OpenCV usage.
    """
    p = params or LineSegmentParams()
    ksize = max(1, int(p.blur_ksize))
    if ksize % 2 == 0:
        ksize += 1

    gray = _image_to_gray(image)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    canny_low = int(p.canny_low)
    canny_high = int(p.canny_high)
    if canny_high < canny_low:
        canny_low, canny_high = canny_high, canny_low
    edges = cv2.Canny(blurred, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=max(0.1, float(p.rho)),
        theta=np.deg2rad(max(0.1, float(p.theta_deg))),
        threshold=max(1, int(p.threshold)),
        minLineLength=max(0, int(p.min_line_length)),
        maxLineGap=max(0, int(p.max_line_gap)),
    )
    if lines is None:
        return HorizontalRectangleResult([], [], [], [])
    raw = [
        (int(x1), int(y1), int(x2), int(y2))
        for x1, y1, x2, y2 in lines[:, 0]
    ]
    pos_tol = float(max(8, int(p.max_line_gap) + 6))
    vertical_merged = _detect_vertical_merged_segments(
        edges, p, pos_tol=pos_tol
    )
    merged, candidates, rectangles = pair_horizontal_rectangles(
        raw,
        pos_tol=pos_tol,
        min_width_over_height=float(p.min_width_over_height),
        min_overlap_frac=float(p.min_overlap_frac),
        min_height=float(p.min_height),
        max_height=float(p.max_height),
        horizontal_scrollbar_boxes=horizontal_scrollbar_boxes,
    )
    return HorizontalRectangleResult(
        raw, merged, candidates, rectangles, vertical_merged
    )


def extract_input_box_rectangles(
    image: Image.Image | np.ndarray,
    params: LineSegmentParams | None = None,
    **kwargs: Any,
) -> list[tuple[int, int, int, int]]:
    """Return only completed rectangle boxes ``(x0, y0, x1, y1)``."""
    return detect_horizontal_rectangles(image, params, **kwargs).rectangles


DEFAULT_INPUT_RECT_IOU_THRESHOLD: float = 0.3


def _xyxy_to_xywh(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    img_w: int | None = None,
    img_h: int | None = None,
) -> tuple[int, int, int, int] | None:
    """Convert ``(x0, y0, x1, y1)`` to clipped ``(x, y, w, h)``, or ``None`` if empty."""
    from cua_mcp.geometry import clip_box

    if x1 <= x0 or y1 <= y0:
        return None
    x, y, w, h = x0, y0, x1 - x0, y1 - y0
    if img_w is not None and img_h is not None:
        x, y, w, h = clip_box(x, y, w, h, img_w, img_h)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def merge_yolo_inputs_with_line_rectangles(
    image: Image.Image | np.ndarray,
    yolo_input_boxes: list[tuple[int, int, int, int]],
    *,
    iou_threshold: float = DEFAULT_INPUT_RECT_IOU_THRESHOLD,
    params: LineSegmentParams | None = None,
    img_w: int | None = None,
    img_h: int | None = None,
    horizontal_scrollbar_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Add Hough input-box rectangles and refine YOLO inputs on high IoU.

    - Line rectangles with no matching YOLO input (IoU ≤ threshold) are kept
      as new input boxes.
    - When IoU with a YOLO input exceeds ``iou_threshold``, the YOLO box is
      replaced by the line rectangle (each YOLO / rectangle used at most once).
    - Unmatched YOLO inputs are kept as-is.
    - Any box overlapping a horizontal scrollbar is dropped.

    Boxes are ``(x, y, w, h)``.
    """
    from cua_mcp.geometry import iou_xywh

    if img_w is None or img_h is None:
        arr = np.asarray(image)
        if arr.ndim >= 2:
            img_h = int(arr.shape[0])
            img_w = int(arr.shape[1])

    scrollbars = horizontal_scrollbar_boxes or []
    rect_boxes: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in extract_input_box_rectangles(
        image,
        params,
        horizontal_scrollbar_boxes=scrollbars,
    ):
        box = _xyxy_to_xywh(x0, y0, x1, y1, img_w=img_w, img_h=img_h)
        if box is not None:
            rect_boxes.append(box)

    if not rect_boxes:
        merged_inputs = list(yolo_input_boxes)
    elif not yolo_input_boxes:
        merged_inputs = rect_boxes
    else:
        used_yolo: set[int] = set()
        merged_inputs = []

        for rect in rect_boxes:
            best_i: int | None = None
            best_iou = 0.0
            for i, yolo in enumerate(yolo_input_boxes):
                if i in used_yolo:
                    continue
                iou = iou_xywh(rect, yolo)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
            if best_i is not None and best_iou > iou_threshold:
                merged_inputs.append(rect)
                used_yolo.add(best_i)
            else:
                merged_inputs.append(rect)

        for i, yolo in enumerate(yolo_input_boxes):
            if i not in used_yolo:
                merged_inputs.append(yolo)

    if scrollbars:
        merged_inputs = _drop_xywh_boxes_overlapping_horizontal_scrollbars(
            merged_inputs,
            scrollbars,
        )
    return merged_inputs
