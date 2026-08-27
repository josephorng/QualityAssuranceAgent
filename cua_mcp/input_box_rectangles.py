"""Detect flat axis-aligned rectangles from horizontal Hough line pairs.

Used for input-box borders: keep near-horizontal segments, merge collinear
pieces, then pair parallels with high overlap and large width/height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class LineSegmentParams:
    blur_ksize: int = 5
    canny_low: int = 50
    canny_high: int = 150
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 100
    min_line_length: int = 20
    max_line_gap: int = 0


@dataclass(frozen=True)
class HorizontalRectangleResult:
    """Pipeline stages from Hough segments to completed input-box rectangles."""

    raw_segments: list[tuple[int, int, int, int]]
    merged_segments: list[tuple[int, int, int, int]]
    candidate_segments: list[tuple[int, int, int, int]]
    rectangles: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1)


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


def pair_horizontal_rectangles(
    segments: list[tuple[int, int, int, int]],
    *,
    pos_tol: float,
    min_width_over_height: float = 5.0,
    min_overlap_frac: float = 0.95,
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

    candidates: set[_AxisSeg] = set()
    completed: set[tuple[int, int, int, int]] = set()

    for i, h1 in enumerate(axis):
        for h2 in axis[i + 1 :]:
            height = abs(h1.mid_y - h2.mid_y)
            if height < 1.0:
                continue
            x_lo = max(h1.xmin, h2.xmin)
            x_hi = min(h1.xmax, h2.xmax)
            overlap = float(x_hi - x_lo)
            if overlap <= 0:
                continue
            if (
                overlap < min_overlap_frac * h1.length
                or overlap < min_overlap_frac * h2.length
            ):
                continue
            width = overlap
            if width / height < min_width_over_height:
                continue
            y_lo = min(h1.mid_y, h2.mid_y)
            y_hi = max(h1.mid_y, h2.mid_y)
            x0 = int(round(x_lo))
            y0 = int(round(y_lo))
            x1 = int(round(x_hi))
            y1 = int(round(y_hi))
            if x1 <= x0 or y1 <= y0:
                continue
            candidates.add(h1)
            candidates.add(h2)
            completed.add((x0, y0, x1, y1))

    return (
        merged,
        [seg.as_tuple() for seg in candidates],
        sorted(completed),
    )


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
    min_width_over_height: float = 5.0,
    min_overlap_frac: float = 0.95,
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
    merged, candidates, rectangles = pair_horizontal_rectangles(
        raw,
        pos_tol=pos_tol,
        min_width_over_height=min_width_over_height,
        min_overlap_frac=min_overlap_frac,
    )
    return HorizontalRectangleResult(raw, merged, candidates, rectangles)


def extract_input_box_rectangles(
    image: Image.Image | np.ndarray,
    params: LineSegmentParams | None = None,
    **kwargs: Any,
) -> list[tuple[int, int, int, int]]:
    """Return only completed rectangle boxes ``(x0, y0, x1, y1)``."""
    return detect_horizontal_rectangles(image, params, **kwargs).rectangles
