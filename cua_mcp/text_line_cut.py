"""
Stage 4 line cut: split residual multi-line ``text`` boxes at row-profile valleys.

Stage 3 (:func:`cua_mcp.yolo_divide_conquer.refine_overlapping_text_with_crops`) can only
swap in boxes the detector re-found, so a tall box whose lines the model keeps merging
survives unchanged. This module adds the missing *split* capability: a deterministic,
model-free pass over the final detections.

Ink is measured as deviation from a **per-row** background estimate, so white-on-black and
black-on-white behave identically, and colored / selected rows and alternating table
striping stay neutral. Text rows become profile peaks; blank rows between lines become
valleys, and cuts land in the valleys. A full-width separator rule is uniform within its
own row, so its row background equals the rule color and it reads as a valley too.

Splits are all-or-nothing per box: when the resulting bands do not look like lines, the
original box is kept, so the worst case equals no Stage 4 at all. See
``OVERLAP_REFINE_LOGIC.md``.
"""

from __future__ import annotations

import cv2
import numpy as np

# Stage 4 on by default (mirrors ``OVERLAP_REFINE_DEFAULT``).
LINE_CUT_DEFAULT: bool = True
LINE_CUT_TEXT_CLASS_ID: int = 0
# Only boxes at least this tall vs the frame median text height are candidates.
# Same constant value as ``OVERLAP_REFINE_TALL_HEIGHT_FACTOR`` so Stage 3 and Stage 4
# agree on what "suspiciously multi-line" means.
LINE_CUT_TALL_HEIGHT_FACTOR: float = 1.85
LINE_CUT_MAX_BOXES: int = 64
LINE_CUT_MIN_BOX_H: int = 16
LINE_CUT_MIN_BOX_W: int = 8

# Ink test: |pixel - row_background| above this is ink. Scaled to the crop's own
# dynamic range so low-contrast themes still register.
LINE_CUT_MIN_DEV: float = 12.0
LINE_CUT_DEV_RANGE_FRAC: float = 0.22
# Rows smoothed over this many neighbors to pick the peak reference level. The gap test
# itself runs on the raw profile: smoothing it would widen every band by a row and swallow
# the 2-3px valleys of tightly-leaded lines.
LINE_CUT_SMOOTH_ROWS: int = 3
# A row is a gap when its ink fraction is at or below
# ``max(LINE_CUT_ABS_GAP, LINE_CUT_GAP_FRAC * p95(smoothed))``. The absolute floor is what
# absorbs single stray pixels in an otherwise blank row.
LINE_CUT_ABS_GAP: float = 0.02
LINE_CUT_GAP_FRAC: float = 0.18
# Interior gap runs shorter than ``max(2, frac * median_line_h)`` do not split a band —
# unless the run contains a fully blank row (see ``LINE_CUT_ABS_GAP``). Compact dropdowns
# separate their lines by a single blank row, so a 2-row floor alone never splits them.
LINE_CUT_MIN_GAP_FRAC: float = 0.10
# Bands are validated against each other, not against the frame median: a compact list can
# have a much smaller line pitch than the median text box of the frame.
# Bands shorter than this fraction of the box's own median band height are slivers (e.g.
# the edge of a highlighted row) and are dropped rather than failing the whole box.
LINE_CUT_SLIVER_H_FRAC: float = 0.5
# Kept bands must be uniform: tallest no more than this × the box's median band height.
LINE_CUT_BAND_UNIFORM_MAX_FRAC: float = 1.8
# No real text line is thinner than this, which keeps glyph-internal blank rows (二, 三)
# from being read as line separators.
LINE_CUT_MIN_BAND_H_PX: int = 5
# The frame median is still an upper bound, so an emitted band is never itself multi-line.
LINE_CUT_BAND_MAX_H_FRAC: float = 1.65
# Bands must retain at least this fraction of the box's total ink.
LINE_CUT_MIN_COVERAGE_FRAC: float = 0.6


def _empty_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
    )


def _ink_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Boolean ink mask for ``crop_bgr``, using a per-row background estimate.

    Polarity-free by construction: ink is any pixel far enough from its own row's
    median. Rows where the "ink" ends up in the majority are inverted so ink always
    means the minority class within that row (text never fills a whole row of a text
    box, but a row median can land on the glyph color in text-dense crops).
    """
    if crop_bgr.size == 0:
        return np.zeros((0, 0), dtype=bool)
    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = crop_bgr.astype(np.float32)

    row_bg = np.median(gray, axis=1, keepdims=True)
    dev = np.abs(gray - row_bg)
    lo, hi = (float(v) for v in np.percentile(gray, (2.0, 98.0)))
    thr = max(LINE_CUT_MIN_DEV, LINE_CUT_DEV_RANGE_FRAC * (hi - lo))

    mask = dev > thr
    if mask.size:
        flip = mask.mean(axis=1) > 0.5
        if bool(np.any(flip)):
            mask[flip] = ~mask[flip]
    return mask


def _row_ink_profile(mask: np.ndarray) -> np.ndarray:
    """Fraction of ink pixels per row."""
    if mask.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return mask.mean(axis=1).astype(np.float32)


def _smooth_profile(profile: np.ndarray, window: int = LINE_CUT_SMOOTH_ROWS) -> np.ndarray:
    if window <= 1 or profile.size <= window:
        return profile
    pad = window // 2
    padded = np.pad(profile, pad, mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)[: len(profile)]


def _gap_row_mask(profile: np.ndarray) -> np.ndarray:
    """
    True for rows that look like blank space between lines.

    The reference peak level comes from a smoothed copy (so one spiky row cannot raise
    the bar), but the decision is per raw row so band edges stay tight.
    """
    if profile.size == 0:
        return np.zeros((0,), dtype=bool)
    p95 = float(np.percentile(_smooth_profile(profile), 95.0))
    thr = max(LINE_CUT_ABS_GAP, LINE_CUT_GAP_FRAC * p95)
    return profile <= thr


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as ``(start, end_exclusive)`` index pairs."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(edges[i]), int(edges[i + 1]))
        for i in range(0, len(edges) - 1, 2)
    ]


def _content_bands(profile: np.ndarray, *, min_gap_rows: int) -> list[tuple[int, int]]:
    """
    Row spans of contiguous content, split at every qualifying valley.

    A valley qualifies when it spans at least ``min_gap_rows`` rows **or** contains a fully
    blank row (``profile <= LINE_CUT_ABS_GAP``). The blank-row rule is what handles compact
    lists: a 12px-pitch dropdown separates its items by exactly one empty row, which the
    row-count floor alone would merge away. Rows that merely dip below the soft gap
    threshold still need the full ``min_gap_rows`` span, so a thin spot inside a glyph row
    cannot split a line.

    Ink runs are the bands, so leading / trailing gaps are dropped (each band is already
    trimmed to its own ink extent) and cutting between two bands is equivalent to cutting
    at the center of the valley that separates them.
    """
    gap = _gap_row_mask(profile)
    blank = profile <= LINE_CUT_ABS_GAP
    ink_runs = _true_runs(~gap)
    if not ink_runs:
        return []

    bands: list[tuple[int, int]] = [ink_runs[0]]
    for start, end in ink_runs[1:]:
        prev_start, prev_end = bands[-1]
        valley_rows = start - prev_end
        has_blank_row = bool(blank[prev_end:start].any())
        if valley_rows < int(min_gap_rows) and not has_blank_row:
            bands[-1] = (prev_start, end)
        else:
            bands.append((start, end))
    return bands


def _drop_sliver_bands(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Drop bands far thinner than the box's own median band.

    Highlighted / selected rows leave a 1px run at the edge of their background block.
    Dropping those is better than letting one sliver fail an otherwise clean split.
    """
    if len(bands) < 2:
        return bands
    heights = np.asarray([y1 - y0 for y0, y1 in bands], dtype=np.float32)
    median_h = float(np.median(heights))
    if median_h <= 0.0:
        return bands
    floor = LINE_CUT_SLIVER_H_FRAC * median_h
    return [b for b, h in zip(bands, heights) if float(h) >= floor]


def _band_x_extent(mask: np.ndarray, y0: int, y1: int) -> tuple[int, int] | None:
    """Ink column extent inside rows ``[y0, y1)``; ``None`` when the band has no ink."""
    if mask.size == 0 or y1 <= y0:
        return None
    cols = np.flatnonzero(mask[y0:y1].any(axis=0))
    if cols.size == 0:
        return None
    return int(cols[0]), int(cols[-1]) + 1


def _bands_look_like_lines(
    bands: list[tuple[int, int]],
    profile: np.ndarray,
    *,
    median_line_h: float,
) -> bool:
    """
    All-or-nothing validation: reject anything that does not look like stacked lines.

    Uniformity is judged **within the box** — stacked lines of one control share a pitch,
    whatever that pitch is. Comparing band heights against the frame median instead would
    reject every compact list whose font is smaller than the frame's median text box.
    ``median_line_h`` is used only as an upper bound, so a band can never itself be
    multi-line.
    """
    if len(bands) < 2:
        return False

    heights = np.asarray([y1 - y0 for y0, y1 in bands], dtype=np.float32)
    median_h = float(np.median(heights))
    if median_h < float(LINE_CUT_MIN_BAND_H_PX):
        return False
    if float(heights.max()) > LINE_CUT_BAND_UNIFORM_MAX_FRAC * median_h:
        return False
    if float(heights.min()) < LINE_CUT_SLIVER_H_FRAC * median_h:
        return False
    if float(heights.max()) > LINE_CUT_BAND_MAX_H_FRAC * median_line_h:
        return False

    total_ink = float(profile.sum())
    if total_ink > 0.0:
        kept_ink = float(sum(profile[y0:y1].sum() for y0, y1 in bands))
        if kept_ink < LINE_CUT_MIN_COVERAGE_FRAC * total_ink:
            return False
    return True


def cut_text_box_into_lines(
    crop_bgr: np.ndarray,
    *,
    median_line_h: float,
) -> list[tuple[int, int, int, int]] | None:
    """
    Split one text-box crop into per-line boxes in **crop-local** ``xyxy`` pixels.

    Returns ``None`` when the crop should be left alone (no valley, or the bands do not
    validate as lines).
    """
    if crop_bgr.size == 0:
        return None

    mask = _ink_mask(crop_bgr)
    profile = _row_ink_profile(mask)
    if profile.size == 0:
        return None

    min_gap_rows = max(2, int(round(LINE_CUT_MIN_GAP_FRAC * max(1.0, median_line_h))))
    bands = _drop_sliver_bands(_content_bands(profile, min_gap_rows=min_gap_rows))
    if not _bands_look_like_lines(
        bands,
        profile,
        median_line_h=max(1.0, median_line_h),
    ):
        return None

    out: list[tuple[int, int, int, int]] = []
    for y0, y1 in bands:
        x_extent = _band_x_extent(mask, y0, y1)
        if x_extent is None:
            return None
        x0, x1 = x_extent
        out.append((x0, y0, x1, y1))
    if len(out) < 2:
        return None
    return out


def cut_multiline_text_boxes(
    bgr: np.ndarray,
    xyxy: np.ndarray,
    scores: np.ndarray,
    cls: np.ndarray,
    *,
    text_class_id: int = LINE_CUT_TEXT_CLASS_ID,
    tall_height_factor: float = LINE_CUT_TALL_HEIGHT_FACTOR,
    max_boxes: int = LINE_CUT_MAX_BOXES,
    enhance_roi: tuple[int, int, int, int] | None = None,
    enhance_roi_pad: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Split tall ``text`` boxes at row-profile valleys.

    Only ``text`` boxes at least ``tall_height_factor`` × the frame's median text height
    are attempted; every other detection passes through untouched and in order. Emitted
    bands are disjoint in Y with no vertical padding, so the downstream per-class merge
    (``cua_mcp.geometry.merge_same_line_boxes``) cannot read two bands as one line and
    fuse the lines we just split.

    When ``enhance_roi`` is set, only boxes whose center lies in the padded ROI are
    considered for cutting; others pass through unchanged.

    Returns ``(xyxy, scores, cls, num_boxes_cut)``.
    """
    xyxy = np.asarray(xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    cls = np.asarray(cls).reshape(-1).astype(np.int32, copy=False)
    if xyxy.ndim != 2:
        xyxy = xyxy.reshape(-1, 4) if xyxy.size else np.zeros((0, 4), dtype=np.float32)
    n = min(len(xyxy), len(scores), len(cls))
    if n == 0 or bgr.size == 0:
        empty_xyxy, empty_scores, empty_cls = _empty_arrays()
        return empty_xyxy, empty_scores, empty_cls, 0
    xyxy, scores, cls = xyxy[:n], scores[:n], cls[:n]

    img_h, img_w = bgr.shape[:2]
    text_heights = [
        float(xyxy[i, 3]) - float(xyxy[i, 1])
        for i in range(n)
        if int(cls[i]) == int(text_class_id)
    ]
    if not text_heights:
        return xyxy, scores, cls, 0
    median_line_h = max(1.0, float(np.median(np.asarray(text_heights, dtype=np.float32))))
    min_tall_h = tall_height_factor * median_line_h

    out_xyxy: list[np.ndarray] = []
    out_scores: list[float] = []
    out_cls: list[int] = []
    attempted = 0
    cut_count = 0

    for i in range(n):
        box = xyxy[i]
        keep_box = True
        if (
            int(cls[i]) == int(text_class_id)
            and attempted < int(max_boxes)
            and (float(box[3]) - float(box[1])) >= min_tall_h
            and (
                enhance_roi is None
                or _xyxy_center_in_enhance_roi(box, enhance_roi, pad=enhance_roi_pad)
            )
        ):
            x0 = int(max(0, np.floor(float(box[0]))))
            y0 = int(max(0, np.floor(float(box[1]))))
            x1 = int(min(img_w, np.ceil(float(box[2]))))
            y1 = int(min(img_h, np.ceil(float(box[3]))))
            if (
                (x1 - x0) >= LINE_CUT_MIN_BOX_W
                and (y1 - y0) >= LINE_CUT_MIN_BOX_H
            ):
                attempted += 1
                lines = cut_text_box_into_lines(
                    bgr[y0:y1, x0:x1],
                    median_line_h=median_line_h,
                )
                if lines:
                    for lx0, ly0, lx1, ly1 in lines:
                        out_xyxy.append(
                            np.asarray(
                                [x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1],
                                dtype=np.float32,
                            )
                        )
                        out_scores.append(float(scores[i]))
                        out_cls.append(int(cls[i]))
                    cut_count += 1
                    keep_box = False

        if keep_box:
            out_xyxy.append(box.astype(np.float32, copy=False))
            out_scores.append(float(scores[i]))
            out_cls.append(int(cls[i]))

    if cut_count == 0:
        return xyxy, scores, cls, 0

    return (
        np.stack(out_xyxy, axis=0).astype(np.float32, copy=False),
        np.asarray(out_scores, dtype=np.float32),
        np.asarray(out_cls, dtype=np.int32),
        cut_count,
    )


def _xyxy_center_in_enhance_roi(
    box: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> bool:
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    rx, ry, rw, rh = (int(v) for v in rect)
    if rw <= 0 or rh <= 0:
        return False
    p = max(0, int(pad))
    return (rx - p) <= cx < (rx + rw + p) and (ry - p) <= cy < (ry + rh + p)
