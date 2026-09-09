"""Fit scrollbar bboxes to nearby end-cap arrow icons after YOLO+OCR."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from cua_mcp.geometry import boxes_overlap, iou_xywh, merge_two_boxes
from cua_mcp.icon_map import (
    is_pua_char,
    is_unknown_icon_record,
    load_icon_map,
    unknown_icon_record,
)
from cua_mcp.select_ui_element import UiDetection
from cua_mcp.yolo_onnx import (
    DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
    DEFAULT_SMALL_TEXT_AS_ELEMENT_MAX_SIDE,
    PICKER_CLASS_UNKNOWN,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
)

# Scrollbar arrow button icons (OCR / icon_map chinese_id). After YOLO+OCR, each
# scrollbar bbox is extended/shrunk along its main axis to include the two end
# buttons nearest the track ends. Matched ends are then unified to the
# canonical ``*滾動箭頭`` labels below.
_SCROLL_ARROW_UP_ID = "向上滾動箭頭"
_SCROLL_ARROW_DOWN_ID = "向下滾動箭頭"
_SCROLL_ARROW_LEFT_ID = "向左滾動箭頭"
_SCROLL_ARROW_RIGHT_ID = "向右滾動箭頭"
_SCROLL_ARROW_UP_IDS: frozenset[str] = frozenset(
    {"向上三角", "向上V箭頭", _SCROLL_ARROW_UP_ID}
)
_SCROLL_ARROW_DOWN_IDS: frozenset[str] = frozenset(
    {"向下三角", "向下V箭頭", _SCROLL_ARROW_DOWN_ID}
)
_SCROLL_ARROW_LEFT_IDS: frozenset[str] = frozenset(
    {"向左三角", "向左V箭頭", _SCROLL_ARROW_LEFT_ID}
)
_SCROLL_ARROW_RIGHT_IDS: frozenset[str] = frozenset(
    {"向右三角", "向右V箭頭", _SCROLL_ARROW_RIGHT_ID}
)
_SCROLL_ARROW_VERTICAL_IDS: frozenset[str] = (
    _SCROLL_ARROW_UP_IDS | _SCROLL_ARROW_DOWN_IDS
)
_SCROLL_ARROW_HORIZONTAL_IDS: frozenset[str] = (
    _SCROLL_ARROW_LEFT_IDS | _SCROLL_ARROW_RIGHT_IDS
)
_SCROLL_ARROW_ALL_IDS: frozenset[str] = (
    _SCROLL_ARROW_VERTICAL_IDS | _SCROLL_ARROW_HORIZONTAL_IDS
)
# Same-family pairs used to *create* scrollbars when YOLO missed the track.
# Unified ``*滾動箭頭`` labels are intentionally excluded (those come from fit).
_PAIR_FAMILY_V_ARROW_VERTICAL: tuple[str, str, bool] = (
    "向上V箭頭",
    "向下V箭頭",
    True,
)
_PAIR_FAMILY_TRIANGLE_VERTICAL: tuple[str, str, bool] = (
    "向上三角",
    "向下三角",
    True,
)
_PAIR_FAMILY_V_ARROW_HORIZONTAL: tuple[str, str, bool] = (
    "向左V箭頭",
    "向右V箭頭",
    False,
)
_PAIR_FAMILY_TRIANGLE_HORIZONTAL: tuple[str, str, bool] = (
    "向左三角",
    "向右三角",
    False,
)
_SCROLLBAR_PAIR_FAMILIES: tuple[tuple[str, str, bool], ...] = (
    _PAIR_FAMILY_V_ARROW_VERTICAL,
    _PAIR_FAMILY_TRIANGLE_VERTICAL,
    _PAIR_FAMILY_V_ARROW_HORIZONTAL,
    _PAIR_FAMILY_TRIANGLE_HORIZONTAL,
)
_UNKNOWN_ICON_CHINESE_ID: str = str(
    unknown_icon_record().get("chinese_id", "未知圖示")
).strip()
# OCR often misreads end-arrow glyphs as tiny text overlapping the arrow box.


def _detection_icon_chinese_ids(det: UiDetection) -> set[str]:
    """Return non-empty ``chinese_id`` values from ``det.icons``."""
    return {
        str(icon.get("chinese_id", "")).strip()
        for icon in (det.icons or [])
        if str(icon.get("chinese_id", "")).strip()
    }


def _is_unknown_icon_detection(det: UiDetection) -> bool:
    """True for unknown-icon / ambiguous OCR boxes usable as scrollbar end caps."""
    if det.class_id == PICKER_CLASS_UNKNOWN or det.class_name == "unknown":
        return True
    if _UNKNOWN_ICON_CHINESE_ID and _UNKNOWN_ICON_CHINESE_ID in _detection_icon_chinese_ids(
        det
    ):
        return True
    return any(
        is_unknown_icon_record(icon)
        for icon in (det.icons or [])
        if isinstance(icon, dict)
    )


def _is_arrow_pool_detection(det: UiDetection, arrow_ids: frozenset[str]) -> bool:
    """True when ``det`` is a known directional arrow or an unknown icon."""
    return bool(_detection_icon_chinese_ids(det) & arrow_ids) or _is_unknown_icon_detection(
        det
    )


def _pua_from_detection(det: UiDetection) -> str:
    """Best-effort PUA codepoint preserved from icons or OCR text."""
    for icon in det.icons or []:
        if not isinstance(icon, dict):
            continue
        pua = icon.get("pua")
        if isinstance(pua, str) and pua:
            return pua
    for ch in det.text or "":
        if is_pua_char(ch):
            return ch
    return ""


@lru_cache(maxsize=16)
def _icon_description_for_chinese_id(chinese_id: str) -> str:
    """Look up an icon_map description for ``chinese_id``, if present."""
    for value in load_icon_map().values():
        if not isinstance(value, dict):
            continue
        if str(value.get("chinese_id", "")).strip() == chinese_id:
            return str(value.get("description", "")).strip()
    return ""


def _arrow_icon_record(chinese_id: str, *, pua: str = "") -> dict[str, Any]:
    """Build an icon metadata dict for a reclassified scrollbar end arrow."""
    return {
        "pua": pua,
        "chinese_id": chinese_id,
        "icon_description": _icon_description_for_chinese_id(chinese_id),
    }


def _is_vertical_scrollbar_bbox(bbox: tuple[int, int, int, int]) -> bool:
    """True when height is at least width (vertical track)."""
    return int(bbox[3]) >= int(bbox[2])


def _arrow_cross_axis_aligned(
    scrollbar_bbox: tuple[int, int, int, int],
    arrow: UiDetection,
    *,
    vertical: bool,
) -> bool:
    """True when ``arrow`` center lies within the scrollbar's cross-axis span.

    Vertical tracks require ``arrow.cx`` in ``[sx, sx+sw]``; horizontal tracks
    require ``arrow.cy`` in ``[sy, sy+sh]``.
    """
    sx, sy, sw, sh = scrollbar_bbox
    if vertical:
        return sx <= arrow.cx <= sx + sw
    return sy <= arrow.cy <= sy + sh


def _scrollbar_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return the center pixel of a ``(x, y, w, h)`` bbox."""
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def _arrow_on_end_side_of_center(
    arrow: UiDetection,
    center_xy: tuple[int, int],
    end: str,
) -> bool:
    """True when ``arrow`` lies on the correct side of the scrollbar center for ``end``.

    ``top`` / ``bottom`` compare ``cy``; ``left`` / ``right`` compare ``cx``.
    """
    cx, cy = center_xy
    if end == "top":
        return arrow.cy < cy
    if end == "bottom":
        return arrow.cy > cy
    if end == "left":
        return arrow.cx < cx
    if end == "right":
        return arrow.cx > cx
    raise ValueError(f"unknown scrollbar end: {end!r}")


def _pick_scrollbar_end_arrow(
    arrows: list[UiDetection],
    *,
    end: str,
    preferred_ids: frozenset[str],
    any_ids: frozenset[str],
    scrollbar_bbox: tuple[int, int, int, int],
    vertical: bool,
    allow_unknown: bool = True,
) -> UiDetection | None:
    """Pick a track-aligned end arrow on the correct side of the scrollbar center.

    Priority: ``preferred_ids`` (expected direction), then unknown icons (when
    ``allow_unknown``), then ``any_ids``. Within a priority tier, choose the
    arrow closest to the original scrollbar center. ``end`` must be ``top`` /
    ``bottom`` / ``left`` / ``right`` and restricts candidates to that side of
    the center (e.g. top requires ``cy < center_y``).
    """
    center_xy = _scrollbar_center(scrollbar_bbox)
    ccx, ccy = center_xy

    def _candidates(
        *,
        ids: frozenset[str] | None = None,
        unknown_only: bool = False,
    ) -> list[UiDetection]:
        out: list[UiDetection] = []
        for arrow in arrows:
            if unknown_only:
                if not _is_unknown_icon_detection(arrow):
                    continue
            elif ids is not None:
                if not (_detection_icon_chinese_ids(arrow) & ids):
                    continue
            else:
                continue
            if not _arrow_cross_axis_aligned(
                scrollbar_bbox, arrow, vertical=vertical
            ):
                continue
            if not _arrow_on_end_side_of_center(arrow, center_xy, end):
                continue
            out.append(arrow)
        return out

    pool = _candidates(ids=preferred_ids)
    if not pool and allow_unknown:
        pool = _candidates(unknown_only=True)
    if not pool:
        pool = _candidates(ids=any_ids)
    if not pool:
        return None
    return min(pool, key=lambda a: (a.cx - ccx) ** 2 + (a.cy - ccy) ** 2)


def _fit_bbox_to_arrows_1d(
    scrollbar_bbox: tuple[int, int, int, int],
    arrow_a: UiDetection,
    arrow_b: UiDetection,
    *,
    vertical: bool,
) -> tuple[int, int, int, int]:
    """Extend/shrink ``scrollbar_bbox`` on one axis so both arrows are inside."""
    sx, sy, sw, sh = scrollbar_bbox
    boxes = (arrow_a.bbox, arrow_b.bbox)
    if vertical:
        top = min(b[1] for b in boxes)
        bottom = max(b[1] + b[3] for b in boxes)
        return sx, top, sw, max(1, bottom - top)
    left = min(b[0] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    return left, sy, max(1, right - left), sh


def _rebuild_detection(
    bbox: tuple[int, int, int, int],
    *,
    class_id: int,
    text: str | None,
    icons: list[dict[str, Any]] | None,
) -> UiDetection:
    """Rebuild a ``UiDetection`` with an updated bbox (and matching center)."""
    x, y, w, h = bbox
    return UiDetection(
        bbox=bbox,
        cx=x + w // 2,
        cy=y + h // 2,
        class_id=class_id,
        class_name=YOLO_CLASS_NAMES.get(class_id, str(class_id)),
        text=text,
        icons=icons if icons else None,
    )


def _unify_end_arrow_label(
    detections: list[UiDetection],
    picked: UiDetection,
    chinese_id: str,
) -> bool:
    """
    Rewrite ``picked`` in ``detections`` to the canonical scrollbar arrow label.

    Unknown icons are promoted to ``element``. Returns True when updated.
    """
    try:
        index = next(i for i, det in enumerate(detections) if det is picked)
    except StopIteration:
        return False
    current_ids = _detection_icon_chinese_ids(picked)
    already_unified = current_ids == {chinese_id} and not _is_unknown_icon_detection(
        picked
    )
    if already_unified and picked.class_id == YOLO_CLASS_ELEMENT:
        return False
    detections[index] = _rebuild_detection(
        picked.bbox,
        class_id=YOLO_CLASS_ELEMENT,
        text=picked.text,
        icons=[_arrow_icon_record(chinese_id, pua=_pua_from_detection(picked))],
    )
    return True


def fit_scrollbar_bboxes_to_arrow_controls(
    detections: list[UiDetection],
    *,
    log_info: Callable[[str], None] | None = None,
) -> list[UiDetection]:
    """
    Extend/shrink each scrollbar bbox to include its two end arrow buttons.

    Vertical scrollbars match ``向上三角`` / ``向下三角`` / ``向上V箭頭`` /
    ``向下V箭頭`` (and the unified ``*滾動箭頭`` labels). Horizontal
    scrollbars use the left/right triangle and V-arrow ids. Unknown icons
    (``未知圖示`` / class ``unknown``) are preferred over other-direction
    arrows when the expected end icon is missing. Each matched end arrow is
    then labeled ``向上滾動箭頭`` / ``向下滾動箭頭`` / ``向左滾動箭頭`` /
    ``向右滾動箭頭``. Matching arrows may be any distance along the track —
    the scrollbar extends to them. When either end lacks a matching
    track-aligned arrow, that scrollbar is left unchanged. When the fitted
    bbox's **open track** (between the end caps) would overlap text or an
    input with meaningful cross-axis penetration, the fit is skipped (bbox and
    end-arrow labels unchanged). Thin parallel grazes (footer text hugging the
    track), tiny text boxes (both sides under
    :data:`~cua_mcp.yolo_onnx.DEFAULT_SMALL_TEXT_AS_ELEMENT_MAX_SIDE`), and
    overlap confined to an end-arrow box are allowed. After a successful fit,
    the bbox grows by :data:`_FIT_SCROLLBAR_MAIN_EXPAND_PX` past each end cap
    (top/bottom or left/right). Overlap with another scrollbar is allowed
    (e.g. V+H corner meetings).
    """
    if not detections:
        return detections

    scrollbars = [
        i
        for i, det in enumerate(detections)
        if det.class_id == YOLO_CLASS_SCROLLBAR or det.class_name == "scrollbar"
    ]
    if not scrollbars:
        return detections

    out = list(detections)
    adjusted = 0
    unified = 0
    skipped_overlap = 0
    for idx in scrollbars:
        # Rebuild pools from ``out`` so prior end-arrow unifications apply.
        vertical_arrows = [
            det
            for det in out
            if _is_arrow_pool_detection(det, _SCROLL_ARROW_VERTICAL_IDS)
        ]
        horizontal_arrows = [
            det
            for det in out
            if _is_arrow_pool_detection(det, _SCROLL_ARROW_HORIZONTAL_IDS)
        ]
        if not vertical_arrows and not horizontal_arrows:
            continue

        sb = out[idx]
        vertical = _is_vertical_scrollbar_bbox(sb.bbox)
        if vertical:
            top = _pick_scrollbar_end_arrow(
                vertical_arrows,
                end="top",
                preferred_ids=_SCROLL_ARROW_UP_IDS,
                any_ids=_SCROLL_ARROW_VERTICAL_IDS,
                scrollbar_bbox=sb.bbox,
                vertical=True,
            )
            bottom = _pick_scrollbar_end_arrow(
                vertical_arrows,
                end="bottom",
                preferred_ids=_SCROLL_ARROW_DOWN_IDS,
                any_ids=_SCROLL_ARROW_VERTICAL_IDS,
                scrollbar_bbox=sb.bbox,
                vertical=True,
            )
            if top is None or bottom is None or top is bottom:
                continue
            new_bbox = _fit_bbox_to_arrows_1d(
                sb.bbox, top, bottom, vertical=True
            )
            start_arrow, end_arrow = top, bottom
            start_label, end_label = _SCROLL_ARROW_UP_ID, _SCROLL_ARROW_DOWN_ID
        else:
            left = _pick_scrollbar_end_arrow(
                horizontal_arrows,
                end="left",
                preferred_ids=_SCROLL_ARROW_LEFT_IDS,
                any_ids=_SCROLL_ARROW_HORIZONTAL_IDS,
                scrollbar_bbox=sb.bbox,
                vertical=False,
            )
            right = _pick_scrollbar_end_arrow(
                horizontal_arrows,
                end="right",
                preferred_ids=_SCROLL_ARROW_RIGHT_IDS,
                any_ids=_SCROLL_ARROW_HORIZONTAL_IDS,
                scrollbar_bbox=sb.bbox,
                vertical=False,
            )
            if left is None or right is None or left is right:
                continue
            new_bbox = _fit_bbox_to_arrows_1d(
                sb.bbox, left, right, vertical=False
            )
            start_arrow, end_arrow = left, right
            start_label, end_label = _SCROLL_ARROW_LEFT_ID, _SCROLL_ARROW_RIGHT_ID

        if not _proposed_pair_bbox_valid(
            new_bbox,
            out,
            ignore=sb,
            reject_scrollbar_overlap=False,
            end_arrows=(start_arrow, end_arrow),
        ):
            skipped_overlap += 1
            continue

        if _unify_end_arrow_label(out, start_arrow, start_label):
            unified += 1
        if _unify_end_arrow_label(out, end_arrow, end_label):
            unified += 1

        new_bbox = _expand_scrollbar_bbox_main_axis(
            new_bbox, vertical=vertical
        )
        if new_bbox != sb.bbox:
            out[idx] = _rebuild_detection(
                new_bbox,
                class_id=sb.class_id,
                text=sb.text,
                icons=sb.icons,
            )
            adjusted += 1

    if (adjusted or unified or skipped_overlap) and log_info is not None:
        log_info(
            f"fit_scrollbar_bboxes_to_arrow_controls: adjusted={adjusted} "
            f"unified_labels={unified} skipped_overlap={skipped_overlap} "
            f"scrollbars={len(scrollbars)}"
        )
    return out


def _pair_cross_axis_aligned(
    a: UiDetection,
    b: UiDetection,
    *,
    vertical: bool,
) -> bool:
    """True when ``a`` and ``b`` share the same column (vertical) or row (horizontal).

    Tolerance is half the larger arrow's cross-axis size:
    ``abs(cx_a - cx_b) <= max(w_a, w_b) // 2`` (vertical) or the height analogue.
    """
    if vertical:
        tol = max(a.bbox[2], b.bbox[2]) // 2
        return abs(a.cx - b.cx) <= tol
    tol = max(a.bbox[3], b.bbox[3]) // 2
    return abs(a.cy - b.cy) <= tol


def _pair_ordered_along_axis(
    start: UiDetection,
    end: UiDetection,
    *,
    vertical: bool,
) -> bool:
    """True when ``start`` sits above (vertical) or left of (horizontal) ``end``."""
    if vertical:
        return start.cy < end.cy
    return start.cx < end.cx


def _pair_main_axis_distance(
    start: UiDetection,
    end: UiDetection,
    *,
    vertical: bool,
) -> int:
    """Absolute center distance along the scrollbar main axis."""
    if vertical:
        return abs(end.cy - start.cy)
    return abs(end.cx - start.cx)


# Created bars grow this many pixels on each side of the cross axis so thin
# arrow unions better match typical track thickness.
_CREATE_SCROLLBAR_CROSS_EXPAND_PX = 2

# Fitted / created bars grow this many pixels past each end cap along the
# main axis (top/bottom or left/right).
_FIT_SCROLLBAR_MAIN_EXPAND_PX = 4


def _expand_scrollbar_bbox_cross_axis(
    bbox: tuple[int, int, int, int],
    *,
    vertical: bool,
    expand_px: int = _CREATE_SCROLLBAR_CROSS_EXPAND_PX,
) -> tuple[int, int, int, int]:
    """Widen ``bbox`` on the cross axis by ``expand_px`` on each side."""
    x, y, w, h = bbox
    if expand_px <= 0:
        return bbox
    if vertical:
        return x - expand_px, y, w + 2 * expand_px, h
    return x, y - expand_px, w, h + 2 * expand_px


def _expand_scrollbar_bbox_main_axis(
    bbox: tuple[int, int, int, int],
    *,
    vertical: bool,
    expand_px: int = _FIT_SCROLLBAR_MAIN_EXPAND_PX,
) -> tuple[int, int, int, int]:
    """Lengthen ``bbox`` on the main axis by ``expand_px`` past each end."""
    x, y, w, h = bbox
    if expand_px <= 0:
        return bbox
    if vertical:
        return x, y - expand_px, w, h + 2 * expand_px
    return x - expand_px, y, w + 2 * expand_px, h


def _is_text_detection(det: UiDetection) -> bool:
    """True for YOLO/OCR text class detections."""
    return det.class_id == YOLO_CLASS_TEXT or det.class_name == "text"


def _is_input_detection(det: UiDetection) -> bool:
    """True for YOLO input / text-field class detections."""
    return det.class_id == YOLO_CLASS_INPUT or det.class_name == "input"


def _is_scrollbar_detection(det: UiDetection) -> bool:
    """True for scrollbar class detections."""
    return det.class_id == YOLO_CLASS_SCROLLBAR or det.class_name == "scrollbar"


def _is_scrollbar_end_arrow_detection(det: UiDetection) -> bool:
    """True when ``det`` is a directional scrollbar end arrow (V/三角/滾動箭頭)."""
    return bool(_detection_icon_chinese_ids(det) & _SCROLL_ARROW_ALL_IDS)


def _text_overlaps_scrollbar_arrow(
    text: UiDetection,
    detections: list[UiDetection],
) -> bool:
    """True when ``text`` overlaps any directional scrollbar end arrow."""
    for det in detections:
        if _is_scrollbar_end_arrow_detection(det) and boxes_overlap(
            text.bbox, det.bbox
        ):
            return True
    return False


def _track_interior_between_end_arrows(
    proposed: tuple[int, int, int, int],
    arrow_a: UiDetection,
    arrow_b: UiDetection,
    *,
    vertical: bool,
) -> tuple[int, int, int, int] | None:
    """Return the open-track bbox between two end caps, or ``None`` if empty.

    Uses ``proposed``'s cross-axis span. Along the main axis, the interior is
    the gap strictly between the two arrow boxes (exclusive of both caps).
    """
    sx, sy, sw, sh = proposed
    ax, ay, aw, ah = arrow_a.bbox
    bx, by, bw, bh = arrow_b.bbox
    if vertical:
        a0, a1 = ay, ay + ah
        b0, b1 = by, by + bh
        if a1 <= b0:
            y0, y1 = a1, b0
        elif b1 <= a0:
            y0, y1 = b1, a0
        else:
            return None
        if y1 <= y0:
            return None
        return sx, y0, sw, y1 - y0
    a0, a1 = ax, ax + aw
    b0, b1 = bx, bx + bw
    if a1 <= b0:
        x0, x1 = a1, b0
    elif b1 <= a0:
        x0, x1 = b1, a0
    else:
        return None
    if x1 <= x0:
        return None
    return x0, sy, x1 - x0, sh


# Fit rejects text/input only when cross-axis penetration into the open track
# exceeds this floor and fraction of track thickness (ignores parallel grazes).
_FIT_OVERLAP_CROSS_MIN_PX = 2
_FIT_OVERLAP_CROSS_FRAC = 0.25


def _overlap_cross_axis_depth(
    track: tuple[int, int, int, int],
    other: tuple[int, int, int, int],
    *,
    vertical: bool,
) -> int:
    """Overlap extent along the track's thin (cross) axis; 0 if no overlap."""
    tx, ty, tw, th = track
    ox, oy, ow, oh = other
    if vertical:
        return max(0, min(tx + tw, ox + ow) - max(tx, ox))
    return max(0, min(ty + th, oy + oh) - max(ty, oy))


def _fit_text_input_overlap_blocks(
    track: tuple[int, int, int, int],
    other: tuple[int, int, int, int],
) -> bool:
    """True when text/input meaningfully cuts into ``track`` (not a parallel graze)."""
    if not boxes_overlap(track, other):
        return False
    vertical = _is_vertical_scrollbar_bbox(track)
    depth = _overlap_cross_axis_depth(track, other, vertical=vertical)
    track_cross = track[2] if vertical else track[3]
    threshold = max(
        _FIT_OVERLAP_CROSS_MIN_PX,
        int(track_cross * _FIT_OVERLAP_CROSS_FRAC),
    )
    return depth > threshold


def _is_tiny_text_bbox(
    bbox: tuple[int, int, int, int],
    *,
    max_side: int = DEFAULT_SMALL_TEXT_AS_ELEMENT_MAX_SIDE,
) -> bool:
    """True when both width and height are strictly under ``max_side``."""
    _x, _y, w, h = bbox
    return w < max_side and h < max_side


def _detection_has_icon_id(det: UiDetection, chinese_id: str) -> bool:
    """True when ``det.icons`` includes ``chinese_id``."""
    return chinese_id in _detection_icon_chinese_ids(det)


def _proposed_pair_bbox_valid(
    proposed: tuple[int, int, int, int],
    detections: list[UiDetection],
    *,
    ignore: UiDetection | None = None,
    reject_scrollbar_overlap: bool = True,
    end_arrows: tuple[UiDetection, UiDetection] | None = None,
) -> bool:
    """Reject proposed bars that overlap text, input, or (optionally) a scrollbar.

    ``ignore`` skips one detection (the scrollbar being fitted) so self-overlap
    does not fail the check. ``reject_scrollbar_overlap`` is True for
    create-from-pairs (avoid duplicating a same-orientation YOLO track; V+H
    corner clips are allowed) and False for fit (allow all scrollbar overlaps).

    When ``end_arrows`` is set (fit path), text/input are rejected only if they
    overlap the **open track between** those caps with meaningful **cross-axis**
    penetration (parallel grazes of a few pixels are ignored). Tiny ``text``
    boxes (both sides under :data:`~cua_mcp.yolo_onnx.DEFAULT_SMALL_TEXT_AS_ELEMENT_MAX_SIDE`)
    are ignored as icon-sized YOLO/OCR noise on the track. Overlap on an end
    arrow or outside the caps is allowed. When ``end_arrows`` is omitted
    (create path), any text/input overlap with ``proposed`` rejects, except
    text that overlaps a directional end arrow (OCR-on-glyph noise).
    """
    interior: tuple[int, int, int, int] | None = None
    proposed_vertical = _is_vertical_scrollbar_bbox(proposed)
    if end_arrows is not None:
        a, b = end_arrows
        interior = _track_interior_between_end_arrows(
            proposed,
            a,
            b,
            vertical=proposed_vertical,
        )

    for det in detections:
        if ignore is not None and det is ignore:
            continue
        if (
            reject_scrollbar_overlap
            and _is_scrollbar_detection(det)
            and boxes_overlap(proposed, det.bbox)
            and _is_vertical_scrollbar_bbox(det.bbox) == proposed_vertical
        ):
            return False
        if _is_input_detection(det) or _is_text_detection(det):
            if end_arrows is not None:
                if _is_text_detection(det) and _is_tiny_text_bbox(det.bbox):
                    continue
                if interior is not None and _fit_text_input_overlap_blocks(
                    interior, det.bbox
                ):
                    return False
                continue
            if _is_input_detection(det) and boxes_overlap(proposed, det.bbox):
                return False
            if _is_text_detection(det) and boxes_overlap(proposed, det.bbox):
                if _text_overlaps_scrollbar_arrow(det, detections):
                    continue
                return False
    return True


def create_scrollbars_from_arrow_pairs(
    detections: list[UiDetection],
    *,
    log_info: Callable[[str], None] | None = None,
) -> list[UiDetection]:
    """
    Create scrollbar detections from same-family opposing arrow/triangle pairs.

    Pairs ``向上/下V箭頭``, ``向上/下三角``, ``向左/右V箭頭``, and ``向左/右三角``
    when both ends share a column (vertical) or row (horizontal). The scrollbar
    bbox is the union of the two arrow boxes, then widened by
    :data:`_CREATE_SCROLLBAR_CROSS_EXPAND_PX` on each side of the cross axis
    and lengthened by :data:`_FIT_SCROLLBAR_MAIN_EXPAND_PX` past each end.
    Skips pairs whose (expanded) union overlaps any text, input, or an existing
    **same-orientation** scrollbar (YOLO miss-fill only; V+H corner clips are
    allowed), except text that overlaps a directional end arrow (OCR-on-glyph
    noise). Each detection is used in at most one created pair. Matched ends
    are unified to ``*滾動箭頭`` labels. Unified ``*滾動箭頭`` icons are not
    pair seeds.
    """
    if not detections:
        return detections

    out = list(detections)
    used: set[int] = set()
    created = 0
    unified = 0

    for start_id, end_id, vertical in _SCROLLBAR_PAIR_FAMILIES:
        start_indices = [
            i
            for i, det in enumerate(out)
            if i not in used and _detection_has_icon_id(det, start_id)
        ]
        end_indices = [
            i
            for i, det in enumerate(out)
            if i not in used and _detection_has_icon_id(det, end_id)
        ]
        if not start_indices or not end_indices:
            continue

        candidates: list[tuple[int, int, int]] = []
        for si in start_indices:
            for ei in end_indices:
                start, end = out[si], out[ei]
                if not _pair_cross_axis_aligned(start, end, vertical=vertical):
                    continue
                if not _pair_ordered_along_axis(start, end, vertical=vertical):
                    continue
                dist = _pair_main_axis_distance(start, end, vertical=vertical)
                candidates.append((dist, si, ei))
        candidates.sort(key=lambda t: t[0])

        for _dist, si, ei in candidates:
            if si in used or ei in used:
                continue
            start, end = out[si], out[ei]
            proposed = _expand_scrollbar_bbox_main_axis(
                _expand_scrollbar_bbox_cross_axis(
                    merge_two_boxes(start.bbox, end.bbox),
                    vertical=vertical,
                ),
                vertical=vertical,
            )
            if not _proposed_pair_bbox_valid(
                proposed, out, reject_scrollbar_overlap=True
            ):
                continue

            out.append(
                _rebuild_detection(
                    proposed,
                    class_id=YOLO_CLASS_SCROLLBAR,
                    text=None,
                    icons=None,
                )
            )
            created += 1
            used.add(si)
            used.add(ei)

            if vertical:
                start_label, end_label = _SCROLL_ARROW_UP_ID, _SCROLL_ARROW_DOWN_ID
            else:
                start_label, end_label = _SCROLL_ARROW_LEFT_ID, _SCROLL_ARROW_RIGHT_ID
            if _unify_end_arrow_label(out, start, start_label):
                unified += 1
            if _unify_end_arrow_label(out, end, end_label):
                unified += 1

    if (created or unified) and log_info is not None:
        log_info(
            f"create_scrollbars_from_arrow_pairs: created={created} "
            f"unified_labels={unified}"
        )
    return out


def _scrollbar_has_v_or_triangle_ends(
    scrollbar: UiDetection,
    detections: list[UiDetection],
) -> bool:
    """True when ``scrollbar`` has opposing V箭頭/三角/滾動箭頭 ends (not unknown)."""
    vertical = _is_vertical_scrollbar_bbox(scrollbar.bbox)
    arrow_ids = (
        _SCROLL_ARROW_VERTICAL_IDS if vertical else _SCROLL_ARROW_HORIZONTAL_IDS
    )
    arrows = [
        det
        for det in detections
        if det is not scrollbar
        and bool(_detection_icon_chinese_ids(det) & arrow_ids)
    ]
    if vertical:
        top = _pick_scrollbar_end_arrow(
            arrows,
            end="top",
            preferred_ids=_SCROLL_ARROW_UP_IDS,
            any_ids=_SCROLL_ARROW_VERTICAL_IDS,
            scrollbar_bbox=scrollbar.bbox,
            vertical=True,
            allow_unknown=False,
        )
        bottom = _pick_scrollbar_end_arrow(
            arrows,
            end="bottom",
            preferred_ids=_SCROLL_ARROW_DOWN_IDS,
            any_ids=_SCROLL_ARROW_VERTICAL_IDS,
            scrollbar_bbox=scrollbar.bbox,
            vertical=True,
            allow_unknown=False,
        )
        return top is not None and bottom is not None and top is not bottom
    left = _pick_scrollbar_end_arrow(
        arrows,
        end="left",
        preferred_ids=_SCROLL_ARROW_LEFT_IDS,
        any_ids=_SCROLL_ARROW_HORIZONTAL_IDS,
        scrollbar_bbox=scrollbar.bbox,
        vertical=False,
        allow_unknown=False,
    )
    right = _pick_scrollbar_end_arrow(
        arrows,
        end="right",
        preferred_ids=_SCROLL_ARROW_RIGHT_IDS,
        any_ids=_SCROLL_ARROW_HORIZONTAL_IDS,
        scrollbar_bbox=scrollbar.bbox,
        vertical=False,
        allow_unknown=False,
    )
    return left is not None and right is not None and left is not right


def drop_scrollbars_without_arrow_ends(
    detections: list[UiDetection],
    *,
    log_info: Callable[[str], None] | None = None,
) -> list[UiDetection]:
    """
    Remove scrollbar detections that lack V箭頭 / 三角 / ``*滾動箭頭`` end caps.

    Unknown-only ends do not count. Non-scrollbar detections are kept unchanged.
    """
    if not detections:
        return detections

    kept: list[UiDetection] = []
    dropped = 0
    for det in detections:
        if not _is_scrollbar_detection(det):
            kept.append(det)
            continue
        if _scrollbar_has_v_or_triangle_ends(det, detections):
            kept.append(det)
        else:
            dropped += 1

    if dropped and log_info is not None:
        log_info(
            f"drop_scrollbars_without_arrow_ends: dropped={dropped} "
            f"kept_scrollbars={sum(1 for d in kept if _is_scrollbar_detection(d))}"
        )
    return kept


def merge_overlapping_scrollbars(
    detections: list[UiDetection],
    *,
    min_iou: float = DEFAULT_MERGE_SAME_CLASS_IOU_THRESHOLD,
    log_info: Callable[[str], None] | None = None,
) -> list[UiDetection]:
    """
    Merge scrollbar detections whose pairwise IoU is strictly greater than ``min_iou``.

    Same default threshold as YOLO same-class merge (0.2). Groups are transitive;
    each group becomes the axis-aligned union bbox. Non-scrollbar detections are
    unchanged.
    """
    if not detections:
        return detections

    sb_idxs = [
        i for i, det in enumerate(detections) if _is_scrollbar_detection(det)
    ]
    if len(sb_idxs) < 2:
        return detections

    parent = {i: i for i in sb_idxs}

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        # Keep the lower index as root so the merged bar stays near first sighting.
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for a_pos, ia in enumerate(sb_idxs):
        for ib in sb_idxs[a_pos + 1 :]:
            if iou_xywh(detections[ia].bbox, detections[ib].bbox) > min_iou:
                _union(ia, ib)

    groups: dict[int, list[int]] = {}
    for i in sb_idxs:
        groups.setdefault(_find(i), []).append(i)

    out: list[UiDetection | None] = list(detections)
    merged_groups = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        merged_groups += 1
        union_box = detections[members[0]].bbox
        for m in members[1:]:
            union_box = merge_two_boxes(union_box, detections[m].bbox)
        keep = min(members)
        out[keep] = _rebuild_detection(
            union_box,
            class_id=YOLO_CLASS_SCROLLBAR,
            text=None,
            icons=None,
        )
        for m in members:
            if m != keep:
                out[m] = None

    if merged_groups and log_info is not None:
        log_info(
            f"merge_overlapping_scrollbars: merged_groups={merged_groups} "
            f"min_iou={min_iou}"
        )
    return [det for det in out if det is not None]


def _icon_chinese_ids_from_candidate(candidate: Any) -> set[str]:
    """Return non-empty ``chinese_id`` values from a dict or ``UiDetection``."""
    if isinstance(candidate, UiDetection):
        return _detection_icon_chinese_ids(candidate)
    if not isinstance(candidate, dict):
        return set()
    return {
        str(icon.get("chinese_id", "")).strip()
        for icon in (candidate.get("icons") or [])
        if isinstance(icon, dict) and str(icon.get("chinese_id", "")).strip()
    }


def is_scrollbar_end_arrow_candidate(candidate: Any) -> bool:
    """True when ``candidate`` is a scrollbar end-cap arrow icon.

    Matches canonical ``*滾動箭頭`` labels and pre-unify triangle/V ids.
    """
    return bool(_icon_chinese_ids_from_candidate(candidate) & _SCROLL_ARROW_ALL_IDS)


def scrollbar_orientation(bbox: tuple[int, int, int, int] | list[int]) -> str:
    """Return ``"vertical"`` when height >= width, else ``"horizontal"``."""
    x, y, w, h = (int(v) for v in bbox[:4])
    _ = x, y
    return "vertical" if h >= w else "horizontal"


def point_in_bbox(
    x: int,
    y: int,
    bbox: tuple[int, int, int, int] | list[int],
) -> bool:
    """True when ``(x, y)`` lies inside an axis-aligned ``(x, y, w, h)`` bbox."""
    bx, by, bw, bh = (int(v) for v in bbox[:4])
    return bx <= int(x) < bx + bw and by <= int(y) < by + bh


def scrollbar_axis_percent(
    x: int,
    y: int,
    bbox: tuple[int, int, int, int] | list[int],
) -> int:
    """Return 0–100 position along the scrollbar's main axis (full fitted bbox).

    Vertical bars use ``y``; horizontal bars use ``x``. The cross-axis
    coordinate is ignored so drag releases outside the bar still project.
    """
    bx, by, bw, bh = (int(v) for v in bbox[:4])
    if scrollbar_orientation((bx, by, bw, bh)) == "vertical":
        span = max(1, bh)
        frac = (int(y) - by) / span
    else:
        span = max(1, bw)
        frac = (int(x) - bx) / span
    return int(max(0, min(100, round(frac * 100))))


def point_from_scrollbar_percent(
    bbox: tuple[int, int, int, int] | list[int],
    percent: int,
) -> tuple[int, int]:
    """Map a 0–100 track percent to a pixel on the scrollbar (cross-axis center)."""
    bx, by, bw, bh = (int(v) for v in bbox[:4])
    pct = max(0, min(100, int(percent)))
    if scrollbar_orientation((bx, by, bw, bh)) == "vertical":
        # Use inclusive end so 100% lands on the last pixel of the track.
        y = by + int(round(pct / 100 * max(0, bh - 1)))
        return bx + bw // 2, y
    x = bx + int(round(pct / 100 * max(0, bw - 1)))
    return x, by + bh // 2
