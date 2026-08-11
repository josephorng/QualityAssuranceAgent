"""Fit scrollbar bboxes to nearby end-cap arrow icons after YOLO+OCR."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from cua_mcp.icon_map import (
    is_pua_char,
    is_unknown_icon_record,
    load_icon_map,
    unknown_icon_record,
)
from cua_mcp.select_ui_element import UiDetection
from cua_mcp.yolo_onnx import (
    PICKER_CLASS_UNKNOWN,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_SCROLLBAR,
)

# Scrollbar arrow button icons (OCR / icon_map chinese_id). After YOLO+OCR, each
# scrollbar bbox is extended/shrunk along its main axis to include the two end
# buttons nearest the track ends.
_SCROLL_ARROW_UP_IDS: frozenset[str] = frozenset({"向上三角", "向上V箭頭"})
_SCROLL_ARROW_DOWN_IDS: frozenset[str] = frozenset({"向下三角", "向下V箭頭"})
_SCROLL_ARROW_LEFT_IDS: frozenset[str] = frozenset({"向左三角", "向左V箭頭"})
_SCROLL_ARROW_RIGHT_IDS: frozenset[str] = frozenset({"向右三角", "向右V箭頭"})
_SCROLL_ARROW_VERTICAL_IDS: frozenset[str] = (
    _SCROLL_ARROW_UP_IDS | _SCROLL_ARROW_DOWN_IDS
)
_SCROLL_ARROW_HORIZONTAL_IDS: frozenset[str] = (
    _SCROLL_ARROW_LEFT_IDS | _SCROLL_ARROW_RIGHT_IDS
)
# Canonical labels used when reclassifying an unknown icon chosen as an end cap.
_RECLASSIFY_UP_ID = "向上V箭頭"
_RECLASSIFY_DOWN_ID = "向下V箭頭"
_RECLASSIFY_LEFT_ID = "向左V箭頭"
_RECLASSIFY_RIGHT_ID = "向右V箭頭"
_UNKNOWN_ICON_CHINESE_ID: str = str(
    unknown_icon_record().get("chinese_id", "未知圖示")
).strip()


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
    """True when ``arrow`` sits on the scrollbar's track (cross-axis overlap/pad)."""
    sx, sy, sw, sh = scrollbar_bbox
    ax, ay, aw, ah = arrow.bbox
    if vertical:
        pad = max(8, sw)
        return ax < sx + sw + pad and ax + aw > sx - pad
    pad = max(8, sh)
    return ay < sy + sh + pad and ay + ah > sy - pad


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
) -> UiDetection | None:
    """Pick a track-aligned end arrow on the correct side of the scrollbar center.

    Priority: ``preferred_ids`` (expected direction), then unknown icons, then
    ``any_ids``. Within a priority tier, choose the arrow closest to the
    original scrollbar center. ``end`` must be ``top`` / ``bottom`` / ``left`` /
    ``right`` and restricts candidates to that side of the center (e.g. top
    requires ``cy < center_y``).
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

    pool = (
        _candidates(ids=preferred_ids)
        or _candidates(unknown_only=True)
        or _candidates(ids=any_ids)
    )
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


def _reclassify_unknown_end_arrow(
    detections: list[UiDetection],
    picked: UiDetection,
    chinese_id: str,
) -> bool:
    """
    If ``picked`` is an unknown icon in ``detections``, rewrite it as ``chinese_id``.

    Returns True when a detection was updated.
    """
    if not _is_unknown_icon_detection(picked):
        return False
    try:
        index = next(i for i, det in enumerate(detections) if det is picked)
    except StopIteration:
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

    Vertical scrollbars use ``向上三角`` / ``向下三角`` / ``向上V箭頭`` /
    ``向下V箭頭``. Horizontal scrollbars use the left/right triangle and V-arrow
    ids. Unknown icons (``未知圖示`` / class ``unknown``) are preferred over
    other-direction arrows when the expected end icon is missing; those
    unknowns are then reclassified to the matching V-arrow label for that end.
    Matching arrows may be any distance along the track — the scrollbar extends
    to them. When either end lacks a matching track-aligned arrow, that
    scrollbar is left unchanged.
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
    reclassified = 0
    for idx in scrollbars:
        # Rebuild pools from ``out`` so prior unknown reclassifications apply.
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
            if _reclassify_unknown_end_arrow(out, top, _RECLASSIFY_UP_ID):
                reclassified += 1
            if _reclassify_unknown_end_arrow(out, bottom, _RECLASSIFY_DOWN_ID):
                reclassified += 1
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
            if _reclassify_unknown_end_arrow(out, left, _RECLASSIFY_LEFT_ID):
                reclassified += 1
            if _reclassify_unknown_end_arrow(out, right, _RECLASSIFY_RIGHT_ID):
                reclassified += 1

        if new_bbox != sb.bbox:
            out[idx] = _rebuild_detection(
                new_bbox,
                class_id=sb.class_id,
                text=sb.text,
                icons=sb.icons,
            )
            adjusted += 1

    if (adjusted or reclassified) and log_info is not None:
        log_info(
            f"fit_scrollbar_bboxes_to_arrow_controls: adjusted={adjusted} "
            f"reclassified_unknown={reclassified} scrollbars={len(scrollbars)}"
        )
    return out
