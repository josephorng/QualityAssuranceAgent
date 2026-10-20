"""Live UIA Edit/Document/ScrollBar + labeled-control refine.

One FindAll under a live hwnd, split by control type, convert to image-local
xywh, drop oversized (near-window) rects, prefer non-empty UIA inputs/scrollbars
over YOLO (+ Hough for inputs), and use labeled Text/Hyperlink/ListItem/Button/
MenuItem/TreeItem/DataItem as OCR assist for YOLO text and element boxes
(split blobs, IoU fill, orphans; collapse high-overlap text+icon clusters).
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Callable

from cua_mcp.geometry import (
    clip_box,
    intersection_area_xywh,
    iou_xywh,
    point_in_rect_xywh,
)
from cua_mcp.input_box_rectangles import DEFAULT_INPUT_RECT_IOU_THRESHOLD
from src.recorder.focus_point import (
    _CLSID_CUIAutomation,
    _CLSCTX_INPROC_SERVER,
    _COINIT_MULTITHREADED,
    _IID_IUIAutomation,
    _S_FALSE,
    _S_OK,
    _UIA_GET_CURRENT_BOUNDING_RECTANGLE_INDEX,
    _com_release,
    _com_vtable,
    _guid_from_string,
)

# Fraction of window area at/above which a UIA rect is treated as window-sized.
DEFAULT_UIA_OVERSIZE_AREA_FRAC = 0.8
# Match ``DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD`` in select_mouse_target (avoid cycle).
DEFAULT_TEXT_ELEMENT_CLUSTER_IOU_THRESHOLD = 0.5
# YOLO mostly inside UIA when intersection covers at least this fraction of YOLO area.
DEFAULT_UIA_CONTAINMENT_FRAC = 0.5
# Drop a smaller element when this fraction of its area lies inside a larger one
# (duplicate YOLO crops of the same icon).
DEFAULT_NESTED_ELEMENT_CONTAINMENT_FRAC = 0.7

# IUIAutomation vtable (after IUnknown 0..2).
_UIA_ELEMENT_FROM_HANDLE = 6
_UIA_CREATE_PROPERTY_CONDITION = 23
_UIA_CREATE_OR_CONDITION = 28

# IUIAutomationElement
_UIA_FIND_ALL = 6
_UIA_GET_CURRENT_PATTERN = 16
_UIA_GET_CURRENT_CONTROL_TYPE = 21
_UIA_GET_CURRENT_NAME = 23

# IUIAutomationElementArray
_UIA_ARRAY_GET_LENGTH = 3
_UIA_ARRAY_GET_ELEMENT = 4

_UIA_CONTROL_TYPE_PROPERTY_ID = 30003
_UIA_BUTTON_CONTROL_TYPE_ID = 50000
_UIA_EDIT_CONTROL_TYPE_ID = 50004
_UIA_HYPERLINK_CONTROL_TYPE_ID = 50005
_UIA_LISTITEM_CONTROL_TYPE_ID = 50007
_UIA_MENUITEM_CONTROL_TYPE_ID = 50011
_UIA_TEXT_CONTROL_TYPE_ID = 50020
_UIA_TREEITEM_CONTROL_TYPE_ID = 50025
_UIA_SCROLLBAR_CONTROL_TYPE_ID = 50028
_UIA_DATAITEM_CONTROL_TYPE_ID = 50029
_UIA_DOCUMENT_CONTROL_TYPE_ID = 50030
_UIA_VALUE_PATTERN = 10002
_UIA_VALUE_GET_CURRENT = 4
_TREE_SCOPE_DESCENDANTS = 0x4
_VT_I4 = 3

_LABELED_TEXT_CONTROL_TYPES = frozenset(
    {
        _UIA_BUTTON_CONTROL_TYPE_ID,
        _UIA_HYPERLINK_CONTROL_TYPE_ID,
        _UIA_LISTITEM_CONTROL_TYPE_ID,
        _UIA_MENUITEM_CONTROL_TYPE_ID,
        _UIA_TEXT_CONTROL_TYPE_ID,
        _UIA_TREEITEM_CONTROL_TYPE_ID,
        _UIA_DATAITEM_CONTROL_TYPE_ID,
    }
)

RectXywh = tuple[int, int, int, int]
ScreenXyxy = tuple[int, int, int, int]
LabeledScreenXyxy = tuple[ScreenXyxy, str]
LabeledRectXywh = tuple[RectXywh, str]

UiaScreenEnumerator = Callable[[int], list[ScreenXyxy]]
# (inputs, scrollbars, labeled_texts)
UiaCombinedScreenEnumerator = Callable[
    [int],
    tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy]],
]


class _VARIANT_UNION(ctypes.Union):
    _fields_ = [
        ("lVal", ctypes.c_long),
        ("llVal", ctypes.c_longlong),
        ("dblVal", ctypes.c_double),
        ("pv", ctypes.c_void_p),
        ("raw", ctypes.c_byte * 16),
    ]


class VARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("_", _VARIANT_UNION),
    ]


def _variant_i4(value: int) -> VARIANT:
    var = VARIANT()
    var.vt = _VT_I4
    var._.lVal = int(value)
    return var


def is_oversized_uia_rect(
    rect_xywh: RectXywh,
    window_xywh: RectXywh,
    *,
    max_area_frac: float = DEFAULT_UIA_OVERSIZE_AREA_FRAC,
) -> bool:
    """True when ``rect`` covers nearly the whole window (or has no area)."""
    _x, _y, w, h = (int(v) for v in rect_xywh)
    _wx, _wy, ww, wh = (int(v) for v in window_xywh)
    if w <= 0 or h <= 0:
        return True
    window_area = float(max(0, ww) * max(0, wh))
    if window_area <= 0.0:
        return True
    return (float(w) * float(h)) / window_area >= float(max_area_frac)


def uia_label_prefer_value(value: str, name: str) -> str:
    """Prefer non-empty stripped Value over Name."""
    stripped_value = str(value or "").strip()
    if stripped_value:
        return stripped_value
    return str(name or "").strip()


def filter_uia_input_rects(
    screen_xyxy: list[ScreenXyxy],
    *,
    window_screen_xywh: RectXywh,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    max_area_frac: float = DEFAULT_UIA_OVERSIZE_AREA_FRAC,
) -> list[RectXywh]:
    """Convert screen xyxy → image-local xywh; drop empty, oversized, out-of-ROI."""
    ox, oy = int(monitor_offset[0]), int(monitor_offset[1])
    out: list[RectXywh] = []
    seen: set[RectXywh] = set()
    for left, top, right, bottom in screen_xyxy:
        if int(right) <= int(left) or int(bottom) <= int(top):
            continue
        local = (
            int(left) - ox,
            int(top) - oy,
            int(right) - int(left),
            int(bottom) - int(top),
        )
        x, y, w, h = clip_box(local[0], local[1], local[2], local[3], image_w, image_h)
        if w <= 0 or h <= 0:
            continue
        box = (x, y, w, h)
        window_local = (
            int(window_screen_xywh[0]) - ox,
            int(window_screen_xywh[1]) - oy,
            int(window_screen_xywh[2]),
            int(window_screen_xywh[3]),
        )
        if is_oversized_uia_rect(box, window_local, max_area_frac=max_area_frac):
            continue
        if ocr_roi is not None and not point_in_rect_xywh(
            x + w * 0.5,
            y + h * 0.5,
            ocr_roi,
            pad=int(ocr_roi_pad),
        ):
            continue
        if box in seen:
            continue
        seen.add(box)
        out.append(box)
    return out


filter_uia_scrollbar_rects = filter_uia_input_rects


def filter_uia_labeled_text_rects(
    screen_labeled: list[LabeledScreenXyxy],
    *,
    window_screen_xywh: RectXywh,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    max_area_frac: float = DEFAULT_UIA_OVERSIZE_AREA_FRAC,
) -> list[LabeledRectXywh]:
    """Convert labeled screen xyxy → image-local xywh + label; drop empty labels."""
    boxes_only = [xyxy for xyxy, _label in screen_labeled]
    filtered = filter_uia_input_rects(
        boxes_only,
        window_screen_xywh=window_screen_xywh,
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
        max_area_frac=max_area_frac,
    )
    # Re-pair labels: filter drops/reorders by geometry; map via converted local box.
    ox, oy = int(monitor_offset[0]), int(monitor_offset[1])
    label_by_box: dict[RectXywh, str] = {}
    for (left, top, right, bottom), label in screen_labeled:
        text = str(label or "").strip()
        if not text:
            continue
        if int(right) <= int(left) or int(bottom) <= int(top):
            continue
        local = (
            int(left) - ox,
            int(top) - oy,
            int(right) - int(left),
            int(bottom) - int(top),
        )
        x, y, w, h = clip_box(local[0], local[1], local[2], local[3], image_w, image_h)
        if w <= 0 or h <= 0:
            continue
        label_by_box[(x, y, w, h)] = text
    out: list[LabeledRectXywh] = []
    seen: set[RectXywh] = set()
    for box in filtered:
        text = label_by_box.get(box)
        if not text or box in seen:
            continue
        seen.add(box)
        out.append((box, text))
    return out


def uia_covers_yolo_inputs(
    uia_boxes: list[RectXywh],
    yolo_input_boxes: list[RectXywh],
    *,
    iou_threshold: float = DEFAULT_INPUT_RECT_IOU_THRESHOLD,
) -> bool:
    """Diagnostic: True when every YOLO input overlaps some UIA box."""
    if not uia_boxes:
        return False
    if not yolo_input_boxes:
        return True
    threshold = float(iou_threshold)
    for yolo in yolo_input_boxes:
        best = 0.0
        for uia in uia_boxes:
            best = max(best, iou_xywh(yolo, uia))
            if best > threshold:
                break
        if best <= threshold:
            return False
    return True


uia_covers_yolo_scrollbars = uia_covers_yolo_inputs


def _uia_rect_inside_blob(
    uia_box: RectXywh,
    blob: RectXywh,
    *,
    iou_threshold: float,
) -> bool:
    ux, uy, uw, uh = (int(v) for v in uia_box)
    if point_in_rect_xywh(ux + uw * 0.5, uy + uh * 0.5, blob, pad=0):
        return True
    return iou_xywh(uia_box, blob) > float(iou_threshold)


def _yolo_mostly_inside_uia(
    yolo: RectXywh,
    uia: RectXywh,
    *,
    frac: float = DEFAULT_UIA_CONTAINMENT_FRAC,
) -> bool:
    """True when intersection covers at least ``frac`` of the YOLO box area."""
    _x, _y, w, h = (int(v) for v in yolo)
    area = float(max(0, w) * max(0, h))
    if area <= 0.0:
        return False
    return intersection_area_xywh(yolo, uia) / area >= float(frac)


def merge_nested_element_boxes(
    boxes: list[RectXywh],
    *,
    frac: float = DEFAULT_NESTED_ELEMENT_CONTAINMENT_FRAC,
) -> list[RectXywh]:
    """Drop element boxes that are mostly contained in a larger element.

    YOLO often emits a tight crop and a loose crop for the same icon. Keeping
    both blocks UIA 1-text+1-icon sibling association. Prefer the larger box.
    """
    if len(boxes) < 2:
        return list(boxes)
    contain_frac = float(frac)
    # Largest first so a kept box is never nested inside a later candidate.
    ordered = sorted(boxes, key=_box_area_xywh, reverse=True)
    kept: list[RectXywh] = []
    for box in ordered:
        if any(
            _yolo_mostly_inside_uia(box, outer, frac=contain_frac) for outer in kept
        ):
            continue
        kept.append(box)
    # Restore roughly top-to-bottom / left-to-right reading order.
    kept.sort(key=lambda b: (int(b[1]), int(b[0])))
    return kept


def _partition_occupant_groups(
    occupants: list[tuple[int, str, int, RectXywh]],
    *,
    cluster_iou_threshold: float,
) -> list[list[tuple[int, str, int, RectXywh]]]:
    """Group remaining blobs that highly overlap; others stay singleton groups.

    ``occupants`` entries are ``(remaining_index, kind, local_idx, box)``.
    """
    if not occupants:
        return []
    thresh = float(cluster_iou_threshold)
    n = len(occupants)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if iou_xywh(occupants[i][3], occupants[j][3]) > thresh:
                _union(i, j)

    groups_map: dict[int, list[tuple[int, str, int, RectXywh]]] = {}
    for i, occ in enumerate(occupants):
        groups_map.setdefault(_find(i), []).append(occ)
    return list(groups_map.values())


def _text_element_sibling_occupants(
    occupants: list[tuple[int, str, int, RectXywh]],
) -> list[tuple[int, str, int, RectXywh]] | None:
    """Return occupants when they are exactly one text + one element; else None.

    Side-by-side icon+label rows share one UIA Name without high mutual IoU.
    """
    if len(occupants) != 2:
        return None
    kinds = {occ[1] for occ in occupants}
    if kinds != {"text", "element"}:
        return None
    return occupants


def _box_area_xywh(box: RectXywh) -> int:
    return max(0, int(box[2])) * max(0, int(box[3]))


def _uia_safe_to_emit_as_candidate(
    uia_box: RectXywh,
    *,
    text_boxes: list[RectXywh],
    element_boxes: list[RectXywh],
    containment_frac: float,
    cluster_iou_threshold: float,
) -> bool:
    """False when emitting this UIA would be an ambiguous click target.

    Replay clicks the UIA center. If several distinct YOLO texts/icons sit
    inside the control, that center can land on the wrong child. Allow only:

    - no YOLO inside, or at most one text and at most one element; or
    - all inside YOLO boxes form a single high-overlap cluster (dual identity).
    """
    occupants: list[tuple[int, str, int, RectXywh]] = []
    for i, box in enumerate(text_boxes):
        if _yolo_mostly_inside_uia(box, uia_box, frac=containment_frac):
            occupants.append((i, "text", i, box))
    for i, box in enumerate(element_boxes):
        if _yolo_mostly_inside_uia(box, uia_box, frac=containment_frac):
            occupants.append((i, "element", i, box))
    if not occupants:
        return True
    n_text = sum(1 for _ri, kind, _li, _b in occupants if kind == "text")
    n_elem = sum(1 for _ri, kind, _li, _b in occupants if kind == "element")
    if n_text <= 1 and n_elem <= 1:
        return True
    groups = _partition_occupant_groups(
        occupants, cluster_iou_threshold=cluster_iou_threshold
    )
    return len(groups) == 1


def _dedupe_overlapping_uia_labels(
    uia_texts: list[LabeledRectXywh],
    *,
    iou_threshold: float,
) -> list[LabeledRectXywh]:
    """Collapse near-duplicate UIA Names (short vs long) to one box.

    Search/list rows often expose both ``Name`` and ``Name, …`` with nearly the
    same rect. Keeping both falsely triggers YOLO blob splits and blocks
    icon+label sibling claims. Prefer the shorter label, then smaller area.
    """
    thresh = float(iou_threshold)
    indexed = list(enumerate(uia_texts))
    indexed.sort(
        key=lambda item: (
            len(str(item[1][1] or "")),
            _box_area_xywh(item[1][0]),
            item[0],
        )
    )
    kept: list[LabeledRectXywh] = []
    kept_boxes: list[RectXywh] = []
    for _i, (box, label) in indexed:
        if any(iou_xywh(box, prev) > thresh for prev in kept_boxes):
            continue
        kept.append((box, str(label or "")))
        kept_boxes.append(box)
    return kept


def _spatially_distinct_uia_indices(
    indices: list[int],
    uia_list: list[LabeledRectXywh],
    *,
    iou_threshold: float,
) -> list[int]:
    """Keep UIA indices whose boxes do not heavily overlap already-kept ones."""
    thresh = float(iou_threshold)
    kept: list[int] = []
    for i in indices:
        box = uia_list[i][0]
        if any(iou_xywh(box, uia_list[j][0]) > thresh for j in kept):
            continue
        kept.append(i)
    return kept


def associate_uia_labels_with_yolo_text(
    yolo_text_boxes: list[RectXywh],
    uia_texts: list[LabeledRectXywh],
    *,
    yolo_element_boxes: list[RectXywh] | None = None,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    iou_threshold: float = DEFAULT_INPUT_RECT_IOU_THRESHOLD,
    cluster_iou_threshold: float = DEFAULT_TEXT_ELEMENT_CLUSTER_IOU_THRESHOLD,
    containment_frac: float = DEFAULT_UIA_CONTAINMENT_FRAC,
) -> tuple[list[LabeledRectXywh], list[RectXywh], list[RectXywh], dict[str, object]]:
    """Split/fill/orphan UIA labels against YOLO text and element boxes.

    Returns
    ``(uia_sourced_detections, yolo_text_for_ocr, yolo_element_for_ocr, meta)``.

    Sourced detections always use the **UIA control rect** (not the YOLO crop)
    so recording hit-tests and replay clicks cover the real control. Claimed
    YOLO boxes are dropped from OCR remainders only.

    Highly overlapped text+element pairs claimed by UIA collapse to one sourced
    text detection; both boxes are dropped from OCR remainders.

    A single UIA may also claim at most one text and one element when both are
    mostly inside it (list-row icon + label), even if those YOLO boxes do not
    overlap each other.

    UIA boxes that contain more than one distinct text or more than one
    distinct element (and are not a single overlap cluster) are never emitted
    as candidates — including as orphans — so replay center-clicks cannot land
    on the wrong nested control.
    """
    threshold = float(iou_threshold)
    cluster_thresh = float(cluster_iou_threshold)
    contain_frac = float(containment_frac)
    text_boxes = list(yolo_text_boxes)
    element_boxes = merge_nested_element_boxes(list(yolo_element_boxes or []))
    uia_list = _dedupe_overlapping_uia_labels(
        list(uia_texts), iou_threshold=cluster_thresh
    )
    used_uia: set[int] = set()
    sourced: list[LabeledRectXywh] = []
    sourced_boxes: set[RectXywh] = set()
    claimed_text: set[int] = set()
    claimed_element: set[int] = set()
    # Per-UIA kinds already claimed (for icon+label sibling completion).
    uia_claim_kinds: dict[int, set[str]] = {}
    split_blobs = 0
    iou_filled = 0
    iou_filled_element = 0
    containment_filled = 0
    containment_filled_element = 0
    sibling_filled = 0
    orphans = 0
    ambiguous_skipped = 0
    cluster_collapsed = 0

    def _safe_to_emit(uia_box: RectXywh) -> bool:
        return _uia_safe_to_emit_as_candidate(
            uia_box,
            text_boxes=text_boxes,
            element_boxes=element_boxes,
            containment_frac=contain_frac,
            cluster_iou_threshold=cluster_thresh,
        )

    # Union blobs: (kind, local_index, box)
    all_blobs: list[tuple[str, int, RectXywh]] = [
        ("text", i, box) for i, box in enumerate(text_boxes)
    ] + [("element", i, box) for i, box in enumerate(element_boxes)]

    def _emit(box: RectXywh, label: str) -> None:
        if box in sourced_boxes:
            return
        sourced_boxes.add(box)
        sourced.append((box, label))

    def _claim_blob(kind: str, local_idx: int) -> None:
        if kind == "text":
            claimed_text.add(local_idx)
        else:
            claimed_element.add(local_idx)

    def _record_uia_kind(ui: int, kind: str) -> None:
        uia_claim_kinds.setdefault(ui, set()).add(kind)

    def _occupants_inside(
        uia_box: RectXywh,
    ) -> list[tuple[int, str, int, RectXywh]]:
        return [
            (ri, kind, local_idx, yolo)
            for ri, (kind, local_idx, yolo) in enumerate(remaining)
            if ri not in used_remaining
            and _yolo_mostly_inside_uia(yolo, uia_box, frac=contain_frac)
        ]

    def _claim_occupant_group(
        ui: int,
        label: str,
        group: list[tuple[int, str, int, RectXywh]],
        *,
        uia_box: RectXywh,
        count_containment: bool,
        count_sibling: bool = False,
    ) -> None:
        nonlocal containment_filled, containment_filled_element, sibling_filled
        if not group:
            return
        used_uia.add(ui)
        if count_containment:
            containment_filled += 1
        if count_sibling:
            sibling_filled += 1
            # Label already emitted by IoU/containment; only skip OCR here.
        else:
            _emit(uia_box, label)
        for ri, kind, local_idx, _yolo in group:
            used_remaining.add(ri)
            _claim_blob(kind, local_idx)
            _record_uia_kind(ui, kind)
            if kind == "element" and count_containment:
                containment_filled_element += 1

    remaining: list[tuple[str, int, RectXywh]] = []
    for kind, local_idx, blob in all_blobs:
        contained = [
            i
            for i, (uia_box, _label) in enumerate(uia_list)
            if i not in used_uia
            and _uia_rect_inside_blob(uia_box, blob, iou_threshold=threshold)
        ]
        # Duplicate Name/Name+suffix UIAs share one rect; only spatially
        # distinct children should split a YOLO blob.
        distinct = _spatially_distinct_uia_indices(
            contained, uia_list, iou_threshold=cluster_thresh
        )
        if len(distinct) >= 2:
            split_blobs += 1
            _claim_blob(kind, local_idx)
            for i in distinct:
                uia_box, label = uia_list[i]
                used_uia.add(i)
                _emit(uia_box, label)
            continue
        remaining.append((kind, local_idx, blob))

    # Greedy 1:1 IoU fill on remaining YOLO blobs vs unused UIA.
    pairs: list[tuple[float, int, int]] = []
    for ri, (_kind, _local_idx, yolo) in enumerate(remaining):
        for ui, (uia_box, _label) in enumerate(uia_list):
            if ui in used_uia:
                continue
            score = iou_xywh(yolo, uia_box)
            if score > threshold:
                pairs.append((score, ri, ui))
    pairs.sort(reverse=True)
    used_remaining: set[int] = set()
    for _score, ri, ui in pairs:
        if ri in used_remaining or ui in used_uia:
            continue
        uia_box, label = uia_list[ui]
        if not _safe_to_emit(uia_box):
            ambiguous_skipped += 1
            used_uia.add(ui)
            continue
        used_remaining.add(ri)
        used_uia.add(ui)
        kind, local_idx, _yolo = remaining[ri]
        _claim_blob(kind, local_idx)
        _record_uia_kind(ui, kind)
        iou_filled += 1
        if kind == "element":
            iou_filled_element += 1
        _emit(uia_box, label)

    # Containment fill: small YOLO mostly inside an exclusive unused UIA.
    # Accept a single overlap cluster, or exactly one text + one element (list
    # row icon beside label) even when those two do not overlap each other.
    for ui, (uia_box, label) in enumerate(uia_list):
        if ui in used_uia:
            continue
        if not _safe_to_emit(uia_box):
            ambiguous_skipped += 1
            used_uia.add(ui)
            continue
        occupants = _occupants_inside(uia_box)
        groups = _partition_occupant_groups(
            occupants, cluster_iou_threshold=cluster_thresh
        )
        sibling = _text_element_sibling_occupants(occupants)
        if len(groups) == 1:
            group = groups[0]
        elif sibling is not None:
            group = sibling
        else:
            continue
        _claim_occupant_group(
            ui, label, group, uia_box=uia_box, count_containment=True
        )

    # Sibling completion: UIA that already claimed text (or element) via IoU /
    # containment may also claim exactly one remaining opposite-kind blob
    # mostly inside it — at most one text and one element per UIA.
    for ui, (uia_box, label) in enumerate(uia_list):
        if ui not in used_uia:
            continue
        kinds = uia_claim_kinds.get(ui, set())
        if kinds >= {"text", "element"}:
            continue
        occupants = _occupants_inside(uia_box)
        if not occupants:
            continue
        if "text" in kinds and "element" not in kinds:
            extras = [occ for occ in occupants if occ[1] == "element"]
            if len(extras) != 1 or any(occ[1] == "text" for occ in occupants):
                continue
            _claim_occupant_group(
                ui,
                label,
                extras,
                uia_box=uia_box,
                count_containment=False,
                count_sibling=True,
            )
        elif "element" in kinds and "text" not in kinds:
            extras = [occ for occ in occupants if occ[1] == "text"]
            if len(extras) != 1 or any(occ[1] == "element" for occ in occupants):
                continue
            _claim_occupant_group(
                ui,
                label,
                extras,
                uia_box=uia_box,
                count_containment=False,
                count_sibling=True,
            )

    all_yolo_for_orphan = text_boxes + element_boxes
    for ui, (uia_box, label) in enumerate(uia_list):
        if ui in used_uia:
            continue
        if not _safe_to_emit(uia_box):
            ambiguous_skipped += 1
            used_uia.add(ui)
            continue
        ux, uy, uw, uh = (int(v) for v in uia_box)
        cx, cy = ux + uw * 0.5, uy + uh * 0.5
        in_blob = any(
            point_in_rect_xywh(cx, cy, blob, pad=0) for blob in all_yolo_for_orphan
        )
        in_roi = ocr_roi is not None and point_in_rect_xywh(
            cx, cy, ocr_roi, pad=int(ocr_roi_pad)
        )
        if in_blob or in_roi:
            orphans += 1
            used_uia.add(ui)
            _emit(uia_box, label)

    text_ocr_idx = {i for i in range(len(text_boxes)) if i not in claimed_text}
    element_ocr_idx = {
        i for i in range(len(element_boxes)) if i not in claimed_element
    }

    def _uia_claims_cluster(
        tbox: RectXywh, ebox: RectXywh, ti: int, ei: int
    ) -> tuple[bool, str | None, RectXywh | None]:
        """Return (claims, label, emit_box) for a high-overlap text+element pair."""
        if ti in claimed_text or ei in claimed_element:
            for sbox, slabel in sourced:
                if (
                    iou_xywh(sbox, tbox) > cluster_thresh
                    or iou_xywh(sbox, ebox) > cluster_thresh
                    or _uia_rect_inside_blob(sbox, tbox, iou_threshold=cluster_thresh)
                    or _uia_rect_inside_blob(sbox, ebox, iou_threshold=cluster_thresh)
                ):
                    return True, slabel, None
            return True, None, None
        for _ui, (ubox, ulabel) in enumerate(uia_list):
            if (
                iou_xywh(ubox, tbox) > cluster_thresh
                or iou_xywh(ubox, ebox) > cluster_thresh
                or _uia_rect_inside_blob(ubox, tbox, iou_threshold=cluster_thresh)
                or _uia_rect_inside_blob(ubox, ebox, iou_threshold=cluster_thresh)
            ):
                return True, ulabel, ubox
        return False, None, None

    for ti, tbox in enumerate(text_boxes):
        for ei, ebox in enumerate(element_boxes):
            if iou_xywh(tbox, ebox) <= cluster_thresh:
                continue
            claims, label, emit_box = _uia_claims_cluster(tbox, ebox, ti, ei)
            if not claims:
                continue
            removed = False
            if ti in text_ocr_idx:
                text_ocr_idx.discard(ti)
                claimed_text.add(ti)
                removed = True
            if ei in element_ocr_idx:
                element_ocr_idx.discard(ei)
                claimed_element.add(ei)
                removed = True
            if emit_box is not None and label:
                before = len(sourced)
                _emit(emit_box, label)
                if len(sourced) > before:
                    removed = True
            if removed:
                cluster_collapsed += 1

    yolo_text_for_ocr = [text_boxes[i] for i in sorted(text_ocr_idx)]
    yolo_element_for_ocr = [element_boxes[i] for i in sorted(element_ocr_idx)]
    element_claimed = len(element_boxes) - len(yolo_element_for_ocr)
    text_ocr_skipped = len(text_boxes) - len(yolo_text_for_ocr)

    meta: dict[str, object] = {
        "uia_texts": len(uia_list),
        "split_blobs": split_blobs,
        "iou_filled": iou_filled,
        "iou_filled_element": iou_filled_element,
        "containment_filled": containment_filled,
        "containment_filled_element": containment_filled_element,
        "sibling_filled": sibling_filled,
        "orphans": orphans,
        "ambiguous_skipped": ambiguous_skipped,
        "text_ocr_skipped": text_ocr_skipped,
        "element_claimed": element_claimed,
        "cluster_collapsed": cluster_collapsed,
        "yolo_text_for_ocr": len(yolo_text_for_ocr),
        "yolo_element_for_ocr": len(yolo_element_for_ocr),
        "uia_sourced": len(sourced),
    }
    return sourced, yolo_text_for_ocr, yolo_element_for_ocr, meta


def _window_screen_xywh(hwnd: int) -> RectXywh | None:
    if os.name != "nt" or not hwnd:
        return None
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(int(hwnd), ctypes.byref(rect)):
        return None
    w = int(rect.right) - int(rect.left)
    h = int(rect.bottom) - int(rect.top)
    if w <= 0 or h <= 0:
        return None
    return (int(rect.left), int(rect.top), w, h)


def _bstr_to_str(pointer: ctypes.c_void_p | None) -> str:
    if not pointer:
        return ""
    try:
        return str(ctypes.wstring_at(pointer) or "")
    finally:
        try:
            ctypes.windll.oleaut32.SysFreeString(pointer)
        except Exception:
            pass


def _element_bounding_xyxy(element: ctypes.c_void_p) -> ScreenXyxy | None:
    get_bounds = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT),
    )(
        _com_vtable(element, _UIA_GET_CURRENT_BOUNDING_RECTANGLE_INDEX + 1)[
            _UIA_GET_CURRENT_BOUNDING_RECTANGLE_INDEX
        ]
    )
    rect = wintypes.RECT()
    hr = get_bounds(element, ctypes.byref(rect))
    if hr != _S_OK:
        return None
    if int(rect.right) <= int(rect.left) or int(rect.bottom) <= int(rect.top):
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _element_control_type(element: ctypes.c_void_p) -> int | None:
    get_ct = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )(
        _com_vtable(element, _UIA_GET_CURRENT_CONTROL_TYPE + 1)[
            _UIA_GET_CURRENT_CONTROL_TYPE
        ]
    )
    control_type = ctypes.c_int(0)
    hr = get_ct(element, ctypes.byref(control_type))
    if hr != _S_OK:
        return None
    return int(control_type.value)


def _element_current_name(element: ctypes.c_void_p) -> str:
    try:
        getter = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(element, _UIA_GET_CURRENT_NAME + 1)[_UIA_GET_CURRENT_NAME])
        bstr = ctypes.c_void_p()
        if getter(element, ctypes.byref(bstr)) == _S_OK:
            return _bstr_to_str(bstr)
    except Exception:
        pass
    return ""


def _element_value_pattern_current(element: ctypes.c_void_p) -> str:
    getter = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    )(_com_vtable(element, _UIA_GET_CURRENT_PATTERN + 1)[_UIA_GET_CURRENT_PATTERN])
    pattern = ctypes.c_void_p()
    hr = getter(element, int(_UIA_VALUE_PATTERN), ctypes.byref(pattern))
    if hr != _S_OK or not pattern:
        return ""
    try:
        value_getter = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(
            _com_vtable(pattern, _UIA_VALUE_GET_CURRENT + 1)[_UIA_VALUE_GET_CURRENT]
        )
        bstr = ctypes.c_void_p()
        if value_getter(pattern, ctypes.byref(bstr)) == _S_OK:
            return _bstr_to_str(bstr)
    except Exception:
        return ""
    finally:
        _com_release(pattern)
    return ""


def _element_label(element: ctypes.c_void_p) -> str:
    return uia_label_prefer_value(
        _element_value_pattern_current(element),
        _element_current_name(element),
    )


def _create_control_type_condition(
    automation: ctypes.c_void_p, control_type_id: int
) -> ctypes.c_void_p | None:
    create_prop = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_int,
        VARIANT,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(automation, _UIA_CREATE_PROPERTY_CONDITION + 1)[
            _UIA_CREATE_PROPERTY_CONDITION
        ]
    )
    cond_ptr = ctypes.c_void_p()
    hr = create_prop(
        automation,
        _UIA_CONTROL_TYPE_PROPERTY_ID,
        _variant_i4(control_type_id),
        ctypes.byref(cond_ptr),
    )
    if hr != _S_OK or not cond_ptr:
        return None
    return cond_ptr


def _create_or_condition(
    automation: ctypes.c_void_p,
    left: ctypes.c_void_p,
    right: ctypes.c_void_p,
) -> ctypes.c_void_p | None:
    create_or = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(_com_vtable(automation, _UIA_CREATE_OR_CONDITION + 1)[_UIA_CREATE_OR_CONDITION])
    or_ptr = ctypes.c_void_p()
    hr = create_or(automation, left, right, ctypes.byref(or_ptr))
    if hr != _S_OK or not or_ptr:
        return None
    return or_ptr


def _fold_or_conditions(
    automation: ctypes.c_void_p, conditions: list[ctypes.c_void_p]
) -> tuple[ctypes.c_void_p | None, list[ctypes.c_void_p]]:
    """OR-fold conditions; returns ``(root, created_or_nodes_to_release)``."""
    if not conditions:
        return None, []
    created: list[ctypes.c_void_p] = []
    current = conditions[0]
    for nxt in conditions[1:]:
        folded = _create_or_condition(automation, current, nxt)
        if folded is None:
            return None, created
        created.append(folded)
        current = folded
    return current, created


def enumerate_edit_document_scrollbar_screen(
    hwnd: int,
) -> tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy]]:
    """One FindAll Edit|Document|ScrollBar plus labeled text-like controls."""
    if os.name != "nt" or not hwnd:
        return [], [], []

    ole32 = ctypes.windll.ole32
    initialized_here = False
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr == _S_OK:
        initialized_here = True
    elif hr not in (_S_OK, _S_FALSE) and (hr & 0xFFFFFFFF) != 0x80010106:
        return [], [], []

    automation: ctypes.c_void_p | None = None
    root: ctypes.c_void_p | None = None
    leaf_conds: list[ctypes.c_void_p] = []
    or_nodes: list[ctypes.c_void_p] = []
    or_cond: ctypes.c_void_p | None = None
    found: ctypes.c_void_p | None = None
    inputs: list[ScreenXyxy] = []
    scrollbars: list[ScreenXyxy] = []
    labeled: list[LabeledScreenXyxy] = []
    try:
        clsid = _guid_from_string(_CLSID_CUIAutomation)
        iid = _guid_from_string(_IID_IUIAutomation)
        automation_ptr = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid),
            None,
            _CLSCTX_INPROC_SERVER,
            ctypes.byref(iid),
            ctypes.byref(automation_ptr),
        )
        if hr != _S_OK or not automation_ptr:
            return [], [], []
        automation = automation_ptr

        element_from_handle = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(automation, _UIA_ELEMENT_FROM_HANDLE + 1)[_UIA_ELEMENT_FROM_HANDLE])
        root_ptr = ctypes.c_void_p()
        hr = element_from_handle(
            automation,
            ctypes.c_void_p(int(hwnd)),
            ctypes.byref(root_ptr),
        )
        if hr != _S_OK or not root_ptr:
            return [], [], []
        root = root_ptr

        type_ids = (
            _UIA_EDIT_CONTROL_TYPE_ID,
            _UIA_DOCUMENT_CONTROL_TYPE_ID,
            _UIA_SCROLLBAR_CONTROL_TYPE_ID,
            _UIA_BUTTON_CONTROL_TYPE_ID,
            _UIA_HYPERLINK_CONTROL_TYPE_ID,
            _UIA_LISTITEM_CONTROL_TYPE_ID,
            _UIA_MENUITEM_CONTROL_TYPE_ID,
            _UIA_TEXT_CONTROL_TYPE_ID,
            _UIA_TREEITEM_CONTROL_TYPE_ID,
            _UIA_DATAITEM_CONTROL_TYPE_ID,
        )
        for type_id in type_ids:
            cond = _create_control_type_condition(automation, type_id)
            if cond is None:
                return [], [], []
            leaf_conds.append(cond)

        or_cond, or_nodes = _fold_or_conditions(automation, leaf_conds)
        if or_cond is None:
            return [], [], []

        find_all = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(root, _UIA_FIND_ALL + 1)[_UIA_FIND_ALL])
        found_ptr = ctypes.c_void_p()
        hr = find_all(
            root,
            _TREE_SCOPE_DESCENDANTS,
            or_cond,
            ctypes.byref(found_ptr),
        )
        if hr != _S_OK or not found_ptr:
            return [], [], []
        found = found_ptr

        get_length = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )(_com_vtable(found, _UIA_ARRAY_GET_LENGTH + 1)[_UIA_ARRAY_GET_LENGTH])
        get_element = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(found, _UIA_ARRAY_GET_ELEMENT + 1)[_UIA_ARRAY_GET_ELEMENT])
        length = ctypes.c_int(0)
        hr = get_length(found, ctypes.byref(length))
        if hr != _S_OK or length.value <= 0:
            return [], [], []

        for index in range(int(length.value)):
            elem_ptr = ctypes.c_void_p()
            hr = get_element(found, index, ctypes.byref(elem_ptr))
            if hr != _S_OK or not elem_ptr:
                continue
            element = elem_ptr
            try:
                box = _element_bounding_xyxy(element)
                if box is None:
                    continue
                control_type = _element_control_type(element)
                if control_type == _UIA_SCROLLBAR_CONTROL_TYPE_ID:
                    scrollbars.append(box)
                elif control_type in (
                    _UIA_EDIT_CONTROL_TYPE_ID,
                    _UIA_DOCUMENT_CONTROL_TYPE_ID,
                ):
                    inputs.append(box)
                elif control_type in _LABELED_TEXT_CONTROL_TYPES:
                    label = _element_label(element)
                    if label:
                        labeled.append((box, label))
            finally:
                _com_release(element)
        return inputs, scrollbars, labeled
    except Exception:
        return [], [], []
    finally:
        _com_release(found)
        for node in reversed(or_nodes):
            _com_release(node)
        for cond in leaf_conds:
            _com_release(cond)
        _com_release(root)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def enumerate_edit_document_screen_xyxy(hwnd: int) -> list[ScreenXyxy]:
    """FindAll Edit|Document under ``hwnd`` (via combined walk); screen xyxy boxes."""
    inputs, _scrollbars, _labeled = enumerate_edit_document_scrollbar_screen(int(hwnd))
    return inputs


def list_uia_input_and_scrollbar_boxes_image_local(
    hwnd: int,
    *,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    max_area_frac: float = DEFAULT_UIA_OVERSIZE_AREA_FRAC,
    enumerate_fn: UiaCombinedScreenEnumerator | None = None,
    window_screen_xywh: RectXywh | None = None,
) -> tuple[list[RectXywh], list[RectXywh], list[LabeledRectXywh]]:
    """Live input + scrollbar + labeled text boxes in image-local xywh (one FindAll)."""
    window = window_screen_xywh or _window_screen_xywh(int(hwnd))
    if window is None:
        return [], [], []
    enumerator = enumerate_fn or enumerate_edit_document_scrollbar_screen
    try:
        input_screen, scrollbar_screen, labeled_screen = enumerator(int(hwnd))
    except Exception:
        return [], [], []
    filter_kwargs = dict(
        window_screen_xywh=window,
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
        max_area_frac=max_area_frac,
    )
    return (
        filter_uia_input_rects(input_screen or [], **filter_kwargs),
        filter_uia_scrollbar_rects(scrollbar_screen or [], **filter_kwargs),
        filter_uia_labeled_text_rects(labeled_screen or [], **filter_kwargs),
    )


def list_uia_input_boxes_image_local(
    hwnd: int,
    *,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    max_area_frac: float = DEFAULT_UIA_OVERSIZE_AREA_FRAC,
    enumerate_fn: UiaScreenEnumerator | None = None,
    window_screen_xywh: RectXywh | None = None,
) -> list[RectXywh]:
    """Live Edit/Document boxes in image-local xywh for one monitor capture."""
    window = window_screen_xywh or _window_screen_xywh(int(hwnd))
    if window is None:
        return []
    if enumerate_fn is not None:
        try:
            screen_boxes = enumerate_fn(int(hwnd))
        except Exception:
            return []
        if not screen_boxes:
            return []
        return filter_uia_input_rects(
            screen_boxes,
            window_screen_xywh=window,
            monitor_offset=monitor_offset,
            image_w=image_w,
            image_h=image_h,
            ocr_roi=ocr_roi,
            ocr_roi_pad=ocr_roi_pad,
            max_area_frac=max_area_frac,
        )
    inputs, _scrollbars, _labeled = list_uia_input_and_scrollbar_boxes_image_local(
        int(hwnd),
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
        max_area_frac=max_area_frac,
        window_screen_xywh=window,
    )
    return inputs


def try_uia_replace_yolo_inputs(
    yolo_input_boxes: list[RectXywh],
    *,
    hwnd: int,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    iou_threshold: float = DEFAULT_INPUT_RECT_IOU_THRESHOLD,
    enumerate_fn: UiaScreenEnumerator | None = None,
    window_screen_xywh: RectXywh | None = None,
) -> tuple[list[RectXywh] | None, dict[str, object]]:
    """Return ``(uia_boxes, meta)`` when UIA finds ≥1 usable input; else ``(None, meta)``."""
    uia_boxes = list_uia_input_boxes_image_local(
        int(hwnd),
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
        enumerate_fn=enumerate_fn,
        window_screen_xywh=window_screen_xywh,
    )
    yolo_covered = uia_covers_yolo_inputs(
        uia_boxes,
        yolo_input_boxes,
        iou_threshold=iou_threshold,
    )
    prefer_uia = bool(uia_boxes)
    meta: dict[str, object] = {
        "uia_inputs": len(uia_boxes),
        "yolo_inputs": len(yolo_input_boxes),
        "covered": prefer_uia,
        "yolo_covered": yolo_covered,
        "used": "uia" if prefer_uia else "hough",
    }
    if prefer_uia:
        return uia_boxes, meta
    return None, meta


def try_uia_refine_inputs_and_scrollbars(
    yolo_input_boxes: list[RectXywh],
    yolo_scrollbar_boxes: list[RectXywh],
    *,
    hwnd: int,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    iou_threshold: float = DEFAULT_INPUT_RECT_IOU_THRESHOLD,
    enumerate_fn: UiaCombinedScreenEnumerator | None = None,
    window_screen_xywh: RectXywh | None = None,
) -> tuple[
    list[RectXywh] | None,
    list[RectXywh] | None,
    list[LabeledRectXywh],
    dict[str, object],
]:
    """One FindAll; prefer-replace inputs/scrollbars; return labeled texts for OCR assist.

    Returns ``(uia_inputs|None, uia_scrollbars|None, uia_texts, meta)``.
    ``uia_texts`` is always a list (possibly empty).
    """
    uia_inputs, uia_scrollbars, uia_texts = list_uia_input_and_scrollbar_boxes_image_local(
        int(hwnd),
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
        enumerate_fn=enumerate_fn,
        window_screen_xywh=window_screen_xywh,
    )
    prefer_inputs = bool(uia_inputs)
    prefer_scrollbars = bool(uia_scrollbars)
    meta: dict[str, object] = {
        "uia_inputs": len(uia_inputs),
        "yolo_inputs": len(yolo_input_boxes),
        "covered": prefer_inputs,
        "yolo_covered": uia_covers_yolo_inputs(
            uia_inputs,
            yolo_input_boxes,
            iou_threshold=iou_threshold,
        ),
        "used": "uia" if prefer_inputs else "hough",
        "uia_scrollbars": len(uia_scrollbars),
        "yolo_scrollbars": len(yolo_scrollbar_boxes),
        "scrollbar_covered": prefer_scrollbars,
        "scrollbar_yolo_covered": uia_covers_yolo_scrollbars(
            uia_scrollbars,
            yolo_scrollbar_boxes,
            iou_threshold=iou_threshold,
        ),
        "scrollbar_used": "uia" if prefer_scrollbars else "yolo",
        "uia_texts": len(uia_texts),
    }
    return (
        uia_inputs if prefer_inputs else None,
        uia_scrollbars if prefer_scrollbars else None,
        uia_texts,
        meta,
    )
