"""Live UIA Edit/Document/ScrollBar + labeled-control enumeration.

One FindAll under a window hwnd. Recording and replay use the snapshot as its
own candidate list, or YOLO+OCR when that list is not enough. The two lists
are not merged.

A control-type ScrollBar is a scrollbar. A Thumb whose AutomationId is
``ScrollbarThumb`` is one too. ``UpButton`` / ``DownButton`` buttons on that
thumb's line are included in the same scrollbar rectangle.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Callable

from cua_mcp.geometry import (
    clip_box,
    intersection_area_xywh,
    point_in_rect_xywh,
)
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
# IUIAutomationElement::get_CurrentAutomationId. Bounding rect is slot 43.
_UIA_GET_CURRENT_AUTOMATION_ID = 29

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
_UIA_THUMB_CONTROL_TYPE_ID = 50027
_UIA_SCROLLBAR_CONTROL_TYPE_ID = 50028
_UIA_DATAITEM_CONTROL_TYPE_ID = 50029
_UIA_DOCUMENT_CONTROL_TYPE_ID = 50030
_UIA_VALUE_PATTERN = 10002
_UIA_VALUE_GET_CURRENT = 4
_TREE_SCOPE_DESCENDANTS = 0x4
_VT_I4 = 3

_SCROLLBAR_THUMB_AUTOMATION_ID = "ScrollbarThumb"
_SCROLLBAR_LINE_BUTTON_AUTOMATION_IDS = frozenset({"UpButton", "DownButton"})
# Same-line when the shared span covers this fraction of the shorter cross-axis side.
_SCROLLBAR_LINE_OVERLAP_FRAC = 0.5
# A thumb-built rect inside a ScrollBar control is a duplicate when the control
# is at most this many times larger than the thumb rect.
_SCROLLBAR_CONTROL_CONTAIN_AREA_RATIO = 4.0

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



MAX_USABLE_UIA_LABEL_LEN = 48
MIN_USABLE_UIA_LANDMARKS = 8


def is_usable_uia_landmark_label(text: str | None) -> bool:
    """True when a UIA string is short enough to name a control.

    URL-like names and ultra-long AutomationIds are omitted from the UIA
    candidate list and do not count toward the sufficiency gate.
    """
    visible = str(text or "").strip()
    if len(visible) < 2:
        return False
    if len(visible) > MAX_USABLE_UIA_LABEL_LEN:
        return False
    lowered = visible.casefold()
    if "://" in visible or lowered.startswith("www."):
        return False
    if visible.count("/") >= 3 and ("." in visible or "%" in visible):
        return False
    return True


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


def _box_area_xywh(box: RectXywh) -> int:
    return max(0, int(box[2])) * max(0, int(box[3]))


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


def _element_automation_id(element: ctypes.c_void_p) -> str:
    try:
        getter = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(
            _com_vtable(element, _UIA_GET_CURRENT_AUTOMATION_ID + 1)[
                _UIA_GET_CURRENT_AUTOMATION_ID
            ]
        )
        bstr = ctypes.c_void_p()
        if getter(element, ctypes.byref(bstr)) == _S_OK:
            return _bstr_to_str(bstr).strip()
    except Exception:
        pass
    return ""


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


def _xyxy_area(box: ScreenXyxy) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _xyxy_intersection_area(left: ScreenXyxy, right: ScreenXyxy) -> int:
    width = min(int(left[2]), int(right[2])) - max(int(left[0]), int(right[0]))
    height = min(int(left[3]), int(right[3])) - max(int(left[1]), int(right[1]))
    if width <= 0 or height <= 0:
        return 0
    return width * height


def _union_xyxy(boxes: list[ScreenXyxy]) -> ScreenXyxy:
    return (
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    )


def _thumb_is_vertical(thumb: ScreenXyxy) -> bool:
    return (int(thumb[3]) - int(thumb[1])) >= (int(thumb[2]) - int(thumb[0]))


def _cross_axis_overlap_frac(thumb: ScreenXyxy, button: ScreenXyxy) -> float:
    """Overlap of the thumb and button on the scrollbar's cross axis, over the shorter side."""
    if _thumb_is_vertical(thumb):
        overlap = min(int(thumb[2]), int(button[2])) - max(int(thumb[0]), int(button[0]))
        shorter = min(int(thumb[2]) - int(thumb[0]), int(button[2]) - int(button[0]))
    else:
        overlap = min(int(thumb[3]), int(button[3])) - max(int(thumb[1]), int(button[1]))
        shorter = min(int(thumb[3]) - int(thumb[1]), int(button[3]) - int(button[1]))
    if overlap <= 0 or shorter <= 0:
        return 0.0
    return float(overlap) / float(shorter)


def _main_axis_gap(thumb: ScreenXyxy, button: ScreenXyxy) -> int:
    """Zero when the boxes meet on the scrollbar axis; otherwise the gap in pixels."""
    if _thumb_is_vertical(thumb):
        if int(button[3]) < int(thumb[1]):
            return int(thumb[1]) - int(button[3])
        if int(thumb[3]) < int(button[1]):
            return int(button[1]) - int(thumb[3])
        return 0
    if int(button[2]) < int(thumb[0]):
        return int(thumb[0]) - int(button[2])
    if int(thumb[2]) < int(button[0]):
        return int(button[0]) - int(thumb[2])
    return 0


def _on_same_scrollbar_line(thumb: ScreenXyxy, button: ScreenXyxy) -> bool:
    """True when ``button`` lies on the thumb's scrollbar line.

    A vertical thumb shares its column (x overlap). A horizontal thumb shares
    its row (y overlap). The shared span must cover at least half of the
    shorter cross-axis side, so a neighbor that only clips the edge does not join.
    """
    return _cross_axis_overlap_frac(thumb, button) >= _SCROLLBAR_LINE_OVERLAP_FRAC


def _scrollbar_control_already_covers(control: ScreenXyxy, part: ScreenXyxy) -> bool:
    """True when ``part`` is the thumb inside an already-found ScrollBar control."""
    part_area = _xyxy_area(part)
    if part_area <= 0:
        return True
    if _xyxy_intersection_area(control, part) / part_area < 0.8:
        return False
    control_area = _xyxy_area(control)
    return control_area <= part_area * _SCROLLBAR_CONTROL_CONTAIN_AREA_RATIO


def compose_scrollbar_screen_rects(
    scrollbar_controls: list[ScreenXyxy],
    thumbs: list[ScreenXyxy],
    line_buttons: list[ScreenXyxy],
) -> tuple[list[ScreenXyxy], list[int]]:
    """ScrollBar controls plus one rect per ``ScrollbarThumb`` and its line buttons.

    Each thumb is a scrollbar. ``UpButton`` / ``DownButton`` boxes on that
    thumb's line are unioned into the same rectangle. A thumb already covered
    by a similar-sized ScrollBar control is not added again.

    The second list is the indexes of ``line_buttons`` absorbed into a thumb.
    """
    grouped: list[list[ScreenXyxy]] = [[thumb] for thumb in thumbs]
    used: list[int] = []
    for index, button in enumerate(line_buttons):
        best_thumb: int | None = None
        best_key: tuple[float, int] | None = None
        for thumb_index, thumb in enumerate(thumbs):
            if not _on_same_scrollbar_line(thumb, button):
                continue
            key = (
                _cross_axis_overlap_frac(thumb, button),
                -_main_axis_gap(thumb, button),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_thumb = thumb_index
        if best_thumb is None:
            continue
        grouped[best_thumb].append(button)
        used.append(index)

    composed: list[ScreenXyxy] = []
    for parts in grouped:
        rect = _union_xyxy(parts)
        if any(
            _scrollbar_control_already_covers(control, rect)
            for control in scrollbar_controls
        ):
            continue
        composed.append(rect)
    return list(scrollbar_controls) + composed, used


def enumerate_edit_document_scrollbar_screen(
    hwnd: int,
) -> tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy]]:
    """One FindAll Edit|Document|ScrollBar|Thumb plus labeled text-like controls."""
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
    scrollbar_thumbs: list[ScreenXyxy] = []
    scrollbar_line_buttons: list[tuple[ScreenXyxy, str]] = []
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
            _UIA_THUMB_CONTROL_TYPE_ID,
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
                automation_id = ""
                if control_type in (
                    _UIA_THUMB_CONTROL_TYPE_ID,
                    _UIA_BUTTON_CONTROL_TYPE_ID,
                ):
                    automation_id = _element_automation_id(element)
                if control_type == _UIA_SCROLLBAR_CONTROL_TYPE_ID:
                    scrollbars.append(box)
                elif (
                    control_type == _UIA_THUMB_CONTROL_TYPE_ID
                    and automation_id == _SCROLLBAR_THUMB_AUTOMATION_ID
                ):
                    scrollbar_thumbs.append(box)
                elif (
                    control_type == _UIA_BUTTON_CONTROL_TYPE_ID
                    and automation_id in _SCROLLBAR_LINE_BUTTON_AUTOMATION_IDS
                ):
                    scrollbar_line_buttons.append((box, _element_label(element)))
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
        scrollbars, used_line_buttons = compose_scrollbar_screen_rects(
            scrollbars,
            scrollbar_thumbs,
            [box for box, _label in scrollbar_line_buttons],
        )
        for index, (box, label) in enumerate(scrollbar_line_buttons):
            if index in used_line_buttons or not label:
                continue
            labeled.append((box, label))
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



def snapshot_uia_screen(hwnd: int) -> dict[str, object] | None:
    """FindAll under ``hwnd`` and return screen-space boxes for later analysis.

    Called on the capture worker at press/release time, while the window's
    accessibility tree is still populated. Returns ``None`` when the hwnd
    cannot be queried. An empty tree is still a snapshot (analysis must not
    FindAll again).
    """
    if os.name != "nt" or not int(hwnd or 0):
        return None
    try:
        inputs, scrollbars, labeled = enumerate_edit_document_scrollbar_screen(int(hwnd))
    except Exception:
        return None
    window = _window_screen_xywh(int(hwnd))
    texts: list[dict[str, object]] = []
    for box, label in labeled:
        text = str(label or "").strip()
        if not text:
            continue
        texts.append({"bbox": [int(v) for v in box], "text": text})
    return {
        "window": [int(v) for v in window] if window is not None else None,
        "inputs": [[int(v) for v in box] for box in inputs],
        "scrollbars": [[int(v) for v in box] for box in scrollbars],
        "texts": texts,
    }


def _snapshot_xyxy_list(raw: object) -> list[ScreenXyxy]:
    if not isinstance(raw, list):
        return []
    out: list[ScreenXyxy] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            out.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
    return out


def uia_snapshot_image_local(
    snapshot: dict[str, object],
    *,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    window_screen_xywh: RectXywh | None = None,
) -> tuple[list[RectXywh] | None, list[RectXywh] | None, list[LabeledRectXywh]]:
    """Convert a click-time snapshot to image-local boxes. Does not call FindAll.

    Empty input/scrollbar lists become ``None`` so YOLO/Hough refine still runs.
    Labeled texts stay a list (possibly empty).
    """
    window_raw = snapshot.get("window")
    window = window_screen_xywh
    if isinstance(window_raw, (list, tuple)) and len(window_raw) == 4:
        window = (
            int(window_raw[0]),
            int(window_raw[1]),
            int(window_raw[2]),
            int(window_raw[3]),
        )
    if window is None:
        return None, None, []
    filter_kwargs = dict(
        window_screen_xywh=window,
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
    )
    texts_raw = snapshot.get("texts")
    labeled: list[LabeledScreenXyxy] = []
    if isinstance(texts_raw, list):
        for item in texts_raw:
            if not isinstance(item, dict):
                continue
            box = item.get("bbox")
            text = str(item.get("text") or "").strip()
            if not text or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            labeled.append(
                ((int(box[0]), int(box[1]), int(box[2]), int(box[3])), text)
            )
    inputs = filter_uia_input_rects(_snapshot_xyxy_list(snapshot.get("inputs")), **filter_kwargs)
    scrollbars = filter_uia_scrollbar_rects(
        _snapshot_xyxy_list(snapshot.get("scrollbars")), **filter_kwargs
    )
    texts = filter_uia_labeled_text_rects(labeled, **filter_kwargs)
    return (inputs or None, scrollbars or None, texts)
