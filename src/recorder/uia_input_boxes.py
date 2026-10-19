"""Live UIA Edit/Document/ScrollBar + labeled-control refine.

One FindAll under a live hwnd, split by control type, convert to image-local
xywh, drop oversized (near-window) rects, prefer non-empty UIA inputs/scrollbars
over YOLO (+ Hough for inputs), and use labeled Text/Hyperlink/ListItem/Button/
MenuItem/TreeItem/DataItem as OCR assist (split blobs, IoU fill, orphans).
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Callable

from cua_mcp.geometry import clip_box, iou_xywh, point_in_rect_xywh
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


def associate_uia_labels_with_yolo_text(
    yolo_text_boxes: list[RectXywh],
    uia_texts: list[LabeledRectXywh],
    *,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    iou_threshold: float = DEFAULT_INPUT_RECT_IOU_THRESHOLD,
) -> tuple[list[LabeledRectXywh], list[RectXywh], dict[str, object]]:
    """Split/fill/orphan UIA labels against YOLO text boxes.

    Returns ``(uia_sourced_detections, yolo_boxes_for_ocr, meta)``.
    """
    threshold = float(iou_threshold)
    uia_list = list(uia_texts)
    used_uia: set[int] = set()
    sourced: list[LabeledRectXywh] = []
    sourced_boxes: set[RectXywh] = set()
    yolo_for_ocr: list[RectXywh] = []
    split_blobs = 0
    iou_filled = 0
    orphans = 0

    def _emit(box: RectXywh, label: str) -> None:
        if box in sourced_boxes:
            return
        sourced_boxes.add(box)
        sourced.append((box, label))

    remaining_yolo: list[RectXywh] = []
    for blob in yolo_text_boxes:
        contained = [
            i
            for i, (uia_box, _label) in enumerate(uia_list)
            if i not in used_uia
            and _uia_rect_inside_blob(uia_box, blob, iou_threshold=threshold)
        ]
        if len(contained) >= 2:
            split_blobs += 1
            for i in contained:
                uia_box, label = uia_list[i]
                used_uia.add(i)
                _emit(uia_box, label)
            continue
        remaining_yolo.append(blob)

    # Greedy 1:1 IoU fill on remaining YOLO text vs unused UIA.
    pairs: list[tuple[float, int, int]] = []
    for yi, yolo in enumerate(remaining_yolo):
        for ui, (uia_box, _label) in enumerate(uia_list):
            if ui in used_uia:
                continue
            score = iou_xywh(yolo, uia_box)
            if score > threshold:
                pairs.append((score, yi, ui))
    pairs.sort(reverse=True)
    used_yolo: set[int] = set()
    for _score, yi, ui in pairs:
        if yi in used_yolo or ui in used_uia:
            continue
        used_yolo.add(yi)
        used_uia.add(ui)
        iou_filled += 1
        _emit(remaining_yolo[yi], uia_list[ui][1])

    for yi, yolo in enumerate(remaining_yolo):
        if yi not in used_yolo:
            yolo_for_ocr.append(yolo)

    for ui, (uia_box, label) in enumerate(uia_list):
        if ui in used_uia:
            continue
        ux, uy, uw, uh = (int(v) for v in uia_box)
        cx, cy = ux + uw * 0.5, uy + uh * 0.5
        in_blob = any(
            point_in_rect_xywh(cx, cy, blob, pad=0) for blob in yolo_text_boxes
        )
        in_roi = ocr_roi is not None and point_in_rect_xywh(
            cx, cy, ocr_roi, pad=int(ocr_roi_pad)
        )
        if in_blob or in_roi:
            orphans += 1
            used_uia.add(ui)
            _emit(uia_box, label)

    meta: dict[str, object] = {
        "uia_texts": len(uia_list),
        "split_blobs": split_blobs,
        "iou_filled": iou_filled,
        "orphans": orphans,
        "text_ocr_skipped": split_blobs + iou_filled,  # YOLO blobs not OCR'd via UIA
        "yolo_text_for_ocr": len(yolo_for_ocr),
        "uia_sourced": len(sourced),
    }
    return sourced, yolo_for_ocr, meta


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
