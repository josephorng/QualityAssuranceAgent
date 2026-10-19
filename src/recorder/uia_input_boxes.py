"""Live UIA Edit/Document bounds for input refine (skip Hough when non-empty).

Enumerates ControlType Edit and Document under a live hwnd, converts to
image-local xywh, drops oversized (near-window) rects, and prefers a non-empty
UIA set over YOLO+Hough (YOLO often over-detects false inputs).
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

# IUIAutomationElementArray
_UIA_ARRAY_GET_LENGTH = 3
_UIA_ARRAY_GET_ELEMENT = 4

_UIA_CONTROL_TYPE_PROPERTY_ID = 30003
_UIA_EDIT_CONTROL_TYPE_ID = 50004
_UIA_DOCUMENT_CONTROL_TYPE_ID = 50030
_TREE_SCOPE_DESCENDANTS = 0x4
_VT_I4 = 3

RectXywh = tuple[int, int, int, int]
ScreenXyxy = tuple[int, int, int, int]

UiaScreenEnumerator = Callable[[int], list[ScreenXyxy]]


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


def uia_covers_yolo_inputs(
    uia_boxes: list[RectXywh],
    yolo_input_boxes: list[RectXywh],
    *,
    iou_threshold: float = DEFAULT_INPUT_RECT_IOU_THRESHOLD,
) -> bool:
    """Diagnostic: True when every YOLO input overlaps some UIA box.

    Prefer-UIA decisions no longer require this; kept for logging / metrics.
    Empty UIA never covers. Zero YOLO with non-empty UIA counts as covered.
    """
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


def enumerate_edit_document_screen_xyxy(hwnd: int) -> list[ScreenXyxy]:
    """FindAll Edit|Document under ``hwnd``; return screen ``(l,t,r,b)`` boxes."""
    if os.name != "nt" or not hwnd:
        return []

    ole32 = ctypes.windll.ole32
    initialized_here = False
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr == _S_OK:
        initialized_here = True
    elif hr not in (_S_OK, _S_FALSE) and (hr & 0xFFFFFFFF) != 0x80010106:
        return []

    automation: ctypes.c_void_p | None = None
    root: ctypes.c_void_p | None = None
    edit_cond: ctypes.c_void_p | None = None
    doc_cond: ctypes.c_void_p | None = None
    or_cond: ctypes.c_void_p | None = None
    found: ctypes.c_void_p | None = None
    out: list[ScreenXyxy] = []
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
            return []
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
            return []
        root = root_ptr

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
        edit_ptr = ctypes.c_void_p()
        hr = create_prop(
            automation,
            _UIA_CONTROL_TYPE_PROPERTY_ID,
            _variant_i4(_UIA_EDIT_CONTROL_TYPE_ID),
            ctypes.byref(edit_ptr),
        )
        if hr != _S_OK or not edit_ptr:
            return []
        edit_cond = edit_ptr

        doc_ptr = ctypes.c_void_p()
        hr = create_prop(
            automation,
            _UIA_CONTROL_TYPE_PROPERTY_ID,
            _variant_i4(_UIA_DOCUMENT_CONTROL_TYPE_ID),
            ctypes.byref(doc_ptr),
        )
        if hr != _S_OK or not doc_ptr:
            return []
        doc_cond = doc_ptr

        create_or = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(automation, _UIA_CREATE_OR_CONDITION + 1)[_UIA_CREATE_OR_CONDITION])
        or_ptr = ctypes.c_void_p()
        hr = create_or(automation, edit_cond, doc_cond, ctypes.byref(or_ptr))
        if hr != _S_OK or not or_ptr:
            return []
        or_cond = or_ptr

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
            return []
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
            return []

        for index in range(int(length.value)):
            elem_ptr = ctypes.c_void_p()
            hr = get_element(found, index, ctypes.byref(elem_ptr))
            if hr != _S_OK or not elem_ptr:
                continue
            element = elem_ptr
            try:
                box = _element_bounding_xyxy(element)
                if box is not None:
                    out.append(box)
            finally:
                _com_release(element)
        return out
    except Exception:
        return []
    finally:
        _com_release(found)
        _com_release(or_cond)
        _com_release(doc_cond)
        _com_release(edit_cond)
        _com_release(root)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


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
    enumerator = enumerate_fn or enumerate_edit_document_screen_xyxy
    try:
        screen_boxes = enumerator(int(hwnd))
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
    """Return ``(uia_boxes, meta)`` when UIA finds ≥1 usable input; else ``(None, meta)``.

    A non-empty UIA Edit/Document set replaces YOLO inputs and skips Hough.
    Fall back to YOLO+Hough only when UIA is empty or fails. ``yolo_covered`` in
    meta is diagnostic only (whether every YOLO box overlapped a UIA rect).
    """
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
