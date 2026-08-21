"""Resolve screen coordinates for where typing is focused."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable


# Screen rect: left, top, right, bottom (inclusive edges for containment).
ScreenRect = tuple[int, int, int, int]
ScreenPoint = tuple[int, int]


@dataclass(frozen=True)
class TypingFocus:
    """Typing focus point plus the caret/UIA rect that produced it (when known)."""

    point: ScreenPoint | None
    rect: ScreenRect | None = None


_CLSCTX_INPROC_SERVER = 1
_COINIT_MULTITHREADED = 0x0
_S_OK = 0
_S_FALSE = 1

# CLSID_CUIAutomation / IID_IUIAutomation / IID_IUIAutomationElement
_CLSID_CUIAutomation = "{ff48dba4-60ef-4201-aa87-54103eef594e}"
_IID_IUIAutomation = "{30cbe57d-d9d0-452a-ab13-7ac5ac4825ee}"
_IID_IUIAutomationElement = "{d22108aa-8ac5-49a5-837b-37bbb3d2861d}"

# IUIAutomation::GetFocusedElement is vtable index 8 (after IUnknown + 5 methods).
_UIA_GET_FOCUSED_ELEMENT_INDEX = 8
# IUIAutomationElement::get_CurrentBoundingRectangle is vtable index 43.
_UIA_GET_CURRENT_BOUNDING_RECTANGLE_INDEX = 43


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


def _guid_from_string(value: str) -> _GUID:
    ole32 = ctypes.windll.ole32
    parsed = _GUID()
    hr = ole32.CLSIDFromString(wintypes.LPCWSTR(value), ctypes.byref(parsed))
    if hr != _S_OK:
        raise OSError(f"CLSIDFromString failed: 0x{hr & 0xFFFFFFFF:08X}")
    return parsed


def _rect_is_valid(rect: ScreenRect | None) -> bool:
    if rect is None:
        return False
    left, top, right, bottom = rect
    return right > left and bottom > top


def _rect_center(rect: ScreenRect) -> ScreenPoint:
    left, top, right, bottom = rect
    return ((left + right) // 2, (top + bottom) // 2)


def _point_in_rect(point: ScreenPoint, rect: ScreenRect) -> bool:
    x, y = point
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def _point_from_rect(
    rect: ScreenRect,
    *,
    last_click_xy: ScreenPoint | None,
) -> ScreenPoint:
    # Only reuse the last click when it actually landed inside the focused field.
    # Otherwise prefer the caret/UIA rect center (e.g. guest-mode click then type
    # in a newly focused omnibox on another monitor).
    if last_click_xy is not None and _point_in_rect(last_click_xy, rect):
        return last_click_xy
    return _rect_center(rect)


def _caret_screen_rect() -> ScreenRect | None:
    """Return the system caret rect in screen coordinates, or None."""
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    thread_id = user32.GetWindowThreadProcessId(hwnd, None)
    if not thread_id:
        return None

    info = _GUITHREADINFO()
    info.cbSize = ctypes.sizeof(_GUITHREADINFO)
    if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
        return None
    if not info.hwndCaret:
        return None
    client = info.rcCaret
    if client.right <= client.left or client.bottom <= client.top:
        return None

    top_left = wintypes.POINT(client.left, client.top)
    bottom_right = wintypes.POINT(client.right, client.bottom)
    if not user32.ClientToScreen(info.hwndCaret, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(info.hwndCaret, ctypes.byref(bottom_right)):
        return None
    return (int(top_left.x), int(top_left.y), int(bottom_right.x), int(bottom_right.y))


def _com_vtable(ptr: ctypes.c_void_p, size: int) -> ctypes.Array[ctypes.c_void_p]:
    return ctypes.cast(
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.POINTER(ctypes.c_void_p * size),
    ).contents


def _com_release(ptr: ctypes.c_void_p | None) -> None:
    if not ptr:
        return
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(_com_vtable(ptr, 3)[2])
    release(ptr)


def _uia_focused_screen_rect() -> ScreenRect | None:
    """Return the focused UIA element's bounding rect in screen coordinates."""
    if os.name != "nt":
        return None

    ole32 = ctypes.windll.ole32
    initialized_here = False
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr in (_S_OK, _S_FALSE):
        initialized_here = hr == _S_OK
    elif hr not in (_S_OK, _S_FALSE):
        # RPC_E_CHANGED_MODE (0x80010106): already initialized with another model.
        if (hr & 0xFFFFFFFF) != 0x80010106:
            return None

    automation: ctypes.c_void_p | None = None
    element: ctypes.c_void_p | None = None
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
            return None
        automation = automation_ptr

        get_focused = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(automation, _UIA_GET_FOCUSED_ELEMENT_INDEX + 1)[_UIA_GET_FOCUSED_ELEMENT_INDEX])
        element_ptr = ctypes.c_void_p()
        hr = get_focused(automation, ctypes.byref(element_ptr))
        if hr != _S_OK or not element_ptr:
            return None
        element = element_ptr

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
        screen = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        if not _rect_is_valid(screen):
            return None
        return screen
    except Exception:
        return None
    finally:
        _com_release(element)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def resolve_typing_focus(
    *,
    last_click_xy: ScreenPoint | None = None,
    mouse_xy: ScreenPoint | None = None,
    caret_rect_fn: Callable[[], ScreenRect | None] | None = None,
    uia_rect_fn: Callable[[], ScreenRect | None] | None = None,
) -> TypingFocus:
    """Resolve where typing is focused on screen.

    Priority:
    1. System caret rect (GetGUIThreadInfo)
    2. UIA focused-element bounds (accepted even when window-sized)
    3. Within a valid rect: last click if inside, else rect center
    4. Last resort: last click, else mouse position
    """
    caret_fn = caret_rect_fn or _caret_screen_rect
    uia_fn = uia_rect_fn or _uia_focused_screen_rect

    rect: ScreenRect | None = None
    try:
        rect = caret_fn()
    except Exception:
        rect = None
    if not _rect_is_valid(rect):
        try:
            rect = uia_fn()
        except Exception:
            rect = None

    if _rect_is_valid(rect):
        assert rect is not None
        return TypingFocus(
            point=_point_from_rect(rect, last_click_xy=last_click_xy),
            rect=rect,
        )

    if last_click_xy is not None:
        return TypingFocus(point=last_click_xy, rect=None)
    return TypingFocus(point=mouse_xy, rect=None)


def resolve_typing_screen_xy(
    *,
    last_click_xy: ScreenPoint | None = None,
    mouse_xy: ScreenPoint | None = None,
    caret_rect_fn: Callable[[], ScreenRect | None] | None = None,
    uia_rect_fn: Callable[[], ScreenRect | None] | None = None,
) -> ScreenPoint | None:
    """Resolve the typing focus point (see ``resolve_typing_focus``)."""
    return resolve_typing_focus(
        last_click_xy=last_click_xy,
        mouse_xy=mouse_xy,
        caret_rect_fn=caret_rect_fn,
        uia_rect_fn=uia_rect_fn,
    ).point
