"""Live UIA Edit/Document/ComboBox/Pane/ScrollBar + labeled-control enumeration.

Budgeted ControlView walk under a window hwnd (legacy FindAllBuildCache via
``CUA_UIA_FINDALL_LEGACY=1``). Recording and replay use the snapshot as its
own candidate list, or YOLO+OCR when that list is not enough. The two lists
are not merged.

A control-type ScrollBar is a scrollbar. A Thumb whose AutomationId is
``ScrollbarThumb`` (or Win32/WinForms Name ``Position``) is one too.
``UpButton`` / ``DownButton`` AutomationIds, or Win32 Names such as
``Line up`` / ``Line down`` / ``Column left`` / ``Column right`` (and
``Page *`` mid-track parts), on that thumb's line are included in the same
scrollbar rectangle.
"""

from __future__ import annotations

import ctypes
import heapq
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

from cua_mcp.geometry import (
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
    _bstr_to_str,
    _com_release,
    _com_vtable,
    _guid_from_string,
    _uia_current_bstr,
)

# Fraction of window area at/above which a UIA rect is treated as window-sized.
DEFAULT_UIA_OVERSIZE_AREA_FRAC = 0.8
# Input-only: drop tall/huge Pane/Edit containers that are not near full-window
# but still dominate FindAll (scrollbars use the plain oversize filter only).
DEFAULT_UIA_INPUT_CONTAINER_AREA_FRAC = 0.4
DEFAULT_UIA_INPUT_CONTAINER_WIDTH_FRAC = 0.7
DEFAULT_UIA_INPUT_CONTAINER_HEIGHT_FRAC = 0.25
# Match ``DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD`` in select_mouse_target (avoid cycle).
DEFAULT_TEXT_ELEMENT_CLUSTER_IOU_THRESHOLD = 0.5
# YOLO mostly inside UIA when intersection covers at least this fraction of YOLO area.
DEFAULT_UIA_CONTAINMENT_FRAC = 0.5
# Drop a smaller element when this fraction of its area lies inside a larger one
# (duplicate YOLO crops of the same icon).
DEFAULT_NESTED_ELEMENT_CONTAINMENT_FRAC = 0.7
# After monitor clip, drop edge slivers (wrong-monitor projection ghosts).
DEFAULT_UIA_CLIP_MIN_SIDE_PX = 2
DEFAULT_UIA_CLIP_KEEP_AREA_FRAC = 0.5

# IUIAutomation vtable (full slots including IUnknown 0..2).
_UIA_ELEMENT_FROM_HANDLE = 6
_UIA_ELEMENT_FROM_POINT = 7
_UIA_GET_CONTROL_VIEW_WALKER = 14
_UIA_CREATE_CACHE_REQUEST = 20
_UIA_CREATE_PROPERTY_CONDITION = 23
_UIA_CREATE_AND_CONDITION = 25
_UIA_CREATE_OR_CONDITION = 28

# IUIAutomationElement
_UIA_FIND_FIRST = 5
_UIA_FIND_ALL = 6
_UIA_FIND_ALL_BUILD_CACHE = 8
_UIA_GET_CURRENT_PATTERN = 16
_UIA_GET_CURRENT_CONTROL_TYPE = 21
_UIA_GET_CURRENT_NAME = 23
# IUIAutomationElement::get_CurrentAutomationId. Bounding rect is slot 43.
_UIA_GET_CURRENT_AUTOMATION_ID = 29
_UIA_GET_CURRENT_NATIVE_WINDOW_HANDLE = 36
_UIA_GET_CACHED_CONTROL_TYPE = 53
_UIA_GET_CACHED_NAME = 55
_UIA_GET_CACHED_AUTOMATION_ID = 61
_UIA_GET_CACHED_BOUNDING_RECTANGLE = 75

# IUIAutomationCacheRequest
_UIA_CACHE_REQUEST_ADD_PROPERTY = 3
_UIA_CACHE_REQUEST_PUT_TREE_SCOPE = 7

# IUIAutomationTreeWalker (IUnknown 0..2)
_UIA_TREE_WALKER_GET_PARENT_ELEMENT = 3
_UIA_TREE_WALKER_GET_FIRST_CHILD_ELEMENT = 4
_UIA_TREE_WALKER_GET_NEXT_SIBLING_ELEMENT = 6

# IUIAutomationElementArray
_UIA_ARRAY_GET_LENGTH = 3
_UIA_ARRAY_GET_ELEMENT = 4

_UIA_BOUNDING_RECTANGLE_PROPERTY_ID = 30001
_UIA_CONTROL_TYPE_PROPERTY_ID = 30003
_UIA_NAME_PROPERTY_ID = 30005
_UIA_AUTOMATION_ID_PROPERTY_ID = 30011
_UIA_BUTTON_CONTROL_TYPE_ID = 50000
_UIA_CUSTOM_CONTROL_TYPE_ID = 50001
_UIA_CHECKBOX_CONTROL_TYPE_ID = 50002
_UIA_COMBOBOX_CONTROL_TYPE_ID = 50003
_UIA_EDIT_CONTROL_TYPE_ID = 50004
_UIA_HYPERLINK_CONTROL_TYPE_ID = 50005
_UIA_LISTITEM_CONTROL_TYPE_ID = 50007
_UIA_MENUITEM_CONTROL_TYPE_ID = 50011
_UIA_RADIOBUTTON_CONTROL_TYPE_ID = 50013
_UIA_TABITEM_CONTROL_TYPE_ID = 50019
_UIA_TEXT_CONTROL_TYPE_ID = 50020
_UIA_TREEITEM_CONTROL_TYPE_ID = 50025
_UIA_THUMB_CONTROL_TYPE_ID = 50027
_UIA_SCROLLBAR_CONTROL_TYPE_ID = 50028
_UIA_DATAITEM_CONTROL_TYPE_ID = 50029
_UIA_DOCUMENT_CONTROL_TYPE_ID = 50030
_UIA_SPLITBUTTON_CONTROL_TYPE_ID = 50031
_UIA_WINDOW_CONTROL_TYPE_ID = 50032
_UIA_PANE_CONTROL_TYPE_ID = 50033
_UIA_HEADER_CONTROL_TYPE_ID = 50034
_UIA_HEADERITEM_CONTROL_TYPE_ID = 50035
# Live FindAll + OCR-clip field hosts treated as input boxes.
_UIA_INPUT_CONTROL_TYPE_IDS = (
    _UIA_EDIT_CONTROL_TYPE_ID,
    _UIA_DOCUMENT_CONTROL_TYPE_ID,
    _UIA_COMBOBOX_CONTROL_TYPE_ID,
    _UIA_PANE_CONTROL_TYPE_ID,
)
# Text hosts used to expand a caret-thin focus into an OCR clip rect.
_OCR_CLIP_FIELD_CONTROL_TYPE_IDS = _UIA_INPUT_CONTROL_TYPE_IDS
_UIA_VALUE_PATTERN = 10002
_UIA_VALUE_GET_CURRENT = 4
_TREE_SCOPE_ELEMENT = 0x1
_TREE_SCOPE_DESCENDANTS = 0x4
_VT_I4 = 3
_VT_BSTR = 8
# Idle hover FindAll: climb to a container covering at least this fraction of
# the top-level window area (or Window/Document / hwnd root).
_HOVER_SUBTREE_AREA_FRAC = 0.15
_HOVER_SUBTREE_MAX_CLIMB = 12

# Budgeted ControlView walk (replaces unbounded FindAllBuildCache).
_DEFAULT_WALK_MAX_HITS = 500
_DEFAULT_WALK_MAX_NODES = 4000
_DEFAULT_WALK_MAX_SECONDS = 2.0
_DEFAULT_WALK_MAX_CHILDREN_PER_PARENT = 80
_WALK_OFFSCREEN_SCORE_BASE = 1_000_000_000.0
_WALK_MISSING_BBOX_SCORE = 1_000_000_000_000.0

# Sparse FindAll → cursor refetch / EFP ROI enrich (recording).
_SPARSE_LABELED_TEXT_FLOOR = 8
_EFP_SCAN_ROI_HALF_PX = 300
_EFP_SCAN_STEP_PX = 28
_EFP_SCAN_MAX_PROBES = 80
_EFP_SCAN_MAX_SECONDS = 1.0
_SNAPSHOT_UIA_ENRICH_TIMEOUT_S = 5.5


@dataclass(frozen=True)
class WalkBudgets:
    """Concurrent stop limits for the budgeted ControlView walk (first trip wins)."""

    max_hits: int = _DEFAULT_WALK_MAX_HITS
    max_nodes: int = _DEFAULT_WALK_MAX_NODES
    max_seconds: float = _DEFAULT_WALK_MAX_SECONDS
    max_children_per_parent: int = _DEFAULT_WALK_MAX_CHILDREN_PER_PARENT


@dataclass(frozen=True)
class WalkResult:
    """Outcome counters for a budgeted walk (hits may be partial when truncated)."""

    hits: int
    nodes: int
    elapsed_s: float
    truncated: bool
    stop_reason: str


_TNode = TypeVar("_TNode")

_SCROLLBAR_THUMB_AUTOMATION_ID = "ScrollbarThumb"
_SCROLLBAR_LINE_BUTTON_AUTOMATION_IDS = frozenset({"UpButton", "DownButton"})
# Win32/WinForms accessible Names when AutomationId is empty / non-WPF.
_SCROLLBAR_THUMB_NAMES = frozenset({"position"})
_SCROLLBAR_LINE_BUTTON_NAMES = frozenset(
    {
        "line up",
        "line down",
        "line left",
        "line right",
        "column left",
        "column right",
        "page up",
        "page down",
        "page left",
        "page right",
    }
)
# Same-line when the shared span covers this fraction of the shorter cross-axis side.
_SCROLLBAR_LINE_OVERLAP_FRAC = 0.5
# A thumb-built rect inside a ScrollBar control is a duplicate when the control
# is at most this many times larger than the thumb rect.
_SCROLLBAR_CONTROL_CONTAIN_AREA_RATIO = 4.0


def is_uia_scrollbar_thumb_part(
    *,
    automation_id: str | None = None,
    name: str | None = None,
) -> bool:
    """True for WPF ``ScrollbarThumb`` or Win32/WinForms Name ``Position``."""
    auto = str(automation_id or "").strip()
    if auto == _SCROLLBAR_THUMB_AUTOMATION_ID:
        return True
    return str(name or "").strip().casefold() in _SCROLLBAR_THUMB_NAMES


def is_uia_scrollbar_line_button_part(
    *,
    automation_id: str | None = None,
    name: str | None = None,
) -> bool:
    """True for WPF Up/DownButton or Win32 Line/Column/Page button Names."""
    auto = str(automation_id or "").strip()
    if auto in _SCROLLBAR_LINE_BUTTON_AUTOMATION_IDS:
        return True
    return str(name or "").strip().casefold() in _SCROLLBAR_LINE_BUTTON_NAMES


def is_winforms_scrollbar_thumb_efp_name(name: str | None) -> bool:
    """True when an EFP leaf Name should seed a scrollbar peer (``Position``)."""
    return str(name or "").strip().casefold() in _SCROLLBAR_THUMB_NAMES

# DataItem (Excel cells, dense grids) is intentionally omitted: FindAll under
# XLMAIN would return every visible cell and block settle/hover for tens of
# seconds. Recording/replay name cells from ElementFromPoint instead
# (``control_type=data_item`` stays in ``UIA_CONTROL_TYPE_SUFFIX``).
_LABELED_TEXT_CONTROL_TYPES = frozenset(
    {
        _UIA_BUTTON_CONTROL_TYPE_ID,
        _UIA_CUSTOM_CONTROL_TYPE_ID,
        _UIA_CHECKBOX_CONTROL_TYPE_ID,
        _UIA_HYPERLINK_CONTROL_TYPE_ID,
        _UIA_LISTITEM_CONTROL_TYPE_ID,
        _UIA_MENUITEM_CONTROL_TYPE_ID,
        _UIA_RADIOBUTTON_CONTROL_TYPE_ID,
        _UIA_TABITEM_CONTROL_TYPE_ID,
        _UIA_TEXT_CONTROL_TYPE_ID,
        _UIA_TREEITEM_CONTROL_TYPE_ID,
        _UIA_SPLITBUTTON_CONTROL_TYPE_ID,
        _UIA_HEADER_CONTROL_TYPE_ID,
        _UIA_HEADERITEM_CONTROL_TYPE_ID,
    }
)

RectXywh = tuple[int, int, int, int]
ScreenXyxy = tuple[int, int, int, int]
# box, visible label, control-type token ("" when the snapshot did not record one).
LabeledScreenXyxy = tuple[ScreenXyxy, str, str]
LabeledRectXywh = tuple[RectXywh, str, str]

# Tokens stored on snapshot texts. Chinese suffixes are the instruction words.
UIA_CONTROL_TYPE_SUFFIX: dict[str, str] = {
    "button": "按鈕",
    "custom": "自訂",
    "check_box": "核取方塊",
    "hyperlink": "超連結",
    "list_item": "清單項目",
    "menu_item": "功能表項目",
    "radio_button": "選項按鈕",
    "tab_item": "索引標籤",
    "text": "文字",
    "tree_item": "樹狀項目",
    "data_item": "資料項目",
    "split_button": "分割按鈕",
    "header": "標頭",
    "header_item": "標頭項目",
}

UiaScreenEnumerator = Callable[[int], list[ScreenXyxy]]
# (inputs, scrollbars, labeled_texts, input_labels)
UiaCombinedScreenEnumerator = Callable[
    [int],
    tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]],
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


def _variant_bstr(value: str) -> tuple[VARIANT, ctypes.c_void_p | None]:
    """Build a VT_BSTR VARIANT; caller must ``SysFreeString`` the returned ptr."""
    oleaut32 = ctypes.windll.oleaut32
    oleaut32.SysAllocString.argtypes = [wintypes.LPCWSTR]
    oleaut32.SysAllocString.restype = ctypes.c_void_p
    bstr = oleaut32.SysAllocString(wintypes.LPCWSTR(str(value)))
    var = VARIANT()
    var.vt = _VT_BSTR
    var._.pv = bstr
    return var, bstr


def _sys_free_string(bstr: ctypes.c_void_p | None) -> None:
    if not bstr:
        return
    try:
        ctypes.windll.oleaut32.SysFreeString(ctypes.c_void_p(int(bstr)))
    except Exception:
        pass


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


def is_uia_input_container_rect(
    rect_xywh: RectXywh,
    window_xywh: RectXywh,
    *,
    max_area_frac: float = DEFAULT_UIA_INPUT_CONTAINER_AREA_FRAC,
    min_width_frac: float = DEFAULT_UIA_INPUT_CONTAINER_WIDTH_FRAC,
    min_height_frac: float = DEFAULT_UIA_INPUT_CONTAINER_HEIGHT_FRAC,
) -> bool:
    """True for tall/huge input hosts that are not single-line fields.

    Wide thin edits (e.g. ``w=1348, h=21``) stay. Full-height scrollbars are
    filtered separately and must not use this helper.
    """
    _x, _y, w, h = (int(v) for v in rect_xywh)
    _wx, _wy, ww, wh = (int(v) for v in window_xywh)
    if w <= 0 or h <= 0 or ww <= 0 or wh <= 0:
        return True
    area_frac = (float(w) * float(h)) / float(ww * wh)
    if area_frac >= float(max_area_frac):
        return True
    return (
        float(w) / float(ww) >= float(min_width_frac)
        and float(h) / float(wh) >= float(min_height_frac)
    )


def compose_uia_input_landmark_label(name: str, automation_id: str) -> str:
    """Combine FindAll Name + AutomationId into a replay landmark label.

    When Name already embeds ``|`` segments (e.g. recorded EFP shape
    ``傳真號碼|Textbox|txt_Fax``), keep it as-is if AutomationId is already
    present. Otherwise append AutomationId via ``Name|AutomationId``.
    """
    name_c = canonicalize_uia_shell_status_label(name)
    auto_c = canonicalize_uia_shell_status_label(automation_id)
    if not auto_c:
        return name_c
    if not name_c:
        return auto_c
    if auto_c == name_c or name_c.endswith("|" + auto_c):
        return name_c
    if auto_c in name_c.split("|"):
        return name_c
    return uia_label_combine_name_value(name_c, auto_c)



MAX_USABLE_UIA_LABEL_LEN = 48

# Windows shell status tails on taskbar / Explorer Names (running count, pinned).
# Applied as trailing suffixes only so app titles like ``日月晶耀 - 晶鑰系統`` stay.
_UIA_SHELL_STATUS_SUFFIX_RE = re.compile(
    r"(?:"
    r"\s*-\s*\d+\s*個執行中視窗"
    r"|\s*-\s*\d+\s*running\s+windows?"
    r"|\s*-\s*已釘選"
    r"|\s*-\s*Pinned"
    r"|\s*[\(（]\s*已釘選\s*[\)）]"
    r"|\s*[\(（]\s*Pinned\s*[\)）]"
    r")+$",
    re.IGNORECASE,
)


def canonicalize_uia_shell_status_label(text: str | None) -> str:
    """Drop variable Windows shell status from a UIA Name/Value.

    Taskbar buttons flip between forms such as ``Excel - 已釘選``,
    ``Chrome - 2 個執行中視窗``, and ``Chrome - 1 個執行中視窗 - 已釘選``.
    Explorer pinned items use ``文件 (已釘選)``. Those tails are not stable
    across sessions, so recording and replay both reduce to the app/item name.
    """
    visible = str(text or "").strip()
    if not visible:
        return ""
    previous = None
    while previous != visible:
        previous = visible
        visible = _UIA_SHELL_STATUS_SUFFIX_RE.sub("", visible).strip()
        visible = visible.rstrip(" \t-–—－").strip()
    return visible


def control_type_token_from_id(control_type_id: int | None) -> str:
    """Stable token for a UIA control type, or ``""`` when unmapped.

    Edit/Document/ComboBox/Pane map to ``\"input\"`` so press-time EFP can
    mark field hosts (FindAll still enumerates them as geometry-only inputs).
    """
    if control_type_id is None:
        return ""
    return {
        _UIA_BUTTON_CONTROL_TYPE_ID: "button",
        _UIA_CUSTOM_CONTROL_TYPE_ID: "custom",
        _UIA_CHECKBOX_CONTROL_TYPE_ID: "check_box",
        _UIA_HYPERLINK_CONTROL_TYPE_ID: "hyperlink",
        _UIA_LISTITEM_CONTROL_TYPE_ID: "list_item",
        _UIA_MENUITEM_CONTROL_TYPE_ID: "menu_item",
        _UIA_RADIOBUTTON_CONTROL_TYPE_ID: "radio_button",
        _UIA_TABITEM_CONTROL_TYPE_ID: "tab_item",
        _UIA_TEXT_CONTROL_TYPE_ID: "text",
        _UIA_TREEITEM_CONTROL_TYPE_ID: "tree_item",
        _UIA_DATAITEM_CONTROL_TYPE_ID: "data_item",
        _UIA_SPLITBUTTON_CONTROL_TYPE_ID: "split_button",
        _UIA_HEADER_CONTROL_TYPE_ID: "header",
        _UIA_HEADERITEM_CONTROL_TYPE_ID: "header_item",
        _UIA_EDIT_CONTROL_TYPE_ID: "input",
        _UIA_DOCUMENT_CONTROL_TYPE_ID: "input",
        _UIA_COMBOBOX_CONTROL_TYPE_ID: "input",
        _UIA_PANE_CONTROL_TYPE_ID: "input",
    }.get(int(control_type_id), "")


def control_token_for_suffix(suffix: str | None) -> str | None:
    """Replay filter token for a UIA class suffix.

    ``文字`` is not Text-only: older instructions used that word for every
    labeled control. Non-text suffixes map to their token.
    """
    word = str(suffix or "").strip()
    if not word or word == "文字":
        return None
    for token, suffix_word in UIA_CONTROL_TYPE_SUFFIX.items():
        if token != "text" and suffix_word == word:
            return token
    return None


def unpack_uia_labeled(item: object) -> tuple[tuple[int, int, int, int], str, str]:
    """Split a labeled UIA row. A missing control type stays ``""``."""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return (0, 0, 0, 0), "", ""
    box = item[0]
    text = str(item[1] or "")
    control_type = str(item[2] or "").strip() if len(item) >= 3 else ""
    if control_type not in UIA_CONTROL_TYPE_SUFFIX:
        control_type = ""
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            rect = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        except (TypeError, ValueError):
            rect = (0, 0, 0, 0)
    else:
        rect = (0, 0, 0, 0)
    return rect, text, control_type


def is_usable_uia_landmark_label(text: str | None) -> bool:
    """True when a UIA string is short enough to name a control.

    URL-like names and ultra-long AutomationIds are omitted from the UIA
    candidate list and do not count toward the sufficiency gate. Shell status
    tails are stripped before the length check so a long `` - N 個執行中視窗``
    suffix does not hide an otherwise usable app name.
    """
    visible = canonicalize_uia_shell_status_label(text)
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


# Same floor as replay FindAll OCR-similarity picks (Search list-item Names, …).
OCR_UIA_LABEL_SOFT_AGREE_THRESHOLD = 0.25


def ocr_uia_labels_soft_agree(ocr_text: str | None, uia_text: str | None) -> bool:
    """True when OCR and UIA hub labels are the same target (exact or soft match).

    Soft match uses ``_label_similarity`` so short OCR hubs like ``Google Chrome``
    agree with longer FindAll Accessibility Names such as
    ``Google Chrome, 應用程式, 按右鍵以檢視選項``.
    """
    left = str(ocr_text or "").strip()
    right = str(uia_text or "").strip()
    if not left or not right:
        return False
    if left.casefold() == right.casefold():
        return True
    from cua_mcp.select_mouse_target import _label_similarity

    return float(_label_similarity(left, right)) >= float(
        OCR_UIA_LABEL_SOFT_AGREE_THRESHOLD
    )


# Host / chrome leaves that must not replace a FindAll hub label (EFP channel
# only). Keep in sync with repair promote gates.
_WEAK_EFP_HUB_LABELS = frozenset(
    {
        "coreinput",
        "core window",
        "corewindow",
    }
)


def is_weak_efp_hub_label(text: str | None) -> bool:
    """True when an EFP leaf name is too generic to become the hub instruction.

    FindAll labels (e.g. ``Launch 檔案總管``) stay the hub; weak EFP names like
    ``CoreInput`` remain on the ``name``/``value`` side-car for verify only.
    """
    visible = canonicalize_uia_shell_status_label(text)
    if not visible:
        return True
    lowered = visible.casefold()
    if lowered in _WEAK_EFP_HUB_LABELS:
        return True
    if lowered.startswith("windows.ui."):
        return True
    return False


def uia_label_combine_name_value(name: str, value: str) -> str:
    """Build a landmark label from UIA Name and ValuePattern CurrentValue.

    Both non-empty and different → ``Name|Value``. Equal → one copy. Only one
    present → that side. Canonicalizes shell status tails on each side first.
    """
    name_c = canonicalize_uia_shell_status_label(name)
    value_c = canonicalize_uia_shell_status_label(value)
    if name_c and value_c:
        if name_c == value_c:
            return name_c
        return f"{name_c}|{value_c}"
    return name_c or value_c


def split_uia_name_value_label(label: str | None) -> tuple[str, str]:
    """Split a combined ``Name|Value`` landmark into Name and Value.

    Inverse of :func:`uia_label_combine_name_value`: the first ``|`` separates
    Name from Value; Value may contain further ``|`` (e.g.
    ``電腦資產|LeftMenu|Asset_PC``). No ``|`` → ``(label, "")``. Each side is
    shell-status-canonicalized after the split (same as combine's inputs).
    """
    visible = str(label or "").strip()
    if not visible:
        return "", ""
    if "|" not in visible:
        return canonicalize_uia_shell_status_label(visible), ""
    name, _sep, value = visible.partition("|")
    return (
        canonicalize_uia_shell_status_label(name),
        canonicalize_uia_shell_status_label(value),
    )


def uia_label_prefer_value(value: str, name: str) -> str:
    """Prefer non-empty stripped Value over Name.

    Prefer :func:`uia_label_combine_name_value` for new landmark labeling so
    FindAll and ElementFromPoint stay in sync.
    """
    stripped_value = str(value or "").strip()
    if stripped_value:
        return canonicalize_uia_shell_status_label(stripped_value)
    return canonicalize_uia_shell_status_label(name)


def resolve_usable_uia_probe_label(
    *,
    preferred: str = "",
    name: str = "",
    value: str = "",
) -> str:
    """Probe label without the landmark usability gate.

    When ``name`` / ``value`` are present, compose via
    :func:`uia_label_combine_name_value`. Otherwise ``preferred`` is used
    (legacy 2-/3-tuple probes). Empty strings still return ``""``.
    """
    if str(name or "").strip() or str(value or "").strip():
        return str(uia_label_combine_name_value(name, value) or "").strip()
    return str(canonicalize_uia_shell_status_label(preferred) or "").strip()


# EFP leaves that can green the recording LED / open the UIA analysis path
# without a covering FindAll walk (cells + concrete labeled chrome).
_EFP_USABLE_LABELED_LEAF_CONTROL_TYPES = frozenset(
    {
        "data_item",
        "button",
        "tab_item",
        "list_item",
        "menu_item",
        "hyperlink",
        "split_button",
        "check_box",
        "radio_button",
        "text",
        "custom",
        "header",
        "header_item",
    }
)


def efp_payload_screen_bbox_xywh(
    payload: dict[str, Any] | None,
) -> tuple[int, int, int, int] | None:
    """Parse EFP ``bbox`` as screen xywh, or ``None`` when missing/invalid."""
    if not isinstance(payload, dict):
        return None
    raw_bbox = payload.get("bbox")
    if not (isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4):
        return None
    try:
        sx, sy, sw, sh = (int(v) for v in raw_bbox[:4])
    except (TypeError, ValueError):
        return None
    if sw <= 0 or sh <= 0:
        return None
    return (sx, sy, sw, sh)


def efp_payload_is_usable_labeled_leaf(payload: dict[str, Any] | None) -> bool:
    """True when idle/press EFP alone can green the UIA LED for a labeled leaf.

    Covers Excel ``data_item`` cells and concrete chrome (button/tab/…) that
    analysis injects as ``uia_efp`` peers. Rejects weak hubs and WinForms
    ``Position`` thumbs.
    """
    if not isinstance(payload, dict):
        return False
    control_type = str(payload.get("control_type") or "").strip()
    if control_type not in _EFP_USABLE_LABELED_LEAF_CONTROL_TYPES:
        return False
    name = str(payload.get("name") or "").strip()
    if is_winforms_scrollbar_thumb_efp_name(name):
        return False
    label = resolve_usable_uia_probe_label(
        name=name,
        value=str(payload.get("value") or "").strip(),
    )
    if (
        not label
        or not is_usable_uia_landmark_label(label)
        or is_weak_efp_hub_label(label)
    ):
        return False
    return efp_payload_screen_bbox_xywh(payload) is not None


def efp_payload_is_usable_data_item(payload: dict[str, Any] | None) -> bool:
    """True when idle/press EFP alone can green the UIA LED for a grid cell.

    DataItem is excluded from FindAll (Excel cell blow-up); a usable
    ``control_type=data_item`` leaf with bbox is enough for readiness / path.
    Rejects weak hubs and WinForms ``Position`` thumbs.
    """
    if not isinstance(payload, dict):
        return False
    if str(payload.get("control_type") or "").strip() != "data_item":
        return False
    return efp_payload_is_usable_labeled_leaf(payload)


def _intersect_local_box(
    lx: int,
    ly: int,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
) -> RectXywh | None:
    """True image-local intersection; ``None`` when the box misses the frame."""
    x1 = max(0, int(lx))
    y1 = max(0, int(ly))
    x2 = min(int(image_w), int(lx) + int(w))
    y2 = min(int(image_h), int(ly) + int(h))
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return None
    return (x1, y1, iw, ih)


def _is_clipped_uia_sliver(
    *,
    orig_w: int,
    orig_h: int,
    clipped_w: int,
    clipped_h: int,
    min_side_px: int = DEFAULT_UIA_CLIP_MIN_SIDE_PX,
    keep_area_frac: float = DEFAULT_UIA_CLIP_KEEP_AREA_FRAC,
) -> bool:
    """True when clip left a tiny edge remnant of a larger off-monitor box."""
    ow, oh = int(orig_w), int(orig_h)
    cw, ch = int(clipped_w), int(clipped_h)
    if cw <= 0 or ch <= 0:
        return True
    min_side = int(min_side_px)
    # Shrunk into a hairline on an edge of a larger control.
    if (ow > min_side and cw <= min_side) or (oh > min_side and ch <= min_side):
        return True
    orig_area = float(max(0, ow) * max(0, oh))
    if orig_area <= 0.0:
        return True
    clipped_area = float(cw * ch)
    if cw == ow and ch == oh:
        return False
    return clipped_area / orig_area < float(keep_area_frac)


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
    drop_container_inputs: bool = False,
) -> list[RectXywh]:
    """Convert screen xyxy → image-local xywh; drop empty, oversized, out-of-ROI.

    When ``drop_container_inputs`` is True, also drop tall/huge Pane/Edit
    containers (not used for scrollbars).
    """
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
        clipped = _intersect_local_box(
            local[0], local[1], local[2], local[3], image_w, image_h
        )
        if clipped is None:
            continue
        x, y, w, h = clipped
        if _is_clipped_uia_sliver(
            orig_w=local[2],
            orig_h=local[3],
            clipped_w=w,
            clipped_h=h,
        ):
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
        if drop_container_inputs and is_uia_input_container_rect(box, window_local):
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


def filter_uia_scrollbar_rects(
    screen_xyxy: list[ScreenXyxy],
    *,
    window_screen_xywh: RectXywh,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    max_area_frac: float = DEFAULT_UIA_OVERSIZE_AREA_FRAC,
    drop_container_inputs: bool = False,
) -> list[RectXywh]:
    """Scrollbar filter: same as inputs but never drops tall container shapes."""
    del drop_container_inputs  # scrollbars may span the full window height
    return filter_uia_input_rects(
        screen_xyxy,
        window_screen_xywh=window_screen_xywh,
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
        max_area_frac=max_area_frac,
        drop_container_inputs=False,
    )


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
    boxes_only = [unpack_uia_labeled(item)[0] for item in screen_labeled]
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
    label_by_box: dict[RectXywh, tuple[str, str]] = {}
    for item in screen_labeled:
        (left, top, right, bottom), label, control_type = unpack_uia_labeled(item)
        text = canonicalize_uia_shell_status_label(label)
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
        clipped = _intersect_local_box(
            local[0], local[1], local[2], local[3], image_w, image_h
        )
        if clipped is None:
            continue
        x, y, w, h = clipped
        label_by_box[(x, y, w, h)] = (text, control_type)
    out: list[LabeledRectXywh] = []
    seen: set[RectXywh] = set()
    for box in filtered:
        paired = label_by_box.get(box)
        if paired is None or box in seen:
            continue
        text, control_type = paired
        if not text:
            continue
        seen.add(box)
        out.append((box, text, control_type))
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


def _element_bounding_xyxy(
    element: ctypes.c_void_p,
    *,
    cached: bool = False,
) -> ScreenXyxy | None:
    slot = (
        _UIA_GET_CACHED_BOUNDING_RECTANGLE
        if cached
        else _UIA_GET_CURRENT_BOUNDING_RECTANGLE_INDEX
    )
    get_bounds = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT),
    )(_com_vtable(element, slot + 1)[slot])
    rect = wintypes.RECT()
    hr = get_bounds(element, ctypes.byref(rect))
    if hr != _S_OK:
        return None
    if int(rect.right) <= int(rect.left) or int(rect.bottom) <= int(rect.top):
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _element_control_type(
    element: ctypes.c_void_p,
    *,
    cached: bool = False,
) -> int | None:
    slot = (
        _UIA_GET_CACHED_CONTROL_TYPE if cached else _UIA_GET_CURRENT_CONTROL_TYPE
    )
    get_ct = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )(_com_vtable(element, slot + 1)[slot])
    control_type = ctypes.c_int(0)
    hr = get_ct(element, ctypes.byref(control_type))
    if hr != _S_OK:
        return None
    return int(control_type.value)


def _element_automation_id(
    element: ctypes.c_void_p,
    *,
    cached: bool = False,
) -> str:
    try:
        slot = (
            _UIA_GET_CACHED_AUTOMATION_ID
            if cached
            else _UIA_GET_CURRENT_AUTOMATION_ID
        )
        return _uia_current_bstr(element, slot).strip()
    except Exception:
        return ""


def _element_current_name(
    element: ctypes.c_void_p,
    *,
    cached: bool = False,
) -> str:
    try:
        slot = _UIA_GET_CACHED_NAME if cached else _UIA_GET_CURRENT_NAME
        return _uia_current_bstr(element, slot)
    except Exception:
        return ""


def _element_native_hwnd(element: ctypes.c_void_p) -> int:
    getter = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(element, _UIA_GET_CURRENT_NATIVE_WINDOW_HANDLE + 1)[
            _UIA_GET_CURRENT_NATIVE_WINDOW_HANDLE
        ]
    )
    handle = ctypes.c_void_p()
    hr = getter(element, ctypes.byref(handle))
    if hr != _S_OK or not handle:
        return 0
    try:
        return int(handle.value or 0)
    except (TypeError, ValueError):
        return 0


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


def _element_label_parts(
    element: ctypes.c_void_p,
    *,
    cached_name: bool = False,
) -> tuple[str, str, str]:
    """Combined Name|Value label, raw Name, and ValuePattern CurrentValue."""
    name = _element_current_name(element, cached=cached_name)
    value = _element_value_pattern_current(element)
    return uia_label_combine_name_value(name, value), name, value


def _element_label(element: ctypes.c_void_p, *, cached_name: bool = False) -> str:
    return _element_label_parts(element, cached_name=cached_name)[0]


_findall_timing_samples: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "findall_timing_samples",
    default=None,
)


@contextmanager
def bind_findall_timing() -> Iterator[list[dict[str, Any]]]:
    """Accumulate ``findall_s`` / ``findall_count`` for each UIA enumerate walk."""
    samples: list[dict[str, Any]] = []
    token = _findall_timing_samples.set(samples)
    try:
        yield samples
    finally:
        _findall_timing_samples.reset(token)


def _note_findall_timing(
    *,
    elapsed_s: float,
    count: int,
    truncated: bool = False,
    stop_reason: str = "",
) -> None:
    samples = _findall_timing_samples.get()
    if samples is not None:
        row: dict[str, Any] = {
            "findall_s": round(float(elapsed_s), 3),
            "findall_count": int(count),
            "walk_truncated": bool(truncated),
        }
        reason = str(stop_reason or "").strip()
        if reason:
            row["walk_stop_reason"] = reason
        samples.append(row)


def summarize_findall_timing(
    samples: list[dict[str, Any]] | None,
) -> tuple[float, int, int]:
    """Return ``(total_seconds, call_count, total_hits)`` for FindAll samples."""
    if not samples:
        return 0.0, 0, 0
    total_s = 0.0
    total_hits = 0
    for row in samples:
        try:
            total_s += float(row.get("findall_s") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            total_hits += int(row.get("findall_count") or 0)
        except (TypeError, ValueError):
            pass
    return float(total_s), int(len(samples)), int(total_hits)


def attach_findall_timing(
    snapshot: dict[str, Any] | None,
    samples: list[dict[str, Any]] | None,
    *,
    wall_s: float | None = None,
) -> dict[str, Any] | None:
    """Stamp ``findall_s`` / hit counters onto a UIA snapshot dict.

    Prefers summed walk samples when present; otherwise falls back to ``wall_s``
    so seed/settle/hover always persist an elapsed time even when the enumerate
    path did not emit samples.
    """
    if not isinstance(snapshot, dict):
        return snapshot
    total_s, calls, hits = summarize_findall_timing(samples)
    if calls <= 0:
        if wall_s is None:
            return snapshot
        try:
            total_s = float(wall_s)
        except (TypeError, ValueError):
            return snapshot
        hits = 0
        for key in ("texts", "inputs", "scrollbars"):
            raw = snapshot.get(key)
            if isinstance(raw, list):
                hits += len(raw)
    out = dict(snapshot)
    out["findall_s"] = round(float(total_s), 3)
    out["findall_count"] = int(hits)
    truncated = any(
        bool(row.get("walk_truncated"))
        for row in (samples or ())
        if isinstance(row, dict)
    )
    if truncated:
        out["walk_truncated"] = True
        reasons = [
            str(row.get("walk_stop_reason") or "").strip()
            for row in (samples or ())
            if isinstance(row, dict)
        ]
        reasons = [reason for reason in reasons if reason]
        if reasons:
            out["walk_stop_reason"] = reasons[-1]
    return out


def _snapshot_window_xywh(snapshot: dict[str, Any] | None) -> RectXywh | None:
    if not isinstance(snapshot, dict):
        return None
    raw = snapshot.get("window")
    if not (isinstance(raw, (list, tuple)) and len(raw) == 4):
        return None
    try:
        return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
    except (TypeError, ValueError):
        return None


def _point_in_screen_xyxy(x: int, y: int, box: ScreenXyxy) -> bool:
    return int(box[0]) <= int(x) < int(box[2]) and int(box[1]) <= int(y) < int(box[3])


def _text_row_screen_xyxy(row: object) -> ScreenXyxy | None:
    if not isinstance(row, dict):
        return None
    box = row.get("bbox")
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def usable_covering_snapshot_text_labels(
    snapshot: dict[str, Any] | None,
    screen_xy: tuple[int, int] | None,
) -> list[str]:
    """Usable non-oversized ``texts`` labels whose bbox covers ``screen_xy``.

    Ignores pane ``inputs`` — huge WinForms panels must not count as cover.
    """
    if not isinstance(snapshot, dict) or screen_xy is None:
        return []
    texts = snapshot.get("texts")
    if not isinstance(texts, list):
        return []
    try:
        px, py = int(screen_xy[0]), int(screen_xy[1])
    except (TypeError, ValueError, IndexError):
        return []
    window = _snapshot_window_xywh(snapshot)
    labels: list[str] = []
    for row in texts:
        if not isinstance(row, dict):
            continue
        label = str(row.get("text") or "").strip()
        if not is_usable_uia_landmark_label(label) or is_weak_efp_hub_label(label):
            continue
        xyxy = _text_row_screen_xyxy(row)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        xywh = (x1, y1, x2 - x1, y2 - y1)
        if window is not None and is_oversized_uia_rect(xywh, window):
            continue
        if _point_in_screen_xyxy(px, py, xyxy):
            labels.append(label)
    return labels


def count_usable_snapshot_text_labels(snapshot: dict[str, Any] | None) -> int:
    """Count unique usable non-weak labeled ``texts`` rows (not pane inputs)."""
    if not isinstance(snapshot, dict):
        return 0
    texts = snapshot.get("texts")
    if not isinstance(texts, list):
        return 0
    seen: set[str] = set()
    for row in texts:
        if not isinstance(row, dict):
            continue
        label = str(row.get("text") or "").strip()
        if not is_usable_uia_landmark_label(label) or is_weak_efp_hub_label(label):
            continue
        key = canonicalize_uia_shell_status_label(label).casefold()
        if key:
            seen.add(key)
    return len(seen)


def uia_snapshot_has_rich_labels(
    snapshot: dict[str, Any] | None,
    *,
    min_labels: int = _SPARSE_LABELED_TEXT_FLOOR,
) -> bool:
    """True when FindAll ``texts`` alone can supply nearby landmarks."""
    return count_usable_snapshot_text_labels(snapshot) >= int(min_labels)


def efp_cover_label_from_payload(payload: dict[str, Any] | None) -> str:
    """Usable EFP landmark label for the cover gate, or ``\"\"`` when unusable."""
    if not isinstance(payload, dict):
        return ""
    label = resolve_usable_uia_probe_label(
        name=str(payload.get("name") or "").strip(),
        value=str(payload.get("value") or "").strip(),
    )
    if (
        not label
        or not is_usable_uia_landmark_label(label)
        or is_weak_efp_hub_label(label)
    ):
        return ""
    return label


def snapshot_efp_label_covered(
    snapshot: dict[str, Any] | None,
    screen_xy: tuple[int, int] | None,
    efp_label: str,
) -> bool:
    """True when a usable FindAll text under ``screen_xy`` soft-agrees with EFP."""
    needle = str(efp_label or "").strip()
    if not needle:
        return False
    covering = usable_covering_snapshot_text_labels(snapshot, screen_xy)
    if not covering:
        return False
    return any(ocr_uia_labels_soft_agree(label, needle) for label in covering)


def uia_snapshot_covers_efp_at_point(
    snapshot: dict[str, Any] | None,
    screen_xy: tuple[int, int] | None,
    *,
    efp_payload: dict[str, Any] | None = None,
    efp_label: str = "",
) -> bool:
    """Hover/settle cover: agreeing labeled text under the cursor (not panes).

    When ``efp_label`` / payload yields a usable EFP label, require soft-agree.
    When no EFP label is available, any usable covering text counts.
    """
    label = str(efp_label or "").strip() or efp_cover_label_from_payload(efp_payload)
    if label:
        return snapshot_efp_label_covered(snapshot, screen_xy, label)
    return bool(usable_covering_snapshot_text_labels(snapshot, screen_xy))


def _merge_findall_timing_fields(
    base: dict[str, Any],
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Sum ``findall_s`` / counts when replacing a snapshot after refetch."""
    out = dict(base)
    if not isinstance(extra, dict):
        return out
    try:
        base_s = float(out.get("findall_s") or 0.0)
    except (TypeError, ValueError):
        base_s = 0.0
    try:
        extra_s = float(extra.get("findall_s") or 0.0)
    except (TypeError, ValueError):
        extra_s = 0.0
    out["findall_s"] = round(base_s + extra_s, 3)
    try:
        base_n = int(out.get("findall_count") or 0)
    except (TypeError, ValueError):
        base_n = 0
    try:
        extra_n = int(extra.get("findall_count") or 0)
    except (TypeError, ValueError):
        extra_n = 0
    # Prefer the replacement walk's leaf count when it found more labels.
    out["findall_count"] = max(base_n, extra_n)
    if extra.get("walk_truncated"):
        out["walk_truncated"] = True
        reason = str(extra.get("walk_stop_reason") or "").strip()
        if reason:
            out["walk_stop_reason"] = reason
    return out


def enrich_uia_snapshot_with_efp_scan(
    snapshot: dict[str, Any] | None,
    *,
    cursor_xy: tuple[int, int],
    max_probes: int = _EFP_SCAN_MAX_PROBES,
    max_seconds: float = _EFP_SCAN_MAX_SECONDS,
    step_px: int = _EFP_SCAN_STEP_PX,
    roi_half_px: int = _EFP_SCAN_ROI_HALF_PX,
) -> dict[str, Any] | None:
    """Merge unique EFP leaves from a cursor-centered grid into ``texts``.

    Must run on the UIA worker thread (uses ``uia_text_element_at_point``).
    """
    if not isinstance(snapshot, dict):
        return snapshot
    try:
        cx, cy = int(cursor_xy[0]), int(cursor_xy[1])
    except (TypeError, ValueError, IndexError):
        return snapshot
    window = _snapshot_window_xywh(snapshot)
    if window is None:
        return snapshot
    wx, wy, ww, wh = window
    if ww <= 0 or wh <= 0:
        return snapshot
    half = max(1, int(roi_half_px))
    step = max(8, int(step_px))
    x0 = max(wx, cx - half)
    y0 = max(wy, cy - half)
    x1 = min(wx + ww - 1, cx + half)
    y1 = min(wy + wh - 1, cy + half)
    if x1 < x0 or y1 < y0:
        return snapshot

    out = dict(snapshot)
    texts = list(out.get("texts") or []) if isinstance(out.get("texts"), list) else []
    occupied: list[tuple[int, int, int, int]] = []
    seen_keys: set[tuple[str, int, int, int, int]] = set()

    def _occupy_xywh(box: tuple[int, int, int, int] | None) -> None:
        if box is None:
            return
        occupied.append(box)

    def _is_occupied(x: int, y: int) -> bool:
        return any(point_in_rect_xywh(int(x), int(y), box) for box in occupied)

    for row in texts:
        xyxy = _text_row_screen_xyxy(row)
        if xyxy is None or not isinstance(row, dict):
            continue
        x1b, y1b, x2b, y2b = xyxy
        _occupy_xywh((x1b, y1b, x2b - x1b, y2b - y1b))
        label = str(row.get("text") or "").strip()
        if label:
            seen_keys.add((label.casefold(), x1b, y1b, x2b, y2b))

    started = time.perf_counter()
    added = 0
    probes = 0
    # Cursor first, then grid.
    points: list[tuple[int, int]] = [(cx, cy)]
    for gy in range(y0, y1 + 1, step):
        for gx in range(x0, x1 + 1, step):
            if gx == cx and gy == cy:
                continue
            points.append((gx, gy))

    for px, py in points:
        if probes >= int(max_probes):
            break
        if time.perf_counter() - started >= float(max_seconds):
            break
        if _is_occupied(px, py):
            continue
        probes += 1
        payload = _element_from_point_name_value_impl(int(px), int(py))
        label = efp_cover_label_from_payload(payload)
        if not label:
            bbox = efp_payload_screen_bbox_xywh(payload)
            _occupy_xywh(bbox)
            continue
        bbox = efp_payload_screen_bbox_xywh(payload)
        if bbox is None:
            continue
        bx, by, bw, bh = bbox
        xyxy = (bx, by, bx + bw, by + bh)
        key = (label.casefold(), xyxy[0], xyxy[1], xyxy[2], xyxy[3])
        if key in seen_keys:
            _occupy_xywh(bbox)
            continue
        if window is not None and is_oversized_uia_rect(bbox, window):
            _occupy_xywh(bbox)
            continue
        entry: dict[str, object] = {
            "bbox": [int(v) for v in xyxy],
            "text": label,
        }
        control_type = str(payload.get("control_type") or "").strip()
        if control_type:
            entry["control_type"] = control_type
        texts.append(entry)
        seen_keys.add(key)
        _occupy_xywh(bbox)
        added += 1

    out["texts"] = texts
    out["efp_scan_s"] = round(time.perf_counter() - started, 3)
    out["efp_scan_added"] = int(added)
    out["efp_scan_probes"] = int(probes)
    return out


def _refetch_and_enrich_uia_snapshot(
    hwnd: int,
    snapshot: dict[str, Any] | None,
    *,
    cursor_xy: tuple[int, int],
    budgets: WalkBudgets | None,
) -> dict[str, Any] | None:
    """One cursor-climbed FindAll refetch on EFP miss, then optional EFP ROI scan.

    Runs on the UIA worker thread only.
    """
    if not isinstance(snapshot, dict):
        return snapshot
    try:
        cx, cy = int(cursor_xy[0]), int(cursor_xy[1])
    except (TypeError, ValueError, IndexError):
        return snapshot

    efp = _element_from_point_name_value_impl(cx, cy)
    efp_label = efp_cover_label_from_payload(efp)
    out: dict[str, Any] = dict(snapshot)

    if efp_label:
        covered = snapshot_efp_label_covered(out, (cx, cy), efp_label)
        if not covered:
            started = time.perf_counter()
            with bind_findall_timing() as samples:
                refetched = _snapshot_uia_screen_impl(
                    int(hwnd),
                    cursor_xy=(cx, cy),
                    budgets=budgets,
                )
            timed = attach_findall_timing(
                refetched if isinstance(refetched, dict) else None,
                samples,
                wall_s=time.perf_counter() - started,
            )
            if isinstance(timed, dict):
                merged = _merge_findall_timing_fields(out, timed)
                # Prefer refetch leaf lists; keep summed timing on merged.
                for key in ("texts", "inputs", "scrollbars", "input_labels", "window"):
                    if key in timed:
                        merged[key] = timed[key]
                merged["findall_refetch"] = 1
                out = merged
            covered = snapshot_efp_label_covered(out, (cx, cy), efp_label)

        need_scan = (not covered) or (
            not uia_snapshot_has_rich_labels(out)
        )
        if need_scan:
            enriched = enrich_uia_snapshot_with_efp_scan(out, cursor_xy=(cx, cy))
            if isinstance(enriched, dict):
                out = enriched
    return out


_element_from_point_samples: ContextVar[list[float] | None] = ContextVar(
    "element_from_point_samples",
    default=None,
)


@contextmanager
def bind_element_from_point_timing() -> Iterator[list[float]]:
    """Accumulate wall times for each ``uia_text_element_at_point`` call."""
    samples: list[float] = []
    token = _element_from_point_samples.set(samples)
    try:
        yield samples
    finally:
        _element_from_point_samples.reset(token)


def _note_element_from_point_elapsed(seconds: float) -> None:
    samples = _element_from_point_samples.get()
    if samples is not None:
        samples.append(float(seconds))


def summarize_element_from_point_timing(
    samples: list[float] | None,
) -> tuple[float, int]:
    """Return ``(total_seconds, call_count)`` for recorded ElementFromPoint probes."""
    if not samples:
        return 0.0, 0
    return float(sum(samples)), int(len(samples))


def uia_text_element_at_point(
    x: int, y: int
) -> tuple[str, RectXywh, str, str, str] | None:
    """Combined Name|Value label, screen xywh, control-type, Name, and Value.

    ``ElementFromPoint`` can see a control that ``FindAll`` on the top-level
    hwnd does not return (Word's start page is one of those). The third value is
    a ``UIA_CONTROL_TYPE_SUFFIX`` token (``button``, ``text``, …) or ``""``.
    Name and Value are the raw UIA strings before composition.
    """
    started = time.perf_counter()
    try:
        return _uia_text_element_at_point_impl(x, y)
    finally:
        _note_element_from_point_elapsed(time.perf_counter() - started)


def _element_from_point_name_value_impl(x: int, y: int) -> dict[str, Any]:
    started = time.perf_counter()
    probed = uia_text_element_at_point(int(x), int(y))
    elapsed_s = round(time.perf_counter() - started, 3)
    out: dict[str, Any] = {"element_from_point_s": elapsed_s}
    if not isinstance(probed, tuple) or len(probed) < 5:
        return out
    name = str(probed[3] or "").strip()
    value = str(probed[4] or "").strip()
    if name or value:
        out["name"] = name
        out["value"] = value
    raw_box = probed[1]
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        try:
            bbox = [int(v) for v in raw_box]
        except (TypeError, ValueError):
            bbox = []
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            out["bbox"] = bbox
    control_type = str(probed[2] or "").strip()
    if control_type:
        out["control_type"] = control_type
    return out


def element_from_point_name_value(x: int, y: int) -> dict[str, Any]:
    """Feasibility probe: raw UIA Name/Value (+ leaf bbox) under a screen point.

    Always returns ``{"element_from_point_s": float}`` for the wall time of
    the probe. When Name and/or Value are non-empty, also includes
    ``"name"`` / ``"value"``. When the live leaf has a box, also includes
    screen ``"bbox"`` xywh and ``"control_type"`` (UIA suffix token). Used to
    stash ElementFromPoint results on recorded pointer steps without requiring
    FindAll to expose the same leaf; analysis may inject the leaf as a
    FindAll-shaped peer for instruction naming. Marshals onto the UIA worker.
    """
    from src.recorder.uia_thread import uia_call

    try:
        result = uia_call(
            _element_from_point_name_value_impl,
            int(x),
            int(y),
            timeout_s=2.5,
        )
    except Exception:
        return {"element_from_point_s": 0.0}
    if isinstance(result, dict):
        return result
    return {"element_from_point_s": 0.0}


def _uia_text_element_at_point_impl(
    x: int, y: int
) -> tuple[str, RectXywh, str, str, str] | None:
    if os.name != "nt":
        return None
    ole32 = ctypes.windll.ole32
    initialized_here = False
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr == _S_OK:
        initialized_here = True
    elif hr not in (_S_OK, _S_FALSE) and (hr & 0xFFFFFFFF) != 0x80010106:
        return None

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

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
        element_from_point = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            _POINT,
            ctypes.POINTER(ctypes.c_void_p),
        )(
            _com_vtable(automation, _UIA_ELEMENT_FROM_POINT + 1)[
                _UIA_ELEMENT_FROM_POINT
            ]
        )
        element_ptr = ctypes.c_void_p()
        hr = element_from_point(
            automation,
            _POINT(int(x), int(y)),
            ctypes.byref(element_ptr),
        )
        if hr != _S_OK or not element_ptr:
            return None
        element = element_ptr
        label, name, value = _element_label_parts(element)
        label = label.strip()
        name = str(name or "").strip()
        value = str(value or "").strip()
        bounds = _element_bounding_xyxy(element)
        if not label or bounds is None:
            return None
        left, top, right, bottom = bounds
        width = int(right) - int(left)
        height = int(bottom) - int(top)
        if width <= 0 or height <= 0:
            return None
        control_type = control_type_token_from_id(_element_control_type(element))
        return (
            label,
            (int(left), int(top), width, height),
            control_type,
            name,
            value,
        )
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


def _create_name_condition(
    automation: ctypes.c_void_p, name: str
) -> ctypes.c_void_p | None:
    """PropertyCondition on ``Name``; frees the temporary BSTR after create."""
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
    var, bstr = _variant_bstr(str(name))
    cond_ptr = ctypes.c_void_p()
    try:
        hr = create_prop(
            automation,
            _UIA_NAME_PROPERTY_ID,
            var,
            ctypes.byref(cond_ptr),
        )
    finally:
        _sys_free_string(bstr)
    if hr != _S_OK or not cond_ptr:
        return None
    return cond_ptr


def _create_and_condition(
    automation: ctypes.c_void_p,
    left: ctypes.c_void_p,
    right: ctypes.c_void_p,
) -> ctypes.c_void_p | None:
    create_and = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(automation, _UIA_CREATE_AND_CONDITION + 1)[
            _UIA_CREATE_AND_CONDITION
        ]
    )
    and_ptr = ctypes.c_void_p()
    hr = create_and(automation, left, right, ctypes.byref(and_ptr))
    if hr != _S_OK or not and_ptr:
        return None
    return and_ptr


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


def _create_findall_cache_request(
    automation: ctypes.c_void_p,
) -> ctypes.c_void_p | None:
    """Cache BoundingRectangle / ControlType / Name / AutomationId on matches."""
    create = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(automation, _UIA_CREATE_CACHE_REQUEST + 1)[
            _UIA_CREATE_CACHE_REQUEST
        ]
    )
    request_ptr = ctypes.c_void_p()
    hr = create(automation, ctypes.byref(request_ptr))
    if hr != _S_OK or not request_ptr:
        return None
    request = request_ptr
    add_prop = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_int,
    )(
        _com_vtable(request, _UIA_CACHE_REQUEST_ADD_PROPERTY + 1)[
            _UIA_CACHE_REQUEST_ADD_PROPERTY
        ]
    )
    put_scope = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_int,
    )(
        _com_vtable(request, _UIA_CACHE_REQUEST_PUT_TREE_SCOPE + 1)[
            _UIA_CACHE_REQUEST_PUT_TREE_SCOPE
        ]
    )
    for prop_id in (
        _UIA_BOUNDING_RECTANGLE_PROPERTY_ID,
        _UIA_CONTROL_TYPE_PROPERTY_ID,
        _UIA_NAME_PROPERTY_ID,
        _UIA_AUTOMATION_ID_PROPERTY_ID,
    ):
        if add_prop(request, int(prop_id)) != _S_OK:
            _com_release(request)
            return None
    if put_scope(request, _TREE_SCOPE_ELEMENT) != _S_OK:
        _com_release(request)
        return None
    return request


def _coinitialize_uia() -> tuple[Any, bool] | tuple[None, bool]:
    """Return ``(ole32, initialized_here)`` or ``(None, False)`` on failure."""
    if os.name != "nt":
        return None, False
    ole32 = ctypes.windll.ole32
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr == _S_OK:
        return ole32, True
    if hr in (_S_OK, _S_FALSE) or (hr & 0xFFFFFFFF) == 0x80010106:
        return ole32, False
    return None, False


def _create_uia_automation() -> ctypes.c_void_p | None:
    ole32 = ctypes.windll.ole32
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
    return automation_ptr


def _element_from_handle(
    automation: ctypes.c_void_p, hwnd: int
) -> ctypes.c_void_p | None:
    element_from_handle = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(automation, _UIA_ELEMENT_FROM_HANDLE + 1)[
            _UIA_ELEMENT_FROM_HANDLE
        ]
    )
    root_ptr = ctypes.c_void_p()
    hr = element_from_handle(
        automation,
        ctypes.c_void_p(int(hwnd)),
        ctypes.byref(root_ptr),
    )
    if hr != _S_OK or not root_ptr:
        return None
    return root_ptr


def _element_from_point(
    automation: ctypes.c_void_p, x: int, y: int
) -> ctypes.c_void_p | None:
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    element_from_point = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        _POINT,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(automation, _UIA_ELEMENT_FROM_POINT + 1)[
            _UIA_ELEMENT_FROM_POINT
        ]
    )
    element_ptr = ctypes.c_void_p()
    hr = element_from_point(
        automation,
        _POINT(int(x), int(y)),
        ctypes.byref(element_ptr),
    )
    if hr != _S_OK or not element_ptr:
        return None
    return element_ptr


def _control_view_parent(
    automation: ctypes.c_void_p, element: ctypes.c_void_p
) -> ctypes.c_void_p | None:
    get_walker = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(automation, _UIA_GET_CONTROL_VIEW_WALKER + 1)[
            _UIA_GET_CONTROL_VIEW_WALKER
        ]
    )
    walker_ptr = ctypes.c_void_p()
    hr = get_walker(automation, ctypes.byref(walker_ptr))
    if hr != _S_OK or not walker_ptr:
        return None
    walker = walker_ptr
    try:
        get_parent = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(
            _com_vtable(walker, _UIA_TREE_WALKER_GET_PARENT_ELEMENT + 1)[
                _UIA_TREE_WALKER_GET_PARENT_ELEMENT
            ]
        )
        parent_ptr = ctypes.c_void_p()
        hr = get_parent(walker, element, ctypes.byref(parent_ptr))
        if hr != _S_OK or not parent_ptr:
            return None
        return parent_ptr
    finally:
        _com_release(walker)


def _com_addref(ptr: ctypes.c_void_p | None) -> None:
    if not ptr:
        return
    addref = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(_com_vtable(ptr, 3)[1])
    addref(ptr)


def _hover_container_ready(
    element: ctypes.c_void_p,
    *,
    top_hwnd: int,
    min_area: int,
) -> bool:
    box = _element_bounding_xyxy(element)
    control_type = _element_control_type(element)
    native = _element_native_hwnd(element)
    area = _xyxy_area(box) if box is not None else 0
    if int(top_hwnd) > 0 and native == int(top_hwnd):
        return True
    if area >= int(min_area):
        return True
    return control_type in (
        _UIA_WINDOW_CONTROL_TYPE_ID,
        _UIA_DOCUMENT_CONTROL_TYPE_ID,
    )


def _climb_hover_findall_root(
    automation: ctypes.c_void_p,
    start: ctypes.c_void_p,
    *,
    top_hwnd: int,
    window_area: int,
) -> ctypes.c_void_p | None:
    """Return an AddRef'd container for hover FindAll (caller must release).

    Stops at the first ancestor whose area is >= ``_HOVER_SUBTREE_AREA_FRAC`` of
    the top-level window, or Window/Document, or whose native hwnd is
    ``top_hwnd``. Returns ``None`` when the caller should fall back to
    ``ElementFromHandle(top_hwnd)``.
    """
    min_area = max(1, int(float(window_area) * float(_HOVER_SUBTREE_AREA_FRAC)))
    current = start
    owned = False
    for _ in range(int(_HOVER_SUBTREE_MAX_CLIMB)):
        if _hover_container_ready(
            current, top_hwnd=int(top_hwnd), min_area=min_area
        ):
            if not owned:
                _com_addref(current)
            return current
        parent = _control_view_parent(automation, current)
        if owned:
            _com_release(current)
        if parent is None:
            return None
        current = parent
        owned = True
    if owned:
        _com_release(current)
    return None


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


def _xywh_to_xyxy_screen(box: RectXywh) -> ScreenXyxy:
    x, y, w, h = (int(v) for v in box[:4])
    return (x, y, x + w, y + h)


def _xyxy_center(box: ScreenXyxy) -> tuple[int, int]:
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    return (x1 + x2) // 2, (y1 + y2) // 2


def match_scrollbar_thumb_center(
    scrollbar_xywh: RectXywh,
    thumb_rects_xyxy: list[ScreenXyxy],
) -> tuple[int, int] | None:
    """Pick the live thumb center aligned with ``scrollbar_xywh``, or ``None``.

    Accepts any Thumb rect on the same line that is strictly shorter than the
    full track on the main axis (name gates are not applied — inside a known
    scrollbar bbox empty-named Win32/Chromium thumbs are valid).
    """
    sx, sy, sw, sh = (int(v) for v in scrollbar_xywh[:4])
    if sw <= 0 or sh <= 0:
        return None
    bar = _xywh_to_xyxy_screen((sx, sy, sw, sh))
    vertical = sh >= sw
    bar_main = sh if vertical else sw
    best: ScreenXyxy | None = None
    best_key: tuple[float, float, int] | None = None
    for raw in thumb_rects_xyxy:
        thumb = (
            int(raw[0]),
            int(raw[1]),
            int(raw[2]),
            int(raw[3]),
        )
        tw = int(thumb[2]) - int(thumb[0])
        th = int(thumb[3]) - int(thumb[1])
        if tw <= 0 or th <= 0:
            continue
        thumb_main = th if vertical else tw
        if thumb_main >= bar_main:
            continue
        if not _on_same_scrollbar_line(bar, thumb):
            continue
        inter = _xyxy_intersection_area(bar, thumb)
        thumb_area = float(tw * th)
        if thumb_area <= 0 or inter / thumb_area < 0.5:
            continue
        key = (
            _cross_axis_overlap_frac(bar, thumb),
            inter / thumb_area,
            thumb_main,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = thumb
    if best is None:
        return None
    return _xyxy_center(best)


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
    """ScrollBar controls plus one rect per thumb and its line buttons.

    Each thumb is a scrollbar (WPF ``ScrollbarThumb`` or WinForms ``Position``).
    Line/page buttons on that thumb's line are unioned into the same rectangle.
    A thumb already covered by a similar-sized ScrollBar control is not added
    again.

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


def _use_legacy_findall() -> bool:
    return str(os.environ.get("CUA_UIA_FINDALL_LEGACY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def default_walk_budgets() -> WalkBudgets:
    return WalkBudgets()


def chebyshev_distance_point_to_xyxy(
    x: int,
    y: int,
    box: ScreenXyxy,
) -> int:
    """Chebyshev distance from a point to the nearest edge of ``box`` (0 if inside)."""
    left, top, right, bottom = (int(v) for v in box)
    dx = 0
    if int(x) < left:
        dx = left - int(x)
    elif int(x) > right:
        dx = int(x) - right
    dy = 0
    if int(y) < top:
        dy = top - int(y)
    elif int(y) > bottom:
        dy = int(y) - bottom
    return max(dx, dy)


def walk_visit_score(
    box: ScreenXyxy | None,
    *,
    clip: ScreenXyxy,
    cursor_xy: tuple[int, int] | None,
) -> float:
    """Lower scores are visited first (near cursor / on-screen preferred)."""
    if box is None:
        return float(_WALK_MISSING_BBOX_SCORE)
    if _xyxy_intersection_area(box, clip) <= 0:
        cx = (int(clip[0]) + int(clip[2])) // 2
        cy = (int(clip[1]) + int(clip[3])) // 2
        if cursor_xy is not None:
            cx, cy = int(cursor_xy[0]), int(cursor_xy[1])
        return float(_WALK_OFFSCREEN_SCORE_BASE) + float(
            chebyshev_distance_point_to_xyxy(cx, cy, box)
        )
    if cursor_xy is not None:
        return float(
            chebyshev_distance_point_to_xyxy(
                int(cursor_xy[0]), int(cursor_xy[1]), box
            )
        )
    cx = (int(clip[0]) + int(clip[2])) // 2
    cy = (int(clip[1]) + int(clip[3])) // 2
    return float(chebyshev_distance_point_to_xyxy(cx, cy, box))


def run_budgeted_priority_walk(
    seed_nodes: list[_TNode],
    *,
    node_bbox: Callable[[_TNode], ScreenXyxy | None],
    node_matches: Callable[[_TNode], bool],
    on_match: Callable[[_TNode], None],
    node_children: Callable[[_TNode], list[_TNode]],
    release_node: Callable[[_TNode], None],
    clip: ScreenXyxy,
    cursor_xy: tuple[int, int] | None,
    budgets: WalkBudgets | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> WalkResult:
    """Priority walk with concurrent hit/node/time budgets (first trip wins).

    ``seed_nodes`` are typically the ControlView children of the walk root
    (TreeScope Descendants excludes the root itself). Caller owns releasing
    any nodes not passed to ``release_node`` only through this helper: every
    seeded/enqueued node is either released after visit or released when
    discarded by the sibling cap.
    """
    limits = budgets or default_walk_budgets()
    max_hits = max(0, int(limits.max_hits))
    max_nodes = max(0, int(limits.max_nodes))
    max_seconds = max(0.0, float(limits.max_seconds))
    max_children = max(0, int(limits.max_children_per_parent))
    started = float(clock())
    hits = 0
    nodes = 0
    stop_reason = "complete"
    truncated = False
    seq = 0
    heap: list[tuple[float, int, _TNode]] = []

    def _push(node: _TNode) -> None:
        nonlocal seq
        score = walk_visit_score(
            node_bbox(node), clip=clip, cursor_xy=cursor_xy
        )
        heapq.heappush(heap, (float(score), int(seq), node))
        seq += 1

    for seed in seed_nodes:
        _push(seed)

    try:
        while heap:
            elapsed = float(clock()) - started
            if max_seconds > 0.0 and elapsed >= max_seconds:
                stop_reason = "max_seconds"
                truncated = True
                break
            if max_nodes > 0 and nodes >= max_nodes:
                stop_reason = "max_nodes"
                truncated = True
                break
            if max_hits > 0 and hits >= max_hits:
                stop_reason = "max_hits"
                truncated = True
                break

            _score, _seq, node = heapq.heappop(heap)
            nodes += 1
            try:
                if node_matches(node):
                    on_match(node)
                    hits += 1
                    if max_hits > 0 and hits >= max_hits:
                        stop_reason = "max_hits"
                        truncated = True
                        break
                children = list(node_children(node) or [])
                if not children:
                    continue
                ranked: list[tuple[float, int, _TNode]] = []
                for index, child in enumerate(children):
                    child_score = walk_visit_score(
                        node_bbox(child), clip=clip, cursor_xy=cursor_xy
                    )
                    ranked.append((float(child_score), int(index), child))
                ranked.sort(key=lambda item: (item[0], item[1]))
                keep = ranked if max_children <= 0 else ranked[:max_children]
                drop = ranked[len(keep) :]
                for _cs, _ci, child in drop:
                    release_node(child)
                for _cs, _ci, child in keep:
                    _push(child)
            finally:
                release_node(node)
    finally:
        while heap:
            _score, _seq, node = heapq.heappop(heap)
            release_node(node)

    elapsed_s = float(clock()) - started
    return WalkResult(
        hits=int(hits),
        nodes=int(nodes),
        elapsed_s=float(elapsed_s),
        truncated=bool(truncated),
        stop_reason=str(stop_reason),
    )


def _enumerate_match_type_ids() -> frozenset[int]:
    return frozenset(
        (
            *_UIA_INPUT_CONTROL_TYPE_IDS,
            _UIA_SCROLLBAR_CONTROL_TYPE_ID,
            _UIA_THUMB_CONTROL_TYPE_ID,
            *_LABELED_TEXT_CONTROL_TYPES,
        )
    )


def _get_control_view_walker(
    automation: ctypes.c_void_p,
) -> ctypes.c_void_p | None:
    get_walker = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(automation, _UIA_GET_CONTROL_VIEW_WALKER + 1)[
            _UIA_GET_CONTROL_VIEW_WALKER
        ]
    )
    walker_ptr = ctypes.c_void_p()
    hr = get_walker(automation, ctypes.byref(walker_ptr))
    if hr != _S_OK or not walker_ptr:
        return None
    return walker_ptr


def _walker_first_child(
    walker: ctypes.c_void_p, element: ctypes.c_void_p
) -> ctypes.c_void_p | None:
    getter = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(walker, _UIA_TREE_WALKER_GET_FIRST_CHILD_ELEMENT + 1)[
            _UIA_TREE_WALKER_GET_FIRST_CHILD_ELEMENT
        ]
    )
    child = ctypes.c_void_p()
    hr = getter(walker, element, ctypes.byref(child))
    if hr != _S_OK or not child:
        return None
    return child


def _walker_next_sibling(
    walker: ctypes.c_void_p, element: ctypes.c_void_p
) -> ctypes.c_void_p | None:
    getter = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(
        _com_vtable(walker, _UIA_TREE_WALKER_GET_NEXT_SIBLING_ELEMENT + 1)[
            _UIA_TREE_WALKER_GET_NEXT_SIBLING_ELEMENT
        ]
    )
    sibling = ctypes.c_void_p()
    hr = getter(walker, element, ctypes.byref(sibling))
    if hr != _S_OK or not sibling:
        return None
    return sibling


def _walker_children(
    walker: ctypes.c_void_p, element: ctypes.c_void_p
) -> list[ctypes.c_void_p]:
    kids: list[ctypes.c_void_p] = []
    child = _walker_first_child(walker, element)
    while child is not None:
        kids.append(child)
        child = _walker_next_sibling(walker, child)
    return kids


def _clip_xyxy_for_root(
    root: ctypes.c_void_p,
    hwnd: int,
) -> ScreenXyxy:
    box = _element_bounding_xyxy(root, cached=False)
    if box is not None:
        return box
    window = _window_screen_xywh(int(hwnd))
    if window is not None:
        left, top, width, height = window
        return (int(left), int(top), int(left + width), int(top + height))
    return (0, 0, 1, 1)


def _classify_enumerated_element(
    element: ctypes.c_void_p,
    box: ScreenXyxy,
    control_type: int | None,
    *,
    cached: bool,
    inputs: list[ScreenXyxy],
    input_labels: list[str],
    scrollbars: list[ScreenXyxy],
    scrollbar_thumbs: list[ScreenXyxy],
    scrollbar_line_buttons: list[tuple[ScreenXyxy, str]],
    labeled: list[LabeledScreenXyxy],
) -> bool:
    """Append element into enumerate buckets. Returns True when it counted as a hit."""
    if control_type is None or box is None:
        return False
    automation_id = ""
    part_name = ""
    if control_type in (
        _UIA_THUMB_CONTROL_TYPE_ID,
        _UIA_BUTTON_CONTROL_TYPE_ID,
    ):
        automation_id = _element_automation_id(element, cached=cached)
        if not automation_id:
            automation_id = _element_automation_id(element, cached=False)
        part_name = _element_current_name(element, cached=cached)
        if not part_name:
            part_name = _element_current_name(element, cached=False)
    if control_type == _UIA_SCROLLBAR_CONTROL_TYPE_ID:
        scrollbars.append(box)
        return True
    if control_type == _UIA_THUMB_CONTROL_TYPE_ID and (
        is_uia_scrollbar_thumb_part(
            automation_id=automation_id,
            name=part_name,
        )
    ):
        scrollbar_thumbs.append(box)
        return True
    if control_type == _UIA_BUTTON_CONTROL_TYPE_ID and (
        is_uia_scrollbar_line_button_part(
            automation_id=automation_id,
            name=part_name,
        )
    ):
        scrollbar_line_buttons.append(
            (
                box,
                part_name or _element_label(element, cached_name=cached),
            )
        )
        return True
    if control_type in _UIA_INPUT_CONTROL_TYPE_IDS:
        name = _element_current_name(element, cached=cached)
        if not name:
            name = _element_current_name(element, cached=False)
        auto_id = _element_automation_id(element, cached=cached)
        if not auto_id:
            auto_id = _element_automation_id(element, cached=False)
        inputs.append(box)
        input_labels.append(compose_uia_input_landmark_label(name, auto_id))
        return True
    if control_type in _LABELED_TEXT_CONTROL_TYPES:
        label = _element_label(element, cached_name=cached)
        if label:
            labeled.append(
                (box, label, control_type_token_from_id(control_type))
            )
            return True
        return False
    return False


def _finalize_enumerate_lists(
    inputs: list[ScreenXyxy],
    input_labels: list[str],
    scrollbars: list[ScreenXyxy],
    scrollbar_thumbs: list[ScreenXyxy],
    scrollbar_line_buttons: list[tuple[ScreenXyxy, str]],
    labeled: list[LabeledScreenXyxy],
) -> tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]]:
    composed, used_line_buttons = compose_scrollbar_screen_rects(
        scrollbars,
        scrollbar_thumbs,
        [box for box, _label in scrollbar_line_buttons],
    )
    for index, (box, label) in enumerate(scrollbar_line_buttons):
        if index in used_line_buttons or not label:
            continue
        labeled.append((box, label, "button"))
    return inputs, composed, labeled, input_labels


def _enumerate_via_budgeted_walk(
    *,
    hwnd: int,
    root: ctypes.c_void_p,
    automation: ctypes.c_void_p,
    cursor_xy: tuple[int, int] | None,
    budgets: WalkBudgets | None,
) -> tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]]:
    empty: tuple[
        list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]
    ] = ([], [], [], [])
    match_types = _enumerate_match_type_ids()
    walker = _get_control_view_walker(automation)
    if walker is None:
        return empty
    inputs: list[ScreenXyxy] = []
    input_labels: list[str] = []
    scrollbars: list[ScreenXyxy] = []
    scrollbar_thumbs: list[ScreenXyxy] = []
    scrollbar_line_buttons: list[tuple[ScreenXyxy, str]] = []
    labeled: list[LabeledScreenXyxy] = []
    clip = _clip_xyxy_for_root(root, int(hwnd))
    try:
        seeds = _walker_children(walker, root)

        def _bbox(node: ctypes.c_void_p) -> ScreenXyxy | None:
            return _element_bounding_xyxy(node, cached=False)

        def _matches(node: ctypes.c_void_p) -> bool:
            control_type = _element_control_type(node, cached=False)
            if control_type is None or int(control_type) not in match_types:
                return False
            box = _element_bounding_xyxy(node, cached=False)
            if box is None:
                return False
            # Thumb/line-button filters may reject; probe classify without append.
            if control_type == _UIA_THUMB_CONTROL_TYPE_ID:
                auto = _element_automation_id(node, cached=False)
                name = _element_current_name(node, cached=False)
                return is_uia_scrollbar_thumb_part(automation_id=auto, name=name)
            if control_type == _UIA_BUTTON_CONTROL_TYPE_ID:
                auto = _element_automation_id(node, cached=False)
                name = _element_current_name(node, cached=False)
                if is_uia_scrollbar_line_button_part(
                    automation_id=auto, name=name
                ):
                    return True
                # Ordinary buttons are labeled-text matches.
                return bool(_element_label(node, cached_name=False))
            if control_type in _LABELED_TEXT_CONTROL_TYPES:
                return bool(_element_label(node, cached_name=False))
            return True

        def _on_match(node: ctypes.c_void_p) -> None:
            box = _element_bounding_xyxy(node, cached=False)
            control_type = _element_control_type(node, cached=False)
            if box is None or control_type is None:
                return
            _classify_enumerated_element(
                node,
                box,
                control_type,
                cached=False,
                inputs=inputs,
                input_labels=input_labels,
                scrollbars=scrollbars,
                scrollbar_thumbs=scrollbar_thumbs,
                scrollbar_line_buttons=scrollbar_line_buttons,
                labeled=labeled,
            )

        result = run_budgeted_priority_walk(
            seeds,
            node_bbox=_bbox,
            node_matches=_matches,
            on_match=_on_match,
            node_children=lambda node: _walker_children(walker, node),
            release_node=_com_release,
            clip=clip,
            cursor_xy=cursor_xy,
            budgets=budgets,
        )
        _note_findall_timing(
            elapsed_s=result.elapsed_s,
            count=result.hits,
            truncated=result.truncated,
            stop_reason=result.stop_reason,
        )
        if result.hits <= 0 and not inputs and not scrollbars and not labeled:
            return empty
        return _finalize_enumerate_lists(
            inputs,
            input_labels,
            scrollbars,
            scrollbar_thumbs,
            scrollbar_line_buttons,
            labeled,
        )
    finally:
        _com_release(walker)


def _enumerate_via_legacy_findall(
    *,
    root: ctypes.c_void_p,
    automation: ctypes.c_void_p,
) -> tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]]:
    empty: tuple[
        list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]
    ] = ([], [], [], [])
    cache_request: ctypes.c_void_p | None = None
    leaf_conds: list[ctypes.c_void_p] = []
    or_nodes: list[ctypes.c_void_p] = []
    found: ctypes.c_void_p | None = None
    inputs: list[ScreenXyxy] = []
    input_labels: list[str] = []
    scrollbars: list[ScreenXyxy] = []
    scrollbar_thumbs: list[ScreenXyxy] = []
    scrollbar_line_buttons: list[tuple[ScreenXyxy, str]] = []
    labeled: list[LabeledScreenXyxy] = []
    try:
        type_ids = tuple(_enumerate_match_type_ids())
        for type_id in type_ids:
            cond = _create_control_type_condition(automation, type_id)
            if cond is None:
                return empty
            leaf_conds.append(cond)
        or_cond, or_nodes = _fold_or_conditions(automation, leaf_conds)
        if or_cond is None:
            return empty
        cache_request = _create_findall_cache_request(automation)
        if cache_request is None:
            return empty
        find_all = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(
            _com_vtable(root, _UIA_FIND_ALL_BUILD_CACHE + 1)[
                _UIA_FIND_ALL_BUILD_CACHE
            ]
        )
        found_ptr = ctypes.c_void_p()
        started = time.perf_counter()
        hr = find_all(
            root,
            _TREE_SCOPE_DESCENDANTS,
            or_cond,
            cache_request,
            ctypes.byref(found_ptr),
        )
        findall_s = time.perf_counter() - started
        if hr != _S_OK or not found_ptr:
            _note_findall_timing(elapsed_s=findall_s, count=0)
            return empty
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
        hit_count = int(length.value) if hr == _S_OK and length.value > 0 else 0
        _note_findall_timing(elapsed_s=findall_s, count=hit_count)
        if hit_count <= 0:
            return empty
        for index in range(hit_count):
            elem_ptr = ctypes.c_void_p()
            hr = get_element(found, index, ctypes.byref(elem_ptr))
            if hr != _S_OK or not elem_ptr:
                continue
            element = elem_ptr
            try:
                box = _element_bounding_xyxy(element, cached=True)
                if box is None:
                    box = _element_bounding_xyxy(element, cached=False)
                if box is None:
                    continue
                control_type = _element_control_type(element, cached=True)
                if control_type is None:
                    control_type = _element_control_type(element, cached=False)
                _classify_enumerated_element(
                    element,
                    box,
                    control_type,
                    cached=True,
                    inputs=inputs,
                    input_labels=input_labels,
                    scrollbars=scrollbars,
                    scrollbar_thumbs=scrollbar_thumbs,
                    scrollbar_line_buttons=scrollbar_line_buttons,
                    labeled=labeled,
                )
            finally:
                _com_release(element)
        return _finalize_enumerate_lists(
            inputs,
            input_labels,
            scrollbars,
            scrollbar_thumbs,
            scrollbar_line_buttons,
            labeled,
        )
    except Exception:
        return empty
    finally:
        _com_release(found)
        _com_release(cache_request)
        for node in reversed(or_nodes):
            _com_release(node)
        for cond in leaf_conds:
            _com_release(cond)


def enumerate_edit_document_scrollbar_screen(
    hwnd: int,
    *,
    root_element: ctypes.c_void_p | None = None,
    automation: ctypes.c_void_p | None = None,
    cursor_xy: tuple[int, int] | None = None,
    budgets: WalkBudgets | None = None,
) -> tuple[list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]]:
    """Budgeted ControlView walk (or legacy FindAll) for inputs/scrollbars/texts.

    Returns ``(inputs, scrollbars, labeled_texts, input_labels)``. ``input_labels``
    is aligned with ``inputs`` (Name/AutomationId landmark text, possibly empty).

    When ``automation`` / ``root_element`` are provided the caller owns them (not
    released here). Otherwise a top-level ``ElementFromHandle(hwnd)`` root is used.
    ``cursor_xy`` biases visit order toward the pointer when set.
    """
    empty: tuple[
        list[ScreenXyxy], list[ScreenXyxy], list[LabeledScreenXyxy], list[str]
    ] = ([], [], [], [])
    if os.name != "nt" or not int(hwnd or 0):
        return empty

    ole32, initialized_here = _coinitialize_uia()
    if ole32 is None:
        return empty

    own_automation = automation is None
    own_root = root_element is None
    automation_ptr: ctypes.c_void_p | None = automation
    root: ctypes.c_void_p | None = root_element
    try:
        if automation_ptr is None:
            automation_ptr = _create_uia_automation()
            if automation_ptr is None:
                return empty
        if root is None:
            root = _element_from_handle(automation_ptr, int(hwnd))
            if root is None:
                return empty
        if _use_legacy_findall():
            return _enumerate_via_legacy_findall(
                root=root, automation=automation_ptr
            )
        return _enumerate_via_budgeted_walk(
            hwnd=int(hwnd),
            root=root,
            automation=automation_ptr,
            cursor_xy=cursor_xy,
            budgets=budgets,
        )
    except Exception:
        return empty
    finally:
        if own_root:
            _com_release(root)
        if own_automation:
            _com_release(automation_ptr)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def enumerate_edit_document_screen_xyxy(hwnd: int) -> list[ScreenXyxy]:
    """FindAll Edit|Document|ComboBox|Pane under ``hwnd`` (via combined walk); screen xyxy boxes."""
    inputs, _scrollbars, _labeled, _input_labels = (
        enumerate_edit_document_scrollbar_screen(int(hwnd))
    )
    return inputs


def _point_inside_screen_xyxy(x: int, y: int, box: ScreenXyxy) -> bool:
    left, top, right, bottom = (int(v) for v in box)
    return left <= int(x) <= right and top <= int(y) <= bottom


def _foreground_hwnd() -> int | None:
    if os.name != "nt":
        return None
    hwnd = int(ctypes.windll.user32.GetForegroundWindow() or 0)
    return hwnd or None


def _with_uia_focused_element(
    consume: Callable[[ctypes.c_void_p], Any],
) -> Any | None:
    """Run ``consume(focused_element)`` under a live GetFocusedElement COM session."""
    if os.name != "nt":
        return None

    from src.recorder.focus_point import _UIA_GET_FOCUSED_ELEMENT_INDEX

    ole32 = ctypes.windll.ole32
    initialized_here = False
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr == _S_OK:
        initialized_here = True
    elif hr not in (_S_OK, _S_FALSE) and (hr & 0xFFFFFFFF) != 0x80010106:
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
        )(
            _com_vtable(automation, _UIA_GET_FOCUSED_ELEMENT_INDEX + 1)[
                _UIA_GET_FOCUSED_ELEMENT_INDEX
            ]
        )
        element_ptr = ctypes.c_void_p()
        hr = get_focused(automation, ctypes.byref(element_ptr))
        if hr != _S_OK or not element_ptr:
            return None
        element = element_ptr
        return consume(element)
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


def uia_focused_element_snapshot() -> dict[str, Any] | None:
    """Post-settle GetFocusedElement payload for recording.

    Returns ``{"name", "value", "bbox", "control_type", "automation_id"}`` when
    the focused leaf has a usable screen box. ``bbox`` is screen xywh (same as
    press-time ``element_from_point``). Empty name/value/type/id fields are
    omitted. Returns ``None`` when UIA focus is unavailable.
    """

    def _read(element: ctypes.c_void_p) -> dict[str, Any] | None:
        bounds = _element_bounding_xyxy(element)
        if bounds is None:
            return None
        left, top, right, bottom = bounds
        width = int(right) - int(left)
        height = int(bottom) - int(top)
        if width <= 0 or height <= 0:
            return None
        out: dict[str, Any] = {
            "bbox": [int(left), int(top), width, height],
        }
        _label, name, value = _element_label_parts(element)
        name = str(name or "").strip()
        value = str(value or "").strip()
        if name:
            out["name"] = name
        if value:
            out["value"] = value
        control_type = control_type_token_from_id(_element_control_type(element))
        if control_type:
            out["control_type"] = control_type
        automation_id = str(_element_automation_id(element) or "").strip()
        if automation_id:
            out["automation_id"] = automation_id
        return out

    result = _with_uia_focused_element(_read)
    return result if isinstance(result, dict) else None


def focused_element_focus_rect(
    payload: dict[str, Any] | None,
) -> tuple[int, int, int, int] | None:
    """Convert a focused-element ``bbox`` xywh into overlay ``focus_rect`` xyxy."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("bbox")
    if not (isinstance(raw, (list, tuple)) and len(raw) == 4):
        return None
    try:
        x, y, w, h = (int(v) for v in raw[:4])
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def _focused_ocr_clip_field_screen_xyxy() -> ScreenXyxy | None:
    """Focused UIA Edit/Document/ComboBox/Pane bounds when control type matches."""

    def _read(element: ctypes.c_void_p) -> ScreenXyxy | None:
        control_type = _element_control_type(element)
        if control_type not in _OCR_CLIP_FIELD_CONTROL_TYPE_IDS:
            return None
        return _element_bounding_xyxy(element)

    result = _with_uia_focused_element(_read)
    if (
        isinstance(result, tuple)
        and len(result) == 4
        and all(isinstance(v, int) for v in result)
    ):
        return result  # type: ignore[return-value]
    return None


def enumerate_ocr_clip_field_screen_xyxy(hwnd: int) -> list[ScreenXyxy]:
    """FindAll Edit|Document|ComboBox|Pane under ``hwnd`` (screen xyxy)."""
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
    leaf_conds: list[ctypes.c_void_p] = []
    or_nodes: list[ctypes.c_void_p] = []
    found: ctypes.c_void_p | None = None
    boxes: list[ScreenXyxy] = []
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

        for type_id in _OCR_CLIP_FIELD_CONTROL_TYPE_IDS:
            cond = _create_control_type_condition(automation, type_id)
            if cond is None:
                return []
            leaf_conds.append(cond)

        or_cond, or_nodes = _fold_or_conditions(automation, leaf_conds)
        if or_cond is None:
            return []

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
        if get_length(found, ctypes.byref(length)) != _S_OK:
            return []
        for index in range(int(length.value)):
            child = ctypes.c_void_p()
            if get_element(found, index, ctypes.byref(child)) != _S_OK or not child:
                continue
            try:
                box = _element_bounding_xyxy(child)
                if box is not None:
                    boxes.append(box)
            finally:
                _com_release(child)
        return boxes
    except Exception:
        return []
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


def select_uia_field_clip_containing_point(
    caret_xy: tuple[int, int],
    *,
    root_hwnd: int | None = None,
    max_area_frac: float = DEFAULT_UIA_OVERSIZE_AREA_FRAC,
    enumerate_fn: Callable[[int], list[ScreenXyxy]] | None = None,
    focused_rect_fn: Callable[[], ScreenXyxy | None] | None = None,
    window_screen_xywh: RectXywh | None = None,
) -> ScreenXyxy | None:
    """Smallest Edit/Document/ComboBox/Pane containing ``caret_xy``; skip window-sized.

    Used to expand a caret-thin ``focus_rect`` into an OCR clip without replacing
    the caret for typing-point / debug overlays. Huge Documents (e.g. an Excel
    sheet) are rejected via ``max_area_frac`` against the root window.
    """
    if os.name != "nt":
        return None
    x, y = int(caret_xy[0]), int(caret_xy[1])
    hwnd = int(root_hwnd) if root_hwnd else (_foreground_hwnd() or 0)
    if not hwnd:
        return None
    window = window_screen_xywh or _window_screen_xywh(hwnd)
    if window is None:
        return None

    enumerator = enumerate_fn or enumerate_ocr_clip_field_screen_xyxy
    try:
        candidates = list(enumerator(hwnd) or [])
    except Exception:
        candidates = []

    focused_fn = focused_rect_fn or _focused_ocr_clip_field_screen_xyxy
    try:
        focused = focused_fn()
    except Exception:
        focused = None
    if focused is not None:
        candidates.append(focused)

    containing: list[ScreenXyxy] = []
    seen: set[ScreenXyxy] = set()
    _wx, _wy, ww, wh = (int(v) for v in window)
    window_xywh = (0, 0, ww, wh)  # area compare is size-only via is_oversized
    for box in candidates:
        left, top, right, bottom = (int(v) for v in box)
        if right <= left or bottom <= top:
            continue
        key = (left, top, right, bottom)
        if key in seen:
            continue
        seen.add(key)
        if not _point_inside_screen_xyxy(x, y, key):
            continue
        # Compare field area to window area (position ignored).
        field_xywh = (0, 0, right - left, bottom - top)
        if is_oversized_uia_rect(field_xywh, window_xywh, max_area_frac=max_area_frac):
            continue
        containing.append(key)

    if not containing:
        return None
    return min(containing, key=_xyxy_area)


def sample_uia_ocr_clip_rect(
    *,
    caret_xy: tuple[int, int] | None,
    focus_rect: tuple[int, int, int, int] | None,
    caret_thin_max_px: int = 8,
    select_fn: Callable[..., ScreenXyxy | None] | None = None,
) -> tuple[int, int, int, int] | None:
    """When ``focus_rect`` is caret-thin, return a UIA field clip in screen LTRB.

    Wide focus rects already work as OCR clips; sampling is skipped for those.
    """
    if caret_xy is None or focus_rect is None:
        return None
    left, top, right, bottom = (int(v) for v in focus_rect)
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    if min(right - left, bottom - top) > int(caret_thin_max_px):
        return None
    selector = select_fn or select_uia_field_clip_containing_point
    try:
        return selector((int(caret_xy[0]), int(caret_xy[1])))
    except Exception:
        return None


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
        enumerated = enumerator(int(hwnd))
    except Exception:
        return [], [], []
    if len(enumerated) >= 4:
        input_screen, scrollbar_screen, labeled_screen, input_labels = enumerated[:4]
    else:
        input_screen, scrollbar_screen, labeled_screen = enumerated[:3]
        input_labels = []
    del input_labels  # prefer-replace path keeps bare boxes; labels via snapshot
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
        filter_uia_input_rects(
            input_screen or [],
            drop_container_inputs=True,
            **filter_kwargs,
        ),
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
    """Live Edit/Document/ComboBox/Pane boxes in image-local xywh for one monitor capture."""
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
            drop_container_inputs=True,
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



def _snapshot_dict_from_lists(
    hwnd: int,
    inputs: list[ScreenXyxy],
    scrollbars: list[ScreenXyxy],
    labeled: list[LabeledScreenXyxy],
    input_labels: list[str] | None = None,
) -> dict[str, object]:
    window = _window_screen_xywh(int(hwnd))
    texts: list[dict[str, object]] = []
    for item in labeled:
        box, label, control_type = unpack_uia_labeled(item)
        text = canonicalize_uia_shell_status_label(label)
        if not text:
            continue
        entry: dict[str, object] = {"bbox": [int(v) for v in box], "text": text}
        if control_type:
            entry["control_type"] = control_type
        texts.append(entry)
    labels = list(input_labels or [])
    while len(labels) < len(inputs):
        labels.append("")
    labels = labels[: len(inputs)]
    out: dict[str, object] = {
        "window": [int(v) for v in window] if window is not None else None,
        "inputs": [[int(v) for v in box] for box in inputs],
        "scrollbars": [[int(v) for v in box] for box in scrollbars],
        "texts": texts,
    }
    if any(str(label or "").strip() for label in labels):
        out["input_labels"] = [str(label or "") for label in labels]
    return out


def _snapshot_uia_screen_impl(
    hwnd: int,
    *,
    cursor_xy: tuple[int, int] | None = None,
    budgets: WalkBudgets | None = None,
) -> dict[str, object] | None:
    """Climb from cursor when given, then budgeted walk under the local root."""
    if os.name != "nt" or not int(hwnd or 0):
        return None
    ole32, initialized_here = _coinitialize_uia()
    if ole32 is None:
        return None
    automation: ctypes.c_void_p | None = None
    point_el: ctypes.c_void_p | None = None
    climb_root: ctypes.c_void_p | None = None
    try:
        automation = _create_uia_automation()
        if automation is None:
            return None
        root: ctypes.c_void_p | None = None
        point: tuple[int, int] | None = None
        if cursor_xy is not None:
            point = (int(cursor_xy[0]), int(cursor_xy[1]))
            point_el = _element_from_point(automation, point[0], point[1])
            window = _window_screen_xywh(int(hwnd))
            window_area = 0
            if window is not None:
                window_area = max(0, int(window[2]) * int(window[3]))
            if point_el is not None and window_area > 0:
                climb_root = _climb_hover_findall_root(
                    automation,
                    point_el,
                    top_hwnd=int(hwnd),
                    window_area=window_area,
                )
            root = climb_root
        if root is None:
            root = _element_from_handle(automation, int(hwnd))
            climb_root = root
        if root is None:
            return None
        inputs, scrollbars, labeled, input_labels = (
            enumerate_edit_document_scrollbar_screen(
                int(hwnd),
                root_element=root,
                automation=automation,
                cursor_xy=point,
                budgets=budgets,
            )
        )
        return _snapshot_dict_from_lists(
            int(hwnd), inputs, scrollbars, labeled, input_labels
        )
    except Exception:
        return None
    finally:
        _com_release(climb_root)
        _com_release(point_el)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def _find_uia_data_item_by_name_impl(
    hwnd: int, name: str
) -> tuple[RectXywh, str, str] | None:
    """Worker-thread FindFirst: DataItem AND Name under ``hwnd``."""
    target = str(name or "").strip()
    if not int(hwnd or 0) or not target:
        return None

    ole32, initialized_here = _coinitialize_uia()
    if ole32 is None:
        return None

    automation: ctypes.c_void_p | None = None
    root: ctypes.c_void_p | None = None
    type_cond: ctypes.c_void_p | None = None
    name_cond: ctypes.c_void_p | None = None
    and_cond: ctypes.c_void_p | None = None
    found: ctypes.c_void_p | None = None
    try:
        automation = _create_uia_automation()
        if automation is None:
            return None
        root = _element_from_handle(automation, int(hwnd))
        if root is None:
            return None
        type_cond = _create_control_type_condition(
            automation, _UIA_DATAITEM_CONTROL_TYPE_ID
        )
        if type_cond is None:
            return None
        name_cond = _create_name_condition(automation, target)
        if name_cond is None:
            return None
        and_cond = _create_and_condition(automation, type_cond, name_cond)
        if and_cond is None:
            return None

        find_first = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(root, _UIA_FIND_FIRST + 1)[_UIA_FIND_FIRST])
        found_ptr = ctypes.c_void_p()
        hr = find_first(
            root,
            _TREE_SCOPE_DESCENDANTS,
            and_cond,
            ctypes.byref(found_ptr),
        )
        # S_FALSE / null element = no match.
        if hr not in (_S_OK, _S_FALSE) or not found_ptr:
            return None
        found = found_ptr
        bounds = _element_bounding_xyxy(found)
        if bounds is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in bounds)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return None
        _label, live_name, live_value = _element_label_parts(found)
        return (
            (x1, y1, w, h),
            str(live_name or ""),
            str(live_value or ""),
        )
    except Exception:
        return None
    finally:
        _com_release(found)
        _com_release(and_cond)
        _com_release(name_cond)
        _com_release(type_cond)
        _com_release(root)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def find_uia_data_item_by_name(
    hwnd: int,
    name: str,
    *,
    timeout_s: float = 2.5,
) -> tuple[RectXywh, str, str] | None:
    """FindFirst under ``hwnd``: ControlType=DataItem AND Name=``name``.

    Returns ``(screen_xywh, live_name, live_value)`` or ``None``. Does not add
    DataItems to the default budgeted FindAll walk — targeted lookup only.
    Marshals onto the UIA worker thread when not already running there.
    """
    if os.name != "nt" or not int(hwnd or 0) or not str(name or "").strip():
        return None
    from src.recorder.uia_thread import uia_call

    try:
        result = uia_call(
            _find_uia_data_item_by_name_impl,
            int(hwnd),
            str(name),
            timeout_s=float(timeout_s),
        )
    except Exception:
        return None
    if (
        isinstance(result, tuple)
        and len(result) == 3
        and isinstance(result[0], tuple)
        and len(result[0]) == 4
    ):
        return result  # type: ignore[return-value]
    return None


# Cap Name FindAll hits so a huge duplicate-Name tree cannot blow memory.
_NAME_FINDALL_MAX_HITS = 64


def _find_uia_elements_by_name_impl(
    hwnd: int,
    name: str,
    control_type_id: int | None,
) -> list[tuple[RectXywh, str, str, str]]:
    """Worker-thread FindAll: Name=``name`` (optional ControlType AND)."""
    target = str(name or "").strip()
    if not int(hwnd or 0) or not target:
        return []

    ole32, initialized_here = _coinitialize_uia()
    if ole32 is None:
        return []

    automation: ctypes.c_void_p | None = None
    root: ctypes.c_void_p | None = None
    type_cond: ctypes.c_void_p | None = None
    name_cond: ctypes.c_void_p | None = None
    and_cond: ctypes.c_void_p | None = None
    query_cond: ctypes.c_void_p | None = None
    found: ctypes.c_void_p | None = None
    out: list[tuple[RectXywh, str, str, str]] = []
    try:
        automation = _create_uia_automation()
        if automation is None:
            return []
        root = _element_from_handle(automation, int(hwnd))
        if root is None:
            return []
        name_cond = _create_name_condition(automation, target)
        if name_cond is None:
            return []
        query_cond = name_cond
        if control_type_id is not None and int(control_type_id) > 0:
            type_cond = _create_control_type_condition(
                automation, int(control_type_id)
            )
            if type_cond is None:
                return []
            and_cond = _create_and_condition(automation, type_cond, name_cond)
            if and_cond is None:
                return []
            query_cond = and_cond

        root_bounds = _element_bounding_xyxy(root)
        window_xywh: RectXywh | None = None
        if root_bounds is not None:
            rx1, ry1, rx2, ry2 = (int(v) for v in root_bounds)
            rw, rh = rx2 - rx1, ry2 - ry1
            if rw > 0 and rh > 0:
                window_xywh = (rx1, ry1, rw, rh)

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
            query_cond,
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
        if get_length(found, ctypes.byref(length)) != _S_OK:
            return []
        hit_count = min(int(length.value), _NAME_FINDALL_MAX_HITS)
        for index in range(hit_count):
            elem_ptr = ctypes.c_void_p()
            if get_element(found, index, ctypes.byref(elem_ptr)) != _S_OK or not elem_ptr:
                continue
            element = elem_ptr
            try:
                bounds = _element_bounding_xyxy(element)
                if bounds is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in bounds)
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                if window_xywh is not None and is_oversized_uia_rect(
                    (x1, y1, w, h), window_xywh
                ):
                    continue
                _label, live_name, live_value = _element_label_parts(element)
                ct = control_type_token_from_id(
                    _element_control_type(element, cached=False)
                )
                out.append(
                    (
                        (x1, y1, w, h),
                        str(live_name or ""),
                        str(live_value or ""),
                        str(ct or ""),
                    )
                )
            finally:
                _com_release(element)
        return out
    except Exception:
        return []
    finally:
        _com_release(found)
        _com_release(and_cond)
        _com_release(name_cond)
        _com_release(type_cond)
        _com_release(root)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def find_uia_elements_by_name(
    hwnd: int,
    name: str,
    *,
    control_type_id: int | None = None,
    timeout_s: float = 2.5,
) -> list[tuple[RectXywh, str, str, str]]:
    """FindAll under ``hwnd``: Name=``name`` (optional ControlType AND).

    Returns ``[(screen_xywh, live_name, live_value, control_type_token), ...]``.
    Zero/oversized rects are dropped. Use the **raw** (pre-lookalike-fold) Name
    for the COM PropertyCondition. Marshals onto the UIA worker thread.
    """
    if os.name != "nt" or not int(hwnd or 0) or not str(name or "").strip():
        return []
    from src.recorder.uia_thread import uia_call

    try:
        result = uia_call(
            _find_uia_elements_by_name_impl,
            int(hwnd),
            str(name),
            int(control_type_id) if control_type_id is not None else None,
            timeout_s=float(timeout_s),
        )
    except Exception:
        return []
    if not isinstance(result, list):
        return []
    cleaned: list[tuple[RectXywh, str, str, str]] = []
    for item in result:
        if (
            isinstance(item, tuple)
            and len(item) == 4
            and isinstance(item[0], tuple)
            and len(item[0]) == 4
        ):
            cleaned.append(item)  # type: ignore[arg-type]
    return cleaned


def control_type_id_from_token(token: str | None) -> int | None:
    """Inverse of :func:`control_type_token_from_id` for Name FindAll filters.

    ``\"text\"`` returns ``None`` so TreeItem/ListItem hubs recorded as generic
    文字 are not excluded by a Text-only PropertyCondition.
    """
    key = str(token or "").strip().lower()
    if not key or key == "text":
        return None
    return {
        "button": _UIA_BUTTON_CONTROL_TYPE_ID,
        "custom": _UIA_CUSTOM_CONTROL_TYPE_ID,
        "check_box": _UIA_CHECKBOX_CONTROL_TYPE_ID,
        "hyperlink": _UIA_HYPERLINK_CONTROL_TYPE_ID,
        "list_item": _UIA_LISTITEM_CONTROL_TYPE_ID,
        "menu_item": _UIA_MENUITEM_CONTROL_TYPE_ID,
        "radio_button": _UIA_RADIOBUTTON_CONTROL_TYPE_ID,
        "tab_item": _UIA_TABITEM_CONTROL_TYPE_ID,
        "tree_item": _UIA_TREEITEM_CONTROL_TYPE_ID,
        "data_item": _UIA_DATAITEM_CONTROL_TYPE_ID,
        "split_button": _UIA_SPLITBUTTON_CONTROL_TYPE_ID,
        "header": _UIA_HEADER_CONTROL_TYPE_ID,
        "header_item": _UIA_HEADERITEM_CONTROL_TYPE_ID,
    }.get(key)


# Cap Thumb FindAll hits so a huge tree cannot blow memory during drag start.
_THUMB_FINDALL_MAX_HITS = 64


def _find_uia_scrollbar_thumb_rects_impl(hwnd: int) -> list[ScreenXyxy]:
    """Worker-thread FindAll: ControlType=Thumb under ``hwnd``."""
    if not int(hwnd or 0):
        return []

    ole32, initialized_here = _coinitialize_uia()
    if ole32 is None:
        return []

    automation: ctypes.c_void_p | None = None
    root: ctypes.c_void_p | None = None
    type_cond: ctypes.c_void_p | None = None
    found: ctypes.c_void_p | None = None
    out: list[ScreenXyxy] = []
    try:
        automation = _create_uia_automation()
        if automation is None:
            return []
        root = _element_from_handle(automation, int(hwnd))
        if root is None:
            return []
        type_cond = _create_control_type_condition(
            automation, _UIA_THUMB_CONTROL_TYPE_ID
        )
        if type_cond is None:
            return []

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
            type_cond,
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
        if get_length(found, ctypes.byref(length)) != _S_OK:
            return []
        hit_count = min(int(length.value), _THUMB_FINDALL_MAX_HITS)
        for index in range(hit_count):
            elem_ptr = ctypes.c_void_p()
            if get_element(found, index, ctypes.byref(elem_ptr)) != _S_OK or not elem_ptr:
                continue
            element = elem_ptr
            try:
                bounds = _element_bounding_xyxy(element)
                if bounds is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in bounds)
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append((x1, y1, x2, y2))
            finally:
                _com_release(element)
        return out
    except Exception:
        return []
    finally:
        _com_release(found)
        _com_release(type_cond)
        _com_release(root)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def find_uia_scrollbar_thumb_center(
    hwnd: int,
    scrollbar_xywh: RectXywh,
    *,
    timeout_s: float = 2.5,
) -> tuple[int, int] | None:
    """Return the live Thumb center for ``scrollbar_xywh`` under ``hwnd``.

    Finds ControlType=Thumb descendants and picks the one aligned with the
    scrollbar track (any Thumb inside a known bar — no name/AutomationId gate).
    Marshals onto the UIA worker thread. Returns ``None`` when no thumb matches.
    """
    sx, sy, sw, sh = (int(v) for v in scrollbar_xywh[:4])
    if os.name != "nt" or not int(hwnd or 0) or sw <= 0 or sh <= 0:
        return None
    from src.recorder.uia_thread import uia_call

    try:
        result = uia_call(
            _find_uia_scrollbar_thumb_rects_impl,
            int(hwnd),
            timeout_s=float(timeout_s),
        )
    except Exception:
        return None
    if not isinstance(result, list):
        return None
    thumbs: list[ScreenXyxy] = []
    for item in result:
        if (
            isinstance(item, tuple)
            and len(item) == 4
            and all(isinstance(v, (int, float)) for v in item)
        ):
            thumbs.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
    return match_scrollbar_thumb_center((sx, sy, sw, sh), thumbs)


def snapshot_uia_screen(
    hwnd: int,
    *,
    cursor_xy: tuple[int, int] | None = None,
    budgets: WalkBudgets | None = None,
    enrich: bool = True,
) -> dict[str, object] | None:
    """Budgeted UIA walk under ``hwnd`` (optional cursor climb) for analysis.

    Called after settle on the window-step thread (and once at recording seed).
    Returns ``None`` when the hwnd cannot be queried. An empty tree is still a
    snapshot (analysis must not FindAll again). Marshals onto the UIA worker
    thread when not already running there.

    Successful snapshots include ``findall_s`` (and usually ``findall_count``);
    truncated walks also stamp ``walk_truncated`` / ``walk_stop_reason``.

    When ``cursor_xy`` is set and ``enrich`` is true, an EFP cover miss triggers
    one cursor-climbed FindAll refetch, then a bounded EFP ROI scan when still
    uncovered or sparse (``findall_refetch`` / ``efp_scan_*`` fields).
    """
    if os.name != "nt" or not int(hwnd or 0):
        return None
    from src.recorder.uia_thread import uia_call

    def _run() -> dict[str, object] | None:
        started = time.perf_counter()
        with bind_findall_timing() as samples:
            if cursor_xy is not None:
                snap = _snapshot_uia_screen_impl(
                    int(hwnd), cursor_xy=cursor_xy, budgets=budgets
                )
            else:
                try:
                    inputs, scrollbars, labeled, input_labels = (
                        enumerate_edit_document_scrollbar_screen(
                            int(hwnd),
                            cursor_xy=None,
                            budgets=budgets,
                        )
                    )
                except Exception:
                    snap = None
                else:
                    snap = _snapshot_dict_from_lists(
                        int(hwnd), inputs, scrollbars, labeled, input_labels
                    )
        timed = attach_findall_timing(
            snap if isinstance(snap, dict) else None,
            samples,
            wall_s=time.perf_counter() - started,
        )
        if (
            enrich
            and cursor_xy is not None
            and isinstance(timed, dict)
        ):
            return _refetch_and_enrich_uia_snapshot(
                int(hwnd),
                timed,
                cursor_xy=(int(cursor_xy[0]), int(cursor_xy[1])),
                budgets=budgets,
            )
        return timed

    timeout_s = (
        float(_SNAPSHOT_UIA_ENRICH_TIMEOUT_S)
        if enrich and cursor_xy is not None
        else 2.5
    )
    try:
        return uia_call(_run, timeout_s=timeout_s)
    except Exception:
        return None


def snapshot_uia_hover_subtree(
    hwnd: int,
    x: int,
    y: int,
) -> dict[str, object] | None:
    """Hover warm-up: ElementFromPoint climb, then budgeted ControlView walk.

    Snapshot ``window`` / boxes stay in screen space under the top-level
    ``hwnd``. Returns ``None`` only when UIA cannot be created; an empty tree is
    still a snapshot dict.
    """
    return snapshot_uia_screen(int(hwnd), cursor_xy=(int(x), int(y)))


def _snapshot_xyxy_list(raw: object) -> list[ScreenXyxy]:
    if not isinstance(raw, list):
        return []
    out: list[ScreenXyxy] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            out.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
    return out


def coerce_uia_input_xywh(item: object) -> tuple[RectXywh, str]:
    """Unpack a bare xywh or labeled input row into ``(box, label)``."""
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        first = item[0]
        if isinstance(first, (list, tuple)) and len(first) == 4:
            box, label, _control_type = unpack_uia_labeled(item)
            return box, str(label or "")
    if isinstance(item, (list, tuple)) and len(item) == 4:
        try:
            return (
                (int(item[0]), int(item[1]), int(item[2]), int(item[3])),
                "",
            )
        except (TypeError, ValueError):
            return (0, 0, 0, 0), ""
    return (0, 0, 0, 0), ""


def uia_snapshot_image_local(
    snapshot: dict[str, object],
    *,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: RectXywh | None = None,
    ocr_roi_pad: int = 0,
    window_screen_xywh: RectXywh | None = None,
) -> tuple[
    list[LabeledRectXywh] | list[RectXywh] | None,
    list[RectXywh] | None,
    list[LabeledRectXywh],
]:
    """Convert a click-time snapshot to image-local boxes. Does not call FindAll.

    Empty input/scrollbar lists become ``None`` so YOLO/Hough refine still runs.
    Labeled texts stay a list (possibly empty). When ``input_labels`` are present,
    inputs are returned as ``LabeledRectXywh`` rows (control_type ``\"\"``).
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
            text = canonicalize_uia_shell_status_label(item.get("text"))
            control_type = str(item.get("control_type") or "").strip()
            if control_type not in UIA_CONTROL_TYPE_SUFFIX:
                control_type = ""
            if not text or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            labeled.append(
                (
                    (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    text,
                    control_type,
                )
            )
    input_screen = _snapshot_xyxy_list(snapshot.get("inputs"))
    labels_raw = snapshot.get("input_labels")
    input_labels: list[str] = []
    if isinstance(labels_raw, list):
        input_labels = [str(item or "") for item in labels_raw]
    while len(input_labels) < len(input_screen):
        input_labels.append("")
    has_input_labels = any(str(label or "").strip() for label in input_labels)
    labeled_out: list[LabeledRectXywh] = []
    bare_out: list[RectXywh] = []
    for index, screen_box in enumerate(input_screen):
        kept = filter_uia_input_rects(
            [screen_box],
            drop_container_inputs=True,
            **filter_kwargs,
        )
        if not kept:
            continue
        local_box = kept[0]
        bare_out.append(local_box)
        labeled_out.append((local_box, input_labels[index], ""))
    if has_input_labels:
        inputs: list[LabeledRectXywh] | list[RectXywh] | None = (
            labeled_out or None
        )
    else:
        inputs = bare_out or None
    scrollbars = filter_uia_scrollbar_rects(
        _snapshot_xyxy_list(snapshot.get("scrollbars")), **filter_kwargs
    )
    texts = filter_uia_labeled_text_rects(labeled, **filter_kwargs)
    return (inputs, scrollbars or None, texts)
