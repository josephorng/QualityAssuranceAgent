"""Changed-only signals stored beside the window-list verify predicate.

Light reads (foreground, window under the cursor, clipboard, process names, caret)
run with the window snapshot, off the mouse hook. UI Automation runs only for the
click, type, or scroll step that needs it, and again on replay only when the
recorded predicate contains that field.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any, Callable

from src.recorder.focus_point import (
    _CLSID_CUIAutomation,
    _CLSCTX_INPROC_SERVER,
    _COINIT_MULTITHREADED,
    _IID_IUIAutomation,
    _S_FALSE,
    _S_OK,
    _UIA_GET_FOCUSED_ELEMENT_INDEX,
    _com_release,
    _com_vtable,
    _guid_from_string,
)
from src.recorder.window_snapshot import (
    ClickWindowInfo,
    WindowInfo,
    _process_name_for_pid,
    _window_info_from_hwnd,
    window_at_point,
)

_AGENT_APP_WINDOW_TITLE = "電腦使用代理"
_CLIPBOARD_MAX_CHARS = 4000
_CARET_REPLAY_SLACK_PX = 24
_CARET_LINE_SLACK_PX = 40
_SCROLL_TOLERANCE_PERCENT = 5.0
_SCROLL_CHANGE_PERCENT = 0.5
_UIA_PARENT_WALK_CONTROL = 4
_UIA_PARENT_WALK_SCROLL = 8

# IUIAutomation vtable indexes (IUnknown is 0..2).
_UIA_ELEMENT_FROM_POINT = 7
_UIA_GET_RAW_VIEW_WALKER = 16
# IUIAutomationElement
_UIA_GET_CURRENT_PATTERN = 16
_UIA_GET_CURRENT_NAME = 23
# IUIAutomationTreeWalker.GetParent
_UIA_WALKER_GET_PARENT = 3

_UIA_TOGGLE_PATTERN = 10015
_UIA_SELECTION_ITEM_PATTERN = 10010
_UIA_EXPAND_COLLAPSE_PATTERN = 10005
_UIA_VALUE_PATTERN = 10002
_UIA_RANGE_VALUE_PATTERN = 10003

# Pattern getters sit after the mutating methods on each interface.
_UIA_TOGGLE_GET_STATE = 4
_UIA_SELECTION_GET_IS_SELECTED = 6
_UIA_EXPAND_GET_STATE = 5
_UIA_VALUE_GET_CURRENT = 4
_UIA_RANGE_GET_VALUE = 4
_UIA_RANGE_GET_MAX = 6
_UIA_RANGE_GET_MIN = 7

_TOGGLE_LABELS = {0: "unchecked", 1: "checked", 2: "indeterminate"}
_EXPAND_LABELS = {0: "collapsed", 1: "expanded", 2: "partial"}
_EXPAND_LEAF = 3

CLICK_KINDS = frozenset(
    {"click", "double_click", "triple_click", "right_click", "middle_click"}
)
TYPE_KINDS = frozenset({"text_input"})
SCROLL_KINDS = frozenset({"scroll"})

_TH32CS_SNAPPROCESS = 0x00000002
# Session helpers and service hosts. A user app that starts with no window yet
# is still recorded; these names are not.
_SYSTEM_PROCESS_NAMES = frozenset(
    {
        "svchost.exe",
        "csrss.exe",
        "smss.exe",
        "lsass.exe",
        "services.exe",
        "wininit.exe",
        "winlogon.exe",
        "dwm.exe",
        "fontdrvhost.exe",
        "sihost.exe",
        "taskhostw.exe",
        "ctfmon.exe",
        "runtimebroker.exe",
        "dllhost.exe",
        "conhost.exe",
        "searchhost.exe",
        "startmenuexperiencehost.exe",
        "textinputhost.exe",
        "shellexperiencehost.exe",
        "securityhealthservice.exe",
        "securityhealthsystray.exe",
        "msmpeng.exe",
        "nissrv.exe",
        "registry",
        "system",
        "idle",
        "spoolsv.exe",
        "audiodg.exe",
        "dashost.exe",
        "searchindexer.exe",
        "wmiprvse.exe",
        "unsecapp.exe",
        "backgroundtaskhost.exe",
        "applicationframehost.exe",
        "systemsettingsbroker.exe",
        "lockapp.exe",
        "smartscreen.exe",
        "memory compression",
    }
)

UiaReader = Callable[[str | None, tuple[int, int] | None], dict[str, Any]]


def _normalize_title(title: str) -> str:
    return " ".join(str(title or "").strip().lower().split())


def make_identity(
    class_name: str | None,
    title: str | None,
    process_name: str | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "class_name": str(class_name or "").strip(),
        "title": str(title or "").strip(),
    }
    process = str(process_name or "").strip()
    if process:
        entry["process_name"] = process
    return entry


def identity_from_window(win: WindowInfo | None) -> dict[str, Any] | None:
    if win is None:
        return None
    process = win.process_name or _process_name_for_pid(win.pid)
    return make_identity(win.class_name, win.title, process)


def identity_from_click_window(info: ClickWindowInfo | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return make_identity(info.class_name, info.title, info.process_name)


def _is_agent_identity(identity: dict[str, Any] | None) -> bool:
    if not isinstance(identity, dict):
        return False
    return str(identity.get("title") or "").strip() == _AGENT_APP_WINDOW_TITLE


def identities_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return left is None and right is None
    return (
        str(left.get("class_name") or "").strip() == str(right.get("class_name") or "").strip()
        and _normalize_title(str(left.get("title") or ""))
        == _normalize_title(str(right.get("title") or ""))
        and str(left.get("process_name") or "").strip()
        == str(right.get("process_name") or "").strip()
    )


def identity_matches_recorded(recorded: Any, live: Any) -> bool:
    """Blank recorded class or process matches any live value (older snapshots)."""
    if not isinstance(recorded, dict) or not isinstance(live, dict):
        return False
    recorded_title = _normalize_title(str(recorded.get("title") or ""))
    live_title = _normalize_title(str(live.get("title") or ""))
    if recorded_title != live_title:
        return False
    recorded_class = str(recorded.get("class_name") or "").strip()
    live_class = str(live.get("class_name") or "").strip()
    if recorded_class and recorded_class != live_class:
        return False
    recorded_process = str(recorded.get("process_name") or "").strip()
    live_process = str(live.get("process_name") or "").strip()
    if recorded_process and recorded_process != live_process:
        return False
    return True


def _foreground_identity() -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    try:
        hwnd = int(ctypes.windll.user32.GetForegroundWindow() or 0)
    except Exception:
        return None
    if hwnd == 0:
        return None
    return identity_from_window(_window_info_from_hwnd(hwnd))


def _point_identity(cursor_xy: tuple[int, int] | None) -> dict[str, Any] | None:
    if cursor_xy is None:
        return None
    try:
        win = window_at_point(int(cursor_xy[0]), int(cursor_xy[1]))
    except Exception:
        return None
    return identity_from_window(win)


def _read_clipboard() -> str | None:
    """Read clipboard text. A clipboard owner can block, so give up quickly."""
    import threading

    result: list[str | None] = []

    def _read() -> None:
        try:
            import pyperclip

            text = pyperclip.paste()
        except Exception:
            result.append(None)
            return
        if text is None:
            result.append("")
            return
        value = str(text)
        if len(value) > _CLIPBOARD_MAX_CHARS:
            value = value[:_CLIPBOARD_MAX_CHARS]
        result.append(value)

    worker = threading.Thread(target=_read, name="verify-clipboard", daemon=True)
    worker.start()
    worker.join(0.25)
    if not result:
        return None
    return result[0]


def _caret_snapshot() -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        thread_id = user32.GetWindowThreadProcessId(hwnd, None)
        if not thread_id:
            return None
        from src.recorder.focus_point import _GUITHREADINFO

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
        return {
            "rect": [
                int(top_left.x),
                int(top_left.y),
                int(bottom_right.x),
                int(bottom_right.y),
            ],
            "hwnd": int(info.hwndCaret),
        }
    except Exception:
        return None


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _kernel32_fn(name: str, restype: Any, *argtypes: Any) -> Callable[..., Any]:
    # Bind a private prototype. Setting argtypes on the shared kernel32 DLL
    # would change every other caller in the process.
    prototype = ctypes.WINFUNCTYPE(restype, *argtypes)
    return prototype((name, ctypes.windll.kernel32))


def _process_names_by_pid() -> dict[int, str]:
    if os.name != "nt":
        return {}
    create_snapshot = _kernel32_fn(
        "CreateToolhelp32Snapshot",
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    process_first = _kernel32_fn(
        "Process32FirstW",
        wintypes.BOOL,
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32W),
    )
    process_next = _kernel32_fn(
        "Process32NextW",
        wintypes.BOOL,
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESSENTRY32W),
    )
    close_handle = _kernel32_fn("CloseHandle", wintypes.BOOL, wintypes.HANDLE)
    snapshot = create_snapshot(_TH32CS_SNAPPROCESS, 0)
    invalid = int(wintypes.HANDLE(-1).value)
    if not snapshot or int(snapshot) == invalid:
        return {}
    names: dict[int, str] = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        if not process_first(snapshot, ctypes.byref(entry)):
            return {}
        while True:
            pid = int(entry.th32ProcessID)
            name = str(entry.szExeFile or "").strip()
            if pid > 0 and name:
                names[pid] = name
            entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            if not process_next(snapshot, ctypes.byref(entry)):
                break
    except Exception:
        return names
    finally:
        close_handle(snapshot)
    return names


def _session_id_for_pid(pid: int) -> int | None:
    if os.name != "nt":
        return None
    try:
        query = _kernel32_fn(
            "ProcessIdToSessionId",
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        session = wintypes.DWORD()
        if not query(int(pid), ctypes.byref(session)):
            return None
        return int(session.value)
    except Exception:
        return None


def _current_session_id() -> int | None:
    if os.name != "nt":
        return None
    try:
        return _session_id_for_pid(int(ctypes.windll.kernel32.GetCurrentProcessId()))
    except Exception:
        return None


def _pid_in_current_session(pid: int, session_id: int | None) -> bool:
    if session_id is None:
        return True
    return _session_id_for_pid(pid) == session_id


def _process_sample(windows: list[WindowInfo]) -> dict[str, Any]:
    names_by_pid = _process_names_by_pid()
    window_pids = {int(win.pid) for win in windows if win.pid}
    pid_names: dict[str, str] = {}
    for pid in window_pids:
        name = names_by_pid.get(pid)
        if name:
            pid_names[str(pid)] = name
    session_id = _current_session_id()
    window_names = {name.lower() for name in pid_names.values()}
    orphans: set[str] = set()
    for pid, name in names_by_pid.items():
        if pid in window_pids:
            continue
        if name.lower() in _SYSTEM_PROCESS_NAMES or name.lower() in window_names:
            continue
        if not _pid_in_current_session(pid, session_id):
            continue
        orphans.add(name)
    return {
        "pid_names": pid_names,
        "orphan_processes": sorted(orphans),
    }


def capture_fast_signals(
    windows: list[WindowInfo] | tuple[WindowInfo, ...],
    *,
    cursor_xy: tuple[int, int] | None = None,
    include_foreground: bool = True,
    include_processes: bool = True,
    include_caret: bool = True,
) -> dict[str, Any]:
    """Foreground, process names, caret, and optional window-under-point.

    No clipboard and no UI Automation. Safe to run on the pre-click thread.
    """
    sample: dict[str, Any] = {}
    if include_foreground:
        sample["foreground"] = _foreground_identity()
    if cursor_xy is not None:
        sample["point_window"] = _point_identity(cursor_xy)
    if include_processes:
        try:
            sample.update(_process_sample(list(windows)))
        except Exception:
            sample["pid_names"] = {}
            sample["orphan_processes"] = []
    if include_caret:
        sample["caret"] = _caret_snapshot()
    return sample


def capture_slow_signals(
    cursor_xy: tuple[int, int] | None,
    windows: list[WindowInfo] | tuple[WindowInfo, ...] | None = None,
) -> dict[str, Any]:
    """Clipboard, process names, and a throttled UI Automation sample.

    Clipboard owners and UI Automation providers can stall, so this stays off
    the mouse hook and off the window-list refresh.
    """
    sample: dict[str, Any] = {}
    text = _read_clipboard()
    if text is not None:
        sample["clipboard"] = text
    try:
        sample.update(_process_sample(list(windows or ())))
    except Exception:
        sample["pid_names"] = {}
        sample["orphan_processes"] = []
    try:
        sample.update(
            read_uia_fields(
                cursor_xy=cursor_xy,
                want_control=cursor_xy is not None,
                want_focused=True,
                want_scroll=cursor_xy is not None,
            )
        )
    except Exception:
        pass
    return sample


def capture_step_signals(
    windows: list[WindowInfo] | tuple[WindowInfo, ...],
    *,
    cursor_xy: tuple[int, int] | None,
    kind: str | None,
    uia_reader: UiaReader | None = None,
) -> dict[str, Any]:
    """After-settle sample. UI Automation runs only for click, type, or scroll."""
    sample = capture_fast_signals(windows, cursor_xy=cursor_xy)
    text = _read_clipboard()
    if text is not None:
        sample["clipboard"] = text
    if kind in CLICK_KINDS or kind in TYPE_KINDS or kind in SCROLL_KINDS:
        reader = uia_reader or read_uia_for_kind
        try:
            extra = reader(kind, cursor_xy)
        except Exception:
            extra = {}
        if isinstance(extra, dict):
            sample.update(extra)
    return sample


def read_uia_for_kind(
    kind: str | None,
    cursor_xy: tuple[int, int] | None,
) -> dict[str, Any]:
    if kind in CLICK_KINDS:
        return read_uia_fields(
            cursor_xy=cursor_xy,
            want_control=True,
            want_focused=False,
            want_scroll=False,
        )
    if kind in TYPE_KINDS:
        return read_uia_fields(
            cursor_xy=cursor_xy,
            want_control=False,
            want_focused=True,
            want_scroll=False,
        )
    if kind in SCROLL_KINDS:
        return read_uia_fields(
            cursor_xy=cursor_xy,
            want_control=False,
            want_focused=False,
            want_scroll=True,
        )
    return {}


def _cursor_xy() -> tuple[int, int] | None:
    try:
        import pyautogui

        pos = pyautogui.position()
        return int(pos.x), int(pos.y)
    except Exception:
        return None


def capture_replay_after_signals(
    windows: list[WindowInfo],
    recorded: dict[str, Any],
    press_point_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read only the after-values the recorded predicate actually asserts.

    ``click_window`` is compared to ``press_point_window``, the root window under
    the cursor immediately before the replay click. This does not call
    ``WindowFromPoint`` again after the step settles.
    """
    need_foreground = isinstance(recorded.get("foreground"), dict)
    need_point = isinstance(recorded.get("click_window"), dict)
    need_clipboard = "clipboard" in recorded
    need_process = bool(recorded.get("process_started") or recorded.get("process_exited"))
    need_caret = isinstance(recorded.get("caret"), list)
    need_focused = isinstance(recorded.get("focused"), dict) and bool(recorded["focused"])
    need_scroll = isinstance(recorded.get("scroll"), (int, float)) and not isinstance(
        recorded.get("scroll"), bool
    )
    cursor = _cursor_xy() if need_scroll else None
    sample: dict[str, Any] = {}
    if need_foreground or need_process or need_caret:
        sample.update(
            capture_fast_signals(
                windows,
                cursor_xy=None,
                include_foreground=need_foreground,
                include_processes=need_process,
                include_caret=need_caret,
            )
        )
    if need_point and isinstance(press_point_window, dict) and press_point_window:
        sample["point_window"] = press_point_window
    if need_clipboard:
        text = _read_clipboard()
        if text is not None:
            sample["clipboard"] = text
    if need_focused or need_scroll:
        sample.update(
            read_uia_fields(
                cursor_xy=cursor,
                want_control=False,
                want_focused=need_focused,
                want_scroll=need_scroll,
            )
        )
    return sample


def predicate_has_signal_assertions(predicate: dict[str, Any] | None) -> bool:
    if not isinstance(predicate, dict):
        return False
    if isinstance(predicate.get("foreground"), dict) and predicate["foreground"]:
        return True
    if isinstance(predicate.get("click_window"), dict) and predicate["click_window"]:
        return True
    if "clipboard" in predicate:
        return True
    if isinstance(predicate.get("caret"), list) and len(predicate["caret"]) == 4:
        return True
    if isinstance(predicate.get("focused"), dict) and predicate["focused"]:
        return True
    if isinstance(predicate.get("scroll"), (int, float)) and not isinstance(
        predicate.get("scroll"), bool
    ):
        return True
    for key in ("process_started", "process_exited"):
        items = predicate.get(key)
        if isinstance(items, list) and any(items):
            return True
    return False


def _pid_name_map(sample: dict[str, Any] | None) -> dict[int, str]:
    if not isinstance(sample, dict):
        return {}
    raw = sample.get("pid_names")
    if not isinstance(raw, dict):
        return {}
    names: dict[int, str] = {}
    for key, value in raw.items():
        try:
            pid = int(key)
        except (TypeError, ValueError):
            continue
        name = str(value or "").strip()
        if name:
            names[pid] = name
    return names


def _name_set(sample: dict[str, Any] | None, key: str) -> set[str]:
    if not isinstance(sample, dict):
        return set()
    raw = sample.get(key)
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _window_pids(windows: list[WindowInfo] | None) -> set[int]:
    if not windows:
        return set()
    return {int(win.pid) for win in windows if win.pid}


def _explained_process_names(
    before_windows: list[WindowInfo] | None,
    after_windows: list[WindowInfo] | None,
    before_signals: dict[str, Any] | None,
    after_signals: dict[str, Any] | None,
    *,
    started: bool,
) -> set[str]:
    before_pids = _window_pids(before_windows)
    after_pids = _window_pids(after_windows)
    names: set[str] = set()
    if started:
        pid_names = _pid_name_map(after_signals)
        for win in after_windows or []:
            if not win.pid or int(win.pid) in before_pids:
                continue
            name = pid_names.get(int(win.pid)) or (win.process_name or "")
            if name:
                names.add(name)
        return names
    pid_names = _pid_name_map(before_signals)
    for win in before_windows or []:
        if not win.pid or int(win.pid) in after_pids:
            continue
        name = pid_names.get(int(win.pid)) or (win.process_name or "")
        if name:
            names.add(name)
    return names


def _caret_parts(sample: dict[str, Any] | None) -> tuple[list[int] | None, int | None]:
    if not isinstance(sample, dict):
        return None, None
    caret = sample.get("caret")
    hwnd: int | None = None
    rect_raw: Any = caret
    if isinstance(caret, dict):
        try:
            hwnd = int(caret.get("hwnd") or 0) or None
        except (TypeError, ValueError):
            hwnd = None
        rect_raw = caret.get("rect")
    if not isinstance(rect_raw, list) or len(rect_raw) != 4:
        return None, hwnd
    try:
        return [int(value) for value in rect_raw], hwnd
    except (TypeError, ValueError):
        return None, hwnd


def _caret_moved_field(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> list[int] | None:
    before_rect, before_hwnd = _caret_parts(before)
    after_rect, after_hwnd = _caret_parts(after)
    if after_rect is None:
        return None
    if before_rect is None:
        return after_rect
    if before_hwnd and after_hwnd and before_hwnd == after_hwnd:
        return None
    if before_hwnd and after_hwnd and before_hwnd != after_hwnd:
        return after_rect
    before_y = (before_rect[1] + before_rect[3]) / 2
    after_y = (after_rect[1] + after_rect[3]) / 2
    if abs(before_y - after_y) <= _CARET_LINE_SLACK_PX:
        return None
    return after_rect


def _focused_changed(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, str] | None:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    if "focused" not in before or "focused" not in after:
        return None
    before_focused = before.get("focused")
    after_focused = after.get("focused")
    if not isinstance(after_focused, dict):
        return None
    before_name = str((before_focused or {}).get("name") or "")
    after_name = str(after_focused.get("name") or "")
    before_value = str((before_focused or {}).get("value") or "")
    after_value = str(after_focused.get("value") or "")
    foreground = after.get("foreground") if isinstance(after, dict) else None
    foreground_title = ""
    if isinstance(foreground, dict):
        foreground_title = _normalize_title(str(foreground.get("title") or ""))
    name_changed = after_name != before_name
    if name_changed and foreground_title and _normalize_title(after_name) == foreground_title:
        name_changed = False
    value_changed = after_value != before_value
    if not name_changed and not value_changed:
        return None
    stored: dict[str, str] = {}
    if name_changed and after_name.strip():
        stored["name"] = after_name
    if value_changed:
        stored["value"] = after_value
    return stored or None


def _scroll_percent(sample: dict[str, Any] | None) -> float | None:
    if not isinstance(sample, dict) or "scroll" not in sample:
        return None
    value = sample.get("scroll")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def signal_verify_fields(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    kind: str | None,
    before_windows: list[WindowInfo] | None = None,
    after_windows: list[WindowInfo] | None = None,
) -> dict[str, Any]:
    """Predicate fields for signals that changed. Omitted fields assert nothing."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {}
    fields: dict[str, Any] = {}

    before_fg = before.get("foreground") if isinstance(before.get("foreground"), dict) else None
    after_fg = after.get("foreground") if isinstance(after.get("foreground"), dict) else None
    if (
        isinstance(after_fg, dict)
        and after_fg
        and not _is_agent_identity(after_fg)
        and not identities_equal(before_fg, after_fg)
    ):
        fields["foreground"] = after_fg

    if kind in CLICK_KINDS:
        before_point = before.get("point_window")
        after_point = after.get("point_window")
        # The press-time window is the one that received the click. The window
        # under the same point after settle is whatever was revealed when a
        # flyout closed, so it is not the click target.
        if isinstance(before_point, dict) and before_point and not identities_equal(
            before_point, after_point
        ):
            fields["click_window"] = before_point

    if "clipboard" in before and "clipboard" in after:
        before_text = str(before.get("clipboard") or "")
        after_text = str(after.get("clipboard") or "")
        if before_text != after_text:
            fields["clipboard"] = after_text

    if "pid_names" in before and "pid_names" in after:
        before_window_names = set(_pid_name_map(before).values())
        after_window_names = set(_pid_name_map(after).values())
        before_orphans = _name_set(before, "orphan_processes")
        after_orphans = _name_set(after, "orphan_processes")
        explained_started = _explained_process_names(
            before_windows,
            after_windows,
            before,
            after,
            started=True,
        )
        explained_exited = _explained_process_names(
            before_windows,
            after_windows,
            before,
            after,
            started=False,
        )
        started = sorted(
            (after_window_names - before_window_names - explained_started)
            | (after_orphans - before_orphans - explained_started)
        )
        exited = sorted(
            (before_window_names - after_window_names - explained_exited)
            | (before_orphans - after_orphans - explained_exited)
        )
        if started:
            fields["process_started"] = started
        if exited:
            fields["process_exited"] = exited

    if kind in TYPE_KINDS:
        caret = _caret_moved_field(before, after)
        if caret is not None:
            fields["caret"] = caret
        focused = _focused_changed(before, after)
        if focused is not None:
            fields["focused"] = focused

    # control_state is captured for debug in signals_before/after but is not a
    # replay assertion: after a flyout closes, ElementFromPoint often hits a
    # different window under the same cursor and the state is noise.

    if kind in SCROLL_KINDS:
        before_scroll = _scroll_percent(before)
        after_scroll = _scroll_percent(after)
        if (
            before_scroll is not None
            and after_scroll is not None
            and abs(before_scroll - after_scroll) > _SCROLL_CHANGE_PERCENT
        ):
            fields["scroll"] = round(after_scroll, 1)

    return fields


def _rects_close(recorded: list[Any], live: list[Any], slack: int) -> bool:
    if len(recorded) != 4 or len(live) != 4:
        return False
    try:
        return all(abs(int(left) - int(right)) <= slack for left, right in zip(recorded, live))
    except (TypeError, ValueError):
        return False


def _tracked_process_names(sample: dict[str, Any]) -> set[str]:
    names = set(_pid_name_map(sample).values())
    names.update(_name_set(sample, "orphan_processes"))
    return names


def signal_assertions_satisfied(
    recorded: dict[str, Any],
    live_after: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Compare recorded after-values to the live after sample.

    A missing field on the recorded predicate adds no assertion. Rebuilding a
    live delta would drop a foreground that was already correct before the step.
    """
    if not predicate_has_signal_assertions(recorded):
        return True, ""
    if not isinstance(live_after, dict):
        return False, "signal snapshot after the step failed"

    if isinstance(recorded.get("foreground"), dict):
        live_fg = live_after.get("foreground")
        if _is_agent_identity(live_fg if isinstance(live_fg, dict) else None):
            return False, "window verify missed foreground"
        if not identity_matches_recorded(recorded["foreground"], live_fg):
            title = str(recorded["foreground"].get("title") or "").strip()
            return False, f"window verify missed foreground: {title or recorded['foreground']}"

    if isinstance(recorded.get("click_window"), dict):
        if not identity_matches_recorded(recorded["click_window"], live_after.get("point_window")):
            title = str(recorded["click_window"].get("title") or "").strip()
            return False, f"window verify missed click_window: {title or recorded['click_window']}"

    if "clipboard" in recorded:
        if str(live_after.get("clipboard") or "") != str(recorded.get("clipboard") or ""):
            return False, "window verify missed clipboard"

    started = recorded.get("process_started")
    if isinstance(started, list) and started:
        live_names = _tracked_process_names(live_after)
        for name in started:
            if str(name) not in live_names:
                return False, f"window verify missed process_started: {name}"

    exited = recorded.get("process_exited")
    if isinstance(exited, list) and exited:
        live_names = _tracked_process_names(live_after)
        for name in exited:
            if str(name) in live_names:
                return False, f"window verify missed process_exited: {name}"

    caret = recorded.get("caret")
    if isinstance(caret, list) and len(caret) == 4:
        live_rect, _hwnd = _caret_parts(live_after)
        if live_rect is None or not _rects_close(caret, live_rect, _CARET_REPLAY_SLACK_PX):
            return False, "window verify missed caret"

    focused = recorded.get("focused")
    if isinstance(focused, dict) and focused:
        live_focused = live_after.get("focused")
        if not isinstance(live_focused, dict):
            return False, "window verify missed focused"
        if "name" in focused and str(live_focused.get("name") or "") != str(focused.get("name") or ""):
            return False, "window verify missed focused"
        if "value" in focused and str(live_focused.get("value") or "") != str(
            focused.get("value") or ""
        ):
            return False, "window verify missed focused"

    if isinstance(recorded.get("scroll"), (int, float)) and not isinstance(
        recorded.get("scroll"), bool
    ):
        live_scroll = _scroll_percent(live_after)
        if live_scroll is None or abs(live_scroll - float(recorded["scroll"])) > _SCROLL_TOLERANCE_PERCENT:
            return False, "window verify missed scroll"

    return True, ""


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


def _uia_get_pattern(element: ctypes.c_void_p, pattern_id: int) -> ctypes.c_void_p | None:
    getter = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    )(_com_vtable(element, _UIA_GET_CURRENT_PATTERN + 1)[_UIA_GET_CURRENT_PATTERN])
    pattern = ctypes.c_void_p()
    hr = getter(element, int(pattern_id), ctypes.byref(pattern))
    if hr != _S_OK or not pattern:
        return None
    return pattern


def _uia_control_state(element: ctypes.c_void_p) -> str | None:
    selection = _uia_get_pattern(element, _UIA_SELECTION_ITEM_PATTERN)
    if selection:
        try:
            getter = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(wintypes.BOOL),
            )(_com_vtable(selection, _UIA_SELECTION_GET_IS_SELECTED + 1)[_UIA_SELECTION_GET_IS_SELECTED])
            selected = wintypes.BOOL()
            if getter(selection, ctypes.byref(selected)) == _S_OK:
                return "selected" if bool(selected.value) else "unselected"
        finally:
            _com_release(selection)
    toggle = _uia_get_pattern(element, _UIA_TOGGLE_PATTERN)
    if toggle:
        try:
            getter = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            )(_com_vtable(toggle, _UIA_TOGGLE_GET_STATE + 1)[_UIA_TOGGLE_GET_STATE])
            state = ctypes.c_int()
            if getter(toggle, ctypes.byref(state)) == _S_OK:
                return _TOGGLE_LABELS.get(int(state.value))
        finally:
            _com_release(toggle)
    expand = _uia_get_pattern(element, _UIA_EXPAND_COLLAPSE_PATTERN)
    if expand:
        try:
            getter = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            )(_com_vtable(expand, _UIA_EXPAND_GET_STATE + 1)[_UIA_EXPAND_GET_STATE])
            state = ctypes.c_int()
            if getter(expand, ctypes.byref(state)) == _S_OK:
                if int(state.value) == _EXPAND_LEAF:
                    return None
                return _EXPAND_LABELS.get(int(state.value))
        finally:
            _com_release(expand)
    return None


def _uia_focused_value(element: ctypes.c_void_p) -> dict[str, str] | None:
    name = ""
    try:
        getter = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(_com_vtable(element, _UIA_GET_CURRENT_NAME + 1)[_UIA_GET_CURRENT_NAME])
        bstr = ctypes.c_void_p()
        if getter(element, ctypes.byref(bstr)) == _S_OK:
            name = _bstr_to_str(bstr)
    except Exception:
        name = ""
    value = ""
    pattern = _uia_get_pattern(element, _UIA_VALUE_PATTERN)
    if pattern:
        try:
            getter = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(_com_vtable(pattern, _UIA_VALUE_GET_CURRENT + 1)[_UIA_VALUE_GET_CURRENT])
            bstr = ctypes.c_void_p()
            if getter(pattern, ctypes.byref(bstr)) == _S_OK:
                value = _bstr_to_str(bstr)
        finally:
            _com_release(pattern)
    if not name and not value:
        return None
    return {"name": name, "value": value}


def _uia_scroll_percent(element: ctypes.c_void_p) -> float | None:
    pattern = _uia_get_pattern(element, _UIA_RANGE_VALUE_PATTERN)
    if not pattern:
        return None
    try:
        def _double_at(index: int) -> float | None:
            getter = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
            )(_com_vtable(pattern, index + 1)[index])
            number = ctypes.c_double()
            if getter(pattern, ctypes.byref(number)) != _S_OK:
                return None
            return float(number.value)

        value = _double_at(_UIA_RANGE_GET_VALUE)
        maximum = _double_at(_UIA_RANGE_GET_MAX)
        minimum = _double_at(_UIA_RANGE_GET_MIN)
        if value is None or maximum is None or minimum is None:
            return None
        span = maximum - minimum
        if span == 0:
            return None
        return (value - minimum) / span * 100.0
    finally:
        _com_release(pattern)


def _uia_parent(walker: ctypes.c_void_p, element: ctypes.c_void_p) -> ctypes.c_void_p | None:
    getter = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )(_com_vtable(walker, _UIA_WALKER_GET_PARENT + 1)[_UIA_WALKER_GET_PARENT])
    parent = ctypes.c_void_p()
    hr = getter(walker, element, ctypes.byref(parent))
    if hr != _S_OK or not parent:
        return None
    return parent


def _walk_for(
    element: ctypes.c_void_p,
    walker: ctypes.c_void_p | None,
    limit: int,
    reader: Callable[[ctypes.c_void_p], Any],
) -> Any:
    current: ctypes.c_void_p | None = element
    owned = False
    for _index in range(max(1, limit)):
        if current is None:
            return None
        try:
            found = reader(current)
        except Exception:
            found = None
        if found is not None:
            if owned:
                _com_release(current)
            return found
        if walker is None:
            break
        parent = _uia_parent(walker, current)
        if owned:
            _com_release(current)
        current = parent
        owned = True
    if owned and current is not None:
        _com_release(current)
    return None


def read_uia_fields(
    *,
    cursor_xy: tuple[int, int] | None,
    want_control: bool,
    want_focused: bool,
    want_scroll: bool,
) -> dict[str, Any]:
    """One automation object for the fields this step asked for. Empty on failure."""
    if os.name != "nt" or not (want_control or want_focused or want_scroll):
        return {}
    ole32 = ctypes.windll.ole32
    initialized_here = False
    hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if hr == _S_OK:
        initialized_here = True
    elif hr not in (_S_OK, _S_FALSE) and (hr & 0xFFFFFFFF) != 0x80010106:
        return {}

    automation: ctypes.c_void_p | None = None
    walker: ctypes.c_void_p | None = None
    point_element: ctypes.c_void_p | None = None
    focused: ctypes.c_void_p | None = None
    fields: dict[str, Any] = {}
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
            return {}
        automation = automation_ptr

        if (want_control or want_scroll) and cursor_xy is not None:
            class _POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            element_from_point = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                _POINT,
                ctypes.POINTER(ctypes.c_void_p),
            )(_com_vtable(automation, _UIA_ELEMENT_FROM_POINT + 1)[_UIA_ELEMENT_FROM_POINT])
            point_ptr = ctypes.c_void_p()
            hr = element_from_point(
                automation,
                _POINT(int(cursor_xy[0]), int(cursor_xy[1])),
                ctypes.byref(point_ptr),
            )
            if hr == _S_OK and point_ptr:
                point_element = point_ptr

        if want_focused:
            get_focused = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(
                _com_vtable(automation, _UIA_GET_FOCUSED_ELEMENT_INDEX + 1)[
                    _UIA_GET_FOCUSED_ELEMENT_INDEX
                ]
            )
            focused_ptr = ctypes.c_void_p()
            hr = get_focused(automation, ctypes.byref(focused_ptr))
            if hr == _S_OK and focused_ptr:
                focused = focused_ptr

        if point_element is not None and (want_control or want_scroll):
            get_walker = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(_com_vtable(automation, _UIA_GET_RAW_VIEW_WALKER + 1)[_UIA_GET_RAW_VIEW_WALKER])
            walker_ptr = ctypes.c_void_p()
            hr = get_walker(automation, ctypes.byref(walker_ptr))
            if hr == _S_OK and walker_ptr:
                walker = walker_ptr

        if want_control and point_element is not None:
            state = _walk_for(
                point_element,
                walker,
                _UIA_PARENT_WALK_CONTROL,
                _uia_control_state,
            )
            if isinstance(state, str) and state:
                fields["control_state"] = state
        if want_scroll and point_element is not None:
            percent = _walk_for(
                point_element,
                walker,
                _UIA_PARENT_WALK_SCROLL,
                _uia_scroll_percent,
            )
            if isinstance(percent, float):
                fields["scroll"] = percent
        if want_focused:
            focused_value = None
            if focused is not None:
                focused_value = _uia_focused_value(focused)
            fields["focused"] = focused_value
        return fields
    except Exception:
        return fields
    finally:
        _com_release(focused)
        _com_release(point_element)
        _com_release(walker)
        _com_release(automation)
        if initialized_here:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass
