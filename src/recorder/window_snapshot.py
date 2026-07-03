from __future__ import annotations

import ctypes
import os
from dataclasses import asdict, dataclass
from typing import Any

# Windows-only: enumerate top-level windows and diff state around pointer events.
# pygetwindow may miss some UWP/Electron windows; WindowFromPoint improves targeting.

_GA_ROOT = 2
_MAXIMIZE_AREA_GROWTH_RATIO = 1.4
_MINIMIZE_AREA_SHRINK_RATIO = 0.15
_MINIMIZED_COORD_THRESHOLD = -30000
_TITLE_BAR_CLICK_Y_MAX = 80
WINDOW_SETTLE_DELAY_S = 0.25
WINDOW_SETTLE_TITLE_BAR_DELAY_S = 0.45


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    pid: int | None
    left: int
    top: int
    width: int
    height: int
    is_minimized: bool
    is_maximized: bool

    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def contains_point(self, x: int, y: int) -> bool:
        if self.width <= 0 or self.height <= 0:
            return False
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WindowInfo:
        return cls(
            hwnd=int(raw["hwnd"]),
            title=str(raw.get("title", "")),
            pid=int(raw["pid"]) if raw.get("pid") is not None else None,
            left=int(raw.get("left", 0)),
            top=int(raw.get("top", 0)),
            width=int(raw.get("width", 0)),
            height=int(raw.get("height", 0)),
            is_minimized=bool(raw.get("is_minimized", False)),
            is_maximized=bool(raw.get("is_maximized", False)),
        )


@dataclass(frozen=True)
class WindowStateChange:
    action: str
    title: str
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "title": self.title, "confidence": self.confidence}


@dataclass(frozen=True)
class WindowDiffResult:
    change: WindowStateChange | None
    debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "change": self.change.to_dict() if self.change is not None else None,
            "debug": self.debug,
        }


def _normalize_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def _window_identity_key(win: WindowInfo) -> tuple[int | None, str]:
    return (win.pid, _normalize_title(win.title))


def _pid_for_hwnd(hwnd: int) -> int | None:
    if os.name != "nt" or hwnd == 0:
        return None
    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
    return int(pid.value) if pid.value else None


def _is_zoomed(hwnd: int) -> bool:
    if os.name != "nt" or hwnd == 0:
        return False
    return bool(ctypes.windll.user32.IsZoomed(int(hwnd)))


def _is_iconic(hwnd: int) -> bool:
    if os.name != "nt" or hwnd == 0:
        return False
    return bool(ctypes.windll.user32.IsIconic(int(hwnd)))


def _window_info_from_pygetwindow(w: Any) -> WindowInfo | None:
    try:
        hwnd = int(getattr(w, "_hWnd", 0) or 0)
    except Exception:
        return None
    if hwnd == 0:
        return None
    title = (getattr(w, "title", None) or "").strip()
    try:
        left = int(w.left)
        top = int(w.top)
        width = int(w.width)
        height = int(w.height)
    except Exception:
        left = top = width = height = 0
    is_minimized = bool(getattr(w, "isMinimized", False)) or _is_iconic(hwnd)
    is_maximized = bool(getattr(w, "isMaximized", False)) or _is_zoomed(hwnd)
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        pid=_pid_for_hwnd(hwnd),
        left=left,
        top=top,
        width=width,
        height=height,
        is_minimized=is_minimized,
        is_maximized=is_maximized,
    )


def snapshot_top_level_windows() -> list[WindowInfo]:
    """Return a snapshot of visible top-level windows (Windows only)."""
    if os.name != "nt":
        return []
    try:
        import pygetwindow as gw
    except Exception:
        return []

    out: list[WindowInfo] = []
    seen: set[int] = set()
    try:
        windows = gw.getAllWindows()
    except Exception:
        return []
    for w in windows:
        info = _window_info_from_pygetwindow(w)
        if info is None or info.hwnd in seen:
            continue
        seen.add(info.hwnd)
        out.append(info)
    return out


def window_at_point(x: int, y: int) -> WindowInfo | None:
    """Return the root top-level window under a desktop point (Windows only)."""
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT(int(x), int(y))
    hwnd = int(user32.WindowFromPoint(pt))
    if hwnd == 0:
        return None
    root = int(user32.GetAncestor(hwnd, _GA_ROOT))
    if root == 0:
        root = hwnd

    try:
        import pygetwindow as gw

        for w in gw.getAllWindows():
            if int(getattr(w, "_hWnd", 0) or 0) == root:
                return _window_info_from_pygetwindow(w)
    except Exception:
        pass

    length = user32.GetWindowTextLengthW(root)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(root, buf, length + 1)
    title = buf.value.strip()
    try:
        from ctypes import wintypes

        rect = wintypes.RECT()
        user32.GetWindowRect(root, ctypes.byref(rect))
        left, top = int(rect.left), int(rect.top)
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
    except Exception:
        left = top = width = height = 0
    return WindowInfo(
        hwnd=root,
        title=title,
        pid=_pid_for_hwnd(root),
        left=left,
        top=top,
        width=width,
        height=height,
        is_minimized=_is_iconic(root),
        is_maximized=_is_zoomed(root),
    )


def _index_by_hwnd(windows: list[WindowInfo]) -> dict[int, WindowInfo]:
    return {w.hwnd: w for w in windows}


def _find_match(target: WindowInfo, windows: list[WindowInfo]) -> WindowInfo | None:
    by_hwnd = _index_by_hwnd(windows)
    if target.hwnd in by_hwnd:
        return by_hwnd[target.hwnd]
    key = _window_identity_key(target)
    if key[1]:
        pid_title_matches = [win for win in windows if _window_identity_key(win) == key]
        if len(pid_title_matches) == 1:
            return pid_title_matches[0]
    if target.title:
        ntitle = _normalize_title(target.title)
        title_matches = [win for win in windows if _normalize_title(win.title) == ntitle]
        if len(title_matches) == 1:
            return title_matches[0]
    return None


def _is_title_bar_click(click_xy: tuple[int, int] | None) -> bool:
    return click_xy is not None and int(click_xy[1]) <= _TITLE_BAR_CLICK_Y_MAX


def settle_delay_for_click(click_xy: tuple[int, int] | None) -> float:
    """Use a longer settle delay for title-bar clicks where animations are slower."""
    if _is_title_bar_click(click_xy):
        return WINDOW_SETTLE_TITLE_BAR_DELAY_S
    return WINDOW_SETTLE_DELAY_S


def _has_minimized_rect(win: WindowInfo) -> bool:
    if win.is_minimized:
        return True
    if win.left <= _MINIMIZED_COORD_THRESHOLD or win.top <= _MINIMIZED_COORD_THRESHOLD:
        return True
    return False


def _pick_target_from_click(
    before: list[WindowInfo],
    click_xy: tuple[int, int] | None,
) -> WindowInfo | None:
    if click_xy is None:
        return None
    x, y = int(click_xy[0]), int(click_xy[1])

    candidates = [w for w in before if w.contains_point(x, y)]

    if _is_title_bar_click(click_xy):
        live = window_at_point(x, y)
        if live is not None:
            matched = _find_match(live, before)
            if matched is not None:
                return matched
            if candidates:
                matched = _find_match(live, candidates)
                if matched is not None:
                    return matched
            if live.title:
                return live

    if not candidates:
        live = window_at_point(x, y)
        if live is not None:
            return _find_match(live, before)
        return None

    if len(candidates) == 1:
        return candidates[0]

    live = window_at_point(x, y)
    if live is not None:
        matched = _find_match(live, candidates)
        if matched is not None:
            return matched

    titled = [w for w in candidates if w.title]
    pool = titled or candidates
    return min(pool, key=lambda w: w.area())


def _looks_maximized(before: WindowInfo, after: WindowInfo) -> bool:
    if after.is_maximized and not before.is_maximized:
        return True
    before_area = before.area()
    after_area = after.area()
    if before_area <= 0 or after_area <= 0:
        return False
    if after_area < before_area * _MAXIMIZE_AREA_GROWTH_RATIO:
        return False
    grew_width = after.width >= int(before.width * 1.2)
    grew_height = after.height >= int(before.height * 1.2)
    return grew_width and grew_height


def _minimize_change(before_win: WindowInfo, after_win: WindowInfo) -> WindowStateChange | None:
    title = before_win.title or f"hwnd:{before_win.hwnd}"
    if not before_win.is_minimized and after_win.is_minimized:
        return WindowStateChange(action="minimize", title=title, confidence="high")
    if not _has_minimized_rect(before_win) and _has_minimized_rect(after_win):
        return WindowStateChange(action="minimize", title=title, confidence="medium")
    before_area = before_win.area()
    after_area = after_win.area()
    if (
        before_area > 10_000
        and after_area > 0
        and after_area < before_area * _MINIMIZE_AREA_SHRINK_RATIO
    ):
        return WindowStateChange(action="minimize", title=title, confidence="medium")
    return None


def _classify_target_change(before_win: WindowInfo, after_win: WindowInfo | None) -> WindowStateChange | None:
    title = before_win.title or f"hwnd:{before_win.hwnd}"
    if after_win is None:
        return WindowStateChange(action="close", title=title, confidence="high")

    minimize = _minimize_change(before_win, after_win)
    if minimize is not None:
        return minimize

    if _looks_maximized(before_win, after_win):
        return WindowStateChange(action="maximize", title=title, confidence="high")

    if (before_win.is_minimized or before_win.is_maximized) and not after_win.is_minimized and not after_win.is_maximized:
        if before_win.area() < after_win.area():
            return WindowStateChange(action="restored", title=title, confidence="medium")

    return None


def _pick_global_minimize(before: list[WindowInfo], after: list[WindowInfo]) -> WindowStateChange | None:
    transitions: list[WindowInfo] = []
    for before_win in before:
        if not before_win.title:
            continue
        after_win = _find_match(before_win, after)
        if after_win is None:
            continue
        if _minimize_change(before_win, after_win) is not None:
            transitions.append(before_win)
    if len(transitions) == 1:
        title = transitions[0].title or f"hwnd:{transitions[0].hwnd}"
        return WindowStateChange(action="minimize", title=title, confidence="medium")
    return None


def _pick_opened_at_click(
    before: list[WindowInfo],
    after: list[WindowInfo],
    click_xy: tuple[int, int] | None,
) -> WindowStateChange | None:
    if click_xy is None:
        return None
    x, y = int(click_xy[0]), int(click_xy[1])
    after_candidates = [w for w in after if w.contains_point(x, y) and w.title]
    for win in after_candidates:
        if _find_match(win, before) is None:
            return WindowStateChange(action="opened", title=win.title, confidence="medium")
    return None


def diff_snapshots(
    before: list[WindowInfo],
    after: list[WindowInfo],
    *,
    click_xy: tuple[int, int] | None = None,
) -> WindowStateChange | None:
    """Compare window lists and return the most likely state change at click_xy."""
    return diff_snapshots_with_debug(before, after, click_xy=click_xy).change


def diff_snapshots_with_debug(
    before: list[WindowInfo],
    after: list[WindowInfo],
    *,
    click_xy: tuple[int, int] | None = None,
) -> WindowDiffResult:
    """Compare window lists and return the detected change plus debug metadata."""
    debug: dict[str, Any] = {
        "windows_before_count": len(before),
        "windows_after_count": len(after),
        "target_hwnd": None,
        "title_bar_click": _is_title_bar_click(click_xy),
        "settle_delay_s": settle_delay_for_click(click_xy),
    }
    if not before and not after:
        return WindowDiffResult(change=None, debug=debug)

    target = _pick_target_from_click(before, click_xy)
    if target is not None:
        debug["target_hwnd"] = target.hwnd
        after_match = _find_match(target, after)
        change = _classify_target_change(target, after_match)
        if change is not None:
            return WindowDiffResult(change=change, debug=debug)

    opened = _pick_opened_at_click(before, after, click_xy)
    if opened is not None:
        return WindowDiffResult(change=opened, debug=debug)

    global_minimize = _pick_global_minimize(before, after)
    if global_minimize is not None:
        return WindowDiffResult(change=global_minimize, debug=debug)

    before_set = {_window_identity_key(w) for w in before if w.title}
    after_set = {_window_identity_key(w) for w in after if w.title}
    removed = before_set - after_set
    added = after_set - before_set
    if len(removed) == 1 and not added:
        key = next(iter(removed))
        for win in before:
            if _window_identity_key(win) == key:
                return WindowDiffResult(
                    change=WindowStateChange(action="close", title=win.title, confidence="medium"),
                    debug=debug,
                )
    if len(added) == 1 and not removed:
        key = next(iter(added))
        for win in after:
            if _window_identity_key(win) == key:
                return WindowDiffResult(
                    change=WindowStateChange(action="opened", title=win.title, confidence="medium"),
                    debug=debug,
                )

    return WindowDiffResult(change=None, debug=debug)


def instruction_for_window_change(change: WindowStateChange | dict[str, Any]) -> str | None:
    """Build a hub-script instruction for a confident window state change."""
    if isinstance(change, WindowStateChange):
        data = change.to_dict()
    else:
        data = change
    confidence = data.get("confidence")
    if confidence not in {"high", "medium"}:
        return None
    action = str(data.get("action", ""))
    title = str(data.get("title", "")).strip()
    if not title:
        return None
    if action == "minimize":
        return f"最小化「{title}」視窗"
    if action == "maximize":
        return f"最大化「{title}」視窗"
    if action == "close":
        return f"關閉「{title}」視窗"
    if action == "restored":
        return f"還原「{title}」視窗"
    return None


def format_window_change_hint(change: WindowStateChange | dict[str, Any] | None) -> str:
    if change is None:
        return "(none)"
    if isinstance(change, WindowStateChange):
        data = change.to_dict()
    else:
        data = change
    action = data.get("action", "unknown")
    title = data.get("title", "")
    confidence = data.get("confidence", "unknown")
    return f"action={action}, title={title!r}, confidence={confidence}"
