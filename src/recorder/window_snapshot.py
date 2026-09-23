from __future__ import annotations

import ctypes
import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# Windows-only: enumerate top-level windows and diff state around pointer events.
# EnumWindows plus per-hwnd reads; WindowFromPoint improves targeting.

_GA_ROOT = 2
_MAXIMIZE_AREA_GROWTH_RATIO = 1.4
_MINIMIZE_AREA_SHRINK_RATIO = 0.15
_MINIMIZED_COORD_THRESHOLD = -30000
_TITLE_BAR_CLICK_Y_MAX = 80
_DWMWA_CAPTION_BUTTON_BOUNDS = 5
_FALLBACK_CAPTION_BUTTON_WIDTH = 46
_FALLBACK_CAPTION_BUTTON_COUNT = 3
_FALLBACK_CAPTION_HEIGHT = 32
# DWM caption button rects are often a few px short of the real hit target
# (esp. maximized Chrome: recorded clicks land below the reported bottom).
_CAPTION_HIT_SLACK_PX = 12
WINDOW_SETTLE_DELAY_S = 1.0
WINDOW_SETTLE_TITLE_BAR_DELAY_S = 1.2
CaptionBounds = tuple[int, int, int, int]
RectXywh = tuple[int, int, int, int]

_TASKBAR_CLASS_NAMES = frozenset(
    {
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
        "NotifyIconOverflowWindow",
    }
)
# Shell / Start / search overlays that should keep their own (often untitled) rect.
_FLYOUT_CLASS_NAMES = frozenset(
    {
        "Windows.UI.Core.CoreWindow",
        "XamlExplorerHostIslandWindow",
        "Windows.Internal.Shell.TabProxyWindow",
        "Shell_Flyout",
        "NetUIHWND",
    }
)
_FLYOUT_TITLES = frozenset({"快顯主機"})


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
    # Screen-space caption min/max/close strip when known (DWM); else hit-test falls back.
    caption_button_bounds: CaptionBounds | None = None

    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def contains_point(self, x: int, y: int) -> bool:
        if self.width <= 0 or self.height <= 0:
            return False
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("caption_button_bounds") is None:
            data.pop("caption_button_bounds", None)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WindowInfo:
        bounds_raw = raw.get("caption_button_bounds")
        bounds: CaptionBounds | None = None
        if isinstance(bounds_raw, (list, tuple)) and len(bounds_raw) == 4:
            bounds = (
                int(bounds_raw[0]),
                int(bounds_raw[1]),
                int(bounds_raw[2]),
                int(bounds_raw[3]),
            )
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
            caption_button_bounds=bounds,
        )


@dataclass(frozen=True)
class ClickWindowInfo:
    """Press-time window under the cursor (screen coords) for ROI gating / replay."""

    hwnd: int
    title: str
    process_name: str | None
    left: int
    top: int
    width: int
    height: int
    is_maximized: bool
    is_taskbar: bool = False
    is_flyout: bool = False
    class_name: str = ""

    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def screen_rect(self) -> RectXywh:
        return (int(self.left), int(self.top), int(self.width), int(self.height))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hwnd": int(self.hwnd),
            "title": self.title,
            "process_name": self.process_name,
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
            "is_maximized": bool(self.is_maximized),
            "is_taskbar": bool(self.is_taskbar),
            "is_flyout": bool(self.is_flyout),
            "class_name": self.class_name,
        }

    def to_local_payload(
        self,
        monitor_offset: tuple[int, int] = (0, 0),
    ) -> dict[str, Any]:
        """Serialize with image-local ``rect`` xywh for the given monitor origin."""
        ox, oy = int(monitor_offset[0]), int(monitor_offset[1])
        return {
            "hwnd": int(self.hwnd),
            "title": self.title,
            "process_name": self.process_name,
            "rect": [
                int(self.left) - ox,
                int(self.top) - oy,
                int(self.width),
                int(self.height),
            ],
            "is_maximized": bool(self.is_maximized),
            "is_taskbar": bool(self.is_taskbar),
            "is_flyout": bool(self.is_flyout),
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ClickWindowInfo | None:
        if not isinstance(raw, dict):
            return None
        rect = raw.get("rect")
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            left, top, width, height = (int(v) for v in rect)
        else:
            try:
                left = int(raw.get("left", 0))
                top = int(raw.get("top", 0))
                width = int(raw.get("width", 0))
                height = int(raw.get("height", 0))
            except (TypeError, ValueError):
                return None
        hwnd_raw = raw.get("hwnd")
        try:
            hwnd = int(hwnd_raw) if hwnd_raw is not None else 0
        except (TypeError, ValueError):
            hwnd = 0
        process_name = raw.get("process_name")
        return cls(
            hwnd=hwnd,
            title=str(raw.get("title", "") or ""),
            process_name=str(process_name) if process_name else None,
            left=left,
            top=top,
            width=width,
            height=height,
            is_maximized=bool(raw.get("is_maximized", False)),
            is_taskbar=bool(raw.get("is_taskbar", False)),
            is_flyout=bool(raw.get("is_flyout", False)),
            class_name=str(raw.get("class_name", "") or ""),
        )


@dataclass(frozen=True)
class WindowStateChange:
    action: str
    title: str
    confidence: str
    # Only set for close: True when click hit the title-bar caption button strip.
    from_title_bar_close: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "title": self.title,
            "confidence": self.confidence,
        }
        if self.from_title_bar_close is not None:
            data["from_title_bar_close"] = self.from_title_bar_close
        return data


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


def _sm_cycaption() -> int:
    if os.name != "nt":
        return _FALLBACK_CAPTION_HEIGHT
    try:
        value = int(ctypes.windll.user32.GetSystemMetrics(4))  # SM_CYCAPTION
        return value if value > 0 else _FALLBACK_CAPTION_HEIGHT
    except Exception:
        return _FALLBACK_CAPTION_HEIGHT


def _dwm_caption_button_bounds_screen(
    hwnd: int,
    window_left: int,
    window_top: int,
) -> CaptionBounds | None:
    """Return caption button strip in screen coords, or None if DWM is unavailable."""
    if os.name != "nt" or hwnd == 0:
        return None
    try:
        from ctypes import wintypes

        rect = wintypes.RECT()
        hr = int(
            ctypes.windll.dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(int(hwnd)),
                ctypes.c_uint(_DWMWA_CAPTION_BUTTON_BOUNDS),
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
        )
        if hr != 0:
            return None
        # DWM returns window-relative coordinates.
        left = int(rect.left) + int(window_left)
        top = int(rect.top) + int(window_top)
        right = int(rect.right) + int(window_left)
        bottom = int(rect.bottom) + int(window_top)
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)
    except Exception:
        return None


def _fallback_caption_button_bounds(win: WindowInfo) -> CaptionBounds | None:
    """Approximate min/max/close strip from window geometry when DWM bounds are missing."""
    if win.width <= 0 or win.height <= 0:
        return None
    if win.is_minimized or win.left <= _MINIMIZED_COORD_THRESHOLD:
        return None
    height = min(_sm_cycaption() + 8, max(win.height // 4, _FALLBACK_CAPTION_HEIGHT))
    strip_w = _FALLBACK_CAPTION_BUTTON_WIDTH * _FALLBACK_CAPTION_BUTTON_COUNT
    strip_w = min(strip_w, max(win.width // 2, _FALLBACK_CAPTION_BUTTON_WIDTH))
    left = win.left + win.width - strip_w
    top = win.top
    right = win.left + win.width
    bottom = win.top + height
    return (left, top, right, bottom)


def dwm_caption_button_bounds(win: WindowInfo) -> CaptionBounds | None:
    """DWM caption strip in screen coords, or stored bounds. No geometry guess.

    A missing result means the window has no system caption buttons (custom
    chrome). Callers that need a guess should use
    :func:`caption_button_bounds_for_window`.
    """
    if win.caption_button_bounds is not None:
        return win.caption_button_bounds
    if not _hwnd_still_valid(win.hwnd):
        return None
    return _dwm_caption_button_bounds_screen(win.hwnd, win.left, win.top)


def caption_button_bounds_for_window(win: WindowInfo) -> CaptionBounds | None:
    """Caption strip for a title-bar hit test.

    Stored bounds win. Otherwise query DWM for this hwnd, then approximate
    from the window rectangle. Snapshots leave the field empty so enumeration
    does not call DWM once per window.
    """
    live = dwm_caption_button_bounds(win)
    if live is not None:
        return live
    return _fallback_caption_button_bounds(win)


# WM_NCHITTEST results for the three system caption buttons.
HTMINBUTTON = 8
HTMAXBUTTON = 9
HTCLOSE = 20
_WM_NCHITTEST = 0x0084
_SMTO_ABORTIFHUNG = 0x0002
_NCHITTEST_TIMEOUT_MS = 30
_NCHITTEST_SAMPLE_STEP_PX = 4


def _nchittest_lparam(x: int, y: int) -> int:
    """Pack screen coordinates into the WM_NCHITTEST lParam (two signed shorts)."""
    return ((int(y) & 0xFFFF) << 16) | (int(x) & 0xFFFF)


def _nchittest_coord_fits(x: int, y: int) -> bool:
    """WM_NCHITTEST carries each axis as a signed 16-bit screen coordinate."""
    return -32768 <= int(x) <= 32767 and -32768 <= int(y) <= 32767


def _nchittest_sender():
    """Bound ``SendMessageTimeoutW`` prototype, or None when it cannot be built."""
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes

        proto = ctypes.WINFUNCTYPE(
            ctypes.c_void_p,
            wintypes.HWND,
            wintypes.UINT,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_size_t),
        )
        return proto(("SendMessageTimeoutW", ctypes.windll.user32))
    except Exception:
        return None


def _send_wm_nchittest(send, hwnd: int, x: int, y: int) -> int | None:
    """Return the hit code, or None when the window does not answer in time."""
    if send is None or hwnd == 0 or not _nchittest_coord_fits(x, y):
        return None
    try:
        from ctypes import wintypes

        result = ctypes.c_size_t(0)
        ok = send(
            wintypes.HWND(int(hwnd)),
            wintypes.UINT(_WM_NCHITTEST),
            ctypes.c_size_t(0),
            ctypes.c_ssize_t(_nchittest_lparam(x, y)),
            wintypes.UINT(_SMTO_ABORTIFHUNG),
            wintypes.UINT(_NCHITTEST_TIMEOUT_MS),
            ctypes.byref(result),
        )
        if not ok:
            return None
        return int(result.value)
    except Exception:
        return None


def sample_caption_nchittest(
    hwnd: int,
    bounds: CaptionBounds,
    *,
    step: int = _NCHITTEST_SAMPLE_STEP_PX,
    slack: int = _CAPTION_HIT_SLACK_PX,
    stop_at_hit: int | None = None,
) -> list[tuple[int, int, int]] | None:
    """Sample WM_NCHITTEST across a caption strip.

    Returns ``(x, y, hit_code)`` rows, or None when hit-testing is unavailable
    (not Windows, coordinates outside the 16-bit screen range, or the window
    does not answer). An empty list means the strip was sampled and no point
    answered. When ``stop_at_hit`` is set, sampling ends after the first row
    that contains that code.
    """
    if os.name != "nt" or hwnd == 0:
        return None
    send = _nchittest_sender()
    if send is None:
        return None
    left, top, right, bottom = (int(v) for v in bounds)
    if right <= left or bottom <= top:
        return []
    stride = max(1, int(step))
    pad = max(0, int(slack))
    height = bottom - top
    y_rows = [(top + bottom) // 2]
    if height >= 8:
        y_rows.append(top + max(1, height // 4))
        y_rows.append(top + max(1, (height * 3) // 4))
    seen_y: set[int] = set()
    samples: list[tuple[int, int, int]] = []
    x_start = left - pad
    x_end = right + pad
    for y in y_rows:
        if y in seen_y:
            continue
        seen_y.add(y)
        if not _nchittest_coord_fits(x_start, y) or not _nchittest_coord_fits(x_end, y):
            return samples or None
        row: list[tuple[int, int, int]] = []
        x = x_start
        while x <= x_end:
            code = _send_wm_nchittest(send, hwnd, x, y)
            if code is None:
                # Keep points that already answered, including this row.
                # A hung window with no samples at all stays on the vision path.
                samples.extend(row)
                return samples or None
            row.append((x, y, code))
            x += stride
        samples.extend(row)
        if stop_at_hit is not None and any(code == int(stop_at_hit) for _, _, code in row):
            return samples
    return samples


def click_hits_caption_buttons(
    click_xy: tuple[int, int] | None,
    win: WindowInfo,
) -> bool:
    """True when click_xy is inside the window's caption button strip (min/max/close)."""
    if click_xy is None:
        return False
    bounds = caption_button_bounds_for_window(win)
    if bounds is None:
        return False
    x, y = int(click_xy[0]), int(click_xy[1])
    left, top, right, bottom = bounds
    slack = _CAPTION_HIT_SLACK_PX
    # Inclusive edges plus slack: DWM rects are tight (and sometimes short of
    # the painted X), so recorded clicks on/just outside the rect still count.
    return (
        left - slack <= x <= right + slack
        and top - slack <= y <= bottom + slack
    )


def _title_bar_height(win: WindowInfo) -> int:
    bounds = caption_button_bounds_for_window(win)
    if bounds is not None:
        return max(int(bounds[3]) - win.top, _FALLBACK_CAPTION_HEIGHT)
    return _sm_cycaption() + 8


def _make_window_info(
    *,
    hwnd: int,
    title: str,
    pid: int | None,
    left: int,
    top: int,
    width: int,
    height: int,
    is_minimized: bool,
    is_maximized: bool,
) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        pid=pid,
        left=left,
        top=top,
        width=width,
        height=height,
        is_minimized=is_minimized,
        is_maximized=is_maximized,
    )


def snapshot_top_level_windows() -> list[WindowInfo]:
    """Return visible top-level windows (Windows only).

    Enumerates hwnds and reads each window's title and rectangle directly.
    Caption-button bounds are filled later, only for a title-bar hit test.
    """
    if os.name != "nt":
        return []
    user32 = ctypes.windll.user32
    hwnds: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def _collect(hwnd: int, _lparam: int) -> int:
        try:
            hwnds.append(int(hwnd or 0))
        except Exception:
            return 1
        return 1

    try:
        if not user32.EnumWindows(_collect, 0):
            return []
    except Exception:
        return []

    out: list[WindowInfo] = []
    seen: set[int] = set()
    for hwnd in hwnds:
        if hwnd == 0 or hwnd in seen:
            continue
        seen.add(hwnd)
        try:
            visible = bool(user32.IsWindowVisible(hwnd))
        except Exception:
            continue
        if not visible:
            continue
        info = _window_info_from_hwnd(hwnd)
        if info is not None:
            out.append(info)
    return out


def window_at_point(x: int, y: int) -> WindowInfo | None:
    """Return the root top-level window under a desktop point (Windows only).

    Reads that hwnd directly. A full ``getAllWindows`` scan here blocks the
    mouse hook long enough for Windows to drop the matching mouse-up.
    """
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
    return _window_info_from_hwnd(root)


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


def _is_title_bar_click(
    click_xy: tuple[int, int] | None,
    windows: Sequence[WindowInfo] | None = None,
) -> bool:
    """True when the click is in a window title-bar / caption strip."""
    if click_xy is None:
        return False
    x, y = int(click_xy[0]), int(click_xy[1])
    if windows:
        containing = [w for w in windows if w.contains_point(x, y)]
        if containing:
            win = min(containing, key=lambda w: w.area())
            if click_hits_caption_buttons(click_xy, win):
                return True
            # Untitled chrome (taskbar, shell strips) must not count as title-bar.
            if win.title and win.top <= y < win.top + _title_bar_height(win):
                return True
    # Absolute screen fallback (primary-monitor title strip / legacy settle heuristic).
    return y <= _TITLE_BAR_CLICK_Y_MAX


def settle_delay_for_click(
    click_xy: tuple[int, int] | None,
    windows: Sequence[WindowInfo] | None = None,
) -> float:
    """Use a longer settle delay for title-bar clicks where animations are slower."""
    if _is_title_bar_click(click_xy, windows):
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

    if _is_title_bar_click(click_xy, before):
        live = window_at_point(x, y)
        if live is not None:
            matched = _find_match(live, before)
            if matched is not None:
                return matched
            if candidates:
                matched = _find_match(live, candidates)
                if matched is not None:
                    return matched
            elif live.title:
                # Only invent a live hwnd when the snapshot has no containing window.
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
            titled = [w for w in candidates if w.title]
            # Taskbar/search clicks often hit an untitled shell strip via
            # WindowFromPoint; prefer a real titled window when one also
            # contains the click so transient strips are not treated as the target.
            if matched.title or not titled:
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


def _restore_change(before_win: WindowInfo, after_win: WindowInfo) -> WindowStateChange | None:
    """Detect unminimize (including taskbar restore) or restore-from-maximize."""
    title = before_win.title or f"hwnd:{before_win.hwnd}"
    if after_win.is_minimized or _has_minimized_rect(after_win):
        return None

    was_minimized = before_win.is_minimized or _has_minimized_rect(before_win)
    if was_minimized and before_win.area() < after_win.area():
        confidence = "high" if before_win.is_minimized else "medium"
        return WindowStateChange(action="restored", title=title, confidence=confidence)

    # Maximized → normal (same growth check as the previous target-path logic).
    if (
        before_win.is_maximized
        and not after_win.is_maximized
        and before_win.area() < after_win.area()
    ):
        return WindowStateChange(action="restored", title=title, confidence="medium")

    return None


def _is_synthetic_window_title(title: str) -> bool:
    """True for fallback titles like hwnd:65934 (not checkable at replay)."""
    return title.startswith("hwnd:")


def _classify_target_change(
    before_win: WindowInfo,
    after_win: WindowInfo | None,
    click_xy: tuple[int, int] | None = None,
) -> WindowStateChange | None:
    title = before_win.title or f"hwnd:{before_win.hwnd}"
    if after_win is None:
        # Untitled shell windows (taskbar strips, etc.) often vanish as a side
        # effect of Start/Search clicks; never treat that as a user close.
        if not before_win.title.strip():
            return None
        return WindowStateChange(
            action="close",
            title=title,
            confidence="high",
            from_title_bar_close=click_hits_caption_buttons(click_xy, before_win),
        )

    minimize = _minimize_change(before_win, after_win)
    if minimize is not None:
        return minimize

    if _looks_maximized(before_win, after_win):
        return WindowStateChange(action="maximize", title=title, confidence="high")

    restore = _restore_change(before_win, after_win)
    if restore is not None:
        return restore

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


def _pick_global_restore(before: list[WindowInfo], after: list[WindowInfo]) -> WindowStateChange | None:
    """Detect a single off-target restore (e.g. taskbar click unminimizes an app)."""
    transitions: list[WindowInfo] = []
    for before_win in before:
        if not before_win.title:
            continue
        after_win = _find_match(before_win, after)
        if after_win is None:
            continue
        if _restore_change(before_win, after_win) is not None:
            transitions.append(after_win)
    if len(transitions) == 1:
        title = transitions[0].title or f"hwnd:{transitions[0].hwnd}"
        return WindowStateChange(action="restored", title=title, confidence="medium")
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


def _window_debug_entry(win: WindowInfo) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "hwnd": win.hwnd,
        "title": win.title,
        "pid": win.pid,
        "left": win.left,
        "top": win.top,
        "width": win.width,
        "height": win.height,
        "is_minimized": win.is_minimized,
        "is_maximized": win.is_maximized,
    }
    if win.caption_button_bounds is not None:
        entry["caption_button_bounds"] = list(win.caption_button_bounds)
    return entry


def _windows_debug_list(windows: list[WindowInfo]) -> list[dict[str, Any]]:
    """Compact before/after program list for event debug (titled windows first)."""
    titled = [w for w in windows if w.title]
    untitled = [w for w in windows if not w.title]
    return [_window_debug_entry(w) for w in (*titled, *untitled)]


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
        "windows_before": _windows_debug_list(before),
        "windows_after": _windows_debug_list(after),
        "target_hwnd": None,
        "title_bar_click": _is_title_bar_click(click_xy, before),
        "settle_delay_s": settle_delay_for_click(click_xy, before),
        "detection_path": None,
    }
    if not before and not after:
        return WindowDiffResult(change=None, debug=debug)

    target = _pick_target_from_click(before, click_xy)
    if target is not None:
        debug["target_hwnd"] = target.hwnd
        after_match = _find_match(target, after)
        change = _classify_target_change(target, after_match, click_xy)
        if change is not None:
            debug["detection_path"] = "target"
            return WindowDiffResult(change=change, debug=debug)

    opened = _pick_opened_at_click(before, after, click_xy)
    if opened is not None:
        debug["detection_path"] = "opened_at_click"
        return WindowDiffResult(change=opened, debug=debug)

    global_minimize = _pick_global_minimize(before, after)
    if global_minimize is not None:
        debug["detection_path"] = "global_minimize"
        return WindowDiffResult(change=global_minimize, debug=debug)

    global_restore = _pick_global_restore(before, after)
    if global_restore is not None:
        debug["detection_path"] = "global_restore"
        return WindowDiffResult(change=global_restore, debug=debug)

    before_set = {_window_identity_key(w) for w in before if w.title}
    after_set = {_window_identity_key(w) for w in after if w.title}
    removed = before_set - after_set
    added = after_set - before_set
    # Title/pid identity can flicker (e.g. Explorer navigates and the title
    # briefly clears) while the hwnd is still alive. Only treat as close/open
    # when the window handle itself is gone / new.
    if len(removed) == 1 and not added:
        key = next(iter(removed))
        for win in before:
            if _window_identity_key(win) == key:
                if _find_match(win, after) is None:
                    debug["detection_path"] = "identity_close"
                    return WindowDiffResult(
                        change=WindowStateChange(
                            action="close",
                            title=win.title,
                            confidence="medium",
                            from_title_bar_close=click_hits_caption_buttons(click_xy, win),
                        ),
                        debug=debug,
                    )
                break
    if len(added) == 1 and not removed:
        key = next(iter(added))
        for win in after:
            if _window_identity_key(win) == key:
                if _find_match(win, before) is None:
                    debug["detection_path"] = "identity_opened"
                    return WindowDiffResult(
                        change=WindowStateChange(
                            action="opened", title=win.title, confidence="medium"
                        ),
                        debug=debug,
                    )
                break

    return WindowDiffResult(change=None, debug=debug)


# Windows shell host that often disappears as a side effect of unrelated clicks
# (taskbar search, Start, etc.). Never treat it as the user's intended action.
_IGNORED_WINDOW_CHANGE_TITLES = frozenset({"快顯主機"})
# Hub app title; trailing restores are dropped during analysis (stop-recording artifact).
_AGENT_APP_WINDOW_TITLE = "電腦使用代理"


def _window_change_data(
    change: WindowStateChange | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(change, WindowStateChange):
        return change.to_dict()
    return change


def _should_ignore_window_change(data: dict[str, Any]) -> bool:
    title = str(data.get("title", "")).strip()
    return title in _IGNORED_WINDOW_CHANGE_TITLES


def is_agent_app_restore(change: WindowStateChange | dict[str, Any] | None) -> bool:
    """True when the change restores the Computer Use Agent hub window."""
    if change is None:
        return False
    data = _window_change_data(change)
    return (
        str(data.get("action", "")).strip() == "restored"
        and str(data.get("title", "")).strip() == _AGENT_APP_WINDOW_TITLE
    )


def resolve_window_change(
    window_change: dict[str, Any] | None,
    window_snapshot_debug: dict[str, Any] | None,
    click_xy: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Prefer captured window_change; otherwise re-diff snapshot debug lists."""
    if isinstance(window_change, dict):
        return window_change
    if not isinstance(window_snapshot_debug, dict):
        return None
    before_raw = window_snapshot_debug.get("windows_before")
    after_raw = window_snapshot_debug.get("windows_after")
    if not isinstance(before_raw, list) or not isinstance(after_raw, list):
        return None
    before: list[WindowInfo] = []
    after: list[WindowInfo] = []
    for raw in before_raw:
        if isinstance(raw, dict) and "hwnd" in raw:
            before.append(WindowInfo.from_dict(raw))
    for raw in after_raw:
        if isinstance(raw, dict) and "hwnd" in raw:
            after.append(WindowInfo.from_dict(raw))
    change = diff_snapshots(before, after, click_xy=click_xy)
    return change.to_dict() if change is not None else None


def expected_outcome_for_window_change(
    change: WindowStateChange | dict[str, Any] | None,
) -> str | None:
    """Build a checkable success criterion from a confident window state change.

    Used when recording key actions such as Enter that open/restore a window, so
    replay verification has a real expected outcome instead of ``(none)``.
    """
    if change is None:
        return None
    data = _window_change_data(change)
    if _should_ignore_window_change(data):
        return None
    if is_agent_app_restore(data):
        return None
    confidence = data.get("confidence")
    if confidence not in {"high", "medium"}:
        return None
    action = str(data.get("action", "")).strip()
    title = str(data.get("title", "")).strip()
    if not title or title == _AGENT_APP_WINDOW_TITLE or _is_synthetic_window_title(title):
        return None
    if action == "opened":
        return f"「{title}」視窗已開啟"
    if action == "restored":
        return f"「{title}」視窗已顯示"
    if action == "maximize":
        return f"「{title}」視窗已最大化並佔滿螢幕"
    if action == "minimize":
        return f"「{title}」視窗已最小化"
    if action == "close":
        return f"「{title}」視窗已關閉"
    return None


def instruction_for_window_change(change: WindowStateChange | dict[str, Any]) -> str | None:
    """Build a hub-script instruction for a confident window state change."""
    data = _window_change_data(change)
    if _should_ignore_window_change(data):
        return None
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
        # Only the title-bar caption X becomes a close instruction; 儲存/取消 stay clicks.
        if not data.get("from_title_bar_close"):
            return None
        return f"關閉「{title}」視窗"
    if action == "restored":
        return f"還原「{title}」視窗"
    return None


def format_window_change_hint(change: WindowStateChange | dict[str, Any] | None) -> str:
    if change is None:
        return "(none)"
    data = _window_change_data(change)
    if _should_ignore_window_change(data):
        return "(none)"
    action = data.get("action", "unknown")
    title = data.get("title", "")
    confidence = data.get("confidence", "unknown")
    if action == "close" and not data.get("from_title_bar_close"):
        return (
            f"action=close (not title-bar X; prefer click label), "
            f"title={title!r}, confidence={confidence}"
        )
    return f"action={action}, title={title!r}, confidence={confidence}"


def _window_class_name(hwnd: int) -> str:
    if os.name != "nt" or hwnd == 0:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetClassNameW(int(hwnd), buf, 256)
        return str(buf.value or "")
    except Exception:
        return ""


def _process_name_for_pid(pid: int | None) -> str | None:
    if os.name != "nt" or pid is None or int(pid) <= 0:
        return None
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return None
        try:
            size = ctypes.c_ulong(260)
            buf = ctypes.create_unicode_buffer(size.value)
            ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)
            )
            if not ok:
                return None
            path = str(buf.value or "").strip()
            if not path:
                return None
            return os.path.basename(path)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        return None


def _is_taskbar_class(class_name: str) -> bool:
    return class_name in _TASKBAR_CLASS_NAMES


def _is_flyout_window(*, class_name: str, title: str) -> bool:
    if title.strip() in _FLYOUT_TITLES:
        return True
    return class_name in _FLYOUT_CLASS_NAMES


def _click_window_from_window_info(win: WindowInfo) -> ClickWindowInfo:
    class_name = _window_class_name(win.hwnd)
    title = win.title or ""
    return ClickWindowInfo(
        hwnd=int(win.hwnd),
        title=title,
        process_name=_process_name_for_pid(win.pid),
        left=int(win.left),
        top=int(win.top),
        width=int(win.width),
        height=int(win.height),
        is_maximized=bool(win.is_maximized),
        is_taskbar=_is_taskbar_class(class_name),
        is_flyout=_is_flyout_window(class_name=class_name, title=title),
        class_name=class_name,
    )


def _hwnd_still_valid(hwnd: int) -> bool:
    if os.name != "nt" or hwnd == 0:
        return False
    try:
        return bool(ctypes.windll.user32.IsWindow(int(hwnd)))
    except Exception:
        return False


def _window_info_from_hwnd(hwnd: int) -> WindowInfo | None:
    if not _hwnd_still_valid(hwnd):
        return None
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(int(hwnd))
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(int(hwnd), buf, length + 1)
    title = buf.value.strip()
    try:
        from ctypes import wintypes

        rect = wintypes.RECT()
        user32.GetWindowRect(int(hwnd), ctypes.byref(rect))
        left, top = int(rect.left), int(rect.top)
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
    except Exception:
        return None
    return _make_window_info(
        hwnd=int(hwnd),
        title=title,
        pid=_pid_for_hwnd(int(hwnd)),
        left=left,
        top=top,
        width=width,
        height=height,
        is_minimized=_is_iconic(int(hwnd)),
        is_maximized=_is_zoomed(int(hwnd)),
    )


def resolve_click_window(x: int, y: int) -> ClickWindowInfo | None:
    """Return the press-time window under ``(x, y)`` for ROI / replay matching.

    Uses :func:`window_at_point` (root under the point). Taskbar and shell flyouts
    keep their own rects — do not replace with a later Explorer/owner window.
    """
    win = window_at_point(int(x), int(y))
    if win is None:
        return None
    return _click_window_from_window_info(win)


def find_matching_click_window(
    click_window: ClickWindowInfo | dict[str, Any] | None,
) -> ClickWindowInfo | None:
    """Prefer a live match for a recorded ``click_window``; else ``None``."""
    info = (
        click_window
        if isinstance(click_window, ClickWindowInfo)
        else ClickWindowInfo.from_dict(click_window) if isinstance(click_window, dict) else None
    )
    if info is None:
        return None

    if info.hwnd and _hwnd_still_valid(info.hwnd):
        live = _window_info_from_hwnd(info.hwnd)
        if live is not None:
            matched = _click_window_from_window_info(live)
            # Keep recorded taskbar/flyout flags when class lookup is noisy.
            if info.is_taskbar or info.is_flyout:
                matched = ClickWindowInfo(
                    hwnd=matched.hwnd,
                    title=matched.title or info.title,
                    process_name=matched.process_name or info.process_name,
                    left=matched.left,
                    top=matched.top,
                    width=matched.width,
                    height=matched.height,
                    is_maximized=matched.is_maximized,
                    is_taskbar=info.is_taskbar or matched.is_taskbar,
                    is_flyout=info.is_flyout or matched.is_flyout,
                    class_name=matched.class_name or info.class_name,
                )
            return matched

    # Fall back: same process + title among top-level windows.
    title_key = _normalize_title(info.title)
    process_name = (info.process_name or "").lower()
    best: ClickWindowInfo | None = None
    best_area_delta: int | None = None
    for win in snapshot_top_level_windows():
        if title_key and _normalize_title(win.title) != title_key:
            continue
        live = _click_window_from_window_info(win)
        if process_name:
            live_proc = (live.process_name or "").lower()
            if live_proc and live_proc != process_name:
                continue
        delta = abs(live.area() - info.area())
        if best is None or best_area_delta is None or delta < best_area_delta:
            best = live
            best_area_delta = delta
    return best


def _clip_xywh_to_image(
    rect: RectXywh,
    *,
    image_w: int,
    image_h: int,
) -> RectXywh | None:
    x, y, w, h = (int(v) for v in rect)
    x2 = x + w
    y2 = y + h
    x = max(0, min(x, image_w))
    y = max(0, min(y, image_h))
    x2 = max(0, min(x2, image_w))
    y2 = max(0, min(y2, image_h))
    w = x2 - x
    h = y2 - y
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def resolve_ocr_roi_local(
    click_window: ClickWindowInfo | dict[str, Any] | None,
    *,
    image_w: int,
    image_h: int,
    monitor_offset: tuple[int, int] = (0, 0),
) -> RectXywh | None:
    """Image-local xywh ROI for OCR/enhance gating, or ``None`` to leave ungated.

    Prefers a live window match (screen rect → local via ``monitor_offset``).
    Falls back to the recorded local ``rect`` / screen fields. Maximized and
    near-full-screen windows still return their clipped rect. Returns ``None``
    only when ``click_window`` is missing or the window does not intersect the image.
    """
    if click_window is None or image_w <= 0 or image_h <= 0:
        return None

    info = (
        click_window
        if isinstance(click_window, ClickWindowInfo)
        else ClickWindowInfo.from_dict(click_window)
    )
    if info is None:
        return None

    ox, oy = int(monitor_offset[0]), int(monitor_offset[1])
    live = find_matching_click_window(info)
    if live is not None:
        local = (
            int(live.left) - ox,
            int(live.top) - oy,
            int(live.width),
            int(live.height),
        )
    else:
        # Recorded payload may already be image-local (``rect``) with offset (0,0),
        # or screen-space left/top when rebuilt without going through to_local_payload.
        if isinstance(click_window, dict) and isinstance(click_window.get("rect"), (list, tuple)):
            local = info.screen_rect()
        else:
            local = (
                int(info.left) - ox,
                int(info.top) - oy,
                int(info.width),
                int(info.height),
            )

    return _clip_xywh_to_image(local, image_w=image_w, image_h=image_h)
