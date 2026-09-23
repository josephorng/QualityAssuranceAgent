from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mss
import pyautogui
import pyperclip
from pynput import keyboard, mouse

from src.common.io_utils import append_text, read_json, write_json
from src.common.run_state import unique_run_folder_name
from src.common.settings import load_settings
from src.eye.capture import capture_all_screens_to_file, resolve_monitor_index
from src.recorder.focus_point import resolve_typing_focus
from src.recorder.frame_similarity import (
    DEFAULT_SIMILARITY_THRESHOLD,
    append_settle_probe_debug_record,
    apply_settle_sample,
    archive_settle_probe_sample,
    compare_frames,
    last_settle_frame_path,
    settle_probe_kept_path,
    settle_probe_staging_path,
)
from src.recorder.models import (
    RecordedEvent,
    SessionManifest,
    event_json_path,
    final_after_screenshot_path,
    next_recording_event_index,
    recording_event_paths,
    screenshot_path_for_event,
    screenshot_path_for_event_end,
    utc_now_iso,
)
from src.recorder.window_snapshot import (
    ClickWindowInfo,
    WindowInfo,
    diff_snapshots_with_debug,
    resolve_click_window,
    settle_delay_for_click,
    snapshot_top_level_windows,
)
from src.recorder.hotkey import is_recording_toggle_hotkey

_DOUBLE_CLICK_INTERVAL_S = 0.35
_DOUBLE_CLICK_MAX_DIST_PX = 8
_DRAG_THRESHOLD_PX = 8
# Must exceed the double-click window so short presses still defer for double-click.
_HOLD_THRESHOLD_S = 0.5
# Pre-type frames are only reused when typing focus is still near the capture point.
# Enter→app launch→type into a new dialog moves focus far away; reuse would keep a
# stale Search/desktop frame as the text_input before-shot.
_PRE_TYPE_FOCUS_MAX_DIST_PX = 48
# After typing pauses, capture a settled frame for the next key_press before-shot.
# LL hooks cannot screenshot before Tab/Enter is delivered; reuse this frame instead.
_PRE_KEY_SETTLE_S = 0.3
# Consecutive-screenshot settle probe (UI stability after an action).
# Inter-sample delays follow Fibonacci seconds: 1, 1, 2, 3, 5, 8, 13, 21, 34, …
# Dense early samples catch immediate settles; backoff cuts captures on long waits.
_SETTLE_PROBE_MAX_WINDOW_S = 55.0
# Retain every settle-probe capture under screenshots/settle_debug/ for MAD tuning.
_SETTLE_PROBE_KEEP_DEBUG_SAMPLES = False
_SETTLE_PROBE_KINDS = frozenset(
    {
        "click",
        "double_click",
        "triple_click",
        "right_click",
        "middle_click",
        "drag",
        "hold",
        "text_input",
        "key_press",
    }
)
_QUEUE_SENTINEL = object()
_DRAIN_MARKER = object()


def settle_probe_interval_s(step: int) -> float:
    """Return the Fibonacci delay (seconds) before settle sample ``step``.

    ``step`` 0 → 1, 1 → 1, 2 → 2, 3 → 3, 4 → 5, 5 → 8, …
    """
    n = max(0, int(step))
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return float(a)


# First probe delay (same as ``settle_probe_interval_s(0)``); kept for callers/tests.
_SETTLE_PROBE_FIRST_S = settle_probe_interval_s(0)


@dataclass
class _SettleProbeState:
    """In-flight consecutive-frame settle measurement for one persisted event."""

    event_index: int
    started_monotonic: float
    cursor_xy: tuple[int, int] | None = None
    kept_path: str | None = None
    kept_age_s: float | None = None
    last_mon_index: int | None = None
    last_mon_offset: tuple[int, int] | None = None
    sample_seq: int = 0
    kept_sample_seq: int | None = None
    # Next Fibonacci index for ``settle_probe_interval_s`` when scheduling a tick.
    interval_step: int = 0


@dataclass(frozen=True)
class _PendingPreTypeShot:
    """Settled UI frame captured before the next text_input or key_press."""

    path: str
    monitor_index: int
    monitor_offset: tuple[int, int]
    focus_xy: tuple[int, int]

    def as_finalize_tuple(self) -> tuple[str, int, tuple[int, int]]:
        return self.path, self.monitor_index, self.monitor_offset


@dataclass
class _DeferredCaptureJob:
    """Screenshot / UIA work deferred off the low-level input hook threads.

    Windows silently removes ``WH_KEYBOARD_LL`` / ``WH_MOUSE_LL`` hooks when the
    callback takes too long. Keep hook handlers cheap; run capture work here.
    """

    action: str
    # begin_text_input | flush_text_input | keyboard_event | settle_pre_key
    # | pending_left_press | pending_right_press | pending_drag_end
    # | emit_left_gesture | emit_right_gesture | mouse_pointer_event
    # | seed_settle_windows | read_clipboard_paste
    meta: dict[str, Any] | None = None
    pending_pre_type: _PendingPreTypeShot | None = None
    pending_pre_key: _PendingPreTypeShot | None = None
    last_click_xy: tuple[int, int] | None = None
    mouse_xy: tuple[int, int] | None = None
    flush_chars: list[str] | None = None
    flush_meta: dict[str, Any] | None = None
    shared_end_index: int | None = None
    shared_end_monitor: int | None = None
    shared_end_offset: tuple[int, int] | None = None
    kind: str | None = None
    cursor_xy: tuple[int, int] | None = None
    end_xy: tuple[int, int] | None = None
    event_index: int | None = None
    timestamp_utc: str | None = None
    key: str | None = None
    keys: list[str] | None = None
    text: str | None = None
    button: str | None = None
    modifiers: list[str] | None = None
    scroll_delta: int | None = None
    duration_seconds: float | None = None
    press_seq: int | None = None
    text_event_index: int | None = None
    windows_before: tuple[WindowInfo, ...] | None = None
    click_window: ClickWindowInfo | None = None
    refresh_pre_type: bool = False


def _pre_type_focus_still_valid(
    pending_focus: tuple[int, int] | None,
    current_focus: tuple[int, int] | None,
    *,
    max_dist_px: float = _PRE_TYPE_FOCUS_MAX_DIST_PX,
) -> bool:
    """Return True when the pre-type frame still matches the live typing focus."""
    if pending_focus is None or current_focus is None:
        return False
    dx = int(pending_focus[0]) - int(current_focus[0])
    dy = int(pending_focus[1]) - int(current_focus[1])
    return (dx * dx + dy * dy) <= int(max_dist_px * max_dist_px)


_LISTENER_STARTUP_TIMEOUT_S = 2.0
_SPECIAL_KEYS = frozenset(
    {
        keyboard.Key.enter,
        keyboard.Key.tab,
        keyboard.Key.backspace,
        keyboard.Key.delete,
        keyboard.Key.esc,
        keyboard.Key.up,
        keyboard.Key.down,
        keyboard.Key.left,
        keyboard.Key.right,
        keyboard.Key.home,
        keyboard.Key.end,
        keyboard.Key.page_up,
        keyboard.Key.page_down,
        keyboard.Key.insert,
        keyboard.Key.f1,
        keyboard.Key.f2,
        keyboard.Key.f3,
        keyboard.Key.f4,
        keyboard.Key.f5,
        keyboard.Key.f6,
        keyboard.Key.f7,
        keyboard.Key.f8,
        keyboard.Key.f9,
        keyboard.Key.f10,
        keyboard.Key.f11,
        keyboard.Key.f12,
    }
)
_HOTKEY_SUPPRESS_KEYS = frozenset(
    {
        keyboard.Key.ctrl,
        keyboard.Key.ctrl_l,
        keyboard.Key.ctrl_r,
        keyboard.Key.shift,
        keyboard.Key.shift_l,
        keyboard.Key.shift_r,
        keyboard.KeyCode.from_char("r"),
        keyboard.KeyCode.from_char("R"),
    }
)

# Windows VK codes for the numeric keypad (Num Lock on). pynput often
# delivers these as KeyCode(vk=…) with char=None, so map them explicitly.
_NUMPAD_VK_TO_CHAR: dict[int, str] = {
    96: "0",
    97: "1",
    98: "2",
    99: "3",
    100: "4",
    101: "5",
    102: "6",
    103: "7",
    104: "8",
    105: "9",
    106: "*",
    107: "+",
    109: "-",
    110: ".",
    111: "/",
}
# Windows VK_A..VK_Z
_VK_A = 65
_VK_Z = 90

IgnoreRect = tuple[int, int, int, int]
# Provider may return one rect, several rects (hub + stop overlay), or None.
IgnoreRectProvider = Callable[[], IgnoreRect | list[IgnoreRect] | None]


@dataclass(frozen=True)
class _QueuedEvent:
    kind: str
    cursor_xy: tuple[int, int] | None
    event_index: int
    timestamp_utc: str
    screenshot_path: str = ""
    monitor_index: int | None = None
    monitor_offset: tuple[int, int] | None = None
    button: str | None = None
    modifiers: list[str] | None = None
    key: str | None = None
    keys: list[str] | None = None
    text: str | None = None
    scroll_delta: int | None = None
    duration_seconds: float | None = None
    anchor_click_xy: tuple[int, int] | None = None
    focus_rect: tuple[int, int, int, int] | None = None
    end_xy: tuple[int, int] | None = None
    end_screenshot_path: str = ""
    end_monitor_index: int | None = None
    end_monitor_offset: tuple[int, int] | None = None
    windows_before: tuple[WindowInfo, ...] | None = None
    # Press-time ClickWindowInfo (screen coords); localized at persist.
    click_window: ClickWindowInfo | None = None


def _pending_capture_path(run_dir: Path) -> Path:
    return run_dir / "screenshots" / "_pending_capture.jpeg"


def _pending_right_capture_path(run_dir: Path) -> Path:
    return run_dir / "screenshots" / "_pending_right_capture.jpeg"


def _pending_pre_type_capture_path(run_dir: Path) -> Path:
    return run_dir / "screenshots" / "_pending_pre_type.jpeg"


def _pending_pre_key_capture_path(run_dir: Path) -> Path:
    return run_dir / "screenshots" / "_pending_pre_key.jpeg"


def _pending_drag_end_capture_path(run_dir: Path, monitor_index: int) -> Path:
    return run_dir / "screenshots" / f"_pending_drag_end_mon{monitor_index}.jpeg"


def _capture_all_monitors_to_pending(
    run_dir: Path,
) -> dict[int, tuple[str, int, tuple[int, int]]]:
    """Capture every physical monitor into temporary drag-end pending files."""
    captures: dict[int, tuple[str, int, tuple[int, int]]] = {}
    with mss.mss() as sct:
        for raw_idx in range(1, len(sct.monitors)):
            mon_idx = resolve_monitor_index(sct, raw_idx)
            monitor = sct.monitors[mon_idx]
            dest = _pending_drag_end_capture_path(run_dir, mon_idx)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shot = sct.grab(monitor)
            from PIL import Image

            img = Image.frombytes("RGB", shot.size, shot.rgb)
            img.save(dest, format="JPEG")
            captures[mon_idx] = (
                str(dest),
                mon_idx,
                (int(monitor["left"]), int(monitor["top"])),
            )
    return captures


def _discard_pending_drag_end_capture_files(
    captures: dict[int, tuple[str, int, tuple[int, int]]] | None,
) -> None:
    if not captures:
        return
    for path, _, _ in captures.values():
        pending = Path(path)
        if pending.is_file():
            pending.unlink()


def _finalize_drag_end_screenshot(
    run_dir: Path,
    index: int,
    end_xy: tuple[int, int],
    pending_captures: dict[int, tuple[str, int, tuple[int, int]]] | None,
    *,
    fallback_mon_idx: int,
    fallback_mon_offset: tuple[int, int],
) -> tuple[str, int, tuple[int, int]]:
    """Pick the pre-captured monitor at ``end_xy`` and save it as ``event_{index}_end``."""
    end_dest = screenshot_path_for_event_end(run_dir, index)
    if not pending_captures:
        try:
            return _capture_screenshot_at_point(end_xy[0], end_xy[1], end_dest)
        except Exception:
            return str(end_dest), fallback_mon_idx, fallback_mon_offset

    raw_idx, _, _, _, _ = _monitor_at_point(end_xy[0], end_xy[1])
    with mss.mss() as sct:
        end_mon_idx = resolve_monitor_index(sct, raw_idx)

    entry = pending_captures.get(end_mon_idx)
    if entry is None:
        try:
            return _capture_screenshot_at_point(end_xy[0], end_xy[1], end_dest)
        except Exception:
            return str(end_dest), fallback_mon_idx, fallback_mon_offset

    src = Path(entry[0])
    end_dest.parent.mkdir(parents=True, exist_ok=True)
    if end_dest.is_file():
        end_dest.unlink()
    if src.is_file():
        src.replace(end_dest)
    else:
        try:
            return _capture_screenshot_at_point(end_xy[0], end_xy[1], end_dest)
        except Exception:
            return str(end_dest), fallback_mon_idx, fallback_mon_offset

    for mon_idx, (path, _, _) in pending_captures.items():
        if mon_idx == end_mon_idx:
            continue
        pending = Path(path)
        if pending.is_file():
            pending.unlink()

    return str(end_dest), entry[1], entry[2]


def _finalize_screenshot(
    run_dir: Path,
    index: int,
    cursor_xy: tuple[int, int],
    pending: tuple[str, int, tuple[int, int]] | None,
) -> tuple[str, int, tuple[int, int]]:
    """Move a pre-captured pending screenshot or capture now into ``event_{index}``."""
    dest = screenshot_path_for_event(run_dir, index)
    if pending is not None:
        src = Path(pending[0])
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file():
                dest.unlink()
            src.replace(dest)
            return str(dest), pending[1], pending[2]
    try:
        return _capture_screenshot_at_point(cursor_xy[0], cursor_xy[1], dest)
    except Exception:
        return str(dest), 0, (0, 0)


def _monitor_at_point(x: int, y: int) -> tuple[int, int, int, int, int]:
    """Return monitor index, left, top, width, height for a desktop point."""
    with mss.mss() as sct:
        for idx in range(1, len(sct.monitors)):
            mon = sct.monitors[idx]
            left = int(mon["left"])
            top = int(mon["top"])
            width = int(mon["width"])
            height = int(mon["height"])
            if left <= x < left + width and top <= y < top + height:
                return idx, left, top, width, height
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        return (
            1 if len(sct.monitors) > 1 else 0,
            int(mon["left"]),
            int(mon["top"]),
            int(mon["width"]),
            int(mon["height"]),
        )


def _capture_screenshot_at_point(x: int, y: int, dest: Path) -> tuple[str, int, tuple[int, int]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    idx, left, top, width, height = _monitor_at_point(x, y)
    with mss.mss() as sct:
        mon_idx = resolve_monitor_index(sct, idx)
        monitor = sct.monitors[mon_idx]
        shot = sct.grab(monitor)
        from PIL import Image

        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(dest, format="JPEG")
        return str(dest), mon_idx, (int(monitor["left"]), int(monitor["top"]))


def capture_final_after_screenshot(
    run_dir: Path,
    events: list[RecordedEvent],
) -> str | None:
    """Capture settled UI after the last action (before the hub window is restored).

    Always grabs the full virtual desktop (all monitors) so playback verification
    matches Eye's multi-monitor live frame. ``events`` is unused (kept for
    call-site compatibility). Returns the saved path, or ``None`` when capture fails.
    """
    _ = events
    dest = final_after_screenshot_path(run_dir)
    try:
        capture_all_screens_to_file(dest)
        return str(dest)
    except Exception:
        return None


def _normalize_button(button: mouse.Button) -> str:
    if button == mouse.Button.right:
        return "right"
    if button == mouse.Button.middle:
        return "middle"
    return "left"


def _ascii_control_to_letter(ch: str) -> str | None:
    """Map Ctrl+letter ASCII control codes (SOH..SUB) back to a..z.

    On Windows, pynput reports Ctrl+A as char='\\x01', Ctrl+C as '\\x03', etc.
    """
    if len(ch) != 1:
        return None
    code = ord(ch)
    if 1 <= code <= 26:
        return chr(ord("a") + code - 1)
    return None


def _vk_to_letter(vk: int) -> str | None:
    if _VK_A <= vk <= _VK_Z:
        return chr(ord("a") + (vk - _VK_A))
    return None


def _key_char(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    """Return the typed character for a key, including numpad / Ctrl VK fallbacks."""
    if not isinstance(key, keyboard.KeyCode):
        return None
    if key.char:
        if key.char.isprintable():
            return key.char
        # Ctrl+letter arrives as a non-printable control character on Windows.
        letter = _ascii_control_to_letter(key.char)
        if letter:
            return letter
    if key.vk is not None:
        vk = int(key.vk)
        numpad = _NUMPAD_VK_TO_CHAR.get(vk)
        if numpad:
            return numpad
        letter = _vk_to_letter(vk)
        if letter:
            return letter
    return None


def _key_token(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if isinstance(key, keyboard.KeyCode):
        ch = _key_char(key)
        if ch:
            return ch
        if key.vk is not None:
            return f"vk_{key.vk}"
        return None
    name = str(key).replace("Key.", "")
    return name


def _modifier_name(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        return "ctrl"
    if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_gr):
        return "alt"
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        return "shift"
    if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
        return "win"
    return None


_MODIFIER_TOKEN_ORDER = ("ctrl", "alt", "shift", "win")


def _ordered_modifiers(mods: set[str] | list[str]) -> list[str] | None:
    present = set(mods)
    ordered = [name for name in _MODIFIER_TOKEN_ORDER if name in present]
    extras = sorted(name for name in present if name not in _MODIFIER_TOKEN_ORDER)
    result = ordered + extras
    return result or None


def _safe_clipboard_text() -> str | None:
    """Best-effort read of the current clipboard as text."""
    try:
        return pyperclip.paste()
    except Exception:
        return None


def _is_paste_hotkey(mods: list[str], token: str) -> bool:
    return "ctrl" in mods and token.lower() == "v"


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    left, top, width, height = rect
    if width <= 0 or height <= 0:
        return False
    return left <= x < left + width and top <= y < top + height


def _normalize_ignore_rects(
    value: IgnoreRect | list[IgnoreRect] | None,
) -> list[IgnoreRect]:
    if value is None:
        return []
    if isinstance(value, list):
        return [rect for rect in value if rect is not None]
    return [value]


def _point_in_any_rect(x: int, y: int, rects: list[IgnoreRect]) -> bool:
    return any(_point_in_rect(x, y, rect) for rect in rects)


def _windows_is_admin() -> bool | None:
    if os.name != "nt":
        return None
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


def _format_windows_error(code: int) -> str:
    if code == 0:
        return "unknown error"
    kernel32 = ctypes.windll.kernel32
    buf = ctypes.create_unicode_buffer(512)
    chars = kernel32.FormatMessageW(0x00001000, None, code, 0, buf, len(buf), None)
    if chars:
        return buf.value.strip()
    return f"Win32 error {code}"


def _probe_low_level_hook() -> tuple[bool, int | None]:
    """Install and remove WH_MOUSE_LL to detect hook blocking."""
    if os.name != "nt":
        return True, None
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hook_proc = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p)(
        lambda *_args: 0
    )
    hook = user32.SetWindowsHookExW(14, hook_proc, None, 0)
    if not hook:
        return False, int(kernel32.GetLastError())
    user32.UnhookWindowsHookEx(hook)
    return True, None


def _listener_thread_error(
    listener: mouse.Listener | keyboard.Listener,
    *,
    timeout_s: float = 0.05,
) -> str | None:
    try:
        listener.join(timeout=timeout_s)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _diagnose_listener_startup(
    label: str,
    listener: mouse.Listener | keyboard.Listener,
    *,
    timeout_s: float = _LISTENER_STARTUP_TIMEOUT_S,
) -> str | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not listener.is_alive():
            detail = _listener_thread_error(listener)
            if detail:
                return f"{label} listener 執行緒已結束 ({detail})"
            return f"{label} listener 執行緒已結束"
        if listener.running:
            detail = _listener_thread_error(listener)
            if detail:
                return f"{label} listener 啟動失敗 ({detail})"
            return None
        time.sleep(0.05)

    if not listener.is_alive():
        detail = _listener_thread_error(listener)
        if detail:
            return f"{label} listener 執行緒已結束 ({detail})"
        return f"{label} listener 執行緒已結束"
    if not listener.running:
        return f"{label} listener 未進入 running 狀態 (等待 {timeout_s:.1f}s 逾時)"
    return None


def _build_input_listener_error(issues: list[str]) -> str:
    lines = ["無法啟動全域輸入監聽。"]
    lines.extend(issues)

    if os.name == "nt":
        admin = _windows_is_admin()
        if admin is not None:
            lines.append(f"目前程序管理員權限：{'是' if admin else '否'}")
        hook_ok, err_code = _probe_low_level_hook()
        if not hook_ok and err_code is not None:
            lines.append(f"Windows low-level hook 探測失敗：{_format_windows_error(err_code)}")

    lines.append(
        "可能原因：防毒軟體封鎖 keyboard/mouse hook、公司安全政策、或 listener 執行緒異常退出。"
    )
    if os.name == "nt" and _windows_is_admin() is False:
        lines.append("若僅在操作「以系統管理員執行」的程式時失敗，可嘗試以系統管理員身分執行此程式。")
    return "\n".join(lines)


def _wait_for_input_listeners(
    mouse_listener: mouse.Listener,
    keyboard_listener: keyboard.Listener,
) -> list[str]:
    issues: list[str] = []
    for label, listener in (("滑鼠", mouse_listener), ("鍵盤", keyboard_listener)):
        issue = _diagnose_listener_startup(label, listener)
        if issue:
            issues.append(issue)
    return issues


class RecordingSession:
    """Capture desktop input events with per-event screenshots."""

    def __init__(self, runs_root: Path | None = None) -> None:
        settings = load_settings()
        self._runs_root = Path(runs_root or settings.recordings_dir)
        self._lock = threading.Lock()
        self._accepting_input = False
        self._run_dir: Path | None = None
        self._run_id: str | None = None
        self._events: list[RecordedEvent] = []
        self._next_index = 1
        self._started_at: str | None = None
        self._ignore_rect_provider: IgnoreRectProvider | None = None
        self._suppress_hotkey_keys = False
        self._mouse_listener: mouse.Listener | None = None
        self._keyboard_listener: keyboard.Listener | None = None
        self._pressed_modifiers: set[str] = set()
        self._pending_click_timer: threading.Timer | None = None
        self._pending_click_coords: tuple[int, int, str] | None = None
        self._pending_click_down_at: float | None = None
        self._pending_click_timestamp_utc: str | None = None
        self._pending_click_modifiers: list[str] | None = None
        self._left_button_down = False
        self._left_press_dragging = False
        self._last_move_xy: tuple[int, int] | None = None
        self._pending_screenshot: tuple[str, int, tuple[int, int]] | None = None
        self._pending_drag_end_captures: dict[int, tuple[str, int, tuple[int, int]]] | None = None
        self._pending_windows_before: tuple[WindowInfo, ...] | None = None
        self._pending_click_window: ClickWindowInfo | None = None
        # Generation tokens so deferred press captures stay tied to the gesture that
        # enqueued them (hook thread must not wait on mss/UIA).
        self._left_press_seq: int = 0
        self._right_press_seq: int = 0
        self._left_press_captures: dict[
            int, tuple[tuple[str, int, tuple[int, int]] | None, tuple[WindowInfo, ...] | None]
        ] = {}
        self._right_press_captures: dict[
            int, tuple[tuple[str, int, tuple[int, int]] | None, tuple[WindowInfo, ...] | None]
        ] = {}
        self._left_press_click_windows: dict[int, ClickWindowInfo | None] = {}
        self._right_press_click_windows: dict[int, ClickWindowInfo | None] = {}
        self._drag_end_captures: dict[int, dict[int, tuple[str, int, tuple[int, int]]]] = {}
        self._pending_right_coords: tuple[int, int] | None = None
        self._pending_right_down_at: float | None = None
        self._pending_right_timestamp_utc: str | None = None
        self._pending_right_modifiers: list[str] | None = None
        self._pending_right_screenshot: tuple[str, int, tuple[int, int]] | None = None
        self._pending_right_windows_before: tuple[WindowInfo, ...] | None = None
        self._pending_right_click_window: ClickWindowInfo | None = None
        self._pending_text_chars: list[str] = []
        self._pending_text_caret: int = 0
        self._pending_text_meta: dict[str, Any] | None = None
        self._paste_seq: int = 0
        # Empty-field frame captured after focus settles (click/Tab/etc.), consumed
        # as the text_input before-shot so we do not race the first typed glyph.
        self._pending_pre_type_screenshot: _PendingPreTypeShot | None = None
        # Settled frame for the next key_press before-shot (cannot capture on LL hook).
        self._pending_pre_key_screenshot: _PendingPreTypeShot | None = None
        self._pending_pre_key_timer: threading.Timer | None = None
        self._settle_probe: _SettleProbeState | None = None
        self._settle_probe_timer: threading.Timer | None = None
        # Serialize settle ticks (timer thread) vs cancel/start (capture worker).
        self._settle_tick_lock = threading.Lock()
        # Last settle-probe frame (path, monitor_index, monitor_offset) for next before-shot.
        self._last_settle_frame: tuple[str, int, tuple[int, int]] | None = None
        # Top-level windows captured with that frame; mouse-down reuses this as
        # windows_before so EnumWindows never runs on the LL hook.
        self._last_settle_windows: tuple[WindowInfo, ...] | None = None
        self._last_pointer_cursor_xy: tuple[int, int] | None = None
        self._event_queue: queue.Queue[object] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._on_event: Callable[[RecordedEvent], None] | None = None
        self._finalizing = False

    def set_on_event(self, callback: Callable[[RecordedEvent], None] | None) -> None:
        self._on_event = callback

    def is_active(self) -> bool:
        with self._lock:
            return self._accepting_input

    def is_finalizing(self) -> bool:
        with self._lock:
            return self._finalizing

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def run_dir(self) -> Path | None:
        with self._lock:
            return self._run_dir

    def start(
        self,
        *,
        ignore_rect: tuple[int, int, int, int] | None = None,
        ignore_rect_provider: IgnoreRectProvider | None = None,
        existing_run_dir: Path | None = None,
    ) -> Path:
        if ignore_rect_provider is not None:
            provider = ignore_rect_provider
        elif ignore_rect is not None:
            provider = lambda rect=ignore_rect: rect
        else:
            provider = lambda: None

        with self._lock:
            if self._accepting_input:
                raise RuntimeError("Recording is already active")
            if existing_run_dir is not None:
                run_dir = Path(existing_run_dir)
                if not run_dir.is_dir():
                    raise RuntimeError(f"Recording folder not found: {run_dir}")
                run_id = run_dir.name
                (run_dir / "events").mkdir(exist_ok=True)
                (run_dir / "screenshots").mkdir(exist_ok=True)
                (run_dir / "yolo_ocr").mkdir(exist_ok=True)
                next_index = next_recording_event_index(run_dir)
                session = read_json(run_dir / "session.json", {})
                started_at = utc_now_iso()
                if isinstance(session, dict):
                    raw_started = session.get("started_at_utc")
                    if isinstance(raw_started, str) and raw_started.strip():
                        started_at = raw_started.strip()
                    raw_run_id = session.get("run_id")
                    if isinstance(raw_run_id, str) and raw_run_id.strip():
                        run_id = raw_run_id.strip()
            else:
                run_id = unique_run_folder_name("recording")
                run_dir = self._runs_root / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "events").mkdir(exist_ok=True)
                (run_dir / "screenshots").mkdir(exist_ok=True)
                (run_dir / "yolo_ocr").mkdir(exist_ok=True)
                next_index = 1
                started_at = utc_now_iso()
            self._run_dir = run_dir
            self._run_id = run_id
            self._events = []
            self._next_index = next_index
            self._started_at = started_at
            self._ignore_rect_provider = provider
            self._accepting_input = False
            self._pressed_modifiers = set()
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
            self._pending_screenshot = None
            self._pending_drag_end_captures = None
            self._pending_windows_before = None
            self._pending_click_window = None
            self._left_press_seq = 0
            self._right_press_seq = 0
            self._left_press_captures = {}
            self._right_press_captures = {}
            self._left_press_click_windows = {}
            self._right_press_click_windows = {}
            self._drag_end_captures = {}
            self._pending_right_coords = None
            self._pending_right_down_at = None
            self._pending_right_timestamp_utc = None
            self._pending_right_modifiers = None
            self._pending_right_screenshot = None
            self._pending_right_windows_before = None
            self._pending_right_click_window = None
            self._pending_text_chars = []
            self._pending_text_meta = None
            self._paste_seq = 0
            self._pending_pre_type_screenshot = None
            self._pending_pre_key_screenshot = None
            self._cancel_pre_key_settle_timer_locked()
            self._settle_probe = None
            self._settle_probe_timer = None
            self._last_settle_frame = None
            self._last_settle_windows = None
            self._last_pointer_cursor_xy = None
            pending = self._pending_click_timer
            self._pending_click_timer = None
            self._event_queue = queue.Queue()

        if pending is not None:
            pending.cancel()

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="screen-recorder-worker",
            daemon=True,
        )
        self._worker_thread.start()

        seeded = threading.Event()
        self._enqueue(
            _DeferredCaptureJob(action="seed_settle_windows", meta={"done": seeded})
        )
        if not seeded.wait(5.0):
            self._log(run_dir, "settle window seed timed out")
        with self._lock:
            self._accepting_input = True

        self._mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
            on_move=self._on_mouse_move,
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener.start()
        self._keyboard_listener.start()

        listener_issues = _wait_for_input_listeners(
            self._mouse_listener,
            self._keyboard_listener,
        )
        if listener_issues:
            error_message = _build_input_listener_error(listener_issues)
            self.stop()
            raise RuntimeError(error_message)

        self._log(
            run_dir,
            "recording continued" if existing_run_dir is not None else "recording started",
        )
        return run_dir

    def begin_stop(self) -> Path | None:
        """Stop capturing input and return promptly; call ``finalize_stop`` to finish."""
        listeners: list[mouse.Listener | keyboard.Listener] = []
        with self._lock:
            if self._finalizing:
                return self._run_dir
            if not self._accepting_input and self._run_dir is None:
                return None
            self._accepting_input = False
            self._finalizing = True
            if self._mouse_listener is not None:
                listeners.append(self._mouse_listener)
                self._mouse_listener = None
            if self._keyboard_listener is not None:
                listeners.append(self._keyboard_listener)
                self._keyboard_listener = None
            run_dir = self._run_dir
            pending = self._pending_click_timer
            pending_coords = self._pending_click_coords
            pending_down_at = self._pending_click_down_at
            left_press_dragging = self._left_press_dragging
            last_move_xy = self._last_move_xy
            pending_right_coords = self._pending_right_coords
            pending_right_down_at = self._pending_right_down_at
            self._pending_click_timer = None

        self._flush_pending_text_input()
        if pending is not None:
            pending.cancel()
        if left_press_dragging and pending_coords is not None and last_move_xy is not None:
            sx, sy, button = pending_coords
            ex, ey = last_move_xy
            if abs(sx - ex) > _DRAG_THRESHOLD_PX or abs(sy - ey) > _DRAG_THRESHOLD_PX:
                self._flush_pending_drag(sx, sy, ex, ey, button)
            else:
                hold_duration = (
                    time.monotonic() - pending_down_at if pending_down_at is not None else 0.0
                )
                if hold_duration >= _HOLD_THRESHOLD_S:
                    self._flush_pending_hold(sx, sy, button, hold_duration)
                else:
                    self._flush_pending_click(sx, sy, button)
        elif pending_coords is not None:
            x, y, button = pending_coords
            hold_duration = (
                time.monotonic() - pending_down_at if pending_down_at is not None else 0.0
            )
            if hold_duration >= _HOLD_THRESHOLD_S:
                self._flush_pending_hold(x, y, button, hold_duration)
            else:
                self._flush_pending_click(x, y, button)

        if pending_right_coords is not None:
            rx, ry = pending_right_coords
            right_hold = (
                time.monotonic() - pending_right_down_at
                if pending_right_down_at is not None
                else 0.0
            )
            if right_hold >= _HOLD_THRESHOLD_S:
                self._flush_pending_right(rx, ry, kind="hold", duration_seconds=right_hold)
            else:
                self._flush_pending_right(rx, ry, kind="right_click")

        with self._lock:
            leftover_drag_end = self._pending_drag_end_captures
            self._pending_drag_end_captures = None
            stale_right, pending_right_shot = self._clear_pending_right_gesture_locked(
                discard_capture=True,
            )
            self._pending_screenshot = None
            self._pending_windows_before = None
            self._pending_click_window = None
            leftover_pre_type = self._pending_pre_type_screenshot
            self._pending_pre_type_screenshot = None
            leftover_pre_key = self._pending_pre_key_screenshot
            self._pending_pre_key_screenshot = None
            self._cancel_pre_key_settle_timer_locked()
        _discard_pending_drag_end_capture_files(leftover_drag_end)
        self._discard_press_capture_entry(stale_right)
        if pending_right_shot is not None:
            right_pending = Path(pending_right_shot[0])
            if right_pending.is_file():
                try:
                    right_pending.unlink()
                except OSError:
                    pass
        if leftover_pre_type is not None:
            pending_pre = Path(leftover_pre_type.path)
            if pending_pre.is_file():
                try:
                    pending_pre.unlink()
                except OSError:
                    pass
        if leftover_pre_key is not None:
            pending_key = Path(leftover_pre_key.path)
            if pending_key.is_file():
                try:
                    pending_key.unlink()
                except OSError:
                    pass

        for listener in listeners:
            try:
                listener.stop()
            except Exception:
                pass

        # Keep the settle probe running through finalize so the last action can
        # match samples against ``final_after`` (do not cancel here).
        self._event_queue.put(_QUEUE_SENTINEL)
        return run_dir

    def finalize_stop(
        self,
        *,
        on_after_screenshot: Callable[[], None] | None = None,
    ) -> Path | None:
        """Wait for queued events, capture final screenshot, and write session artifacts.

        ``on_after_screenshot`` runs on this thread immediately after the final-after
        frame is captured (or capture fails), before slower session HTML / cleanup work.
        """
        with self._lock:
            if not self._finalizing:
                return self._run_dir
            run_dir = self._run_dir
            run_id = self._run_id
            started_at = self._started_at
            worker = self._worker_thread

        if worker is not None and worker.is_alive():
            worker.join(timeout=15)

        with self._lock:
            events = list(self._events)

        if run_dir is None or run_id is None or started_at is None:
            self._cancel_settle_probe(cleanup_files=True, reason="recording_stop_empty")
            with self._lock:
                self._finalizing = False
            if on_after_screenshot is not None:
                try:
                    on_after_screenshot()
                except Exception:
                    pass
            return None

        # Capture settled UI before the hub restores (caller deiconifies after this).
        final_after = capture_final_after_screenshot(run_dir, events)
        if final_after is not None:
            self._log(run_dir, f"final after screenshot saved path={final_after}")
            self._finish_settle_against_final_after(Path(final_after))
        else:
            self._log(run_dir, "final after screenshot capture failed")
            self._cancel_settle_probe(
                cleanup_files=True,
                reason="recording_stop_no_final_after",
            )
        if on_after_screenshot is not None:
            try:
                on_after_screenshot()
            except Exception:
                pass

        event_paths = recording_event_paths(run_dir)
        manifest = SessionManifest(
            run_id=run_id,
            started_at_utc=started_at,
            stopped_at_utc=utc_now_iso(),
            event_count=len(event_paths),
            events=[path.relative_to(run_dir).as_posix() for path in event_paths],
            final_after_screenshot=(
                "screenshots/final_after.jpeg" if final_after is not None else None
            ),
        )
        write_json(run_dir / "session.json", manifest.to_dict())
        self._log(run_dir, f"recording stopped events={len(event_paths)}")
        try:
            from src.common.session_html import write_recording_html_from_run

            write_recording_html_from_run(run_dir)
        except Exception as exc:
            self._log(run_dir, f"recording html write failed: {exc}")

        with self._lock:
            self._finalizing = False
        return run_dir

    def stop(self) -> Path | None:
        """Synchronously stop and finalize a recording session."""
        self.begin_stop()
        if not self.is_finalizing():
            return None
        return self.finalize_stop()

    def set_suppress_hotkey_keys(self, suppress: bool) -> None:
        with self._lock:
            self._suppress_hotkey_keys = suppress

    def _log(self, run_dir: Path, text: str) -> None:
        append_text(run_dir / "record.log", f"{utc_now_iso()} {text}\n")

    def _notify_event(self, event: RecordedEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass

    def _current_ignore_rects(self) -> list[IgnoreRect]:
        provider = self._ignore_rect_provider
        if provider is None:
            return []
        try:
            return _normalize_ignore_rects(provider())
        except Exception:
            return []

    def _should_ignore_mouse_point(self, x: int, y: int) -> bool:
        with self._lock:
            if not self._accepting_input:
                return True
        return _point_in_any_rect(x, y, self._current_ignore_rects())

    def _enqueue(self, item: _QueuedEvent) -> None:
        self._event_queue.put(item)

    def _capture_immediate_screenshot(
        self,
        run_dir: Path,
        index: int,
        cursor_xy: tuple[int, int],
        *,
        dest: Path | None = None,
    ) -> tuple[str, int, tuple[int, int]]:
        target = dest if dest is not None else screenshot_path_for_event(run_dir, index)
        try:
            return _capture_screenshot_at_point(cursor_xy[0], cursor_xy[1], target)
        except Exception:
            return str(target), 0, (0, 0)

    def _refresh_last_settle_windows(self) -> None:
        """Store the current top-level window list next to the settle frame.

        Called from the settle loop and once at recording start (worker thread),
        never from the mouse hook.
        """
        try:
            windows = tuple(snapshot_top_level_windows())
        except Exception:
            return
        with self._lock:
            self._last_settle_windows = windows

    def _cached_windows_before(self) -> tuple[WindowInfo, ...] | None:
        """Pre-click window list from the last settle sample (or the start seed)."""
        with self._lock:
            cached = self._last_settle_windows
        if not cached:
            return None
        return cached

    def _discard_press_capture_entry(
        self,
        entry: tuple[
            tuple[str, int, tuple[int, int]] | None,
            tuple[WindowInfo, ...] | None,
        ]
        | None,
    ) -> None:
        if entry is None:
            return
        shot = entry[0]
        if shot is None:
            return
        path = Path(shot[0])
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass

    def _discard_drag_end_capture_entry(
        self,
        captures: dict[int, tuple[str, int, tuple[int, int]]] | None,
    ) -> None:
        _discard_pending_drag_end_capture_files(captures)

    def _capture_pending_left_press(
        self,
        run_dir: Path,
        x: int,
        y: int,
        *,
        text_event_index: int | None = None,
        text_meta: dict[str, Any] | None = None,
    ) -> None:
        """Reuse the settle-loop window cache; defer only the before-shot off the hook.

        ``windows_before`` is the last settle snapshot (seeded at recording start),
        so title-bar close still diffs against the pre-click window list without
        enumerating windows inside ``WH_MOUSE_LL``.
        """
        _ = run_dir
        windows_before = self._cached_windows_before()
        click_window = self._click_window_payload_at(x, y)
        with self._lock:
            self._left_press_seq += 1
            seq = self._left_press_seq
            self._pending_screenshot = None
            self._pending_windows_before = windows_before or None
            self._pending_click_window = click_window
            self._left_press_captures[seq] = (None, windows_before or None)
            self._left_press_click_windows[seq] = click_window
        self._enqueue(
            _DeferredCaptureJob(
                action="pending_left_press",
                cursor_xy=(x, y),
                press_seq=seq,
                text_event_index=text_event_index,
                flush_meta=text_meta,
            )
        )

    def _capture_pending_right_press(self, run_dir: Path, x: int, y: int) -> None:
        """Reuse the settle-loop window cache; defer only the right-press before-shot."""
        _ = run_dir
        windows_before = self._cached_windows_before()
        click_window = self._click_window_payload_at(x, y)
        with self._lock:
            self._right_press_seq += 1
            seq = self._right_press_seq
            self._pending_right_screenshot = None
            self._pending_right_windows_before = windows_before or None
            self._pending_right_click_window = click_window
            self._right_press_captures[seq] = (None, windows_before or None)
            self._right_press_click_windows[seq] = click_window
        self._enqueue(
            _DeferredCaptureJob(
                action="pending_right_press",
                cursor_xy=(x, y),
                press_seq=seq,
            )
        )

    def _pending_screenshot_from_settle_or_capture(
        self,
        run_dir: Path,
        x: int,
        y: int,
        dest: Path,
    ) -> tuple[str, int, tuple[int, int]]:
        """Prefer the last settle-probe frame on the same monitor; else live capture.

        Reusing the settled frame avoids mid-animation before-shots when the user
        clicks soon after a navigation/animation from the previous action.
        """
        click_mon, _, _, _, _ = _monitor_at_point(x, y)
        with self._lock:
            last = self._last_settle_frame
        if last is not None:
            last_path, last_mon, last_offset = last
            src = Path(last_path)
            if src.is_file() and int(last_mon) == int(click_mon):
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if dest.is_file():
                        dest.unlink()
                except OSError:
                    pass
                try:
                    import shutil

                    shutil.copy2(src, dest)
                    return str(dest), int(last_mon), tuple(last_offset)  # type: ignore[return-value]
                except OSError:
                    pass
        return _capture_screenshot_at_point(x, y, dest)

    def _publish_last_settle_frame(
        self,
        source: Path,
        mon_index: int,
        mon_offset: tuple[int, int],
    ) -> None:
        """Copy ``source`` into the durable next-before candidate frame."""
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None or not source.is_file():
            return
        dest = last_settle_frame_path(run_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest.resolve() != source.resolve():
                import shutil

                shutil.copy2(source, dest)
        except OSError:
            return
        with self._lock:
            self._last_settle_frame = (str(dest), int(mon_index), (int(mon_offset[0]), int(mon_offset[1])))
        self._refresh_last_settle_windows()

    def _queue_event(self, item: _QueuedEvent) -> None:
        self._enqueue(item)

    def _remember_pointer_cursor(self, cursor_xy: tuple[int, int]) -> None:
        with self._lock:
            self._last_pointer_cursor_xy = cursor_xy

    def _queue_pointer_event_immediate(
        self,
        *,
        kind: str,
        cursor_xy: tuple[int, int],
        button: str | None = None,
        scroll_delta: int | None = None,
        modifiers: list[str] | None = None,
        timestamp_utc: str | None = None,
    ) -> None:
        """Reserve an index and enqueue screenshot/UIA work off the mouse hook thread."""
        action_timestamp_utc = timestamp_utc or utc_now_iso()
        with self._lock:
            run_dir = self._run_dir
            if run_dir is None:
                return
            index = self._next_index
            self._next_index += 1
        self._flush_pending_text_input(shared_end_index=index)
        self._remember_pointer_cursor(cursor_xy)
        # Pre-click window list comes from the settle cache. click_window is one
        # WindowFromPoint (no full window scan) and stays on the hook.
        # Screenshot stays deferred.
        windows_before = self._cached_windows_before()
        click_window = self._click_window_payload_at(int(cursor_xy[0]), int(cursor_xy[1]))
        self._enqueue(
            _DeferredCaptureJob(
                action="mouse_pointer_event",
                kind=kind,
                cursor_xy=cursor_xy,
                event_index=index,
                timestamp_utc=action_timestamp_utc,
                button=button,
                modifiers=modifiers,
                scroll_delta=scroll_delta,
                windows_before=windows_before or None,
                click_window=click_window,
                refresh_pre_type=True,
            )
        )

    def _queue_keyboard_event_immediate(
        self,
        *,
        kind: str,
        cursor_xy: tuple[int, int] | None,
        key: str | None = None,
        keys: list[str] | None = None,
        text: str | None = None,
        timestamp_utc: str | None = None,
        event_index: int | None = None,
    ) -> None:
        """Reserve indices and enqueue screenshot work off the keyboard hook thread."""
        action_timestamp_utc = timestamp_utc or utc_now_iso()
        with self._lock:
            if self._run_dir is None:
                return
            if event_index is None:
                index = self._next_index
                self._next_index += 1
            else:
                index = event_index
            # Hand off the list object. A paste slot inside it can still be
            # filled by an earlier queued clipboard read before this job runs.
            flush_chars = self._pending_text_chars
            flush_meta = self._pending_text_meta
            self._pending_text_chars = []
            self._pending_text_caret = 0
            self._pending_text_meta = None
            self._cancel_pre_key_settle_timer_locked()
            pending_pre_key = self._pending_pre_key_screenshot
            self._pending_pre_key_screenshot = None
        shared_index = index if cursor_xy is not None else None
        self._enqueue(
            _DeferredCaptureJob(
                action="keyboard_event",
                kind=kind,
                cursor_xy=cursor_xy,
                event_index=index,
                timestamp_utc=action_timestamp_utc,
                key=key,
                keys=keys,
                text=text,
                flush_chars=flush_chars or None,
                flush_meta=flush_meta,
                shared_end_index=shared_index,
                pending_pre_key=pending_pre_key,
                refresh_pre_type=True,
            )
        )

    def _capture_typing_ocr_end_shot(
        self,
        run_dir: Path,
        text_index: int,
        x: int,
        y: int,
    ) -> tuple[str, int, tuple[int, int]] | None:
        """Capture the OCR-only post-typing frame at the typing focus point."""
        dest = screenshot_path_for_event_end(run_dir, text_index)
        if dest.is_file():
            try:
                mon_idx, left, top, _, _ = _monitor_at_point(x, y)
                return str(dest), mon_idx, (left, top)
            except Exception:
                return str(dest), 0, (0, 0)
        try:
            return self._capture_immediate_screenshot(
                run_dir,
                text_index,
                (x, y),
                dest=dest,
            )
        except Exception:
            return None

    def _discard_pending_pre_type_locked(self) -> None:
        pending = self._pending_pre_type_screenshot
        self._pending_pre_type_screenshot = None
        if pending is None:
            return
        path = Path(pending.path)
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass

    def _discard_pending_pre_key_locked(self) -> None:
        pending = self._pending_pre_key_screenshot
        self._pending_pre_key_screenshot = None
        if pending is None:
            return
        path = Path(pending.path)
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass

    def _cancel_pre_key_settle_timer_locked(self) -> None:
        pending = self._pending_pre_key_timer
        self._pending_pre_key_timer = None
        if pending is not None:
            pending.cancel()

    def _schedule_pre_key_settle(self) -> None:
        """After typing pauses, capture a before-shot for the next Tab/Enter/etc."""
        with self._lock:
            if (
                self._run_dir is None
                or not self._accepting_input
                or self._finalizing
                or self._pending_text_meta is None
            ):
                return
            self._cancel_pre_key_settle_timer_locked()
            timer = threading.Timer(_PRE_KEY_SETTLE_S, self._on_pre_key_settle_timer)
            timer.daemon = True
            self._pending_pre_key_timer = timer
            timer.start()

    def _on_pre_key_settle_timer(self) -> None:
        with self._lock:
            self._pending_pre_key_timer = None
            if (
                self._run_dir is None
                or not self._accepting_input
                or self._finalizing
                or self._pending_text_meta is None
            ):
                return
        self._enqueue(_DeferredCaptureJob(action="settle_pre_key"))

    def _mirror_shot_to_pre_key(
        self,
        shot: _PendingPreTypeShot,
        run_dir: Path,
    ) -> None:
        """Copy a settled frame into the pre-key slot for the next key_press."""
        with self._lock:
            self._discard_pending_pre_key_locked()
            if (
                self._run_dir is None
                or not self._accepting_input
                or self._finalizing
            ):
                return
        dest = _pending_pre_key_capture_path(run_dir)
        src = Path(shot.path)
        try:
            if src.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
            elif not dest.is_file():
                return
        except OSError:
            return
        with self._lock:
            if (
                self._run_dir is None
                or not self._accepting_input
                or self._finalizing
            ):
                if dest.is_file():
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                return
            self._pending_pre_key_screenshot = _PendingPreTypeShot(
                path=str(dest),
                monitor_index=shot.monitor_index,
                monitor_offset=shot.monitor_offset,
                focus_xy=shot.focus_xy,
            )

    def _refresh_pending_pre_key(self) -> None:
        """Capture the current typing UI as the next key_press before-shot."""
        with self._lock:
            run_dir = self._run_dir
            if (
                run_dir is None
                or not self._accepting_input
                or self._finalizing
                or self._pending_text_meta is None
            ):
                return
            last_click_xy = self._last_pointer_cursor_xy
            text_meta = self._pending_text_meta
            self._discard_pending_pre_key_locked()

        mouse_xy = None
        try:
            pos = pyautogui.position()
            mouse_xy = (int(pos.x), int(pos.y))
        except Exception:
            pass

        focus_xy = None
        if isinstance(text_meta, dict):
            raw_focus = text_meta.get("cursor_xy")
            if isinstance(raw_focus, (tuple, list)) and len(raw_focus) == 2:
                focus_xy = (int(raw_focus[0]), int(raw_focus[1]))
        if focus_xy is None:
            typing_focus = resolve_typing_focus(
                last_click_xy=last_click_xy,
                mouse_xy=mouse_xy,
            )
            focus_xy = typing_focus.point
        if focus_xy is None:
            return

        pending_dest = _pending_pre_key_capture_path(run_dir)
        try:
            path, mon_idx, mon_offset = _capture_screenshot_at_point(
                focus_xy[0],
                focus_xy[1],
                pending_dest,
            )
        except Exception:
            return

        with self._lock:
            if (
                self._run_dir is None
                or not self._accepting_input
                or self._finalizing
                or self._pending_text_meta is None
            ):
                stale = Path(path)
                if stale.is_file():
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                return
            self._pending_pre_key_screenshot = _PendingPreTypeShot(
                path=path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                focus_xy=(int(focus_xy[0]), int(focus_xy[1])),
            )

    def _refresh_pending_pre_type(
        self,
        cursor_xy: tuple[int, int] | None = None,
    ) -> None:
        """Capture an empty-field frame after focus settles for the next text_input."""
        with self._lock:
            run_dir = self._run_dir
            if (
                run_dir is None
                or not self._accepting_input
                or self._finalizing
                or self._pending_text_meta is not None
            ):
                return
            last_click_xy = self._last_pointer_cursor_xy
            self._discard_pending_pre_type_locked()

        mouse_xy = cursor_xy or last_click_xy
        try:
            pos = pyautogui.position()
            mouse_xy = (int(pos.x), int(pos.y))
        except Exception:
            pass

        typing_focus = resolve_typing_focus(
            last_click_xy=last_click_xy,
            mouse_xy=mouse_xy,
        )
        focus_xy = typing_focus.point
        if focus_xy is None:
            return

        pending_dest = _pending_pre_type_capture_path(run_dir)
        try:
            path, mon_idx, mon_offset = _capture_screenshot_at_point(
                focus_xy[0],
                focus_xy[1],
                pending_dest,
            )
        except Exception:
            return

        shot = _PendingPreTypeShot(
            path=path,
            monitor_index=mon_idx,
            monitor_offset=mon_offset,
            focus_xy=(int(focus_xy[0]), int(focus_xy[1])),
        )
        with self._lock:
            if (
                self._run_dir is None
                or not self._accepting_input
                or self._finalizing
                or self._pending_text_meta is not None
            ):
                stale = Path(path)
                if stale.is_file():
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                return
            self._pending_pre_type_screenshot = shot
        # Same settled UI is the best before-shot for a following Tab/Enter.
        self._mirror_shot_to_pre_key(shot, run_dir)

    def _flush_pending_text_input(
        self,
        *,
        shared_end_index: int | None = None,
        shared_end_monitor: int | None = None,
        shared_end_offset: tuple[int, int] | None = None,
    ) -> None:
        """Detach pending typed text and enqueue OCR/screenshot work off-hook."""
        with self._lock:
            chars = self._pending_text_chars
            meta = self._pending_text_meta
            self._pending_text_chars = []
            self._pending_text_caret = 0
            self._pending_text_meta = None
            self._cancel_pre_key_settle_timer_locked()
        if not chars or meta is None:
            return
        self._enqueue(
            _DeferredCaptureJob(
                action="flush_text_input",
                flush_chars=chars,
                flush_meta=meta,
                shared_end_index=shared_end_index,
                shared_end_monitor=shared_end_monitor,
                shared_end_offset=shared_end_offset,
            )
        )

    def _begin_pending_text_input(
        self,
        cursor_xy: tuple[int, int] | None,
        *,
        timestamp_utc: str | None = None,
    ) -> None:
        """Reserve the text-input event; defer UIA + before-shot off the hook thread.

        Prefer a pre-captured empty-field frame (taken after the focusing click /
        key), falling back to a live grab only when none is available.
        """
        mouse_xy = cursor_xy
        try:
            pos = pyautogui.position()
            mouse_xy = (int(pos.x), int(pos.y))
        except Exception:
            pass
        with self._lock:
            if self._run_dir is None:
                return
            last_click_xy = self._last_pointer_cursor_xy
            index = self._next_index
            self._next_index += 1
            self._pending_text_chars = []
            self._pending_text_caret = 0
            pending_pre_type = self._pending_pre_type_screenshot
            self._pending_pre_type_screenshot = None
            provisional_xy = mouse_xy or last_click_xy or cursor_xy
            meta: dict[str, Any] = {
                "index": index,
                "cursor_xy": provisional_xy,
                "anchor_click_xy": None,
                "focus_rect": None,
                "timestamp_utc": timestamp_utc or utc_now_iso(),
                "screenshot_path": "",
                "monitor_index": None,
                "monitor_offset": None,
                "capture_ready": threading.Event(),
            }
            self._pending_text_meta = meta
        self._enqueue(
            _DeferredCaptureJob(
                action="begin_text_input",
                meta=meta,
                pending_pre_type=pending_pre_type,
                last_click_xy=last_click_xy,
                mouse_xy=mouse_xy,
            )
        )

    def _append_text_input_char(
        self,
        char: str,
        cursor_xy: tuple[int, int] | None,
        *,
        timestamp_utc: str | None = None,
    ) -> None:
        with self._lock:
            if self._run_dir is None:
                return
            starting_burst = self._pending_text_meta is None

        if starting_burst:
            self._begin_pending_text_input(cursor_xy, timestamp_utc=timestamp_utc)

        with self._lock:
            caret = max(0, min(self._pending_text_caret, len(self._pending_text_chars)))
            self._pending_text_chars[caret:caret] = [char]
            self._pending_text_caret = caret + 1
        self._schedule_pre_key_settle()

    def _paste_slot_token(self, paste_id: int) -> str:
        return f"\ue000paste:{paste_id}\ue001"

    def _defer_clipboard_paste(
        self,
        cursor_xy: tuple[int, int] | None,
        *,
        timestamp_utc: str,
        hotkey_keys: list[str],
    ) -> None:
        """Enqueue a clipboard read. The hook must not call pyperclip.

        A slot is reserved at the current caret so keys typed while the read
        blocks stay after the pasted text. The worker fills that same list
        even if a later flush already detached it.
        """
        with self._lock:
            if self._run_dir is None:
                return
            need_begin = self._pending_text_meta is None
        if need_begin:
            self._begin_pending_text_input(cursor_xy, timestamp_utc=timestamp_utc)
        with self._lock:
            if self._pending_text_meta is None:
                return
            self._paste_seq += 1
            slot = self._paste_slot_token(self._paste_seq)
            chars = self._pending_text_chars
            caret = max(0, min(self._pending_text_caret, len(chars)))
            chars.insert(caret, slot)
            self._pending_text_caret = caret + 1
            raw_index = self._pending_text_meta.get("index")
            text_index = raw_index if isinstance(raw_index, int) else None
        self._enqueue(
            _DeferredCaptureJob(
                action="read_clipboard_paste",
                cursor_xy=cursor_xy,
                timestamp_utc=timestamp_utc,
                keys=hotkey_keys,
                meta={"paste_slot": slot, "chars": chars, "text_index": text_index},
            )
        )

    def _fill_paste_slot_locked(self, chars: list[str], slot: str, text: str | None) -> None:
        try:
            idx = chars.index(slot)
        except ValueError:
            return
        replacement = list(text) if text else []
        chars[idx : idx + 1] = replacement
        if chars is self._pending_text_chars and self._pending_text_caret > idx:
            self._pending_text_caret += len(replacement) - 1

    def _worker_read_clipboard_paste(self, job: _DeferredCaptureJob) -> None:
        pasted = _safe_clipboard_text()
        text = pasted if pasted is not None and pasted.strip() else None
        meta = job.meta or {}
        slot = meta.get("paste_slot")
        chars = meta.get("chars")
        if isinstance(slot, str) and isinstance(chars, list):
            with self._lock:
                self._fill_paste_slot_locked(chars, slot, text)
            if text:
                self._schedule_pre_key_settle()
                return
            self._persist_failed_paste_hotkey(job, chars)
            return
        if text:
            self._append_text_input_text(
                text,
                job.cursor_xy,
                timestamp_utc=job.timestamp_utc,
            )
            return
        self._persist_failed_paste_hotkey(job, None)

    def _persist_failed_paste_hotkey(
        self,
        job: _DeferredCaptureJob,
        chars: list[str] | None,
    ) -> None:
        """Record Ctrl+V as a hotkey on this worker, before stop's sentinel.

        An empty or failed clipboard read must not enqueue the hotkey behind
        the stop sentinel, or the key press is dropped.
        """
        with self._lock:
            if self._run_dir is None:
                return
            detached = chars is not None and chars is not self._pending_text_chars
            flush_chars: list[str] = []
            flush_meta = None
            orphan_index: int | None = None
            if not detached:
                flush_chars = self._pending_text_chars
                flush_meta = self._pending_text_meta
                self._pending_text_chars = []
                self._pending_text_caret = 0
                self._pending_text_meta = None
            chars_empty = chars is None or not chars
            if chars_empty:
                raw_index = (job.meta or {}).get("text_index")
                if isinstance(raw_index, int):
                    orphan_index = raw_index
                if not detached:
                    flush_meta = None
                    flush_chars = []
            if orphan_index is None:
                orphan_index = self._next_index
                self._next_index += 1
            self._cancel_pre_key_settle_timer_locked()
            pending_pre_key = self._pending_pre_key_screenshot
            self._pending_pre_key_screenshot = None
        if flush_chars and flush_meta is not None:
            self._worker_flush_text_input(
                _DeferredCaptureJob(
                    action="flush_text_input",
                    flush_chars=flush_chars,
                    flush_meta=flush_meta,
                )
            )
        self._worker_keyboard_event(
            _DeferredCaptureJob(
                action="keyboard_event",
                kind="hotkey",
                cursor_xy=job.cursor_xy,
                event_index=orphan_index,
                timestamp_utc=job.timestamp_utc,
                keys=job.keys,
                pending_pre_key=pending_pre_key,
                refresh_pre_type=True,
            )
        )

    def _append_text_input_text(
        self,
        text: str,
        cursor_xy: tuple[int, int] | None,
        *,
        timestamp_utc: str | None = None,
    ) -> None:
        if not text:
            return
        with self._lock:
            if self._run_dir is None:
                return
            starting_burst = self._pending_text_meta is None

        if starting_burst:
            self._begin_pending_text_input(cursor_xy, timestamp_utc=timestamp_utc)

        with self._lock:
            caret = max(0, min(self._pending_text_caret, len(self._pending_text_chars)))
            self._pending_text_chars[caret:caret] = list(text)
            self._pending_text_caret = caret + len(text)
        self._schedule_pre_key_settle()

    def _edit_pending_text_input(self, key: keyboard.Key | keyboard.KeyCode) -> bool:
        """Apply in-burst Backspace/Delete/Left/Right/Home/End. Returns True if handled."""
        if key not in (
            keyboard.Key.backspace,
            keyboard.Key.delete,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.home,
            keyboard.Key.end,
        ):
            return False
        with self._lock:
            if self._pending_text_meta is None:
                return False
            caret = max(0, min(self._pending_text_caret, len(self._pending_text_chars)))
            if key == keyboard.Key.backspace:
                if caret > 0:
                    del self._pending_text_chars[caret - 1]
                    self._pending_text_caret = caret - 1
                else:
                    self._pending_text_caret = caret
            elif key == keyboard.Key.delete:
                if caret < len(self._pending_text_chars):
                    del self._pending_text_chars[caret]
                self._pending_text_caret = caret
            elif key == keyboard.Key.left:
                self._pending_text_caret = max(0, caret - 1)
            elif key == keyboard.Key.right:
                self._pending_text_caret = min(len(self._pending_text_chars), caret + 1)
            elif key == keyboard.Key.home:
                self._pending_text_caret = 0
            else:  # end
                self._pending_text_caret = len(self._pending_text_chars)
            edited = True
        if edited:
            self._schedule_pre_key_settle()
        return edited

    def _cancel_pending_click_timer(self) -> None:
        with self._lock:
            pending = self._pending_click_timer
            self._pending_click_timer = None
        if pending is not None:
            pending.cancel()

    def _flush_superseded_pending_left_gesture(self) -> None:
        """Emit a pending deferred left click/hold/drag before a new press replaces it.

        Left clicks stay pending for ``_DOUBLE_CLICK_INTERVAL_S`` after mouse-up.
        A second press at a different location used to cancel that timer and
        overwrite ``_pending_click_coords`` without emitting — dropping the first
        click (e.g.「篩選」then a quick flyout pick). Flush first, same as stop.
        """
        self._cancel_pending_click_timer()
        with self._lock:
            pending_coords = self._pending_click_coords
            pending_down_at = self._pending_click_down_at
            left_press_dragging = self._left_press_dragging
            last_move_xy = self._last_move_xy
        if pending_coords is None:
            return
        sx, sy, button = pending_coords
        if left_press_dragging and last_move_xy is not None:
            ex, ey = last_move_xy
            if abs(sx - ex) > _DRAG_THRESHOLD_PX or abs(sy - ey) > _DRAG_THRESHOLD_PX:
                self._flush_pending_drag(sx, sy, ex, ey, button)
                return
        hold_duration = (
            time.monotonic() - pending_down_at if pending_down_at is not None else 0.0
        )
        if hold_duration >= _HOLD_THRESHOLD_S:
            self._flush_pending_hold(sx, sy, button, hold_duration)
            return
        self._flush_pending_click(sx, sy, button)

    def _discard_pending_drag_end_captures(self) -> None:
        with self._lock:
            captures = self._pending_drag_end_captures
            self._pending_drag_end_captures = None
        _discard_pending_drag_end_capture_files(captures)

    def _capture_pending_drag_end_screens(self) -> None:
        with self._lock:
            if not self._left_press_dragging or self._run_dir is None:
                return
            if self._pending_drag_end_captures is not None:
                return
            if self._left_press_seq in self._drag_end_captures:
                return
            press_seq = self._left_press_seq
        self._enqueue(
            _DeferredCaptureJob(
                action="pending_drag_end",
                press_seq=press_seq,
            )
        )

    def _clear_pending_left_gesture(self) -> None:
        self._discard_pending_drag_end_captures()
        with self._lock:
            seq = self._left_press_seq
            stale = self._left_press_captures.pop(seq, None)
            drag_stale = self._drag_end_captures.pop(seq, None)
            self._left_press_seq += 1
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
            self._pending_screenshot = None
            self._pending_windows_before = None
            self._pending_click_window = None
            self._left_press_click_windows.pop(seq, None)
        self._discard_press_capture_entry(stale)
        self._discard_drag_end_capture_entry(drag_stale)

    def _clear_pending_right_gesture_locked(
        self,
        *,
        discard_capture: bool = True,
    ) -> tuple[
        tuple[tuple[str, int, tuple[int, int]] | None, tuple[WindowInfo, ...] | None] | None,
        tuple[str, int, tuple[int, int]] | None,
    ]:
        seq = self._right_press_seq
        stale = self._right_press_captures.pop(seq, None) if discard_capture else None
        if discard_capture:
            self._right_press_seq += 1
        self._pending_right_coords = None
        self._pending_right_down_at = None
        self._pending_right_timestamp_utc = None
        self._pending_right_modifiers = None
        pending_shot = self._pending_right_screenshot
        self._pending_right_screenshot = None
        self._pending_right_windows_before = None
        self._pending_right_click_window = None
        self._right_press_click_windows.pop(seq, None)
        return stale, pending_shot

    def _clear_pending_right_gesture(self) -> None:
        with self._lock:
            stale, pending_shot = self._clear_pending_right_gesture_locked(
                discard_capture=True,
            )
        self._discard_press_capture_entry(stale)
        if pending_shot is not None:
            src = Path(pending_shot[0])
            if src.is_file():
                try:
                    src.unlink()
                except OSError:
                    pass

    def _snapshot_pressed_modifiers(self) -> list[str] | None:
        with self._lock:
            return _ordered_modifiers(self._pressed_modifiers)

    def _schedule_deferred_click(self, x: int, y: int, button: str, down_at: float) -> None:
        delay = max(0.0, _DOUBLE_CLICK_INTERVAL_S - (time.monotonic() - down_at))
        self._cancel_pending_click_timer()
        timer = threading.Timer(
            delay,
            self._emit_pending_click,
            args=(x, y, button),
        )
        timer.daemon = True
        with self._lock:
            self._pending_click_timer = timer
        timer.start()

    def _flush_pending_click(self, x: int, y: int, button: str) -> None:
        with self._lock:
            run_dir = self._run_dir
            if run_dir is None:
                return
            if self._pending_click_coords != (x, y, button):
                return
            click_index = self._next_index
            self._next_index += 1
            press_seq = self._left_press_seq
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
        self._flush_pending_text_input(shared_end_index=click_index)
        self._discard_pending_drag_end_captures()
        self._remember_pointer_cursor((x, y))
        self._enqueue(
            _DeferredCaptureJob(
                action="emit_left_gesture",
                kind="click",
                cursor_xy=(x, y),
                event_index=click_index,
                timestamp_utc=timestamp_utc,
                button=button,
                modifiers=modifiers,
                press_seq=press_seq,
                refresh_pre_type=True,
            )
        )

    def _flush_pending_hold(
        self,
        x: int,
        y: int,
        button: str,
        duration_seconds: float,
    ) -> None:
        with self._lock:
            run_dir = self._run_dir
            if run_dir is None:
                return
            if self._pending_click_coords != (x, y, button):
                return
            hold_index = self._next_index
            self._next_index += 1
            press_seq = self._left_press_seq
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
        self._flush_pending_text_input(shared_end_index=hold_index)
        self._discard_pending_drag_end_captures()
        self._remember_pointer_cursor((x, y))
        self._enqueue(
            _DeferredCaptureJob(
                action="emit_left_gesture",
                kind="hold",
                cursor_xy=(x, y),
                event_index=hold_index,
                timestamp_utc=timestamp_utc,
                button=button,
                modifiers=modifiers,
                duration_seconds=round(float(duration_seconds), 3),
                press_seq=press_seq,
                refresh_pre_type=True,
            )
        )

    def _flush_pending_right(
        self,
        x: int,
        y: int,
        *,
        kind: str,
        duration_seconds: float | None = None,
    ) -> None:
        with self._lock:
            run_dir = self._run_dir
            if run_dir is None:
                return
            if self._pending_right_coords != (x, y):
                return
            right_index = self._next_index
            self._next_index += 1
            press_seq = self._right_press_seq
            timestamp_utc = self._pending_right_timestamp_utc or utc_now_iso()
            modifiers = self._pending_right_modifiers
            # Keep press_seq + capture dict entry for the emit worker.
            self._clear_pending_right_gesture_locked(discard_capture=False)
        self._flush_pending_text_input(shared_end_index=right_index)
        self._remember_pointer_cursor((x, y))
        self._enqueue(
            _DeferredCaptureJob(
                action="emit_right_gesture",
                kind=kind,
                cursor_xy=(x, y),
                event_index=right_index,
                timestamp_utc=timestamp_utc,
                button="right",
                modifiers=modifiers,
                duration_seconds=(
                    round(float(duration_seconds), 3) if duration_seconds is not None else None
                ),
                press_seq=press_seq,
                refresh_pre_type=True,
            )
        )

    def _flush_pending_drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        button: str,
    ) -> None:
        with self._lock:
            run_dir = self._run_dir
            if run_dir is None:
                return
            if self._pending_click_coords != (x1, y1, button):
                return
            drag_index = self._next_index
            self._next_index += 1
            press_seq = self._left_press_seq
            pending_drag_end = self._pending_drag_end_captures
            self._pending_drag_end_captures = None
            if pending_drag_end is not None:
                self._drag_end_captures[press_seq] = pending_drag_end
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
        self._flush_pending_text_input(shared_end_index=drag_index)
        self._remember_pointer_cursor((x2, y2))
        self._enqueue(
            _DeferredCaptureJob(
                action="emit_left_gesture",
                kind="drag",
                cursor_xy=(x1, y1),
                end_xy=(x2, y2),
                event_index=drag_index,
                timestamp_utc=timestamp_utc,
                button=button,
                modifiers=modifiers,
                press_seq=press_seq,
                refresh_pre_type=True,
            )
        )

    def _worker_loop(self) -> None:
        while True:
            item = self._event_queue.get()
            if item is _QUEUE_SENTINEL:
                return
            if isinstance(item, tuple) and len(item) == 2 and item[0] is _DRAIN_MARKER:
                done = item[1]
                if isinstance(done, threading.Event):
                    done.set()
                continue
            if isinstance(item, _DeferredCaptureJob):
                self._process_deferred_capture_job(item)
            elif isinstance(item, _QueuedEvent):
                self._persist_queued_event(item)

    def wait_for_deferred_work(self, timeout: float = 2.0) -> None:
        """Block until previously enqueued capture jobs/events are processed.

        Used by tests; production stop path drains via the worker sentinel.
        """
        done = threading.Event()
        self._event_queue.put((_DRAIN_MARKER, done))
        if not done.wait(timeout):
            raise TimeoutError(f"deferred capture work did not finish within {timeout:.1f}s")

    def _process_deferred_capture_job(self, job: _DeferredCaptureJob) -> None:
        if job.action == "begin_text_input":
            self._worker_begin_text_input(job)
        elif job.action == "flush_text_input":
            self._worker_flush_text_input(job)
        elif job.action == "keyboard_event":
            self._worker_keyboard_event(job)
        elif job.action == "settle_pre_key":
            self._refresh_pending_pre_key()
        elif job.action == "pending_left_press":
            self._worker_pending_left_press(job)
        elif job.action == "pending_right_press":
            self._worker_pending_right_press(job)
        elif job.action == "pending_drag_end":
            self._worker_pending_drag_end(job)
        elif job.action == "emit_left_gesture":
            self._worker_emit_left_gesture(job)
        elif job.action == "emit_right_gesture":
            self._worker_emit_right_gesture(job)
        elif job.action == "mouse_pointer_event":
            self._worker_mouse_pointer_event(job)
        elif job.action == "seed_settle_windows":
            try:
                self._refresh_last_settle_windows()
            finally:
                done = (job.meta or {}).get("done")
                if isinstance(done, threading.Event):
                    done.set()
        elif job.action == "read_clipboard_paste":
            self._worker_read_clipboard_paste(job)
        # settle_probe_tick runs on its own timer thread (not this queue) so
        # window-settle sleeps on click persist cannot starve comparisons.

    def _cancel_settle_probe(
        self,
        *,
        cleanup_files: bool = True,
        reason: str | None = None,
    ) -> None:
        """Stop the settle probe timer and clear probe state."""
        # Wait out any in-flight timer-thread tick before clearing probe state.
        with self._settle_tick_lock:
            with self._lock:
                timer = self._settle_probe_timer
                self._settle_probe_timer = None
                probe = self._settle_probe
                self._settle_probe = None
                run_dir = self._run_dir
            if timer is not None:
                timer.cancel()
            if probe is not None and run_dir is not None and reason:
                age_s = time.monotonic() - probe.started_monotonic
                kept_age = (
                    f"{probe.kept_age_s:.3f}" if probe.kept_age_s is not None else "none"
                )
                self._log(
                    run_dir,
                    "settle probe cancelled "
                    f"event={probe.event_index} reason={reason} "
                    f"age_s={age_s:.3f} kept_age_s={kept_age}",
                )
            if not cleanup_files or probe is None or run_dir is None:
                return
            for path in (
                settle_probe_staging_path(run_dir, probe.event_index),
                settle_probe_kept_path(run_dir, probe.event_index),
                Path(probe.kept_path) if probe.kept_path else None,
            ):
                if path is None:
                    continue
                try:
                    # Never delete the durable next-before candidate.
                    if path.resolve() == last_settle_frame_path(run_dir).resolve():
                        continue
                except OSError:
                    pass
                try:
                    if path.is_file():
                        path.unlink()
                except OSError:
                    pass

    def _schedule_settle_probe_tick(self, delay_s: float, event_index: int) -> None:
        with self._lock:
            if self._settle_probe is None or self._settle_probe.event_index != event_index:
                return
            prev = self._settle_probe_timer
            self._settle_probe_timer = None

        if prev is not None:
            prev.cancel()

        def _fire() -> None:
            try:
                self._run_settle_probe_tick(event_index)
            except Exception as exc:
                with self._lock:
                    run_dir = self._run_dir
                if run_dir is not None:
                    self._log(
                        run_dir,
                        f"settle probe tick failed event={event_index}: {exc}",
                    )

        timer = threading.Timer(max(0.0, float(delay_s)), _fire)
        timer.daemon = True
        with self._lock:
            if self._settle_probe is None or self._settle_probe.event_index != event_index:
                return
            self._settle_probe_timer = timer
        timer.start()

    def _consume_settle_probe_delay_s(self, probe: _SettleProbeState) -> float:
        """Advance Fibonacci backoff on ``probe`` and return the next delay."""
        delay = settle_probe_interval_s(probe.interval_step)
        probe.interval_step += 1
        return delay

    def _schedule_next_settle_probe_tick(self, event_index: int) -> None:
        """Schedule the next settle sample using the probe's Fibonacci step."""
        with self._lock:
            probe = self._settle_probe
            if probe is None or probe.event_index != event_index:
                return
            delay = self._consume_settle_probe_delay_s(probe)
        self._schedule_settle_probe_tick(delay, event_index)

    def _event_already_has_observed_settle(self, event_index: int) -> bool:
        with self._lock:
            run_dir = self._run_dir
            events = list(self._events)
        if run_dir is None:
            return False
        for event in events:
            if event.index != event_index:
                continue
            observed = event.observed_settle_seconds
            return (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and float(observed) > 0
            )
        path = event_json_path(run_dir, event_index)
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            return False
        observed = raw.get("observed_settle_seconds")
        return (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and float(observed) > 0
        )

    def _finish_settle_against_final_after(self, final_after: Path) -> None:
        """Match the last action's settle probe to the all-monitor final-after frame.

        Consecutive-pair probing stops here. New samples use the full virtual
        desktop (same geometry as ``final_after``) until similar or the max window.
        """
        final_path = Path(final_after)
        if not final_path.is_file():
            self._cancel_settle_probe(
                cleanup_files=True,
                reason="recording_stop_final_after_missing",
            )
            return

        # Stop the timer but keep probe state for age / event index.
        with self._settle_tick_lock:
            with self._lock:
                timer = self._settle_probe_timer
                self._settle_probe_timer = None
                probe = self._settle_probe
                run_dir = self._run_dir
            if timer is not None:
                timer.cancel()

        if probe is None or run_dir is None:
            return

        if self._event_already_has_observed_settle(probe.event_index):
            self._cancel_settle_probe(
                cleanup_files=True,
                reason="recording_stop_already_observed",
            )
            return

        self._log(
            run_dir,
            f"settle probe match_final_after start event={probe.event_index} "
            f"path={final_path}",
        )

        staging = settle_probe_staging_path(run_dir, probe.event_index)
        while True:
            sample_age = time.monotonic() - probe.started_monotonic
            if sample_age > _SETTLE_PROBE_MAX_WINDOW_S:
                self._log(
                    run_dir,
                    f"settle probe match_final_after timeout event={probe.event_index} "
                    f"age_s={sample_age:.3f}",
                )
                self._cancel_settle_probe(
                    cleanup_files=True,
                    reason="recording_stop_final_after_timeout",
                )
                return

            try:
                # final_after is always all-monitors; samples must match that geometry.
                capture_all_screens_to_file(staging)
            except Exception as exc:
                self._log(
                    run_dir,
                    f"settle probe match_final_after capture failed "
                    f"event={probe.event_index}: {exc}",
                )
                time.sleep(self._consume_settle_probe_delay_s(probe))
                continue

            similar, mad = compare_frames(
                staging,
                final_path,
                threshold=DEFAULT_SIMILARITY_THRESHOLD,
            )
            mad_text = f"{mad:.6f}" if mad is not None else "error"
            self._archive_settle_debug_sample(
                run_dir=run_dir,
                event_index=probe.event_index,
                source=staging,
                age_s=sample_age,
                tag="final",
                record={
                    "kind": "match_final_after",
                    "mad": mad,
                    "threshold": DEFAULT_SIMILARITY_THRESHOLD,
                    "similar": similar,
                    "final_after": final_path.name,
                },
            )
            self._log(
                run_dir,
                f"settle probe match_final_after compare event={probe.event_index} "
                f"age_s={sample_age:.3f} mad={mad_text} "
                f"threshold={DEFAULT_SIMILARITY_THRESHOLD} similar={similar}",
            )
            if similar:
                self._publish_last_settle_frame(staging, 0, (0, 0))
                self._persist_observed_settle(probe.event_index, sample_age)
                with self._lock:
                    self._settle_probe = None
                    self._settle_probe_timer = None
                for path in (
                    staging,
                    settle_probe_kept_path(run_dir, probe.event_index),
                ):
                    try:
                        if (
                            path.is_file()
                            and path.resolve()
                            != last_settle_frame_path(run_dir).resolve()
                        ):
                            path.unlink()
                    except OSError:
                        pass
                self._log(
                    run_dir,
                    f"settle probe match_final_after stable event={probe.event_index} "
                    f"observed_settle_seconds={round(float(sample_age), 3)}",
                )
                return

            time.sleep(self._consume_settle_probe_delay_s(probe))

    def _start_settle_probe(self, event: RecordedEvent) -> None:
        if event.kind not in _SETTLE_PROBE_KINDS:
            return
        self._cancel_settle_probe(
            cleanup_files=True,
            reason=f"replaced_by_event_{event.index}",
        )
        cursor = event.cursor_xy or event.end_xy or event.anchor_click_xy
        with self._lock:
            if self._run_dir is None:
                return
            self._settle_probe = _SettleProbeState(
                event_index=event.index,
                started_monotonic=time.monotonic(),
                cursor_xy=cursor,
                interval_step=0,
            )
            run_dir = self._run_dir
            first_delay = self._consume_settle_probe_delay_s(self._settle_probe)
        self._log(
            run_dir,
            f"settle probe start event={event.index} "
            f"first_s={first_delay} fib_backoff=True "
            f"max_window_s={_SETTLE_PROBE_MAX_WINDOW_S} "
            f"threshold={DEFAULT_SIMILARITY_THRESHOLD}",
        )
        self._schedule_settle_probe_tick(first_delay, event.index)

    def _promote_settle_staging(self, staging: Path, kept_dest: Path) -> Path:
        kept_dest.parent.mkdir(parents=True, exist_ok=True)
        if staging.resolve() == kept_dest.resolve():
            return kept_dest
        try:
            if kept_dest.is_file():
                kept_dest.unlink()
        except OSError:
            pass
        try:
            staging.replace(kept_dest)
        except OSError:
            try:
                import shutil

                shutil.copy2(staging, kept_dest)
                staging.unlink(missing_ok=True)
            except OSError:
                return staging
        return kept_dest

    def _persist_observed_settle(self, event_index: int, observed_settle_seconds: float) -> None:
        with self._lock:
            run_dir = self._run_dir
            events = self._events
        if run_dir is None:
            return
        path = event_json_path(run_dir, event_index)
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            return
        seconds = round(float(observed_settle_seconds), 3)
        raw["observed_settle_seconds"] = seconds
        write_json(path, raw)
        with self._lock:
            for index, event in enumerate(events):
                if event.index == event_index:
                    events[index] = RecordedEvent.from_dict(raw)
                    break
        self._log(
            run_dir,
            f"settle probe stable event={event_index} observed_settle_seconds={seconds}",
        )

    def _run_settle_probe_tick(self, event_index: int) -> None:
        """Capture/compare one settle sample. Safe to call from the timer thread."""
        if not isinstance(event_index, int):
            return
        hit_max_window = False
        with self._settle_tick_lock:
            with self._lock:
                probe = self._settle_probe
                run_dir = self._run_dir
                if (
                    probe is None
                    or run_dir is None
                    or probe.event_index != event_index
                ):
                    return
                started = probe.started_monotonic
                kept_path = Path(probe.kept_path) if probe.kept_path else None
                kept_age = probe.kept_age_s
                cursor = probe.cursor_xy

            sample_age = time.monotonic() - started
            if sample_age > _SETTLE_PROBE_MAX_WINDOW_S:
                hit_max_window = True
            else:
                self._settle_probe_tick_body(
                    event_index=event_index,
                    run_dir=run_dir,
                    probe=probe,
                    kept_path=kept_path,
                    kept_age=kept_age,
                    cursor=cursor,
                    sample_age=sample_age,
                )
        if hit_max_window:
            self._cancel_settle_probe(cleanup_files=True, reason="max_window")

    def _archive_settle_debug_sample(
        self,
        *,
        run_dir: Path,
        event_index: int,
        source: Path,
        age_s: float,
        tag: str,
        record: dict[str, Any],
    ) -> int | None:
        """Persist a settle sample + JSONL row when debug retention is enabled."""
        if not _SETTLE_PROBE_KEEP_DEBUG_SAMPLES:
            return None
        with self._lock:
            probe = self._settle_probe
            if probe is None or probe.event_index != event_index:
                return None
            probe.sample_seq += 1
            sample_index = probe.sample_seq
        archived = archive_settle_probe_sample(
            run_dir=run_dir,
            event_index=event_index,
            sample_index=sample_index,
            source=source,
            age_s=age_s,
            tag=tag,
        )
        payload = {
            "sample": sample_index,
            "age_s": round(float(age_s), 3),
            "tag": tag,
            "path": archived.name if archived is not None else None,
            **record,
        }
        append_settle_probe_debug_record(run_dir, event_index, payload)
        return sample_index

    def _settle_probe_tick_body(
        self,
        *,
        event_index: int,
        run_dir: Path,
        probe: _SettleProbeState,
        kept_path: Path | None,
        kept_age: float | None,
        cursor: tuple[int, int] | None,
        sample_age: float,
    ) -> None:
        """Inner settle tick; caller holds ``_settle_tick_lock``."""
        staging = settle_probe_staging_path(run_dir, event_index)
        mon_idx = 0
        mon_offset: tuple[int, int] = (0, 0)
        try:
            if cursor is not None:
                _, mon_idx, mon_offset = _capture_screenshot_at_point(
                    int(cursor[0]), int(cursor[1]), staging
                )
            else:
                capture_all_screens_to_file(staging)
        except Exception as exc:
            self._log(run_dir, f"settle probe capture failed event={event_index}: {exc}")
            self._schedule_next_settle_probe_tick(event_index)
            return

        similar: bool | None = None
        mad: float | None = None
        kept_sample_seq: int | None = None
        with self._lock:
            if self._settle_probe is not None and self._settle_probe.event_index == event_index:
                kept_sample_seq = self._settle_probe.kept_sample_seq

        if kept_path is None or not kept_path.is_file():
            sample_index = self._archive_settle_debug_sample(
                run_dir=run_dir,
                event_index=event_index,
                source=staging,
                age_s=sample_age,
                tag="sample",
                record={"kind": "first_keep"},
            )
            self._log(
                run_dir,
                f"settle probe sample event={event_index} age_s={sample_age:.3f} "
                "first_keep=True",
            )
            if sample_index is not None:
                with self._lock:
                    if (
                        self._settle_probe is not None
                        and self._settle_probe.event_index == event_index
                    ):
                        self._settle_probe.kept_sample_seq = sample_index
        else:
            similar, mad = compare_frames(
                kept_path,
                staging,
                threshold=DEFAULT_SIMILARITY_THRESHOLD,
            )
            mad_text = f"{mad:.6f}" if mad is not None else "error"
            kept_age_text = (
                f"{kept_age:.3f}" if kept_age is not None else "none"
            )
            sample_index = self._archive_settle_debug_sample(
                run_dir=run_dir,
                event_index=event_index,
                source=staging,
                age_s=sample_age,
                tag="sample",
                record={
                    "kind": "compare",
                    "kept_sample": kept_sample_seq,
                    "kept_age_s": (
                        round(float(kept_age), 3) if kept_age is not None else None
                    ),
                    "mad": mad,
                    "threshold": DEFAULT_SIMILARITY_THRESHOLD,
                    "similar": similar,
                },
            )
            self._log(
                run_dir,
                f"settle probe compare event={event_index} "
                f"age_s={sample_age:.3f} kept_age_s={kept_age_text} "
                f"mad={mad_text} threshold={DEFAULT_SIMILARITY_THRESHOLD} "
                f"similar={similar}",
            )

        new_kept, new_age, observed = apply_settle_sample(
            kept_path=kept_path,
            kept_age_s=kept_age,
            staging_path=staging,
            sample_age_s=sample_age,
            similar=similar,
        )

        if observed is not None:
            # Publish stable frame as next before-shot candidate, then stop probe.
            publish_src = Path(new_kept) if new_kept is not None else None
            pub_mon = mon_idx
            pub_off = mon_offset
            with self._lock:
                if probe.last_mon_index is not None and probe.last_mon_offset is not None:
                    pub_mon = probe.last_mon_index
                    pub_off = probe.last_mon_offset
            if publish_src is not None and publish_src.is_file():
                self._publish_last_settle_frame(publish_src, pub_mon, pub_off)
            self._persist_observed_settle(event_index, observed)
            with self._lock:
                timer = self._settle_probe_timer
                self._settle_probe_timer = None
                self._settle_probe = None
            if timer is not None:
                timer.cancel()
            # Drop probe-only temps; keep durable ``_last_settle_frame.jpeg``
            # and settle_debug archives.
            for path in (staging, settle_probe_kept_path(run_dir, event_index)):
                try:
                    if path.is_file() and path.resolve() != last_settle_frame_path(run_dir).resolve():
                        path.unlink()
                except OSError:
                    pass
            if (
                publish_src is not None
                and publish_src.is_file()
                and publish_src.resolve() != last_settle_frame_path(run_dir).resolve()
            ):
                try:
                    publish_src.unlink()
                except OSError:
                    pass
            return

        official = settle_probe_kept_path(run_dir, event_index)
        if new_kept is not None:
            official = self._promote_settle_staging(Path(new_kept), official)
        self._publish_last_settle_frame(official, mon_idx, mon_offset)
        with self._lock:
            if self._settle_probe is None or self._settle_probe.event_index != event_index:
                return
            self._settle_probe.kept_path = str(official)
            self._settle_probe.kept_age_s = new_age
            self._settle_probe.last_mon_index = int(mon_idx)
            self._settle_probe.last_mon_offset = (int(mon_offset[0]), int(mon_offset[1]))
            # Current sample is now the kept frame (first keep or replace on change).
            if sample_index is not None:
                self._settle_probe.kept_sample_seq = sample_index
        self._schedule_next_settle_probe_tick(event_index)

    def _discard_pending_pre_type_file(
        self,
        pending: _PendingPreTypeShot | None,
    ) -> None:
        if pending is None:
            return
        path = Path(pending.path)
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass

    def _click_window_payload_at(self, x: int, y: int) -> ClickWindowInfo | None:
        """Resolve press-time window under the cursor (screen coords)."""
        try:
            return resolve_click_window(int(x), int(y))
        except Exception:
            return None

    def _worker_pending_left_press(self, job: _DeferredCaptureJob) -> None:
        if job.press_seq is None or job.cursor_xy is None:
            return
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            return

        if job.text_event_index is not None and job.flush_meta is not None:
            focus_xy = self._typing_focus_xy_for_ocr(job.flush_meta)
            if focus_xy is not None:
                self._capture_typing_ocr_end_shot(
                    run_dir,
                    int(job.text_event_index),
                    focus_xy[0],
                    focus_xy[1],
                )

        pending_dest = _pending_capture_path(run_dir)
        try:
            info = self._pending_screenshot_from_settle_or_capture(
                run_dir,
                int(job.cursor_xy[0]),
                int(job.cursor_xy[1]),
                pending_dest,
            )
        except Exception:
            info = None
        with self._lock:
            prev = self._left_press_captures.get(job.press_seq)
            windows_before = prev[1] if prev is not None else None
            self._left_press_captures[job.press_seq] = (info, windows_before)
            if self._left_press_seq == job.press_seq:
                self._pending_screenshot = info
                if windows_before is not None:
                    self._pending_windows_before = windows_before

    def _worker_pending_right_press(self, job: _DeferredCaptureJob) -> None:
        if job.press_seq is None or job.cursor_xy is None:
            return
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            return
        pending_dest = _pending_right_capture_path(run_dir)
        try:
            info = self._pending_screenshot_from_settle_or_capture(
                run_dir,
                int(job.cursor_xy[0]),
                int(job.cursor_xy[1]),
                pending_dest,
            )
        except Exception:
            info = None
        with self._lock:
            prev = self._right_press_captures.get(job.press_seq)
            windows_before = prev[1] if prev is not None else None
            self._right_press_captures[job.press_seq] = (info, windows_before)
            if self._right_press_seq == job.press_seq:
                self._pending_right_screenshot = info
                if windows_before is not None:
                    self._pending_right_windows_before = windows_before

    def _worker_pending_drag_end(self, job: _DeferredCaptureJob) -> None:
        if job.press_seq is None:
            return
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            return
        try:
            captures = _capture_all_monitors_to_pending(run_dir)
        except Exception:
            return
        with self._lock:
            if job.press_seq not in self._left_press_captures and job.press_seq != self._left_press_seq:
                # Gesture already cleared without emit; drop files.
                _discard_pending_drag_end_capture_files(captures)
                return
            if job.press_seq in self._drag_end_captures:
                _discard_pending_drag_end_capture_files(captures)
                return
            self._drag_end_captures[job.press_seq] = captures
            if self._left_press_seq == job.press_seq:
                self._pending_drag_end_captures = captures

    def _worker_emit_left_gesture(self, job: _DeferredCaptureJob) -> None:
        if job.event_index is None or job.kind is None or job.cursor_xy is None:
            return
        with self._lock:
            run_dir = self._run_dir
            press_seq = job.press_seq
            entry = (
                self._left_press_captures.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            click_window = (
                self._left_press_click_windows.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            drag_end = (
                self._drag_end_captures.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            if entry is None and press_seq is not None and self._left_press_seq == press_seq:
                entry = (self._pending_screenshot, self._pending_windows_before)
                click_window = self._pending_click_window
                self._pending_screenshot = None
                self._pending_windows_before = None
                self._pending_click_window = None
            if drag_end is None and press_seq is not None and self._left_press_seq == press_seq:
                drag_end = self._pending_drag_end_captures
                self._pending_drag_end_captures = None
        if run_dir is None:
            self._discard_press_capture_entry(entry)
            self._discard_drag_end_capture_entry(drag_end)
            return

        pending_shot = entry[0] if entry is not None else None
        pending_windows = entry[1] if entry is not None else None
        shot_path, mon_idx, mon_offset = _finalize_screenshot(
            run_dir,
            job.event_index,
            job.cursor_xy,
            pending_shot,
        )
        end_shot_path = ""
        end_mon_idx: int | None = None
        end_mon_offset: tuple[int, int] | None = None
        if job.kind == "drag" and job.end_xy is not None:
            end_shot_path, end_mon_idx, end_mon_offset = _finalize_drag_end_screenshot(
                run_dir,
                job.event_index,
                job.end_xy,
                drag_end,
                fallback_mon_idx=mon_idx,
                fallback_mon_offset=mon_offset,
            )
        else:
            self._discard_drag_end_capture_entry(drag_end)

        self._persist_queued_event(
            _QueuedEvent(
                kind=job.kind,
                cursor_xy=job.cursor_xy,
                end_xy=job.end_xy,
                event_index=job.event_index,
                timestamp_utc=job.timestamp_utc or utc_now_iso(),
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                end_screenshot_path=end_shot_path,
                end_monitor_index=end_mon_idx,
                end_monitor_offset=end_mon_offset,
                button=job.button,
                modifiers=job.modifiers,
                duration_seconds=job.duration_seconds,
                windows_before=pending_windows,
                click_window=click_window,
            )
        )
        if job.refresh_pre_type:
            refresh_xy = job.end_xy if job.kind == "drag" and job.end_xy is not None else job.cursor_xy
            self._refresh_pending_pre_type(refresh_xy)

    def _worker_emit_right_gesture(self, job: _DeferredCaptureJob) -> None:
        if job.event_index is None or job.kind is None or job.cursor_xy is None:
            return
        with self._lock:
            run_dir = self._run_dir
            press_seq = job.press_seq
            entry = (
                self._right_press_captures.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            click_window = (
                self._right_press_click_windows.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            if entry is None and press_seq is not None and self._right_press_seq == press_seq:
                entry = (self._pending_right_screenshot, self._pending_right_windows_before)
                click_window = self._pending_right_click_window
                self._pending_right_screenshot = None
                self._pending_right_windows_before = None
                self._pending_right_click_window = None
        if run_dir is None:
            self._discard_press_capture_entry(entry)
            return
        pending_shot = entry[0] if entry is not None else None
        pending_windows = entry[1] if entry is not None else None
        shot_path, mon_idx, mon_offset = _finalize_screenshot(
            run_dir,
            job.event_index,
            job.cursor_xy,
            pending_shot,
        )
        self._persist_queued_event(
            _QueuedEvent(
                kind=job.kind,
                cursor_xy=job.cursor_xy,
                event_index=job.event_index,
                timestamp_utc=job.timestamp_utc or utc_now_iso(),
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                button=job.button or "right",
                modifiers=job.modifiers,
                duration_seconds=job.duration_seconds,
                windows_before=pending_windows,
                click_window=click_window,
            )
        )
        if job.refresh_pre_type:
            self._refresh_pending_pre_type(job.cursor_xy)

    def _worker_mouse_pointer_event(self, job: _DeferredCaptureJob) -> None:
        if job.event_index is None or job.kind is None or job.cursor_xy is None:
            return
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            return
        shot_path, mon_idx, mon_offset = self._capture_immediate_screenshot(
            run_dir,
            job.event_index,
            job.cursor_xy,
        )
        self._persist_queued_event(
            _QueuedEvent(
                kind=job.kind,
                cursor_xy=job.cursor_xy,
                event_index=job.event_index,
                timestamp_utc=job.timestamp_utc or utc_now_iso(),
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                button=job.button,
                modifiers=job.modifiers,
                scroll_delta=job.scroll_delta,
                windows_before=job.windows_before,
                click_window=job.click_window,
            )
        )
        if job.refresh_pre_type:
            self._refresh_pending_pre_type(job.cursor_xy)

    def _worker_begin_text_input(self, job: _DeferredCaptureJob) -> None:
        meta = job.meta
        if meta is None:
            self._discard_pending_pre_type_file(job.pending_pre_type)
            return
        ready = meta.get("capture_ready")
        try:
            if meta.get("capture_closed"):
                self._discard_pending_pre_type_file(job.pending_pre_type)
                return

            with self._lock:
                run_dir = self._run_dir
            if run_dir is None:
                self._discard_pending_pre_type_file(job.pending_pre_type)
                return

            typing_focus = resolve_typing_focus(
                last_click_xy=job.last_click_xy,
                mouse_xy=job.mouse_xy,
            )
            focus_xy = typing_focus.point
            index = int(meta["index"])
            if meta.get("capture_closed"):
                self._discard_pending_pre_type_file(job.pending_pre_type)
                return

            meta["cursor_xy"] = focus_xy
            meta["focus_rect"] = typing_focus.rect

            pending_pre_type = job.pending_pre_type
            if pending_pre_type is not None and not _pre_type_focus_still_valid(
                pending_pre_type.focus_xy,
                focus_xy,
            ):
                # Focus moved (e.g. Enter launched an app; login field auto-focused).
                # Reusing the old frame would show the previous UI as the before-shot.
                self._discard_pending_pre_type_file(pending_pre_type)
                pending_pre_type = None

            shot_path = ""
            mon_idx: int | None = None
            mon_offset: tuple[int, int] | None = None
            if focus_xy is not None:
                shot_path, mon_idx, mon_offset = _finalize_screenshot(
                    run_dir,
                    index,
                    focus_xy,
                    (
                        pending_pre_type.as_finalize_tuple()
                        if pending_pre_type is not None
                        else None
                    ),
                )
            else:
                self._discard_pending_pre_type_file(pending_pre_type)

            if meta.get("capture_closed"):
                # Flushed while we captured; drop a late before-shot file if unused.
                return

            meta["screenshot_path"] = shot_path
            meta["monitor_index"] = mon_idx
            meta["monitor_offset"] = mon_offset
        finally:
            if isinstance(ready, threading.Event):
                ready.set()

    def _worker_flush_text_input(self, job: _DeferredCaptureJob) -> None:
        chars = job.flush_chars
        meta = job.flush_meta
        if not chars or meta is None:
            return

        meta["capture_closed"] = True
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            return

        index = int(meta["index"])
        shot_xy = meta.get("cursor_xy")
        if not meta.get("screenshot_path") and shot_xy is not None:
            shot_path, mon_idx, mon_offset = _finalize_screenshot(
                run_dir,
                index,
                (int(shot_xy[0]), int(shot_xy[1])),
                None,
            )
            meta["screenshot_path"] = shot_path
            meta["monitor_index"] = mon_idx
            meta["monitor_offset"] = mon_offset

        end_shot_path = ""
        end_mon_idx: int | None = None
        end_mon_offset: tuple[int, int] | None = None
        # Prefer a pre-key settle frame (captured before Tab/Enter) over a live
        # grab that may already show the key's focus change.
        preferred_end = job.pending_pre_key
        if preferred_end is not None:
            dest = screenshot_path_for_event_end(run_dir, index)
            src = Path(preferred_end.path)
            if src.is_file():
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if src.resolve() != dest.resolve():
                        dest.write_bytes(src.read_bytes())
                    end_shot_path = str(dest)
                    end_mon_idx = preferred_end.monitor_index
                    end_mon_offset = preferred_end.monitor_offset
                except OSError:
                    end_shot_path = ""

        # OCR / after-frame for typing must be on the typing focus monitor, not the
        # next pointer event's monitor (which can be a different display).
        if not end_shot_path and shot_xy is not None:
            ocr_shot = self._capture_typing_ocr_end_shot(
                run_dir,
                index,
                int(shot_xy[0]),
                int(shot_xy[1]),
            )
            if ocr_shot is not None:
                end_shot_path, end_mon_idx, end_mon_offset = ocr_shot

        if not end_shot_path and job.shared_end_index is not None:
            end_shot_path = str(screenshot_path_for_event(run_dir, job.shared_end_index))
            end_mon_idx = job.shared_end_monitor
            end_mon_offset = job.shared_end_offset

        self._persist_queued_event(
            _QueuedEvent(
                kind="text_input",
                cursor_xy=meta.get("cursor_xy"),
                event_index=index,
                timestamp_utc=str(meta["timestamp_utc"]),
                screenshot_path=str(meta.get("screenshot_path") or ""),
                monitor_index=meta.get("monitor_index"),
                monitor_offset=meta.get("monitor_offset"),
                end_screenshot_path=end_shot_path,
                end_monitor_index=end_mon_idx,
                end_monitor_offset=end_mon_offset,
                text="".join(chars),
                anchor_click_xy=None,
                focus_rect=meta.get("focus_rect"),
            )
        )

    def _typing_focus_xy_for_ocr(self, text_meta: dict[str, Any]) -> tuple[int, int] | None:
        """Return focus coords for OCR, waiting for deferred begin-text work if needed."""
        ready = text_meta.get("capture_ready")
        if isinstance(ready, threading.Event) and not ready.is_set():
            ready.wait(timeout=2.0)
        focus_xy = text_meta.get("cursor_xy")
        if isinstance(focus_xy, (tuple, list)) and len(focus_xy) == 2:
            return int(focus_xy[0]), int(focus_xy[1])
        return None

    def _worker_keyboard_event(self, job: _DeferredCaptureJob) -> None:
        pending_pre_key = job.pending_pre_key
        if job.flush_chars and job.flush_meta is not None:
            self._worker_flush_text_input(
                _DeferredCaptureJob(
                    action="flush_text_input",
                    flush_chars=job.flush_chars,
                    flush_meta=job.flush_meta,
                    shared_end_index=job.shared_end_index,
                    shared_end_monitor=job.shared_end_monitor,
                    shared_end_offset=job.shared_end_offset,
                    pending_pre_key=pending_pre_key,
                )
            )

        with self._lock:
            run_dir = self._run_dir
        if run_dir is None or job.event_index is None or job.kind is None:
            if pending_pre_key is not None:
                self._discard_pending_pre_type_file(pending_pre_key)
            return

        shot_path = ""
        mon_idx: int | None = None
        mon_offset: tuple[int, int] | None = None
        if job.cursor_xy is not None:
            shot_path, mon_idx, mon_offset = _finalize_screenshot(
                run_dir,
                job.event_index,
                job.cursor_xy,
                (
                    pending_pre_key.as_finalize_tuple()
                    if pending_pre_key is not None
                    else None
                ),
            )
            pending_pre_key = None
        elif pending_pre_key is not None:
            self._discard_pending_pre_type_file(pending_pre_key)
            pending_pre_key = None

        self._persist_queued_event(
            _QueuedEvent(
                kind=job.kind,
                cursor_xy=job.cursor_xy,
                event_index=job.event_index,
                timestamp_utc=job.timestamp_utc or utc_now_iso(),
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                key=job.key,
                keys=job.keys,
                text=job.text,
            )
        )
        if job.refresh_pre_type:
            self._refresh_pending_pre_type(job.cursor_xy)

    def _resolve_window_change(
        self,
        item: _QueuedEvent,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
        if not item.windows_before:
            return None, None, None
        time.sleep(settle_delay_for_click(item.cursor_xy, item.windows_before))
        try:
            windows_after = snapshot_top_level_windows()
        except Exception:
            return None, None, None
        result = diff_snapshots_with_debug(
            list(item.windows_before),
            windows_after,
            click_xy=item.cursor_xy,
        )
        if result.change is None:
            return None, None, result.debug
        return result.change.to_dict(), result.change.title or None, result.debug

    def _persist_queued_event(self, item: _QueuedEvent) -> None:
        with self._lock:
            if self._run_dir is None:
                return
            run_dir = self._run_dir

        window_change, target_title, snapshot_debug = self._resolve_window_change(item)

        click_window_payload: dict[str, Any] | None = None
        if item.click_window is not None:
            offset = item.monitor_offset if item.monitor_offset is not None else (0, 0)
            click_window_payload = item.click_window.to_local_payload(offset)

        event = RecordedEvent(
            index=item.event_index,
            timestamp_utc=item.timestamp_utc,
            kind=item.kind,
            cursor_xy=item.cursor_xy,
            end_xy=item.end_xy,
            button=item.button,
            modifiers=item.modifiers,
            key=item.key,
            keys=item.keys,
            text=item.text,
            scroll_delta=item.scroll_delta,
            duration_seconds=item.duration_seconds,
            screenshot_path=item.screenshot_path,
            monitor_index=item.monitor_index,
            monitor_offset=item.monitor_offset,
            end_screenshot_path=item.end_screenshot_path,
            end_monitor_index=item.end_monitor_index,
            end_monitor_offset=item.end_monitor_offset,
            anchor_click_xy=item.anchor_click_xy,
            focus_rect=item.focus_rect,
            window_change=window_change,
            target_window_title=target_title,
            window_snapshot_debug=snapshot_debug,
            click_window=click_window_payload,
        )
        with self._lock:
            if self._run_dir is None:
                return
            write_json(event_json_path(run_dir, event.index), event.to_dict())
            self._events.append(event)
        self._notify_event(event)
        if event.kind in _SETTLE_PROBE_KINDS:
            self._start_settle_probe(event)
        else:
            self._cancel_settle_probe(
                cleanup_files=True,
                reason=f"non_probe_event_{event.kind}_{event.index}",
            )

    def _emit_pending_click(self, x: int, y: int, button: str) -> None:
        self._flush_pending_click(x, y, button)

    def _on_mouse_move(self, x: int, y: int) -> None:
        ix, iy = int(x), int(y)
        with self._lock:
            self._last_move_xy = (ix, iy)
            if not self._left_button_down or self._pending_click_coords is None:
                return
            sx, sy, _btn = self._pending_click_coords
            dragging = self._left_press_dragging
        if dragging:
            return
        if abs(sx - ix) > _DRAG_THRESHOLD_PX or abs(sy - iy) > _DRAG_THRESHOLD_PX:
            with self._lock:
                self._left_press_dragging = True
            self._cancel_pending_click_timer()
            self._capture_pending_drag_end_screens()

    def _on_left_mouse_down(
        self,
        ix: int,
        iy: int,
        btn: str,
        *,
        timestamp_utc: str | None = None,
    ) -> None:
        now = time.monotonic()
        action_timestamp_utc = timestamp_utc or utc_now_iso()
        with self._lock:
            pending_coords = self._pending_click_coords
            down_at = self._pending_click_down_at
            pending_timer = self._pending_click_timer
        if (
            pending_coords is not None
            and down_at is not None
            and now - down_at <= _DOUBLE_CLICK_INTERVAL_S
        ):
            px, py, pbtn = pending_coords
            if (
                pbtn == "left"
                and abs(px - ix) <= _DOUBLE_CLICK_MAX_DIST_PX
                and abs(py - iy) <= _DOUBLE_CLICK_MAX_DIST_PX
            ):
                if pending_timer is not None:
                    pending_timer.cancel()
                self._clear_pending_left_gesture()
                with self._lock:
                    self._pending_click_timer = None
                self._queue_pointer_event_immediate(
                    kind="double_click",
                    cursor_xy=(ix, iy),
                    button=btn,
                    modifiers=self._snapshot_pressed_modifiers(),
                    timestamp_utc=action_timestamp_utc,
                )
                return

        # Not a double-click: keep any deferred first click instead of dropping it.
        if pending_coords is not None:
            self._flush_superseded_pending_left_gesture()
            now = time.monotonic()
        else:
            self._cancel_pending_click_timer()
        with self._lock:
            run_dir = self._run_dir
            has_pending_text = bool(self._pending_text_chars)
            text_meta = self._pending_text_meta
            self._pending_click_coords = (ix, iy, btn)
            self._pending_click_down_at = now
            self._pending_click_timestamp_utc = action_timestamp_utc
            self._pending_click_modifiers = _ordered_modifiers(self._pressed_modifiers)
            self._left_button_down = True
            self._left_press_dragging = False
            self._last_move_xy = (ix, iy)
        if (
            run_dir is not None
            and has_pending_text
            and text_meta is not None
        ):
            self._capture_pending_left_press(
                run_dir,
                ix,
                iy,
                text_event_index=int(text_meta["index"]),
                text_meta=text_meta,
            )
        elif run_dir is not None:
            self._capture_pending_left_press(run_dir, ix, iy)

    def _on_left_mouse_up(self, ix: int, iy: int) -> None:
        with self._lock:
            pending_coords = self._pending_click_coords
            down_at = self._pending_click_down_at
            dragging = self._left_press_dragging
            self._left_button_down = False
            self._last_move_xy = (ix, iy)
        if pending_coords is None:
            return
        sx, sy, button = pending_coords
        if dragging:
            # Interim movement may arm drag, but a release near the press point
            # is a click (or hold), not a drag.
            if (
                abs(sx - ix) > _DRAG_THRESHOLD_PX
                or abs(sy - iy) > _DRAG_THRESHOLD_PX
            ):
                self._flush_pending_drag(sx, sy, ix, iy, button)
                return
            with self._lock:
                self._left_press_dragging = False
        hold_duration = time.monotonic() - down_at if down_at is not None else 0.0
        if hold_duration >= _HOLD_THRESHOLD_S:
            self._flush_pending_hold(sx, sy, button, hold_duration)
            return
        if down_at is None:
            self._flush_pending_click(sx, sy, button)
            return
        self._schedule_deferred_click(sx, sy, button, down_at)

    def _on_right_mouse_down(
        self,
        ix: int,
        iy: int,
        *,
        timestamp_utc: str | None = None,
    ) -> None:
        action_timestamp_utc = timestamp_utc or utc_now_iso()
        self._clear_pending_right_gesture()
        with self._lock:
            run_dir = self._run_dir
            has_pending_text = bool(self._pending_text_chars)
            text_meta = self._pending_text_meta
            self._pending_right_coords = (ix, iy)
            self._pending_right_down_at = time.monotonic()
            self._pending_right_timestamp_utc = action_timestamp_utc
            self._pending_right_modifiers = _ordered_modifiers(self._pressed_modifiers)
        if (
            run_dir is not None
            and has_pending_text
            and text_meta is not None
        ):
            # OCR end-shot for the prior text burst is handled when left-click
            # also flushes; right-click only starts a new gesture off-hook.
            pass
        if run_dir is not None:
            self._capture_pending_right_press(run_dir, ix, iy)

    def _on_right_mouse_up(self, ix: int, iy: int) -> None:
        with self._lock:
            pending_coords = self._pending_right_coords
            down_at = self._pending_right_down_at
        if pending_coords is None:
            return
        sx, sy = pending_coords
        hold_duration = time.monotonic() - down_at if down_at is not None else 0.0
        if hold_duration >= _HOLD_THRESHOLD_S:
            self._flush_pending_right(sx, sy, kind="hold", duration_seconds=hold_duration)
            return
        self._flush_pending_right(sx, sy, kind="right_click")

    def _on_mouse_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        timestamp_utc = utc_now_iso()
        if self._should_ignore_mouse_point(int(x), int(y)):
            return
        btn = _normalize_button(button)
        ix, iy = int(x), int(y)
        if btn == "left":
            if pressed:
                self._on_left_mouse_down(ix, iy, btn, timestamp_utc=timestamp_utc)
            else:
                self._on_left_mouse_up(ix, iy)
            return
        if btn == "right":
            if pressed:
                self._on_right_mouse_down(ix, iy, timestamp_utc=timestamp_utc)
            else:
                self._on_right_mouse_up(ix, iy)
            return
        if not pressed:
            return
        kind = "middle_click" if btn == "middle" else "click"
        self._queue_pointer_event_immediate(
            kind=kind,
            cursor_xy=(ix, iy),
            button=btn,
            modifiers=self._snapshot_pressed_modifiers(),
            timestamp_utc=timestamp_utc,
        )

    def _on_mouse_scroll(self, x: int, y: int, _dx: int, dy: int) -> None:
        timestamp_utc = utc_now_iso()
        if self._should_ignore_mouse_point(int(x), int(y)):
            return
        clicks = int(dy)
        if clicks == 0:
            clicks = -1 if dy < 0 else 1
        self._queue_pointer_event_immediate(
            kind="scroll",
            cursor_xy=(int(x), int(y)),
            scroll_delta=clicks,
            timestamp_utc=timestamp_utc,
        )

    def _on_key_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        timestamp_utc = utc_now_iso()
        with self._lock:
            suppress = self._suppress_hotkey_keys
            active = self._accepting_input
        if not active:
            return
        if suppress and key in _HOTKEY_SUPPRESS_KEYS:
            mod = _modifier_name(key)
            if mod:
                with self._lock:
                    self._pressed_modifiers.add(mod)
            return

        mod = _modifier_name(key)
        if mod:
            with self._lock:
                self._pressed_modifiers.add(mod)
            return

        try:
            pos = pyautogui.position()
            cursor_xy = (int(pos.x), int(pos.y))
        except Exception:
            cursor_xy = None

        with self._lock:
            mods = sorted(self._pressed_modifiers)

        token = _key_token(key)
        if token is None:
            return

        typed = _key_char(key)
        # Shift alone only changes case/symbols — treat printable keys as typing,
        # not hotkeys. Ctrl/Alt/Win (+ optional Shift) remain hotkeys.
        if mods == ["shift"]:
            if typed and typed.isprintable():
                self._append_text_input_char(typed, cursor_xy, timestamp_utc=timestamp_utc)
                return
            if key == keyboard.Key.space:
                self._append_text_input_char(" ", cursor_xy, timestamp_utc=timestamp_utc)
                return

        if mods:
            if is_recording_toggle_hotkey(mods + [token]):
                # Global recording toggle (Ctrl+Shift+R) — never record as a step.
                return
            if _is_paste_hotkey(mods, token):
                # Clipboard owners can block pyperclip.paste for a long time.
                # Read it on the worker, same as screenshots, so this hook returns.
                self._defer_clipboard_paste(
                    cursor_xy,
                    timestamp_utc=timestamp_utc,
                    hotkey_keys=mods + [token],
                )
                return
            self._queue_keyboard_event_immediate(
                kind="hotkey",
                cursor_xy=cursor_xy,
                keys=mods + [token],
                timestamp_utc=timestamp_utc,
            )
            return

        if typed and typed.isprintable():
            self._append_text_input_char(typed, cursor_xy, timestamp_utc=timestamp_utc)
            return

        if key == keyboard.Key.space:
            self._append_text_input_char(" ", cursor_xy, timestamp_utc=timestamp_utc)
            return

        if not mods and self._edit_pending_text_input(key):
            return

        if key in _SPECIAL_KEYS or isinstance(key, keyboard.Key):
            self._queue_keyboard_event_immediate(
                kind="key_press",
                cursor_xy=cursor_xy,
                key=token,
                timestamp_utc=timestamp_utc,
            )
            return

    def _on_key_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        mod = _modifier_name(key)
        if mod:
            with self._lock:
                self._pressed_modifiers.discard(mod)
