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
from src.eye.capture import resolve_monitor_index
from src.recorder.focus_point import resolve_typing_focus
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
    WindowInfo,
    diff_snapshots_with_debug,
    settle_delay_for_click,
    snapshot_top_level_windows,
)

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
_QUEUE_SENTINEL = object()
_DRAIN_MARKER = object()


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

    action: str  # begin_text_input | flush_text_input | keyboard_event | settle_pre_key
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
    event_index: int | None = None
    timestamp_utc: str | None = None
    key: str | None = None
    keys: list[str] | None = None
    text: str | None = None
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

IgnoreRectProvider = Callable[[], tuple[int, int, int, int] | None]


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

    Uses the last event's end/cursor point to pick a monitor. Returns the saved
    path, or ``None`` when capture fails.
    """
    dest = final_after_screenshot_path(run_dir)
    anchor: tuple[int, int] | None = None
    for event in reversed(events):
        if event.end_xy is not None:
            anchor = event.end_xy
            break
        if event.cursor_xy is not None:
            anchor = event.cursor_xy
            break
    try:
        if anchor is None:
            with mss.mss() as sct:
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                anchor = (
                    int(monitor["left"] + monitor["width"] // 2),
                    int(monitor["top"] + monitor["height"] // 2),
                )
        path, _, _ = _capture_screenshot_at_point(anchor[0], anchor[1], dest)
        return path
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
        self._pending_right_coords: tuple[int, int] | None = None
        self._pending_right_down_at: float | None = None
        self._pending_right_timestamp_utc: str | None = None
        self._pending_right_modifiers: list[str] | None = None
        self._pending_right_screenshot: tuple[str, int, tuple[int, int]] | None = None
        self._pending_right_windows_before: tuple[WindowInfo, ...] | None = None
        self._pending_text_chars: list[str] = []
        self._pending_text_caret: int = 0
        self._pending_text_meta: dict[str, Any] | None = None
        # Empty-field frame captured after focus settles (click/Tab/etc.), consumed
        # as the text_input before-shot so we do not race the first typed glyph.
        self._pending_pre_type_screenshot: _PendingPreTypeShot | None = None
        # Settled frame for the next key_press before-shot (cannot capture on LL hook).
        self._pending_pre_key_screenshot: _PendingPreTypeShot | None = None
        self._pending_pre_key_timer: threading.Timer | None = None
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
            self._accepting_input = True
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
            self._pending_right_coords = None
            self._pending_right_down_at = None
            self._pending_right_timestamp_utc = None
            self._pending_right_modifiers = None
            self._pending_right_screenshot = None
            self._pending_right_windows_before = None
            self._pending_text_chars = []
            self._pending_text_meta = None
            self._pending_pre_type_screenshot = None
            self._pending_pre_key_screenshot = None
            self._cancel_pre_key_settle_timer_locked()
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
            self._clear_pending_right_gesture_locked()
            self._pending_screenshot = None
            self._pending_windows_before = None
            leftover_pre_type = self._pending_pre_type_screenshot
            self._pending_pre_type_screenshot = None
            leftover_pre_key = self._pending_pre_key_screenshot
            self._pending_pre_key_screenshot = None
            self._cancel_pre_key_settle_timer_locked()
        _discard_pending_drag_end_capture_files(leftover_drag_end)
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

        self._event_queue.put(_QUEUE_SENTINEL)
        return run_dir

    def finalize_stop(self) -> Path | None:
        """Wait for queued events, capture final screenshot, and write session artifacts."""
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
            with self._lock:
                self._finalizing = False
            return None

        # Capture settled UI before the hub restores (caller deiconifies after finalize).
        final_after = capture_final_after_screenshot(run_dir, events)
        if final_after is not None:
            self._log(run_dir, f"final after screenshot saved path={final_after}")
        else:
            self._log(run_dir, "final after screenshot capture failed")

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

    def _current_ignore_rect(self) -> tuple[int, int, int, int] | None:
        provider = self._ignore_rect_provider
        if provider is None:
            return None
        try:
            return provider()
        except Exception:
            return None

    def _should_ignore_mouse_point(self, x: int, y: int) -> bool:
        with self._lock:
            if not self._accepting_input:
                return True
        return _point_in_rect(x, y, self._current_ignore_rect())

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

    def _snapshot_windows_before(self) -> tuple[WindowInfo, ...]:
        try:
            return tuple(snapshot_top_level_windows())
        except Exception:
            return ()

    def _capture_pending_left_press(self, run_dir: Path, x: int, y: int) -> None:
        """Capture the screen on mouse-down before the click is delivered to apps."""
        pending_dest = _pending_capture_path(run_dir)
        windows_before = self._snapshot_windows_before()
        try:
            info = _capture_screenshot_at_point(x, y, pending_dest)
        except Exception:
            info = None
        with self._lock:
            self._pending_screenshot = info
            self._pending_windows_before = windows_before

    def _capture_pending_right_press(self, run_dir: Path, x: int, y: int) -> None:
        """Capture the screen on right mouse-down before the click is delivered."""
        pending_dest = _pending_right_capture_path(run_dir)
        windows_before = self._snapshot_windows_before()
        try:
            info = _capture_screenshot_at_point(x, y, pending_dest)
        except Exception:
            info = None
        with self._lock:
            self._pending_right_screenshot = info
            self._pending_right_windows_before = windows_before

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
        action_timestamp_utc = timestamp_utc or utc_now_iso()
        with self._lock:
            run_dir = self._run_dir
            if run_dir is None:
                return
            index = self._next_index
            self._next_index += 1
        self._flush_pending_text_input(shared_end_index=index)
        windows_before = self._snapshot_windows_before()
        shot_path, mon_idx, mon_offset = self._capture_immediate_screenshot(
            run_dir,
            index,
            cursor_xy,
        )
        self._remember_pointer_cursor(cursor_xy)
        self._queue_event(
            _QueuedEvent(
                kind=kind,
                cursor_xy=cursor_xy,
                event_index=index,
                timestamp_utc=action_timestamp_utc,
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                button=button,
                modifiers=modifiers,
                scroll_delta=scroll_delta,
                windows_before=windows_before or None,
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
    ) -> None:
        """Reserve indices and enqueue screenshot work off the keyboard hook thread."""
        action_timestamp_utc = timestamp_utc or utc_now_iso()
        with self._lock:
            if self._run_dir is None:
                return
            index = self._next_index
            self._next_index += 1
            flush_chars = list(self._pending_text_chars)
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
            chars = list(self._pending_text_chars)
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
            run_dir = self._run_dir

        try:
            captures = _capture_all_monitors_to_pending(run_dir)
        except Exception:
            return

        with self._lock:
            if not self._left_press_dragging:
                _discard_pending_drag_end_capture_files(captures)
                return
            if self._pending_drag_end_captures is not None:
                _discard_pending_drag_end_capture_files(captures)
                return
            self._pending_drag_end_captures = captures

    def _clear_pending_left_gesture(self) -> None:
        self._discard_pending_drag_end_captures()
        with self._lock:
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
            self._pending_screenshot = None
            self._pending_windows_before = None

    def _clear_pending_right_gesture_locked(self) -> None:
        self._pending_right_coords = None
        self._pending_right_down_at = None
        self._pending_right_timestamp_utc = None
        self._pending_right_modifiers = None
        self._pending_right_screenshot = None
        self._pending_right_windows_before = None

    def _discard_pending_right_capture_file(self) -> None:
        with self._lock:
            pending = self._pending_right_screenshot
            self._pending_right_screenshot = None
        if pending is None:
            return
        src = Path(pending[0])
        if src.is_file():
            try:
                src.unlink()
            except OSError:
                pass

    def _clear_pending_right_gesture(self) -> None:
        self._discard_pending_right_capture_file()
        with self._lock:
            self._clear_pending_right_gesture_locked()

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
            pending_shot = self._pending_screenshot
            pending_windows = self._pending_windows_before
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            self._pending_screenshot = None
            self._pending_windows_before = None
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
        self._flush_pending_text_input(
            shared_end_index=click_index,
            shared_end_monitor=pending_shot[1] if pending_shot is not None else None,
            shared_end_offset=pending_shot[2] if pending_shot is not None else None,
        )
        self._discard_pending_drag_end_captures()
        shot_path, mon_idx, mon_offset = _finalize_screenshot(
            run_dir,
            click_index,
            (x, y),
            pending_shot,
        )
        self._remember_pointer_cursor((x, y))
        self._queue_event(
            _QueuedEvent(
                kind="click",
                cursor_xy=(x, y),
                event_index=click_index,
                timestamp_utc=timestamp_utc,
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                button=button,
                modifiers=modifiers,
                windows_before=pending_windows,
            )
        )
        self._refresh_pending_pre_type((x, y))

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
            pending_shot = self._pending_screenshot
            pending_windows = self._pending_windows_before
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            self._pending_screenshot = None
            self._pending_windows_before = None
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
        self._flush_pending_text_input(
            shared_end_index=hold_index,
            shared_end_monitor=pending_shot[1] if pending_shot is not None else None,
            shared_end_offset=pending_shot[2] if pending_shot is not None else None,
        )
        self._discard_pending_drag_end_captures()
        shot_path, mon_idx, mon_offset = _finalize_screenshot(
            run_dir,
            hold_index,
            (x, y),
            pending_shot,
        )
        self._remember_pointer_cursor((x, y))
        self._queue_event(
            _QueuedEvent(
                kind="hold",
                cursor_xy=(x, y),
                event_index=hold_index,
                timestamp_utc=timestamp_utc,
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                button=button,
                modifiers=modifiers,
                duration_seconds=round(float(duration_seconds), 3),
                windows_before=pending_windows,
            )
        )
        self._refresh_pending_pre_type((x, y))

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
            pending_shot = self._pending_right_screenshot
            pending_windows = self._pending_right_windows_before
            timestamp_utc = self._pending_right_timestamp_utc or utc_now_iso()
            modifiers = self._pending_right_modifiers
            self._clear_pending_right_gesture_locked()
        self._flush_pending_text_input(
            shared_end_index=right_index,
            shared_end_monitor=pending_shot[1] if pending_shot is not None else None,
            shared_end_offset=pending_shot[2] if pending_shot is not None else None,
        )
        shot_path, mon_idx, mon_offset = _finalize_screenshot(
            run_dir,
            right_index,
            (x, y),
            pending_shot,
        )
        self._remember_pointer_cursor((x, y))
        self._queue_event(
            _QueuedEvent(
                kind=kind,
                cursor_xy=(x, y),
                event_index=right_index,
                timestamp_utc=timestamp_utc,
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                button="right",
                modifiers=modifiers,
                duration_seconds=(
                    round(float(duration_seconds), 3) if duration_seconds is not None else None
                ),
                windows_before=pending_windows,
            )
        )
        self._refresh_pending_pre_type((x, y))

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
            pending_shot = self._pending_screenshot
            pending_windows = self._pending_windows_before
            pending_drag_end = self._pending_drag_end_captures
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            self._pending_screenshot = None
            self._pending_windows_before = None
            self._pending_drag_end_captures = None
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
        self._flush_pending_text_input(
            shared_end_index=drag_index,
            shared_end_monitor=pending_shot[1] if pending_shot is not None else None,
            shared_end_offset=pending_shot[2] if pending_shot is not None else None,
        )
        shot_path, mon_idx, mon_offset = _finalize_screenshot(
            run_dir,
            drag_index,
            (x1, y1),
            pending_shot,
        )
        end_shot_path, end_mon_idx, end_mon_offset = _finalize_drag_end_screenshot(
            run_dir,
            drag_index,
            (x2, y2),
            pending_drag_end,
            fallback_mon_idx=mon_idx,
            fallback_mon_offset=mon_offset,
        )
        self._remember_pointer_cursor((x2, y2))
        self._queue_event(
            _QueuedEvent(
                kind="drag",
                cursor_xy=(x1, y1),
                end_xy=(x2, y2),
                event_index=drag_index,
                timestamp_utc=timestamp_utc,
                screenshot_path=shot_path,
                monitor_index=mon_idx,
                monitor_offset=mon_offset,
                end_screenshot_path=end_shot_path,
                end_monitor_index=end_mon_idx,
                end_monitor_offset=end_mon_offset,
                button=button,
                modifiers=modifiers,
                windows_before=pending_windows,
            )
        )
        self._refresh_pending_pre_type((x2, y2))

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
        )
        with self._lock:
            if self._run_dir is None:
                return
            write_json(event_json_path(run_dir, event.index), event.to_dict())
            self._events.append(event)
        self._notify_event(event)

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
            focus_xy = self._typing_focus_xy_for_ocr(text_meta)
            if focus_xy is not None:
                self._capture_typing_ocr_end_shot(
                    run_dir,
                    int(text_meta["index"]),
                    focus_xy[0],
                    focus_xy[1],
                )
        if run_dir is not None:
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
            focus_xy = self._typing_focus_xy_for_ocr(text_meta)
            if focus_xy is not None:
                self._capture_typing_ocr_end_shot(
                    run_dir,
                    int(text_meta["index"]),
                    focus_xy[0],
                    focus_xy[1],
                )
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
            if _is_paste_hotkey(mods, token):
                pasted = _safe_clipboard_text()
                if pasted is not None and pasted.strip():
                    self._append_text_input_text(
                        pasted,
                        cursor_xy,
                        timestamp_utc=timestamp_utc,
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
