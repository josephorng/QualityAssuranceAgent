from __future__ import annotations

import copy
import ctypes
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mss
import pyautogui
from pynput import keyboard, mouse

from src.common.io_utils import append_text, read_clipboard_text, read_json, write_json
from cua_mcp.geometry import point_in_rect_xywh
from src.common.run_state import unique_run_folder_name
from src.common.settings import load_settings
from src.eye.capture import capture_all_screens_to_file, resolve_monitor_index
from src.recorder.focus_point import resolve_typing_focus, sample_debug_focus_rect
from src.recorder.frame_similarity import (
    DEFAULT_SIMILARITY_THRESHOLD,
    append_settle_probe_debug_record,
    apply_settle_sample,
    archive_settle_probe_sample,
    compare_frames,
    rgb_frames_similar,
    last_settle_frame_path,
    settle_probe_kept_path,
    settle_probe_staging_path,
)
from src.recorder.models import (
    RecordedEvent,
    SessionManifest,
    event_json_path,
    final_after_screenshot_path,
    hydrate_window_snapshot_debug,
    next_recording_event_index,
    recording_event_paths,
    screenshot_path_for_event,
    screenshot_path_for_event_end,
    utc_now_iso,
    write_event_json,
)
from src.recorder.verify_signals import (
    capture_fast_signals,
    capture_slow_signals,
    capture_step_signals,
    identity_from_click_window,
    read_uia_for_kind,
)
from src.recorder.window_snapshot import (
    ClickWindowInfo,
    WindowInfo,
    _windows_debug_list,
    diff_snapshots_with_debug,
    resolve_click_window,
    settle_delay_for_click,
    snapshot_top_level_windows,
)
from src.recorder.hotkey import is_recording_toggle_hotkey
from src.recorder.ime_state import ImeInputState, read_ime_input_state

_DOUBLE_CLICK_INTERVAL_S = 0.35
_DOUBLE_CLICK_MAX_DIST_PX = 8
_DRAG_THRESHOLD_PX = 8
# Distinct from None, which means mouse-up reserved the drag-end slot
# before the mouse-down monitor grab finished.
_DRAG_END_UNCLAIMED = object()
# Must exceed the double-click window so short presses still defer for double-click.
_HOLD_THRESHOLD_S = 0.5
# Pre-type frames are only reused when typing focus is still near the capture point.
# Enter→app launch→type into a new dialog moves focus far away; reuse would keep a
# stale Search/desktop frame as the text_input before-shot.
_PRE_TYPE_FOCUS_MAX_DIST_PX = 48
# After typing pauses, capture a settled frame for the next key_press before-shot.
# LL hooks cannot screenshot before Tab/Enter is delivered; reuse this frame instead.
_PRE_KEY_SETTLE_S = 0.3
# After Chinese IME Enter, wait for composition to commit before sealing
# ``signals_after`` (avoid mid-zhuyin settle cache and late unsealed live samples).
_IME_COMMIT_SETTLE_S = 0.3
# Consecutive-screenshot settle probe (UI stability after an action).
# Samples: the first shot is immediate, then every 1s. Each sample grabs the
# full virtual desktop so a window appearing on another monitor is still seen.
# The probe only measures observed_settle; it does not publish the next step's
# before-shot. A separate before-shot loop round-robins each physical monitor
# and keeps the latest raw RGB frame per monitor in memory. A click pins that
# raw frame; the worker JPEG-encodes it only when the frame is adopted.
_SETTLE_PROBE_INTERVAL_S = 1.0
_SETTLE_PROBE_MAX_WINDOW_S = 55.0
# Retain every settle-probe capture under screenshots/settle_debug/ for MAD tuning.
_SETTLE_PROBE_KEEP_DEBUG_SAMPLES = False
# Window list and UI Automation. Kept off the settle-probe timer.
_PRE_CLICK_CONTEXT_INTERVAL_S = 0.25
# UI Automation on a closing window can stall. The per-step thread gives up
# and records ``resolve_failed`` on ``window_snapshot_debug`` rather than
# blocking the next click.
_WINDOW_UIA_TIMEOUT_S = 1.0
# Stop waits this long for in-flight window-step threads to patch event JSON.
_WINDOW_STEP_JOIN_TIMEOUT_S = 2.0
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
    """Return the delay (seconds) before settle sample ``step``.

    Sample 0 is immediate. Every later sample waits ``_SETTLE_PROBE_INTERVAL_S``.
    """
    if int(step) <= 0:
        return 0.0
    return _SETTLE_PROBE_INTERVAL_S


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
    # Next sample index for ``settle_probe_interval_s`` when scheduling a tick.
    interval_step: int = 0
    # First similar plateau recorded to event JSON; probe keeps sampling after.
    observed_recorded: bool = False


@dataclass(frozen=True)
class _PendingPreTypeShot:
    """Settled UI frame captured before the next text_input or key_press."""

    path: str
    monitor_index: int
    monitor_offset: tuple[int, int]
    focus_xy: tuple[int, int]

    def as_finalize_tuple(self) -> tuple[str, int, tuple[int, int]]:
        return self.path, self.monitor_index, self.monitor_offset


@dataclass(frozen=True)
class _RawBeforeShot:
    """In-memory before-shot from the grab-only loop (encode on adopt)."""

    frame_id: int
    rgb: bytes
    size: tuple[int, int]
    monitor_index: int
    monitor_offset: tuple[int, int]


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
    # Newest before-shot raw frame finished before this hook. Pinned until adopt.
    settle_frame: _RawBeforeShot | None = None
    # Hook-safe latch diagnosis (no I/O); worker writes it to record.log on adopt.
    settle_latch_note: str | None = None
    windows_before: tuple[WindowInfo, ...] | None = None
    click_window: ClickWindowInfo | None = None
    # Drag release window under end_xy (screen coords); localized at persist.
    end_click_window: ClickWindowInfo | None = None
    signals_before: dict[str, Any] | None = None
    refresh_pre_type: bool = False
    # time.monotonic() when the gesture happened, for the window-settle sleep.
    action_monotonic: float | None = None
    # Text flushed by a later gesture: after-sample is the settle cache from
    # before that gesture, not a live read taken after it has changed the UI.
    window_after_sealed: bool = False
    sealed_windows_after: tuple[WindowInfo, ...] | None = None
    sealed_signals_after: dict[str, Any] | None = None


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
# Set-1 scan codes for the numpad cluster. Dedicated arrows share these
# scans but set LLKHF_EXTENDED; the numpad key does not.
_NUMPAD_SCAN_TO_CHAR: dict[int, str] = {
    71: "7",
    72: "8",
    73: "9",
    75: "4",
    76: "5",
    77: "6",
    79: "1",
    80: "2",
    81: "3",
    82: "0",
    83: ".",
}
# Navigation virtual keys that share a scan code with a numpad key, plus the
# numpad digit/decimal virtual keys themselves.
_NUMPAD_OR_NAV_VKS = frozenset(
    {
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        45,
        46,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        110,
    }
)
_VK_NUMLOCK = 0x90
_VK_PACKET = 0xE7
_LLKHF_EXTENDED = 0x01
_WM_KEYDOWN = 0x0100
_WM_SYSKEYDOWN = 0x0104
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
    # Drag release ClickWindowInfo under end_xy (screen coords); localized at persist.
    end_click_window: ClickWindowInfo | None = None
    # Press-time UIA snapshot (screen xyxy). None when the walk did not run.
    uia_snapshot: dict[str, Any] | None = None
    # Drag-release UIA snapshot.
    end_uia_snapshot: dict[str, Any] | None = None
    # Light/UIA sample copied from the pre-click cache at input time.
    signals_before: dict[str, Any] | None = None
    # time.monotonic() when the gesture happened. The window-step thread sleeps
    # only the remainder of the settle delay from this instant.
    action_monotonic: float | None = None
    # Text flushed by the next gesture. The after list is the pre-gesture cache.
    window_after_sealed: bool = False
    windows_after: tuple[WindowInfo, ...] | None = None
    signals_after: dict[str, Any] | None = None
    # Typing-burst IME summary persisted on text_input events.
    ime: dict[str, Any] | None = None


def _pending_capture_path(run_dir: Path, press_seq: int | None = None) -> Path:
    if press_seq is None:
        return run_dir / "screenshots" / "_pending_capture.jpeg"
    return run_dir / "screenshots" / f"_pending_capture_{int(press_seq)}.jpeg"


def _pending_right_capture_path(run_dir: Path, press_seq: int | None = None) -> Path:
    if press_seq is None:
        return run_dir / "screenshots" / "_pending_right_capture.jpeg"
    return run_dir / "screenshots" / f"_pending_right_capture_{int(press_seq)}.jpeg"


def _pending_pre_type_capture_path(run_dir: Path) -> Path:
    return run_dir / "screenshots" / "_pending_pre_type.jpeg"


def _pending_pre_key_capture_path(run_dir: Path) -> Path:
    return run_dir / "screenshots" / "_pending_pre_key.jpeg"


def _pending_drag_end_capture_path(run_dir: Path, monitor_index: int) -> Path:
    return run_dir / "screenshots" / f"_pending_drag_end_mon{monitor_index}.jpeg"


def _capture_all_monitors_to_pending(
    run_dir: Path,
) -> dict[int, tuple[str, int, tuple[int, int]]]:
    """Capture every physical monitor at mouse-down, before a drag ghost exists."""
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
        try:
            if pending.is_file():
                pending.unlink()
        except OSError:
            pass


def _finalize_drag_end_screenshot(
    run_dir: Path,
    index: int,
    end_xy: tuple[int, int],
    pending_captures: dict[int, tuple[str, int, tuple[int, int]]] | None,
    *,
    fallback_mon_idx: int,
    fallback_mon_offset: tuple[int, int],
) -> tuple[str, int, tuple[int, int]]:
    """Pick the mouse-down monitor at ``end_xy`` and save it as ``event_{index}_end``.

    Used when the release is on a different monitor from the press shot.
    Same-monitor drops copy the press screenshot instead.
    """
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


def _release_monitor_index(end_xy: tuple[int, int]) -> int | None:
    """Resolved mss monitor index under the drag release point."""
    try:
        raw_idx, _, _, _, _ = _monitor_at_point(end_xy[0], end_xy[1])
    except Exception:
        return None
    try:
        with mss.mss() as sct:
            return int(resolve_monitor_index(sct, raw_idx))
    except Exception:
        return int(raw_idx)


def _copy_mouse_down_screenshot_as_drag_end(
    run_dir: Path,
    index: int,
    press_shot_path: str,
    press_mon_idx: int,
    press_mon_offset: tuple[int, int],
) -> tuple[str, int, tuple[int, int]]:
    """Use the mouse-down press frame as the drag end shot (same pixels)."""
    end_dest = screenshot_path_for_event_end(run_dir, index)
    src = Path(press_shot_path)
    end_dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file() and src.resolve() != end_dest.resolve():
        try:
            if end_dest.is_file():
                end_dest.unlink()
        except OSError:
            pass
        import shutil

        shutil.copy2(src, end_dest)
    return str(end_dest), press_mon_idx, press_mon_offset


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


_VK_LBUTTON = 0x01


def _normalize_button(button: mouse.Button) -> str:
    if button == mouse.Button.right:
        return "right"
    if button == mouse.Button.middle:
        return "middle"
    return "left"


def _os_mouse_button_down(vk: int) -> bool | None:
    """Return whether the OS currently reports ``vk`` as down.

    ``None`` means unavailable (non-Windows or query failed) — callers must
    trust pynput and must not synthesize a release.
    """
    if os.name != "nt":
        return None
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(int(vk)) & 0x8000)
    except Exception:
        return None


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


@dataclass(frozen=True)
class _HardwareKeySample:
    """One key-down as delivered by the low-level hook, before pynput translation."""

    vk: int
    scan: int
    extended: bool


def _numlock_on() -> bool:
    """Return whether NumLock is toggled on.

    ``GetKeyState``'s low bit is the toggle state. ``GetAsyncKeyState`` only
    reports whether the key is currently held down.
    """
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.user32.GetKeyState(_VK_NUMLOCK) & 1)
    except Exception:
        return False


def _numpad_char_from_hardware(sample: _HardwareKeySample | None) -> str | None:
    """Return a numpad digit when the hardware key is the numpad cluster.

    A dedicated arrow uses the same scan code with the extended bit set, and
    NumLock off makes the numpad key itself that arrow. ``VK_PACKET`` carries
    a UTF-16 character in ``scanCode``, so it is excluded by the virtual-key set.
    """
    if sample is None or sample.extended or sample.vk == _VK_PACKET:
        return None
    if sample.vk not in _NUMPAD_OR_NAV_VKS:
        return None
    if not _numlock_on():
        return None
    return _NUMPAD_SCAN_TO_CHAR.get(sample.scan)


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
    return read_clipboard_text()


def _is_paste_hotkey(mods: list[str], token: str) -> bool:
    return "ctrl" in mods and token.lower() == "v"


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    return point_in_rect_xywh(x, y, rect)


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


def _window_resolve_failure_debug(
    reason: str,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    """Minimal ``window_snapshot_debug`` when the window-step cannot finish."""
    payload: dict[str, Any] = {"resolve_failed": str(reason)}
    if detail:
        payload["resolve_failed_detail"] = str(detail)[:500]
    return payload


def _apply_next_before_to_debug(
    debug: dict[str, Any],
    windows: tuple[WindowInfo, ...] | None,
    signals: dict[str, Any] | None,
) -> None:
    """Replace the after-cache with the next gesture's before-cache.

    ``point_window`` stays from this gesture's own after sample so
    ``click_window`` still names the window that received this click.
    """
    if windows is not None:
        debug["windows_after"] = _windows_debug_list(list(windows))
        debug["windows_after_count"] = len(windows)
    if signals is None:
        return
    merged = dict(signals)
    merged.pop("point_window", None)
    existing = debug.get("signals_after")
    if isinstance(existing, dict) and "point_window" in existing:
        merged["point_window"] = existing["point_window"]
    debug["signals_after"] = merged


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
        self._hardware_key_presses: deque[_HardwareKeySample] = deque()
        self._pending_click_timer: threading.Timer | None = None
        self._pending_click_coords: tuple[int, int, str] | None = None
        self._pending_click_down_at: float | None = None
        self._pending_click_timestamp_utc: str | None = None
        self._pending_click_modifiers: list[str] | None = None
        # Reserved at mouse-down so keyboard events during the press sort after
        # the eventual click/hold/drag (index assigned at up used to invert order).
        self._pending_click_event_index: int | None = None
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
        # Settled FindAll from the previous window-step (or recording seed).
        # Press copies this as uia_snapshot when hwnd matches and the settle
        # was published after that previous event (see after_index).
        self._last_settle_uia: dict[str, Any] | None = None
        # Event index whose window-step published ``_last_settle_uia`` (0 = seed).
        self._last_settle_uia_after_index: int | None = None
        self._left_press_uia: dict[int, dict[str, Any] | None] = {}
        self._right_press_uia: dict[int, dict[str, Any] | None] = {}
        # Legacy press-time FindAll slots. New presses no longer start a live
        # walk (it often finishes after the click mutates the tree). Kept so
        # emit can still join an in-flight future from an older build/session.
        self._uia_executor: ThreadPoolExecutor | None = None
        self._left_press_uia_futures: dict[int, Future[dict[str, Any] | None]] = {}
        self._right_press_uia_futures: dict[int, Future[dict[str, Any] | None]] = {}
        self._uia_inflight: set[Future[dict[str, Any] | None]] = set()
        # Emit waits for pending press fallback and/or drag end_uia before notify.
        self._uia_notify_wait: dict[int, int] = {}
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
        # Burst IME summary while buffering typed chars (None = not yet sampled).
        self._pending_ime_saw_chinese: bool | None = None
        self._pending_ime_lang_id: int | None = None
        # Last known IME 中/英 mode for the foreground thread (None = unknown).
        self._last_ime_chinese: bool | None = None
        self._last_ime_lang_id: int | None = None
        self._paste_seq: int = 0
        # Empty-field frame captured after focus settles (click/Tab/etc.), consumed
        # as the text_input before-shot so we do not race the first typed glyph.
        self._pending_pre_type_screenshot: _PendingPreTypeShot | None = None
        # Settled frame for the next key_press before-shot (cannot capture on LL hook).
        self._pending_pre_key_screenshot: _PendingPreTypeShot | None = None
        self._pending_pre_key_timer: threading.Timer | None = None
        # Chinese IME Enter: delay flush until composition commits, then seal.
        self._pending_ime_commit_timer: threading.Timer | None = None
        self._settle_probe: _SettleProbeState | None = None
        self._settle_probe_timer: threading.Timer | None = None
        # Serialize settle ticks (timer thread) vs cancel/start (capture worker).
        self._settle_tick_lock = threading.Lock()
        self._before_shot_stop = threading.Event()
        self._before_shot_thread: threading.Thread | None = None
        # Event JSON is patched by the probe timer and by per-step window threads.
        self._event_file_lock = threading.Lock()
        self._window_step_threads: list[threading.Thread] = []
        # text_input vision notify waits until the window step writes signals_after.
        self._pending_text_notify: set[int] = set()
        # Latest event whose after-cache is still the 0.25s sample. The next
        # gesture replaces it with the settle cache copied as that gesture's before.
        self._after_owner_index: int | None = None
        self._after_seals: dict[
            int, tuple[tuple[WindowInfo, ...] | None, dict[str, Any] | None]
        ] = {}
        # Latest finished before-shot per physical monitor (raw RGB in memory).
        # The hook pins by frame_id; the worker JPEG-encodes on adopt.
        # ``_last_settle_frame`` is the most recently published entry (any monitor).
        # The settle probe does not write these.
        self._last_settle_frames: dict[int, _RawBeforeShot] = {}
        self._last_settle_frame: _RawBeforeShot | None = None
        self._before_shot_rr_index: int = 0
        self._settle_pins: dict[int, int] = {}
        self._before_shot_frame_id: int = 0
        # Monitor rectangles so a click can match a settle sample without grabbing.
        self._pre_click_monitors: tuple[tuple[int, int, int, int, int], ...] = ()
        self._pre_click_stop = threading.Event()
        # Window list and UI Automation. Separate from the settle-probe timer.
        self._pre_click_context_thread: threading.Thread | None = None
        self._context_epoch = 0
        self._window_refresh_lock = threading.Lock()
        # Top-level windows captured with that frame; mouse-down reuses this as
        # windows_before so EnumWindows never runs on the LL hook.
        self._last_settle_windows: tuple[WindowInfo, ...] | None = None
        self._last_settle_signals: dict[str, Any] | None = None
        # First scan of a recording. Hooks start before it finishes; an action
        # that latched an empty cache uses this list once the worker reaches it.
        self._settle_windows_ready = threading.Event()
        self._start_windows: tuple[WindowInfo, ...] | None = None
        self._start_signals: dict[str, Any] | None = None
        self._slow_signal_mono: float = 0.0
        self._left_press_signals: dict[int, dict[str, Any] | None] = {}
        self._right_press_signals: dict[int, dict[str, Any] | None] = {}
        self._pending_signals_before: dict[str, Any] | None = None
        self._pending_right_signals_before: dict[str, Any] | None = None
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
            self._hardware_key_presses.clear()
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._pending_click_event_index = None
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
            self._last_settle_uia = None
            self._last_settle_uia_after_index = None
            self._left_press_uia = {}
            self._right_press_uia = {}
            self._left_press_uia_futures = {}
            self._right_press_uia_futures = {}
            self._uia_inflight = set()
            self._uia_notify_wait = {}
            self._drag_end_captures = {}
            self._pending_right_coords = None
            self._pending_right_down_at = None
            self._pending_right_timestamp_utc = None
            self._pending_right_modifiers = None
            self._pending_right_screenshot = None
            self._pending_right_windows_before = None
            self._pending_right_click_window = None
            self._pending_text_chars = []
            self._pending_text_caret = 0
            self._pending_text_meta = None
            self._pending_ime_saw_chinese = None
            self._pending_ime_lang_id = None
            self._last_ime_chinese = None
            self._last_ime_lang_id = None
            self._paste_seq = 0
            self._pending_pre_type_screenshot = None
            self._pending_pre_key_screenshot = None
            self._cancel_pre_key_settle_timer_locked()
            self._cancel_ime_commit_timer_locked()
            self._settle_probe = None
            self._settle_probe_timer = None
            self._before_shot_stop.clear()
            self._before_shot_thread = None
            self._window_step_threads = []
            self._pending_text_notify = set()
            self._last_settle_frames = {}
            self._last_settle_frame = None
            self._before_shot_rr_index = 0
            self._settle_pins = {}
            self._before_shot_frame_id = 0
            self._last_settle_windows = None
            self._last_settle_signals = None
            self._last_settle_uia = None
            self._last_settle_uia_after_index = None
            self._start_windows = None
            self._start_signals = None
            self._settle_windows_ready.clear()
            self._slow_signal_mono = 0.0
            self._left_press_signals = {}
            self._right_press_signals = {}
            self._pending_signals_before = None
            self._pending_right_signals_before = None
            self._pre_click_monitors = ()
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
        self._start_uia_executor()

        # Snapshot stays the first worker job. Hooks do not wait for it.
        self._enqueue(_DeferredCaptureJob(action="seed_settle_windows"))
        with self._lock:
            self._accepting_input = True

        self._start_pre_click_pump()
        self._start_before_shot_loop()

        self._mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
            on_move=self._on_mouse_move,
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
            win32_event_filter=self._on_win32_key_filter,
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
            if pending_coords is not None:
                press_windows = self._pending_windows_before
                press_signals = self._pending_signals_before
            else:
                press_windows = self._pending_right_windows_before
                press_signals = self._pending_right_signals_before
            self._pending_click_timer = None

        self._stop_pre_click_pump(invalidate_context=True)
        self._stop_before_shot_loop()

        # A pending click/drag/right-click is flushed next and has already been
        # delivered. Seal the text after-sample from that press copy so a cache
        # refresh cannot give the typing step a different foreground.
        following_gesture = pending_coords is not None or pending_right_coords is not None
        # Drop a pending Chinese-IME Enter delay; stop must not leave text hanging.
        self._cancel_ime_commit_timer()
        self._flush_pending_text_input(
            seal_after_to_cache=following_gesture,
            sealed_windows=press_windows,
            sealed_signals=press_signals,
            seal_sample_supplied=following_gesture,
        )
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
            self._pending_signals_before = None
            self._pending_click_window = None
            self._pending_click_event_index = None
            leftover_pre_type = self._pending_pre_type_screenshot
            self._pending_pre_type_screenshot = None
            leftover_pre_key = self._pending_pre_key_screenshot
            self._pending_pre_key_screenshot = None
            self._cancel_pre_key_settle_timer_locked()
            self._cancel_ime_commit_timer_locked()
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
        self._shutdown_uia_executor()
        self._join_window_step_threads()
        self._flush_pending_text_notifies()

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

    def _log_key_press(
        self,
        *,
        action: str,
        token: str | None = None,
        typed: str | None = None,
        mods: list[str] | None = None,
        key: keyboard.Key | keyboard.KeyCode | None = None,
        ime: ImeInputState | None = None,
    ) -> None:
        """Append one key-down line to ``record.log`` for typing/debug traces.

        Safe to call from the keyboard hook: failures never raise. Logs the
        pending text buffer after the press was applied so Backspace/edits are
        visible next to the key that changed them.
        """
        try:
            with self._lock:
                run_dir = self._run_dir
                if run_dir is None:
                    return
                if self._pending_text_meta is not None:
                    pending = "".join(self._pending_text_chars)
                    caret = self._pending_text_caret
                    raw_index = self._pending_text_meta.get("index")
                    text_event = raw_index if isinstance(raw_index, int) else None
                else:
                    pending = None
                    caret = None
                    text_event = None
                lang_id = self._last_ime_lang_id
                chinese = self._last_ime_chinese
            if ime is not None:
                lang_label = ime.lang_label
                mode_label = ime.mode_label
            else:
                lang_label = f"0x{lang_id:04x}" if lang_id is not None else "?"
                mode_label = (
                    "zh" if chinese is True else "en" if chinese is False else "?"
                )
            mods_s = "+".join(mods) if mods else "-"
            key_s = str(key) if key is not None else "-"
            pending_s = repr(pending) if pending is not None else "-"
            caret_s = str(caret) if caret is not None else "-"
            event_s = str(text_event) if text_event is not None else "-"
            self._log(
                run_dir,
                "key "
                f"action={action} token={token!r} typed={typed!r} mods={mods_s} "
                f"key={key_s} ime={mode_label} lang={lang_label} "
                f"pending_text={pending_s} caret={caret_s} text_event={event_s}",
            )
        except Exception:
            pass

    def _sync_ime_state(self) -> ImeInputState:
        """Sample IME 中/英 mode; wipe the pending text buffer on zh→en switches."""
        state = read_ime_input_state()
        wiped: str | None = None
        with self._lock:
            prev = self._last_ime_chinese
            if state.lang_id is not None:
                self._last_ime_lang_id = state.lang_id
            if state.chinese_mode is not None:
                self._last_ime_chinese = state.chinese_mode
            if prev is True and state.chinese_mode is False:
                if self._pending_text_chars:
                    wiped = "".join(self._pending_text_chars)
                self._pending_text_chars = []
                self._pending_text_caret = 0
                # Composition Latin keys are gone; restart IME summary for the
                # English burst that follows.
                self._pending_ime_saw_chinese = None
                self._pending_ime_lang_id = None
            run_dir = self._run_dir
        if wiped is not None and run_dir is not None:
            try:
                self._log(
                    run_dir,
                    "key "
                    f"action=ime_zh_to_en_wipe wiped={wiped!r} "
                    f"ime={state.mode_label} lang={state.lang_label}",
                )
            except Exception:
                pass
        return state

    def _flush_text_without_key_step(self) -> None:
        """Schedule a delayed sealed flush after Chinese IME Enter (no key step).

        Wait briefly so composition commits to characters, then live-sample and
        seal ``signals_after``. Do not seal the pre-Enter settle cache (still
        mid-composition) and do not leave an unsealed after for a late window
        step to race against later UI.
        """
        self._schedule_ime_commit_flush()

    def _notify_event(self, event: RecordedEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass

    def _notify_pending_text_input(self, event_index: int) -> None:
        """Fire deferred vision notify once the window step has patched (or failed)."""
        with self._lock:
            if event_index not in self._pending_text_notify:
                return
            self._pending_text_notify.discard(event_index)
            event: RecordedEvent | None = None
            for candidate in self._events:
                if candidate.index == event_index:
                    event = candidate
                    break
        if event is not None:
            self._notify_event(event)

    def _flush_pending_text_notifies(self) -> None:
        """Notify any text_input still waiting after window-step threads join."""
        with self._lock:
            pending = sorted(self._pending_text_notify)
        for event_index in pending:
            self._notify_pending_text_input(event_index)

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

        Called from the context loop, the settle loop, and once at recording
        start (worker thread), never from the mouse hook or the grab loop.
        Light verify signals are sampled here too. Clipboard and UI Automation
        stay on ``_refresh_slow_verify_signals``.
        """
        with self._lock:
            run_dir = self._run_dir
            epoch = self._context_epoch
        with self._window_refresh_lock:
            try:
                windows = tuple(snapshot_top_level_windows())
            except Exception:
                return
            try:
                signals = capture_fast_signals()
            except Exception:
                signals = {}
            with self._lock:
                if self._run_dir is not run_dir or epoch != self._context_epoch:
                    return
                self._last_settle_windows = windows
                merged = dict(self._last_settle_signals or {})
                merged.update(signals)
                self._last_settle_signals = merged

    def _refresh_slow_verify_signals(self) -> None:
        """Refresh clipboard and UI Automation at most once a second.

        A clipboard owner or a wedged UI Automation provider can stall. This
        runs on the context thread, so that stall cannot block a settle-probe grab.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._slow_signal_mono < 1.0:
                return
            self._slow_signal_mono = now
            cursor = self._last_pointer_cursor_xy
            run_dir = self._run_dir
            epoch = self._context_epoch
        if cursor is None:
            try:
                pos = pyautogui.position()
                cursor = (int(pos.x), int(pos.y))
            except Exception:
                cursor = None
        try:
            slow = capture_slow_signals(cursor)
        except Exception:
            return
        if not slow:
            return
        with self._lock:
            if self._run_dir is not run_dir or epoch != self._context_epoch:
                return
            merged = dict(self._last_settle_signals or {})
            merged.update(slow)
            self._last_settle_signals = merged

    def _cached_windows_before(self) -> tuple[WindowInfo, ...] | None:
        """Pre-click window list from the last settle sample (or the start seed)."""
        with self._lock:
            cached = self._last_settle_windows
        if not cached:
            return None
        return cached

    def _cached_signals_before(self) -> dict[str, Any] | None:
        """Pre-click signal sample. The hook only copies this dict."""
        with self._lock:
            cached = self._last_settle_signals
        if not cached:
            return None
        return dict(cached)

    def _claim_after_owner(self, event_index: int) -> None:
        """This event's after-cache is the 0.25s sample until a later gesture seals it.

        A newer owner is left in place. The last event is never sealed, so its
        0.25s sample remains the after-cache when recording stops.
        """
        with self._lock:
            if event_index in self._after_seals:
                return
            owner = self._after_owner_index
            if owner is None or owner == event_index:
                self._after_owner_index = event_index

    def _seal_open_after_cache(
        self,
        windows: tuple[WindowInfo, ...] | None,
        signals: dict[str, Any] | None,
    ) -> None:
        """Write this gesture's before-cache as the previous event's after-cache."""
        if not windows and not signals:
            return
        seal = (
            tuple(windows) if windows else None,
            dict(signals) if signals else None,
        )
        with self._lock:
            owner = self._after_owner_index
            if owner is None:
                return
            self._after_owner_index = None
            self._after_seals[owner] = seal
            run_dir = self._run_dir
        if run_dir is None:
            return
        # The hook must not rewrite event JSON. A window step that has not
        # finished applies the seal itself; this thread patches one that has.
        thread = threading.Thread(
            target=self._apply_stored_after_seal,
            args=(run_dir, owner),
            name=f"window-after-seal-{owner}",
            daemon=True,
        )
        thread.start()
        with self._lock:
            self._window_step_threads.append(thread)

    def _apply_stored_after_seal(self, run_dir: Path, event_index: int) -> None:
        """Patch an after-cache that the window step already wrote."""
        with self._event_json_lock():
            with self._lock:
                seal = self._after_seals.get(event_index)
            if seal is None:
                return
            if not self._patch_sealed_after_cache(run_dir, event_index, seal):
                return
            with self._lock:
                if self._after_seals.get(event_index) == seal:
                    self._after_seals.pop(event_index, None)

    def _patch_sealed_after_cache(
        self,
        run_dir: Path,
        event_index: int,
        seal: tuple[tuple[WindowInfo, ...] | None, dict[str, Any] | None],
    ) -> bool:
        """Patch a finished window debug. Caller holds the event JSON lock."""
        path = event_json_path(run_dir, event_index)
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            return False
        debug = raw.get("window_snapshot_debug")
        if not isinstance(debug, dict):
            return False
        # Sidecar peels leave signals_* as paths; hydrate so point_window survives.
        hydrated = hydrate_window_snapshot_debug(run_dir, event_index, debug)
        if isinstance(hydrated, dict):
            debug = hydrated
        _apply_next_before_to_debug(debug, seal[0], seal[1])
        raw["window_snapshot_debug"] = debug
        write_event_json(run_dir, raw)
        with self._lock:
            events = self._events
            for index, event in enumerate(events):
                if event.index == event_index:
                    events[index] = RecordedEvent.from_dict(raw)
                    break
        return True

    def _start_windows_before(self) -> tuple[WindowInfo, ...] | None:
        """Window list from the recording-start scan, if that scan has finished."""
        with self._lock:
            cached = self._start_windows
        if not cached:
            return None
        return cached

    def _start_signals_before(self) -> dict[str, Any] | None:
        with self._lock:
            cached = self._start_signals
        if not cached:
            return None
        return dict(cached)

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
        signals_before = self._cached_signals_before()
        self._seal_open_after_cache(windows_before, signals_before)
        click_window = self._click_window_payload_at(x, y)
        # Settle-cache only. A press-time live FindAll races the click itself
        # (Excel File → Backstage turns 「檔案」 into 「返回」 before FindAll
        # returns). Miss → None → analysis uses YOLO on the before-screenshot.
        uia_snapshot = self._cached_uia_before(click_window, cursor_xy=(x, y))
        with self._lock:
            self._left_press_seq += 1
            seq = self._left_press_seq
            self._pending_screenshot = None
            self._pending_windows_before = windows_before or None
            self._pending_signals_before = signals_before
            self._pending_click_window = click_window
            self._left_press_captures[seq] = (None, windows_before or None)
            self._left_press_signals[seq] = signals_before
            self._left_press_click_windows[seq] = click_window
            self._left_press_uia[seq] = uia_snapshot
        settle_frame, settle_latch_note = self._latch_settle_frame(x, y)
        self._enqueue(
            _DeferredCaptureJob(
                action="pending_left_press",
                cursor_xy=(x, y),
                press_seq=seq,
                text_event_index=text_event_index,
                flush_meta=text_meta,
                settle_frame=settle_frame,
                settle_latch_note=settle_latch_note,
            )
        )
        # Other monitors, captured now so a later cross-monitor drop is pre-drag.
        self._capture_pending_drag_end_screens()

    def _capture_pending_right_press(self, run_dir: Path, x: int, y: int) -> None:
        """Reuse the settle-loop window cache; defer only the right-press before-shot."""
        _ = run_dir
        windows_before = self._cached_windows_before()
        signals_before = self._cached_signals_before()
        self._seal_open_after_cache(windows_before, signals_before)
        click_window = self._click_window_payload_at(x, y)
        uia_snapshot = self._cached_uia_before(click_window, cursor_xy=(x, y))
        with self._lock:
            self._right_press_seq += 1
            seq = self._right_press_seq
            self._pending_right_screenshot = None
            self._pending_right_windows_before = windows_before or None
            self._pending_right_signals_before = signals_before
            self._pending_right_click_window = click_window
            self._right_press_captures[seq] = (None, windows_before or None)
            self._right_press_signals[seq] = signals_before
            self._right_press_click_windows[seq] = click_window
            self._right_press_uia[seq] = uia_snapshot
        settle_frame, settle_latch_note = self._latch_settle_frame(x, y)
        self._enqueue(
            _DeferredCaptureJob(
                action="pending_right_press",
                cursor_xy=(x, y),
                press_seq=seq,
                settle_frame=settle_frame,
                settle_latch_note=settle_latch_note,
            )
        )

    def _start_pre_click_pump(self) -> None:
        """Refresh windows and UI Automation off the mouse hook."""
        self._stop_pre_click_pump()
        self._pre_click_stop.clear()
        context = threading.Thread(
            target=self._pre_click_context_loop,
            name="screen-recorder-preclick-context",
            daemon=True,
        )
        with self._lock:
            self._pre_click_context_thread = context
        self._refresh_pre_click_monitors()
        context.start()

    def _stop_pre_click_pump(self, *, invalidate_context: bool = False) -> None:
        self._pre_click_stop.set()
        with self._lock:
            if invalidate_context:
                self._context_epoch += 1
            context = self._pre_click_context_thread
            self._pre_click_context_thread = None
        if (
            context is not None
            and context.is_alive()
            and context is not threading.current_thread()
        ):
            context.join(timeout=1.0)

    def _pre_click_context_loop(self) -> None:
        """Refresh the window list and UI Automation until stop.

        Never runs on the hook or the grab thread. Waits for the recording-start
        scan so a second EnumWindows cannot replace that list before an early
        click records it.
        """
        me = threading.current_thread()
        while not self._pre_click_stop.is_set():
            if not self._settle_windows_ready.is_set():
                if self._pre_click_stop.wait(0.02):
                    return
                continue
            with self._lock:
                if self._pre_click_context_thread is not me:
                    return
                run_dir = self._run_dir
                accepting = self._accepting_input
            if run_dir is None or not accepting:
                if self._pre_click_stop.wait(_PRE_CLICK_CONTEXT_INTERVAL_S):
                    return
                continue
            self._refresh_last_settle_windows()
            self._refresh_slow_verify_signals()
            self._refresh_pre_click_monitors()
            if self._pre_click_stop.wait(_PRE_CLICK_CONTEXT_INTERVAL_S):
                return

    def _refresh_pre_click_monitors(self) -> None:
        """Cache monitor rectangles so a click can match a frame without grabbing."""
        try:
            with mss.mss() as sct:
                monitors = tuple(
                    (
                        idx,
                        int(sct.monitors[idx]["left"]),
                        int(sct.monitors[idx]["top"]),
                        int(sct.monitors[idx]["width"]),
                        int(sct.monitors[idx]["height"]),
                    )
                    for idx in range(1, len(sct.monitors))
                )
        except Exception:
            return
        with self._lock:
            self._pre_click_monitors = monitors

    def _monitor_index_for_point(self, x: int, y: int) -> int | None:
        with self._lock:
            monitors = self._pre_click_monitors
        for index, left, top, width, height in monitors:
            if left <= int(x) < left + width and top <= int(y) < top + height:
                return int(index)
        try:
            return int(_monitor_at_point(x, y)[0])
        except Exception:
            return None

    def _alloc_before_shot_frame_id(self) -> int:
        with self._lock:
            self._before_shot_frame_id += 1
            return int(self._before_shot_frame_id)

    def _install_raw_before_shot(self, frame: _RawBeforeShot) -> bool:
        """Install a grab-only raw frame for ``frame.monitor_index``."""
        mon_key = int(frame.monitor_index)
        with self._lock:
            self._last_settle_frames[mon_key] = frame
            self._last_settle_frame = frame
        return True

    def _publish_last_settle_frame(
        self,
        source: Path,
        mon_index: int,
        mon_offset: tuple[int, int],
        *,
        event_index: int | None = None,
    ) -> bool:
        """Copy a finalize/stop JPEG to the durable debug path.

        Does not update per-monitor click pins (those are raw frames from the
        before-shot loop). ``mon_index`` / ``mon_offset`` / ``event_index`` are
        accepted for call-site compatibility.
        """
        _ = mon_index, mon_offset, event_index
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None or not source.is_file():
            return False
        durable = last_settle_frame_path(run_dir)
        durable.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil

            if source.resolve() != durable.resolve():
                shutil.copy2(source, durable)
        except OSError:
            return False
        return True

    def _latest_settle_frame_for_monitor(
        self,
        monitor_index: int,
    ) -> _RawBeforeShot | None:
        """Return the newest published before-shot for ``monitor_index``."""
        mon_key = int(monitor_index)
        with self._lock:
            published = self._last_settle_frames.get(mon_key)
            if published is not None:
                return published
            last = self._last_settle_frame
            if last is not None and int(last.monitor_index) == mon_key:
                return last
        return None

    def _latch_settle_frame(
        self,
        x: int,
        y: int,
    ) -> tuple[_RawBeforeShot | None, str]:
        """Pin the newest finished before-shot for the click's monitor.

        Hook-safe: no capture, no encode, no log I/O. The note string is logged
        later on the worker when the frame is adopted (or when adopt falls back).
        """
        click_mon = self._monitor_index_for_point(x, y)
        if click_mon is None:
            return None, f"miss xy=({int(x)},{int(y)}) reason=no_click_monitor"
        with self._lock:
            published = self._last_settle_frames.get(int(click_mon))
            if published is None:
                last = self._last_settle_frame
                if last is not None and int(last.monitor_index) == int(click_mon):
                    published = last
            if published is None:
                published_mons = sorted(int(k) for k in self._last_settle_frames.keys())
                last = self._last_settle_frame
                last_mon = int(last.monitor_index) if last is not None else None
                return (
                    None,
                    f"miss xy=({int(x)},{int(y)}) reason=no_frame_for_mon "
                    f"click_mon={int(click_mon)} published_mons={published_mons} "
                    f"last_mon={last_mon}",
                )
            frame_id = int(published.frame_id)
            self._settle_pins[frame_id] = self._settle_pins.get(frame_id, 0) + 1
            return (
                published,
                f"ok xy=({int(x)},{int(y)}) frame_id={frame_id} "
                f"mon={int(published.monitor_index)} "
                f"offset={tuple(published.monitor_offset)} "
                f"size={tuple(published.size)}",
            )

    def _log_before_shot(self, run_dir: Path | None, text: str) -> None:
        """Append a before-shot debug line; never raise (worker / tests)."""
        if run_dir is None:
            return
        log_fn = getattr(self, "_log", None)
        if not callable(log_fn):
            return
        try:
            log_fn(run_dir, f"before-shot {text}")
        except Exception:
            pass

    def _log_before_shot_finalize(
        self,
        run_dir: Path | None,
        *,
        event_index: int | None,
        press_seq: int | None,
        pending_shot: tuple[str, int, tuple[int, int]] | None,
        side: str,
    ) -> None:
        """Log whether emit reused the press pending JPEG or live-captures."""
        if pending_shot is None:
            self._log_before_shot(
                run_dir,
                f"finalize side={side} event={event_index} press_seq={press_seq} "
                f"pending=missing path=live_capture",
            )
            return
        src = Path(pending_shot[0])
        if src.is_file():
            self._log_before_shot(
                run_dir,
                f"finalize side={side} event={event_index} press_seq={press_seq} "
                f"pending=ok mon={pending_shot[1]} src={src.name}",
            )
            return
        self._log_before_shot(
            run_dir,
            f"finalize side={side} event={event_index} press_seq={press_seq} "
            f"pending=missing_file src={src.name} path=live_capture",
        )

    def _unpin_settle_frame(
        self,
        info: _RawBeforeShot | None,
    ) -> None:
        if info is None:
            return
        frame_id = int(info.frame_id)
        with self._lock:
            count = self._settle_pins.get(frame_id, 0) - 1
            if count > 0:
                self._settle_pins[frame_id] = count
                return
            self._settle_pins.pop(frame_id, None)

    def _encode_raw_before_shot(
        self,
        frame: _RawBeforeShot,
        dest: Path,
        *,
        write_durable: bool = True,
    ) -> tuple[tuple[str, int, tuple[int, int]] | None, str]:
        """JPEG-encode a raw before-shot to ``dest`` (worker / adopt path)."""
        try:
            from PIL import Image

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dest.is_file():
                    dest.unlink()
            except OSError:
                pass
            img = Image.frombytes("RGB", frame.size, frame.rgb)
            img.save(dest, format="JPEG")
            if write_durable:
                with self._lock:
                    run_dir = self._run_dir
                if run_dir is not None:
                    durable = last_settle_frame_path(run_dir)
                    try:
                        import shutil

                        if dest.resolve() != durable.resolve():
                            shutil.copy2(dest, durable)
                    except OSError:
                        pass
            return (
                (
                    str(dest),
                    int(frame.monitor_index),
                    (int(frame.monitor_offset[0]), int(frame.monitor_offset[1])),
                ),
                "encode_ok",
            )
        except Exception as exc:
            return None, f"encode_fail {type(exc).__name__}: {exc}"

    def _take_latched_settle_frame(
        self,
        latched: _RawBeforeShot | None,
        x: int,
        y: int,
        dest: Path,
    ) -> tuple[tuple[str, int, tuple[int, int]] | None, str]:
        """Encode the raw frame pinned at the hook when it matches the click monitor."""
        if latched is None:
            return None, "adopt_skip reason=no_latched_frame"
        try:
            click_mon = self._monitor_index_for_point(x, y)
            if click_mon is None:
                return None, (
                    f"adopt_skip reason=no_click_monitor_at_adopt "
                    f"latched_mon={int(latched.monitor_index)} "
                    f"xy=({int(x)},{int(y)})"
                )
            if int(latched.monitor_index) != int(click_mon):
                return None, (
                    f"adopt_skip reason=monitor_mismatch "
                    f"latched_mon={int(latched.monitor_index)} "
                    f"click_mon={int(click_mon)} xy=({int(x)},{int(y)})"
                )
            encoded, encode_note = self._encode_raw_before_shot(latched, dest)
            if encoded is None:
                return None, (
                    f"adopt_skip reason=encode_fail "
                    f"frame_id={int(latched.frame_id)} mon={int(latched.monitor_index)} "
                    f"{encode_note}"
                )
            return encoded, (
                f"adopt_ok frame_id={int(latched.frame_id)} "
                f"mon={int(latched.monitor_index)} dest={dest.name}"
            )
        except Exception as exc:
            return None, f"adopt_skip reason=exception {type(exc).__name__}: {exc}"
        finally:
            self._unpin_settle_frame(latched)

    def _copy_published_settle_frame(
        self,
        published: _RawBeforeShot,
        dest: Path,
    ) -> tuple[tuple[str, int, tuple[int, int]] | None, str]:
        encoded, encode_note = self._encode_raw_before_shot(published, dest)
        if encoded is None:
            return None, (
                f"published_encode_fail frame_id={int(published.frame_id)} "
                f"mon={int(published.monitor_index)} {encode_note}"
            )
        return encoded, (
            f"published_ok frame_id={int(published.frame_id)} "
            f"mon={int(published.monitor_index)} dest={dest.name}"
        )

    def _pending_screenshot_from_settle_or_capture(
        self,
        run_dir: Path,
        x: int,
        y: int,
        dest: Path,
        settle_frame: _RawBeforeShot | None = None,
        *,
        settle_latch_note: str | None = None,
        press_seq: int | None = None,
    ) -> tuple[str, int, tuple[int, int]]:
        """Adopt the before-shot frame pinned at the hook (JPEG on the worker).

        That frame is the newest before-shot loop sample for this click's
        monitor finished before mouse-down. If the pin is missing or the wrong
        monitor, reuse the latest published frame for the click monitor.
        Otherwise grab live.
        """
        prefix = f"press_seq={press_seq} " if press_seq is not None else ""
        if settle_latch_note:
            self._log_before_shot(run_dir, f"{prefix}latch {settle_latch_note}")
        taken, adopt_note = self._take_latched_settle_frame(settle_frame, x, y, dest)
        if taken is not None:
            self._log_before_shot(run_dir, f"{prefix}path=latched {adopt_note}")
            return taken
        self._log_before_shot(run_dir, f"{prefix}{adopt_note}")
        click_mon = self._monitor_index_for_point(x, y)
        if click_mon is None:
            try:
                click_mon, _, _, _, _ = _monitor_at_point(x, y)
            except Exception:
                click_mon = None
        if click_mon is not None:
            last = self._latest_settle_frame_for_monitor(int(click_mon))
            if last is not None:
                copied, published_note = self._copy_published_settle_frame(last, dest)
                if copied is not None:
                    self._log_before_shot(
                        run_dir,
                        f"{prefix}path=published_fallback "
                        f"click_mon={int(click_mon)} {published_note}",
                    )
                    return copied
                self._log_before_shot(
                    run_dir,
                    f"{prefix}published_fallback_failed "
                    f"click_mon={int(click_mon)} {published_note}",
                )
        self._log_before_shot(
            run_dir,
            f"{prefix}path=live_fallback xy=({int(x)},{int(y)}) "
            f"click_mon={click_mon} dest={dest.name}",
        )
        return _capture_screenshot_at_point(x, y, dest)

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
        event_index: int | None = None,
    ) -> None:
        """Reserve an index and enqueue screenshot/UIA work off the mouse hook thread.

        Pass ``event_index`` to reuse an index reserved earlier (e.g. first press
        of a double-click).
        """
        action_timestamp_utc = timestamp_utc or utc_now_iso()
        with self._lock:
            run_dir = self._run_dir
            if run_dir is None:
                return
            if event_index is None:
                index = self._next_index
                self._next_index += 1
            else:
                index = event_index
        # One settle-cache copy feeds the flushed text, this event's before-sample,
        # and the previous event's after-cache.
        windows_before = self._cached_windows_before()
        signals_before = self._cached_signals_before()
        self._flush_pending_text_input(
            shared_end_index=index,
            seal_after_to_cache=True,
            sealed_windows=windows_before,
            sealed_signals=signals_before,
            seal_sample_supplied=True,
        )
        self._remember_pointer_cursor(cursor_xy)
        # Pre-click window list comes from the settle cache. click_window is one
        # WindowFromPoint (no full window scan) and stays on the hook.
        # Screenshot stays deferred.
        self._seal_open_after_cache(windows_before, signals_before)
        self._claim_after_owner(index)
        click_window = self._click_window_payload_at(int(cursor_xy[0]), int(cursor_xy[1]))
        settle_frame, settle_latch_note = self._latch_settle_frame(
            int(cursor_xy[0]),
            int(cursor_xy[1]),
        )
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
                signals_before=signals_before,
                refresh_pre_type=True,
                settle_frame=settle_frame,
                settle_latch_note=settle_latch_note,
                action_monotonic=time.monotonic(),
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
            ime_payload = self._pending_ime_payload_locked()
            self._pending_text_chars = []
            self._pending_text_caret = 0
            self._pending_text_meta = None
            self._pending_ime_saw_chinese = None
            self._pending_ime_lang_id = None
            self._cancel_pre_key_settle_timer_locked()
            pending_pre_key = self._pending_pre_key_screenshot
            self._pending_pre_key_screenshot = None
        if flush_meta is not None and ime_payload is not None:
            flush_meta["ime"] = ime_payload
        shared_index = index if cursor_xy is not None else None
        windows_before = self._cached_windows_before()
        signals_before = self._cached_signals_before()
        self._seal_open_after_cache(windows_before, signals_before)
        self._claim_after_owner(index)
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
                action_monotonic=time.monotonic(),
                flush_chars=flush_chars or None,
                flush_meta=flush_meta,
                shared_end_index=shared_index,
                pending_pre_key=pending_pre_key,
                windows_before=windows_before,
                signals_before=signals_before,
                refresh_pre_type=True,
                window_after_sealed=windows_before is not None,
                sealed_windows_after=windows_before,
                sealed_signals_after=signals_before,
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

    def _cancel_ime_commit_timer_locked(self) -> None:
        pending = self._pending_ime_commit_timer
        self._pending_ime_commit_timer = None
        if pending is not None:
            pending.cancel()

    def _cancel_ime_commit_timer(self) -> None:
        with self._lock:
            self._cancel_ime_commit_timer_locked()

    def _schedule_ime_commit_flush(self) -> None:
        """After Chinese IME Enter, flush once composition has time to commit."""
        with self._lock:
            if (
                self._run_dir is None
                or not self._accepting_input
                or self._finalizing
                or self._pending_text_meta is None
            ):
                return
            self._cancel_ime_commit_timer_locked()
            timer = threading.Timer(
                _IME_COMMIT_SETTLE_S, self._on_ime_commit_settle_timer
            )
            timer.daemon = True
            self._pending_ime_commit_timer = timer
            timer.start()

    def _on_ime_commit_settle_timer(self) -> None:
        with self._lock:
            self._pending_ime_commit_timer = None
            if (
                self._run_dir is None
                or self._finalizing
                or self._pending_text_meta is None
            ):
                return
            run_dir = self._run_dir
        try:
            self._log(
                run_dir,
                "key action=ime_commit_flush_scheduled "
                f"delay_s={_IME_COMMIT_SETTLE_S}",
            )
        except Exception:
            pass
        self._enqueue(_DeferredCaptureJob(action="flush_ime_commit_text"))

    def _flush_ime_commit_with_sealed_sample(self) -> None:
        """Live-sample committed UIA and seal it onto the pending text_input.

        Used by the delayed IME Enter worker job and by stop when no following
        gesture already supplied an after-sample.
        """
        with self._lock:
            if not self._pending_text_chars or self._pending_text_meta is None:
                return
            meta = self._pending_text_meta
            cursor = meta.get("cursor_xy")
        cursor_xy: tuple[int, int] | None = None
        if isinstance(cursor, (tuple, list)) and len(cursor) == 2:
            try:
                cursor_xy = (int(cursor[0]), int(cursor[1]))
            except (TypeError, ValueError):
                cursor_xy = None
        try:
            signals = capture_step_signals(
                cursor_xy=cursor_xy,
                kind="text_input",
            )
        except Exception:
            signals = {}
        if not isinstance(signals, dict):
            signals = {}
        windows = self._cached_windows_before() or ()
        self._flush_pending_text_input(
            seal_after_to_cache=True,
            sealed_windows=windows,
            sealed_signals=signals,
            seal_sample_supplied=True,
        )

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

    def _pending_ime_payload_locked(self) -> dict[str, Any] | None:
        """Build ``ime`` dict from the current burst summary, or None."""
        if (
            self._pending_ime_saw_chinese is None
            and self._pending_ime_lang_id is None
        ):
            return None
        payload: dict[str, Any] = {}
        if self._pending_ime_saw_chinese is not None:
            payload["chinese"] = bool(self._pending_ime_saw_chinese)
        if self._pending_ime_lang_id is not None:
            payload["lang_id"] = int(self._pending_ime_lang_id)
        return payload or None

    def _note_pending_ime_locked(self) -> None:
        """Fold the latest IME sample into the pending typing-burst summary."""
        lang = self._last_ime_lang_id
        chinese = self._last_ime_chinese
        if lang is not None:
            self._pending_ime_lang_id = int(lang)
        if chinese is True:
            self._pending_ime_saw_chinese = True
        elif chinese is False and self._pending_ime_saw_chinese is None:
            self._pending_ime_saw_chinese = False

    def _flush_pending_text_input(
        self,
        *,
        shared_end_index: int | None = None,
        shared_end_monitor: int | None = None,
        shared_end_offset: tuple[int, int] | None = None,
        seal_after_to_cache: bool = False,
        sealed_windows: tuple[WindowInfo, ...] | None = None,
        sealed_signals: dict[str, Any] | None = None,
        seal_sample_supplied: bool = False,
    ) -> None:
        """Detach pending typed text and enqueue OCR/screenshot work off-hook.

        ``seal_after_to_cache`` is set when a later gesture is flushing this
        text. The after-sample is that gesture's already-copied settle cache,
        not a second read. A live window read would run after the gesture and
        attribute its window change to the typing step. ``point_window`` stays
        off this sample; the gesture records its own click target.
        """
        with self._lock:
            # A following gesture (or stop) wins over a delayed Chinese IME Enter.
            self._cancel_ime_commit_timer_locked()
            chars = self._pending_text_chars
            meta = self._pending_text_meta
            ime_payload = self._pending_ime_payload_locked()
            self._pending_text_chars = []
            self._pending_text_caret = 0
            self._pending_text_meta = None
            self._pending_ime_saw_chinese = None
            self._pending_ime_lang_id = None
            self._cancel_pre_key_settle_timer_locked()
        if not chars or meta is None:
            return
        if ime_payload is not None:
            meta["ime"] = ime_payload
        if seal_after_to_cache and not seal_sample_supplied:
            sealed_windows = self._cached_windows_before()
            sealed_signals = self._cached_signals_before()
        elif not seal_after_to_cache:
            sealed_windows = None
            sealed_signals = None
        if isinstance(sealed_signals, dict):
            sealed_signals = dict(sealed_signals)
            sealed_signals.pop("point_window", None)
        sealed = sealed_windows is not None or sealed_signals is not None
        self._enqueue(
            _DeferredCaptureJob(
                action="flush_text_input",
                flush_chars=chars,
                flush_meta=meta,
                shared_end_index=shared_end_index,
                shared_end_monitor=shared_end_monitor,
                shared_end_offset=shared_end_offset,
                window_after_sealed=sealed and seal_after_to_cache,
                sealed_windows_after=sealed_windows if seal_after_to_cache else None,
                sealed_signals_after=sealed_signals if seal_after_to_cache else None,
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
            self._pending_ime_saw_chinese = None
            self._pending_ime_lang_id = None
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
                "windows_before": self._last_settle_windows,
                "signals_before": (
                    dict(self._last_settle_signals) if self._last_settle_signals else None
                ),
            }
            self._pending_text_meta = meta
        self._seal_open_after_cache(
            meta.get("windows_before") if isinstance(meta.get("windows_before"), tuple) else None,
            meta.get("signals_before") if isinstance(meta.get("signals_before"), dict) else None,
        )
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
            self._note_pending_ime_locked()
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
            ime_payload = None
            if not detached:
                flush_chars = self._pending_text_chars
                flush_meta = self._pending_text_meta
                ime_payload = self._pending_ime_payload_locked()
                self._pending_text_chars = []
                self._pending_text_caret = 0
                self._pending_text_meta = None
                self._pending_ime_saw_chinese = None
                self._pending_ime_lang_id = None
            chars_empty = chars is None or not chars
            if chars_empty:
                raw_index = (job.meta or {}).get("text_index")
                if isinstance(raw_index, int):
                    orphan_index = raw_index
                if not detached:
                    flush_meta = None
                    flush_chars = []
                    ime_payload = None
            if orphan_index is None:
                orphan_index = self._next_index
                self._next_index += 1
            self._cancel_pre_key_settle_timer_locked()
            pending_pre_key = self._pending_pre_key_screenshot
            self._pending_pre_key_screenshot = None
        if flush_meta is not None and ime_payload is not None:
            flush_meta["ime"] = ime_payload
        if flush_chars and flush_meta is not None:
            sealed_windows = self._cached_windows_before()
            self._worker_flush_text_input(
                _DeferredCaptureJob(
                    action="flush_text_input",
                    flush_chars=flush_chars,
                    flush_meta=flush_meta,
                    window_after_sealed=sealed_windows is not None,
                    sealed_windows_after=sealed_windows,
                    sealed_signals_after=self._cached_signals_before(),
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
        """Grab every monitor at mouse-down for a possible cross-monitor drop."""
        with self._lock:
            if self._run_dir is None:
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

    def _take_pending_left_event_index_locked(self) -> int:
        """Consume the mouse-down reserved index, or allocate if missing."""
        reserved = self._pending_click_event_index
        self._pending_click_event_index = None
        if reserved is not None:
            return int(reserved)
        index = self._next_index
        self._next_index += 1
        return index

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
            self._pending_click_event_index = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
            self._pending_screenshot = None
            self._pending_windows_before = None
            self._pending_signals_before = None
            self._pending_click_window = None
            self._left_press_click_windows.pop(seq, None)
            self._left_press_uia.pop(seq, None)
            self._left_press_uia_futures.pop(seq, None)
            self._left_press_signals.pop(seq, None)
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
        self._pending_right_signals_before = None
        self._pending_right_click_window = None
        self._right_press_click_windows.pop(seq, None)
        self._right_press_uia.pop(seq, None)
        self._right_press_uia_futures.pop(seq, None)
        self._right_press_signals.pop(seq, None)
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
            click_index = self._take_pending_left_event_index_locked()
            press_seq = self._left_press_seq
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            action_monotonic = self._pending_click_down_at or time.monotonic()
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
            press_windows = self._pending_windows_before
            press_signals = self._pending_signals_before
        self._claim_after_owner(click_index)
        self._flush_pending_text_input(
            shared_end_index=click_index,
            seal_after_to_cache=True,
            sealed_windows=press_windows,
            sealed_signals=press_signals,
            seal_sample_supplied=True,
        )
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
                action_monotonic=action_monotonic,
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
            hold_index = self._take_pending_left_event_index_locked()
            press_seq = self._left_press_seq
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            action_monotonic = self._pending_click_down_at or time.monotonic()
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
            press_windows = self._pending_windows_before
            press_signals = self._pending_signals_before
        self._claim_after_owner(hold_index)
        self._flush_pending_text_input(
            shared_end_index=hold_index,
            seal_after_to_cache=True,
            sealed_windows=press_windows,
            sealed_signals=press_signals,
            seal_sample_supplied=True,
        )
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
                action_monotonic=action_monotonic,
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
            action_monotonic = self._pending_right_down_at or time.monotonic()
            press_windows = self._pending_right_windows_before
            press_signals = self._pending_right_signals_before
            # Keep press_seq + capture dict entry for the emit worker.
            self._clear_pending_right_gesture_locked(discard_capture=False)
        self._claim_after_owner(right_index)
        self._flush_pending_text_input(
            shared_end_index=right_index,
            seal_after_to_cache=True,
            sealed_windows=press_windows,
            sealed_signals=press_signals,
            seal_sample_supplied=True,
        )
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
                action_monotonic=action_monotonic,
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
            drag_index = self._take_pending_left_event_index_locked()
            press_seq = self._left_press_seq
            pending_drag_end = self._pending_drag_end_captures
            self._pending_drag_end_captures = None
            # Claim the slot even when the mouse-down grab is still queued.
            # None means the worker should fill this slot. Stop must not delete
            # those files out from under the emit job.
            if pending_drag_end is not None or press_seq not in self._drag_end_captures:
                self._drag_end_captures[press_seq] = pending_drag_end
            timestamp_utc = self._pending_click_timestamp_utc or utc_now_iso()
            modifiers = self._pending_click_modifiers
            action_monotonic = self._pending_click_down_at or time.monotonic()
            self._pending_click_timer = None
            self._pending_click_coords = None
            self._pending_click_down_at = None
            self._pending_click_timestamp_utc = None
            self._pending_click_modifiers = None
            self._left_button_down = False
            self._left_press_dragging = False
            self._last_move_xy = None
            press_windows = self._pending_windows_before
            press_signals = self._pending_signals_before
        self._claim_after_owner(drag_index)
        end_click_window = self._click_window_payload_at(x2, y2)
        self._flush_pending_text_input(
            shared_end_index=drag_index,
            seal_after_to_cache=True,
            sealed_windows=press_windows,
            sealed_signals=press_signals,
            seal_sample_supplied=True,
        )
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
                action_monotonic=action_monotonic,
                end_click_window=end_click_window,
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
        elif job.action == "flush_ime_commit_text":
            self._flush_ime_commit_with_sealed_sample()
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
                with self._lock:
                    self._start_windows = self._last_settle_windows
                    self._start_signals = (
                        dict(self._last_settle_signals) if self._last_settle_signals else None
                    )
                # Seed naming tree so event 1 has a before FindAll without press-time walk.
                self._collect_and_publish_settle_uia(
                    self._foreground_hwnd(),
                    after_event_index=0,
                )
            finally:
                self._settle_windows_ready.set()
                done = (job.meta or {}).get("done")
                if isinstance(done, threading.Event):
                    done.set()
        elif job.action == "read_clipboard_paste":
            self._worker_read_clipboard_paste(job)
        # settle_probe_tick runs on its own timer thread (not this queue).
        # Window settle and UI Automation run on a per-step thread started
        # after the event is saved, so they cannot delay the next probe.

    def _cancel_settle_probe(
        self,
        *,
        cleanup_files: bool = True,
        reason: str | None = None,
    ) -> None:
        """Stop the settle probe timer and clear probe state.

        Does not wait for an in-flight timer tick. The tick checks the current
        event index and no-ops before publishing when this slot was replaced.
        """
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
        # Skip cleanup while a tick still holds the lock (it may be writing these
        # paths). Recording stop / a later idle cancel will remove leftovers.
        acquired = self._settle_tick_lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            self._cleanup_settle_probe_temp_files(run_dir, probe)
        finally:
            self._settle_tick_lock.release()

    def _cleanup_settle_probe_temp_files(
        self,
        run_dir: Path,
        probe: _SettleProbeState,
    ) -> None:
        """Delete probe-only temps for ``probe``. Caller holds ``_settle_tick_lock``."""
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

    def _settle_probe_is_current(self, event_index: int) -> bool:
        with self._lock:
            probe = self._settle_probe
            return probe is not None and probe.event_index == event_index

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
        """Advance the sample counter on ``probe`` and return the next delay."""
        delay = settle_probe_interval_s(probe.interval_step)
        probe.interval_step += 1
        return delay

    def _schedule_next_settle_probe_tick(self, event_index: int) -> None:
        """Schedule the next settle sample using the fixed 1s interval."""
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

    def _pointer_cursor_for_capture(self) -> tuple[int, int] | None:
        with self._lock:
            cursor = self._last_pointer_cursor_xy
        if cursor is not None:
            return cursor
        try:
            pos = pyautogui.position()
            return (int(pos.x), int(pos.y))
        except Exception:
            return None

    def _grab_screenshot(
        self,
        cursor: tuple[int, int] | None,
        dest: Path,
        *,
        monitor_index: int | None = None,
    ) -> tuple[int, tuple[int, int]]:
        """Capture one frame for before-shot or settle (separate mss contexts)."""
        if monitor_index is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with mss.mss() as sct:
                mon_idx = resolve_monitor_index(sct, int(monitor_index))
                monitor = sct.monitors[mon_idx]
                shot = sct.grab(monitor)
                from PIL import Image

                img = Image.frombytes("RGB", shot.size, shot.rgb)
                img.save(dest, format="JPEG")
                return int(mon_idx), (int(monitor["left"]), int(monitor["top"]))
        if cursor is not None:
            _, mon_idx, mon_offset = _capture_screenshot_at_point(
                int(cursor[0]), int(cursor[1]), dest
            )
            return int(mon_idx), (int(mon_offset[0]), int(mon_offset[1]))
        capture_all_screens_to_file(dest)
        return 0, (0, 0)

    def _start_before_shot_loop(self) -> None:
        """Grab fresh per-monitor raw frames for the whole recording.

        Round-robins each physical monitor into in-memory RGB. Mouse-down pins
        that monitor's latest finished frame; the worker JPEG-encodes on adopt.
        The loop does not wait for a settle probe, so the next action still
        gets a frame when the previous probe has not sampled yet.
        """
        self._before_shot_stop.clear()
        thread = threading.Thread(
            target=self._before_shot_loop,
            name="screen-recorder-before-shot",
            daemon=True,
        )
        with self._lock:
            self._before_shot_thread = thread
            run_dir = self._run_dir
        thread.start()
        if run_dir is not None:
            self._log(run_dir, "before-shot loop start")

    def _stop_before_shot_loop(self) -> None:
        stop = getattr(self, "_before_shot_stop", None)
        if stop is not None:
            stop.set()
        with self._lock:
            thread = getattr(self, "_before_shot_thread", None)
            self._before_shot_thread = None
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

    def _before_shot_monitor_targets(
        self,
    ) -> tuple[tuple[int, int, int, int, int], ...]:
        """Physical monitors to round-robin (mss index 1..N)."""
        with self._lock:
            monitors = self._pre_click_monitors
        if monitors:
            return monitors
        self._refresh_pre_click_monitors()
        with self._lock:
            return self._pre_click_monitors

    def _next_before_shot_monitor(
        self,
        monitors: tuple[tuple[int, int, int, int, int], ...],
    ) -> int | None:
        if not monitors:
            return None
        with self._lock:
            rr = self._before_shot_rr_index % len(monitors)
            self._before_shot_rr_index = rr + 1
        return int(monitors[rr][0])

    def _before_shot_loop(self) -> None:
        try:
            with mss.mss() as sct:
                while not self._before_shot_stop.is_set():
                    published = self._capture_and_publish_before_shot(sct)
                    if self._before_shot_stop.is_set():
                        return
                    if not published and self._before_shot_stop.wait(0.05):
                        return
        except Exception as exc:
            with self._lock:
                run_dir = self._run_dir
            if run_dir is not None:
                self._log(run_dir, f"before-shot loop failed: {exc}")

    def _grab_raw_before_shot(
        self,
        sct: Any,
        monitor_index: int,
    ) -> _RawBeforeShot:
        """Grab one monitor into an in-memory RGB frame (no JPEG)."""
        mon_idx = resolve_monitor_index(sct, int(monitor_index))
        monitor = sct.monitors[mon_idx]
        shot = sct.grab(monitor)
        return _RawBeforeShot(
            frame_id=self._alloc_before_shot_frame_id(),
            rgb=bytes(shot.rgb),
            size=(int(shot.width), int(shot.height)),
            monitor_index=int(mon_idx),
            monitor_offset=(int(monitor["left"]), int(monitor["top"])),
        )

    def _capture_and_publish_before_shot(self, sct: Any | None = None) -> bool:
        """Grab the next round-robin monitor into the in-memory before-shot slot."""
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None or self._before_shot_stop.is_set():
            return False
        monitors = self._before_shot_monitor_targets()
        mon_target = self._next_before_shot_monitor(monitors)
        try:
            if mon_target is None:
                cursor = self._pointer_cursor_for_capture()
                if cursor is None:
                    return False
                mon_target = self._monitor_index_for_point(int(cursor[0]), int(cursor[1]))
                if mon_target is None:
                    try:
                        mon_target, _, _, _, _ = _monitor_at_point(
                            int(cursor[0]), int(cursor[1])
                        )
                    except Exception:
                        return False
            if sct is None:
                with mss.mss() as owned:
                    frame = self._grab_raw_before_shot(owned, int(mon_target))
            else:
                frame = self._grab_raw_before_shot(sct, int(mon_target))
        except Exception as exc:
            self._log(run_dir, f"before-shot capture failed: {exc}")
            return False
        if self._before_shot_stop.is_set():
            return False
        return self._install_raw_before_shot(frame)

    def _start_baseline_settle_probe(self) -> None:
        """Grab immediately, then every 1s, until the next action replaces this probe.

        Event index 0 is not a recorded step. Recording start uses the
        before-shot loop for pre-click frames; this probe is not started there.
        """
        cursor = self._pointer_cursor_for_capture()
        with self._lock:
            if self._run_dir is None:
                return
            self._settle_probe = _SettleProbeState(
                event_index=0,
                started_monotonic=time.monotonic(),
                cursor_xy=cursor,
                interval_step=0,
            )
            run_dir = self._run_dir
            first_delay = self._consume_settle_probe_delay_s(self._settle_probe)
        self._log(
            run_dir,
            f"settle probe start event=0 "
            f"first_s={first_delay} interval_s={_SETTLE_PROBE_INTERVAL_S} "
            f"max_window_s={_SETTLE_PROBE_MAX_WINDOW_S} "
            f"threshold={DEFAULT_SIMILARITY_THRESHOLD}",
        )
        self._schedule_settle_probe_tick(first_delay, 0)

    def _settle_probe_is_current_or_newer(self, event_index: int) -> bool:
        with self._lock:
            probe = self._settle_probe
            return probe is not None and probe.event_index >= event_index

    def _start_settle_probe(self, event: RecordedEvent) -> None:
        if event.kind not in _SETTLE_PROBE_KINDS:
            return
        if self._settle_probe_is_current_or_newer(event.index):
            return
        self._cancel_settle_probe(
            cleanup_files=True,
            reason=f"replaced_by_event_{event.index}",
        )
        if self._settle_probe_is_current_or_newer(event.index):
            return
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
            f"first_s={first_delay} interval_s={_SETTLE_PROBE_INTERVAL_S} "
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

    def _event_json_lock(self) -> threading.Lock:
        """Lock for event JSON. Sessions built without ``__init__`` get one on first use."""
        lock = getattr(self, "_event_file_lock", None)
        if lock is not None:
            return lock
        lock = threading.Lock()
        self._event_file_lock = lock
        return lock

    def _persist_observed_settle(self, event_index: int, observed_settle_seconds: float) -> None:
        """Record the first similar-plateau age. Later plateaus leave the value alone."""
        with self._lock:
            run_dir = self._run_dir
            events = self._events
        if run_dir is None:
            return
        path = event_json_path(run_dir, event_index)
        with self._event_json_lock():
            raw = read_json(path, None)
            if not isinstance(raw, dict):
                return
            existing = raw.get("observed_settle_seconds")
            if isinstance(existing, (int, float)) and not isinstance(existing, bool):
                return
            seconds = round(float(observed_settle_seconds), 3)
            raw["observed_settle_seconds"] = seconds
            write_event_json(run_dir, raw)
        with self._lock:
            for index, event in enumerate(events):
                if event.index == event_index:
                    events[index] = RecordedEvent.from_dict(raw)
                    break
        self._log(
            run_dir,
            f"settle probe first plateau event={event_index} "
            f"observed_settle_seconds={seconds}",
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

            sample_age = time.monotonic() - started
            if sample_age > _SETTLE_PROBE_MAX_WINDOW_S:
                hit_max_window = True
            else:
                self._settle_probe_tick_body(
                    event_index=event_index,
                    run_dir=run_dir,
                    kept_path=kept_path,
                    kept_age=kept_age,
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
        kept_path: Path | None,
        kept_age: float | None,
        sample_age: float,
    ) -> None:
        """Inner settle tick; caller holds ``_settle_tick_lock``.

        If this event was replaced while capturing, drop the sample. This tick
        measures settle only and does not publish a before-shot.

        Always grabs the full virtual desktop (all monitors). A click on one
        monitor must still see a window that opens or closes on another.
        """
        staging = settle_probe_staging_path(run_dir, event_index)
        mon_idx = 0
        mon_offset: tuple[int, int] = (0, 0)
        try:
            mon_idx, mon_offset = self._grab_screenshot(None, staging)
        except Exception as exc:
            self._log(run_dir, f"settle probe capture failed event={event_index}: {exc}")
            self._schedule_next_settle_probe_tick(event_index)
            return

        # Replaced while capturing: drop this sample (do not publish or reschedule).
        if not self._settle_probe_is_current(event_index):
            try:
                if staging.is_file():
                    staging.unlink()
            except OSError:
                pass
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

        if not self._settle_probe_is_current(event_index):
            try:
                if staging.is_file():
                    staging.unlink()
            except OSError:
                pass
            return

        new_kept, new_age, observed = apply_settle_sample(
            kept_path=kept_path,
            kept_age_s=kept_age,
            staging_path=staging,
            sample_age_s=sample_age,
            similar=similar,
        )

        # First similar pair records observed_settle. Later pairs leave that
        # value alone. The probe keeps sampling until the next event replaces
        # it. Before-shots are published by the before-shot loop.
        if observed is not None:
            record_observed = False
            with self._lock:
                if (
                    self._settle_probe is not None
                    and self._settle_probe.event_index == event_index
                    and not self._settle_probe.observed_recorded
                ):
                    self._settle_probe.observed_recorded = True
                    record_observed = True
            if record_observed:
                self._persist_observed_settle(event_index, observed)

        if not self._settle_probe_is_current(event_index):
            return

        official = settle_probe_kept_path(run_dir, event_index)
        if new_kept is not None:
            official = self._promote_settle_staging(Path(new_kept), official)
        # Settle samples stay on the probe. The before-shot loop is the only
        # writer of raw ``_last_settle_frames``.
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

    def _foreground_hwnd(self) -> int:
        if os.name != "nt":
            return 0
        try:
            return int(ctypes.windll.user32.GetForegroundWindow() or 0)
        except Exception:
            return 0

    def _start_uia_executor(self) -> None:
        """One worker for press-time fallback FindAll only (cache miss / hwnd mismatch)."""
        previous = self._uia_executor
        self._uia_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="recording-uia-fallback",
        )
        if previous is not None:
            previous.shutdown(wait=False, cancel_futures=True)

    def _shutdown_uia_executor(self) -> None:
        with self._lock:
            executor = self._uia_executor
            self._uia_executor = None
            pending = list(self._uia_inflight)
        if pending:
            wait(pending, timeout=15)
        if executor is None:
            return
        executor.shutdown(wait=False, cancel_futures=True)

    def _begin_press_uia_fallback(
        self,
        press_seq: int,
        click_window: ClickWindowInfo | None,
        *,
        side: str,
    ) -> None:
        """No-op: press-time live FindAll was removed (post-click tree race).

        Kept as a stub so older tests that patch this name still import cleanly.
        """
        _ = (press_seq, click_window, side)

    def _uia_snapshot_for_window(
        self, click_window: ClickWindowInfo | None
    ) -> dict[str, Any] | None:
        if click_window is None or not int(click_window.hwnd or 0):
            return None
        from src.recorder.uia_input_boxes import snapshot_uia_screen

        hwnd = int(click_window.hwnd)
        try:
            value = snapshot_uia_screen(hwnd)
        except Exception:
            return None
        if not isinstance(value, dict):
            return None
        snapshot = dict(value)
        snapshot["hwnd"] = hwnd
        return snapshot

    def _ready_uia_snapshot(
        self, future: Future[dict[str, Any] | None] | None
    ) -> tuple[dict[str, Any] | None, bool]:
        """Return ``(snapshot, still_running)``."""
        if future is None:
            return None, False
        if not future.done():
            return None, True
        with self._lock:
            self._uia_inflight.discard(future)
        return self._snapshot_from_future(future), False

    def _snapshot_from_future(
        self, future: Future[dict[str, Any] | None]
    ) -> dict[str, Any] | None:
        try:
            value = future.result()
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def _arm_deferred_uia_snapshots(
        self,
        event_index: int,
        pending: list[tuple[str, Future[dict[str, Any] | None]]],
    ) -> None:
        for field, future in pending:
            future.add_done_callback(
                lambda done, idx=event_index, name=field: self._apply_deferred_uia_snapshot(
                    idx, name, done
                )
            )

    def _apply_deferred_uia_snapshot(
        self,
        event_index: int,
        field: str,
        future: Future[dict[str, Any] | None],
    ) -> None:
        snapshot = self._snapshot_from_future(future)
        self._patch_event_uia_field(event_index, field, snapshot)
        with self._lock:
            self._uia_inflight.discard(future)
        self._release_uia_notify_slot(event_index)

    def _release_uia_notify_slot(self, event_index: int) -> None:
        """Decrement pending UIA wait; notify vision when the last slot clears."""
        with self._lock:
            left = self._uia_notify_wait.get(event_index, 0) - 1
            if left <= 0:
                self._uia_notify_wait.pop(event_index, None)
                event = next(
                    (ev for ev in self._events if ev.index == event_index),
                    None,
                )
            else:
                self._uia_notify_wait[event_index] = left
                event = None
        if event is not None and event.kind != "text_input":
            self._notify_event(event)

    def _copy_uia_snapshot(
        self, snapshot: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not isinstance(snapshot, dict):
            return None
        return copy.deepcopy(snapshot)

    def _uia_snapshot_matches_click(
        self,
        snapshot: dict[str, Any] | None,
        click_window: ClickWindowInfo | None,
        *,
        cursor_xy: tuple[int, int] | None = None,
    ) -> bool:
        """True when the settled tree belongs to the window under this click."""
        if not isinstance(snapshot, dict) or click_window is None:
            return False
        click_hwnd = int(click_window.hwnd or 0)
        try:
            cached_hwnd = int(snapshot.get("hwnd") or 0)
        except (TypeError, ValueError):
            cached_hwnd = 0
        # Known hwnds that disagree → wrong tree (YOLO). Do not rescue via rect.
        if click_hwnd > 0 and cached_hwnd > 0:
            return click_hwnd == cached_hwnd
        window = snapshot.get("window")
        if (
            cursor_xy is not None
            and isinstance(window, (list, tuple))
            and len(window) == 4
        ):
            return point_in_rect_xywh(
                int(cursor_xy[0]),
                int(cursor_xy[1]),
                (int(window[0]), int(window[1]), int(window[2]), int(window[3])),
            )
        return False

    def _previous_event_index_for_settle_cache(self) -> int:
        """Event index the next press expects settle UIA to have been published after.

        ``0`` means recording seed (no events yet). Otherwise the last persisted
        event index — settle must be from that event's window-step, not an older
        one still in flight for a later sibling.
        """
        with self._lock:
            if not self._events:
                return 0
            return int(self._events[-1].index)

    def _cached_uia_before(
        self,
        click_window: ClickWindowInfo | None,
        *,
        cursor_xy: tuple[int, int] | None = None,
    ) -> dict[str, Any] | None:
        """Copy last settled FindAll when it matches this click; else None.

        A ``None`` result leaves ``uia_snapshot`` empty so analysis names the
        target from the before-screenshot (YOLO/OCR). Press-time live FindAll
        is not used: it often completes after the click has already changed the
        tree.

        Reuse also requires the settle to have been published after the previous
        event (or seed index 0). A same-hwnd cache from an older step is treated
        as a miss so navigation/scroll between presses does not reuse a stale
        tree.
        """
        previous_index = self._previous_event_index_for_settle_cache()
        with self._lock:
            cached = self._last_settle_uia
            after_index = self._last_settle_uia_after_index
        if after_index != previous_index:
            return None
        if not self._uia_snapshot_matches_click(
            cached, click_window, cursor_xy=cursor_xy
        ):
            return None
        return self._copy_uia_snapshot(cached)

    def _collect_and_publish_settle_uia(
        self,
        hwnd: int,
        *,
        after_event_index: int,
    ) -> dict[str, Any] | None:
        """FindAll under one hwnd after settle; publish as next press before-cache.

        ``after_event_index`` is the event whose window-step produced this tree
        (``0`` for recording seed). The next press reuses it only when that index
        is still the previous event.
        """
        hwnd = int(hwnd or 0)
        if hwnd <= 0:
            return None
        from src.recorder.uia_input_boxes import snapshot_uia_screen

        try:
            value = snapshot_uia_screen(hwnd)
        except Exception:
            return None
        if not isinstance(value, dict):
            return None
        snapshot = dict(value)
        snapshot["hwnd"] = hwnd
        with self._lock:
            self._last_settle_uia = snapshot
            self._last_settle_uia_after_index = int(after_event_index)
        return self._copy_uia_snapshot(snapshot)

    def _opened_hwnd_from_change(
        self,
        change: dict[str, Any] | None,
        debug: dict[str, Any] | None,
    ) -> int:
        if not isinstance(change, dict):
            return 0
        if str(change.get("action") or "").strip() != "opened":
            return 0
        title = str(change.get("title") or "").strip()
        if not title or not isinstance(debug, dict):
            return 0
        after_raw = debug.get("windows_after")
        if not isinstance(after_raw, list):
            return 0
        for entry in after_raw:
            if not isinstance(entry, dict):
                continue
            entry_title = str(entry.get("title") or "").strip()
            if entry_title != title and title not in entry_title and entry_title not in title:
                continue
            try:
                hwnd = int(entry.get("hwnd") or 0)
            except (TypeError, ValueError):
                hwnd = 0
            if hwnd > 0:
                return hwnd
        return 0

    def _pick_settle_uia_hwnd(
        self,
        item: _QueuedEvent,
        change: dict[str, Any] | None,
        debug: dict[str, Any] | None,
    ) -> int:
        """Prefer foreground after settle; never FindAll from the desktop root."""
        fg = self._foreground_hwnd()
        if fg > 0:
            return fg
        opened = self._opened_hwnd_from_change(change, debug)
        if opened > 0:
            return opened
        if item.kind == "drag" and item.end_click_window is not None:
            end_hwnd = int(item.end_click_window.hwnd or 0)
            if end_hwnd > 0:
                return end_hwnd
        if item.click_window is not None:
            return int(item.click_window.hwnd or 0)
        return 0

    def _patch_event_uia_field(
        self,
        event_index: int,
        field: str,
        snapshot: dict[str, Any] | None,
    ) -> RecordedEvent | None:
        """Write one snapshot onto a persisted event. Returns the patched event."""
        if field not in {"uia_snapshot", "end_uia_snapshot"}:
            return None
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            return None
        patched: RecordedEvent | None = None
        with self._event_json_lock():
            path = event_json_path(run_dir, event_index)
            raw = read_json(path, None)
            if isinstance(raw, dict) and isinstance(snapshot, dict):
                raw[field] = snapshot
                write_event_json(run_dir, raw)
            with self._lock:
                events = self._events
                for index, event in enumerate(events):
                    if event.index != event_index:
                        continue
                    if isinstance(raw, dict):
                        event = RecordedEvent.from_dict(raw)
                        events[index] = event
                    patched = event
                    break
        return patched

    def _release_drag_uia_notify(self, item: _QueuedEvent) -> None:
        """Clear one notify slot after drag window-step finished end_uia (or skipped)."""
        if item.kind != "drag":
            return
        self._release_uia_notify_slot(item.event_index)

    def _worker_pending_left_press(self, job: _DeferredCaptureJob) -> None:
        if job.press_seq is None or job.cursor_xy is None:
            self._unpin_settle_frame(job.settle_frame)
            return
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            self._unpin_settle_frame(job.settle_frame)
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

        pending_dest = _pending_capture_path(run_dir, job.press_seq)
        try:
            info = self._pending_screenshot_from_settle_or_capture(
                run_dir,
                int(job.cursor_xy[0]),
                int(job.cursor_xy[1]),
                pending_dest,
                job.settle_frame,
                settle_latch_note=job.settle_latch_note,
                press_seq=job.press_seq,
            )
        except Exception as exc:
            self._log_before_shot(
                run_dir,
                f"press_seq={job.press_seq} adopt_exception "
                f"{type(exc).__name__}: {exc}",
            )
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
            self._unpin_settle_frame(job.settle_frame)
            return
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            self._unpin_settle_frame(job.settle_frame)
            return
        pending_dest = _pending_right_capture_path(run_dir, job.press_seq)
        try:
            info = self._pending_screenshot_from_settle_or_capture(
                run_dir,
                int(job.cursor_xy[0]),
                int(job.cursor_xy[1]),
                pending_dest,
                job.settle_frame,
                settle_latch_note=job.settle_latch_note,
                press_seq=job.press_seq,
            )
        except Exception as exc:
            self._log_before_shot(
                run_dir,
                f"press_seq={job.press_seq} adopt_exception "
                f"{type(exc).__name__}: {exc}",
            )
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
            claimed = self._drag_end_captures.get(job.press_seq, _DRAG_END_UNCLAIMED)
            gesture_gone = (
                job.press_seq not in self._left_press_captures
                and job.press_seq != self._left_press_seq
                and claimed is _DRAG_END_UNCLAIMED
            )
            if gesture_gone:
                # Gesture already cleared without emit; drop files.
                _discard_pending_drag_end_capture_files(captures)
                return
            if claimed is _DRAG_END_UNCLAIMED:
                self._drag_end_captures[job.press_seq] = captures
                if self._left_press_seq == job.press_seq:
                    self._pending_drag_end_captures = captures
                return
            if claimed is None:
                # Mouse-up already reserved this slot. Keep the files off the
                # pending pointer so stop() cannot unlink them before emit.
                self._drag_end_captures[job.press_seq] = captures
                return
            _discard_pending_drag_end_capture_files(captures)

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
            uia_snapshot = (
                self._left_press_uia.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            uia_future = (
                self._left_press_uia_futures.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            signals_before = (
                self._left_press_signals.pop(press_seq, None)
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
                if signals_before is None:
                    signals_before = self._pending_signals_before
                self._pending_screenshot = None
                self._pending_windows_before = None
                self._pending_signals_before = None
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
        if not pending_windows:
            pending_windows = self._start_windows_before()
        if not signals_before:
            signals_before = self._start_signals_before()
        self._log_before_shot_finalize(
            run_dir,
            event_index=job.event_index,
            press_seq=press_seq,
            pending_shot=pending_shot,
            side="left",
        )
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
            release_mon = _release_monitor_index(job.end_xy)
            if (
                release_mon is not None
                and int(release_mon) == int(mon_idx)
                and shot_path
                and Path(shot_path).is_file()
            ):
                # Drop stayed on the press monitor: both anchors share that frame.
                end_shot_path, end_mon_idx, end_mon_offset = (
                    _copy_mouse_down_screenshot_as_drag_end(
                        run_dir,
                        job.event_index,
                        shot_path,
                        mon_idx,
                        mon_offset,
                    )
                )
                self._discard_drag_end_capture_entry(drag_end)
            else:
                end_shot_path, end_mon_idx, end_mon_offset = (
                    _finalize_drag_end_screenshot(
                        run_dir,
                        job.event_index,
                        job.end_xy,
                        drag_end,
                        fallback_mon_idx=mon_idx,
                        fallback_mon_offset=mon_offset,
                    )
                )
        else:
            self._discard_drag_end_capture_entry(drag_end)

        pending_uia: list[tuple[str, Future[dict[str, Any] | None]]] = []
        if uia_snapshot is None and uia_future is not None:
            ready, still_running = self._ready_uia_snapshot(uia_future)
            if ready is not None:
                uia_snapshot = ready
            elif still_running:
                pending_uia.append(("uia_snapshot", uia_future))
        wait_slots = len(pending_uia)
        # Drag drop naming uses end_uia from this step's settled FindAll.
        if job.kind == "drag":
            wait_slots += 1
        if wait_slots:
            with self._lock:
                self._uia_notify_wait[int(job.event_index)] = wait_slots

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
                end_click_window=job.end_click_window,
                uia_snapshot=uia_snapshot,
                end_uia_snapshot=None,
                signals_before=signals_before,
                action_monotonic=job.action_monotonic,
            )
        )
        self._arm_deferred_uia_snapshots(int(job.event_index), pending_uia)
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
            uia_snapshot = (
                self._right_press_uia.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            uia_future = (
                self._right_press_uia_futures.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            signals_before = (
                self._right_press_signals.pop(press_seq, None)
                if press_seq is not None
                else None
            )
            if entry is None and press_seq is not None and self._right_press_seq == press_seq:
                entry = (self._pending_right_screenshot, self._pending_right_windows_before)
                click_window = self._pending_right_click_window
                if signals_before is None:
                    signals_before = self._pending_right_signals_before
                self._pending_right_screenshot = None
                self._pending_right_windows_before = None
                self._pending_right_signals_before = None
                self._pending_right_click_window = None
        if run_dir is None:
            self._discard_press_capture_entry(entry)
            return
        pending_shot = entry[0] if entry is not None else None
        pending_windows = entry[1] if entry is not None else None
        if not pending_windows:
            pending_windows = self._start_windows_before()
        if not signals_before:
            signals_before = self._start_signals_before()
        self._log_before_shot_finalize(
            run_dir,
            event_index=job.event_index,
            press_seq=press_seq,
            pending_shot=pending_shot,
            side="right",
        )
        shot_path, mon_idx, mon_offset = _finalize_screenshot(
            run_dir,
            job.event_index,
            job.cursor_xy,
            pending_shot,
        )
        pending_uia: list[tuple[str, Future[dict[str, Any] | None]]] = []
        if uia_snapshot is None and uia_future is not None:
            ready, still_running = self._ready_uia_snapshot(uia_future)
            if ready is not None:
                uia_snapshot = ready
            elif still_running:
                pending_uia.append(("uia_snapshot", uia_future))
                with self._lock:
                    self._uia_notify_wait[int(job.event_index)] = len(pending_uia)
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
                uia_snapshot=uia_snapshot,
                signals_before=signals_before,
                action_monotonic=job.action_monotonic,
            )
        )
        self._arm_deferred_uia_snapshots(int(job.event_index), pending_uia)
        if job.refresh_pre_type:
            self._refresh_pending_pre_type(job.cursor_xy)

    def _worker_mouse_pointer_event(self, job: _DeferredCaptureJob) -> None:
        if job.event_index is None or job.kind is None or job.cursor_xy is None:
            self._unpin_settle_frame(job.settle_frame)
            return
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            self._unpin_settle_frame(job.settle_frame)
            return
        dest = screenshot_path_for_event(run_dir, job.event_index)
        try:
            shot_path, mon_idx, mon_offset = self._pending_screenshot_from_settle_or_capture(
                run_dir,
                int(job.cursor_xy[0]),
                int(job.cursor_xy[1]),
                dest,
                job.settle_frame,
                settle_latch_note=job.settle_latch_note,
                press_seq=job.press_seq,
            )
        except Exception as exc:
            self._log_before_shot(
                run_dir,
                f"event={job.event_index} adopt_exception "
                f"{type(exc).__name__}: {exc} path=live_capture",
            )
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
                windows_before=job.windows_before or self._start_windows_before(),
                click_window=job.click_window,
                signals_before=job.signals_before or self._start_signals_before(),
                action_monotonic=job.action_monotonic,
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

        ime_raw = meta.get("ime")
        ime_payload = ime_raw if isinstance(ime_raw, dict) else None
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
                windows_before=meta.get("windows_before") or self._start_windows_before(),
                signals_before=meta.get("signals_before") or self._start_signals_before(),
                action_monotonic=job.action_monotonic,
                window_after_sealed=job.window_after_sealed,
                windows_after=job.sealed_windows_after,
                signals_after=job.sealed_signals_after,
                ime=ime_payload,
            )
        )
        # So a later gesture can stamp its before-cache as this typing after
        # (or refine a sealed IME-commit sample) via ``_seal_open_after_cache``.
        self._claim_after_owner(index)

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
                    window_after_sealed=job.window_after_sealed,
                    sealed_windows_after=job.sealed_windows_after,
                    sealed_signals_after=job.sealed_signals_after,
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
                windows_before=job.windows_before or self._start_windows_before(),
                signals_before=job.signals_before or self._start_signals_before(),
                action_monotonic=job.action_monotonic,
            )
        )
        if job.refresh_pre_type:
            self._refresh_pending_pre_type(job.cursor_xy)

    def _sleep_remaining_window_settle(self, item: _QueuedEvent) -> None:
        """Sleep only the settle time still left since the gesture."""
        delay = settle_delay_for_click(item.cursor_xy, item.windows_before)
        started = item.action_monotonic
        if started is None:
            time.sleep(delay)
            return
        remaining = delay - (time.monotonic() - float(started))
        if remaining > 0:
            time.sleep(remaining)

    def _uia_reader_or_timeout(self) -> tuple[Callable[..., dict[str, Any]], dict[str, bool]]:
        """Run UI Automation on a side thread and report when it exceeds the timeout."""
        timed_out = {"value": False}

        def reader(kind: str | None, cursor_xy: tuple[int, int] | None) -> dict[str, Any]:
            done = threading.Event()
            box: dict[str, Any] = {}

            def _run() -> None:
                try:
                    value = read_uia_for_kind(kind, cursor_xy)
                except Exception:
                    value = {}
                box["value"] = value if isinstance(value, dict) else {}
                done.set()

            thread = threading.Thread(target=_run, name="window-step-uia", daemon=True)
            thread.start()
            if not done.wait(_WINDOW_UIA_TIMEOUT_S):
                timed_out["value"] = True
                return {}
            value = box.get("value", {})
            return value if isinstance(value, dict) else {}

        return reader, timed_out

    def _resolve_window_change(
        self,
        item: _QueuedEvent,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
        if not item.windows_before and not item.signals_before:
            # Still keep a sealed after-sample (e.g. delayed IME Enter commit).
            if item.window_after_sealed:
                debug = _window_resolve_failure_debug("missing_before_cache")
                debug["signals_before"] = {}
                debug["signals_after"] = dict(item.signals_after or {})
                debug["signal_kind"] = item.kind
                return None, None, debug
            return None, None, _window_resolve_failure_debug("missing_before_cache")
        signals_before = dict(item.signals_before or {})
        point_before = identity_from_click_window(item.click_window)
        if point_before is not None:
            signals_before["point_window"] = point_before
        if item.window_after_sealed:
            # The next gesture already happened. Use the cache copied before it
            # instead of enumerating windows now.
            windows_after = list(item.windows_after or [])
            signals_after = dict(item.signals_after or {})
        else:
            self._sleep_remaining_window_settle(item)
            windows_after = []
            if item.windows_before:
                try:
                    windows_after = snapshot_top_level_windows()
                except Exception as exc:
                    return None, None, _window_resolve_failure_debug(
                        "snapshot_error",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
            uia_reader, uia_timed_out = self._uia_reader_or_timeout()
            try:
                signals_after = capture_step_signals(
                    cursor_xy=item.cursor_xy,
                    kind=item.kind,
                    uia_reader=uia_reader,
                )
            except Exception:
                signals_after = {}
            if uia_timed_out["value"]:
                return None, None, _window_resolve_failure_debug("uia_timeout")
        change: dict[str, Any] | None = None
        title: str | None = None
        debug: dict[str, Any] = {}
        if item.windows_before:
            result = diff_snapshots_with_debug(
                list(item.windows_before),
                windows_after,
                click_xy=item.cursor_xy,
            )
            debug = dict(result.debug)
            if result.change is not None:
                change = result.change.to_dict()
                title = result.change.title or None
        debug["signals_before"] = signals_before
        debug["signals_after"] = signals_after
        debug["signal_kind"] = item.kind
        return change, title, debug

    def _persist_queued_event(self, item: _QueuedEvent) -> None:
        with self._lock:
            if self._run_dir is None:
                return
            run_dir = self._run_dir

        click_window_payload: dict[str, Any] | None = None
        if item.click_window is not None:
            offset = item.monitor_offset if item.monitor_offset is not None else (0, 0)
            click_window_payload = item.click_window.to_local_payload(offset)
        end_click_window_payload: dict[str, Any] | None = None
        if item.end_click_window is not None:
            end_offset = (
                item.end_monitor_offset if item.end_monitor_offset is not None else (0, 0)
            )
            end_click_window_payload = item.end_click_window.to_local_payload(end_offset)

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
            click_window=click_window_payload,
            end_click_window=end_click_window_payload,
            uia_snapshot=item.uia_snapshot if isinstance(item.uia_snapshot, dict) else None,
            end_uia_snapshot=(
                item.end_uia_snapshot if isinstance(item.end_uia_snapshot, dict) else None
            ),
            ime=item.ime if isinstance(item.ime, dict) else None,
        )
        with self._event_json_lock():
            write_event_json(run_dir, event.to_dict())
        with self._lock:
            if self._run_dir is None:
                return
            self._events.append(event)
            if event.kind == "text_input":
                # Resolve must see signals_after.focused.value; notify after patch.
                self._pending_text_notify.add(event.index)
            uia_waiting = self._uia_notify_wait.get(event.index, 0)
        if event.kind != "text_input" and uia_waiting == 0:
            self._notify_event(event)
        if event.kind in _SETTLE_PROBE_KINDS:
            self._start_settle_probe(event)
        else:
            self._cancel_settle_probe(
                cleanup_files=True,
                reason=f"non_probe_event_{event.kind}_{event.index}",
            )
        self._start_window_step(item)

    def _start_window_step(self, item: _QueuedEvent) -> None:
        """Diff windows and sample debug focus_rect off the fast lane."""
        thread = threading.Thread(
            target=self._window_step_body,
            args=(item,),
            name=f"window-step-{item.event_index}",
            daemon=True,
        )
        with self._lock:
            self._window_step_threads = [
                running for running in self._window_step_threads if running.is_alive()
            ]
            self._window_step_threads.append(thread)
        thread.start()

    def _sample_debug_focus_rect_for_item(
        self,
        item: _QueuedEvent,
    ) -> tuple[int, int, int, int] | None:
        """Focused-element bounds after settle, for screenshot overlays only."""
        return sample_debug_focus_rect(
            last_click_xy=item.cursor_xy,
            mouse_xy=item.cursor_xy,
        )

    def _window_step_body(self, item: _QueuedEvent) -> None:
        change: dict[str, Any] | None = None
        title: str | None = None
        debug: dict[str, Any] | None = None
        focus_rect: tuple[int, int, int, int] | None = None
        need_window = bool(
            item.windows_before or item.signals_before or item.window_after_sealed
        )
        if need_window:
            change, title, debug = self._resolve_window_change(item)
            # Sealed after-cache means the next gesture already ran; sampling
            # focus now would describe that later gesture, not this one.
            if not item.window_after_sealed:
                focus_rect = self._sample_debug_focus_rect_for_item(item)
        else:
            self._sleep_remaining_window_settle(item)
            focus_rect = self._sample_debug_focus_rect_for_item(item)
            # Same outcome as ``_resolve_window_change`` with an empty before-cache.
            debug = _window_resolve_failure_debug("missing_before_cache")
        if not item.window_after_sealed:
            # One settled FindAll for the next press (and drag end_uia on this event).
            settle_hwnd = self._pick_settle_uia_hwnd(item, change, debug)
            settle_snap = self._collect_and_publish_settle_uia(
                settle_hwnd,
                after_event_index=int(item.event_index),
            )
            if item.kind == "drag" and settle_snap is not None:
                self._patch_event_uia_field(
                    item.event_index, "end_uia_snapshot", settle_snap
                )
        if change is None and title is None and debug is None and focus_rect is None:
            # Still release deferred text_input / drag vision notify.
            self._notify_pending_text_input(item.event_index)
            self._release_drag_uia_notify(item)
            return
        self._patch_window_change(
            item.event_index,
            change,
            title,
            debug,
            focus_rect=focus_rect,
        )
        self._release_drag_uia_notify(item)

    def _patch_window_change(
        self,
        event_index: int,
        change: dict[str, Any] | None,
        title: str | None,
        debug: dict[str, Any] | None,
        *,
        focus_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            self._notify_pending_text_input(event_index)
            return
        path = event_json_path(run_dir, event_index)
        with self._event_json_lock():
            with self._lock:
                seal = self._after_seals.pop(event_index, None)
            if seal is not None and isinstance(debug, dict):
                _apply_next_before_to_debug(debug, seal[0], seal[1])
            raw = read_json(path, None)
            if not isinstance(raw, dict):
                if seal is not None:
                    with self._lock:
                        self._after_seals.setdefault(event_index, seal)
            else:
                if change is not None or title is not None or debug is not None:
                    raw["window_change"] = change
                    raw["target_window_title"] = title
                    raw["window_snapshot_debug"] = debug
                # Keep typing-time focus_rect when already set; fill clicks/keys here.
                if focus_rect is not None and raw.get("focus_rect") is None:
                    raw["focus_rect"] = [
                        int(focus_rect[0]),
                        int(focus_rect[1]),
                        int(focus_rect[2]),
                        int(focus_rect[3]),
                    ]
                write_event_json(run_dir, raw)
                with self._lock:
                    events = self._events
                    for index, event in enumerate(events):
                        if event.index == event_index:
                            events[index] = RecordedEvent.from_dict(raw)
                            break
        # Outside the JSON lock so vision prefetch cannot deadlock on event I/O.
        self._notify_pending_text_input(event_index)

    def _join_window_step_threads(self) -> None:
        with self._lock:
            threads = list(self._window_step_threads)
        deadline = time.monotonic() + _WINDOW_STEP_JOIN_TIMEOUT_S
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)

    def _emit_pending_click(self, x: int, y: int, button: str) -> None:
        self._flush_pending_click(x, y, button)

    def _reconcile_sticky_left_press_from_os(self) -> bool:
        """Close a tracked left press when the OS already reports the button up.

        Missed pynput mouse-ups leave ``_left_button_down`` stuck. Recover by
        synthesizing release at the press point. Returns True when a press was
        closed.
        """
        with self._lock:
            if not self._left_button_down or self._pending_click_coords is None:
                return False
            sx, sy, _btn = self._pending_click_coords
        os_down = _os_mouse_button_down(_VK_LBUTTON)
        if os_down is not False:
            return False
        self._on_left_mouse_up(sx, sy, synthesized=True)
        return True

    def _on_mouse_move(self, x: int, y: int) -> None:
        ix, iy = int(x), int(y)
        with self._lock:
            self._last_move_xy = (ix, iy)
            self._last_pointer_cursor_xy = (ix, iy)
            if not self._left_button_down or self._pending_click_coords is None:
                return
            sx, sy, _btn = self._pending_click_coords
            dragging = self._left_press_dragging
        # Missed mouse-up: OS says released but we still track a press. Synthesize
        # up at the press point so a later far move cannot become a false drag
        # (toolbar click → flyout pick).
        if self._reconcile_sticky_left_press_from_os():
            return
        if dragging:
            return
        if abs(sx - ix) > _DRAG_THRESHOLD_PX or abs(sy - iy) > _DRAG_THRESHOLD_PX:
            with self._lock:
                self._left_press_dragging = True
            self._cancel_pending_click_timer()

    def _on_left_mouse_down(
        self,
        ix: int,
        iy: int,
        btn: str,
        *,
        timestamp_utc: str | None = None,
    ) -> None:
        # A click will flush pending text after the double-click window; cancel
        # a delayed Chinese IME Enter flush so it cannot race and seal first.
        self._cancel_ime_commit_timer()
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
                with self._lock:
                    # Reuse the first-press reserved index for the double-click.
                    double_index = self._take_pending_left_event_index_locked()
                    self._pending_click_timer = None
                self._clear_pending_left_gesture()
                self._queue_pointer_event_immediate(
                    kind="double_click",
                    cursor_xy=(ix, iy),
                    button=btn,
                    modifiers=self._snapshot_pressed_modifiers(),
                    timestamp_utc=action_timestamp_utc,
                    event_index=double_index,
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
            self._pending_click_event_index = self._next_index
            self._next_index += 1
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

    def _on_left_mouse_up(
        self,
        ix: int,
        iy: int,
        *,
        synthesized: bool = False,
    ) -> None:
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
        # OS/key recovery for a missed up: wall-clock since down is "how long we
        # were stuck", not how long the user held — never promote to hold.
        # Flush immediately (no double-click deferral): a recovered press is not
        # awaiting a second click, and deferral can race with stop into a hold.
        if synthesized:
            self._flush_pending_click(sx, sy, button)
            return
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
        self._cancel_ime_commit_timer()
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
        btn = _normalize_button(button)
        ix, iy = int(x), int(y)
        # Never drop a release while we already track that button as down —
        # ignore_rect must not swallow the matching up (sticky press → false drag).
        if not pressed:
            if btn == "left":
                with self._lock:
                    tracking_left = self._left_button_down
                if tracking_left:
                    self._on_left_mouse_up(ix, iy)
                    return
            elif btn == "right":
                with self._lock:
                    tracking_right = self._pending_right_coords is not None
                if tracking_right:
                    self._on_right_mouse_up(ix, iy)
                    return
        if self._should_ignore_mouse_point(ix, iy):
            return
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

    def _on_win32_key_filter(self, msg: int, data: Any) -> None:
        """Stash the hardware scan code for this key-down. Must stay cheap.

        pynput calls this on the low-level hook thread and then drops
        ``scanCode`` and ``flags``. Returning ``None`` still delivers the key.
        """
        if int(msg) not in (_WM_KEYDOWN, _WM_SYSKEYDOWN):
            return None
        try:
            sample = _HardwareKeySample(
                vk=int(data.vkCode),
                scan=int(data.scanCode),
                extended=bool(int(data.flags) & _LLKHF_EXTENDED),
            )
        except Exception:
            return None
        with self._lock:
            self._hardware_key_presses.append(sample)
        return None

    def _pop_hardware_key_sample_locked(self) -> _HardwareKeySample | None:
        if not self._hardware_key_presses:
            return None
        return self._hardware_key_presses.popleft()

    def _on_key_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        timestamp_utc = utc_now_iso()
        with self._lock:
            suppress = self._suppress_hotkey_keys
            active = self._accepting_input
            # Pop even when this press is ignored so the queue stays aligned
            # with key-down callbacks.
            hardware = self._pop_hardware_key_sample_locked()
        if not active:
            return
        # Close a sticky left press before typing/hotkeys so a missed mouse-up
        # cannot invert gesture vs text order or inflate into a false hold.
        self._reconcile_sticky_left_press_from_os()
        # Sample before applying the key so a prior 中→英 switch (e.g. Shift
        # release, or language-bar click) clears composition Latin keys first.
        ime = self._sync_ime_state()
        if suppress and key in _HOTKEY_SUPPRESS_KEYS:
            mod = _modifier_name(key)
            if mod:
                with self._lock:
                    self._pressed_modifiers.add(mod)
            self._log_key_press(
                action="suppress", token=_key_token(key), key=key, ime=ime
            )
            return

        mod = _modifier_name(key)
        if mod:
            with self._lock:
                self._pressed_modifiers.add(mod)
            self._log_key_press(action="mod_down", token=mod, key=key, ime=ime)
            return

        try:
            pos = pyautogui.position()
            cursor_xy = (int(pos.x), int(pos.y))
        except Exception:
            cursor_xy = None

        with self._lock:
            mods = sorted(self._pressed_modifiers)

        hardware_char = _numpad_char_from_hardware(hardware)
        token = hardware_char or _key_token(key)
        if token is None:
            self._log_key_press(
                action="ignored_no_token", mods=mods, key=key, ime=ime
            )
            return

        typed = hardware_char or _key_char(key)
        # Shift alone only changes case/symbols — treat printable keys as typing,
        # not hotkeys. Ctrl/Alt/Win (+ optional Shift) remain hotkeys.
        if mods == ["shift"]:
            if typed and typed.isprintable():
                self._append_text_input_char(typed, cursor_xy, timestamp_utc=timestamp_utc)
                self._log_key_press(
                    action="type",
                    token=token,
                    typed=typed,
                    mods=mods,
                    key=key,
                    ime=ime,
                )
                return
            if key == keyboard.Key.space:
                self._append_text_input_char(" ", cursor_xy, timestamp_utc=timestamp_utc)
                self._log_key_press(
                    action="type",
                    token=token,
                    typed=" ",
                    mods=mods,
                    key=key,
                    ime=ime,
                )
                return

        if mods:
            if is_recording_toggle_hotkey(mods + [token]):
                # Global recording toggle (Ctrl+Shift+R) — never record as a step.
                self._log_key_press(
                    action="recording_toggle",
                    token=token,
                    typed=typed,
                    mods=mods,
                    key=key,
                    ime=ime,
                )
                return
            if _is_paste_hotkey(mods, token):
                # Clipboard owners can block pyperclip.paste for a long time.
                # Read it on the worker, same as screenshots, so this hook returns.
                self._defer_clipboard_paste(
                    cursor_xy,
                    timestamp_utc=timestamp_utc,
                    hotkey_keys=mods + [token],
                )
                self._log_key_press(
                    action="paste",
                    token=token,
                    typed=typed,
                    mods=mods,
                    key=key,
                    ime=ime,
                )
                return
            self._queue_keyboard_event_immediate(
                kind="hotkey",
                cursor_xy=cursor_xy,
                keys=mods + [token],
                timestamp_utc=timestamp_utc,
            )
            self._log_key_press(
                action="hotkey",
                token=token,
                typed=typed,
                mods=mods,
                key=key,
                ime=ime,
            )
            return

        if typed and typed.isprintable():
            self._append_text_input_char(typed, cursor_xy, timestamp_utc=timestamp_utc)
            self._log_key_press(
                action="type",
                token=token,
                typed=typed,
                mods=mods,
                key=key,
                ime=ime,
            )
            return

        if key == keyboard.Key.space:
            self._append_text_input_char(" ", cursor_xy, timestamp_utc=timestamp_utc)
            self._log_key_press(
                action="type",
                token=token,
                typed=" ",
                mods=mods,
                key=key,
                ime=ime,
            )
            return

        if not mods and self._edit_pending_text_input(key):
            self._log_key_press(
                action=f"edit_{token}",
                token=token,
                typed=typed,
                mods=mods,
                key=key,
                ime=ime,
            )
            return

        # Chinese IME: Enter commits the composition / candidate — keep the
        # flushed text_input step, but do not record Enter as its own step.
        if key == keyboard.Key.enter and ime.chinese_mode is True:
            self._flush_text_without_key_step()
            self._log_key_press(
                action="ime_commit_enter",
                token=token,
                typed=typed,
                mods=mods,
                key=key,
                ime=ime,
            )
            return

        if key in _SPECIAL_KEYS or isinstance(key, keyboard.Key):
            self._queue_keyboard_event_immediate(
                kind="key_press",
                cursor_xy=cursor_xy,
                key=token,
                timestamp_utc=timestamp_utc,
            )
            self._log_key_press(
                action="key_press",
                token=token,
                typed=typed,
                mods=mods,
                key=key,
                ime=ime,
            )
            return

        self._log_key_press(
            action="dropped", token=token, typed=typed, mods=mods, key=key, ime=ime
        )

    def _on_key_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        mod = _modifier_name(key)
        if mod:
            with self._lock:
                self._pressed_modifiers.discard(mod)
            # Shift toggles 中/英 on many Chinese IMEs at key-up; wipe here so
            # composition Latin keys do not stick around for the English burst.
            if mod == "shift":
                with self._lock:
                    active = self._accepting_input
                if active:
                    self._sync_ime_state()
