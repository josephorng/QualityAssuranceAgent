"""Edge-reveal stop control: hidden until the cursor hits the top screen border."""

from __future__ import annotations

from typing import Any, Callable

import customtkinter as ctk
import mss
import pyautogui

from src.recorder.hotkey import RECORDING_HOTKEY_DISPLAY

_EDGE_TRIGGER_PX = 4
_HIDE_DELAY_MS = 450
_POLL_MS = 80
_OVERLAY_MARGIN = 16


class RecordingStopOverlay:
    """Topmost stop control that appears only when the mouse touches the top edge."""

    def __init__(
        self,
        master: Any,
        *,
        on_stop: Callable[[], None],
        hotkey_hint: str = RECORDING_HOTKEY_DISPLAY,
    ) -> None:
        self._on_stop = on_stop
        self._master = master
        self._window: ctk.CTkToplevel | None = None
        self._hotkey_hint = hotkey_hint
        self._visible = False
        self._hide_after_id: str | None = None
        self._poll_after_id: str | None = None
        self._destroyed = False
        self._width = 200
        self._height = 100
        self._monitors_cache: list[dict[str, int]] = []
        self._monitors_cache_at = 0.0
        self._build(master)
        self._schedule_poll()

    def _build(self, master: Any) -> None:
        win = ctk.CTkToplevel(master)
        self._window = win
        win.title("錄製中")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        try:
            win.overrideredirect(True)
        except Exception:
            pass
        try:
            win.transient(master.winfo_toplevel())
        except Exception:
            pass

        frame = ctk.CTkFrame(win, corner_radius=10)
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(
            frame,
            text="錄製中",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(
            frame,
            text=f"按 {self._hotkey_hint} 也可停止",
            font=ctk.CTkFont(size=11),
            text_color=("gray30", "gray65"),
        ).pack(anchor="w", pady=(0, 10), padx=14)

        ctk.CTkButton(
            frame,
            text="停止錄製",
            width=160,
            height=34,
            fg_color=("#c0392b", "#a93226"),
            hover_color=("#a93226", "#922b21"),
            command=self._handle_stop,
        ).pack(padx=14, pady=(0, 14))

        win.update_idletasks()
        self._width = max(int(win.winfo_reqwidth()), 200)
        self._height = max(int(win.winfo_reqheight()), 100)
        # Start hidden; reveal only when the cursor hits the top border.
        try:
            win.withdraw()
        except Exception:
            pass

    def _schedule_poll(self) -> None:
        if self._destroyed or self._window is None:
            return
        try:
            self._poll_after_id = self._master.after(_POLL_MS, self._poll_cursor)
        except Exception:
            self._poll_after_id = None

    def _cancel_poll(self) -> None:
        after_id = self._poll_after_id
        self._poll_after_id = None
        if after_id is None:
            return
        try:
            self._master.after_cancel(after_id)
        except Exception:
            pass

    def _cancel_hide(self) -> None:
        after_id = self._hide_after_id
        self._hide_after_id = None
        if after_id is None:
            return
        try:
            self._master.after_cancel(after_id)
        except Exception:
            pass

    def _poll_cursor(self) -> None:
        self._poll_after_id = None
        if self._destroyed or self._window is None:
            return
        try:
            pos = pyautogui.position()
            x, y = int(pos.x), int(pos.y)
        except Exception:
            self._schedule_poll()
            return

        at_top = self._cursor_at_top_edge(x, y)
        over_overlay = self._cursor_over_overlay(x, y)
        if at_top or over_overlay:
            self._cancel_hide()
            self._reveal_near_cursor(x, y)
        elif self._visible:
            self._schedule_hide()
        self._schedule_poll()

    @staticmethod
    def _read_monitors() -> list[dict[str, int]]:
        try:
            with mss.mss() as sct:
                return [
                    {
                        "left": int(mon["left"]),
                        "top": int(mon["top"]),
                        "width": int(mon["width"]),
                        "height": int(mon["height"]),
                    }
                    for mon in sct.monitors[1:]
                ]
        except Exception:
            return []

    def _monitors(self) -> list[dict[str, int]]:
        import time

        now = time.monotonic()
        if not self._monitors_cache or now - self._monitors_cache_at > 2.0:
            self._monitors_cache = self._read_monitors()
            self._monitors_cache_at = now
        return self._monitors_cache

    def _monitor_for_point(self, x: int, y: int) -> dict[str, int] | None:
        for mon in self._monitors():
            left = int(mon["left"])
            top = int(mon["top"])
            width = int(mon["width"])
            height = int(mon["height"])
            if left <= x < left + width and top <= y < top + height:
                return mon
        monitors = self._monitors()
        return monitors[0] if monitors else None

    def _cursor_at_top_edge(self, x: int, y: int) -> bool:
        mon = self._monitor_for_point(x, y)
        if mon is None:
            return y <= _EDGE_TRIGGER_PX
        return y <= int(mon["top"]) + _EDGE_TRIGGER_PX

    def _cursor_over_overlay(self, x: int, y: int) -> bool:
        if not self._visible:
            return False
        rect = self.ignore_rect()
        if rect is None:
            return False
        left, top, width, height = rect
        return left <= x < left + width and top <= y < top + height

    def _reveal_near_cursor(self, x: int, y: int) -> None:
        win = self._window
        if win is None:
            return
        mon = self._monitor_for_point(x, y)
        if mon is None:
            try:
                mon = {
                    "left": 0,
                    "top": 0,
                    "width": int(win.winfo_screenwidth()),
                    "height": int(win.winfo_screenheight()),
                }
            except Exception:
                mon = {"left": 0, "top": 0, "width": 1280, "height": 720}

        width = self._width
        height = self._height
        # Center horizontally on the monitor whose top edge was touched.
        ox = int(mon["left"]) + max(
            _OVERLAY_MARGIN,
            (int(mon["width"]) - width) // 2,
        )
        oy = int(mon["top"]) + _OVERLAY_MARGIN
        try:
            win.geometry(f"{width}x{height}+{ox}+{oy}")
            win.attributes("-topmost", True)
            if not self._visible:
                win.deiconify()
                win.lift()
            self._visible = True
        except Exception:
            pass

    def _schedule_hide(self) -> None:
        if self._hide_after_id is not None or not self._visible:
            return
        try:
            self._hide_after_id = self._master.after(_HIDE_DELAY_MS, self._hide)
        except Exception:
            self._hide_after_id = None

    def _hide(self) -> None:
        self._hide_after_id = None
        if self._destroyed or not self._visible:
            return
        win = self._window
        if win is None:
            return
        try:
            win.withdraw()
        except Exception:
            pass
        self._visible = False

    def _handle_stop(self) -> None:
        try:
            self._on_stop()
        except Exception:
            pass

    def ignore_rect(self) -> tuple[int, int, int, int] | None:
        """Only ignore clicks while the overlay is actually visible."""
        win = self._window
        if win is None or not self._visible:
            return None
        try:
            if not win.winfo_exists() or not win.winfo_viewable():
                return None
            width = int(win.winfo_width())
            height = int(win.winfo_height())
            if width <= 1 or height <= 1:
                width = self._width
                height = self._height
            if width <= 0 or height <= 0:
                return None
            return (
                int(win.winfo_rootx()),
                int(win.winfo_rooty()),
                width,
                height,
            )
        except Exception:
            return None

    def destroy(self) -> None:
        self._destroyed = True
        self._cancel_poll()
        self._cancel_hide()
        win = self._window
        self._window = None
        self._visible = False
        if win is None:
            return
        try:
            win.destroy()
        except Exception:
            pass
