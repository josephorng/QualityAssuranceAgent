"""Edge-reveal controls: hidden until the cursor hits the top screen border."""

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
_STEP_SNIPPET_LIMIT = 42
_STEP_ERROR_LIMIT = 160


def clip_step_snippet(text: str, *, limit: int = _STEP_SNIPPET_LIMIT) -> str:
    """Collapse whitespace and shorten a step instruction for the overlay label."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 1:
        return "…"
    return cleaned[: limit - 1] + "…"


class _EdgeRevealOverlay:
    """Topmost window that appears only when the mouse touches the top edge."""

    def __init__(self, master: Any) -> None:
        self._master = master
        self._window: ctk.CTkToplevel | None = None
        self._visible = False
        self._hide_after_id: str | None = None
        self._poll_after_id: str | None = None
        self._destroyed = False
        self._pinned = False
        self._width = 200
        self._height = 100
        self._monitors_cache: list[dict[str, int]] = []
        self._monitors_cache_at = 0.0
        self._build(master)
        self._measure_and_hide()
        self._schedule_poll()

    def _build(self, master: Any) -> None:
        raise NotImplementedError

    def _min_size(self) -> tuple[int, int]:
        return (200, 100)

    def _measure_and_hide(self) -> None:
        win = self._window
        if win is None:
            return
        try:
            win.update_idletasks()
            min_width, min_height = self._min_size()
            self._width = max(int(win.winfo_reqwidth()), min_width)
            self._height = max(int(win.winfo_reqheight()), min_height)
        except Exception:
            pass
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

        if self._pinned:
            self._cancel_hide()
            if not self._visible:
                self._reveal_near_cursor(x, y)
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

    def reveal(self) -> None:
        """Show the overlay on the monitor under the cursor."""
        if self._destroyed or self._window is None:
            return
        self._cancel_hide()
        try:
            pos = pyautogui.position()
            x, y = int(pos.x), int(pos.y)
        except Exception:
            x, y = 0, 0
        self._reveal_near_cursor(x, y)

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


class RecordingStopOverlay(_EdgeRevealOverlay):
    """Topmost stop control that appears only when the mouse touches the top edge."""

    def __init__(
        self,
        master: Any,
        *,
        on_stop: Callable[[], None],
        hotkey_hint: str = RECORDING_HOTKEY_DISPLAY,
    ) -> None:
        self._on_stop = on_stop
        self._hotkey_hint = hotkey_hint
        super().__init__(master)

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

    def _handle_stop(self) -> None:
        try:
            self._on_stop()
        except Exception:
            pass


class ReplayControlOverlay(_EdgeRevealOverlay):
    """Pause, stop, and edit controls for a replay, revealed at the top screen edge."""

    def __init__(
        self,
        master: Any,
        *,
        on_pause: Callable[[], None],
        on_resume: Callable[[], None],
        on_stop: Callable[[], None],
        on_edit: Callable[[], None],
        on_previous: Callable[[], None],
        on_next: Callable[[], None],
        edit_enabled: bool = False,
    ) -> None:
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_stop = on_stop
        self._on_edit = on_edit
        self._on_previous = on_previous
        self._on_next = on_next
        self._paused = False
        self._edit_enabled = edit_enabled
        self._nav_index: int | None = None
        self._nav_total = 0
        self._step_label: ctk.CTkLabel | None = None
        self._error_label: ctk.CTkLabel | None = None
        self._nav_frame: ctk.CTkFrame | None = None
        self._pause_btn: ctk.CTkButton | None = None
        self._edit_btn: ctk.CTkButton | None = None
        self._prev_btn: ctk.CTkButton | None = None
        self._next_btn: ctk.CTkButton | None = None
        super().__init__(master)

    def _min_size(self) -> tuple[int, int]:
        return (340, 196)

    def _build(self, master: Any) -> None:
        win = ctk.CTkToplevel(master)
        self._window = win
        win.title("執行中")
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
            text="執行中",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(12, 2))
        self._step_label = ctk.CTkLabel(
            frame,
            text="尚未開始",
            font=ctk.CTkFont(size=11),
            text_color=("gray30", "gray65"),
            wraplength=292,
            justify="left",
            anchor="w",
        )
        self._step_label.pack(anchor="w", padx=14, pady=(0, 8), fill="x")
        self._error_label = ctk.CTkLabel(
            frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=("#b91c1c", "#f87171"),
            wraplength=292,
            justify="left",
            anchor="w",
        )

        nav = ctk.CTkFrame(frame, fg_color="transparent")
        self._nav_frame = nav
        nav.pack(padx=14, pady=(0, 8), fill="x")
        self._prev_btn = ctk.CTkButton(
            nav,
            text="上一步",
            width=100,
            height=32,
            command=self._handle_previous,
            state="disabled",
        )
        self._prev_btn.pack(side="left", padx=(0, 8))
        self._next_btn = ctk.CTkButton(
            nav,
            text="下一步",
            width=100,
            height=32,
            command=self._handle_next,
            state="disabled",
        )
        self._next_btn.pack(side="left")

        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.pack(padx=14, pady=(0, 14), fill="x")
        self._pause_btn = ctk.CTkButton(
            buttons,
            text="暫停",
            width=88,
            height=34,
            command=self._handle_pause,
        )
        self._pause_btn.pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            buttons,
            text="停止",
            width=88,
            height=34,
            fg_color=("#c0392b", "#a93226"),
            hover_color=("#a93226", "#922b21"),
            command=self._handle_stop,
        ).pack(side="left", padx=(0, 8))
        self._edit_btn = ctk.CTkButton(
            buttons,
            text="編輯步驟",
            width=100,
            height=34,
            command=self._handle_edit,
            state="normal" if self._edit_enabled else "disabled",
        )
        self._edit_btn.pack(side="left")

    def set_paused(self, paused: bool) -> None:
        """Switch the pause button between 暫停 and 繼續."""
        self._paused = paused
        button = self._pause_btn
        if button is None or self._destroyed:
            return
        if paused:
            button.configure(text="繼續", command=self._handle_resume)
        else:
            button.configure(text="暫停", command=self._handle_pause)

    def set_edit_enabled(self, enabled: bool) -> None:
        """Enable edit only when the running script is a recording folder."""
        self._edit_enabled = enabled
        button = self._edit_btn
        if button is None or self._destroyed:
            return
        button.configure(state="normal" if enabled else "disabled")

    def set_step(
        self,
        index: int | None,
        snippet: str,
        total: int = 0,
        error: str = "",
    ) -> None:
        """Show the selected step as ``步驟 N / total`` plus a short instruction."""
        label = self._step_label
        if label is None or self._destroyed:
            return
        self._nav_index = index
        self._nav_total = max(0, int(total))
        if index is None:
            text = f"尚未開始 / {self._nav_total}" if self._nav_total else "尚未開始"
        else:
            text = f"步驟 {index + 1}"
            if self._nav_total:
                text = f"{text} / {self._nav_total}"
            clipped = clip_step_snippet(snippet)
            if clipped:
                text = f"{text}：{clipped}"
        label.configure(text=text)
        self._apply_error(error)
        self._sync_nav_buttons()
        self._measure_and_hide_if_hidden()
        if self._pinned and not self._visible:
            self.reveal()

    def _apply_error(self, text: str) -> None:
        """Show ``text`` under the step, and keep the overlay up while it is shown."""
        label = self._error_label
        if label is None or self._destroyed:
            return
        cleaned = clip_step_snippet(text, limit=_STEP_ERROR_LIMIT)
        self._pinned = bool(cleaned)
        if not cleaned:
            self._cancel_hide()
            try:
                label.pack_forget()
            except Exception:
                pass
            return
        label.configure(text=cleaned)
        try:
            if not label.winfo_ismapped():
                pack_kwargs: dict[str, Any] = {
                    "anchor": "w",
                    "padx": 14,
                    "pady": (0, 8),
                    "fill": "x",
                }
                if self._nav_frame is not None:
                    pack_kwargs["before"] = self._nav_frame
                label.pack(**pack_kwargs)
        except Exception:
            pass

    def _sync_nav_buttons(self) -> None:
        previous = self._prev_btn
        nxt = self._next_btn
        if previous is not None and not self._destroyed:
            can_go_back = self._nav_index is not None and self._nav_index > 0
            previous.configure(state="normal" if can_go_back else "disabled")
        if nxt is not None and not self._destroyed:
            can_go_forward = (
                self._nav_index is not None
                and self._nav_total > 0
                and self._nav_index < self._nav_total - 1
            )
            nxt.configure(state="normal" if can_go_forward else "disabled")

    def _measure_and_hide_if_hidden(self) -> None:
        """Grow to fit the step label without revealing a hidden overlay."""
        was_visible = self._visible
        win = self._window
        if win is None:
            return
        try:
            win.update_idletasks()
            min_width, min_height = self._min_size()
            self._width = max(int(win.winfo_reqwidth()), min_width)
            self._height = max(int(win.winfo_reqheight()), min_height)
            if was_visible:
                win.geometry(
                    f"{self._width}x{self._height}+{int(win.winfo_x())}+{int(win.winfo_y())}"
                )
        except Exception:
            pass

    def _handle_pause(self) -> None:
        try:
            self._on_pause()
        except Exception:
            pass

    def _handle_resume(self) -> None:
        try:
            self._on_resume()
        except Exception:
            pass

    def _handle_stop(self) -> None:
        try:
            self._on_stop()
        except Exception:
            pass

    def _handle_edit(self) -> None:
        try:
            self._on_edit()
        except Exception:
            pass

    def _handle_previous(self) -> None:
        try:
            self._on_previous()
        except Exception:
            pass

    def _handle_next(self) -> None:
        try:
            self._on_next()
        except Exception:
            pass
