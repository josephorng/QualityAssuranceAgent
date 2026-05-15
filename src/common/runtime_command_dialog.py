"""GUI prompt for runtime command mode: enter next step or end the run."""

from __future__ import annotations

import threading
from collections.abc import Callable
from queue import Empty, Queue
from typing import TYPE_CHECKING

from src.common.ctk_dialogs import is_ctk_window, show_ctk_message

if TYPE_CHECKING:
    import tkinter as tk

_last_runtime_command: str | None = None

_runtime_command_provider: Callable[[], str | None] | None = None

# Set when runtime coordinator exits because the user ended the run at the prompt (End run / close dialog).
_runtime_user_ended_at_prompt: bool = False


def reset_runtime_user_ended_at_prompt() -> None:
    global _runtime_user_ended_at_prompt
    _runtime_user_ended_at_prompt = False


def note_runtime_user_ended_at_prompt() -> None:
    global _runtime_user_ended_at_prompt
    _runtime_user_ended_at_prompt = True


def consume_runtime_user_ended_at_prompt() -> bool:
    """Return whether the last coordinator run ended because the user chose End run, then clear the flag."""
    global _runtime_user_ended_at_prompt
    v = _runtime_user_ended_at_prompt
    _runtime_user_ended_at_prompt = False
    return v


def set_runtime_command_provider(provider: Callable[[], str | None] | None) -> None:
    global _runtime_command_provider
    _runtime_command_provider = provider


def get_last_runtime_command() -> str | None:
    return _last_runtime_command


def _prompt_runtime_command_console_fallback() -> str | None:
    cmd = input("Runtime command (empty to stop): ").strip()
    return None if not cmd else cmd


def show_runtime_command_ctk(parent: object, previous_command: str | None = None) -> str | None:
    import customtkinter as ctk

    result: dict[str, str | None] = {"action": "end", "cmd": None}

    dialog = ctk.CTkToplevel(parent)
    dialog.title("Runtime command")
    dialog.resizable(True, False)
    dialog.attributes("-topmost", True)
    dialog.after(100, lambda: dialog.attributes("-topmost", False))
    try:
        dialog.transient(parent.winfo_toplevel())
    except Exception:
        pass

    outer = ctk.CTkFrame(dialog, fg_color="transparent")
    outer.pack(fill="both", expand=True, padx=16, pady=16)
    outer.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        master=outer,
        text="Enter the next command for this step:",
        font=ctk.CTkFont(size=14),
    ).grid(row=0, column=0, sticky="w", pady=(0, 8))

    entry = ctk.CTkEntry(master=outer, width=520, height=36, font=ctk.CTkFont(size=14))
    entry.grid(row=1, column=0, sticky="ew", pady=(0, 14))

    btn_row = ctk.CTkFrame(outer, fg_color="transparent")
    btn_row.grid(row=2, column=0, sticky="ew")

    def on_run() -> None:
        text = entry.get().strip()
        if not text:
            show_ctk_message(
                dialog,
                "Runtime command",
                "Enter a non-empty command, or choose End run.",
                kind="warning",
            )
            return
        result["action"] = "run"
        result["cmd"] = text
        dialog.destroy()

    def on_end() -> None:
        result["action"] = "end"
        result["cmd"] = None
        dialog.destroy()

    def on_use_previous() -> None:
        if not previous_command:
            return
        entry.delete(0, "end")
        entry.insert(0, previous_command)
        entry.focus_set()

    ctk.CTkButton(btn_row, text="Previous command", width=150, command=on_use_previous).pack(
        side="left", padx=(0, 10)
    )
    ctk.CTkButton(btn_row, text="Run step", width=120, command=on_run).pack(side="left", padx=(0, 10))
    ctk.CTkButton(btn_row, text="End run", width=100, command=on_end).pack(side="left")

    def on_return(_event: object) -> str:
        on_run()
        return "break"

    entry.bind("<Return>", on_return)
    dialog.protocol("WM_DELETE_WINDOW", on_end)

    def _focus_when_shown() -> None:
        dialog.lift()
        entry.focus_force()

    dialog.after_idle(_focus_when_shown)
    dialog.update_idletasks()
    w, h = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
    sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
    dialog.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
    try:
        dialog.grab_set()
    except Exception:
        pass

    root = parent.winfo_toplevel()
    root.wait_window(dialog)

    if result["action"] == "run" and result["cmd"]:
        return str(result["cmd"])
    return None


def show_runtime_command_ttk(parent: "tk.Misc", previous_command: str | None = None) -> str | None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    result: dict[str, str | None] = {"action": "end", "cmd": None}

    dialog = tk.Toplevel(parent)
    dialog.title("Runtime command")
    dialog.resizable(True, False)
    dialog.attributes("-topmost", True)
    dialog.after(100, lambda: dialog.attributes("-topmost", False))
    try:
        dialog.transient(parent.winfo_toplevel())
    except tk.TclError:
        pass

    frame = ttk.Frame(dialog, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(0, weight=1)

    ttk.Label(frame, text="Enter the next command for this step:").grid(row=0, column=0, columnspan=2, sticky="w")
    entry_var = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=entry_var, width=56)
    entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 10))
    frame.columnconfigure(0, weight=1)

    btn_row = ttk.Frame(frame)
    btn_row.grid(row=2, column=0, columnspan=2, sticky="ew")

    def on_run() -> None:
        text = entry_var.get().strip()
        if not text:
            messagebox.showwarning(
                "Runtime command",
                "Enter a non-empty command, or choose End run.",
                parent=dialog,
            )
            return
        result["action"] = "run"
        result["cmd"] = text
        dialog.destroy()

    def on_end() -> None:
        result["action"] = "end"
        result["cmd"] = None
        dialog.destroy()

    def on_use_previous() -> None:
        if not previous_command:
            return
        entry_var.set(previous_command)
        entry.icursor(tk.END)
        entry.focus_set()

    ttk.Button(btn_row, text="Previous command", command=on_use_previous).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_row, text="Run step", command=on_run).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_row, text="End run", command=on_end).pack(side=tk.LEFT)

    def on_return(_event: object) -> str:
        on_run()
        return "break"

    entry.bind("<Return>", on_return)
    dialog.protocol("WM_DELETE_WINDOW", on_end)

    def _focus_entry_when_shown() -> None:
        dialog.lift()
        entry.focus_force()
        entry.icursor(tk.END)

    dialog.after_idle(_focus_entry_when_shown)
    dialog.update_idletasks()
    w, h = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
    sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
    dialog.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
    try:
        dialog.grab_set()
    except tk.TclError:
        pass

    top = parent.winfo_toplevel()
    top.wait_window(dialog)

    if result["action"] == "run" and result["cmd"]:
        return str(result["cmd"])
    return None


def show_runtime_command_toplevel(parent: "tk.Misc", previous_command: str | None = None) -> str | None:
    if is_ctk_window(parent):
        return show_runtime_command_ctk(parent, previous_command)
    return show_runtime_command_ttk(parent, previous_command)


def _prompt_runtime_command_tk_standalone(previous_command: str | None = None) -> str | None:
    try:
        import customtkinter as ctk
    except ImportError:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            return show_runtime_command_ttk(root, previous_command)
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass

    app = ctk.CTk()
    app.withdraw()
    try:
        return show_runtime_command_ctk(app, previous_command)
    finally:
        try:
            app.destroy()
        except Exception:
            pass


class RuntimeCommandHubBridge:
    """
    Bridges worker-thread ``prompt_runtime_command_popup`` calls to a Tk dialog on the main thread.

    Start polling before starting the worker; call ``stop`` after the worker joins.
    """

    def __init__(
        self,
        tk_parent: "tk.Misc",
        poll_interval_ms: int = 20,
        *,
        on_runtime_command: Callable[[str], None] | None = None,
    ) -> None:
        import tkinter as tk

        self._tk: tk.Misc = tk_parent
        self._poll_interval_ms = poll_interval_ms
        self._on_runtime_command = on_runtime_command
        self._q: Queue[tuple[threading.Event, list[str | None]]] = Queue()
        self._active = False

    def _provide(self) -> str | None:
        ev = threading.Event()
        slot: list[str | None] = [None]
        self._q.put((ev, slot))
        ev.wait()
        return slot[0]

    def _poll(self) -> None:
        if not self._active:
            return
        try:
            ev, slot = self._q.get_nowait()
        except Empty:
            self._tk.after(self._poll_interval_ms, self._poll)
            return
        cmd = show_runtime_command_toplevel(self._tk, get_last_runtime_command())
        if cmd and self._on_runtime_command is not None:
            self._on_runtime_command(cmd)
        slot[0] = cmd
        ev.set()
        self._tk.after(0, self._poll)

    def start(self) -> None:
        self._active = True
        set_runtime_command_provider(self._provide)
        self._tk.after(0, self._poll)

    def stop(self) -> None:
        self._active = False
        set_runtime_command_provider(None)


def prompt_runtime_command_popup() -> str | None:
    """
    Blocking prompt: run the next step with the returned command, or None if the user ends the run.

    Uses an injected provider (hub bridge), else a Tk Toplevel when possible, else stdin.
    """
    global _last_runtime_command

    cmd: str | None
    if _runtime_command_provider is not None:
        cmd = _runtime_command_provider()
    else:
        try:
            import tkinter as tk  # noqa: PLC0415
        except ImportError:
            cmd = _prompt_runtime_command_console_fallback()
        else:
            try:
                cmd = _prompt_runtime_command_tk_standalone(previous_command=_last_runtime_command)
            except tk.TclError:
                cmd = _prompt_runtime_command_console_fallback()

    if cmd:
        _last_runtime_command = cmd
    return cmd
