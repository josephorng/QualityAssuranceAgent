"""CustomTkinter dialogs that follow the active theme (same look as the main hub)."""

from __future__ import annotations

from typing import Any


def is_ctk_window(widget: Any) -> bool:
    try:
        from customtkinter import CTk, CTkToplevel

        return isinstance(widget, (CTk, CTkToplevel))
    except ImportError:
        return False


def show_ctk_message(
    master: Any,
    title: str,
    message: str,
    *,
    kind: str = "info",
) -> None:
    """Modal OK dialog; ``kind`` is ``info``, ``warning``, or ``error`` (accent only)."""
    import customtkinter as ctk

    dialog = ctk.CTkToplevel(master)
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.attributes("-topmost", True)
    dialog.after(120, lambda: dialog.attributes("-topmost", False))
    try:
        dialog.transient(master.winfo_toplevel())
    except Exception:
        pass

    inner = ctk.CTkFrame(dialog, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=22, pady=22)

    ctk.CTkLabel(
        master=inner,
        text=message,
        wraplength=420,
        justify="left",
        font=ctk.CTkFont(size=14),
    ).pack(anchor="w", pady=(0, 18))

    def _close() -> None:
        dialog.destroy()

    dialog.protocol("WM_DELETE_WINDOW", _close)

    btn = ctk.CTkButton(master=inner, text="確定", width=120, height=36, command=_close)
    if kind == "warning":
        btn.configure(fg_color="#B8860B", hover_color="#DAA520")
    elif kind == "error":
        btn.configure(fg_color="#C0392B", hover_color="#E74C3C")
    btn.pack()

    try:
        dialog.grab_set()
    except Exception:
        pass

    dialog.update_idletasks()
    w, h = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
    sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
    dialog.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    root = master.winfo_toplevel()
    root.wait_window(dialog)
