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


def prompt_script_continue_or_end(
    master: Any,
    message: str,
) -> bool:
    """Return True to keep adding runtime steps to the open script, False to end the session."""
    import customtkinter as ctk

    result = {"continue": False}

    dialog = ctk.CTkToplevel(master)
    dialog.title("電腦使用代理")
    dialog.resizable(False, False)
    dialog.attributes("-topmost", True)
    dialog.after(120, lambda: dialog.attributes("-topmost", False))
    try:
        dialog.transient(master.winfo_toplevel())
    except Exception:
        pass

    inner = ctk.CTkFrame(dialog, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=22, pady=22)

    body = message.strip()
    if body:
        body += "\n\n"
    body += "要繼續逐步新增指令到目前腳本，還是結束此工作階段？"
    ctk.CTkLabel(
        master=inner,
        text=body,
        wraplength=420,
        justify="left",
        font=ctk.CTkFont(size=14),
    ).pack(anchor="w", pady=(0, 18))

    btn_row = ctk.CTkFrame(inner, fg_color="transparent")
    btn_row.pack()

    def on_continue() -> None:
        result["continue"] = True
        dialog.destroy()

    def on_end() -> None:
        result["continue"] = False
        dialog.destroy()

    dialog.protocol("WM_DELETE_WINDOW", on_end)

    ctk.CTkButton(
        master=btn_row, text="繼續新增", width=120, height=36, command=on_continue
    ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(
        master=btn_row, text="結束工作階段", width=120, height=36, command=on_end
    ).pack(side="left")

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
    return bool(result["continue"])


def prompt_append_recording_instructions(
    master: Any,
    message: str,
    *,
    folder_name: str = "",
    allow_append: bool = True,
) -> tuple[str, str]:
    """Ask what to do with generated recording instructions.

    Returns ``(choice, folder_name)``. ``choice`` is ``append``, ``open_review``,
    or ``close`` (closing the dialog counts as close). ``folder_name`` is the
    trimmed value from the recording-folder field (may be empty).
    """
    import customtkinter as ctk

    result: dict[str, str] = {"choice": "close", "folder_name": folder_name.strip()}

    dialog = ctk.CTkToplevel(master)
    dialog.title("錄製分析完成")
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
    ).pack(anchor="w", pady=(0, 14))

    ctk.CTkLabel(
        master=inner,
        text="錄製資料夾名稱",
        font=ctk.CTkFont(size=13),
        text_color=("gray30", "gray70"),
    ).pack(anchor="w", pady=(0, 6))
    name_entry = ctk.CTkEntry(
        master=inner,
        width=420,
        height=36,
        font=ctk.CTkFont(size=14),
        placeholder_text="例如：開啟神網",
    )
    name_entry.pack(fill="x", pady=(0, 18))
    if folder_name.strip():
        name_entry.insert(0, folder_name.strip())

    btn_row = ctk.CTkFrame(inner, fg_color="transparent")
    btn_row.pack()

    def _finish(choice: str) -> None:
        name = name_entry.get().strip()
        original = folder_name.strip()
        if name and name != original:
            illegal = '\\/:*?"<>|'
            if name in {".", ".."} or any(ch in name for ch in illegal) or len(name) > 191:
                show_ctk_message(
                    dialog,
                    "錄製分析完成",
                    '資料夾名稱無效（不可為 . 或 ..，也不可含 \\ / : * ? " < > |）。',
                    kind="warning",
                )
                return
        result["choice"] = choice
        result["folder_name"] = name
        dialog.destroy()

    def on_append() -> None:
        _finish("append")

    def on_open_review() -> None:
        _finish("open_review")

    def on_close() -> None:
        _finish("close")

    dialog.protocol("WM_DELETE_WINDOW", on_close)

    if allow_append:
        ctk.CTkButton(
            master=btn_row, text="加入腳本", width=120, height=36, command=on_append
        ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(
        master=btn_row, text="開啟錄製紀錄", width=140, height=36, command=on_open_review
    ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(master=btn_row, text="關閉", width=120, height=36, command=on_close).pack(
        side="left"
    )

    try:
        dialog.grab_set()
    except Exception:
        pass

    def _focus_name() -> None:
        name_entry.focus_set()
        try:
            name_entry.select_range(0, "end")
        except Exception:
            pass

    dialog.after(80, _focus_name)

    dialog.update_idletasks()
    w, h = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
    sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
    dialog.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    root = master.winfo_toplevel()
    root.wait_window(dialog)
    choice = result["choice"]
    if choice not in ("append", "open_review", "close"):
        choice = "close"
    return choice, result.get("folder_name", "")


def prompt_unsaved_script_changes(
    master: Any,
    *,
    message: str = "腳本內容已變更，但尚未儲存。要儲存變更嗎？",
    save_button_text: str = "儲存",
) -> str:
    """Warn that the script editor has unsaved edits.

    Returns ``save``, ``discard``, or ``cancel`` (closing the dialog counts as cancel).
    """
    import customtkinter as ctk

    result = {"choice": "cancel"}

    dialog = ctk.CTkToplevel(master)
    dialog.title("尚未儲存")
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

    btn_row = ctk.CTkFrame(inner, fg_color="transparent")
    btn_row.pack()

    def on_save() -> None:
        result["choice"] = "save"
        dialog.destroy()

    def on_discard() -> None:
        result["choice"] = "discard"
        dialog.destroy()

    def on_cancel() -> None:
        result["choice"] = "cancel"
        dialog.destroy()

    dialog.protocol("WM_DELETE_WINDOW", on_cancel)

    ctk.CTkButton(
        master=btn_row, text=save_button_text, width=100, height=36, command=on_save
    ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(
        master=btn_row, text="不要儲存", width=100, height=36, command=on_discard
    ).pack(side="left", padx=(0, 10))
    ctk.CTkButton(master=btn_row, text="取消", width=100, height=36, command=on_cancel).pack(
        side="left"
    )

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
    choice = result["choice"]
    return choice if choice in ("save", "discard", "cancel") else "cancel"
