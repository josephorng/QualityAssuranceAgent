"""GUI prompt for runtime command mode: enter next step or end the run."""

from __future__ import annotations

_last_runtime_command: str | None = None


def _prompt_runtime_command_console_fallback() -> str | None:
    cmd = input("Runtime command (empty to stop): ").strip()
    return None if not cmd else cmd


def _prompt_runtime_command_tk(previous_command: str | None = None) -> str | None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    result: dict[str, str] = {"action": "end"}

    root = tk.Tk()
    root.title("Runtime command")
    root.resizable(True, False)
    root.attributes("-topmost", True)
    root.after(100, lambda: root.attributes("-topmost", False))

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

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
                parent=root,
            )
            return
        result["action"] = "run"
        result["cmd"] = text
        root.destroy()

    def on_end() -> None:
        result["action"] = "end"
        root.destroy()

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
    root.protocol("WM_DELETE_WINDOW", on_end)

    entry.focus_set()
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    root.mainloop()

    if result["action"] == "run":
        return result["cmd"]
    return None


def prompt_runtime_command_popup() -> str | None:
    """
    Blocking prompt: run the next step with the returned command, or None if the user ends the run.

    Uses a Tk window when available; falls back to stdin if Tk cannot be used.
    """
    global _last_runtime_command

    cmd: str | None
    try:
        import tkinter as tk  # noqa: PLC0415
    except ImportError:
        cmd = _prompt_runtime_command_console_fallback()
    else:
        try:
            cmd = _prompt_runtime_command_tk(previous_command=_last_runtime_command)
        except tk.TclError:
            cmd = _prompt_runtime_command_console_fallback()

    if cmd:
        _last_runtime_command = cmd
    return cmd
