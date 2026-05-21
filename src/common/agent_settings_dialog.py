"""Modal dialog to edit agent settings (persisted in runs/agent_settings.json)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.common.ctk_dialogs import show_ctk_message
from src.common.settings import (
    BACKEND_PRESETS,
    load_agent_settings_dict,
    preset_for_backend,
    save_agent_settings_dict,
    validate_agent_settings_dict,
)

_BACKEND_LABELS: dict[str, str] = {
    "ollama": "ollama (local)",
    "vllm": "ollama (公司主機)",
}
_LABEL_TO_BACKEND = {label: key for key, label in _BACKEND_LABELS.items()}
_BACKEND_MENU_VALUES = [_BACKEND_LABELS["ollama"], _BACKEND_LABELS["vllm"]]


def _backend_to_label(backend: str) -> str:
    key = backend.strip().lower()
    return _BACKEND_LABELS.get(key, _BACKEND_LABELS["ollama"])


def _label_to_backend(label: str) -> str:
    text = label.strip()
    if text in _LABEL_TO_BACKEND:
        return _LABEL_TO_BACKEND[text]
    key = text.lower()
    if key in BACKEND_PRESETS:
        return key
    return "ollama"


def _preset_summary(backend: str) -> tuple[str, str]:
    key = backend.strip().lower()
    preset = BACKEND_PRESETS[key]
    host_key = "ollama_host" if key == "ollama" else "vllm_host"
    return preset["brain_lm"], preset[host_key]


def open_agent_settings_dialog(
    master: Any,
    *,
    on_saved: Callable[[], None] | None = None,
) -> None:
    import customtkinter as ctk

    initial = load_agent_settings_dict()
    backend_initial = str(initial.get("llm_backend", "ollama")).strip().lower()
    if backend_initial not in BACKEND_PRESETS:
        backend_initial = "ollama"

    dialog = ctk.CTkToplevel(master)
    dialog.title("代理設定")
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
        inner,
        text="代理設定",
        font=ctk.CTkFont(size=18, weight="bold"),
    ).pack(anchor="w", pady=(0, 12))

    backend_var = ctk.StringVar(value=_backend_to_label(backend_initial))
    debug_var = ctk.BooleanVar(value=bool(initial.get("debug", True)))

    backend_row = ctk.CTkFrame(inner, fg_color="transparent")
    backend_row.pack(fill="x", pady=(0, 10))
    ctk.CTkLabel(backend_row, text="LLM 後端", width=120, anchor="w").pack(side="left")
    ctk.CTkOptionMenu(
        backend_row,
        values=_BACKEND_MENU_VALUES,
        variable=backend_var,
        width=240,
        command=lambda _: _sync_preset_labels(),
    ).pack(side="left")

    preset_box = ctk.CTkFrame(inner, fg_color=("gray90", "gray20"), corner_radius=8)
    preset_box.pack(fill="x", pady=(0, 10))
    preset_inner = ctk.CTkFrame(preset_box, fg_color="transparent")
    preset_inner.pack(fill="x", padx=14, pady=12)
    model_label = ctk.CTkLabel(
        preset_inner,
        text="",
        font=ctk.CTkFont(size=13),
        anchor="w",
        justify="left",
    )
    model_label.pack(anchor="w")
    host_label = ctk.CTkLabel(
        preset_inner,
        text="",
        font=ctk.CTkFont(size=13),
        anchor="w",
        justify="left",
    )
    host_label.pack(anchor="w", pady=(4, 0))

    def _sync_preset_labels() -> None:
        backend = _label_to_backend(backend_var.get())
        backend_var.set(_backend_to_label(backend))
        model, host = _preset_summary(backend)
        model_label.configure(text=f"模型：{model}")
        host_label.configure(text=f"主機：{host}")

    _sync_preset_labels()

    debug_row = ctk.CTkFrame(inner, fg_color="transparent")
    debug_row.pack(fill="x", pady=(0, 10))
    ctk.CTkLabel(debug_row, text="除錯模式", width=120, anchor="w").pack(side="left")
    ctk.CTkCheckBox(debug_row, text="", variable=debug_var, width=28).pack(side="left")

    ctk.CTkLabel(
        inner,
        text="變更將於下次執行時生效。",
        font=ctk.CTkFont(size=12),
        text_color=("gray30", "gray70"),
    ).pack(anchor="w", pady=(4, 14))

    btn_row = ctk.CTkFrame(inner, fg_color="transparent")
    btn_row.pack(fill="x")

    def _close() -> None:
        dialog.destroy()

    def _save() -> None:
        backend = _label_to_backend(backend_var.get())
        try:
            payload = preset_for_backend(backend)
            payload["debug"] = debug_var.get()
            validated = validate_agent_settings_dict(payload)
            save_agent_settings_dict(validated)
        except ValueError as e:
            show_ctk_message(dialog, "代理設定", str(e), kind="warning")
            return
        except OSError as e:
            show_ctk_message(dialog, "代理設定", f"無法儲存設定：\n{e}", kind="error")
            return
        if on_saved is not None:
            on_saved()
        show_ctk_message(dialog, "代理設定", "設定已儲存。", kind="info")
        _close()

    ctk.CTkButton(btn_row, text="取消", width=100, command=_close).pack(side="right", padx=(8, 0))
    ctk.CTkButton(btn_row, text="儲存", width=100, command=_save).pack(side="right")

    dialog.protocol("WM_DELETE_WINDOW", _close)

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
