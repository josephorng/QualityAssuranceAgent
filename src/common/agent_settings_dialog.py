"""Modal dialog to edit agent settings (persisted in runs/agent_settings.json)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from src.common.ctk_dialogs import show_ctk_message
from src.common.settings import (
    BACKEND_PRESETS,
    VISION_BACKEND_PRESETS,
    canonicalize_llm_backend,
    canonicalize_vision_backend,
    load_agent_settings_dict,
    preset_for_backend,
    preset_for_vision_backend,
    probe_llm_backend,
    probe_vision_backend,
    save_agent_settings_dict,
    validate_agent_settings_dict,
)

_BACKEND_LABELS: dict[str, str] = {
    "ollama_local": "ollama (local)",
    "ollama_local_12b": "ollama (local 12b)",
    "ollama_server": "ollama (公司主機)",
    "vllm_server": "vLLM (192.168.4.134)",
}
_LABEL_TO_BACKEND = {label: key for key, label in _BACKEND_LABELS.items()}
_BACKEND_MENU_VALUES = [
    _BACKEND_LABELS["ollama_local"],
    _BACKEND_LABELS["ollama_local_12b"],
    _BACKEND_LABELS["ollama_server"],
    _BACKEND_LABELS["vllm_server"],
]

_VISION_BACKEND_LABELS: dict[str, str] = {
    "triton_local": "triton（本機 127.0.0.1）",
    "triton_192_168_0_17": "triton（192.168.0.17）",
}
_VISION_LABEL_TO_BACKEND = {label: key for key, label in _VISION_BACKEND_LABELS.items()}
_VISION_BACKEND_MENU_VALUES = [
    _VISION_BACKEND_LABELS["triton_local"],
    _VISION_BACKEND_LABELS["triton_192_168_0_17"],
]


def _backend_to_label(backend: str) -> str:
    key = canonicalize_llm_backend(backend)
    return _BACKEND_LABELS.get(key, _BACKEND_LABELS["ollama_local"])


def _label_to_backend(label: str) -> str:
    text = label.strip()
    if text in _LABEL_TO_BACKEND:
        return _LABEL_TO_BACKEND[text]
    key = canonicalize_llm_backend(text)
    if key in BACKEND_PRESETS:
        return key
    return "ollama_local"


def _vision_backend_to_label(backend: str) -> str:
    key = canonicalize_vision_backend(backend)
    return _VISION_BACKEND_LABELS[key]


def _label_to_vision_backend(label: str) -> str:
    text = label.strip()
    if text in _VISION_LABEL_TO_BACKEND:
        return _VISION_LABEL_TO_BACKEND[text]
    key = canonicalize_vision_backend(text)
    if key in VISION_BACKEND_PRESETS:
        return key
    return "triton_local"


def _vision_preset_summary(backend: str) -> tuple[str, str]:
    preset = preset_for_vision_backend(backend)
    return "模式：Triton 推論（本機 ONNX 已停用）", f"Triton 主機：{preset['triton_http_url']}"


def _preset_summary(backend: str) -> tuple[str, str]:
    key = canonicalize_llm_backend(backend)
    preset = BACKEND_PRESETS[key]
    return preset["brain_lm"], preset["ollama_host"]


def open_agent_settings_dialog(
    master: Any,
    *,
    on_saved: Callable[[], None] | None = None,
) -> None:
    import customtkinter as ctk

    initial = load_agent_settings_dict()
    backend_initial = canonicalize_llm_backend(str(initial.get("llm_backend", "ollama_local")))
    if backend_initial not in BACKEND_PRESETS:
        backend_initial = "ollama_local"
    vision_initial = str(initial.get("vision_backend", "triton_local"))
    try:
        vision_initial = canonicalize_vision_backend(vision_initial)
    except ValueError:
        vision_initial = "triton_local"

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
    vision_var = ctk.StringVar(value=_vision_backend_to_label(vision_initial))

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

    test_btn = ctk.CTkButton(backend_row, text="測試 LLM 連線", width=130)
    test_btn.pack(side="left", padx=(10, 0))

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

    vision_row = ctk.CTkFrame(inner, fg_color="transparent")
    vision_row.pack(fill="x", pady=(0, 10))
    ctk.CTkLabel(vision_row, text="Vision 後端", width=120, anchor="w").pack(side="left")
    ctk.CTkOptionMenu(
        vision_row,
        values=_VISION_BACKEND_MENU_VALUES,
        variable=vision_var,
        width=240,
        command=lambda _: _sync_vision_preset_labels(),
    ).pack(side="left")

    vision_test_btn = ctk.CTkButton(vision_row, text="測試 Vision 連線", width=130)
    vision_test_btn.pack(side="left", padx=(10, 0))

    vision_preset_box = ctk.CTkFrame(inner, fg_color=("gray90", "gray20"), corner_radius=8)
    vision_preset_box.pack(fill="x", pady=(0, 10))
    vision_preset_inner = ctk.CTkFrame(vision_preset_box, fg_color="transparent")
    vision_preset_inner.pack(fill="x", padx=14, pady=12)
    vision_mode_label = ctk.CTkLabel(
        vision_preset_inner,
        text="",
        font=ctk.CTkFont(size=13),
        anchor="w",
        justify="left",
    )
    vision_mode_label.pack(anchor="w")
    vision_host_label = ctk.CTkLabel(
        vision_preset_inner,
        text="",
        font=ctk.CTkFont(size=13),
        anchor="w",
        justify="left",
    )
    vision_host_label.pack(anchor="w", pady=(4, 0))

    def _sync_vision_preset_labels() -> None:
        backend = _label_to_vision_backend(vision_var.get())
        vision_var.set(_vision_backend_to_label(backend))
        mode, host = _vision_preset_summary(backend)
        vision_mode_label.configure(text=mode)
        vision_host_label.configure(text=host)

    _sync_vision_preset_labels()

    ctk.CTkLabel(
        inner,
        text="變更將於下次執行或錄製分析時生效。",
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
            vision_backend = _label_to_vision_backend(vision_var.get())
            payload.update(preset_for_vision_backend(vision_backend))
            payload["vision_backend"] = vision_backend
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
        _close()

    def _on_probe_done(
        ok: bool,
        message: str,
        *,
        title: str,
        button: Any,
        default_text: str,
    ) -> None:
        if not dialog.winfo_exists():
            return
        button.configure(state="normal", text=default_text)
        show_ctk_message(
            dialog,
            title,
            message,
            kind="info" if ok else "error",
        )

    def _test_llm_connection() -> None:
        backend = _label_to_backend(backend_var.get())
        test_btn.configure(state="disabled", text="測試中…")

        def _worker() -> None:
            try:
                ok, message = probe_llm_backend(backend)
            except ValueError as e:
                ok, message = False, str(e)
            dialog.after(
                0,
                lambda: _on_probe_done(
                    ok,
                    message,
                    title="測試 LLM 連線",
                    button=test_btn,
                    default_text="測試 LLM 連線",
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    test_btn.configure(command=_test_llm_connection)

    def _test_vision_connection() -> None:
        vision_backend = _label_to_vision_backend(vision_var.get())
        vision_test_btn.configure(state="disabled", text="測試中…")

        def _worker() -> None:
            try:
                ok, message = probe_vision_backend(vision_backend)
            except ValueError as e:
                ok, message = False, str(e)
            dialog.after(
                0,
                lambda: _on_probe_done(
                    ok,
                    message,
                    title="測試 Vision 連線",
                    button=vision_test_btn,
                    default_text="測試 Vision 連線",
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    vision_test_btn.configure(command=_test_vision_connection)

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
