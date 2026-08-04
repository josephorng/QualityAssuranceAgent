"""Modal dialog to edit agent settings (persisted in runs/agent_settings.json)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from src.common.ctk_dialogs import show_ctk_message
from src.common.monitor_prompt import (
    PRIMARY_MONITOR_MARKER,
    EyeMonitorChoice,
    list_eye_monitor_choices,
)
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

SCRIPT_FONT_SIZE_MIN = 10
SCRIPT_FONT_SIZE_MAX = 28
SCRIPT_FONT_SIZE_DEFAULT = 14


def clamp_script_font_size(value: Any) -> int:
    try:
        size = int(round(float(value)))
    except (TypeError, ValueError):
        return SCRIPT_FONT_SIZE_DEFAULT
    return max(SCRIPT_FONT_SIZE_MIN, min(SCRIPT_FONT_SIZE_MAX, size))


def _backend_to_label(backend: str) -> str:
    key = canonicalize_llm_backend(backend)
    return _BACKEND_LABELS.get(key, _BACKEND_LABELS["vllm_server"])


def _label_to_backend(label: str) -> str:
    text = label.strip()
    if text in _LABEL_TO_BACKEND:
        return _LABEL_TO_BACKEND[text]
    key = canonicalize_llm_backend(text)
    if key in BACKEND_PRESETS:
        return key
    return "vllm_server"


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
    return "triton_192_168_0_17"


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
    monitor_indices: list[int] | None = None,
    on_monitors_changed: Callable[[list[int]], None] | None = None,
    script_font_size: int = SCRIPT_FONT_SIZE_DEFAULT,
    on_script_font_size_changed: Callable[[int], None] | None = None,
) -> None:
    import customtkinter as ctk

    initial = load_agent_settings_dict()
    backend_initial = canonicalize_llm_backend(str(initial.get("llm_backend", "vllm_server")))
    if backend_initial not in BACKEND_PRESETS:
        backend_initial = "vllm_server"
    vision_initial = str(initial.get("vision_backend", "triton_192_168_0_17"))
    try:
        vision_initial = canonicalize_vision_backend(vision_initial)
    except ValueError:
        vision_initial = "triton_192_168_0_17"
    initial_font_size = clamp_script_font_size(script_font_size)
    saved_ok = False

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

    tabs = ctk.CTkTabview(inner, width=560, height=340)
    tabs.pack(fill="both", expand=True, pady=(0, 12))
    tab_agent = tabs.add("代理設定")
    tab_script = tabs.add("腳本編輯器")
    tab_monitor = tabs.add("螢幕畫面")

    backend_var = ctk.StringVar(value=_backend_to_label(backend_initial))
    vision_var = ctk.StringVar(value=_vision_backend_to_label(vision_initial))

    # ── Tab: 代理設定 ───────────────────────────────────────────────
    backend_row = ctk.CTkFrame(tab_agent, fg_color="transparent")
    backend_row.pack(fill="x", pady=(8, 10))
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

    preset_box = ctk.CTkFrame(tab_agent, fg_color=("gray90", "gray20"), corner_radius=8)
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

    vision_row = ctk.CTkFrame(tab_agent, fg_color="transparent")
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

    vision_preset_box = ctk.CTkFrame(tab_agent, fg_color=("gray90", "gray20"), corner_radius=8)
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
        tab_agent,
        text="後端設定將於下次執行或錄製分析時生效。",
        font=ctk.CTkFont(size=12),
        text_color=("gray30", "gray70"),
    ).pack(anchor="w", pady=(4, 0))

    # ── Tab: 腳本編輯器 ─────────────────────────────────────────────
    ctk.CTkLabel(
        tab_script,
        text="調整腳本與智能模式編輯框的字體大小。拖曳時即時預覽。",
        font=ctk.CTkFont(size=12),
        text_color=("gray30", "gray70"),
    ).pack(anchor="w", pady=(8, 10))

    font_row = ctk.CTkFrame(tab_script, fg_color="transparent")
    font_row.pack(fill="x", pady=(0, 10))
    ctk.CTkLabel(font_row, text="字體大小", width=120, anchor="w").pack(side="left")
    font_value_label = ctk.CTkLabel(
        font_row,
        text=str(initial_font_size),
        width=36,
        anchor="e",
    )
    font_value_label.pack(side="right")

    def _apply_font_size_preview(value: float) -> None:
        size = clamp_script_font_size(value)
        font_value_label.configure(text=str(size))
        if on_script_font_size_changed is not None:
            on_script_font_size_changed(size)

    font_slider = ctk.CTkSlider(
        font_row,
        from_=SCRIPT_FONT_SIZE_MIN,
        to=SCRIPT_FONT_SIZE_MAX,
        number_of_steps=SCRIPT_FONT_SIZE_MAX - SCRIPT_FONT_SIZE_MIN,
        command=_apply_font_size_preview,
    )
    font_slider.set(initial_font_size)
    font_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

    ctk.CTkLabel(
        tab_script,
        text="字體大小儲存後立即套用。",
        font=ctk.CTkFont(size=12),
        text_color=("gray30", "gray70"),
    ).pack(anchor="w", pady=(4, 0))

    # ── Tab: 螢幕畫面 ───────────────────────────────────────────────
    ctk.CTkLabel(
        tab_monitor,
        text="勾選要納入截取的每台顯示器。",
        font=ctk.CTkFont(size=12),
        text_color=("gray30", "gray70"),
    ).pack(anchor="w", pady=(8, 10))

    monitor_scroll = ctk.CTkScrollableFrame(tab_monitor, height=180)
    monitor_scroll.pack(fill="both", expand=True, pady=(0, 8))
    _monitor_checkboxes: list[Any] = []
    _monitor_indices_list: list[int] = []
    remembered = list(monitor_indices or [])

    def _format_monitor_label(c: EyeMonitorChoice) -> str:
        return f"{c.title} — {c.detail}"

    def _rebuild_monitors() -> None:
        for w in monitor_scroll.winfo_children():
            w.destroy()
        _monitor_checkboxes.clear()
        _monitor_indices_list.clear()
        try:
            choices = list_eye_monitor_choices()
        except Exception as e:
            show_ctk_message(dialog, "顯示器", f"無法列出顯示器：\n{e}", kind="error")
            choices = []
        default_on: list[int] = []
        for i, c in enumerate(choices):
            if PRIMARY_MONITOR_MARKER in c.title:
                default_on.append(i)
        if not default_on:
            default_on = [0] if choices else []
        for i, c in enumerate(choices):
            _monitor_indices_list.append(c.index)
            cb = ctk.CTkCheckBox(
                monitor_scroll,
                text=_format_monitor_label(c),
                font=ctk.CTkFont(size=13),
            )
            cb.pack(anchor="w", padx=4, pady=3)
            _monitor_checkboxes.append(cb)
            if remembered and c.index in remembered:
                cb.select()
            elif not remembered and i in default_on:
                cb.select()

    _rebuild_monitors()

    monitor_btn_row = ctk.CTkFrame(tab_monitor, fg_color="transparent")
    monitor_btn_row.pack(fill="x", pady=(0, 6))
    ctk.CTkButton(
        monitor_btn_row, text="重新整理", width=100, command=_rebuild_monitors
    ).pack(side="left")

    def _get_selected_monitor_indices() -> list[int]:
        out: list[int] = []
        for midx, cb in zip(_monitor_indices_list, _monitor_checkboxes):
            if cb.get():
                out.append(midx)
        return out

    ctk.CTkLabel(
        tab_monitor,
        text="螢幕設定將於下次執行或錄製分析時生效。",
        font=ctk.CTkFont(size=12),
        text_color=("gray30", "gray70"),
    ).pack(anchor="w", pady=(4, 0))

    btn_row = ctk.CTkFrame(inner, fg_color="transparent")
    btn_row.pack(fill="x")

    def _close() -> None:
        nonlocal saved_ok
        if not saved_ok and on_script_font_size_changed is not None:
            on_script_font_size_changed(initial_font_size)
        dialog.destroy()

    def _save() -> None:
        nonlocal saved_ok
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
        final_size = clamp_script_font_size(font_slider.get())
        if on_script_font_size_changed is not None:
            on_script_font_size_changed(final_size)
        if on_monitors_changed is not None:
            on_monitors_changed(_get_selected_monitor_indices())
        if on_saved is not None:
            on_saved()
        saved_ok = True
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
