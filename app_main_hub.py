"""CustomTkinter hub: runs from an opened script file or step-by-step when no file is set."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import customtkinter as ctk
from tkinter import filedialog

from main import analyze_screen_recording, dismiss_nuitka_onefile_splash, prepare_run_session, run_coordinator_sync
from src.common.agent_settings_dialog import open_agent_settings_dialog
from src.common.ctk_dialogs import (
    prompt_append_recording_instructions,
    prompt_script_continue_or_end,
    show_ctk_message,
)
from src.common.io_utils import append_text, pop_last_nonempty_line, read_json, write_json
from src.common.monitor_prompt import (
    PRIMARY_MONITOR_MARKER,
    EyeMonitorChoice,
    list_eye_monitor_choices,
)
from src.common.run_state import get_run_state_manager, unique_run_folder_name
from src.common.session_report import should_write_session_report, write_session_report
from src.common.runtime_command_dialog import (
    RuntimeCommandHubBridge,
    consume_runtime_user_ended_at_prompt,
    reset_runtime_user_ended_at_prompt,
)
from src.common.runtime_context import USE_TOOL_CACHE_ENV
from src.common.script_helper import parse_executable_lines_from_text
from src.common.settings import ROOT_DIR, apply_startup_ollama_host_probe, load_settings
from src.recorder.capture import RecordingSession
from src.recorder.hotkey import RecordingHotkeyManager

# Step-mode runtime command transcript (append during run); not hub UI preferences.
_RUNTIME_COMMAND_TRANSCRIPT_NAME = "runtime_commands_cache.txt"
_RUNTIME_COMMAND_LABEL = "逐步執行命令"
_HUB_UI_STATE_NAME = "hub_ui.json"
_HUB_UI_VERSION = 1


def _default_hub_ui_dict() -> dict[str, Any]:
    return {
        "version": _HUB_UI_VERSION,
        "appearance_dark": True,
        "selected_monitor_indices": [],
        "last_script_path": None,
        "use_tool_cache": False,
        "recording_hotkey_enabled": True,
    }


def _coerce_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for x in value:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _normalize_hub_ui_state(raw: Any) -> dict[str, Any]:
    base = _default_hub_ui_dict()
    if not isinstance(raw, dict):
        return base
    base["version"] = int(raw.get("version", _HUB_UI_VERSION))
    base["appearance_dark"] = bool(raw.get("appearance_dark", True))
    base["selected_monitor_indices"] = _coerce_int_list(raw.get("selected_monitor_indices"))
    lsp = raw.get("last_script_path")
    base["last_script_path"] = lsp if isinstance(lsp, str) or lsp is None else None
    base["use_tool_cache"] = bool(raw.get("use_tool_cache", False))
    base["recording_hotkey_enabled"] = bool(raw.get("recording_hotkey_enabled", True))
    return base


def _read_hub_ui_state() -> dict[str, Any]:
    try:
        path = Path(load_settings().runs_dir) / _HUB_UI_STATE_NAME
        return _normalize_hub_ui_state(read_json(path, {}))
    except (OSError, ValueError, TypeError):
        return _default_hub_ui_dict()


def _hub_ui_state_path() -> Path:
    return Path(load_settings().runs_dir) / _HUB_UI_STATE_NAME


@dataclass
class _WorkerArgs:
    step_mode: bool
    eye_monitor_indices: list[int]
    script_raw: str
    script_disk_path: Path | None
    run_folder_name: str | None = None
    use_tool_cache: bool = False


class MainHub(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("電腦使用代理")
        self.geometry("960x780")
        self.minsize(880, 880)

        hub = _read_hub_ui_state()
        self._remember_monitor_indices: list[int] = list(hub["selected_monitor_indices"])
        self._appearance_dark = bool(hub["appearance_dark"])
        self._use_tool_cache = bool(hub.get("use_tool_cache", False))
        self._recording_hotkey_enabled = bool(hub.get("recording_hotkey_enabled", True))
        self._suppress_hub_monitor_persist = False
        ctk.set_appearance_mode("dark" if self._appearance_dark else "light")
        ctk.set_default_color_theme("dark-blue")

        self._script_path: Path | None = None
        # When set, Save writes here (step-mode transcript under runs_dir); not a user-opened script.
        self._runtime_commands_cache_path: Path | None = None
        self._worker_thread: threading.Thread | None = None
        self._bridge: RuntimeCommandHubBridge | None = None
        self._worker_outcome: tuple[str, str] = ("ok", "")
        self._user_requested_stop = False
        self._stop_cancel_remaining = 0

        self._monitor_labels: list[str] = []
        self._monitor_indices: list[int] = []
        self._monitor_checkboxes: list[ctk.CTkCheckBox] = []

        self._post_run_unlink: Path | None = None
        self._script_controls: list[Any] = []
        self._last_run_was_script_mode = False
        self._last_script_run_folder: str | None = None
        self._active_run_root: Path | None = None

        self._recording_session = RecordingSession()
        self._recording_session.set_on_event(self._on_recording_event)
        self._recording_hotkey = RecordingHotkeyManager()
        self._recording_analysis_thread: threading.Thread | None = None
        self._analysis_cancel_event = threading.Event()
        self._suppress_script_cache_sync = False
        self._sync_cache_after_id: str | None = None
        self._record_btn: ctk.CTkButton | None = None
        self._analysis_progress_frame: ctk.CTkFrame | None = None
        self._analysis_progress: ctk.CTkProgressBar | None = None

        self._build_header()
        self._build_monitor_row()
        self._build_script_section()
        self._build_actions_row()
        self._build_status()

        self._refresh_monitors()
        last_script = hub.get("last_script_path")
        if isinstance(last_script, str) and last_script.strip():
            p = Path(last_script)
            if p.is_file():
                self._script_path = p
                self._runtime_commands_cache_path = None
                text = p.read_text(encoding="utf-8")
                self._suppress_script_cache_sync = True
                try:
                    self._script_text.delete("0.0", "end")
                    self._script_text.insert("0.0", text)
                finally:
                    self._suppress_script_cache_sync = False
                self._refresh_script_path_label()
        if self._script_path is None:
            self._try_load_last_runtime_command_cache()

        self._status.configure(text="正在檢查 Ollama 主機…")
        self._start_ollama_host_probe()

        if self._recording_hotkey_enabled:
            self._recording_hotkey.register(self._schedule_toggle_recording)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_ollama_host_probe(self) -> None:
        def work() -> None:
            ok, message = apply_startup_ollama_host_probe()
            self.after(0, lambda: self._on_ollama_host_probe_done(ok, message))

        threading.Thread(target=work, daemon=True).start()

    def _on_ollama_host_probe_done(self, ok: bool, message: str) -> None:
        if ok:
            self._status.configure(text=message, text_color=("gray20", "gray65"))
        else:
            self._status.configure(
                text=message,
                text_color=("#b91c1c", "#f87171"),
            )

    def _refresh_script_path_label(self) -> None:
        if self._script_path is not None:
            self._script_path_label.configure(text=str(self._script_path.resolve()))
        elif self._runtime_commands_cache_path is not None:
            self._script_path_label.configure(text=_RUNTIME_COMMAND_LABEL)
        else:
            self._script_path_label.configure(text="未載入檔案")

    def _try_load_last_runtime_command_cache(self) -> None:
        """If no script file is open, show the last runtime command cache for editing and Save."""
        if self._script_path is not None:
            return
        settings = load_settings()
        cache_path = Path(settings.runs_dir) / _RUNTIME_COMMAND_TRANSCRIPT_NAME
        if not cache_path.is_file():
            return
        raw = cache_path.read_text(encoding="utf-8")
        if not raw.strip():
            return
        self._runtime_commands_cache_path = cache_path
        self._suppress_script_cache_sync = True
        try:
            self._script_text.delete("0.0", "end")
            self._script_text.insert("0.0", raw)
        finally:
            self._suppress_script_cache_sync = False
        self._refresh_script_path_label()

    def _append_runtime_command_to_script_view(self, cmd: str) -> None:
        """Underlying Tk Text ignores ``insert`` while the widget is ``disabled`` (as during a run)."""
        self._script_text.configure(state="normal")
        self._script_text.insert("end", cmd + "\n")
        self._script_text.configure(state="disabled")

    def _pop_last_runtime_command_from_cache(self) -> None:
        p = self._runtime_commands_cache_path
        if p is None:
            return
        pop_last_nonempty_line(p)
        self._script_text.configure(state="normal")
        self._script_text.delete("0.0", "end")
        if p.is_file():
            self._script_text.insert("0.0", p.read_text(encoding="utf-8"))
        self._script_text.configure(state="disabled")

    def _pop_last_runtime_command_from_script_file(self) -> None:
        p = self._script_path
        if p is None:
            return
        pop_last_nonempty_line(p)
        self._script_text.configure(state="normal")
        self._script_text.delete("0.0", "end")
        if p.is_file():
            self._script_text.insert("0.0", p.read_text(encoding="utf-8"))
        self._script_text.configure(state="disabled")

    def _refresh_runtime_script_text_from_cache(self) -> None:
        """After a runtime-command run, reload the cache file into the script textbox (disk is source of truth)."""
        p = self._runtime_commands_cache_path
        if p is None or not p.is_file():
            return
        self._suppress_script_cache_sync = True
        try:
            self._script_text.delete("0.0", "end")
            self._script_text.insert("0.0", p.read_text(encoding="utf-8"))
        finally:
            self._suppress_script_cache_sync = False
        self._refresh_script_path_label()

    def _sync_script_text_to_runtime_cache(self) -> bool:
        """Write the script textbox to runtime_commands_cache.txt when no script file is open."""
        if self._script_path is not None:
            return False
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return False
        cache_path = self._runtime_command_transcript_path()
        self._runtime_commands_cache_path = cache_path
        body = self._script_text.get("0.0", "end").rstrip() + "\n"
        cache_path.write_text(body, encoding="utf-8")
        self._refresh_script_path_label()
        return True

    def _schedule_sync_script_text_to_runtime_cache(self) -> None:
        if self._suppress_script_cache_sync or self._script_path is not None:
            return
        if self._sync_cache_after_id is not None:
            self.after_cancel(self._sync_cache_after_id)
        self._sync_cache_after_id = self.after(250, self._run_scheduled_sync_script_text_to_runtime_cache)

    def _run_scheduled_sync_script_text_to_runtime_cache(self) -> None:
        self._sync_cache_after_id = None
        self._sync_script_text_to_runtime_cache()

    def _on_script_text_modified(self, _event: object = None) -> None:
        textbox = self._script_text._textbox
        if textbox.edit_modified():
            textbox.edit_modified(False)
            self._schedule_sync_script_text_to_runtime_cache()

    def _bind_script_text_cache_sync(self) -> None:
        self._script_text._textbox.bind("<<Modified>>", self._on_script_text_modified)

    def _build_header(self) -> None:
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(20, 8))

        top_row = ctk.CTkFrame(head, fg_color="transparent")
        top_row.pack(fill="x")

        theme_row = ctk.CTkFrame(top_row, fg_color="transparent")
        theme_row.pack(side="right", anchor="ne")
        self._settings_btn = ctk.CTkButton(
            theme_row,
            text="\u2699",
            width=32,
            height=32,
            corner_radius=16,
            font=ctk.CTkFont(size=15),
            command=self._open_settings,
        )
        self._settings_btn.pack(side="left", padx=(0, 6))
        self._appearance_toggle_btn = ctk.CTkButton(
            theme_row,
            text="\u2600",
            width=32,
            height=32,
            corner_radius=16,
            font=ctk.CTkFont(size=15),
            command=self._toggle_appearance,
        )
        self._appearance_toggle_btn.pack(side="left")

        left_col = ctk.CTkFrame(top_row, fg_color="transparent")
        left_col.pack(side="left", fill="x", expand=True, padx=(0, 16))
        ctk.CTkLabel(
            left_col,
            text="電腦使用代理",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            left_col,
            text="設定一次執行，選擇螢幕畫面，然後開始。",
            font=ctk.CTkFont(size=14),
            text_color=("gray30", "gray70"),
        ).pack(anchor="w", pady=(4, 0))
        self._sync_appearance_toggle_button()

    def _persist_hub_ui_state(self) -> None:
        try:
            data = {
                "version": _HUB_UI_VERSION,
                "appearance_dark": self._appearance_dark,
                "selected_monitor_indices": self._selected_monitor_indices(),
                "last_script_path": str(self._script_path.resolve())
                if self._script_path is not None
                else None,
                "use_tool_cache": self._tool_cache_enabled_for_run(),
                "recording_hotkey_enabled": self._recording_hotkey_enabled,
            }
            write_json(_hub_ui_state_path(), data)
            self._remember_monitor_indices = list(data["selected_monitor_indices"])
        except OSError:
            pass

    def _open_settings(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        open_agent_settings_dialog(
            self,
            on_saved=lambda: self._status.configure(text="設定已儲存"),
        )

    def _toggle_appearance(self) -> None:
        self._appearance_dark = not self._appearance_dark
        ctk.set_appearance_mode("dark" if self._appearance_dark else "light")
        self._sync_appearance_toggle_button()
        self._persist_hub_ui_state()

    def _sync_appearance_toggle_button(self) -> None:
        self._appearance_toggle_btn.configure(
            text="\u2600" if self._appearance_dark else "\u263e"
        )

    def _build_monitor_row(self) -> None:
        box = ctk.CTkFrame(self, corner_radius=12)
        box.pack(fill="x", padx=24, pady=8)
        ctk.CTkLabel(box, text="螢幕畫面", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=16, pady=(14, 4)
        )
        ctk.CTkLabel(
            box,
            text="勾選要納入截取的每台顯示器。",
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70"),
            wraplength=860,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))
        row = ctk.CTkFrame(box, fg_color="transparent")
        self._monitor_checks_scroll = ctk.CTkScrollableFrame(row, height=200)
        self._monitor_checks_scroll.pack(side="left", fill="both", expand=True)
        self._monitor_refresh_btn = ctk.CTkButton(
            row, text="重新整理", width=100, command=self._refresh_monitors
        )
        self._monitor_refresh_btn.pack(side="left", padx=(10, 0), anchor="n")
        row.pack(fill="x", padx=16, pady=(0, 14))

    def _refresh_monitors(self) -> None:
        try:
            choices = list_eye_monitor_choices()
        except Exception as e:
            show_ctk_message(self, "顯示器", f"無法列出顯示器：\n{e}", kind="error")
            choices = []
        self._monitor_labels = [self._format_monitor_row(c) for c in choices]
        self._monitor_indices = [c.index for c in choices]
        self._suppress_hub_monitor_persist = True
        try:
            self._rebuild_monitor_checkboxes()
            self._apply_remembered_monitor_selection()
        finally:
            self._suppress_hub_monitor_persist = False

    def _apply_remembered_monitor_selection(self) -> None:
        valid = [i for i in self._remember_monitor_indices if i in self._monitor_indices]
        if not valid:
            return
        for midx, cb in zip(self._monitor_indices, self._monitor_checkboxes):
            if midx in valid:
                cb.select()
            else:
                cb.deselect()

    def _on_monitor_checkbox_changed(self) -> None:
        if self._suppress_hub_monitor_persist:
            return
        self._remember_monitor_indices = self._selected_monitor_indices()
        self._persist_hub_ui_state()

    def _rebuild_monitor_checkboxes(self) -> None:
        for w in self._monitor_checks_scroll.winfo_children():
            w.destroy()
        self._monitor_checkboxes.clear()
        default_on: list[int] = []
        for i, label in enumerate(self._monitor_labels):
            if PRIMARY_MONITOR_MARKER in label:
                default_on.append(i)
        if not default_on:
            default_on = [0] if self._monitor_labels else []
        for i, label in enumerate(self._monitor_labels):
            cb = ctk.CTkCheckBox(
                self._monitor_checks_scroll,
                text=label,
                font=ctk.CTkFont(size=13),
                command=self._on_monitor_checkbox_changed,
            )
            cb.pack(anchor="w", padx=4, pady=3)
            self._monitor_checkboxes.append(cb)
            if i in default_on:
                cb.select()
            else:
                cb.deselect()

    @staticmethod
    def _format_monitor_row(c: EyeMonitorChoice) -> str:
        return f"{c.title} — {c.detail}"

    def _selected_monitor_indices(self) -> list[int]:
        out: list[int] = []
        for midx, cb in zip(self._monitor_indices, self._monitor_checkboxes):
            if cb.get():
                out.append(midx)
        return out

    def _build_script_section(self) -> None:
        box = ctk.CTkFrame(self, corner_radius=12)
        box.pack(fill="both", expand=True, padx=24, pady=8)
        ctk.CTkLabel(box, text="腳本", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=16, pady=(14, 4)
        )
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=4)
        b_open = ctk.CTkButton(row, text="開啟…", width=100, command=self._script_open)
        b_open.pack(side="left", padx=(0, 8))
        b_save = ctk.CTkButton(row, text="儲存", width=100, command=self._script_save)
        b_save.pack(side="left", padx=(0, 8))
        b_sas = ctk.CTkButton(row, text="另存新檔…", width=100, command=self._script_save_as)
        b_sas.pack(side="left", padx=(0, 8))
        b_clear = ctk.CTkButton(row, text="清空", width=100, command=self._script_clear)
        b_clear.pack(side="left", padx=(0, 8))
        self._record_btn = ctk.CTkButton(
            row, text="開始錄製", width=100, command=self._on_record_button
        )
        self._record_btn.pack(side="left")
        self._script_path_label = ctk.CTkLabel(
            box,
            text="未載入檔案",
            font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray65"),
        )
        self._script_path_label.pack(anchor="w", padx=16, pady=(4, 8))
        self._script_text = ctk.CTkTextbox(box, font=ctk.CTkFont(size=14), wrap="word")
        self._script_text.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self._bind_script_text_cache_sync()
        self._script_controls.extend(
            [b_open, b_save, b_sas, b_clear, self._record_btn, self._script_text]
        )

    def _build_actions_row(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(12, 8))
        self._use_tool_cache_checkbox = ctk.CTkCheckBox(
            row,
            text="使用快取工具（略過 LLM）",
            font=ctk.CTkFont(size=13),
            command=self._on_use_tool_cache_changed,
        )
        self._use_tool_cache_checkbox.pack(pady=(0, 10))
        if self._use_tool_cache:
            self._use_tool_cache_checkbox.select()
        btn_row = ctk.CTkFrame(row, fg_color="transparent")
        btn_row.pack()
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(2, weight=1)
        self._run_btn = ctk.CTkButton(
            btn_row,
            text="開始執行",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=200,
            command=self._on_start_run,
        )
        self._run_btn.grid(row=0, column=1)
        self._analysis_progress_frame = ctk.CTkFrame(row, fg_color="transparent")
        self._analysis_progress = ctk.CTkProgressBar(
            self._analysis_progress_frame,
            width=420,
            height=14,
        )
        self._analysis_progress.pack(fill="x")
        self._analysis_progress.set(0)
        self._analysis_progress_frame.pack(fill="x", pady=(20, 0))
        self._analysis_progress_frame.pack_forget()

    def _on_use_tool_cache_changed(self) -> None:
        self._use_tool_cache = self._use_tool_cache_checkbox.get() == 1
        self._persist_hub_ui_state()

    def _tool_cache_enabled_for_run(self) -> bool:
        return self._use_tool_cache_checkbox.get() == 1

    def _set_run_button_idle(self) -> None:
        self._run_btn.configure(text="開始執行", command=self._on_start_run, state="normal")

    def _set_run_button_running(self) -> None:
        self._run_btn.configure(text="停止執行", command=self._on_stop_run, state="normal")

    def _build_status(self) -> None:
        self._status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13))
        self._status.pack(anchor="w", padx=28, pady=(0, 16))

    def _is_analysis_running(self) -> bool:
        thread = self._recording_analysis_thread
        return thread is not None and thread.is_alive()

    def _show_analysis_progress(self) -> None:
        if self._analysis_progress_frame is not None:
            self._analysis_progress_frame.pack(fill="x", pady=(20, 0))
        if self._analysis_progress is not None:
            self._analysis_progress.set(0)

    def _hide_analysis_progress(self) -> None:
        if self._analysis_progress_frame is not None:
            self._analysis_progress_frame.pack_forget()
        if self._analysis_progress is not None:
            self._analysis_progress.set(0)

    def _set_hub_controls_idle(self) -> None:
        self._run_btn.configure(state="normal")
        self._settings_btn.configure(state="normal")
        for cb in self._monitor_checkboxes:
            cb.configure(state="normal")
        self._monitor_refresh_btn.configure(state="normal")
        for w in self._script_controls:
            w.configure(state="normal")
        self._use_tool_cache_checkbox.configure(state="normal")
        if self._record_btn is not None:
            self._record_btn.configure(text="開始錄製", state="normal", command=self._on_record_button)
        self._hide_analysis_progress()

    def _set_hub_controls_recording(self) -> None:
        self._run_btn.configure(state="disabled")
        self._settings_btn.configure(state="disabled")
        for cb in self._monitor_checkboxes:
            cb.configure(state="disabled")
        self._monitor_refresh_btn.configure(state="disabled")
        for w in self._script_controls:
            if w is self._record_btn:
                continue
            w.configure(state="disabled")
        self._use_tool_cache_checkbox.configure(state="disabled")
        if self._record_btn is not None:
            self._record_btn.configure(text="停止錄製", state="normal", command=self._on_record_button)
        self._hide_analysis_progress()

    def _set_hub_controls_analyzing(self) -> None:
        self._analysis_cancel_event.clear()
        self._run_btn.configure(state="disabled")
        self._settings_btn.configure(state="disabled")
        for cb in self._monitor_checkboxes:
            cb.configure(state="disabled")
        self._monitor_refresh_btn.configure(state="disabled")
        for w in self._script_controls:
            if w is self._record_btn:
                continue
            w.configure(state="disabled")
        self._use_tool_cache_checkbox.configure(state="disabled")
        if self._record_btn is not None:
            self._record_btn.configure(text="停止分析", state="normal", command=self._on_record_button)
        self._show_analysis_progress()

    def _update_analysis_progress(self, current: int, total: int) -> None:
        if self._analysis_progress is not None and total > 0:
            self._analysis_progress.set(current / total)
        settings = load_settings()
        self._status.configure(
            text=f"分析錄製中 ({current}/{total})… {settings.brain_lm}",
            text_color=("gray20", "gray65"),
        )

    def _request_cancel_analysis(self) -> None:
        if not self._is_analysis_running():
            return
        self._analysis_cancel_event.set()
        if self._record_btn is not None:
            self._record_btn.configure(state="disabled")
        self._status.configure(text="正在停止分析…", text_color=("gray20", "gray65"))

    def _script_open(self) -> None:
        initial = ROOT_DIR / "scripts"
        path = filedialog.askopenfilename(
            parent=self,
            title="開啟腳本",
            initialdir=str(initial) if initial.is_dir() else str(ROOT_DIR),
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        self._script_path = p
        self._runtime_commands_cache_path = None
        text = p.read_text(encoding="utf-8")
        self._suppress_script_cache_sync = True
        try:
            self._script_text.delete("0.0", "end")
            self._script_text.insert("0.0", text)
        finally:
            self._suppress_script_cache_sync = False
        self._refresh_script_path_label()
        self._persist_hub_ui_state()

    def _script_save(self) -> None:
        if self._script_path is not None:
            body = self._script_text.get("0.0", "end").rstrip() + "\n"
            self._script_path.write_text(body, encoding="utf-8")
            self._status.configure(text=f"已儲存 {self._script_path.name}")
            self._persist_hub_ui_state()
            return
        if self._sync_script_text_to_runtime_cache():
            self._status.configure(text="已儲存執行命令")
            return
        self._script_save_as()

    def _script_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="腳本另存新檔",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
            initialdir=str(ROOT_DIR / "scripts"),
        )
        if not path:
            return
        p = Path(path)
        p.write_text(self._script_text.get("0.0", "end").rstrip() + "\n", encoding="utf-8")
        self._script_path = p
        self._runtime_commands_cache_path = None
        self._refresh_script_path_label()
        self._status.configure(text=f"已另存新檔 {p.name}")
        self._persist_hub_ui_state()

    def _script_clear(self) -> None:
        """Unload any opened path / cache binding and empty the script editor."""
        self._script_path = None
        self._runtime_commands_cache_path = None
        self._script_text.configure(state="normal")
        self._suppress_script_cache_sync = True
        try:
            self._script_text.delete("0.0", "end")
        finally:
            self._suppress_script_cache_sync = False
        self._refresh_script_path_label()
        self._status.configure(text="", text_color=("gray20", "gray65"))
        self._persist_hub_ui_state()

    def _on_stop_run(self) -> None:
        from main import request_coordinator_cancel

        self._user_requested_stop = True
        self._status.configure(text="正在停止…")
        if self._bridge is not None:
            self._bridge.request_stop()
        if not request_coordinator_cancel():
            self._stop_cancel_remaining = 30
            self.after(50, self._try_coordinator_cancel)

    def _try_coordinator_cancel(self) -> None:
        from main import request_coordinator_cancel

        if self._worker_thread is None or not self._worker_thread.is_alive():
            return
        if request_coordinator_cancel():
            return
        self._stop_cancel_remaining -= 1
        if self._stop_cancel_remaining > 0:
            self.after(50, self._try_coordinator_cancel)

    def _on_close(self) -> None:
        if self._is_analysis_running():
            self._analysis_cancel_event.set()
        if self._recording_session.is_active():
            self._stop_recording(analyze=False)
        self._recording_hotkey.unregister()
        self.destroy()

    def _hub_ignore_rect(self) -> tuple[int, int, int, int] | None:
        try:
            if not self.winfo_viewable():
                return None
        except Exception:
            return None
        self.update_idletasks()
        width = int(self.winfo_width())
        height = int(self.winfo_height())
        if width <= 0 or height <= 0:
            return None
        return (
            int(self.winfo_rootx()),
            int(self.winfo_rooty()),
            width,
            height,
        )

    def _recording_ignore_rect_provider(self) -> tuple[int, int, int, int] | None:
        return self._hub_ignore_rect()

    def _schedule_toggle_recording(self) -> None:
        self.after(0, self._toggle_recording)

    def _on_record_button(self) -> None:
        if self._is_analysis_running():
            self._request_cancel_analysis()
            return
        self._toggle_recording()

    def _toggle_recording(self) -> None:
        if self._is_analysis_running():
            return
        if self._worker_thread and self._worker_thread.is_alive():
            show_ctk_message(self, "錄製", "請先停止執行再開始錄製。", kind="warning")
            return
        if self._recording_session.is_active():
            self._stop_recording(analyze=True)
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if self._recording_session.is_active():
            return
        try:
            self._recording_session.set_suppress_hotkey_keys(True)
            run_dir = self._recording_session.start(
                ignore_rect_provider=self._recording_ignore_rect_provider,
            )
        except Exception as exc:
            self._recording_session.set_suppress_hotkey_keys(False)
            show_ctk_message(self, "錄製", f"無法開始錄製：{exc}", kind="error")
            return
        finally:
            self.after(400, lambda: self._recording_session.set_suppress_hotkey_keys(False))

        self._set_hub_controls_recording()
        self._status.configure(
            text=f"錄製中 (0 個事件)… {run_dir.name}",
            text_color=("gray20", "gray65"),
        )
        try:
            self.iconify()
        except Exception:
            pass

    def _on_recording_event(self) -> None:
        count = self._recording_session.event_count()
        run_dir = self._recording_session.run_dir()
        name = run_dir.name if run_dir is not None else ""
        self.after(
            0,
            lambda: self._status.configure(
                text=f"錄製中 ({count} 個事件)… {name}",
                text_color=("gray20", "gray65"),
            ),
        )

    def _stop_recording(self, *, analyze: bool) -> None:
        if not self._recording_session.is_active():
            return
        run_dir = self._recording_session.stop()
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass
        if run_dir is None:
            self._set_hub_controls_idle()
            self._status.configure(text="錄製已停止。")
            return
        event_count = self._recording_session.event_count()
        if analyze and event_count > 0:
            self._set_hub_controls_analyzing()
            settings = load_settings()
            self._status.configure(
                text=f"分析錄製中 (0/{event_count})… {settings.brain_lm}",
                text_color=("gray20", "gray65"),
            )
            self._update_analysis_progress(0, event_count)
            self._recording_analysis_thread = threading.Thread(
                target=self._analyze_recording_worker,
                args=(run_dir,),
                daemon=True,
            )
            self._recording_analysis_thread.start()
        else:
            self._set_hub_controls_idle()
            self._status.configure(text=f"錄製已停止（{event_count} 個事件）。")

    def _analyze_recording_worker(self, run_dir: Path) -> None:
        def on_progress(current: int, total: int) -> None:
            self.after(0, lambda c=current, t=total: self._update_analysis_progress(c, t))

        def should_cancel() -> bool:
            return self._analysis_cancel_event.is_set()

        try:
            report = analyze_screen_recording(
                run_dir,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
        except Exception as exc:
            self.after(0, lambda: self._set_hub_controls_idle())
            self.after(0, lambda: setattr(self, "_recording_analysis_thread", None))
            self.after(
                0,
                lambda: show_ctk_message(
                    self,
                    "錄製分析",
                    f"分析失敗：{exc}",
                    kind="error",
                ),
            )
            self.after(0, lambda: self._status.configure(text="錄製分析失敗。"))
            return
        self.after(0, lambda: self._on_recording_analysis_done(report))

    def _on_recording_analysis_done(self, report: dict[str, Any]) -> None:
        self._recording_analysis_thread = None
        self._set_hub_controls_idle()
        cached = int(report.get("cached", 0))
        skipped = int(report.get("skipped", 0))
        recorded = int(report.get("recorded", 0))
        cancelled = bool(report.get("cancelled", False))
        instructions = report.get("instructions")
        lines: list[str] = []
        if isinstance(instructions, list):
            lines = [str(x) for x in instructions if str(x).strip()]
        if cancelled:
            processed = int(report.get("processed", 0))
            msg = (
                f"分析已停止（已完成 {processed}/{recorded} 個事件）。\n"
                f"已寫入快取 {cached} 筆，略過 {skipped} 筆。"
            )
            self._status.configure(text=f"分析已停止（{processed}/{recorded}）。")
            show_ctk_message(self, "錄製分析已停止", msg, kind="warning")
            return
        msg = (
            f"錄製 {recorded} 個事件。\n"
            f"已寫入快取 {cached} 筆，略過 {skipped} 筆。"
        )
        self._status.configure(text=f"已寫入快取 {cached} 筆（略過 {skipped}）。")
        if lines and prompt_append_recording_instructions(self, msg):
            self._script_text.configure(state="normal")
            if self._script_text.get("0.0", "end").strip():
                self._script_text.insert("end", "\n")
            self._script_text.insert("end", "\n".join(lines) + "\n")
            self._sync_script_text_to_runtime_cache()
        else:
            show_ctk_message(self, "錄製分析完成", msg, kind="info")

    def _begin_worker_run(self, args: _WorkerArgs) -> None:
        self._set_run_button_running()
        self._settings_btn.configure(state="disabled")
        for cb in self._monitor_checkboxes:
            cb.configure(state="disabled")
        self._monitor_refresh_btn.configure(state="disabled")
        for w in self._script_controls:
            w.configure(state="disabled")
        self._use_tool_cache_checkbox.configure(state="disabled")
        self._status.configure(text="執行中…")
        self._worker_thread = threading.Thread(target=self._worker_main, args=(args,), daemon=True)
        self._worker_thread.start()
        self.after(80, self._poll_worker_finished)
        self.after_idle(self.iconify)

    def _runtime_command_transcript_path(self) -> Path:
        settings = load_settings()
        runs_root = Path(settings.runs_dir)
        runs_root.mkdir(parents=True, exist_ok=True)
        return runs_root / _RUNTIME_COMMAND_TRANSCRIPT_NAME

    def _start_runtime_after_script(self, eye_indices: list[int]) -> None:
        """After a script run completes, enter runtime step mode and append to the open script or cache."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if self._script_path is not None:
            transcript_path = self._script_path
            on_undo = self._pop_last_runtime_command_from_script_file
        else:
            if self._runtime_commands_cache_path is None:
                self._runtime_commands_cache_path = self._runtime_command_transcript_path()
            transcript_path = self._runtime_commands_cache_path
            on_undo = self._pop_last_runtime_command_from_cache

        self._user_requested_stop = False
        self._post_run_unlink = None
        self._last_run_was_script_mode = False

        def on_runtime_command(cmd: str) -> None:
            append_text(transcript_path, cmd + "\n")
            self._append_runtime_command_to_script_view(cmd)

        self._bridge = RuntimeCommandHubBridge(
            self,
            on_runtime_command=on_runtime_command,
            on_undo_last_runtime_command=on_undo,
        )
        self._bridge.start()
        args = _WorkerArgs(
            step_mode=True,
            eye_monitor_indices=eye_indices,
            script_raw="",
            script_disk_path=None,
            run_folder_name=self._last_script_run_folder,
            use_tool_cache=self._tool_cache_enabled_for_run(),
        )
        self._begin_worker_run(args)

    def _on_start_run(self) -> None:
        if self._recording_session.is_active():
            show_ctk_message(self, "執行", "請先停止錄製再開始執行。", kind="warning")
            return
        if self._recording_analysis_thread and self._recording_analysis_thread.is_alive():
            show_ctk_message(self, "執行", "請等待錄製分析完成，或按「停止分析」。", kind="warning")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._user_requested_stop = False
        self._post_run_unlink = None
        self._last_script_run_folder = None
        eye_indices = self._selected_monitor_indices()
        if not eye_indices:
            show_ctk_message(
                self,
                "顯示器",
                "請至少選擇一台要截取的顯示器。",
                kind="warning",
            )
            return

        raw = self._script_text.get("0.0", "end")
        steps = parse_executable_lines_from_text(raw)

        if steps:
            # Opened script file, cache transcript, or typed commands → run as one script task.
            if self._script_path is not None:
                script_disk_path = self._script_path
            else:
                self._sync_script_text_to_runtime_cache()
                script_disk_path = self._runtime_commands_cache_path
            self._last_run_was_script_mode = True
            self._bridge = None
            args = _WorkerArgs(
                step_mode=False,
                eye_monitor_indices=eye_indices,
                script_raw=raw,
                script_disk_path=script_disk_path,
                use_tool_cache=self._tool_cache_enabled_for_run(),
            )
        else:
            # Empty script box → interactive step-by-step runtime commands.
            self._last_run_was_script_mode = False
            cache_path = self._runtime_command_transcript_path()
            self._runtime_commands_cache_path = cache_path
            cache_path.write_text("", encoding="utf-8")
            self._script_text.configure(state="normal")
            self._suppress_script_cache_sync = True
            try:
                self._script_text.delete("0.0", "end")
            finally:
                self._suppress_script_cache_sync = False
            self._refresh_script_path_label()

            args = _WorkerArgs(
                step_mode=True,
                eye_monitor_indices=eye_indices,
                script_raw="",
                script_disk_path=None,
                use_tool_cache=self._tool_cache_enabled_for_run(),
            )

            def on_runtime_command(cmd: str) -> None:
                append_text(cache_path, cmd + "\n")
                self._append_runtime_command_to_script_view(cmd)

            self._bridge = RuntimeCommandHubBridge(
                self,
                on_runtime_command=on_runtime_command,
                on_undo_last_runtime_command=self._pop_last_runtime_command_from_cache,
            )
            self._bridge.start()

        self._begin_worker_run(args)

    def _resolve_session_end_reason(self, *, kind: str, user_stopped: bool) -> str:
        if user_stopped:
            return "user_stopped"
        if kind == "err":
            return "error"
        try:
            reason = get_run_state_manager().session_end_reason
            if reason:
                return reason
        except RuntimeError:
            pass
        return "completed"

    def _maybe_write_session_report(
        self,
        *,
        script_finished: bool,
        user_continues_runtime: bool,
        session_end_reason: str,
    ) -> None:
        if not should_write_session_report(
            script_finished=script_finished,
            user_continues_runtime=user_continues_runtime,
        ):
            return
        run_root = self._active_run_root
        if run_root is None:
            return
        report_path = write_session_report(run_root, session_end_reason=session_end_reason)
        try:
            get_run_state_manager().log_info(f"Report written to {report_path}")
        except RuntimeError:
            pass
        self._active_run_root = None

    def _worker_main(self, args: _WorkerArgs) -> None:
        if args.use_tool_cache:
            os.environ[USE_TOOL_CACHE_ENV] = "1"
        else:
            os.environ.pop(USE_TOOL_CACHE_ENV, None)
        try:
            if args.step_mode:
                reset_runtime_user_ended_at_prompt()
                settings = load_settings()
                runs_root = Path(settings.runs_dir)
                folder_name = args.run_folder_name or unique_run_folder_name("runtime_command")
                manager, paths, run_id = prepare_run_session(
                    runs_root=runs_root,
                    task="runtime_command",
                    runtime_mode=True,
                    selected_script_path=None,
                    script_steps=None,
                    eye_monitor_indices=args.eye_monitor_indices,
                    clear_runs_root=False,
                    run_folder_name=folder_name,
                )
                self._active_run_root = paths.root
                manager.log_info("Master starting coordinator module runtime")
                run_coordinator_sync()
                if consume_runtime_user_ended_at_prompt():
                    self._worker_outcome = ("ok_quiet", "")
                else:
                    self._worker_outcome = ("ok", f"執行 {run_id} 已完成。")
                manager.log_info("Master stopped.")
            else:
                script_path = args.script_disk_path
                raw = args.script_raw
                steps = parse_executable_lines_from_text(raw)
                if script_path is None:
                    fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="qa_script_", text=True)
                    os.close(fd)
                    script_path = Path(tmp)
                    self._post_run_unlink = script_path
                script_path.write_text(raw.rstrip() + "\n", encoding="utf-8")
                task = steps[0]
                settings = load_settings()
                runs_root = Path(settings.runs_dir)
                manager, paths, run_id = prepare_run_session(
                    runs_root=runs_root,
                    task=task,
                    runtime_mode=False,
                    selected_script_path=script_path,
                    script_steps=steps,
                    eye_monitor_indices=args.eye_monitor_indices,
                    clear_runs_root=False,
                    run_folder_name=None,
                )
                self._active_run_root = paths.root
                manager.log_info("Master starting coordinator module runtime")
                run_coordinator_sync()
                self._last_script_run_folder = run_id
                self._worker_outcome = ("ok", f"執行 {run_id} 已完成。")
                manager.log_info("Master stopped.")
        except asyncio.CancelledError:
            self._worker_outcome = ("ok_quiet", "")
        except BaseException as e:
            self._worker_outcome = ("err", str(e))

    def _poll_worker_finished(self) -> None:
        if self._worker_thread is None:
            return
        if self._worker_thread.is_alive():
            self.after(80, self._poll_worker_finished)
            return
        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None
        if self._post_run_unlink is not None:
            try:
                self._post_run_unlink.unlink(missing_ok=True)
            except OSError:
                pass
            self._post_run_unlink = None
        user_stopped = self._user_requested_stop
        self._user_requested_stop = False
        kind, msg = self._worker_outcome
        script_finished = self._last_run_was_script_mode and kind == "ok"
        self._last_run_was_script_mode = False
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass
        self._set_run_button_idle()
        self._settings_btn.configure(state="normal")
        for cb in self._monitor_checkboxes:
            cb.configure(state="normal")
        self._monitor_refresh_btn.configure(state="normal")
        for w in self._script_controls:
            w.configure(state="normal")
        self._use_tool_cache_checkbox.configure(state="normal")
        self._refresh_runtime_script_text_from_cache()
        if kind == "err":
            self._status.configure(text=f"錯誤：{msg}")
        elif user_stopped:
            self._status.configure(text="執行已停止。")
        elif script_finished:
            pass
        elif kind == "ok_quiet" and msg.strip():
            self._status.configure(text=msg.strip())
        elif kind in ("ok", "ok_quiet"):
            self._status.configure(text="就緒")
        session_end_reason = self._resolve_session_end_reason(kind=kind, user_stopped=user_stopped)
        if user_stopped and kind != "err":
            self._maybe_write_session_report(
                script_finished=script_finished,
                user_continues_runtime=False,
                session_end_reason=session_end_reason,
            )
            return
        if kind == "err":
            self._maybe_write_session_report(
                script_finished=script_finished,
                user_continues_runtime=False,
                session_end_reason=session_end_reason,
            )
            show_ctk_message(self, "電腦使用代理", msg, kind="error")
            return
        if script_finished:
            if prompt_script_continue_or_end(self, msg):
                eye_indices = self._selected_monitor_indices()
                if not eye_indices:
                    show_ctk_message(
                        self,
                        "顯示器",
                        "請至少選擇一台要截取的顯示器。",
                        kind="warning",
                    )
                    self._status.configure(text=msg.strip() or "就緒")
                    return
                self._start_runtime_after_script(eye_indices)
            else:
                self._maybe_write_session_report(
                    script_finished=True,
                    user_continues_runtime=False,
                    session_end_reason=session_end_reason,
                )
                self._last_script_run_folder = None
                self._status.configure(text=msg.strip() or "就緒")
            return
        self._maybe_write_session_report(
            script_finished=False,
            user_continues_runtime=False,
            session_end_reason=session_end_reason,
        )
        if kind == "ok":
            show_ctk_message(self, "電腦使用代理", msg, kind="info")


def run_main_hub() -> None:
    app = MainHub()
    app.update_idletasks()
    dismiss_nuitka_onefile_splash()
    app.mainloop()


if __name__ == "__main__":
    run_main_hub()
