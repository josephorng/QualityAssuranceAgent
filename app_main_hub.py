"""CustomTkinter hub: runs from an opened script file or step-by-step when no file is set."""

from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk

from main import analyze_screen_recording, dismiss_nuitka_onefile_splash, prepare_run_session, run_coordinator_sync
from src.common.agent_settings_dialog import clamp_script_font_size, open_agent_settings_dialog
from src.common.ctk_dialogs import (
    prompt_append_recording_instructions,
    prompt_script_continue_or_end,
    prompt_unsaved_script_changes,
    show_ctk_message,
)
from src.common.folder_dialogs import ask_directories
from src.common.io_utils import append_text, pop_last_nonempty_line, read_json, write_json
from src.common.run_control import (
    pause_run,
    reset_run_control,
    resume_run,
    set_step_status_callback,
    wait_while_paused_blocking,
)
from src.common.run_state import (
    get_run_state_manager,
    unique_named_run_folder,
    unique_run_folder_name,
)
from src.common.session_html import write_runs_index_html
from src.common.runs_report_server import (
    ensure_runs_report_server,
    rename_recording_folder,
    stop_runs_report_server,
)
from src.common.session_report import should_write_session_report, write_session_report
from src.common.runtime_command_dialog import (
    RuntimeCommandHubBridge,
    consume_runtime_user_ended_at_prompt,
    reset_runtime_user_ended_at_prompt,
)
from src.common.runtime_context import USE_TOOL_CACHE_ENV
from src.common.script_helper import (
    collect_recording_script_text,
    executable_source_line_numbers,
    format_script_lines_with_outcomes,
    is_recording_script_path,
    is_runnable_script_path,
    load_runnable_script_text,
    parse_executable_lines_from_text,
    partition_recording_dirs,
    recording_run_dir,
    resolve_runnable_script_path,
    script_display_name,
)
from src.common.smart_mode import normalize_smart_goal, resolve_hub_run_mode
from src.common.settings import (
    ROOT_DIR,
    apply_startup_ollama_host_probe,
    apply_startup_triton_probe,
    load_settings,
    reports_serve_root,
)
from src.recorder.capture import RecordingSession
from src.recorder.hotkey import RecordingHotkeyManager
from src.recorder.models import RecordedEvent
from src.recorder.vision_prefetch import VisionPrefetchWorker

# Step-mode runtime command transcript (append during run); not hub UI preferences.
_RUNTIME_COMMAND_TRANSCRIPT_NAME = "runtime_commands_cache.txt"
_RUNTIME_COMMAND_LABEL = "逐步執行命令"
_SMART_GOAL_CACHE_NAME = "smart_goal_cache.txt"
_SMART_GOAL_CACHE_LABEL = "智能模式目標暫存"
_HUB_UI_STATE_NAME = "hub_ui.json"
_HUB_UI_VERSION = 3
_MODE_TAB_SINGLE = "單一腳本"
_MODE_TAB_QUEUE = "佇列執行"
_MODE_TAB_SMART = "智能模式"

_RECORDING_FOLDER_RENAME_ERRORS = {
    "invalid run name": "資料夾名稱無效（不可為空白，也不可含 \\ / : * ? \" < > |）。",
    "a folder with that name already exists": "已有相同名稱的資料夾。",
    "run folder not found": "找不到錄製資料夾。",
    "invalid run id": "找不到錄製資料夾。",
    "name must be a string": "資料夾名稱無效。",
}


def _recording_folder_rename_error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return _RECORDING_FOLDER_RENAME_ERRORS.get(text, f"無法重新命名資料夾：{text}")


def _resolved_path_key(path: Path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _path_is_recording_folder(path: Path, folder_key: str) -> bool:
    """True when ``path`` is the recording folder (or legacy script.txt inside it)."""
    key = _resolved_path_key(path)
    if key == folder_key:
        return True
    resolved = Path(key)
    return resolved.name == "script.txt" and str(resolved.parent) == folder_key


def _default_hub_ui_dict() -> dict[str, Any]:
    return {
        "version": _HUB_UI_VERSION,
        "appearance_dark": True,
        "script_font_size": 14,
        "selected_monitor_indices": [],
        "last_script_path": None,
        "last_smart_goal_path": None,
        "selected_mode": _MODE_TAB_SINGLE,
        "use_tool_cache": False,
        "recording_hotkey_enabled": True,
        "queue_script_paths": [],
    }


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str) and x.strip()]


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
    if "script_font_size" in raw:
        base["script_font_size"] = clamp_script_font_size(raw.get("script_font_size"))
    elif "ui_scale" in raw:
        # Migrate legacy whole-app scale into editor point size (base 14).
        try:
            legacy_scale = float(raw.get("ui_scale") or 1.0)
        except (TypeError, ValueError):
            legacy_scale = 1.0
        base["script_font_size"] = clamp_script_font_size(14 * legacy_scale)
    else:
        base["script_font_size"] = 14
    base["selected_monitor_indices"] = _coerce_int_list(raw.get("selected_monitor_indices"))
    lsp = raw.get("last_script_path")
    base["last_script_path"] = lsp if isinstance(lsp, str) or lsp is None else None
    lsg = raw.get("last_smart_goal_path")
    base["last_smart_goal_path"] = lsg if isinstance(lsg, str) or lsg is None else None
    selected_mode = raw.get("selected_mode")
    if selected_mode in (_MODE_TAB_SINGLE, _MODE_TAB_QUEUE, _MODE_TAB_SMART):
        base["selected_mode"] = selected_mode
    base["use_tool_cache"] = bool(raw.get("use_tool_cache", False))
    base["recording_hotkey_enabled"] = bool(raw.get("recording_hotkey_enabled", True))
    base["queue_script_paths"] = _coerce_str_list(raw.get("queue_script_paths"))
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
    run_mode: str
    eye_monitor_indices: list[int]
    script_raw: str
    script_disk_path: Path | None
    run_folder_name: str | None = None
    use_tool_cache: bool = False
    queue_paths: list[Path] | None = None
    # Original UI indices aligned with queue_paths (for mid-queue starts / status icons).
    queue_path_indices: list[int] | None = None
    smart_goal: str = ""

    @property
    def step_mode(self) -> bool:
        """Backward-compatible alias for runtime command mode."""
        return self.run_mode == "runtime"


class MainHub(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("電腦使用代理")
        self.geometry("960x620")
        self.minsize(880, 680)

        hub = _read_hub_ui_state()
        self._remember_monitor_indices: list[int] = list(hub["selected_monitor_indices"])
        self._appearance_dark = bool(hub["appearance_dark"])
        self._script_font_size = clamp_script_font_size(hub.get("script_font_size", 14))
        self._use_tool_cache = bool(hub.get("use_tool_cache", False))
        self._recording_hotkey_enabled = bool(hub.get("recording_hotkey_enabled", True))
        ctk.set_appearance_mode("dark" if self._appearance_dark else "light")
        ctk.set_default_color_theme("dark-blue")
        # Undo any previously saved whole-app widget scaling from the old setting.
        ctk.set_widget_scaling(1.0)

        self._script_path: Path | None = None
        self._smart_goal_path: Path | None = None
        # When set, Save writes here (step-mode transcript under runs_dir); not a user-opened script.
        self._runtime_commands_cache_path: Path | None = None
        self._worker_thread: threading.Thread | None = None
        self._bridge: RuntimeCommandHubBridge | None = None
        self._worker_outcome: tuple[str, str] = ("ok", "")
        self._user_requested_stop = False
        self._stop_cancel_remaining = 0


        self._post_run_unlink: Path | None = None
        self._script_controls: list[Any] = []
        self._smart_controls: list[Any] = []
        self._queue_controls: list[Any] = []
        self._queue_item_controls: list[Any] = []
        self._queue_paths: list[Path] = []
        self._queue_selected: int | None = None
        self._queue_mode_active = False
        self._queue_results: list[tuple[str, str]] = []
        self._queue_status_by_index: dict[int, str] = {}
        self._queue_run_root_by_index: dict[int, Path] = {}
        self._script_step_status_by_line: dict[int, str] = {}
        self._script_step_line_numbers: list[int] = []
        self._script_step_status_baseline = "\n"
        self._last_report_html: Path | None = None
        self._last_run_was_script_mode = False
        self._last_script_run_folder: str | None = None
        self._active_run_root: Path | None = None
        self._smart_baseline = "\n"

        self._recording_session = RecordingSession()
        self._recording_session.set_on_event(self._on_recording_event)
        self._recording_hotkey = RecordingHotkeyManager()
        self._recording_analysis_thread: threading.Thread | None = None
        self._recording_finalize_thread: threading.Thread | None = None
        self._vision_prefetch = VisionPrefetchWorker()
        self._analysis_cancel_event = threading.Event()
        self._suppress_script_cache_sync = False
        self._sync_cache_after_id: str | None = None
        self._suppress_smart_cache_sync = False
        self._smart_sync_cache_after_id: str | None = None
        self._script_baseline = "\n"
        self._record_btn: ctk.CTkButton | None = None
        self._recording_report_btn: ctk.CTkButton | None = None
        self._analyze_recording_btn: ctk.CTkButton | None = None
        self._analysis_progress_frame: ctk.CTkFrame | None = None
        self._analysis_progress: ctk.CTkProgressBar | None = None

        self._build_header()
        # Pack bottom chrome first so the expanding script section cannot clip it
        # when the window is restored (not maximized).
        self._build_status()
        self._build_actions_row()
        self._build_script_section()
        last_script = hub.get("last_script_path")
        if isinstance(last_script, str) and last_script.strip():
            p = resolve_runnable_script_path(Path(last_script))
            if is_runnable_script_path(p):
                self._load_script_into_editor(p)
        if self._script_path is None:
            self._try_load_last_runtime_command_cache()
        self._mark_script_clean()
        self._apply_script_editor_lock()

        last_smart = hub.get("last_smart_goal_path")
        if isinstance(last_smart, str) and last_smart.strip():
            sp = Path(last_smart)
            if sp.is_file():
                self._smart_goal_path = sp
                self._suppress_smart_cache_sync = True
                try:
                    self._smart_text.delete("0.0", "end")
                    self._smart_text.insert("0.0", sp.read_text(encoding="utf-8"))
                    self._reset_textbox_undo(self._smart_text)
                finally:
                    self._suppress_smart_cache_sync = False
                self._refresh_smart_path_label()
                self._mark_smart_clean()
        if self._smart_goal_path is None:
            self._try_load_smart_goal_cache()

        restored_queue: list[Path] = []
        for raw_path in hub.get("queue_script_paths", []):
            if not str(raw_path).strip():
                continue
            restored_queue.append(resolve_runnable_script_path(Path(raw_path)))
        self._queue_paths = restored_queue
        self._refresh_queue_list()

        selected_mode = hub.get("selected_mode")
        if selected_mode in (_MODE_TAB_SINGLE, _MODE_TAB_QUEUE, _MODE_TAB_SMART):
            try:
                self._mode_tabs.set(selected_mode)
            except Exception:
                pass
        self._sync_tool_cache_checkbox_for_mode()

        self._status.configure(text="正在檢查 Ollama 與 Triton…")
        self._start_startup_probes()

        if self._recording_hotkey_enabled:
            self._recording_hotkey.register(self._schedule_toggle_recording)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_startup_probes(self) -> None:
        results: queue.SimpleQueue[tuple[bool, str, bool, str]] = queue.SimpleQueue()

        def work() -> None:
            ollama_ok, ollama_message = apply_startup_ollama_host_probe()
            triton_ok, triton_message = apply_startup_triton_probe()
            results.put(
                (ollama_ok, ollama_message, triton_ok, triton_message)
            )

        def poll_results() -> None:
            try:
                result = results.get_nowait()
            except queue.Empty:
                self.after(50, poll_results)
                return
            self._on_startup_probes_done(*result)

        threading.Thread(target=work, daemon=True).start()
        self.after(50, poll_results)

    def _on_startup_probes_done(
        self,
        ollama_ok: bool,
        ollama_message: str,
        triton_ok: bool,
        triton_message: str,
    ) -> None:
        message = f"{ollama_message} | {triton_message}"
        if not ollama_ok:
            self._status.configure(text=message, text_color=("#b91c1c", "#f87171"))
        elif not triton_ok:
            self._status.configure(text=message, text_color=("#b45309", "#fbbf24"))
        else:
            self._status.configure(text=message, text_color=("gray20", "gray65"))

    def _start_ollama_host_probe(self) -> None:
        results: queue.SimpleQueue[tuple[bool, str]] = queue.SimpleQueue()

        def work() -> None:
            ok, message = apply_startup_ollama_host_probe()
            results.put((ok, message))

        def poll_results() -> None:
            try:
                result = results.get_nowait()
            except queue.Empty:
                self.after(50, poll_results)
                return
            self._on_ollama_host_probe_done(*result)

        threading.Thread(target=work, daemon=True).start()
        self.after(50, poll_results)

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
            display = script_display_name(self._script_path)
            if is_recording_script_path(self._script_path):
                self._script_path_label.configure(
                    text=f"{display}（錄製，唯讀 — 請在錄製紀錄頁編輯）"
                )
            else:
                self._script_path_label.configure(text=str(self._script_path.resolve()))
        elif self._runtime_commands_cache_path is not None:
            self._script_path_label.configure(text=_RUNTIME_COMMAND_LABEL)
        else:
            self._script_path_label.configure(text="未載入檔案")

    def _is_recording_script_open(self) -> bool:
        return self._script_path is not None and is_recording_script_path(self._script_path)

    def _record_button_idle_label(self) -> str:
        return "繼續錄製" if self._is_recording_script_open() else "開始錄製"

    def _refresh_record_button_idle_label(self) -> None:
        if self._record_btn is None:
            return
        if self._recording_session.is_active() or self._is_recording_finalizing():
            return
        if self._is_analysis_running():
            return
        try:
            self._record_btn.configure(text=self._record_button_idle_label())
        except Exception:
            pass

    def _apply_script_editor_lock(self) -> None:
        """Recording review HTML is the only editor; hub shows a read-only preview."""
        if not hasattr(self, "_script_text"):
            return
        read_only = self._is_recording_script_open()
        self._script_text.configure(state="disabled" if read_only else "normal")
        self._refresh_script_toolbar_for_recording()

    def _refresh_script_toolbar_for_recording(self) -> None:
        """Toggle save / report buttons for recording vs plain script."""
        recording = self._is_recording_script_open()
        clear_btn = getattr(self, "_script_clear_btn", None)
        save_btn = getattr(self, "_script_save_btn", None)
        save_as_btn = getattr(self, "_script_save_as_btn", None)
        report_btn = self._recording_report_btn

        def _pack_before_clear(btn: Any) -> None:
            if btn is None or btn.winfo_ismapped():
                return
            if clear_btn is not None:
                btn.pack(side="left", padx=(0, 8), before=clear_btn)
            else:
                btn.pack(side="left", padx=(0, 8))

        if recording:
            if save_btn is not None:
                save_btn.pack_forget()
            if save_as_btn is not None:
                save_as_btn.pack_forget()
            _pack_before_clear(report_btn)
        else:
            if report_btn is not None:
                report_btn.pack_forget()
            _pack_before_clear(save_btn)
            _pack_before_clear(save_as_btn)

    def _on_open_recording_report(self) -> None:
        if not self._is_recording_script_open() or self._script_path is None:
            return
        rec = recording_run_dir(self._script_path)
        if rec is None:
            return
        self._open_recording_review_html(rec, rec.name)

    def _load_script_into_editor(self, path: Path) -> None:
        """Load a script file or recording folder into the single-script editor."""
        p = resolve_runnable_script_path(Path(path))
        self._script_path = p
        self._runtime_commands_cache_path = None
        try:
            text = load_runnable_script_text(p)
        except OSError:
            text = ""
        self._suppress_script_cache_sync = True
        try:
            self._script_text.configure(state="normal")
            self._script_text.delete("0.0", "end")
            self._script_text.insert("0.0", text)
            self._reset_textbox_undo(self._script_text)
        finally:
            self._suppress_script_cache_sync = False
        self._clear_script_step_statuses()
        self._refresh_script_line_numbers()
        self._refresh_script_path_label()
        self._mark_script_clean()
        self._apply_script_editor_lock()
        self._refresh_record_button_idle_label()
        self._persist_hub_ui_state()

    def _validate_recording_folder_for_script(self, folder: Path) -> Path | None:
        """Return the recording folder if it is a valid recording, else None."""
        run_dir = Path(folder)
        if not self._is_recording_folder(run_dir):
            return None
        return run_dir

    def _ask_recording_directory(self, *, title: str) -> Path | None:
        folders = self._ask_recording_directories(title=title, allow_multiple=False)
        if not folders:
            return None
        return folders[0]

    def _ask_recording_directories(
        self, *, title: str, allow_multiple: bool = True
    ) -> list[Path]:
        settings = load_settings()
        initial = Path(settings.recordings_dir)
        initialdir = str(initial) if initial.is_dir() else str(ROOT_DIR)
        if allow_multiple:
            return ask_directories(parent=self, title=title, initialdir=initialdir)
        folder = filedialog.askdirectory(
            parent=self,
            title=title,
            initialdir=initialdir,
        )
        if not folder:
            return []
        return [Path(folder)]

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
            self._reset_textbox_undo(self._script_text)
        finally:
            self._suppress_script_cache_sync = False
        self._refresh_script_path_label()
        self._mark_script_clean()

    def _append_runtime_command_to_script_view(self, cmd: str) -> None:
        """Underlying Tk Text ignores ``insert`` while the widget is ``disabled`` (as during a run)."""
        self._script_text.configure(state="normal")
        self._script_text.insert("end", cmd + "\n")
        self._script_text.configure(state="disabled")
        self._mark_script_clean()
        # Keep prior step icons; treat the appended runtime line as part of the status baseline.
        self._script_step_status_baseline = self._script_editor_normalized_text()
        self._refresh_script_line_numbers()

    def _pop_last_runtime_command_from_cache(self) -> None:
        p = self._runtime_commands_cache_path
        if p is None:
            return
        pop_last_nonempty_line(p)
        self._script_text.configure(state="normal")
        self._script_text.delete("0.0", "end")
        if p.is_file():
            self._script_text.insert("0.0", p.read_text(encoding="utf-8"))
        self._reset_textbox_undo(self._script_text)
        self._script_text.configure(state="disabled")
        self._mark_script_clean()
        self._prune_script_step_statuses_to_current_text()

    def _pop_last_runtime_command_from_script_file(self) -> None:
        p = self._script_path
        if p is None:
            return
        pop_last_nonempty_line(p)
        self._script_text.configure(state="normal")
        self._script_text.delete("0.0", "end")
        if p.is_file():
            self._script_text.insert("0.0", p.read_text(encoding="utf-8"))
        self._reset_textbox_undo(self._script_text)
        self._script_text.configure(state="disabled")
        self._mark_script_clean()
        self._prune_script_step_statuses_to_current_text()

    def _refresh_runtime_script_text_from_cache(self) -> None:
        """After a runtime-command run, reload the cache file into the script textbox (disk is source of truth)."""
        p = self._runtime_commands_cache_path
        if p is None or not p.is_file():
            return
        self._suppress_script_cache_sync = True
        try:
            self._script_text.delete("0.0", "end")
            self._script_text.insert("0.0", p.read_text(encoding="utf-8"))
            self._reset_textbox_undo(self._script_text)
        finally:
            self._suppress_script_cache_sync = False
        self._refresh_script_path_label()
        self._mark_script_clean()
        self._prune_script_step_statuses_to_current_text()

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
        self._mark_script_clean()
        return True

    def _script_editor_normalized_text(self) -> str:
        return self._script_text.get("0.0", "end").rstrip() + "\n"

    def _mark_script_clean(self) -> None:
        self._script_baseline = self._script_editor_normalized_text()

    def _is_script_dirty(self) -> bool:
        if self._is_recording_script_open():
            return False
        return self._script_editor_normalized_text() != self._script_baseline

    def _confirm_proceed_with_unsaved_script(self) -> bool:
        """If the script editor has unsaved edits, prompt; return True when the action may proceed."""
        if not self._is_script_dirty():
            return True
        choice = prompt_unsaved_script_changes(self)
        if choice == "cancel":
            return False
        if choice == "discard":
            return True
        return self._script_save()

    def _confirm_proceed_before_new_script(self) -> bool:
        """Before clearing the editor (開新檔案), confirm save when content is not on disk."""
        if not self._script_editor_normalized_text().strip():
            return True
        if self._script_path is not None:
            if not self._is_script_dirty():
                return True
            return self._confirm_proceed_with_unsaved_script()
        if self._is_script_dirty():
            message = "腳本尚未存成檔案，且內容已變更。要另存為檔案嗎？"
        else:
            message = "腳本尚未存成檔案（目前僅存在執行命令暫存）。要另存為檔案嗎？"
        choice = prompt_unsaved_script_changes(
            self, message=message, save_button_text="另存新檔"
        )
        if choice == "cancel":
            return False
        if choice == "discard":
            return True
        return self._script_save_as()

    def _on_mode_tab_changed(self) -> None:
        try:
            selected = self._mode_tabs.get()
        except Exception:
            return
        self._persist_hub_ui_state()
        self._sync_tool_cache_checkbox_for_mode()
        if selected != _MODE_TAB_QUEUE:
            return
        if self._confirm_proceed_with_unsaved_script():
            return
        try:
            self._mode_tabs.set(_MODE_TAB_SINGLE)
        except Exception:
            pass

    def _refresh_smart_path_label(self) -> None:
        if self._smart_goal_path is not None:
            self._smart_path_label.configure(text=str(self._smart_goal_path.resolve()))
        elif self._smart_editor_normalized_text().strip():
            self._smart_path_label.configure(text=_SMART_GOAL_CACHE_LABEL)
        else:
            self._smart_path_label.configure(text="未載入目標檔案")

    def _smart_goal_cache_path(self) -> Path:
        return Path(load_settings().runs_dir) / _SMART_GOAL_CACHE_NAME

    def _try_load_smart_goal_cache(self) -> None:
        cache_path = self._smart_goal_cache_path()
        if not cache_path.is_file():
            return
        raw = cache_path.read_text(encoding="utf-8")
        if not raw.strip():
            return
        self._suppress_smart_cache_sync = True
        try:
            self._smart_text.delete("0.0", "end")
            self._smart_text.insert("0.0", raw)
            self._reset_textbox_undo(self._smart_text)
        finally:
            self._suppress_smart_cache_sync = False
        self._refresh_smart_path_label()
        self._mark_smart_clean()

    def _sync_smart_text_to_cache(self) -> bool:
        """Persist an unsaved smart goal under runs_dir."""
        if self._smart_goal_path is not None:
            return False
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return False
        cache_path = self._smart_goal_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(self._smart_editor_normalized_text(), encoding="utf-8")
        self._refresh_smart_path_label()
        self._mark_smart_clean()
        return True

    def _schedule_sync_smart_text_to_cache(self) -> None:
        if self._suppress_smart_cache_sync or self._smart_goal_path is not None:
            return
        if self._smart_sync_cache_after_id is not None:
            self.after_cancel(self._smart_sync_cache_after_id)
        self._smart_sync_cache_after_id = self.after(
            250, self._run_scheduled_sync_smart_text_to_cache
        )

    def _run_scheduled_sync_smart_text_to_cache(self) -> None:
        self._smart_sync_cache_after_id = None
        self._sync_smart_text_to_cache()

    def _on_smart_text_modified(self, _event: object = None) -> None:
        textbox = self._smart_text._textbox
        if textbox.edit_modified():
            textbox.edit_modified(False)
            self._schedule_sync_smart_text_to_cache()

    def _smart_editor_normalized_text(self) -> str:
        return self._smart_text.get("0.0", "end").rstrip() + "\n"

    def _mark_smart_clean(self) -> None:
        self._smart_baseline = self._smart_editor_normalized_text()

    def _is_smart_dirty(self) -> bool:
        return self._smart_editor_normalized_text() != self._smart_baseline

    def _confirm_proceed_with_unsaved_smart(self) -> bool:
        if not self._is_smart_dirty():
            return True
        choice = prompt_unsaved_script_changes(self, message="智能模式目標尚未儲存。要先儲存嗎？")
        if choice == "cancel":
            return False
        if choice == "discard":
            return True
        return self._smart_save()

    def _smart_open(self) -> None:
        if not self._confirm_proceed_with_unsaved_smart():
            return
        initial = ROOT_DIR / "scripts"
        path = filedialog.askopenfilename(
            parent=self,
            title="開啟智能模式目標",
            initialdir=str(initial) if initial.is_dir() else str(ROOT_DIR),
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        self._smart_goal_path = p
        self._suppress_smart_cache_sync = True
        try:
            self._smart_text.delete("0.0", "end")
            self._smart_text.insert("0.0", text)
            self._reset_textbox_undo(self._smart_text)
        finally:
            self._suppress_smart_cache_sync = False
        self._refresh_smart_path_label()
        self._mark_smart_clean()
        self._persist_hub_ui_state()

    def _smart_save(self) -> bool:
        if self._smart_goal_path is None:
            if self._sync_smart_text_to_cache():
                self._status.configure(text="已儲存智能模式目標")
                return True
            return self._smart_save_as()
        body = self._smart_editor_normalized_text()
        self._smart_goal_path.write_text(body, encoding="utf-8")
        self._mark_smart_clean()
        self._persist_hub_ui_state()
        return True

    def _smart_save_as(self) -> bool:
        initial = ROOT_DIR / "scripts"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="另存智能模式目標",
            initialdir=str(initial) if initial.is_dir() else str(ROOT_DIR),
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
        )
        if not path:
            return False
        p = Path(path)
        p.write_text(self._smart_editor_normalized_text(), encoding="utf-8")
        self._smart_goal_path = p
        self._refresh_smart_path_label()
        self._mark_smart_clean()
        self._persist_hub_ui_state()
        return True

    def _confirm_proceed_before_new_smart(self) -> bool:
        """Before clearing the smart editor (開新檔案), confirm save when content is not on disk."""
        if not self._smart_editor_normalized_text().strip():
            return True
        if self._smart_goal_path is not None:
            if not self._is_smart_dirty():
                return True
            return self._confirm_proceed_with_unsaved_smart()
        if self._is_smart_dirty():
            message = "智能模式目標尚未存成檔案，且內容已變更。要另存為檔案嗎？"
        else:
            message = "智能模式目標尚未存成檔案（目前僅存在目標暫存）。要另存為檔案嗎？"
        choice = prompt_unsaved_script_changes(
            self, message=message, save_button_text="另存新檔"
        )
        if choice == "cancel":
            return False
        if choice == "discard":
            return True
        return self._smart_save_as()

    def _smart_clear(self) -> None:
        if not self._confirm_proceed_before_new_smart():
            return
        self._smart_goal_path = None
        self._suppress_smart_cache_sync = True
        try:
            self._smart_text.delete("0.0", "end")
            self._reset_textbox_undo(self._smart_text)
        finally:
            self._suppress_smart_cache_sync = False
        cache_path = self._smart_goal_cache_path()
        if cache_path.is_file():
            cache_path.write_text("", encoding="utf-8")
        self._refresh_smart_path_label()
        self._mark_smart_clean()
        self._persist_hub_ui_state()

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
            # Keep ✓/✗ until the next run, or until the script body actually changes.
            # Re-enabling the editor after a run can fire <<Modified>> without content changes.
            if (
                not self._suppress_script_cache_sync
                and self._script_step_status_by_line
                and self._script_editor_normalized_text() != self._script_step_status_baseline
            ):
                self._clear_script_step_statuses()
            self._refresh_script_line_numbers()
            self._schedule_sync_script_text_to_runtime_cache()

    def _script_line_count(self) -> int:
        """Return the number of logical lines in the script editor (at least 1)."""
        textbox = self._script_text._textbox
        return max(1, int(float(textbox.index("end-1c"))))

    def _clear_script_step_statuses(self) -> None:
        self._script_step_status_by_line = {}
        self._script_step_line_numbers = []
        self._script_step_status_baseline = "\n"

    def _reset_script_step_status_tracking(self, raw: str) -> None:
        """Clear icons and map executable step indices to 1-based source lines for a script run."""
        self._script_step_status_by_line = {}
        self._script_step_line_numbers = executable_source_line_numbers(raw)
        self._script_step_status_baseline = raw.rstrip() + "\n"
        self._refresh_script_line_numbers()

    def _prune_script_step_statuses_to_current_text(self) -> None:
        """Drop status icons for lines that are no longer executable (e.g. after runtime undo)."""
        raw = self._script_text.get("0.0", "end")
        keep = set(executable_source_line_numbers(raw))
        self._script_step_status_by_line = {
            line: status
            for line, status in self._script_step_status_by_line.items()
            if line in keep
        }
        self._script_step_status_baseline = self._script_editor_normalized_text()
        self._refresh_script_line_numbers()

    def _on_step_status_from_worker(self, step_index: int, status: str) -> None:
        """Apply a green check / red cross next to the matching script line (coordinator thread)."""
        if status not in ("ok", "fail"):
            return

        def apply() -> None:
            line: int | None = None
            if self._script_step_line_numbers:
                if 0 <= step_index < len(self._script_step_line_numbers):
                    line = self._script_step_line_numbers[step_index]
            else:
                numbers = executable_source_line_numbers(self._script_text.get("0.0", "end"))
                if numbers:
                    line = numbers[-1]
            if line is None:
                return
            self._script_step_status_by_line[line] = status
            self._refresh_script_line_numbers()

        self.after(0, apply)

    def _refresh_script_line_numbers(self) -> None:
        """Rebuild the read-only gutter so line numbers stay aligned with script text."""
        textbox = self._script_text._textbox
        line_count = self._script_line_count()
        # One number per logical line; blank rows for soft-wrapped display continuations.
        rows: list[str] = []
        status_tags: list[tuple[str, int]] = []
        gutter_row = 1
        for i in range(1, line_count + 1):
            start = f"{i}.0"
            end = f"{i}.end"
            try:
                counted = textbox.count(start, end, "displaylines")
                display_lines = int(counted[0]) + 1 if counted else 1
            except (tk.TclError, TypeError, ValueError, IndexError):
                display_lines = 1
            status = self._script_step_status_by_line.get(i)
            if status == "ok":
                icon = "\u2713"
            elif status == "fail":
                icon = "\u2717"
            else:
                icon = " "
            rows.append(f"{icon} {i}")
            if status in ("ok", "fail"):
                status_tags.append((status, gutter_row))
            rows.extend("" for _ in range(max(0, display_lines - 1)))
            gutter_row += display_lines
        numbers = "\n".join(rows) if rows else "1"
        digit_width = max(2, len(str(line_count)))
        gutter_width = 28 + (digit_width + 2) * 10

        gutter = self._script_line_numbers
        gutter.configure(state="normal", width=gutter_width)
        gutter.delete("0.0", "end")
        gutter.insert("0.0", numbers)
        gutter_tb = gutter._textbox
        gutter_tb.tag_configure("linenum", justify="right")
        appearance = ctk.get_appearance_mode()
        ok_fg = "#22c55e" if appearance == "Dark" else "#16a34a"
        fail_fg = "#f87171" if appearance == "Dark" else "#b91c1c"
        gutter_tb.tag_configure("ok", foreground=ok_fg)
        gutter_tb.tag_configure("fail", foreground=fail_fg)
        gutter_tb.tag_add("linenum", "1.0", "end")
        for status, row in status_tags:
            gutter_tb.tag_add(status, f"{row}.0", f"{row}.1")
        gutter.configure(state="disabled")
        self._sync_script_line_number_scroll()

    def _sync_script_line_number_scroll(self, *_args: object) -> None:
        """Keep the gutter vertically scrolled with the script textbox."""
        try:
            first, _last = self._script_text._textbox.yview()
            self._script_line_numbers._textbox.yview_moveto(first)
        except tk.TclError:
            return

    def _on_script_text_configured(self, *_args: object) -> None:
        """Recompute gutter rows when wrap width changes, then re-sync scroll."""
        try:
            width = int(self._script_text._textbox.winfo_width())
        except (tk.TclError, TypeError, ValueError):
            self._sync_script_line_number_scroll()
            return
        if getattr(self, "_script_text_last_wrap_width", None) == width:
            self._sync_script_line_number_scroll()
            return
        self._script_text_last_wrap_width = width
        self._refresh_script_line_numbers()

    @staticmethod
    def _reset_textbox_undo(textbox: ctk.CTkTextbox) -> None:
        """Clear undo/redo stacks after programmatic content replacement."""
        try:
            textbox.edit_reset()
        except tk.TclError:
            return

    def _bind_textbox_undo_redo(self, textbox: ctk.CTkTextbox) -> None:
        """Enable Ctrl+Z undo and Ctrl+Shift+Z redo while the editor is editable."""
        tb = textbox._textbox

        def _undo(_event: object = None) -> str:
            if str(tb.cget("state")) != "normal":
                return "break"
            try:
                tb.edit_undo()
            except tk.TclError:
                pass
            return "break"

        def _redo(_event: object = None) -> str:
            if str(tb.cget("state")) != "normal":
                return "break"
            try:
                tb.edit_redo()
            except tk.TclError:
                pass
            return "break"

        # Windows: Ctrl+Z → undo; Ctrl+Shift+Z → redo (Tk default redo is Ctrl+Y only).
        tb.bind("<Control-z>", _undo)
        tb.bind("<Control-Shift-z>", _redo)
        tb.bind("<Control-Shift-Z>", _redo)

    def _bind_script_text_cache_sync(self) -> None:
        textbox = self._script_text._textbox
        textbox.bind("<<Modified>>", self._on_script_text_modified)
        textbox.bind(
            "<MouseWheel>",
            lambda _e: self.after_idle(self._sync_script_line_number_scroll),
            add="+",
        )
        textbox.bind(
            "<Button-4>",
            lambda _e: self.after_idle(self._sync_script_line_number_scroll),
            add="+",
        )
        textbox.bind(
            "<Button-5>",
            lambda _e: self.after_idle(self._sync_script_line_number_scroll),
            add="+",
        )
        textbox.bind(
            "<KeyRelease>",
            lambda _e: self.after_idle(self._sync_script_line_number_scroll),
            add="+",
        )
        textbox.bind(
            "<ButtonRelease-1>",
            lambda _e: self.after_idle(self._sync_script_line_number_scroll),
            add="+",
        )
        textbox.bind(
            "<Configure>",
            lambda _e: self.after_idle(self._on_script_text_configured),
            add="+",
        )

        gutter = self._script_line_numbers._textbox
        gutter.configure(takefocus=0, cursor="arrow")
        gutter.bind("<MouseWheel>", self._forward_script_scroll_from_gutter)
        gutter.bind("<Button-4>", self._forward_script_scroll_from_gutter)
        gutter.bind("<Button-5>", self._forward_script_scroll_from_gutter)
        gutter.bind("<Key>", lambda _e: "break")
        gutter.bind("<Button-1>", lambda _e: "break")

        self._install_script_yscroll_hook()
        self._refresh_script_line_numbers()

    def _forward_script_scroll_from_gutter(self, event: Any) -> str:
        textbox = self._script_text._textbox
        delta = getattr(event, "delta", 0)
        if delta:
            textbox.yview_scroll(int(-1 * (delta / 120)), "units")
        else:
            num = getattr(event, "num", 0)
            if num == 4:
                textbox.yview_scroll(-1, "units")
            elif num == 5:
                textbox.yview_scroll(1, "units")
        self._sync_script_line_number_scroll()
        return "break"

    def _install_script_yscroll_hook(self) -> None:
        """Attach yscrollcommand so gutter tracks CTkTextbox's built-in scrollbar."""
        textbox = self._script_text._textbox
        y_scrollbar = self._script_text._y_scrollbar

        def yscroll_set(first: str, last: str) -> None:
            y_scrollbar.set(first, last)
            self._sync_script_line_number_scroll()

        textbox.configure(yscrollcommand=yscroll_set)

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
            selected_mode = _MODE_TAB_SINGLE
            try:
                selected_mode = self._mode_tabs.get()
            except Exception:
                pass
            data = {
                "version": _HUB_UI_VERSION,
                "appearance_dark": self._appearance_dark,
                "script_font_size": self._script_font_size,
                "selected_monitor_indices": self._selected_monitor_indices(),
                "last_script_path": str(self._script_path.resolve())
                if self._script_path is not None
                else None,
                "last_smart_goal_path": str(self._smart_goal_path.resolve())
                if self._smart_goal_path is not None
                else None,
                "selected_mode": selected_mode,
                "use_tool_cache": bool(self._use_tool_cache),
                "recording_hotkey_enabled": self._recording_hotkey_enabled,
                "queue_script_paths": [str(p) for p in self._queue_paths],
            }
            write_json(_hub_ui_state_path(), data)
            self._remember_monitor_indices = list(data["selected_monitor_indices"])
        except OSError:
            pass

    def _open_settings(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        def _on_monitors_changed(indices: list[int]) -> None:
            self._remember_monitor_indices = indices
            self._persist_hub_ui_state()

        def _on_script_font_size_changed(size: int) -> None:
            self._apply_script_font_size(size, persist=False)

        def _on_saved() -> None:
            self._persist_hub_ui_state()
            self._status.configure(text="設定已儲存")

        open_agent_settings_dialog(
            self,
            on_saved=_on_saved,
            monitor_indices=self._remember_monitor_indices,
            on_monitors_changed=_on_monitors_changed,
            script_font_size=self._script_font_size,
            on_script_font_size_changed=_on_script_font_size_changed,
        )

    def _apply_script_font_size(self, size: int, *, persist: bool = True) -> None:
        self._script_font_size = clamp_script_font_size(size)
        font = ctk.CTkFont(size=self._script_font_size)
        if getattr(self, "_script_text", None) is not None:
            self._script_text.configure(font=font)
        if getattr(self, "_script_line_numbers", None) is not None:
            self._script_line_numbers.configure(font=font)
        if getattr(self, "_smart_text", None) is not None:
            self._smart_text.configure(font=font)
        if persist:
            self._persist_hub_ui_state()

    def _toggle_appearance(self) -> None:
        self._appearance_dark = not self._appearance_dark
        ctk.set_appearance_mode("dark" if self._appearance_dark else "light")
        self._sync_appearance_toggle_button()
        self._persist_hub_ui_state()

    def _sync_appearance_toggle_button(self) -> None:
        self._appearance_toggle_btn.configure(
            text="\u2600" if self._appearance_dark else "\u263e"
        )

    def _selected_monitor_indices(self) -> list[int]:
        """Return currently remembered monitor indices (managed by settings dialog)."""
        return list(self._remember_monitor_indices)

    def _build_script_section(self) -> None:
        box = ctk.CTkFrame(self, corner_radius=12)
        box.pack(fill="both", expand=True, padx=24, pady=8)
        self._mode_tabs = ctk.CTkTabview(box, command=self._on_mode_tab_changed)
        self._mode_tabs.pack(fill="both", expand=True, padx=8, pady=8)
        self._tab_single = self._mode_tabs.add(_MODE_TAB_SINGLE)
        self._tab_smart = self._mode_tabs.add(_MODE_TAB_SMART)
        self._tab_queue = self._mode_tabs.add(_MODE_TAB_QUEUE)
        self._build_single_script_tab(self._tab_single)
        self._build_smart_mode_tab(self._tab_smart)
        self._build_queue_tab(self._tab_queue)

    def _build_single_script_tab(self, parent: Any) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=4)
        b_open = ctk.CTkButton(row, text="開啟", width=80, command=self._script_open)
        b_open.pack(side="left", padx=(0, 8))
        b_open_rec = ctk.CTkButton(
            row, text="開啟錄製", width=100, command=self._script_open_recording
        )
        b_open_rec.pack(side="left", padx=(0, 8))
        self._recording_report_btn = ctk.CTkButton(
            row, text="錄製報告", width=100, command=self._on_open_recording_report
        )
        # Shown only while a recording folder is the open script.
        b_save = ctk.CTkButton(row, text="儲存", width=80, command=self._script_save)
        b_save.pack(side="left", padx=(0, 8))
        self._script_save_btn = b_save
        b_sas = ctk.CTkButton(row, text="另存新檔", width=100, command=self._script_save_as)
        b_sas.pack(side="left", padx=(0, 8))
        self._script_save_as_btn = b_sas
        b_clear = ctk.CTkButton(row, text="開新檔案", width=100, command=self._script_clear)
        b_clear.pack(side="left", padx=(0, 8))
        self._script_clear_btn = b_clear
        self._analyze_recording_btn = ctk.CTkButton(
            row,
            text="分析錄製",
            width=110,
            command=self._on_analyze_recording_folder,
        )
        self._analyze_recording_btn.pack(side="left")
        self._script_path_label = ctk.CTkLabel(
            parent,
            text="未載入檔案",
            font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray65"),
        )
        self._script_path_label.pack(anchor="w", padx=8, pady=(4, 8))

        editor = ctk.CTkFrame(parent, fg_color="transparent")
        editor.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        script_font = ctk.CTkFont(size=self._script_font_size)
        self._script_line_numbers = ctk.CTkTextbox(
            editor,
            font=script_font,
            width=44,
            activate_scrollbars=False,
            wrap="none",
            fg_color=("gray90", "gray20"),
            text_color=("gray40", "gray60"),
        )
        self._script_line_numbers.pack(side="left", fill="y", padx=(0, 4))
        self._script_line_numbers.insert("0.0", "1")
        self._script_line_numbers.configure(state="disabled")
        self._script_text = ctk.CTkTextbox(editor, font=script_font, wrap="word", undo=True)
        self._script_text.pack(side="left", fill="both", expand=True)
        self._bind_script_text_cache_sync()
        self._bind_textbox_undo_redo(self._script_text)
        self._script_controls.extend(
            [
                b_open,
                b_open_rec,
                self._recording_report_btn,
                b_save,
                b_sas,
                b_clear,
                self._analyze_recording_btn,
                self._script_text,
            ]
        )
        self._refresh_script_toolbar_for_recording()

    def _build_smart_mode_tab(self, parent: Any) -> None:
        ctk.CTkLabel(
            parent,
            text="將整段文字視為一個目標；執行時由多模態 LLM 規劃、執行並驗證每一步。",
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70"),
            wraplength=820,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(4, 4))
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=4)
        b_open = ctk.CTkButton(row, text="開啟", width=80, command=self._smart_open)
        b_open.pack(side="left", padx=(0, 8))
        b_save = ctk.CTkButton(row, text="儲存", width=80, command=self._smart_save)
        b_save.pack(side="left", padx=(0, 8))
        b_sas = ctk.CTkButton(row, text="另存新檔", width=100, command=self._smart_save_as)
        b_sas.pack(side="left", padx=(0, 8))
        b_clear = ctk.CTkButton(row, text="開新檔案", width=100, command=self._smart_clear)
        b_clear.pack(side="left", padx=(0, 8))
        self._smart_path_label = ctk.CTkLabel(
            parent,
            text="未載入目標檔案",
            font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray65"),
        )
        self._smart_path_label.pack(anchor="w", padx=8, pady=(4, 8))
        editor = ctk.CTkFrame(parent, fg_color="transparent")
        editor.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        smart_font = ctk.CTkFont(size=self._script_font_size)
        self._smart_text = ctk.CTkTextbox(editor, font=smart_font, wrap="word", undo=True)
        self._smart_text.pack(side="left", fill="both", expand=True)
        self._smart_text._textbox.bind("<<Modified>>", self._on_smart_text_modified)
        self._bind_textbox_undo_redo(self._smart_text)
        self._smart_controls.extend([b_open, b_save, b_sas, b_clear, self._smart_text])

    def _build_queue_tab(self, parent: Any) -> None:
        ctk.CTkLabel(
            parent,
            text="加入多個腳本檔案或錄製資料夾，依序執行；若某個腳本失敗，會繼續執行下一個。",
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70"),
            wraplength=820,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(4, 8))
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=4)
        b_add = ctk.CTkButton(row, text="新增檔案…", width=100, command=self._queue_add_files)
        b_add.pack(side="left", padx=(0, 8))
        b_add_rec = ctk.CTkButton(
            row, text="加入錄製", width=100, command=self._queue_add_recording
        )
        b_add_rec.pack(side="left", padx=(0, 8))
        b_clear = ctk.CTkButton(row, text="清空", width=80, command=self._queue_clear)
        b_clear.pack(side="left")
        self._queue_warning_label = ctk.CTkLabel(
            parent,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("#b91c1c", "#f87171"),
            wraplength=820,
            justify="left",
        )
        self._queue_list_frame = ctk.CTkScrollableFrame(parent)
        self._queue_list_frame.pack(fill="both", expand=True, padx=8, pady=(8, 8))
        self._queue_controls.extend([b_add, b_add_rec, b_clear])

    def _queue_controls_busy(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _set_queue_control_widgets_state(self, state: str) -> None:
        for w in self._queue_controls:
            w.configure(state=state)
        if state == "normal":
            # Rebuild rows so first/last ↑/↓ stay correctly disabled.
            self._refresh_queue_list()
            return
        for w in self._queue_item_controls:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _update_queue_missing_warning(self) -> None:
        label = getattr(self, "_queue_warning_label", None)
        list_frame = getattr(self, "_queue_list_frame", None)
        if label is None or list_frame is None:
            return
        missing = [p for p in self._queue_paths if not is_runnable_script_path(p)]
        if missing:
            names = "、".join(script_display_name(p) for p in missing)
            label.configure(
                text=f"警告：以下項目找不到檔案或資料夾（執行時將略過）：{names}"
            )
            if not label.winfo_ismapped():
                # CTkScrollableFrame.pack() targets _parent_frame; before= must match that sibling.
                list_container = getattr(list_frame, "_parent_frame", list_frame)
                label.pack(fill="x", padx=8, pady=(0, 4), before=list_container)
        else:
            label.configure(text="")
            if label.winfo_ismapped():
                label.pack_forget()

    def _refresh_queue_list(self) -> None:
        frame = getattr(self, "_queue_list_frame", None)
        if frame is None:
            return
        self._queue_item_controls = []
        for w in frame.winfo_children():
            w.destroy()
        self._update_queue_missing_warning()
        if not self._queue_paths:
            ctk.CTkLabel(
                frame,
                text="尚未加入任何腳本檔案。",
                font=ctk.CTkFont(size=13),
                text_color=("gray40", "gray60"),
            ).pack(anchor="w", padx=6, pady=6)
            return
        busy = self._queue_controls_busy()
        item_state = "disabled" if busy else "normal"
        missing_color = ("#b91c1c", "#f87171")
        for i, p in enumerate(self._queue_paths):
            selected = i == self._queue_selected
            fg = ("gray75", "gray28") if selected else "transparent"
            missing = not is_runnable_script_path(p)
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=2)
            status = self._queue_status_by_index.get(i)
            if status is None and missing:
                icon, icon_color = "\u26a0", missing_color
            else:
                icon, icon_color = self._queue_status_icon(status)
            ctk.CTkLabel(
                row,
                text=icon,
                width=20,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=icon_color,
            ).pack(side="left", padx=(0, 2))
            icon_btn_kw = {
                "width": 28,
                "fg_color": "transparent",
                "hover_color": ("gray70", "gray35"),
                "text_color": ("gray10", "gray90"),
            }
            b_up = ctk.CTkButton(
                row,
                text="\u2191",
                command=lambda idx=i: self._queue_move_up(idx),
                state="disabled" if busy or i <= 0 else "normal",
                **icon_btn_kw,
            )
            b_up.pack(side="left", padx=(0, 2))
            b_down = ctk.CTkButton(
                row,
                text="\u2193",
                command=lambda idx=i: self._queue_move_down(idx),
                state="disabled" if busy or i >= len(self._queue_paths) - 1 else "normal",
                **icon_btn_kw,
            )
            b_down.pack(side="left", padx=(0, 2))
            b_remove = ctk.CTkButton(
                row,
                text="\u2715",
                command=lambda idx=i: self._queue_remove_at(idx),
                state=item_state,
                **icon_btn_kw,
            )
            b_remove.pack(side="left", padx=(0, 6))
            self._queue_item_controls.extend([b_up, b_down, b_remove])
            item_label = script_display_name(p)
            if missing:
                item_label += "（找不到）"
            btn = ctk.CTkButton(
                row,
                text=f"{i + 1}. {item_label}",
                anchor="w",
                fg_color=fg,
                # Neutral hover (not theme blue) so dark text stays readable in light mode.
                hover_color=("gray65", "gray35"),
                hover=not selected,
                text_color=missing_color if missing else ("gray10", "gray90"),
                command=lambda idx=i: self._queue_select(idx),
            )
            btn.pack(side="left", fill="x", expand=True)
            run_root = self._queue_run_root_by_index.get(i)
            if run_root is not None and (run_root / "session_steps.html").is_file():
                report_btn = ctk.CTkButton(
                    row,
                    text="報告",
                    width=56,
                    command=lambda root=run_root: self._open_report_html(root / "session_steps.html"),
                )
                report_btn.pack(side="left", padx=(6, 0))
            start_btn = ctk.CTkButton(
                row,
                text="開始執行",
                width=88,
                command=lambda idx=i: self._queue_start_from(idx),
                state=item_state,
            )
            start_btn.pack(side="left", padx=(6, 0))
            edit_btn = ctk.CTkButton(
                row,
                text="編輯",
                width=56,
                command=lambda idx=i: self._queue_edit_file(idx),
                state=item_state,
            )
            edit_btn.pack(side="left", padx=(6, 0))
            self._queue_item_controls.extend([start_btn, edit_btn])

    @staticmethod
    def _queue_status_icon(status: str | None) -> tuple[str, tuple[str, str]]:
        if status == "ok":
            return "\u2713", ("#16a34a", "#22c55e")
        if status == "fail":
            return "\u2717", ("#b91c1c", "#f87171")
        if status == "skipped":
            return "\u2013", ("gray40", "gray60")
        if status == "stopped":
            return "\u25a0", ("#b45309", "#fbbf24")
        return " ", ("gray40", "gray60")

    @staticmethod
    def _queue_status_from_session_end_reason(reason: str | None) -> str:
        """Map coordinator session_end_reason to a queue row status."""
        if reason == "step_failed":
            return "fail"
        return "ok"

    def _mark_queue_result(self, index: int, status: str, run_root: Path | None = None) -> None:
        def apply() -> None:
            self._queue_status_by_index[index] = status
            if run_root is not None:
                self._queue_run_root_by_index[index] = run_root
            self._refresh_queue_list()

        self.after(0, apply)

    def _queue_select(self, index: int) -> None:
        self._queue_selected = index
        self._refresh_queue_list()

    def _queue_start_from(self, index: int) -> None:
        """Start queue execution from the given list item through the end."""
        if self._recording_session.is_active():
            show_ctk_message(self, "執行", "請先停止錄製再開始執行。", kind="warning")
            return
        if self._is_recording_finalizing():
            show_ctk_message(self, "執行", "請等待錄製完成。", kind="warning")
            return
        if self._recording_analysis_thread and self._recording_analysis_thread.is_alive():
            show_ctk_message(self, "執行", "請等待錄製分析完成，或按「停止分析」。", kind="warning")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            return
        if index < 0 or index >= len(self._queue_paths):
            return
        eye_indices = self._selected_monitor_indices()
        if not eye_indices:
            show_ctk_message(
                self,
                "顯示器",
                "請至少選擇一台要截取的顯示器。",
                kind="warning",
            )
            return
        self._queue_selected = index
        self._user_requested_stop = False
        self._post_run_unlink = None
        self._last_script_run_folder = None
        self._start_queue_run(eye_indices, start_index=index)

    def _queue_edit_file(self, index: int) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        if index < 0 or index >= len(self._queue_paths):
            return
        p = resolve_runnable_script_path(self._queue_paths[index])
        if not is_runnable_script_path(p):
            show_ctk_message(self, "編輯", f"找不到檔案：\n{p}", kind="error")
            return
        if is_recording_script_path(p):
            rec = recording_run_dir(p)
            if rec is None:
                show_ctk_message(self, "編輯", f"找不到檔案：\n{p}", kind="error")
                return
            self._open_recording_review_html(rec, rec.name)
            return
        if not self._confirm_proceed_with_unsaved_script():
            return
        self._load_script_into_editor(p)
        try:
            self._mode_tabs.set("單一腳本")
        except Exception:
            pass
        self._status.configure(text=f"已在單一腳本開啟 {script_display_name(p)}")

    def _queue_add_files(self) -> None:
        initial = ROOT_DIR / "scripts"
        paths = filedialog.askopenfilenames(
            parent=self,
            title="新增腳本檔案",
            initialdir=str(initial) if initial.is_dir() else str(ROOT_DIR),
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
        )
        if not paths:
            return
        for path in paths:
            self._queue_paths.append(Path(path))
        self._queue_status_by_index = {}
        self._queue_run_root_by_index = {}
        self._refresh_queue_list()
        self._persist_hub_ui_state()

    def _queue_add_recording(self) -> None:
        folders = self._ask_recording_directories(
            title="選擇錄製資料夾（可按 Ctrl 選取多個）"
        )
        if not folders:
            return
        added, invalid = partition_recording_dirs(folders, existing=self._queue_paths)
        for script in added:
            self._queue_paths.append(script)
        if added:
            self._queue_status_by_index = {}
            self._queue_run_root_by_index = {}
            self._refresh_queue_list()
            self._persist_hub_ui_state()
            if len(added) == 1:
                self._status.configure(text=f"已加入錄製 {script_display_name(added[0])}")
            else:
                self._status.configure(text=f"已加入 {len(added)} 個錄製")
        elif not invalid:
            self._status.configure(text="所選錄製已在佇列中")
        if invalid and not added:
            show_ctk_message(
                self,
                "加入錄製",
                "所選資料夾不是有效的錄製（需有 session.json）。",
                kind="error",
            )
            return
        if invalid:
            skipped = "\n".join(p.name or str(p) for p in invalid)
            show_ctk_message(
                self,
                "加入錄製",
                f"已略過非錄製資料夾：\n{skipped}",
                kind="warning",
            )

    def _queue_move_up(self, index: int) -> None:
        if index <= 0 or index >= len(self._queue_paths):
            return
        self._queue_paths[index - 1], self._queue_paths[index] = (
            self._queue_paths[index],
            self._queue_paths[index - 1],
        )
        self._queue_selected = index - 1
        self._queue_status_by_index = {}
        self._queue_run_root_by_index = {}
        self._refresh_queue_list()
        self._persist_hub_ui_state()

    def _queue_move_down(self, index: int) -> None:
        if index < 0 or index >= len(self._queue_paths) - 1:
            return
        self._queue_paths[index + 1], self._queue_paths[index] = (
            self._queue_paths[index],
            self._queue_paths[index + 1],
        )
        self._queue_selected = index + 1
        self._queue_status_by_index = {}
        self._queue_run_root_by_index = {}
        self._refresh_queue_list()
        self._persist_hub_ui_state()

    def _queue_remove_at(self, index: int) -> None:
        if index < 0 or index >= len(self._queue_paths):
            return
        del self._queue_paths[index]
        self._queue_status_by_index = {}
        self._queue_run_root_by_index = {}
        if not self._queue_paths:
            self._queue_selected = None
        elif self._queue_selected is not None:
            if self._queue_selected == index:
                self._queue_selected = min(index, len(self._queue_paths) - 1)
            elif self._queue_selected > index:
                self._queue_selected -= 1
        self._refresh_queue_list()
        self._persist_hub_ui_state()

    def _queue_clear(self) -> None:
        self._queue_paths = []
        self._queue_selected = None
        self._queue_status_by_index = {}
        self._queue_run_root_by_index = {}
        self._refresh_queue_list()
        self._persist_hub_ui_state()

    def _build_actions_row(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(side="bottom", fill="x", padx=24, pady=(12, 8))
        self._use_tool_cache_checkbox = ctk.CTkCheckBox(
            row,
            text="使用快取工具（略過 LLM）",
            font=ctk.CTkFont(size=13),
            command=self._on_use_tool_cache_changed,
        )
        self._use_tool_cache_checkbox.pack(pady=(0, 10))
        if self._use_tool_cache:
            self._use_tool_cache_checkbox.select()
        self._actions_btn_row = ctk.CTkFrame(row, fg_color="transparent")
        btn_row = self._actions_btn_row
        btn_row.pack()
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(6, weight=1)
        self._pause_btn = ctk.CTkButton(
            btn_row,
            text="暫停",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=120,
            command=self._on_pause_run,
        )
        self._run_btn = ctk.CTkButton(
            btn_row,
            text="開始執行",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=120,
            command=self._on_start_run,
        )
        self._run_btn.grid(row=0, column=2)
        self._record_btn = ctk.CTkButton(
            btn_row,
            text="開始錄製",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=120,
            command=self._on_record_button,
        )
        self._record_btn.grid(row=0, column=3, padx=(12, 0))
        self._open_report_btn = ctk.CTkButton(
            btn_row,
            text="開啟報告",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=120,
            command=self._open_last_report,
        )
        self._open_reports_index_btn = ctk.CTkButton(
            btn_row,
            text="報告列表",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=120,
            command=self._open_reports_index,
        )
        self._open_reports_index_btn.grid(row=0, column=5, padx=(12, 0), sticky="w")
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

    def _sync_tool_cache_checkbox_for_mode(self) -> None:
        """Hide tool-cache option in 智能模式; cache replay is not applicable there."""
        checkbox = getattr(self, "_use_tool_cache_checkbox", None)
        if checkbox is None:
            return
        try:
            selected = self._mode_tabs.get()
        except Exception:
            return
        if selected == _MODE_TAB_SMART:
            checkbox.pack_forget()
            return
        btn_row = getattr(self, "_actions_btn_row", None)
        if btn_row is not None:
            checkbox.pack(pady=(0, 10), before=btn_row)
        else:
            checkbox.pack(pady=(0, 10))

    def _tool_cache_enabled_for_run(self) -> bool:
        try:
            if self._mode_tabs.get() == _MODE_TAB_SMART:
                return False
        except Exception:
            pass
        return self._use_tool_cache_checkbox.get() == 1

    def _show_report_button(self, html_path: Path) -> None:
        if not html_path.is_file():
            return
        self._last_report_html = html_path
        self._open_report_btn.grid(row=0, column=4, padx=(12, 0), sticky="w")

    def _hide_report_button(self) -> None:
        self._last_report_html = None
        self._open_report_btn.grid_remove()

    def _open_report_html(self, html_path: Path) -> None:
        if not html_path.is_file():
            show_ctk_message(self, "報告", f"找不到報告檔案：\n{html_path}", kind="error")
            return
        try:
            url = self._report_http_url(html_path)
            webbrowser.open(url)
        except Exception as e:
            show_ctk_message(self, "報告", f"無法開啟報告：\n{e}", kind="error")

    def _report_http_url(self, html_path: Path) -> str:
        """Serve reports over localhost so interactive report APIs work.

        Uses the common parent of ``runs_dir`` and ``recordings_dir`` when they are
        siblings so index links like ``../recordings/<id>/recording_steps.html``
        resolve instead of 404ing under a runs-only document root.
        """
        settings = load_settings()
        runs_root = Path(settings.runs_dir).resolve()
        recordings_root = Path(settings.recordings_dir).resolve()
        resolved = html_path.resolve()
        serve_root = reports_serve_root(runs_root, recordings_root)
        candidates: list[Path] = [serve_root]
        for extra in (runs_root, recordings_root):
            if extra not in candidates:
                candidates.append(extra)
        for root in candidates:
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError:
                continue
            server = ensure_runs_report_server(root)
            return f"{server.base_url}/{relative}"
        return resolved.as_uri()

    def _open_last_report(self) -> None:
        if self._last_report_html is None:
            return
        self._open_report_html(self._last_report_html)

    def _open_reports_index(self) -> None:
        runs_root = Path(load_settings().runs_dir)
        try:
            index_path = write_runs_index_html(runs_root)
        except Exception as e:
            show_ctk_message(self, "報告", f"無法建立報告列表：\n{e}", kind="error")
            return
        self._open_report_html(index_path)

    def _set_run_button_idle(self) -> None:
        self._run_btn.configure(text="開始執行", command=self._on_start_run, state="normal")
        self._pause_btn.grid_remove()

    def _set_run_button_running(self) -> None:
        self._run_btn.configure(text="停止執行", command=self._on_stop_run, state="normal")
        self._pause_btn.configure(text="暫停", command=self._on_pause_run, state="normal")
        self._pause_btn.grid(row=0, column=1, padx=(0, 12))

    def _set_pause_button_paused(self) -> None:
        self._pause_btn.configure(text="繼續", command=self._on_resume_run, state="normal")
        self._pause_btn.grid(row=0, column=1, padx=(0, 12))

    def _build_status(self) -> None:
        self._status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13))
        self._status.pack(side="bottom", anchor="w", padx=28, pady=(0, 16))

    def _is_analysis_running(self) -> bool:
        thread = self._recording_analysis_thread
        return thread is not None and thread.is_alive()

    def _is_recording_finalizing(self) -> bool:
        thread = self._recording_finalize_thread
        return (
            self._recording_session.is_finalizing()
            or (thread is not None and thread.is_alive())
        )

    def _is_recording_busy(self) -> bool:
        return self._recording_session.is_active() or self._is_recording_finalizing()

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
        for w in self._script_controls:
            w.configure(state="normal")
        for w in self._smart_controls:
            w.configure(state="normal")
        self._set_queue_control_widgets_state("normal")
        self._use_tool_cache_checkbox.configure(state="normal")
        if self._record_btn is not None:
            self._record_btn.configure(
                text=self._record_button_idle_label(),
                state="normal",
                command=self._on_record_button,
            )
        self._hide_analysis_progress()
        self._apply_script_editor_lock()

    def _set_hub_controls_recording(self) -> None:
        self._run_btn.configure(state="disabled")
        self._settings_btn.configure(state="disabled")
        for w in self._script_controls:
            if w is self._record_btn:
                continue
            w.configure(state="disabled")
        for w in self._smart_controls:
            w.configure(state="disabled")
        self._set_queue_control_widgets_state("disabled")
        self._use_tool_cache_checkbox.configure(state="disabled")
        if self._record_btn is not None:
            self._record_btn.configure(text="停止錄製", state="normal", command=self._on_record_button)
        self._hide_analysis_progress()

    def _set_hub_controls_finalizing(self) -> None:
        self._run_btn.configure(state="disabled")
        self._settings_btn.configure(state="disabled")
        for w in self._script_controls:
            if w is self._record_btn:
                continue
            w.configure(state="disabled")
        for w in self._smart_controls:
            w.configure(state="disabled")
        self._set_queue_control_widgets_state("disabled")
        self._use_tool_cache_checkbox.configure(state="disabled")
        if self._record_btn is not None:
            self._record_btn.configure(text="停止分析", state="normal", command=self._on_record_button)
        self._hide_analysis_progress()

    def _set_hub_controls_analyzing(self) -> None:
        self._analysis_cancel_event.clear()
        self._run_btn.configure(state="disabled")
        self._settings_btn.configure(state="disabled")
        for w in self._script_controls:
            if w is self._record_btn:
                continue
            w.configure(state="disabled")
        for w in self._smart_controls:
            w.configure(state="disabled")
        self._set_queue_control_widgets_state("disabled")
        self._use_tool_cache_checkbox.configure(state="disabled")
        if self._record_btn is not None:
            self._record_btn.configure(text="停止分析", state="normal", command=self._on_record_button)
        self._show_analysis_progress()

    def _update_analysis_progress(self, current: int, total: int) -> None:
        if self._analysis_progress is not None and total > 0:
            self._analysis_progress.set(current / total)
        settings = load_settings()
        if total > 0:
            pct = int(round(100.0 * current / total))
            status = f"分析錄製中 ({pct}%)… {settings.brain_lm}"
        else:
            status = f"分析錄製中… {settings.brain_lm}"
        self._status.configure(
            text=status,
            text_color=("gray20", "gray65"),
        )

    def _begin_vision_prefetch_progress(self, total: int) -> None:
        if total <= 0:
            return
        self._show_analysis_progress()
        self._update_vision_prefetch_progress(0, total)

    def _update_vision_prefetch_progress(self, current: int, total: int) -> None:
        if total <= 0:
            return
        if self._analysis_progress is not None:
            self._analysis_progress.set(current / total)
        pct = int(round(100.0 * current / total))
        self._status.configure(
            text=f"正在處理視覺資料 ({pct}%)…",
            text_color=("gray20", "gray65"),
        )

    def _request_cancel_analysis(self) -> None:
        if not self._is_analysis_running() and not self._is_recording_finalizing():
            return
        self._analysis_cancel_event.set()
        if self._record_btn is not None:
            self._record_btn.configure(state="disabled")
        self._status.configure(text="正在停止分析…", text_color=("gray20", "gray65"))

    def _script_open(self) -> None:
        if not self._confirm_proceed_with_unsaved_script():
            return
        initial = ROOT_DIR / "scripts"
        path = filedialog.askopenfilename(
            parent=self,
            title="開啟腳本",
            initialdir=str(initial) if initial.is_dir() else str(ROOT_DIR),
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
        )
        if not path:
            return
        self._load_script_into_editor(Path(path))

    def _script_open_recording(self) -> None:
        if not self._confirm_proceed_with_unsaved_script():
            return
        folder = self._ask_recording_directory(title="開啟錄製資料夾")
        if folder is None:
            return
        script = self._validate_recording_folder_for_script(folder)
        if script is None:
            show_ctk_message(
                self,
                "開啟錄製",
                f"所選資料夾不是有效的錄製（需有 session.json）。",
                kind="error",
            )
            return
        self._load_script_into_editor(script)
        self._status.configure(text=f"已開啟錄製 {script_display_name(script)}（唯讀）")

    def _script_save(self) -> bool:
        if self._is_recording_script_open():
            self._status.configure(
                text="錄製腳本請在「錄製紀錄」頁面編輯並儲存",
                text_color=("gray20", "gray65"),
            )
            return False
        if self._script_path is not None:
            body = self._script_text.get("0.0", "end").rstrip() + "\n"
            self._script_path.write_text(body, encoding="utf-8")
            self._mark_script_clean()
            self._status.configure(text=f"已儲存 {self._script_path.name}")
            self._persist_hub_ui_state()
            return True
        if self._sync_script_text_to_runtime_cache():
            self._status.configure(text="已儲存執行命令")
            return True
        return self._script_save_as()

    def _script_save_as(self) -> bool:
        if self._is_recording_script_open():
            self._status.configure(
                text="錄製腳本請在「錄製紀錄」頁面編輯並儲存",
                text_color=("gray20", "gray65"),
            )
            return False
        path = filedialog.asksaveasfilename(
            parent=self,
            title="腳本另存新檔",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
            initialdir=str(ROOT_DIR / "scripts"),
        )
        if not path:
            return False
        p = Path(path)
        # Unlock temporarily if needed to read text (should already be editable for non-recording).
        body = self._script_text.get("0.0", "end").rstrip() + "\n"
        p.write_text(body, encoding="utf-8")
        self._load_script_into_editor(p)
        self._status.configure(text=f"已另存新檔 {p.name}")
        return True

    def _save_recording_instructions_as_new_script(self, lines: list[str]) -> bool:
        """Save analysis instructions to a new file and open it as the current script."""
        if not self._confirm_proceed_with_unsaved_script():
            return False
        path = filedialog.asksaveasfilename(
            parent=self,
            title="存成新檔",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("全部", "*.*")],
            initialdir=str(ROOT_DIR / "scripts"),
        )
        if not path:
            return False
        body = "\n".join(lines).rstrip() + "\n"
        p = Path(path)
        p.write_text(body, encoding="utf-8")
        self._load_script_into_editor(p)
        self._status.configure(text=f"已存成新檔並開啟 {p.name}")
        return True

    def _script_clear(self) -> None:
        """Unload any opened path / cache binding and empty the script editor."""
        if not self._confirm_proceed_before_new_script():
            return
        self._script_path = None
        self._runtime_commands_cache_path = None
        self._script_text.configure(state="normal")
        self._suppress_script_cache_sync = True
        try:
            self._script_text.delete("0.0", "end")
            self._reset_textbox_undo(self._script_text)
        finally:
            self._suppress_script_cache_sync = False
        self._clear_script_step_statuses()
        self._refresh_script_line_numbers()
        self._refresh_script_path_label()
        self._mark_script_clean()
        self._apply_script_editor_lock()
        self._refresh_record_button_idle_label()
        self._status.configure(text="", text_color=("gray20", "gray65"))
        self._persist_hub_ui_state()

    def _on_pause_run(self) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return
        pause_run()
        self._set_pause_button_paused()
        self._status.configure(text="已暫停（點繼續以恢復）")

    def _on_resume_run(self) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return
        resume_run()
        self._set_run_button_running()
        self._status.configure(text="執行中…")

    def _on_stop_run(self) -> None:
        from main import request_coordinator_cancel

        self._user_requested_stop = True
        self._status.configure(text="正在停止…")
        # Unblock a paused wait so cancel can proceed promptly.
        resume_run()
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
        if not self._confirm_proceed_with_unsaved_script():
            return
        if not self._confirm_proceed_with_unsaved_smart():
            return
        if self._is_analysis_running():
            self._analysis_cancel_event.set()
        if self._recording_session.is_active():
            self._recording_session.stop()
            self._vision_prefetch.drain_and_stop(timeout=5.0)
        elif self._is_recording_finalizing():
            self._wait_for_recording_finalize()
            self._vision_prefetch.drain_and_stop(timeout=5.0)
        self._recording_hotkey.unregister()
        stop_runs_report_server()
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

    @staticmethod
    def _is_recording_folder(run_dir: Path) -> bool:
        return run_dir.is_dir() and (
            (run_dir / "session.json").is_file() or (run_dir / "events").is_dir()
        )

    @staticmethod
    def _count_recording_events(run_dir: Path) -> int:
        manifest = read_json(run_dir / "session.json", {})
        if isinstance(manifest, dict):
            events = manifest.get("events")
            if isinstance(events, list):
                return len(events)
        events_dir = run_dir / "events"
        if events_dir.is_dir():
            return len(list(events_dir.glob("event_*.json")))
        return 0

    def _on_analyze_recording_folder(self) -> None:
        if self._is_analysis_running():
            return
        if self._worker_thread and self._worker_thread.is_alive():
            show_ctk_message(self, "錄製分析", "請先停止執行再分析錄製。", kind="warning")
            return
        if self._recording_session.is_active():
            show_ctk_message(self, "錄製分析", "請先停止錄製再分析。", kind="warning")
            return
        if self._is_recording_finalizing():
            show_ctk_message(self, "錄製分析", "請等待錄製完成。", kind="warning")
            return

        settings = load_settings()
        initial = Path(settings.recordings_dir)
        folder = filedialog.askdirectory(
            parent=self,
            title="選擇錄製資料夾",
            initialdir=str(initial) if initial.is_dir() else str(ROOT_DIR),
        )
        if not folder:
            return

        run_dir = Path(folder)
        if not self._is_recording_folder(run_dir):
            show_ctk_message(
                self,
                "錄製分析",
                "所選資料夾不是有效的錄製工作階段。",
                kind="error",
            )
            return

        event_count = self._count_recording_events(run_dir)
        if event_count <= 0:
            show_ctk_message(self, "錄製分析", "此錄製資料夾沒有事件可分析。", kind="warning")
            return

        self._start_recording_analysis(run_dir, event_count)

    def _on_record_button(self) -> None:
        if self._is_analysis_running() or self._is_recording_finalizing():
            self._request_cancel_analysis()
            return
        self._toggle_recording()

    def _toggle_recording(self) -> None:
        if self._is_analysis_running():
            return
        if self._is_recording_finalizing():
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
        if self._is_recording_finalizing():
            return
        if self._recording_session.is_active():
            return
        existing_run_dir: Path | None = None
        if self._is_recording_script_open() and self._script_path is not None:
            existing_run_dir = Path(self._script_path)
        try:
            self._recording_session.set_suppress_hotkey_keys(True)
            run_dir = self._recording_session.start(
                ignore_rect_provider=self._recording_ignore_rect_provider,
                existing_run_dir=existing_run_dir,
            )
        except Exception as exc:
            self._recording_session.set_suppress_hotkey_keys(False)
            show_ctk_message(self, "錄製", f"無法開始錄製：{exc}", kind="error")
            return
        finally:
            self.after(400, lambda: self._recording_session.set_suppress_hotkey_keys(False))

        self._set_hub_controls_recording()
        self._vision_prefetch.start(run_dir)
        verb = "繼續錄製中" if existing_run_dir is not None else "錄製中"
        self._status.configure(
            text=f"{verb} (0 個事件)… {run_dir.name}",
            text_color=("gray20", "gray65"),
        )
        try:
            self.iconify()
        except Exception:
            pass

    def _on_recording_event(self, event: RecordedEvent) -> None:
        self._vision_prefetch.enqueue(event)
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
        if not self._recording_session.is_active() or self._is_recording_finalizing():
            return
        self._analysis_cancel_event.clear()
        self._recording_session.begin_stop()
        self._set_hub_controls_finalizing()
        self._status.configure(text="正在完成錄製…", text_color=("gray20", "gray65"))
        self._start_recording_finalize_worker(analyze=analyze)

    def _start_recording_finalize_worker(self, *, analyze: bool) -> None:
        if self._recording_finalize_thread is not None and self._recording_finalize_thread.is_alive():
            return

        def worker() -> None:
            try:
                run_dir = self._recording_session.finalize_stop()
                event_count = self._recording_session.event_count()

                def on_prefetch_progress(current: int, total: int) -> None:
                    if total <= 0:
                        return
                    if current == 0:
                        self.after(
                            0,
                            lambda t=total: self._begin_vision_prefetch_progress(t),
                        )
                    else:
                        self.after(
                            0,
                            lambda c=current, t=total: self._update_vision_prefetch_progress(
                                c, t
                            ),
                        )

                self._vision_prefetch.drain_and_stop(on_progress=on_prefetch_progress)
                self.after(
                    0,
                    lambda: self._on_recording_finalize_done(run_dir, event_count, analyze),
                )
            except Exception as exc:
                self._vision_prefetch.drain_and_stop(timeout=5.0)
                self.after(0, lambda: self._on_recording_finalize_failed(exc))

        self._recording_finalize_thread = threading.Thread(
            target=worker,
            name="recording-finalize",
            daemon=True,
        )
        self._recording_finalize_thread.start()

    def _wait_for_recording_finalize(self) -> None:
        thread = self._recording_finalize_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=20)

    def _on_recording_finalize_done(
        self,
        run_dir: Path | None,
        event_count: int,
        analyze: bool,
    ) -> None:
        self._recording_finalize_thread = None
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass
        if run_dir is None:
            self._set_hub_controls_idle()
            self._status.configure(text="錄製已停止。")
            return
        if analyze and self._analysis_cancel_event.is_set():
            self._set_hub_controls_idle()
            self._status.configure(text="分析已停止。")
            show_ctk_message(self, "錄製分析已停止", "分析已停止。", kind="warning")
            if self._is_recording_script_open() and run_dir is not None:
                self._load_script_into_editor(run_dir)
            return
        if analyze and event_count > 0:
            total_events = self._count_recording_events(run_dir)
            self._start_recording_analysis(run_dir, max(event_count, total_events))
        else:
            self._set_hub_controls_idle()
            self._status.configure(text=f"錄製已停止（{event_count} 個事件）。")
            if self._is_recording_script_open() and run_dir is not None:
                self._load_script_into_editor(run_dir)

    def _on_recording_finalize_failed(self, exc: Exception) -> None:
        self._recording_finalize_thread = None
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass
        self._set_hub_controls_idle()
        show_ctk_message(self, "錄製", f"完成錄製失敗：{exc}", kind="error")
        self._status.configure(text="完成錄製失敗。")

    def _start_recording_analysis(self, run_dir: Path, event_count: int) -> None:
        self._set_hub_controls_analyzing()
        settings = load_settings()
        self._status.configure(
            text=f"分析錄製中 (0%)… {settings.brain_lm}",
            text_color=("gray20", "gray65"),
        )
        self._update_analysis_progress(0, max(1, event_count * 3))
        self._recording_analysis_thread = threading.Thread(
            target=self._analyze_recording_worker,
            args=(run_dir,),
            daemon=True,
        )
        self._recording_analysis_thread.start()

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
        self.after(0, lambda: self._on_recording_analysis_done(report, run_dir))

    def _on_recording_analysis_done(
        self,
        report: dict[str, Any],
        run_dir: Path | None = None,
    ) -> None:
        self._recording_analysis_thread = None
        self._set_hub_controls_idle()
        cached = int(report.get("cached", 0))
        skipped = int(report.get("skipped", 0))
        recorded = int(report.get("recorded", 0))
        cancelled = bool(report.get("cancelled", False))
        instructions = report.get("instructions")
        expected_outcomes_raw = report.get("expected_outcomes")
        lines: list[str] = []
        if isinstance(instructions, list):
            instruction_lines = [str(x) for x in instructions if str(x).strip()]
            outcomes: list[str | None] = []
            if isinstance(expected_outcomes_raw, list):
                for item in expected_outcomes_raw:
                    if isinstance(item, str) and item.strip():
                        outcomes.append(item.strip())
                    else:
                        outcomes.append(None)
            while len(outcomes) < len(instruction_lines):
                outcomes.append(None)
            lines = format_script_lines_with_outcomes(instruction_lines, outcomes)
        if cancelled:
            processed = int(report.get("processed", 0))
            msg = (
                f"分析已停止（已完成 {processed}/{recorded} 個事件）。\n"
                f"已寫入快取 {cached} 筆，略過 {skipped} 筆。"
            )
            self._status.configure(text=f"分析已停止（{processed}/{recorded}）。")
            show_ctk_message(self, "錄製分析已停止", msg, kind="warning")
            if self._is_recording_script_open() and self._script_path is not None:
                self._load_script_into_editor(self._script_path)
            return
        msg = (
            f"錄製 {recorded} 個事件。\n"
            f"已寫入快取 {cached} 筆，略過 {skipped} 筆。"
        )
        self._status.configure(text=f"已寫入快取 {cached} 筆（略過 {skipped}）。")
        run_id_raw = report.get("run_id")
        current_name = run_id_raw.strip() if isinstance(run_id_raw, str) else ""
        if not current_name and run_dir is not None:
            current_name = run_dir.name
        continue_open = self._is_recording_script_open()
        choice, folder_name = prompt_append_recording_instructions(
            self,
            msg,
            folder_name=current_name,
            allow_append=not continue_open and (
                (run_dir is not None and Path(run_dir).is_dir()) or bool(current_name)
            ),
        )
        if current_name:
            current_name = self._rename_recording_folder_from_dialog(
                current_name,
                folder_name,
                run_dir=run_dir,
            )
        folder = self._resolve_analyzed_recording_folder(current_name, run_dir)
        if continue_open:
            target = self._script_path
            if target is None:
                target = folder
            if target is not None and Path(target).is_dir():
                self._load_script_into_editor(Path(target))
            if choice == "open_review":
                self._open_recording_review_html(folder, current_name)
            elif choice == "add_queue":
                self._append_recording_to_queue(folder)
            return
        if choice == "append":
            if folder is not None and self._is_recording_folder(folder):
                self._load_script_into_editor(folder)
                self._status.configure(
                    text=f"已開啟錄製 {script_display_name(folder)}（唯讀）"
                )
            else:
                show_ctk_message(
                    self,
                    "錄製分析完成",
                    "找不到錄製資料夾，無法加入腳本。",
                    kind="warning",
                )
        elif choice == "open_review":
            self._open_recording_review_html(folder, current_name)
        elif choice == "add_queue":
            self._append_recording_to_queue(folder)

    def _append_recording_to_queue(self, folder: Path | None) -> None:
        """Add a recording folder to the queue and open the queue tab."""
        if folder is None or not self._is_recording_folder(folder):
            show_ctk_message(
                self,
                "錄製分析完成",
                "找不到錄製資料夾，無法加入佇列。",
                kind="warning",
            )
            return
        added, _invalid = partition_recording_dirs([folder], existing=self._queue_paths)
        if added:
            for script in added:
                self._queue_paths.append(script)
            self._queue_status_by_index = {}
            self._queue_run_root_by_index = {}
            self._refresh_queue_list()
            self._persist_hub_ui_state()
            self._status.configure(text=f"已加入錄製 {script_display_name(added[0])}")
        else:
            self._status.configure(text="所選錄製已在佇列中")
        try:
            self._mode_tabs.set(_MODE_TAB_QUEUE)
        except Exception:
            pass

    def _resolve_analyzed_recording_folder(
        self,
        run_id: str,
        run_dir: Path | None,
    ) -> Path | None:
        """Return the on-disk folder for an analyzed recording after optional rename."""
        if self._script_path is not None and is_recording_script_path(self._script_path):
            if not run_id or Path(self._script_path).name == run_id:
                return Path(self._script_path)
        if run_dir is not None:
            candidate = Path(run_dir)
            if run_id and candidate.name != run_id:
                renamed = candidate.parent / run_id
                if renamed.is_dir():
                    return renamed
            if candidate.is_dir():
                return candidate
        if not run_id:
            return None
        settings = load_settings()
        for root in (Path(settings.runs_dir), Path(settings.recordings_dir)):
            candidate = root / run_id
            if candidate.is_dir():
                return candidate
        return None

    def _open_recording_review_html(
        self,
        folder: Path | None,
        run_id: str,
    ) -> None:
        html_path: Path | None = None
        if folder is not None:
            candidate = Path(folder) / "recording_steps.html"
            if candidate.is_file():
                html_path = candidate
            elif Path(folder).is_dir():
                try:
                    from src.common.session_html import write_recording_html_from_run

                    html_path = write_recording_html_from_run(
                        Path(folder), update_index=False
                    )
                except Exception:
                    html_path = None
        if html_path is None and run_id:
            settings = load_settings()
            for root in (Path(settings.runs_dir), Path(settings.recordings_dir)):
                candidate = root / run_id / "recording_steps.html"
                if candidate.is_file():
                    html_path = candidate
                    break
        if html_path is None or not html_path.is_file():
            show_ctk_message(self, "錄製報告", "找不到錄製報告路徑。", kind="warning")
            return
        self._open_report_html(html_path)

    def _rename_recording_folder_from_dialog(
        self,
        run_id: str,
        new_name: str,
        *,
        run_dir: Path | None = None,
    ) -> str:
        """Rename the recording folder if the dialog name differs. Returns the effective id."""
        cleaned = new_name.strip()
        if not cleaned or cleaned == run_id:
            return run_id
        if run_dir is not None and Path(run_dir).is_dir():
            runs_root = Path(run_dir).parent
            source_id = Path(run_dir).name
        else:
            runs_root = Path(load_settings().runs_dir)
            source_id = run_id
        try:
            result = rename_recording_folder(runs_root, source_id, cleaned)
        except ValueError as exc:
            show_ctk_message(
                self,
                "錄製分析完成",
                _recording_folder_rename_error_text(exc),
                kind="warning",
            )
            return run_id
        except OSError as exc:
            show_ctk_message(
                self,
                "錄製分析完成",
                f"無法重新命名資料夾：{exc}",
                kind="error",
            )
            return run_id
        new_id = str(result.get("new_id") or cleaned)
        self._retarget_recording_folder_refs(runs_root / source_id, runs_root / new_id)
        return new_id

    def _retarget_recording_folder_refs(self, old_dir: Path, new_dir: Path) -> None:
        """Update editor/queue paths after a recording folder rename."""
        old_key = _resolved_path_key(old_dir)
        try:
            new_resolved = new_dir.resolve()
        except OSError:
            new_resolved = new_dir

        changed = False
        if self._script_path is not None and _path_is_recording_folder(
            self._script_path, old_key
        ):
            self._script_path = new_resolved
            self._refresh_script_path_label()
            changed = True

        new_queue: list[Path] = []
        queue_changed = False
        for path in self._queue_paths:
            if _path_is_recording_folder(path, old_key):
                new_queue.append(new_resolved)
                queue_changed = True
            else:
                new_queue.append(path)
        if queue_changed:
            self._queue_paths = new_queue
            self._refresh_queue_list()
            changed = True

        if changed:
            self._persist_hub_ui_state()

    def _begin_worker_run(self, args: _WorkerArgs) -> None:
        reset_run_control()
        if args.run_mode in ("script", "runtime"):
            set_step_status_callback(self._on_step_status_from_worker)
        self._set_run_button_running()
        self._hide_report_button()
        self._settings_btn.configure(state="disabled")
        for w in self._script_controls:
            w.configure(state="disabled")
        for w in self._smart_controls:
            w.configure(state="disabled")
        self._set_queue_control_widgets_state("disabled")
        self._use_tool_cache_checkbox.configure(state="disabled")
        if self._record_btn is not None:
            self._record_btn.configure(state="disabled")
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
        if self._script_path is not None and not is_recording_script_path(self._script_path):
            transcript_path = self._script_path
            on_undo = self._pop_last_runtime_command_from_script_file
        else:
            if self._runtime_commands_cache_path is None:
                self._runtime_commands_cache_path = self._runtime_command_transcript_path()
            transcript_path = self._runtime_commands_cache_path
            on_undo = self._pop_last_runtime_command_from_cache
            if self._is_recording_script_open():
                # Do not append runtime steps into a recording folder.
                self._script_path = None
                self._apply_script_editor_lock()
                self._refresh_script_path_label()

        self._user_requested_stop = False
        self._post_run_unlink = None
        self._last_run_was_script_mode = False
        # Keep script-run icons; switch to last-line marking for new runtime commands.
        self._script_step_line_numbers = []

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
            run_mode="runtime",
            eye_monitor_indices=eye_indices,
            script_raw="",
            script_disk_path=None,
            run_folder_name=self._last_script_run_folder,
            use_tool_cache=self._tool_cache_enabled_for_run(),
        )
        self._begin_worker_run(args)

    def _start_queue_run(self, eye_indices: list[int], *, start_index: int = 0) -> None:
        indexed = [
            (i, p)
            for i, p in enumerate(self._queue_paths)
            if i >= start_index and is_runnable_script_path(p)
        ]
        if not indexed:
            msg = (
                "請先新增至少一個存在的腳本檔案。"
                if not self._queue_paths
                else "從此項目起沒有可執行的腳本檔案。"
            )
            show_ctk_message(self, "佇列執行", msg, kind="warning")
            return
        paths = [p for _, p in indexed]
        path_indices = [i for i, _ in indexed]
        self._queue_mode_active = True
        self._queue_results = []
        self._queue_status_by_index = {}
        self._queue_run_root_by_index = {}
        self._refresh_queue_list()
        self._last_run_was_script_mode = False
        self._bridge = None
        args = _WorkerArgs(
            run_mode="queue",
            eye_monitor_indices=eye_indices,
            script_raw="",
            script_disk_path=None,
            use_tool_cache=self._tool_cache_enabled_for_run(),
            queue_paths=list(paths),
            queue_path_indices=list(path_indices),
        )
        self._begin_worker_run(args)

    def _start_smart_run(self, eye_indices: list[int]) -> None:
        goal = normalize_smart_goal(self._smart_text.get("0.0", "end"))
        if not goal:
            show_ctk_message(
                self,
                "智能模式",
                "請先輸入要達成的目標文字。",
                kind="warning",
            )
            return
        if self._smart_goal_path is not None and self._is_smart_dirty():
            self._smart_save()
        if self._smart_goal_path is not None:
            script_disk_path = self._smart_goal_path
        else:
            self._sync_smart_text_to_cache()
            script_disk_path = self._smart_goal_cache_path()
        self._last_run_was_script_mode = False
        self._bridge = None
        args = _WorkerArgs(
            run_mode="smart",
            eye_monitor_indices=eye_indices,
            script_raw="",
            script_disk_path=script_disk_path,
            use_tool_cache=False,
            smart_goal=goal,
        )
        self._begin_worker_run(args)

    def _on_start_run(self) -> None:
        if self._recording_session.is_active():
            show_ctk_message(self, "執行", "請先停止錄製再開始執行。", kind="warning")
            return
        if self._is_recording_finalizing():
            show_ctk_message(self, "執行", "請等待錄製完成。", kind="warning")
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

        selected_tab = self._mode_tabs.get()
        raw = self._script_text.get("0.0", "end")
        if self._is_recording_script_open() and self._script_path is not None:
            rec = recording_run_dir(self._script_path)
            if rec is not None:
                raw = collect_recording_script_text(rec)
        steps = parse_executable_lines_from_text(raw)
        run_mode = resolve_hub_run_mode(
            selected_tab=selected_tab,
            script_has_steps=bool(steps),
        )

        if run_mode == "queue":
            self._start_queue_run(eye_indices)
            return
        if run_mode == "smart":
            self._start_smart_run(eye_indices)
            return

        if run_mode == "script":
            # Opened script file, cache transcript, or typed commands → run as one script task.
            if self._script_path is not None:
                script_disk_path = self._script_path
            else:
                self._sync_script_text_to_runtime_cache()
                script_disk_path = self._runtime_commands_cache_path
            self._last_run_was_script_mode = True
            self._bridge = None
            self._reset_script_step_status_tracking(raw)
            args = _WorkerArgs(
                run_mode="script",
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
                self._reset_textbox_undo(self._script_text)
            finally:
                self._suppress_script_cache_sync = False
            self._refresh_script_path_label()
            self._clear_script_step_statuses()
            self._refresh_script_line_numbers()
            self._mark_script_clean()

            args = _WorkerArgs(
                run_mode="runtime",
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
        html_path = run_root / "session_steps.html"
        if html_path.is_file():
            self._show_report_button(html_path)
        self._active_run_root = None

    def _set_queue_status(self, text: str) -> None:
        self.after(0, lambda: self._status.configure(text=text, text_color=("gray20", "gray65")))

    def _run_queue_worker(self, args: _WorkerArgs) -> None:
        paths = args.queue_paths or []
        path_indices = args.queue_path_indices
        total = len(paths)
        results: list[tuple[str, str]] = []
        settings = load_settings()
        runs_root = Path(settings.runs_dir)
        os.environ["CUA_WRITE_SESSION_REPORT"] = "1"
        try:
            for i, script_path in enumerate(paths, start=1):
                if self._user_requested_stop:
                    break
                wait_while_paused_blocking()
                if self._user_requested_stop:
                    break
                ui_index = (
                    path_indices[i - 1]
                    if path_indices is not None and i - 1 < len(path_indices)
                    else i - 1
                )
                script_path = resolve_runnable_script_path(script_path)
                name = script_display_name(script_path)
                try:
                    raw = load_runnable_script_text(script_path)
                except OSError as e:
                    results.append((name, "fail"))
                    self._mark_queue_result(ui_index, "fail")
                    self._set_queue_status(f"佇列執行 ({i}/{total})：{name} 無法讀取（{e}）")
                    continue
                steps = parse_executable_lines_from_text(raw)
                if not steps:
                    results.append((name, "skipped"))
                    self._mark_queue_result(ui_index, "skipped")
                    self._set_queue_status(f"佇列執行 ({i}/{total})：{name} 無可執行步驟，略過")
                    continue
                self._set_queue_status(f"佇列執行 ({i}/{total})：{name}")
                run_root_for_row: Path | None = None
                try:
                    rec = recording_run_dir(script_path)
                    folder_name = (
                        unique_named_run_folder(rec.name) if rec is not None else None
                    )
                    manager, paths_obj, run_id = prepare_run_session(
                        runs_root=runs_root,
                        task=steps[0],
                        runtime_mode=False,
                        selected_script_path=script_path,
                        script_steps=steps,
                        eye_monitor_indices=args.eye_monitor_indices,
                        clear_runs_root=False,
                        run_folder_name=folder_name,
                    )
                    run_root_for_row = paths_obj.root
                    self._active_run_root = paths_obj.root
                    manager.log_info(f"Queue starting coordinator for {name}")
                    run_coordinator_sync()
                    # Coordinator sets session_end_reason on the process-wide manager
                    # (get_run_state_manager), not the local prepare_run_session instance.
                    try:
                        end_reason = get_run_state_manager().session_end_reason
                    except RuntimeError:
                        end_reason = None
                    if end_reason is None and run_root_for_row is not None:
                        report = read_json(run_root_for_row / "report.json", {})
                        if isinstance(report, dict):
                            raw = report.get("session_end_reason")
                            end_reason = raw if isinstance(raw, str) else None
                    manager.log_info(
                        f"Queue script stopped (session_end_reason={end_reason!r})."
                    )
                    status = self._queue_status_from_session_end_reason(end_reason)
                    results.append((name, status))
                    self._mark_queue_result(ui_index, status, run_root_for_row)
                    if status == "fail":
                        self._set_queue_status(
                            f"佇列執行 ({i}/{total})：{name} 失敗（步驟未完成）"
                        )
                except asyncio.CancelledError:
                    results.append((name, "stopped"))
                    self._mark_queue_result(ui_index, "stopped", run_root_for_row)
                    break
                except BaseException as e:  # noqa: BLE001 - continue queue on any script failure
                    results.append((name, "fail"))
                    self._mark_queue_result(ui_index, "fail", run_root_for_row)
                    self._set_queue_status(f"佇列執行 ({i}/{total})：{name} 失敗（{e}）")
        finally:
            os.environ.pop("CUA_WRITE_SESSION_REPORT", None)
            self._active_run_root = None
        self._queue_results = results
        ok = sum(1 for _, s in results if s == "ok")
        fail = sum(1 for _, s in results if s == "fail")
        skipped = sum(1 for _, s in results if s == "skipped")
        self._worker_outcome = ("ok", f"佇列完成：成功 {ok}、失敗 {fail}、略過 {skipped}。")

    def _worker_main(self, args: _WorkerArgs) -> None:
        if args.use_tool_cache:
            os.environ[USE_TOOL_CACHE_ENV] = "1"
        else:
            os.environ.pop(USE_TOOL_CACHE_ENV, None)
        try:
            if args.run_mode == "queue" or args.queue_paths is not None:
                self._run_queue_worker(args)
                return
            if args.run_mode == "smart":
                settings = load_settings()
                runs_root = Path(settings.runs_dir)
                goal = normalize_smart_goal(args.smart_goal)
                folder_name = args.run_folder_name or unique_run_folder_name("smart")
                manager, paths, run_id = prepare_run_session(
                    runs_root=runs_root,
                    task=goal.splitlines()[0][:80] if goal else "smart",
                    runtime_mode=False,
                    selected_script_path=args.script_disk_path,
                    script_steps=None,
                    eye_monitor_indices=args.eye_monitor_indices,
                    clear_runs_root=False,
                    run_folder_name=folder_name,
                    smart_mode=True,
                    smart_goal=goal,
                )
                self._active_run_root = paths.root
                manager.log_info("Master starting smart coordinator")
                run_coordinator_sync(smart_mode=True)
                self._worker_outcome = ("ok", f"智能模式執行 {run_id} 已完成。")
                manager.log_info("Master stopped.")
                return
            if args.run_mode == "runtime" or args.step_mode:
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
                rec = recording_run_dir(script_path) if script_path is not None else None
                if rec is not None:
                    raw = collect_recording_script_text(rec)
                    script_path = rec
                steps = parse_executable_lines_from_text(raw)
                if rec is None:
                    if script_path is None:
                        fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="qa_script_", text=True)
                        os.close(fd)
                        script_path = Path(tmp)
                        self._post_run_unlink = script_path
                    script_path.write_text(raw.rstrip() + "\n", encoding="utf-8")
                task = steps[0]
                settings = load_settings()
                runs_root = Path(settings.runs_dir)
                folder_name = (
                    unique_named_run_folder(rec.name) if rec is not None else None
                )
                manager, paths, run_id = prepare_run_session(
                    runs_root=runs_root,
                    task=task,
                    runtime_mode=False,
                    selected_script_path=script_path,
                    script_steps=steps,
                    eye_monitor_indices=args.eye_monitor_indices,
                    clear_runs_root=False,
                    run_folder_name=folder_name,
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
        reset_run_control()
        self._set_run_button_idle()
        self._set_hub_controls_idle()

        if self._queue_mode_active:
            self._queue_mode_active = False
            self._active_run_root = None
            results = self._queue_results
            ok = sum(1 for _, s in results if s == "ok")
            fail = sum(1 for _, s in results if s == "fail")
            skipped = sum(1 for _, s in results if s == "skipped")
            stopped = sum(1 for _, s in results if s == "stopped")
            failed_names = [n for n, s in results if s == "fail"]
            summary = f"佇列完成：成功 {ok}、失敗 {fail}、略過 {skipped}。"
            if stopped:
                summary = f"佇列已停止：成功 {ok}、失敗 {fail}、略過 {skipped}。"
            if failed_names:
                summary += "\n失敗檔案：\n" + "\n".join(f"• {n}" for n in failed_names)
            self._status.configure(
                text=f"佇列已停止（成功 {ok}／失敗 {fail}）。"
                if stopped
                else f"佇列完成（成功 {ok}／失敗 {fail}）。"
            )
            show_ctk_message(
                self,
                "佇列執行",
                summary,
                kind="warning" if (fail or stopped) else "info",
            )
            return

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
