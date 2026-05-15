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

from main import prepare_run_session, run_coordinator_sync
from src.common.ctk_dialogs import show_ctk_message
from src.common.io_utils import append_text, read_json, write_json
from src.common.monitor_prompt import (
    PRIMARY_MONITOR_MARKER,
    EyeMonitorChoice,
    list_eye_monitor_choices,
)
from src.common.run_state import unique_run_folder_name
from src.common.runtime_command_dialog import (
    RuntimeCommandHubBridge,
    consume_runtime_user_ended_at_prompt,
    reset_runtime_user_ended_at_prompt,
)
from src.common.script_helper import parse_executable_lines_from_text
from src.common.settings import ROOT_DIR, load_settings

# Step-mode runtime command transcript (append during run); not hub UI preferences.
_RUNTIME_COMMAND_TRANSCRIPT_NAME = "runtime_commands_cache.txt"
_HUB_UI_STATE_NAME = "hub_ui.json"
_HUB_UI_VERSION = 1


def _default_hub_ui_dict() -> dict[str, Any]:
    return {
        "version": _HUB_UI_VERSION,
        "appearance_dark": True,
        "selected_monitor_indices": [],
        "last_script_path": None,
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


class MainHub(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("電腦使用代理")
        self.geometry("960x780")
        self.minsize(880, 880)

        hub = _read_hub_ui_state()
        self._remember_monitor_indices: list[int] = list(hub["selected_monitor_indices"])
        self._appearance_dark = bool(hub["appearance_dark"])
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
                self._script_text.delete("0.0", "end")
                self._script_text.insert("0.0", text)
                self._script_path_label.configure(text=str(p.resolve()))
        if self._script_path is None:
            self._try_load_last_runtime_command_cache()

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
        self._script_text.delete("0.0", "end")
        self._script_text.insert("0.0", raw)
        self._script_path_label.configure(text=str(cache_path.resolve()))

    def _append_runtime_command_to_script_view(self, cmd: str) -> None:
        """Underlying Tk Text ignores ``insert`` while the widget is ``disabled`` (as during a run)."""
        self._script_text.configure(state="normal")
        self._script_text.insert("end", cmd + "\n")
        self._script_text.configure(state="disabled")

    def _refresh_runtime_script_text_from_cache(self) -> None:
        """After a runtime-command run, reload the cache file into the script textbox (disk is source of truth)."""
        p = self._runtime_commands_cache_path
        if p is None or not p.is_file():
            return
        self._script_text.delete("0.0", "end")
        self._script_text.insert("0.0", p.read_text(encoding="utf-8"))
        self._script_path_label.configure(text=str(p.resolve()))

    def _build_header(self) -> None:
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(20, 8))

        top_row = ctk.CTkFrame(head, fg_color="transparent")
        top_row.pack(fill="x")

        theme_row = ctk.CTkFrame(top_row, fg_color="transparent")
        theme_row.pack(side="right", anchor="ne")
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
            }
            write_json(_hub_ui_state_path(), data)
            self._remember_monitor_indices = list(data["selected_monitor_indices"])
        except OSError:
            pass

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
        b_clear.pack(side="left")
        self._script_path_label = ctk.CTkLabel(
            box,
            text="未載入檔案",
            font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray65"),
        )
        self._script_path_label.pack(anchor="w", padx=16, pady=(4, 8))
        self._script_text = ctk.CTkTextbox(box, font=ctk.CTkFont(size=14), wrap="word")
        self._script_text.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self._script_controls.extend([b_open, b_save, b_sas, b_clear, self._script_text])

    def _build_actions_row(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(12, 8))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(2, weight=1)
        self._run_btn = ctk.CTkButton(
            row,
            text="開始執行",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=200,
            command=self._on_start_run,
        )
        self._run_btn.grid(row=0, column=1)

    def _set_run_button_idle(self) -> None:
        self._run_btn.configure(text="開始執行", command=self._on_start_run, state="normal")

    def _set_run_button_running(self) -> None:
        self._run_btn.configure(text="停止執行", command=self._on_stop_run, state="normal")

    def _build_status(self) -> None:
        self._status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13))
        self._status.pack(anchor="w", padx=28, pady=(0, 16))

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
        self._script_text.delete("0.0", "end")
        self._script_text.insert("0.0", text)
        self._script_path_label.configure(text=str(p.resolve()))
        self._persist_hub_ui_state()

    def _script_save(self) -> None:
        body = self._script_text.get("0.0", "end").rstrip() + "\n"
        if self._script_path is not None:
            self._script_path.write_text(body, encoding="utf-8")
            self._status.configure(text=f"已儲存 {self._script_path.name}")
            self._persist_hub_ui_state()
            return
        if self._runtime_commands_cache_path is not None:
            self._runtime_commands_cache_path.write_text(body, encoding="utf-8")
            self._status.configure(text=f"已儲存 {self._runtime_commands_cache_path.name}")
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
        self._script_path_label.configure(text=str(p.resolve()))
        self._status.configure(text=f"已另存新檔 {p.name}")
        self._persist_hub_ui_state()

    def _script_clear(self) -> None:
        """Unload any opened path / cache binding and empty the script editor."""
        self._script_path = None
        self._runtime_commands_cache_path = None
        self._script_text.configure(state="normal")
        self._script_text.delete("0.0", "end")
        self._script_path_label.configure(text="未載入檔案")
        self._status.configure(text="")
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

    def _on_start_run(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._user_requested_stop = False
        self._post_run_unlink = None
        # Script file on disk (Open or Save as) → script mode; otherwise step-by-step (runtime commands).
        step_mode = self._script_path is None
        eye_indices = self._selected_monitor_indices()
        if not eye_indices:
            show_ctk_message(
                self,
                "顯示器",
                "請至少選擇一台要截取的顯示器。",
                kind="warning",
            )
            return

        if step_mode:
            settings = load_settings()
            runs_root = Path(settings.runs_dir)
            runs_root.mkdir(parents=True, exist_ok=True)
            cache_path = runs_root / _RUNTIME_COMMAND_TRANSCRIPT_NAME
            self._runtime_commands_cache_path = cache_path
            cache_path.write_text("", encoding="utf-8")
            self._script_text.configure(state="normal")
            self._script_text.delete("0.0", "end")
            self._script_path_label.configure(text=str(cache_path.resolve()))

            args = _WorkerArgs(
                step_mode=True,
                eye_monitor_indices=eye_indices,
                script_raw="",
                script_disk_path=None,
            )

            def on_runtime_command(cmd: str) -> None:
                append_text(cache_path, cmd + "\n")
                self._append_runtime_command_to_script_view(cmd)

            self._bridge = RuntimeCommandHubBridge(self, on_runtime_command=on_runtime_command)
            self._bridge.start()
        else:
            raw = self._script_text.get("0.0", "end")
            steps = parse_executable_lines_from_text(raw)
            if not steps:
                show_ctk_message(
                    self,
                    "執行",
                    "腳本沒有可執行行數（為空或僅有 # 註解）。",
                    kind="warning",
                )
                return
            self._runtime_commands_cache_path = None
            args = _WorkerArgs(
                step_mode=False,
                eye_monitor_indices=eye_indices,
                script_raw=raw,
                script_disk_path=self._script_path,
            )
            self._bridge = None

        self._set_run_button_running()
        for cb in self._monitor_checkboxes:
            cb.configure(state="disabled")
        self._monitor_refresh_btn.configure(state="disabled")
        for w in self._script_controls:
            w.configure(state="disabled")
        self._status.configure(text="執行中…")

        self._worker_thread = threading.Thread(target=self._worker_main, args=(args,), daemon=True)
        self._worker_thread.start()
        self.after(80, self._poll_worker_finished)
        self.after_idle(self.iconify)

    def _worker_main(self, args: _WorkerArgs) -> None:
        try:
            if args.step_mode:
                reset_runtime_user_ended_at_prompt()
                settings = load_settings()
                runs_root = Path(settings.runs_dir)
                folder_name = unique_run_folder_name("runtime_command")
                manager, _, run_id = prepare_run_session(
                    runs_root=runs_root,
                    task="runtime_command",
                    runtime_mode=True,
                    selected_script_path=None,
                    script_steps=None,
                    eye_monitor_indices=args.eye_monitor_indices,
                    clear_runs_root=False,
                    run_folder_name=folder_name,
                )
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
                manager, _, run_id = prepare_run_session(
                    runs_root=runs_root,
                    task=task,
                    runtime_mode=False,
                    selected_script_path=script_path,
                    script_steps=steps,
                    eye_monitor_indices=args.eye_monitor_indices,
                    clear_runs_root=False,
                    run_folder_name=None,
                )
                manager.log_info("Master starting coordinator module runtime")
                run_coordinator_sync()
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
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass
        self._set_run_button_idle()
        for cb in self._monitor_checkboxes:
            cb.configure(state="normal")
        self._monitor_refresh_btn.configure(state="normal")
        for w in self._script_controls:
            w.configure(state="normal")
        self._refresh_runtime_script_text_from_cache()
        if kind == "err":
            self._status.configure(text=f"錯誤：{msg}")
        elif user_stopped:
            self._status.configure(text="執行已停止。")
        elif kind == "ok_quiet" and msg.strip():
            self._status.configure(text=msg.strip())
        elif kind in ("ok", "ok_quiet"):
            self._status.configure(text="就緒")
        if user_stopped and kind != "err":
            return
        if kind == "ok":
            show_ctk_message(self, "電腦使用代理", msg, kind="info")
        elif kind == "err":
            show_ctk_message(self, "電腦使用代理", msg, kind="error")


def run_main_hub() -> None:
    app = MainHub()
    app.mainloop()


if __name__ == "__main__":
    run_main_hub()
