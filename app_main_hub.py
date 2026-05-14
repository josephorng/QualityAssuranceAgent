"""CustomTkinter hub: script vs step-by-step runs with monitor selection."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import customtkinter as ctk
from tkinter import filedialog

from main import prepare_run_session, run_coordinator_sync
from src.common.ctk_dialogs import show_ctk_message
from src.common.monitor_prompt import EyeMonitorChoice, list_eye_monitor_choices
from src.common.run_state import unique_run_folder_name
from src.common.runtime_command_dialog import RuntimeCommandHubBridge
from src.common.script_helper import parse_executable_lines_from_text
from src.common.settings import ROOT_DIR, load_settings


@dataclass
class _WorkerArgs:
    step_mode: bool
    eye_index: int
    script_raw: str
    script_disk_path: Path | None


class MainHub(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Quality Assurance Agent")
        self.geometry("960x780")
        self.minsize(880, 880)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._script_path: Path | None = None
        self._worker_thread: threading.Thread | None = None
        self._bridge: RuntimeCommandHubBridge | None = None
        self._worker_outcome: tuple[str, str] = ("ok", "")
        self._last_commands_file: Path | None = None

        self._monitor_labels: list[str] = []
        self._monitor_indices: list[int] = []

        self._post_run_unlink: Path | None = None
        self._script_controls: list[Any] = []
        self._step_controls: list[Any] = []

        self._build_header()
        self._build_monitor_row()
        self._build_mode_row()
        self._build_script_section()
        self._build_step_section()
        self._build_actions_row()
        self._build_status()

        self._on_mode_change("Script file")
        self._refresh_monitors()

    def _build_header(self) -> None:
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(20, 8))
        ctk.CTkLabel(
            head,
            text="Quality Assurance Agent",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            head,
            text="Configure a run, choose the capture monitor, then start.",
            font=ctk.CTkFont(size=14),
            text_color=("gray30", "gray70"),
        ).pack(anchor="w", pady=(4, 0))

        theme_row = ctk.CTkFrame(head, fg_color="transparent")
        theme_row.pack(anchor="w", pady=(12, 0))
        ctk.CTkLabel(theme_row, text="Appearance:").pack(side="left", padx=(0, 8))
        self._theme_menu = ctk.CTkOptionMenu(
            theme_row,
            values=["Dark", "Light", "System"],
            command=self._on_theme_change,
            width=120,
        )
        self._theme_menu.set("Dark")
        self._theme_menu.pack(side="left")

    def _on_theme_change(self, value: str) -> None:
        ctk.set_appearance_mode(value.lower())

    def _build_monitor_row(self) -> None:
        box = ctk.CTkFrame(self, corner_radius=12)
        box.pack(fill="x", padx=24, pady=8)
        ctk.CTkLabel(box, text="Eye capture monitor", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=16, pady=(14, 4)
        )
        ctk.CTkLabel(
            box,
            text="Screenshots and coordinates use this region.",
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70"),
        ).pack(anchor="w", padx=16, pady=(0, 8))
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(0, 14))
        self._monitor_menu = ctk.CTkOptionMenu(row, values=["(loading…)"], width=560)
        self._monitor_menu.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Refresh", width=100, command=self._refresh_monitors).pack(
            side="left", padx=(10, 0)
        )

    def _refresh_monitors(self) -> None:
        try:
            choices = list_eye_monitor_choices()
        except Exception as e:
            show_ctk_message(self, "Monitors", f"Could not list displays:\n{e}", kind="error")
            choices = [EyeMonitorChoice(0, "All screens (fallback)", "—")]
        self._monitor_labels = [self._format_monitor_row(c) for c in choices]
        self._monitor_indices = [c.index for c in choices]
        self._monitor_menu.configure(values=self._monitor_labels)
        self._monitor_menu.set(self._monitor_labels[0])

    @staticmethod
    def _format_monitor_row(c: EyeMonitorChoice) -> str:
        return f"[{c.index}] {c.title} — {c.detail}"

    def _selected_monitor_index(self) -> int:
        label = self._monitor_menu.get()
        try:
            i = self._monitor_labels.index(label)
        except ValueError:
            return self._monitor_indices[0] if self._monitor_indices else 1
        return self._monitor_indices[i]

    def _build_mode_row(self) -> None:
        box = ctk.CTkFrame(self, corner_radius=12)
        box.pack(fill="x", padx=24, pady=8)
        ctk.CTkLabel(box, text="Run mode", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=16, pady=(14, 8)
        )
        self._mode_seg = ctk.CTkSegmentedButton(
            box,
            values=["Script file", "Step-by-step"],
            command=self._on_mode_change,
            height=36,
        )
        self._mode_seg.pack(fill="x", padx=16, pady=(0, 14))
        self._mode_seg.set("Script file")

    def _on_mode_change(self, value: str) -> None:
        script = value == "Script file"
        self._script_section.pack_forget()
        self._step_section.pack_forget()
        if script:
            self._script_section.pack(fill="both", expand=True, padx=24, pady=8)
        else:
            self._step_section.pack(fill="x", padx=24, pady=8)
        for w in self._script_controls:
            w.configure(state="normal" if script else "disabled")
        for w in self._step_controls:
            w.configure(state="normal" if not script else "disabled")

    def _build_script_section(self) -> None:
        box = ctk.CTkFrame(self, corner_radius=12)
        self._script_section = box
        ctk.CTkLabel(box, text="Script mode", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=16, pady=(14, 4)
        )
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=4)
        b_open = ctk.CTkButton(row, text="Open…", width=100, command=self._script_open)
        b_open.pack(side="left", padx=(0, 8))
        b_save = ctk.CTkButton(row, text="Save", width=100, command=self._script_save)
        b_save.pack(side="left", padx=(0, 8))
        b_sas = ctk.CTkButton(row, text="Save as…", width=100, command=self._script_save_as)
        b_sas.pack(side="left")
        self._script_path_label = ctk.CTkLabel(
            box,
            text="No file loaded",
            font=ctk.CTkFont(size=12),
            text_color=("gray20", "gray65"),
        )
        self._script_path_label.pack(anchor="w", padx=16, pady=(4, 8))
        self._script_text = ctk.CTkTextbox(box, font=ctk.CTkFont(size=14), wrap="word")
        self._script_text.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self._script_controls.extend([b_open, b_save, b_sas, self._script_text])

    def _build_step_section(self) -> None:
        box = ctk.CTkFrame(self, corner_radius=12)
        self._step_section = box
        ctk.CTkLabel(box, text="Step-by-step mode", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=16, pady=(14, 4)
        )
        ctk.CTkLabel(
            box,
            text="Each run is stored under the default runs folder (see settings / runs_dir). "
            "Commands are written to runtime_commands.txt in that run folder. "
            "Use Save commands as… to copy that file elsewhere.",
            font=ctk.CTkFont(size=13),
            text_color=("gray20", "gray65"),
            wraplength=860,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(4, 12))
        b_save_cmd = ctk.CTkButton(
            box,
            text="Save commands as…",
            width=180,
            command=self._save_commands_as,
        )
        b_save_cmd.pack(anchor="w", padx=16, pady=(0, 14))
        self._step_controls.append(b_save_cmd)

    def _build_actions_row(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(12, 8))
        self._run_btn = ctk.CTkButton(
            row,
            text="Start run",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=44,
            width=200,
            command=self._on_start_run,
        )
        self._run_btn.pack(side="left")

    def _build_status(self) -> None:
        self._status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13))
        self._status.pack(anchor="w", padx=28, pady=(0, 16))

    def _script_open(self) -> None:
        initial = ROOT_DIR / "scripts"
        path = filedialog.askopenfilename(
            parent=self,
            title="Open script",
            initialdir=str(initial) if initial.is_dir() else str(ROOT_DIR),
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        self._script_path = p
        text = p.read_text(encoding="utf-8")
        self._script_text.delete("0.0", "end")
        self._script_text.insert("0.0", text)
        self._script_path_label.configure(text=str(p.resolve()))

    def _script_save(self) -> None:
        if self._script_path is None:
            self._script_save_as()
            return
        self._script_path.write_text(self._script_text.get("0.0", "end").rstrip() + "\n", encoding="utf-8")
        self._status.configure(text=f"Saved {self._script_path.name}")

    def _script_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save script as",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
            initialdir=str(ROOT_DIR / "scripts"),
        )
        if not path:
            return
        p = Path(path)
        p.write_text(self._script_text.get("0.0", "end").rstrip() + "\n", encoding="utf-8")
        self._script_path = p
        self._script_path_label.configure(text=str(p.resolve()))
        self._status.configure(text=f"Saved as {p.name}")

    def _save_commands_as(self) -> None:
        src = self._last_commands_file
        if src is None or not src.is_file():
            show_ctk_message(
                self,
                "Save commands",
                "No runtime_commands.txt yet. Finish a step-by-step run first, or use Save commands as after a run.",
                kind="info",
            )
            return
        dest = filedialog.asksaveasfilename(
            parent=self,
            title="Save commands as",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if not dest:
            return
        shutil.copy2(src, dest)
        self._status.configure(text=f"Commands copied to {Path(dest).name}")

    def _on_start_run(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._post_run_unlink = None
        step_mode = self._mode_seg.get() == "Step-by-step"
        eye = self._selected_monitor_index()

        if step_mode:
            args = _WorkerArgs(
                step_mode=True,
                eye_index=eye,
                script_raw="",
                script_disk_path=None,
            )
            self._bridge = RuntimeCommandHubBridge(self)
            self._bridge.start()
        else:
            raw = self._script_text.get("0.0", "end")
            steps = parse_executable_lines_from_text(raw)
            if not steps:
                show_ctk_message(
                    self,
                    "Run",
                    "Script has no executable lines (empty or only # comments).",
                    kind="warning",
                )
                return
            args = _WorkerArgs(
                step_mode=False,
                eye_index=eye,
                script_raw=raw,
                script_disk_path=self._script_path,
            )
            self._bridge = None

        self._run_btn.configure(state="disabled")
        self._mode_seg.configure(state="disabled")
        self._monitor_menu.configure(state="disabled")
        for w in self._script_controls:
            w.configure(state="disabled")
        for w in self._step_controls:
            w.configure(state="disabled")
        self._status.configure(text="Running…")

        self._worker_thread = threading.Thread(target=self._worker_main, args=(args,), daemon=True)
        self._worker_thread.start()
        self.after(80, self._poll_worker_finished)
        self.after_idle(self.iconify)

    def _worker_main(self, args: _WorkerArgs) -> None:
        try:
            if args.step_mode:
                settings = load_settings()
                runs_root = Path(settings.runs_dir)
                folder_name = unique_run_folder_name("runtime_command")
                manager, paths, run_id = prepare_run_session(
                    runs_root=runs_root,
                    task="runtime_command",
                    runtime_mode=True,
                    selected_script_path=None,
                    script_steps=None,
                    eye_monitor_index=args.eye_index,
                    clear_runs_root=False,
                    run_folder_name=folder_name,
                )
                self._last_commands_file = paths.root / "runtime_commands.txt"
                manager.log_info("Master starting coordinator module runtime")
                run_coordinator_sync()
                self._worker_outcome = ("ok", f"Run {run_id} finished.")
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
                    eye_monitor_index=args.eye_index,
                    clear_runs_root=False,
                    run_folder_name=None,
                )
                self._last_commands_file = None
                manager.log_info("Master starting coordinator module runtime")
                run_coordinator_sync()
                self._worker_outcome = ("ok", f"Run {run_id} finished.")
                manager.log_info("Master stopped.")
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
        kind, msg = self._worker_outcome
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass
        self._run_btn.configure(state="normal")
        self._mode_seg.configure(state="normal")
        self._monitor_menu.configure(state="normal")
        self._on_mode_change(self._mode_seg.get())
        self._status.configure(text="Ready" if kind == "ok" else f"Error: {msg}")
        if kind == "ok":
            show_ctk_message(self, "QualityAssuranceAgent", msg, kind="info")
        else:
            show_ctk_message(self, "QualityAssuranceAgent", msg, kind="error")


def run_main_hub() -> None:
    app = MainHub()
    app.mainloop()


if __name__ == "__main__":
    run_main_hub()
