from __future__ import annotations

import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

from src.common.settings import ROOT_DIR


LOG_LINE_RE = re.compile(r"^\[(?P<timestamp>[^\]]+)\]\s+\[(?P<source>[^\]]+)\]\s*(?P<message>.*)$")


@dataclass(frozen=True)
class LogEntry:
    raw: str
    timestamp: str
    source: str
    message: str


def discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(
        [item for item in runs_root.iterdir() if item.is_dir() and (item / "run.log").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def parse_run_log(path: Path) -> list[LogEntry]:
    if not path.exists():
        return []
    entries: list[LogEntry] = []
    last_timestamp = ""
    last_source = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_LINE_RE.match(line)
        if match is None:
            # Treat non-header lines as continuations of the previous structured log entry.
            entries.append(
                LogEntry(
                    raw=line,
                    timestamp=last_timestamp,
                    source=last_source or "unparsed",
                    message=line,
                )
            )
            continue
        last_timestamp = match.group("timestamp")
        last_source = match.group("source")
        entries.append(
            LogEntry(
                raw=line,
                timestamp=last_timestamp,
                source=last_source,
                message=match.group("message"),
            )
        )
    return entries


class RunLogViewerApp:
    def __init__(self, root: tk.Tk, runs_root: Path):
        self.root = root
        self.runs_root = runs_root
        self.run_dirs: list[Path] = []
        self.current_entries: list[LogEntry] = []
        self.current_sources: list[str] = []
        self.current_run_log: Path | None = None
        self.auto_refresh_job: str | None = None
        self.last_loaded_mtime_ns: int | None = None

        self.status_var = tk.StringVar(value="Ready")
        self.search_var = tk.StringVar(value="")
        self.auto_refresh_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.refresh_runs(select_first=True)
        self._start_auto_refresh()

    def _build_ui(self) -> None:
        self.root.title("Run Log Viewer")
        self.root.geometry("1320x860")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Runs (folders with run.log)").grid(row=0, column=0, sticky="w")
        self.run_list = tk.Listbox(left, exportselection=False, width=56, height=14)
        self.run_list.grid(row=1, column=0, sticky="nsew")
        self.run_list.bind("<<ListboxSelect>>", self._on_run_selected)

        run_buttons = ttk.Frame(left)
        run_buttons.grid(row=2, column=0, sticky="ew", pady=(6, 8))
        run_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(run_buttons, text="Refresh Runs", command=self.refresh_runs).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(run_buttons, text="Reload Log", command=self.reload_selected_log).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(left, text="Sources (multi-select)").grid(row=3, column=0, sticky="w")
        self.source_list = tk.Listbox(left, selectmode=tk.EXTENDED, exportselection=False, width=56, height=16)
        self.source_list.grid(row=4, column=0, sticky="nsew")
        self.source_list.bind("<<ListboxSelect>>", self._on_filters_changed)

        source_buttons = ttk.Frame(left)
        source_buttons.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        source_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(source_buttons, text="Select All", command=self._select_all_sources).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(source_buttons, text="Clear All", command=self._clear_all_sources).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        controls = ttk.Frame(right)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Search text:").grid(row=0, column=0, sticky="w")
        search_entry = ttk.Entry(controls, textvariable=self.search_var)
        search_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        search_entry.bind("<KeyRelease>", self._on_filters_changed)

        ttk.Checkbutton(
            controls,
            text="Auto refresh (2s)",
            variable=self.auto_refresh_var,
            command=self._on_auto_refresh_toggle,
        ).grid(row=0, column=2, sticky="w", padx=(10, 0))

        text_wrap = ttk.Frame(right)
        text_wrap.grid(row=1, column=0, sticky="nsew")
        text_wrap.rowconfigure(0, weight=1)
        text_wrap.columnconfigure(0, weight=1)

        self.text = tk.Text(text_wrap, wrap="none", font=("Consolas", 10))
        y_scroll = ttk.Scrollbar(text_wrap, orient="vertical", command=self.text.yview)
        x_scroll = ttk.Scrollbar(text_wrap, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

    def refresh_runs(self, select_first: bool = False) -> None:
        selected_name = self._selected_run_name()
        self.run_dirs = discover_runs(self.runs_root)
        self.run_list.delete(0, tk.END)
        for run_dir in self.run_dirs:
            self.run_list.insert(tk.END, run_dir.name)

        if not self.run_dirs:
            self._set_text("")
            self.status_var.set(f"No run folders with run.log found in {self.runs_root}")
            self.current_run_log = None
            self.current_entries = []
            self.current_sources = []
            self.source_list.delete(0, tk.END)
            self.last_loaded_mtime_ns = None
            return

        chosen_idx = 0 if select_first else None
        if selected_name:
            for idx, run_dir in enumerate(self.run_dirs):
                if run_dir.name == selected_name:
                    chosen_idx = idx
                    break
        if chosen_idx is None:
            chosen_idx = 0

        self.run_list.selection_clear(0, tk.END)
        self.run_list.selection_set(chosen_idx)
        self.run_list.see(chosen_idx)
        self._on_run_selected()

    def _selected_run_name(self) -> str | None:
        selected = self.run_list.curselection()
        if not selected:
            return None
        idx = selected[0]
        if idx >= len(self.run_dirs):
            return None
        return self.run_dirs[idx].name

    def _on_run_selected(self, _event: object | None = None) -> None:
        selected = self.run_list.curselection()
        if not selected:
            return
        idx = selected[0]
        if idx >= len(self.run_dirs):
            return
        run_dir = self.run_dirs[idx]
        self.current_run_log = run_dir / "run.log"
        self.last_loaded_mtime_ns = None
        self.reload_selected_log(select_all_sources=True)

    def reload_selected_log(self, select_all_sources: bool = False) -> None:
        if self.current_run_log is None:
            return
        self.current_entries = parse_run_log(self.current_run_log)
        try:
            self.last_loaded_mtime_ns = self.current_run_log.stat().st_mtime_ns
        except OSError:
            self.last_loaded_mtime_ns = None

        old_selected_sources = self._selected_sources()
        self.current_sources = sorted({entry.source for entry in self.current_entries})
        self.source_list.delete(0, tk.END)
        for source in self.current_sources:
            self.source_list.insert(tk.END, source)

        if self.current_sources:
            if select_all_sources:
                self._select_all_sources()
            else:
                for i, source in enumerate(self.current_sources):
                    if source in old_selected_sources:
                        self.source_list.selection_set(i)
                if not self.source_list.curselection():
                    self._select_all_sources()

        self._render_filtered_entries()

    def _selected_sources(self) -> set[str]:
        indices = self.source_list.curselection()
        return {self.source_list.get(i) for i in indices}

    def _on_filters_changed(self, _event: object | None = None) -> None:
        self._render_filtered_entries()

    def _render_filtered_entries(self) -> None:
        allowed_sources = self._selected_sources()
        search_text = self.search_var.get().strip().lower()

        filtered: list[str] = []
        for entry in self.current_entries:
            if allowed_sources and entry.source not in allowed_sources:
                continue
            if search_text and search_text not in entry.raw.lower():
                continue
            filtered.append(entry.raw)

        self._set_text("\n".join(filtered))
        run_name = self.current_run_log.parent.name if self.current_run_log else "-"
        self.status_var.set(
            f"Run: {run_name} | Showing {len(filtered)}/{len(self.current_entries)} lines | Sources selected: {len(allowed_sources)}"
        )

    def _set_text(self, content: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        if content:
            self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def _select_all_sources(self) -> None:
        self.source_list.selection_set(0, tk.END)
        self._render_filtered_entries()

    def _clear_all_sources(self) -> None:
        self.source_list.selection_clear(0, tk.END)
        self._render_filtered_entries()

    def _on_auto_refresh_toggle(self) -> None:
        if self.auto_refresh_var.get():
            self._start_auto_refresh()
        else:
            self._stop_auto_refresh()

    def _start_auto_refresh(self) -> None:
        self._stop_auto_refresh()
        self.auto_refresh_job = self.root.after(2000, self._auto_refresh_tick)

    def _stop_auto_refresh(self) -> None:
        if self.auto_refresh_job is not None:
            self.root.after_cancel(self.auto_refresh_job)
            self.auto_refresh_job = None

    def _auto_refresh_tick(self) -> None:
        if self.auto_refresh_var.get():
            self._refresh_if_log_changed()
            self.auto_refresh_job = self.root.after(2000, self._auto_refresh_tick)
        else:
            self.auto_refresh_job = None

    def _refresh_if_log_changed(self) -> None:
        if self.current_run_log is None:
            return
        try:
            mtime_ns = self.current_run_log.stat().st_mtime_ns
        except OSError:
            return
        if self.last_loaded_mtime_ns is None or mtime_ns != self.last_loaded_mtime_ns:
            self.reload_selected_log()


def run_app(runs_root: Path | None = None) -> None:
    base = runs_root if runs_root is not None else ROOT_DIR / "runs"
    root = tk.Tk()
    RunLogViewerApp(root, base)
    root.mainloop()


if __name__ == "__main__":
    run_app()
