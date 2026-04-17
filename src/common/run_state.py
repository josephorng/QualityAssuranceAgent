from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.io_utils import append_text, read_text, write_json


def slugify(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return safe or "task"


def ts_name() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class RunPaths:
    root: Path
    eye_dir: Path
    thinking_dir: Path
    storage_dir: Path
    hand_csv: Path
    brain_txt: Path
    storage_json: Path
    debug_log: Path


class RunStateManager:
    def __init__(self, runs_root: Path, memory_max_chars: int = 16000) -> None:
        self.runs_root = runs_root
        self.memory_max_chars = memory_max_chars
        self.paths: RunPaths | None = None

    def init_run(self, task_input: str, run_folder_name: str | None = None) -> RunPaths:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        folder_name = run_folder_name or f"{slugify(task_input)[:40]}_{ts_name()}"
        root = self.runs_root / folder_name
        eye_dir = root / "eye"
        thinking_dir = root / "thinking"
        storage_dir = root / "storage"
        hand_csv = root / "hand.csv"
        brain_txt = root / "brain.txt"
        storage_json = root / "storage.json"
        debug_log = root / "run.log"

        eye_dir.mkdir(parents=True, exist_ok=True)
        thinking_dir.mkdir(parents=True, exist_ok=True)
        storage_dir.mkdir(parents=True, exist_ok=True)
        if not brain_txt.exists():
            brain_txt.write_text("", encoding="utf-8")
        if not hand_csv.exists():
            hand_csv.write_text("", encoding="utf-8")
        if not storage_json.exists():
            write_json(storage_json, [])
        if not debug_log.exists():
            debug_log.write_text("", encoding="utf-8")

        self.paths = RunPaths(
            root=root,
            eye_dir=eye_dir,
            thinking_dir=thinking_dir,
            storage_dir=storage_dir,
            hand_csv=hand_csv,
            brain_txt=brain_txt,
            storage_json=storage_json,
            debug_log=debug_log,
        )
        self.log_debug(f"Run initialized for task: {task_input}")
        return self.paths

    def require_paths(self) -> RunPaths:
        if self.paths is None:
            raise RuntimeError("Run state not initialized")
        return self.paths

    def log_debug(self, text: str) -> None:
        paths = self.require_paths()
        append_text(paths.debug_log, f"[{datetime.utcnow().isoformat()}] {text}\n")

    def append_brain_memory(self, text: str) -> None:
        paths = self.require_paths()
        append_text(paths.brain_txt, text + "\n")
        current = read_text(paths.brain_txt)
        if len(current) > self.memory_max_chars:
            compact = current[-self.memory_max_chars :]
            paths.brain_txt.write_text(compact, encoding="utf-8")

    def write_thinking_record(self, screenshot_name: str, thought: str, decision: dict[str, Any]) -> Path:
        paths = self.require_paths()
        out = paths.thinking_dir / f"{ts_name()}_{screenshot_name}.json"
        write_json(
            out,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "screenshot_name": screenshot_name,
                "thought": thought,
                "decision": decision,
            },
        )
        return out
