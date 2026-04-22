from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.io_utils import append_text, read_text, write_json


def slugify(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return safe or "task"


def ts_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class RunPaths:
    root: Path
    eye_dir: Path
    thinking_dir: Path
    yolo_ocr_dir: Path
    storage_dir: Path
    hand_csv: Path
    long_term_memory_txt: Path
    storage_json: Path
    info_log: Path


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
        yolo_ocr_dir = root / "yolo_ocr"
        storage_dir = root / "storage"
        hand_csv = root / "hand.csv"
        long_term_memory_txt = root / "long_term_memory.txt"
        storage_json = root / "storage.json"
        info_log = root / "run.log"

        eye_dir.mkdir(parents=True, exist_ok=True)
        thinking_dir.mkdir(parents=True, exist_ok=True)
        yolo_ocr_dir.mkdir(parents=True, exist_ok=True)
        storage_dir.mkdir(parents=True, exist_ok=True)
        if not long_term_memory_txt.exists():
            long_term_memory_txt.write_text("", encoding="utf-8")
        if not hand_csv.exists():
            hand_csv.write_text("", encoding="utf-8")
        if not storage_json.exists():
            write_json(storage_json, [])
        if not info_log.exists():
            info_log.write_text("", encoding="utf-8")

        self.paths = RunPaths(
            root=root,
            eye_dir=eye_dir,
            thinking_dir=thinking_dir,
            yolo_ocr_dir=yolo_ocr_dir,
            storage_dir=storage_dir,
            hand_csv=hand_csv,
            long_term_memory_txt=long_term_memory_txt,
            storage_json=storage_json,
            info_log=info_log,
        )
        self.log_info(f"Run initialized for task: {task_input}")
        return self.paths

    def require_paths(self) -> RunPaths:
        if self.paths is None:
            raise RuntimeError("Run state not initialized")
        return self.paths

    def log_info(self, text: str) -> None:
        paths = self.require_paths()
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] {text}")
        append_text(paths.info_log, f"[{ts}] {text}\n")

    def append_brain_memory(self, text: str) -> None:
        paths = self.require_paths()
        append_text(paths.long_term_memory_txt, text + "\n")
        current = read_text(paths.long_term_memory_txt)
        if len(current) > self.memory_max_chars:
            compact = current[-self.memory_max_chars :]
            paths.long_term_memory_txt.write_text(compact, encoding="utf-8")

    def write_thinking_record(self, screenshot_name: str, thought: str, decision: dict[str, Any]) -> Path:
        paths = self.require_paths()
        image_name = Path(screenshot_name).name
        out = paths.thinking_dir / Path(image_name).with_suffix(".json").name
        write_json(
            out,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "screenshot_name": image_name,
                "thought": thought,
                "decision": decision,
            },
        )
        return out


_manager: RunStateManager | None = None


def get_run_state_manager() -> RunStateManager:
    """Return the process-wide manager used by brain, eye, hand, and ollama client."""
    global _manager
    if _manager is None:
        from src.common.runtime_context import get_runtime_env
        from src.common.settings import load_settings

        settings = load_settings()
        run_root, _, _ = get_runtime_env()
        _manager = RunStateManager(run_root.parent, settings.brain_memory_max_chars)
    return _manager
