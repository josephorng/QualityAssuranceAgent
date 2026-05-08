from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.common.io_utils import append_text, write_json


def slugify(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return safe or "task"


def ts_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class RunPaths:
    root: Path
    eye_dir: Path
    yolo_ocr_dir: Path
    yolo_ui_dir: Path
    storage_dir: Path
    hand_csv: Path
    storage_json: Path
    info_log: Path


class RunStateManager:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root
        self.paths: RunPaths | None = None

    def init_run(self, task_input: str, run_folder_name: str | None = None) -> RunPaths:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        folder_name = run_folder_name or f"{slugify(task_input)[:40]}_{ts_name()}"
        root = self.runs_root / folder_name
        eye_dir = root / "eye"
        yolo_ocr_dir = root / "yolo_ocr"
        yolo_ui_dir = root / "yolo_ui"
        storage_dir = root / "storage"
        hand_csv = root / "hand.csv"
        storage_json = root / "storage.json"
        info_log = root / "run.log"

        eye_dir.mkdir(parents=True, exist_ok=True)
        yolo_ocr_dir.mkdir(parents=True, exist_ok=True)
        yolo_ui_dir.mkdir(parents=True, exist_ok=True)
        storage_dir.mkdir(parents=True, exist_ok=True)
        if not hand_csv.exists():
            hand_csv.write_text("", encoding="utf-8")
        if not storage_json.exists():
            write_json(storage_json, [])
        if not info_log.exists():
            info_log.write_text("", encoding="utf-8")

        self.paths = RunPaths(
            root=root,
            eye_dir=eye_dir,
            yolo_ocr_dir=yolo_ocr_dir,
            yolo_ui_dir=yolo_ui_dir,
            storage_dir=storage_dir,
            hand_csv=hand_csv,
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
        caller_label = "unknown"
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_file = Path(frame.f_back.f_code.co_filename)
            caller_no_suffix = caller_file.with_suffix("")
            parts = caller_no_suffix.parts
            if "src" in parts:
                src_index = parts.index("src")
                caller_label = "/".join(parts[src_index + 1 :]) or caller_no_suffix.name
            else:
                caller_label = caller_no_suffix.name
        line = f"[{ts}] [{caller_label}] {text}"
        print(line)
        append_text(paths.info_log, line + "\n")

    def log_error(self, text: str) -> None:
        paths = self.require_paths()
        ts = datetime.now(timezone.utc).isoformat()
        caller_label = "unknown"
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_file = Path(frame.f_back.f_code.co_filename)
            caller_no_suffix = caller_file.with_suffix("")
            parts = caller_no_suffix.parts
            if "src" in parts:
                src_index = parts.index("src")
                caller_label = "/".join(parts[src_index + 1 :]) or caller_no_suffix.name
            else:
                caller_label = caller_no_suffix.name
        line = f"[{ts}] [ERROR] [{caller_label}] {text}"
        print(line)
        append_text(paths.info_log, line + "\n")

_manager: RunStateManager | None = None


def get_run_state_manager() -> RunStateManager:
    """Return the process-wide manager used by brain, eye, hand, and ollama client."""
    global _manager
    if _manager is None:
        from src.common.runtime_context import get_runtime_env

        run_root, _ = get_runtime_env()
        _manager = RunStateManager(run_root.parent)
    return _manager
