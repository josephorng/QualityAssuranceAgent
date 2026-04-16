from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def generate_run_folder_name(task: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", task.lower()).strip("-")
    if not stem:
        stem = "task"
    return f"{utc_stamp()}-{stem[:36]}"


@dataclass
class RunPaths:
    root: Path
    eye_dir: Path
    thinking_dir: Path
    storage_dir: Path
    hand_csv: Path
    brain_txt: Path
    storage_json: Path
    log_file: Path


def init_run_session(project_root: Path, task: str) -> RunPaths:
    runs_dir = project_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_root = runs_dir / generate_run_folder_name(task)

    eye_dir = run_root / "eye"
    thinking_dir = run_root / "thinking"
    storage_dir = run_root / "storage"
    eye_dir.mkdir(parents=True, exist_ok=True)
    thinking_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    hand_csv = run_root / "hand.csv"
    with hand_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "image_name", "tool", "arguments", "result"]
        )
        writer.writeheader()

    brain_txt = run_root / "brain.txt"
    brain_txt.write_text("", encoding="utf-8")

    storage_json = run_root / "storage.json"
    storage_json.write_text("[]", encoding="utf-8")

    log_file = run_root / f"{run_root.name}.log"
    log_file.write_text(
        json.dumps({"created_at": utc_stamp(), "task": task}) + "\n", encoding="utf-8"
    )

    return RunPaths(
        root=run_root,
        eye_dir=eye_dir,
        thinking_dir=thinking_dir,
        storage_dir=storage_dir,
        hand_csv=hand_csv,
        brain_txt=brain_txt,
        storage_json=storage_json,
        log_file=log_file,
    )
