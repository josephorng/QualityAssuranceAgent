from __future__ import annotations

import json
import os
import signal
import shutil
import asyncio
from pathlib import Path

from src.common.monitor_prompt import prompt_eye_monitor_index
from src.common.run_state import RunStateManager
from src.common.script_helper import list_script_files, parse_script_lines
from src.runtime.coordinator import RuntimeCoordinator
from src.common.runtime_context import (
    SCRIPT_LINES_ENV,
    SCRIPT_PATH_ENV,
    set_runtime_env,
)
from src.common.settings import ROOT_DIR, load_settings


def clear_runs_folder(runs_root: Path) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    for item in runs_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def select_script_input(root_dir: Path) -> tuple[str, Path, list[str]]:
    scripts_dir = root_dir / "scripts"
    scripts = list_script_files(scripts_dir)
    if not scripts:
        raise RuntimeError("No .txt scripts found in scripts/ directory.")

    while True:
        print("Available scripts:")
        for idx, script in enumerate(scripts, start=1):
            print(f"  {idx}) {script.name}")
        selected = input("Select script number: ").strip()
        if not selected.isdigit():
            print("Please enter a valid number.")
            continue
        selected_index = int(selected) - 1
        if selected_index < 0 or selected_index >= len(scripts):
            print("Selected number is out of range.")
            continue
        script_path = scripts[selected_index]
        script_steps = parse_script_lines(script_path)
        if not script_steps:
            print("Selected script has no executable lines. Add steps and try again.")
            continue
        task = script_steps[0]
        return task, script_path, script_steps


def main() -> None:
    task, selected_script_path, script_steps = select_script_input(ROOT_DIR)

    settings = load_settings()
    runs_root = Path(settings.runs_dir)
    clear_runs_folder(runs_root)
    manager = RunStateManager(runs_root=runs_root)
    paths = manager.init_run(task)
    run_id = paths.root.name

    set_runtime_env(paths.root, task, run_id)
    os.environ[SCRIPT_PATH_ENV] = str(selected_script_path)
    os.environ[SCRIPT_LINES_ENV] = json.dumps(script_steps, ensure_ascii=False)
    eye_monitor_index = prompt_eye_monitor_index()
    os.environ["EYE_MONITOR_INDEX"] = str(eye_monitor_index)
    print(f"[master] Eye capture monitor index: {eye_monitor_index} (0 = all screens)")
    manager.log_info(f"Eye capture monitor index set to {eye_monitor_index}")

    try:
        manager.log_info("Master starting coordinator module runtime")
        coordinator = RuntimeCoordinator()
        asyncio.run(coordinator.run())
    except KeyboardInterrupt:
        manager.log_info("KeyboardInterrupt received. shutting down coordinator.")
    finally:
        manager.log_info("Master stopped.")


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
