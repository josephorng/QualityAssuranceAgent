from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import shutil
import signal
from pathlib import Path

from src.common.monitor_prompt import prompt_eye_monitor_index
from src.common.run_state import RunStateManager, RunPaths, reset_run_state_manager
from src.common.script_helper import list_script_files, parse_script_lines
from src.common.settings import ROOT_DIR, load_settings
from src.runtime.coordinator import RuntimeCoordinator
from src.common.runtime_context import (
    RUNTIME_COMMAND_MODE_ENV,
    SCRIPT_LINES_ENV,
    SCRIPT_PATH_ENV,
    set_runtime_env,
)


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


def prompt_run_mode() -> bool:
    """Return True if the user chose runtime command mode (one command per coordinator step)."""
    while True:
        print("Choose run mode:")
        print("  1) Script file (steps from scripts/*.txt)")
        print("  2) Runtime command (enter one command per step)")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("Invalid choice. Please try again.")


def show_completion_popup(message: str, title: str = "QualityAssuranceAgent") -> None:
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(0, message, title, 0)
        except Exception:
            print(f"[master] {title}: {message}")
    else:
        print(f"[master] {title}: {message}")


def prepare_run_session(
    *,
    runs_root: Path,
    task: str,
    runtime_mode: bool,
    selected_script_path: Path | None,
    script_steps: list[str] | None,
    eye_monitor_index: int,
    clear_runs_root: bool,
    run_folder_name: str | None = None,
) -> tuple[RunStateManager, RunPaths, str]:
    """Create run directory, set process env for script/runtime mode and monitor index."""
    if clear_runs_root:
        clear_runs_folder(runs_root)
    manager = RunStateManager(runs_root=runs_root)
    paths = manager.init_run(task, run_folder_name)
    run_id = paths.root.name

    set_runtime_env(paths.root, run_id)
    if runtime_mode:
        os.environ[RUNTIME_COMMAND_MODE_ENV] = "1"
    else:
        os.environ.pop(RUNTIME_COMMAND_MODE_ENV, None)
        if selected_script_path is None or script_steps is None:
            raise ValueError("Script mode requires selected_script_path and script_steps")
        os.environ[SCRIPT_PATH_ENV] = str(selected_script_path)
        os.environ[SCRIPT_LINES_ENV] = json.dumps(script_steps, ensure_ascii=False)
    os.environ["EYE_MONITOR_INDEX"] = str(eye_monitor_index)
    manager.log_info(f"Eye capture monitor index set to {eye_monitor_index}")
    return manager, paths, run_id


def run_coordinator_sync() -> None:
    """Run one coordinator lifecycle; caller must set env and ``prepare_run_session`` first."""
    reset_run_state_manager()
    coordinator = RuntimeCoordinator()
    asyncio.run(coordinator.run())


def cli_main() -> None:
    runtime_mode = prompt_run_mode()
    if runtime_mode:
        task = "runtime_command"
        selected_script_path: Path | None = None
        script_steps: list[str] | None = None
    else:
        task, selected_script_path, script_steps = select_script_input(ROOT_DIR)

    settings = load_settings()
    runs_root = Path(settings.runs_dir)
    eye_monitor_index = prompt_eye_monitor_index()
    manager, _, run_id = prepare_run_session(
        runs_root=runs_root,
        task=task,
        runtime_mode=runtime_mode,
        selected_script_path=selected_script_path,
        script_steps=script_steps,
        eye_monitor_index=eye_monitor_index,
        clear_runs_root=True,
        run_folder_name=None,
    )
    print(f"[master] Eye capture monitor index: {eye_monitor_index} (0 = all screens)")
    completion_message = f"Run {run_id} finished."

    try:
        manager.log_info("Master starting coordinator module runtime")
        run_coordinator_sync()
    except KeyboardInterrupt:
        manager.log_info("KeyboardInterrupt received. shutting down coordinator.")
        completion_message = f"Run {run_id} interrupted by user."
    finally:
        manager.log_info("Master stopped.")
        show_completion_popup(completion_message)


def launch_gui() -> None:
    from app_main_hub import run_main_hub

    run_main_hub()


def main() -> None:
    parser = argparse.ArgumentParser(description="QualityAssuranceAgent master")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Interactive stdin mode (legacy); default opens the graphical hub.",
    )
    args = parser.parse_args()
    if args.cli:
        cli_main()
    else:
        launch_gui()


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
