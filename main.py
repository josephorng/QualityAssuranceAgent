from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import threading
from pathlib import Path

from src.common.llm_factory import reset_llm_client
from src.common.run_state import RunStateManager, RunPaths, reset_run_state_manager
# from src.common.settings import load_settings
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


def prepare_run_session(
    *,
    runs_root: Path,
    task: str,
    runtime_mode: bool,
    selected_script_path: Path | None,
    script_steps: list[str] | None,
    eye_monitor_indices: list[int],
    clear_runs_root: bool,
    run_folder_name: str | None = None,
) -> tuple[RunStateManager, RunPaths, str]:
    """Create run directory, set process env for script/runtime mode and monitor selection."""
    if clear_runs_root:
        clear_runs_folder(runs_root)
    if not eye_monitor_indices:
        raise ValueError("eye_monitor_indices must be non-empty")
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
    primary = eye_monitor_indices[0]
    os.environ["EYE_MONITOR_INDEX"] = str(primary)
    if len(eye_monitor_indices) > 1:
        os.environ["EYE_MONITOR_INDICES"] = ",".join(str(i) for i in eye_monitor_indices)
    else:
        os.environ.pop("EYE_MONITOR_INDICES", None)
    manager.log_info(f"Eye capture monitors primary={primary} all={eye_monitor_indices}")
    return manager, paths, run_id


_coordinator_loop_lock = threading.Lock()
_coordinator_loop: asyncio.AbstractEventLoop | None = None
_coordinator_main_task: asyncio.Task[None] | None = None


def request_coordinator_cancel() -> bool:
    """Cancel the coordinator task created by ``run_coordinator_sync`` (safe from another thread)."""
    with _coordinator_loop_lock:
        loop = _coordinator_loop
        task = _coordinator_main_task
    if loop is None or task is None:
        return False
    try:
        if loop.is_closed():
            return False
        loop.call_soon_threadsafe(task.cancel)
        return True
    except RuntimeError:
        return False


def run_coordinator_sync() -> None:
    """Run one coordinator lifecycle; caller must set env and ``prepare_run_session`` first."""
    reset_run_state_manager()
    # ``asyncio.run`` closes its loop when the run ends; drop LLM clients so the next
    # run builds fresh async transports instead of reusing ones bound to a closed loop.
    reset_llm_client()

    async def _main() -> None:
        global _coordinator_loop, _coordinator_main_task
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        with _coordinator_loop_lock:
            _coordinator_loop = loop
            _coordinator_main_task = task
        try:
            coordinator = RuntimeCoordinator()
            await coordinator.run()
        finally:
            with _coordinator_loop_lock:
                _coordinator_loop = None
                _coordinator_main_task = None

    asyncio.run(_main())


def launch_gui() -> None:
    from app_main_hub import run_main_hub

    run_main_hub()


def main() -> None:
    launch_gui()


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
