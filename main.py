from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.common.llm_factory import reset_llm_client
from src.common.run_state import RunStateManager, RunPaths, reset_run_state_manager
from src.runtime.coordinator import RuntimeCoordinator
from src.common.runtime_context import (
    RUNTIME_COMMAND_MODE_ENV,
    SCRIPT_LINES_ENV,
    SCRIPT_PATH_ENV,
    SMART_GOAL_ENV,
    SMART_MODE_ENV,
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
    smart_mode: bool = False,
    smart_goal: str | None = None,
) -> tuple[RunStateManager, RunPaths, str]:
    """Create run directory, set process env for script/runtime/smart mode and monitor selection."""
    if clear_runs_root:
        clear_runs_folder(runs_root)
    if not eye_monitor_indices:
        raise ValueError("eye_monitor_indices must be non-empty")
    if smart_mode and runtime_mode:
        raise ValueError("smart_mode and runtime_mode are mutually exclusive")
    manager = RunStateManager(runs_root=runs_root)
    paths = manager.init_run(task, run_folder_name)
    run_id = paths.root.name

    set_runtime_env(paths.root, run_id)
    os.environ.pop(RUNTIME_COMMAND_MODE_ENV, None)
    os.environ.pop(SMART_MODE_ENV, None)
    os.environ.pop(SMART_GOAL_ENV, None)
    os.environ.pop(SCRIPT_PATH_ENV, None)
    os.environ.pop(SCRIPT_LINES_ENV, None)

    if smart_mode:
        goal = (smart_goal or task or "").strip()
        if not goal:
            raise ValueError("Smart mode requires a non-empty goal")
        os.environ[SMART_MODE_ENV] = "1"
        os.environ[SMART_GOAL_ENV] = goal
    elif runtime_mode:
        os.environ[RUNTIME_COMMAND_MODE_ENV] = "1"
    else:
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


def run_coordinator_sync(*, smart_mode: bool = False) -> None:
    """Run one coordinator lifecycle; caller must set env and ``prepare_run_session`` first."""
    reset_run_state_manager()
    # ``asyncio.run`` closes its loop when the run ends; drop LLM clients so the next
    # run builds fresh async transports instead of reusing ones bound to a closed loop.
    reset_llm_client()

    from src.common.runtime_context import get_runtime_env

    run_root, _ = get_runtime_env()

    async def _main() -> None:
        global _coordinator_loop, _coordinator_main_task
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        with _coordinator_loop_lock:
            _coordinator_loop = loop
            _coordinator_main_task = task
        try:
            if smart_mode:
                from src.runtime.smart_coordinator import SmartCoordinator

                coordinator: Any = SmartCoordinator()
            else:
                coordinator = RuntimeCoordinator()
            await coordinator.run()
        finally:
            with _coordinator_loop_lock:
                _coordinator_loop = None
                _coordinator_main_task = None

    try:
        asyncio.run(_main())
    finally:
        if os.environ.get("CUA_WRITE_SESSION_REPORT") == "1":
            from src.common.run_state import get_run_state_manager
            from src.common.session_report import write_session_report

            manager = get_run_state_manager()
            reason = manager.session_end_reason or "completed"
            write_session_report(run_root, session_end_reason=reason)


def dismiss_nuitka_onefile_splash() -> None:
    """Close Nuitka's onefile splash once the hub window is ready (Windows onefile builds only)."""
    parent = os.environ.get("NUITKA_ONEFILE_PARENT")
    if not parent:
        return
    try:
        pid = int(parent)
    except ValueError:
        return
    splash_filename = os.path.join(
        tempfile.gettempdir(),
        f"onefile_{pid}_splash_feedback.tmp",
    )
    if os.path.exists(splash_filename):
        os.unlink(splash_filename)



def analyze_screen_recording(
    run_dir: Path,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Analyze a finished screen recording session and write hub-script instructions."""
    from src.recorder.orchestrator import analyze_recording_session

    # Mirror run_coordinator_sync: rebuild the LLM client from agent settings and
    # avoid reusing async transports bound to a prior asyncio.run() loop.
    reset_llm_client()

    return asyncio.run(
        analyze_recording_session(
            run_dir,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
    )


def launch_gui() -> None:
    from app_main_hub import run_main_hub

    run_main_hub()


def main() -> None:
    from src.common.settings import apply_vision_env_from_settings

    apply_vision_env_from_settings()
    launch_gui()


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
