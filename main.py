from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CRASH_LOG_FILENAME = "ComputerAgent_crash.log"


def _application_dir_for_crash_log() -> Path:
    """Directory containing the distributed exe (onefile-safe), or project root in dev."""
    # Nuitka onefile runs from a temp unpack dir; argv[0] is still the launched .exe.
    if getattr(sys, "frozen", False) or globals().get("__compiled__") is not None:
        if sys.argv:
            launched = Path(sys.argv[0]).expanduser()
            try:
                launched = launched.resolve()
            except OSError:
                pass
            if launched.suffix.lower() == ".exe" and launched.parent.is_dir():
                return launched.parent
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _crash_log_path() -> Path:
    return _application_dir_for_crash_log() / CRASH_LOG_FILENAME


def write_crash_log(
    exc: BaseException | None = None,
    *,
    exc_info: tuple[type[BaseException], BaseException, Any] | None = None,
) -> Path | None:
    """Write a crash report to ComputerAgent_crash.log (exe dir, else %TEMP%)."""
    if exc_info is not None:
        tb_text = "".join(traceback.format_exception(*exc_info))
    elif exc is not None:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        tb_text = "".join(traceback.format_exception(*sys.exc_info()))

    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"executable={sys.executable!r}",
        f"cwd={os.getcwd()!r}",
        f"frozen={getattr(sys, 'frozen', False)!r}",
        "",
        tb_text.rstrip(),
        "",
    ]
    text = "\n".join(lines)
    candidates = [_crash_log_path(), Path(tempfile.gettempdir()) / CRASH_LOG_FILENAME]
    for path in candidates:
        try:
            path.write_text(text, encoding="utf-8")
            return path
        except OSError:
            continue
    return None


def _ensure_stdio_streams() -> None:
    """Windows GUI builds (--windows-console-mode=disable) leave stdout/stderr as None.

    CustomTkinter writes font warnings to sys.stderr; without a stream that raises
    AttributeError: 'NoneType' object has no attribute 'write'.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")


def _install_crash_logging() -> None:
    """Log uncaught exceptions (main thread and other threads) to ComputerAgent_crash.log."""

    def _excepthook(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: Any,
    ) -> None:
        write_crash_log(exc_info=(exc_type, exc, tb))
        try:
            dismiss_nuitka_onefile_splash()
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not None and args.exc_value is not None:
            write_crash_log(exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        previous(args)

    previous = threading.excepthook
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


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
) -> tuple[Any, Any, str]:
    """Create run directory, set process env for script/runtime/smart mode and monitor selection."""
    from src.common.run_state import RunStateManager
    from src.common.runtime_context import (
        RUNTIME_COMMAND_MODE_ENV,
        SCRIPT_LINES_ENV,
        SCRIPT_OUTCOMES_ENV,
        SCRIPT_PATH_ENV,
        SMART_GOAL_ENV,
        SMART_MODE_ENV,
        set_runtime_env,
    )

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
    os.environ.pop(SCRIPT_OUTCOMES_ENV, None)

    if smart_mode:
        goal = (smart_goal or task or "").strip()
        if not goal:
            raise ValueError("Smart mode requires a non-empty goal")
        os.environ[SMART_MODE_ENV] = "1"
        os.environ[SMART_GOAL_ENV] = goal
        if selected_script_path is not None:
            os.environ[SCRIPT_PATH_ENV] = str(selected_script_path)
    elif runtime_mode:
        os.environ[RUNTIME_COMMAND_MODE_ENV] = "1"
    else:
        if selected_script_path is None or script_steps is None:
            raise ValueError("Script mode requires selected_script_path and script_steps")
        os.environ[SCRIPT_PATH_ENV] = str(selected_script_path)
        os.environ[SCRIPT_LINES_ENV] = json.dumps(script_steps, ensure_ascii=False)
        from src.common.script_helper import (
            collect_recording_script_text,
            parse_script_steps_with_outcomes,
            recording_run_dir,
        )

        rec = recording_run_dir(selected_script_path)
        if rec is not None:
            script_raw = collect_recording_script_text(rec)
        else:
            try:
                script_raw = Path(selected_script_path).read_text(encoding="utf-8")
            except OSError:
                script_raw = "\n".join(script_steps)
        _steps, outcomes = parse_script_steps_with_outcomes(script_raw)
        if len(outcomes) != len(script_steps):
            outcomes = [None] * len(script_steps)
        os.environ[SCRIPT_OUTCOMES_ENV] = json.dumps(outcomes, ensure_ascii=False)
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
    from src.common.llm_factory import reset_llm_client
    from src.common.run_state import reset_run_state_manager
    from src.common.runtime_context import get_runtime_env
    from src.runtime.coordinator import RuntimeCoordinator

    reset_run_state_manager()
    # ``asyncio.run`` closes its loop when the run ends; drop LLM clients so the next
    # run builds fresh async transports instead of reusing ones bound to a closed loop.
    reset_llm_client()

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
    from src.common.llm_factory import reset_llm_client
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
    _ensure_stdio_streams()
    _install_crash_logging()
    try:
        if os.name == "nt":
            signal.signal(signal.SIGINT, signal.default_int_handler)
        main()
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except BaseException as exc:
        log_path = write_crash_log(exc)
        try:
            dismiss_nuitka_onefile_splash()
        except Exception:
            pass
        if log_path is not None:
            print(f"Crash logged to: {log_path}", file=sys.stderr)
        raise SystemExit(1) from exc
