from __future__ import annotations

import asyncio
import ctypes
import faulthandler
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
FAULT_LOG_FILENAME = "ComputerAgent_fault.log"


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


def _fault_log_path() -> Path:
    return _application_dir_for_crash_log() / FAULT_LOG_FILENAME


_FAULT_HEARTBEAT_INTERVAL_S = 60.0
_fault_log_path_active: Path | None = None
_fault_heartbeat_stop = threading.Event()

# faulthandler dumps every Windows SEH with the error bit set, including
# transient COM/RPC failures raised as first-chance exceptions during UIA.
# HRESULT forms (0x8001xxxx) and NTSTATUS RPC_NT_* forms (0xC002xxxx) both
# appear; continuing execution lets the COM call return failure instead of
# killing the process.
#
# EXCEPTION_BREAKPOINT (0x80000003) also shows up when UIA talks to an app
# that is still launching (DebugBreak / provider assert). Without a debugger
# attached that becomes a fatal dump; skip the int3 and keep recording.
_RPC_E_CALL_REJECTED = 0x80010001
_RPC_E_SERVERCALL_RETRYLATER = 0x8001010A
_RPC_E_CANTCALLOUT_ININPUTSYNCCALL = 0x8001010D
_RPC_NT_SERVER_UNAVAILABLE = 0xC0020015
_RPC_NT_CALL_FAILED = 0xC0020017
_RPC_NT_CALL_FAILED_DNE = 0xC0020018
_RPC_NT_INTERNAL_ERROR = 0xC0020043
_EXCEPTION_BREAKPOINT = 0x80000003
_EXCEPTION_NONCONTINUABLE = 0x1
# AMD64 CONTEXT.Rip / x86 CONTEXT.Eip offsets (winnt.h).
_CONTEXT_RIP_OFFSET = 0xF8
_CONTEXT_EIP_OFFSET = 0xB8
_NONFATAL_COM_EXCEPTION_CODES = frozenset(
    {
        _RPC_E_CALL_REJECTED,
        _RPC_E_SERVERCALL_RETRYLATER,
        _RPC_E_CANTCALLOUT_ININPUTSYNCCALL,
        _RPC_NT_SERVER_UNAVAILABLE,
        _RPC_NT_CALL_FAILED,
        _RPC_NT_CALL_FAILED_DNE,
        _RPC_NT_INTERNAL_ERROR,
        _EXCEPTION_BREAKPOINT,
    }
)
_EXCEPTION_CONTINUE_EXECUTION = -1
_EXCEPTION_CONTINUE_SEARCH = 0
# Keep the WINFUNCTYPE callback alive for the process lifetime.
_nonfatal_com_veh_handler: Any = None
_nonfatal_com_veh_handle: Any = None


def _advance_past_breakpoint(context_addr: int) -> None:
    """Skip the 1-byte int3 (0xCC) so CONTINUE_EXECUTION does not re-hit it."""
    if not context_addr:
        return
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        rip = ctypes.c_uint64.from_address(context_addr + _CONTEXT_RIP_OFFSET)
        rip.value += 1
    else:
        eip = ctypes.c_uint32.from_address(context_addr + _CONTEXT_EIP_OFFSET)
        eip.value += 1


def _install_nonfatal_com_veh() -> None:
    """Swallow known non-fatal COM SEH so faulthandler does not dump them.

    Registered after ``faulthandler.enable`` so this handler runs first
    (VEH is LIFO). Allowlisted codes return ``EXCEPTION_CONTINUE_EXECUTION``
    and never reach faulthandler; real fatals fall through.
    """
    global _nonfatal_com_veh_handler, _nonfatal_com_veh_handle
    if os.name != "nt":
        return
    if _nonfatal_com_veh_handle is not None:
        return

    class _EXCEPTION_RECORD(ctypes.Structure):
        pass

    _EXCEPTION_RECORD._fields_ = [
        ("ExceptionCode", ctypes.c_ulong),
        ("ExceptionFlags", ctypes.c_ulong),
        ("ExceptionRecord", ctypes.POINTER(_EXCEPTION_RECORD)),
        ("ExceptionAddress", ctypes.c_void_p),
        ("NumberParameters", ctypes.c_ulong),
        ("ExceptionInformation", ctypes.c_ulonglong * 15),
    ]

    class _EXCEPTION_POINTERS(ctypes.Structure):
        _fields_ = [
            ("ExceptionRecord", ctypes.POINTER(_EXCEPTION_RECORD)),
            ("ContextRecord", ctypes.c_void_p),
        ]

    handler_type = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.POINTER(_EXCEPTION_POINTERS)
    )

    @handler_type
    def _handler(exc_info: Any) -> int:
        try:
            record = exc_info.contents.ExceptionRecord
            if not record:
                return _EXCEPTION_CONTINUE_SEARCH
            flags = int(record.contents.ExceptionFlags) & 0xFFFFFFFF
            if flags & _EXCEPTION_NONCONTINUABLE:
                return _EXCEPTION_CONTINUE_SEARCH
            code = int(record.contents.ExceptionCode) & 0xFFFFFFFF
            if code not in _NONFATAL_COM_EXCEPTION_CODES:
                return _EXCEPTION_CONTINUE_SEARCH
            if code == _EXCEPTION_BREAKPOINT:
                ctx = exc_info.contents.ContextRecord
                _advance_past_breakpoint(int(ctx) if ctx else 0)
            return _EXCEPTION_CONTINUE_EXECUTION
        except Exception:
            pass
        return _EXCEPTION_CONTINUE_SEARCH

    add_veh = ctypes.windll.kernel32.AddVectoredExceptionHandler
    add_veh.argtypes = [ctypes.c_ulong, handler_type]
    add_veh.restype = ctypes.c_void_p
    # FirstHandler=1: run before later-registered handlers when possible;
    # still register after faulthandler.enable so we outrank its VEH.
    handle = add_veh(1, _handler)
    if handle:
        _nonfatal_com_veh_handler = _handler
        _nonfatal_com_veh_handle = handle


def _write_fault_marker(path: Path, kind: str, **fields: Any) -> None:
    """Append a human-readable marker. Safe under concurrent faulthandler dumps."""
    parts = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"kind={kind}",
        f"pid={os.getpid()}",
    ]
    for key, value in fields.items():
        parts.append(f"{key}={value!r}")
    line = "--- " + " ".join(parts) + " ---\n"
    try:
        with open(path, "a", encoding="utf-8", errors="replace") as handle:
            handle.write(line)
            handle.flush()
    except OSError:
        pass


def _fault_heartbeat_loop(path: Path) -> None:
    """Periodic timestamps so SEH dumps (no built-in time) can be bracketed."""
    while not _fault_heartbeat_stop.wait(_FAULT_HEARTBEAT_INTERVAL_S):
        _write_fault_marker(path, "heartbeat")


def _install_faulthandler() -> None:
    """Dump native / fatal-error traces (segfaults bypass sys.excepthook).

    Windows SEH dumps from faulthandler have no timestamps of their own, so we
    write a session banner at enable time and a heartbeat every
    ``_FAULT_HEARTBEAT_INTERVAL_S`` seconds. Correlate a dump with the nearest
    preceding/following ``kind=heartbeat`` / ``kind=session_start`` marker.
    """
    global _fault_log_path_active
    candidates = [_fault_log_path(), Path(tempfile.gettempdir()) / FAULT_LOG_FILENAME]
    for path in candidates:
        try:
            # Keep the handle open for the process lifetime; faulthandler needs it.
            fault_file = open(path, "a", encoding="utf-8", errors="replace")
            faulthandler.enable(file=fault_file, all_threads=True)
            # After faulthandler's VEH so ours runs first (LIFO) and can
            # suppress known non-fatal COM codes before they are dumped.
            _install_nonfatal_com_veh()
            _fault_log_path_active = path
            _write_fault_marker(
                path,
                "session_start",
                executable=sys.executable,
                cwd=os.getcwd(),
                frozen=getattr(sys, "frozen", False),
            )
            _fault_heartbeat_stop.clear()
            threading.Thread(
                target=_fault_heartbeat_loop,
                args=(path,),
                name="cua-fault-heartbeat",
                daemon=True,
            ).start()
            return
        except OSError:
            continue


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
        SCRIPT_BASELINE_AFTER_ENV,
        SCRIPT_EVENT_INDICES_ENV,
        SCRIPT_LINES_ENV,
        SCRIPT_OUTCOMES_ENV,
        SCRIPT_PATH_ENV,
        SCRIPT_WINDOW_VERIFY_ENV,
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
    os.environ.pop(SCRIPT_BASELINE_AFTER_ENV, None)
    os.environ.pop(SCRIPT_WINDOW_VERIFY_ENV, None)
    os.environ.pop(SCRIPT_EVENT_INDICES_ENV, None)

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
            collect_recording_baseline_after_paths,
            collect_recording_instruction_event_indices,
            collect_recording_script_text,
            collect_recording_window_verifies,
            parse_script_steps_with_outcomes,
            recording_run_dir,
        )

        rec = recording_run_dir(selected_script_path)
        if rec is not None:
            script_raw = collect_recording_script_text(rec)
            try:
                from src.recorder.compile_tool_calls import ensure_recording_tool_cache

                ensure_recording_tool_cache(rec)
            except Exception as exc:
                manager.log_info(f"ensure_recording_tool_cache failed: {exc}")
        else:
            try:
                script_raw = Path(selected_script_path).read_text(encoding="utf-8")
            except OSError:
                script_raw = "\n".join(script_steps)
        _steps, outcomes = parse_script_steps_with_outcomes(script_raw)
        if len(outcomes) != len(script_steps):
            outcomes = [None] * len(script_steps)
        os.environ[SCRIPT_OUTCOMES_ENV] = json.dumps(outcomes, ensure_ascii=False)
        if rec is not None:
            baselines = collect_recording_baseline_after_paths(rec)
            if len(baselines) != len(script_steps):
                baselines = [None] * len(script_steps)
            os.environ[SCRIPT_BASELINE_AFTER_ENV] = json.dumps(baselines, ensure_ascii=False)
            verifies = collect_recording_window_verifies(rec)
            if len(verifies) != len(script_steps):
                verifies = [{} for _ in script_steps]
            os.environ[SCRIPT_WINDOW_VERIFY_ENV] = json.dumps(verifies, ensure_ascii=False)
            event_indices = collect_recording_instruction_event_indices(rec)
            if len(event_indices) != len(script_steps):
                event_indices = [None] * len(script_steps)
            os.environ[SCRIPT_EVENT_INDICES_ENV] = json.dumps(
                event_indices, ensure_ascii=False
            )
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
    from src.runtime.coordinator import RuntimeCoordinator

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

    from cua_mcp import hand_tools

    hand_tools.backup_clipboard()
    try:
        asyncio.run(_main())
    finally:
        hand_tools.restore_clipboard()


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
    on_progress: Callable[[int, int, str], None] | None = None,
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
    _install_faulthandler()
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
