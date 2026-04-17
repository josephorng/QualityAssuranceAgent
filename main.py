from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.common.run_state import RunStateManager
from src.common.runtime_context import set_runtime_env
from src.common.settings import ROOT_DIR, load_settings


def with_suppressed_debugpy_warning(env: dict[str, str]) -> dict[str, str]:
    warning_filter = "ignore:incompatible copy of pydevd already imported:UserWarning"
    current = env.get("PYTHONWARNINGS", "").strip()
    if current:
        env["PYTHONWARNINGS"] = f"{current},{warning_filter}"
    else:
        env["PYTHONWARNINGS"] = warning_filter
    return env


def build_server_command(module: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        f"{module}:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]


def launch_service(name: str, module: str, port: int, env: dict[str, str]) -> subprocess.Popen:
    cmd = build_server_command(module, port)
    print(f"[master] launching {name} on {port}")
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env)


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def resolve_task(cli_task: str | None) -> str:
    if cli_task and cli_task.strip():
        return cli_task.strip()
    while True:
        task = input("Enter task: ").strip()
        if task:
            return task
        print("Task cannot be empty.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Computer Use Agent master process")
    parser.add_argument("--task", help="Initial user task text")
    args = parser.parse_args()
    task = resolve_task(args.task)

    settings = load_settings()
    runs_root = Path(settings.runs_dir)
    manager = RunStateManager(runs_root=runs_root, memory_max_chars=settings.brain_memory_max_chars)
    paths = manager.init_run(task)
    run_id = paths.root.name

    set_runtime_env(paths.root, task, run_id)
    env = os.environ.copy()
    env = with_suppressed_debugpy_warning(env)

    services = {
        "eye": ("src.eye.server", settings.eye_port),
        "brain": ("src.brain.server", settings.brain_port),
        "hand": ("src.hand.server", settings.hand_port),
    }
    procs: dict[str, subprocess.Popen] = {
        name: launch_service(name, module, port, env) for name, (module, port) in services.items()
    }

    manager.log_debug("Master started all services")

    try:
        while True:
            time.sleep(2)
            for name, proc in list(procs.items()):
                code = proc.poll()
                if code is not None:
                    manager.log_debug(f"{name} exited with code {code}. restarting.")
                    module, port = services[name]
                    procs[name] = launch_service(name, module, port, env)
    except KeyboardInterrupt:
        manager.log_debug("KeyboardInterrupt received. shutting down services.")
    finally:
        for proc in procs.values():
            terminate_process(proc)
        manager.log_debug("Master stopped.")


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
