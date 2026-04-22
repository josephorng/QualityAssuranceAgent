from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ServiceConfig:
    name: str
    launch_mode: str  # uvicorn | module
    module: str
    port: int | None = None


def build_module_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def launch_service(service: ServiceConfig, env: dict[str, str]) -> subprocess.Popen:
    if service.launch_mode == "uvicorn":
        if service.port is None:
            raise ValueError(f"port is required for uvicorn service: {service.name}")
        cmd = build_server_command(service.module, service.port)
        display = f"{service.name} on {service.port}"
    elif service.launch_mode == "module":
        cmd = build_module_command(service.module)
        display = service.name
    else:
        raise ValueError(f"unknown launch mode {service.launch_mode!r}")
    print(f"[master] launching {display}")
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
        "eye": ServiceConfig(
            name="eye",
            launch_mode="uvicorn",
            module="src.eye.server",
            port=settings.eye_port,
        ),
        "brain": ServiceConfig(
            name="brain",
            launch_mode="uvicorn",
            module="src.brain.server",
            port=settings.brain_port,
        ),
        "hand": ServiceConfig(
            name="hand",
            launch_mode="uvicorn",
            module="src.hand.server",
            port=settings.hand_port,
        ),
        "mcp": ServiceConfig(
            name="mcp",
            launch_mode="module",
            module="cua_mcp.tools",
        ),
    }
    procs: dict[str, subprocess.Popen] = {
        name: launch_service(service, env) for name, service in services.items()
    }

    manager.log_info("Master started all services")

    try:
        while True:
            time.sleep(2)
            for name, proc in list(procs.items()):
                code = proc.poll()
                if code is not None:
                    manager.log_info(f"{name} exited with code {code}. restarting.")
                    procs[name] = launch_service(services[name], env)
    except KeyboardInterrupt:
        manager.log_info("KeyboardInterrupt received. shutting down services.")
    finally:
        for proc in procs.values():
            terminate_process(proc)
        manager.log_info("Master stopped.")


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
