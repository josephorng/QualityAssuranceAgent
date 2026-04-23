from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI

from cua_mcp.tools import execute_tool_call
from src.common.io_utils import append_csv_row
from src.common.models import HandExecutionResult, ToolCommand
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings
from datetime import timezone


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(
        f"[hand] startup run_root={run_root} run_id={_run_id} "
        f"port={settings.hand_port} brain={settings.brain_port}"
    )
    print(f"[hand] task: {task_input[:200]}{'…' if len(task_input) > 200 else ''}")
    manager.log_info(
        f"Hand server initialized run_id={_run_id} "
        f"port={settings.hand_port} brain={settings.brain_port}"
    )
    try:
        yield
    finally:
        manager.log_info("Hand lifespan shutdown")


app = FastAPI(title="Hand Server", lifespan=lifespan)
settings = load_settings()
run_root, task_input, _run_id = get_runtime_env()
manager = get_run_state_manager()
manager.init_run(task_input, run_root.name)
_busy = False
_lock = asyncio.Lock()


def _resolve_screenshot_path(screenshot_name: str | None) -> str | None:
    if not screenshot_name:
        return None
    candidate = Path(screenshot_name)
    return str(manager.require_paths().eye_dir / candidate.name)


def _exec_action(cmd: ToolCommand) -> HandExecutionResult:
    action = cmd.action
    args = cmd.args
    screenshot_path = _resolve_screenshot_path(cmd.screenshot_name)
    try:
        executed_args = execute_tool_call(action, args, screenshot_path or "")
        return HandExecutionResult(
            ok=True,
            action=action,
            args=executed_args if isinstance(executed_args, dict) else args,
            timestamp=datetime.now(timezone.utc),
            screenshot_name=screenshot_path,
            message=cmd.reason or "executed",
        )
    except Exception as exc:
        return HandExecutionResult(
            ok=False,
            action=action,
            args=args,
            timestamp=datetime.now(timezone.utc),
            screenshot_name=screenshot_path,
            message=str(exc),
        )


async def _callback_brain(result: HandExecutionResult) -> None:
    manager.log_info(
        f"Hand callback -> brain action={result.action} ok={result.ok} "
        f"screenshot={result.screenshot_name}"
    )
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(
            f"http://127.0.0.1:{settings.brain_port}/action_done",
            json=result.model_dump(mode="json"),
        )
    manager.log_info(
        f"Hand callback sent action={result.action} ok={result.ok} "
        f"screenshot={result.screenshot_name}"
    )


@app.get("/state")
async def state() -> dict[str, bool]:
    return {"busy": _busy}


@app.post("/execute")
async def execute(cmd: ToolCommand) -> dict[str, str]:
    global _busy
    async with _lock:
        manager.log_info(
            f"Hand received execute action={cmd.action} screenshot={cmd.screenshot_name}"
        )
        _busy = True
        result = _exec_action(cmd)
        append_csv_row(
            manager.require_paths().hand_csv,
            fieldnames=["timestamp", "action", "args", "ok", "screenshot_name", "message"],
            row={
                "timestamp": result.timestamp.isoformat(),
                "action": result.action,
                "args": result.args,
                "ok": result.ok,
                "screenshot_name": result.screenshot_name or "",
                "message": result.message,
            },
        )
        if result.ok:
            manager.log_info(f"Hand action: {result.action}, ok={result.ok}")
        else:
            manager.log_info(
                f"Hand action failed: {result.action}, ok={result.ok}, error={result.message}"
            )
        print(
            f"[hand] execute action={result.action} ok={result.ok} "
            f"screenshot={cmd.screenshot_name!r} message={result.message!r}"
        )
        try:
            await _callback_brain(result)
        finally:
            _busy = False
            manager.log_info(
                f"Hand finished execute action={result.action} ok={result.ok} "
                f"screenshot={cmd.screenshot_name}"
            )
    return {"status": "done"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
