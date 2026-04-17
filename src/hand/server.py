from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import pyautogui
from fastapi import FastAPI

from src.common.io_utils import append_csv_row
from src.common.models import HandExecutionResult, ToolCommand
from src.common.run_state import RunStateManager
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(
        f"[hand] startup run_root={run_root} run_id={_run_id} "
        f"port={settings.hand_port} brain={settings.brain_port}"
    )
    print(f"[hand] task: {task_input[:200]}{'…' if len(task_input) > 200 else ''}")
    yield


app = FastAPI(title="Hand Server", lifespan=lifespan)
settings = load_settings()
run_root, task_input, _run_id = get_runtime_env()
manager = RunStateManager(run_root.parent, settings.brain_memory_max_chars)
manager.init_run(task_input, run_root.name)
_busy = False
_lock = asyncio.Lock()


def _exec_action(cmd: ToolCommand) -> HandExecutionResult:
    action = cmd.action
    args = cmd.args
    try:
        if action == "click":
            pyautogui.click(x=args["x"], y=args["y"], button=args.get("button", "left"))
        elif action == "type":
            pyautogui.typewrite(args["text"], interval=args.get("interval", 0.02))
        elif action == "hotkey":
            keys = args.get("keys", [])
            if not keys:
                raise ValueError("hotkey requires keys")
            pyautogui.hotkey(*keys)
        elif action == "move":
            pyautogui.moveTo(x=args["x"], y=args["y"], duration=args.get("duration", 0.2))
        elif action == "wait":
            seconds = float(args.get("seconds", 1.0))
            pyautogui.sleep(seconds)
        else:
            raise ValueError(f"unsupported action: {action}")
        return HandExecutionResult(
            ok=True,
            action=action,
            args=args,
            timestamp=datetime.utcnow(),
            screenshot_name=cmd.screenshot_name,
            message=cmd.reason or "executed",
        )
    except Exception as exc:
        return HandExecutionResult(
            ok=False,
            action=action,
            args=args,
            timestamp=datetime.utcnow(),
            screenshot_name=cmd.screenshot_name,
            message=str(exc),
        )


async def _callback_brain(result: HandExecutionResult) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(
            f"http://127.0.0.1:{settings.brain_port}/action_done",
            json=result.model_dump(mode="json"),
        )


@app.get("/state")
async def state() -> dict[str, bool]:
    return {"busy": _busy}


@app.post("/execute")
async def execute(cmd: ToolCommand) -> dict[str, str]:
    global _busy
    async with _lock:
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
        manager.log_debug(f"Hand action: {result.action}, ok={result.ok}")
        print(
            f"[hand] execute action={result.action} ok={result.ok} "
            f"screenshot={cmd.screenshot_name!r} message={result.message!r}"
        )
        try:
            await _callback_brain(result)
        finally:
            _busy = False
    return {"status": "done"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
