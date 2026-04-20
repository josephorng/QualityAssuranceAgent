from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from fastapi import FastAPI

from cua_mcp.tools import (
    ACTION_TOOL_NAMES,
    execute_tool_call,
)
from src.common.models import BrainTaskState, EyeEvent, HandExecutionResult, ToolCommand
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings

settings = load_settings()
ollama = OllamaClient(settings.ollama_host)
run_root, task_input, _run_id = get_runtime_env()
manager = RunStateManager(run_root.parent, settings.brain_memory_max_chars)
manager.init_run(task_input, run_root.name)

print(
    f"[brain] init run_root={run_root} run_id={_run_id} "
    f"ports brain={settings.brain_port} hand={settings.hand_port} eye={settings.eye_port} "
    f"ollama={settings.ollama_host} lm={settings.brain_lm}"
)
print(f"[brain] task: {task_input[:200]}{'…' if len(task_input) > 200 else ''}")
manager.log_debug(
    f"Brain server initialized run_id={_run_id} "
    f"ports brain={settings.brain_port} hand={settings.hand_port} eye={settings.eye_port}"
)


@dataclass
class BrainRuntime:
    active: BrainTaskState | None = None
    stack: deque[BrainTaskState] = field(default_factory=deque)
    queue: asyncio.Queue[EyeEvent] = field(default_factory=asyncio.Queue)
    previous_action: HandExecutionResult | None = None
    processing: bool = False


runtime = BrainRuntime()


async def _is_interruption(active: BrainTaskState, new_event: EyeEvent) -> bool:
    prompt = get_prompt("classify_interruption")
    full_prompt = (
        f"{prompt}\n\nActive:\n{active.event.description}\n\n"
        f"New:\n{new_event.description}\n\nTask:\n{task_input}"
    )
    out = await ollama.generate_json(
        settings.brain_lm,
        full_prompt,
        fallback={"interruption": True, "replace_state": False, "reason": "fallback"},
    )
    return bool(out.get("interruption", True))


def _best_stack_match(new_event: EyeEvent) -> BrainTaskState | None:
    for state in reversed(runtime.stack):
        if state.event.screenshot_name == new_event.screenshot_name:
            return state
    return None


async def _dispatch_to_hand(command: ToolCommand) -> None:
    manager.log_debug(
        f"Brain dispatching to hand action={command.action} screenshot={command.screenshot_name}"
    )
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(
            f"http://127.0.0.1:{settings.hand_port}/execute",
            json=command.model_dump(mode="json"),
        )
    manager.log_debug(
        f"Brain dispatch complete action={command.action} screenshot={command.screenshot_name}"
    )


async def _decide_action(event: EyeEvent) -> tuple[str, ToolCommand, dict]:
    prompt = get_prompt("brain_decide_action")
    prev_action_text = runtime.previous_action.model_dump_json() if runtime.previous_action else "none"
    memory = manager.require_paths().long_term_memory_txt.read_text(encoding="utf-8")
    full_prompt = (
        f"{prompt}\n\nTask:\n{task_input}\n\nDescription:\n{event.description}\n\n"
        f"PreviousAction:\n{prev_action_text}\n\nMemory:\n{memory}\n\n"
        "Use tool calls only. Keep calling tools as needed. "
        "When ready to execute a UI action, call one of the available action tools."
    )
    max_iterations = 8

    for _ in range(max_iterations):
        assistant_message, tool_calls = await ollama.generate(
            settings.brain_lm,
            prompt=full_prompt,
            use_tools=True,
            store_messages=True,
        )
        
        manager.log_debug(f"Brain generated assistant_message={assistant_message} tool_calls={tool_calls}")

        if not tool_calls:
            raise ValueError("No tool calls returned")

        for call in tool_calls:
            tool_name = call.name
            arguments_raw = call.arguments

            if isinstance(arguments_raw, str):
                try:
                    arguments: dict = json.loads(arguments_raw)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(arguments_raw, dict):
                arguments = arguments_raw
            else:
                arguments = {}

            if tool_name in ACTION_TOOL_NAMES:
                reason = assistant_message
                cmd = ToolCommand(
                    action=tool_name,
                    args=arguments,
                    screenshot_name=event.screenshot_name,
                    reason=reason,
                )
                return reason, cmd, {"message": assistant_message}

            try:
                tool_result = execute_tool_call(tool_name, arguments)
            except Exception as exc:
                tool_result = {"error": str(exc)}

            full_prompt += f"\n\nToolResult from {tool_name}:\n{json.dumps(tool_result, ensure_ascii=True)}"

    ollama.clear_message_history()
    raise ValueError("Tool loop failed")

async def _brain_loop() -> None:
    while True:
        event = await runtime.queue.get()
        manager.log_debug(f"Brain dequeued event screenshot={event.screenshot_name}")
        runtime.processing = True
        active = runtime.active
        if active is not None:
            if await _is_interruption(active, event):
                manager.log_debug(
                    f"Brain interruption accepted active={active.event.screenshot_name} "
                    f"new={event.screenshot_name}"
                )
                runtime.stack.append(active)
                manager.append_brain_memory(
                    f"[{datetime.utcnow().isoformat()}] Interrupted: {active.event.screenshot_name}"
                )
                runtime.active = BrainTaskState(event=event)
            else:
                manager.log_debug(
                    f"Brain ignored event screenshot={event.screenshot_name} while active="
                    f"{active.event.screenshot_name}"
                )
                runtime.processing = False
                continue
        else:
            match = _best_stack_match(event)
            if match is not None:
                runtime.stack.remove(match)
                runtime.active = match
                manager.log_debug(f"Brain resumed stacked state screenshot={match.event.screenshot_name}")
            else:
                runtime.active = BrainTaskState(event=event)
                manager.log_debug(f"Brain created active state screenshot={event.screenshot_name}")

        thought, command, raw_response = await _decide_action(event)
        reason_preview = (thought[:160] + "…") if len(thought) > 160 else thought
        print(
            f"[brain] decide screenshot={event.screenshot_name!r} "
            f"action={command.action!r} args={command.args!r} reason={reason_preview!r}"
        )
        manager.write_thinking_record(event.screenshot_name, thought, raw_response)
        manager.append_brain_memory(f"[{datetime.utcnow().isoformat()}] {thought}")
        await _dispatch_to_hand(command)
        runtime.active = None
        runtime.processing = False
        manager.log_debug(f"Brain finished processing screenshot={event.screenshot_name}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[brain] lifespan: starting _brain_loop")
    manager.log_debug("Brain lifespan startup")
    task = asyncio.create_task(_brain_loop())
    try:
        yield
    finally:
        manager.log_debug("Brain lifespan shutdown")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Brain Server", lifespan=lifespan)


@app.post("/new_event")
async def new_event(event: EyeEvent) -> dict[str, str]:
    preview = (event.description[:120] + "…") if len(event.description) > 120 else event.description
    print(f"[brain] new_event queued screenshot={event.screenshot_name!r} description={preview!r}")
    manager.log_debug(f"Brain queued new_event screenshot={event.screenshot_name}")
    await runtime.queue.put(event)
    return {"status": "queued"}


@app.post("/action_done")
async def action_done(result: HandExecutionResult) -> dict[str, str]:
    print(
        f"[brain] action_done action={result.action!r} ok={result.ok} "
        f"screenshot={result.screenshot_name!r} message={result.message!r}"
    )
    runtime.previous_action = result
    manager.log_debug(
        f"Brain received action_done action={result.action} ok={result.ok} "
        f"screenshot={result.screenshot_name}"
    )
    manager.append_brain_memory(
        f"[{datetime.utcnow().isoformat()}] ActionDone: {result.action} ok={result.ok} message={result.message}"
    )
    return {"status": "ack"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
