from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from fastapi import FastAPI

from cua_mcp.tools import HAND_TOOL_NAMES
from src.common.models import BrainTaskState, EyeEvent, HandExecutionResult, ToolCommand
from src.common.ollama_client import OllamaClient
from src.common.prompting import render_prompt_with_skills
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings

settings = load_settings()
ollama = OllamaClient(settings.ollama_host)
run_root, task_input, _run_id = get_runtime_env()
manager = get_run_state_manager()
manager.init_run(task_input, run_root.name)

print(
    f"[brain] init run_root={run_root} run_id={_run_id} "
    f"ports brain={settings.brain_port} hand={settings.hand_port} eye={settings.eye_port} "
    f"ollama={settings.ollama_host} lm={settings.brain_lm}"
)
print(f"[brain] task: {task_input[:200]}{'…' if len(task_input) > 200 else ''}")
manager.log_info(
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


def _with_default_image_path(tool_name: str, arguments: dict, screenshot_path: str) -> dict:
    """
    Ensure vision tools always receive the current screenshot path.

    Models sometimes emit placeholders like `screen.png`. When that happens,
    fall back to the real screenshot path from the current event.
    """
    if tool_name not in {"get_coordinates", "detect_objects"}:
        return arguments

    patched = dict(arguments)
    patched["image_path"] = screenshot_path
    return patched


async def _is_interruption(active: BrainTaskState, new_event: EyeEvent) -> bool:
    manager.log_info(f"Brain classifying interruption active={active.event.screenshot_name} new={new_event.screenshot_name}")
    out, _ = await ollama.generate_json(
        settings.brain_lm,
        prompt=render_prompt_with_skills("classify_interruption"),
        fallback={"interruption": True, "replace_state": False, "reason": "fallback"},
        image_paths=[active.event.screenshot_path, new_event.screenshot_path],
    )
    decision = bool(out.get("interruption", True))
    manager.log_info(
        f"Brain interruption classified decision={decision} "
        f"active={active.event.screenshot_name} new={new_event.screenshot_name} "
        f"reason={out.get('reason', 'n/a')}"
    )
    return decision


def _best_stack_match(new_event: EyeEvent) -> BrainTaskState | None:
    for state in reversed(runtime.stack):
        if state.event.screenshot_name == new_event.screenshot_name:
            return state
    return None


async def _dispatch_to_hand(command: ToolCommand) -> None:
    manager.log_info(
        f"Brain dispatching to hand action={command.action} screenshot={command.screenshot_name}"
    )
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(
            f"http://127.0.0.1:{settings.hand_port}/execute",
            json=command.model_dump(mode="json"),
        )
    manager.log_info(
        f"Brain dispatch complete action={command.action} screenshot={command.screenshot_name}"
    )


async def _decide_action(event: EyeEvent) -> tuple[str, ToolCommand, dict]:
    prompt = render_prompt_with_skills("brain_decide_action")
    prev_action_text = runtime.previous_action.model_dump_json() if runtime.previous_action else "none"
    memory = manager.require_paths().long_term_memory_txt.read_text(encoding="utf-8")
    full_prompt = f"{prompt}\n\nTask:\n{task_input}\n\n"
    if prev_action_text != "none":
        full_prompt += f"PreviousAction:\n{prev_action_text}\n\n"
    if memory != "":
        full_prompt += f"Memory:\n{memory}"
    max_iterations = 8
    manager.log_info(
        f"Brain deciding action screenshot={event.screenshot_name} "
        f"has_previous_action={prev_action_text != 'none'} memory_chars={len(memory)}"
    )

    for idx in range(max_iterations):
        manager.log_info(
            f"Brain tool-loop iteration={idx + 1}/{max_iterations} "
            f"screenshot={event.screenshot_name}"
        )
        assistant_message, tool_calls = await ollama.generate(
            settings.brain_lm,
            prompt=full_prompt,
            image_paths=[event.screenshot_path],
            use_tools=True,
            store_messages=True,
        )
        
        manager.log_info(f"Brain generated assistant_message={assistant_message} tool_calls={tool_calls}")

        if len(tool_calls) == 0:
            manager.log_info("Brain no tool calls returned")
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

            arguments = _with_default_image_path(tool_name, arguments, event.screenshot_path)

            if tool_name not in HAND_TOOL_NAMES:
                manager.log_info(f"Brain received unknown tool {tool_name}")
                full_prompt += (
                    f"\n\nToolResult from {tool_name}:\n"
                    f"{json.dumps({'error': f'unknown tool: {tool_name}'}, ensure_ascii=False)}"
                )
                continue

            manager.log_info(
                f"Brain selected hand action={tool_name} screenshot={event.screenshot_name}"
            )
            reason = assistant_message
            cmd = ToolCommand(
                action=tool_name,
                args=arguments,
                screenshot_name=event.screenshot_name,
                reason=reason,
            )
            return reason, cmd, {"message": assistant_message}

    ollama.clear_message_history()
    manager.log_info(f"Brain tool-loop exhausted screenshot={event.screenshot_name}")
    raise ValueError("Tool loop failed")

async def _brain_loop() -> None:
    while True:
        event = await runtime.queue.get()
        manager.log_info(
            f"Brain dequeued event screenshot={event.screenshot_name} "
            f"queue_size={runtime.queue.qsize()} stack_size={len(runtime.stack)}"
        )
        runtime.processing = True
        active = runtime.active
        if active is not None:
            if await _is_interruption(active, event):
                manager.log_info(
                    f"Brain interruption accepted active={active.event.screenshot_name} "
                    f"new={event.screenshot_name}"
                )
                runtime.stack.append(active)
                manager.append_brain_memory(
                    f"[{datetime.utcnow().isoformat()}] Interrupted: {active.event.screenshot_name}"
                )
                runtime.active = BrainTaskState(event=event)
            else:
                manager.log_info(
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
                manager.log_info(f"Brain resumed stacked state screenshot={match.event.screenshot_name}")
            else:
                runtime.active = BrainTaskState(event=event)
                manager.log_info(f"Brain created active state screenshot={event.screenshot_name}")

        manager.log_info(f"Brain starting decision for screenshot={event.screenshot_name}")
        thought, command, raw_response = await _decide_action(event)
        reason_preview = (thought[:160] + "…") if len(thought) > 160 else thought
        manager.log_info(
            f"Brain decided action screenshot={event.screenshot_name!r} "
            f"action={command.action!r} args={command.args!r} reason={reason_preview!r}"
        )
        manager.write_thinking_record(event.screenshot_name, thought, raw_response)
        manager.append_brain_memory(f"[{datetime.utcnow().isoformat()}] {thought}")
        await _dispatch_to_hand(command)
        runtime.active = None
        runtime.processing = False
        manager.log_info(f"Brain finished processing screenshot={event.screenshot_name}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[brain] lifespan: starting _brain_loop")
    manager.log_info("Brain lifespan startup")
    task = asyncio.create_task(_brain_loop())
    try:
        yield
    finally:
        manager.log_info("Brain lifespan shutdown")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Brain Server", lifespan=lifespan)


@app.post("/new_event")
async def new_event(event: EyeEvent) -> dict[str, str]:
    print(
        f"[brain] new_event queued screenshot={event.screenshot_name!r} "
        f"path={event.screenshot_path!r}"
    )
    manager.log_info(
        f"Brain queued new_event screenshot={event.screenshot_name} "
        f"processing={runtime.processing} queue_size_before_put={runtime.queue.qsize()}"
    )
    await runtime.queue.put(event)
    manager.log_info(
        f"Brain queued new_event screenshot={event.screenshot_name} "
        f"queue_size_after_put={runtime.queue.qsize()}"
    )
    return {"status": "queued"}


@app.post("/action_done")
async def action_done(result: HandExecutionResult) -> dict[str, str]:
    print(
        f"[brain] action_done action={result.action!r} ok={result.ok} "
        f"screenshot={result.screenshot_name!r} message={result.message!r}"
    )
    runtime.previous_action = result
    manager.log_info(
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
