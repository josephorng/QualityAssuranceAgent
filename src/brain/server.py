"""Brain FastAPI service: queues eye screenshots, drives `steps.json` lifecycle, and dispatches tool commands to hand."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI

from cua_mcp import steps as step_tools
from cua_mcp.tools import HAND_TOOL_NAMES
from src.common.models import BrainTaskState, EyeEvent, HandExecutionResult, ToolCommand
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
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
    """In-process state for the brain worker loop and step orchestration."""

    active: BrainTaskState | None = None
    stack: deque[BrainTaskState] = field(default_factory=deque)
    queue: asyncio.Queue[EyeEvent] = field(default_factory=asyncio.Queue)
    previous_action: HandExecutionResult | None = None
    active_step_path: str | None = None
    latest_event: EyeEvent | None = None
    finished: bool = False
    processing: bool = False


runtime = BrainRuntime()


def _ensure_seed_step() -> None:
    """If `steps.json` has no leaf steps yet, append one root-level step from `task_input`."""
    summary = step_tools.check_task()
    if summary.get("leaf_count", 0) > 0:
        return
    step_tools.create_new_steps(
        target_path="",
        new_steps=[
            {
                "image": "",
                "goal": task_input,
                "instruction": task_input,
                "result": "",
            }
        ],
    )


def _step_prompt_block(step: dict) -> str:
    """Format the current step fields for inclusion in LLM prompts."""
    return (
        "CurrentStep:\n"
        f"- goal: {step.get('goal', '')}\n"
        f"- instruction: {step.get('instruction', '')}\n"
        f"- tool: {step.get('tool')}\n" if step.get("tool") else ""
    )


def _previous_action_text(action: HandExecutionResult | None) -> str:
    return action.model_dump_json(exclude={"timestamp", "screenshot_name"}) if action else "none"


async def _verify_pending_step(pending_path: str, event: EyeEvent) -> bool:
    """Ask the LM whether the step at `pending_path` is complete given the latest screenshot and last hand result."""
    pending_step = step_tools.get_step(pending_path)
    previous_action_text = _previous_action_text(runtime.previous_action)
    prompt = (
        "Determine whether the pending task step is now completed based on the latest screenshot.\n"
        'Return strict JSON: {"done": <bool>, "reason": <string>}.\n\n'
        f"Task:\n{task_input}\n\n"
        f"{_step_prompt_block(pending_step)}\n"
        f"PreviousAction:\n{previous_action_text}\n"
    )
    out, _ = await ollama.generate_json(
        settings.brain_lm,
        prompt=prompt,
        fallback={"done": False, "reason": "fallback"},
        image_paths=[event.screenshot_path],
    )
    done = bool(out.get("done", False))
    manager.log_info(
        f"Brain verify pending path={pending_path} done={done} reason={out.get('reason', '')}"
    )
    return done


async def _generate_retry_steps(step: dict, event: EyeEvent) -> list[dict]:
    """Produce up to three alternative sibling step dicts after a failed verification."""
    prompt = (
        "Create alternative next steps to retry the same goal with a different method.\n"
        'Return strict JSON: {"steps": [{"goal": "...", "instruction": "...", "tool": "...", "arguments": {}, "result": ""}]}\n'
        "Each step must be executable and concise.\n\n"
        f"Task:\n{task_input}\n\n"
        f"FailedStep:\n{json.dumps(step, ensure_ascii=False)}\n"
    )
    out, _ = await ollama.generate_json(
        settings.brain_lm,
        prompt=prompt,
        fallback={"steps": [{"goal": step.get("goal", task_input), "instruction": step.get("instruction", task_input), "result": ""}]},
        image_paths=[event.screenshot_path],
    )
    raw_steps = out.get("steps", [])
    if not isinstance(raw_steps, list):
        return []
    cleaned: list[dict] = []
    for candidate in raw_steps:
        if not isinstance(candidate, dict):
            continue
        cleaned.append(
            {
                "image": str(candidate.get("image", "")),
                "goal": str(candidate.get("goal", step.get("goal", task_input))),
                "instruction": str(candidate.get("instruction", step.get("instruction", task_input))),
                "tool": str(candidate.get("tool", "")),
                "arguments": candidate.get("arguments", {}),
                "result": str(candidate.get("result", "")),
            }
        )
    return cleaned[:3] if cleaned else []


async def _check_task_and_expand(event: EyeEvent) -> bool:
    """When no leaves are pending or undone, ask the LM if the whole task is done; finish or append root steps. Returns True if the run should stop."""
    summary = step_tools.check_task()
    if summary.get("pending_count", 0) > 0 or summary.get("undone_count", 0) > 0:
        return False
    prompt = (
        "Decide whether the full user task is truly complete.\n"
        'Return strict JSON: {"complete": <bool>, "reason": <string>, "steps": [{"goal":"...","instruction":"...","result":""}]}\n'
        "If not complete, provide one or more additional executable steps in `steps`.\n\n"
        f"Task:\n{task_input}\n"
    )
    out, _ = await ollama.generate_json(
        settings.brain_lm,
        prompt=prompt,
        fallback={"complete": True, "reason": "fallback", "steps": []},
        image_paths=[event.screenshot_path],
    )
    if bool(out.get("complete", False)):
        runtime.finished = True
        manager.log_info(f"Brain task complete reason={out.get('reason', '')}")
        return True
    raw_steps = out.get("steps", [])
    if isinstance(raw_steps, list) and raw_steps:
        created = step_tools.create_new_steps(target_path="", new_steps=[s for s in raw_steps if isinstance(s, dict)])
        manager.log_info(f"Brain check_task created additional steps count={created.get('count', 0)}")
    return False


def _with_default_image_path(arguments: dict, screenshot_path: str) -> dict:
    """
    Ensure vision tools always receive the current screenshot path.

    Models sometimes emit placeholders like `screen.png`. When that happens,
    fall back to the real screenshot path from the current event.
    """
    if "image_path" in arguments:
        patched = dict(arguments)
        patched["image_path"] = screenshot_path
        return patched
    else:
        return arguments


async def _is_interruption(active: BrainTaskState, new_event: EyeEvent) -> bool:
    """Return whether the new screenshot should interrupt processing of the active task state (two-image LM classify)."""
    manager.log_info(f"Brain classifying interruption active={active.event.screenshot_name} new={new_event.screenshot_name}")
    out, _ = await ollama.generate_json(
        settings.brain_lm,
        prompt=get_prompt("classify_interruption"),
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
    """Find the most recent stacked brain state whose screenshot name matches the new event."""
    for state in reversed(runtime.stack):
        if state.event.screenshot_name == new_event.screenshot_name:
            return state
    return None


async def _dispatch_to_hand(command: ToolCommand) -> None:
    """POST the tool command to the hand server's `/execute` endpoint."""
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
    """Choose the next hand tool call: honor scripted `tool` on the active step, else one Ollama tool call from vision + context."""
    prompt = get_prompt("brain_decide_action")
    prev_action_text = _previous_action_text(runtime.previous_action)
    memory = manager.require_paths().long_term_memory_txt.read_text(encoding="utf-8")
    step_path = runtime.active_step_path
    step = step_tools.get_step(step_path) if step_path else {"goal": task_input, "instruction": task_input}
    preferred_tool = str(step.get("tool", "")).strip()
    preferred_args = step.get("arguments", {})
    if preferred_tool in HAND_TOOL_NAMES:
        reason = f"Execute scripted step tool: {preferred_tool}"
        command = ToolCommand(
            action=preferred_tool,
            args=preferred_args if isinstance(preferred_args, dict) else {},
            screenshot_name=event.screenshot_name,
            reason=reason,
        )
        return reason, command, {"message": reason, "scripted": True}

    full_prompt = f"{prompt}\n\nTask:\n{task_input}\n\n{_step_prompt_block(step)}\n"
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

        if len(tool_calls) == 0 :
            manager.log_info("Brain no tool calls returned")
            raise ValueError("No tool calls returned")
        if len(tool_calls) > 1:
            manager.log_info(f"Brain multiple tool calls returned: {tool_calls}")
            raise ValueError("Multiple tool calls returned")

        tool_name, arguments = tool_calls[0].name, tool_calls[0].arguments
        arguments = _with_default_image_path(arguments, event.screenshot_path)
        manager.log_info(
            f"Brain selected hand action={tool_name} screenshot={event.screenshot_name}"
        )
        reason = assistant_message
        cmd = ToolCommand(
            action=tool_name,
            args=arguments,
            screenshot_name=event.screenshot_name,
            reason=assistant_message,
        )
        return reason, cmd, {"message": assistant_message}

    ollama.clear_message_history()
    manager.log_info(f"Brain tool-loop exhausted screenshot={event.screenshot_name}")
    raise ValueError("Tool loop failed")

async def _brain_loop() -> None:
    """Main worker: dequeue eye events, verify pending steps, run check_task gate, pick next step, decide action, dispatch to hand."""
    _ensure_seed_step()
    while True:
        event = await runtime.queue.get()
        runtime.latest_event = event
        tree = step_tools.read_steps_tree()
        tree["image"] = event.screenshot_name
        step_tools.write_steps_tree(tree)
        manager.log_info(
            f"Brain dequeued event screenshot={event.screenshot_name} "
            f"queue_size={runtime.queue.qsize()} stack_size={len(runtime.stack)}"
        )
        if runtime.finished:
            manager.log_info("Brain ignored event because runtime is finished")
            continue
        runtime.processing = True

        pending_path = step_tools.get_pending_step_path()
        if pending_path is not None and runtime.previous_action is not None:
            done = await _verify_pending_step(pending_path, event)
            if done:
                step_tools.mark_step_result(pending_path, "Done", message="verified complete")
            else:
                failed_step = step_tools.get_step(pending_path)
                step_tools.mark_step_result(pending_path, "Failed", message="verification failed")
                retry_steps = await _generate_retry_steps(failed_step, event)
                if retry_steps:
                    step_tools.create_new_steps(target_path=pending_path, new_steps=retry_steps)
            runtime.previous_action = None

        if await _check_task_and_expand(event):
            runtime.processing = False
            continue

        next_path = step_tools.get_next_actionable_step_path()
        if next_path is None:
            runtime.processing = False
            manager.log_info("Brain has no actionable step after check_task")
            continue

        runtime.active_step_path = next_path
        step_tools.set_step_image(next_path, event.screenshot_name)
        step_tools.mark_step_result(next_path, "Pending")
        runtime.active = BrainTaskState(event=event)
        manager.log_info(f"Brain starting decision for screenshot={event.screenshot_name}")
        thought, command, raw_response = await _decide_action(event)
        reason_preview = (thought[:160] + "…") if len(thought) > 160 else thought
        manager.log_info(
            f"Brain decided action screenshot={event.screenshot_name!r} "
            f"action={command.action!r} args={command.args!r} reason={reason_preview!r}"
        )
        manager.write_thinking_record(event.screenshot_name, thought, raw_response)
        manager.append_brain_memory(f"[{datetime.now(timezone.utc).isoformat()}] {thought}")
        await _dispatch_to_hand(command)
        runtime.active = None
        runtime.processing = False
        manager.log_info(f"Brain finished processing screenshot={event.screenshot_name}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start `_brain_loop` on app startup and cancel it on shutdown."""
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
    """Queue a new screenshot event from eye for the brain worker."""
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
    """Hand callback after executing a tool; stores result for the next verify/decide cycle."""
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
        f"[{datetime.now(timezone.utc).isoformat()}] ActionDone: {result.action} ok={result.ok} message={result.message}"
    )
    return {"status": "ack"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for the brain service."""
    return {"status": "ok"}


@app.get("/runtime_status")
async def runtime_status() -> dict[str, bool]:
    """Expose whether a task slice is active, the loop is busy, and the run has finished."""
    return {
        "has_active": runtime.active is not None,
        "processing": runtime.processing,
        "finished": runtime.finished,
    }
