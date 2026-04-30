from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from cua_mcp import steps as step_tools
from cua_mcp.tool_module import HAND_TOOL_NAMES
from src.common.models import BrainTaskState, EyeEvent, HandExecutionResult, ToolCommand
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import SCRIPT_LINES_ENV, get_runtime_env
from src.common.settings import load_settings


@dataclass
class BrainRuntime:
    active: BrainTaskState | None = None
    previous_action: HandExecutionResult | None = None
    active_step_path: str | None = None
    latest_event: EyeEvent | None = None
    finished: bool = False
    processing: bool = False


@dataclass
class BrainCycleResult:
    commands: list[ToolCommand] = field(default_factory=list)
    thought: str = ""
    raw_response: dict = field(default_factory=dict)
    request_capture: bool = False
    finished: bool = False


class BrainModule:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.ollama = OllamaClient(self.settings.ollama_host)
        self.run_root, self.task_input, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.task_input, self.run_root.name)
        self.runtime = BrainRuntime()
        self.script_lines = self._script_seed_steps()
        self.manager.log_info(
            f"Brain module initialized run_id={self.run_id} "
            f"ports brain={self.settings.brain_port} hand={self.settings.hand_port}"
        )

    def _script_seed_steps(self) -> list[str]:
        raw = os.environ.get(SCRIPT_LINES_ENV, "")
        payload: list[str] | None = None
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    payload = [item for item in parsed if isinstance(item, str)]
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            payload = [self.task_input]
        lines: list[str] = []
        for item in payload:
            cleaned = item.strip()
            if cleaned:
                lines.append(cleaned)
        return lines or [self.task_input]

    def _ensure_seed_step(self) -> None:
        summary = step_tools.check_task()
        if summary.get("leaf_count", 0) > 0:
            return
        step_tools.create_root_steps(
            [
                {"image": "", "goal": line, "instruction": line, "result": ""}
                for line in self.script_lines
            ]
        )

    def _step_prompt_block(self, step: dict) -> str:
        return (
            "CurrentStep:\n"
            f"- goal: {step.get('goal', '')}\n"
            f"- instruction: {step.get('instruction', '')}\n"
            f"- tool: {step.get('tool')}\n" if step.get("tool") else ""
        )

    def _current_step_goal(self, step: dict | None = None) -> str:
        if isinstance(step, dict):
            goal = str(step.get("goal", "")).strip()
            if goal:
                return goal
        if self.runtime.active_step_path:
            active_step = step_tools.get_step(self.runtime.active_step_path)
            active_goal = str(active_step.get("goal", "")).strip()
            if active_goal:
                return active_goal
        next_path = step_tools.get_next_actionable_step_path()
        if next_path:
            next_step = step_tools.get_step(next_path)
            next_goal = str(next_step.get("goal", "")).strip()
            if next_goal:
                return next_goal
        return self.script_lines[0] if self.script_lines else self.task_input

    def _previous_action_text(self, action: HandExecutionResult | None) -> str:
        return action.model_dump_json(exclude={"timestamp", "screenshot_name"}) if action else "none"

    async def _verify_pending_step(self, pending_path: str, event: EyeEvent) -> bool:
        pending_step = step_tools.get_step(pending_path)
        prompt = (
            "Determine whether the pending task step is now completed based on the latest screenshot.\n"
            'Return strict JSON: {"done": <bool>, "reason": <string>}.\n\n'
            f"CurrentStepGoal:\n{self._current_step_goal(pending_step)}\n\n"
            f"{self._step_prompt_block(pending_step)}\n"
            f"PreviousAction:\n{self._previous_action_text(self.runtime.previous_action)}\n"
        )
        out, _ = await self.ollama.generate_json(
            self.settings.brain_lm,
            prompt=prompt,
            fallback={"done": False, "reason": "fallback"},
            image_paths=[event.screenshot_path],
        )
        done = bool(out.get("done", False))
        self.manager.log_info(
            f"Brain verify pending path={pending_path} done={done} reason={out.get('reason', '')}"
        )
        return done

    async def _generate_retry_steps(self, step: dict, event: EyeEvent) -> list[dict]:
        step_goal = self._current_step_goal(step)
        prompt = (
            "Create alternative next steps to retry the same goal with a different method.\n"
            'Return strict JSON: {"steps": [{"goal": "...", "instruction": "...", "tool": "...", "arguments": {}, "result": ""}]}\n'
            "Each step must be executable and concise.\n\n"
            f"CurrentStepGoal:\n{step_goal}\n\n"
            f"FailedStep:\n{json.dumps(step, ensure_ascii=False)}\n"
        )
        out, _ = await self.ollama.generate_json(
            self.settings.brain_lm,
            prompt=prompt,
            fallback={"steps": [{"goal": step.get("goal", step_goal), "instruction": step.get("instruction", step_goal), "result": ""}]},
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
                    "goal": str(candidate.get("goal", step.get("goal", ""))),
                    "instruction": str(candidate.get("instruction", step.get("instruction", ""))),
                    "tool": str(candidate.get("tool", "")),
                    "arguments": candidate.get("arguments", {}),
                    "result": str(candidate.get("result", "")),
                }
            )
        return cleaned[:3] if cleaned else []

    async def _check_task_and_expand(self, event: EyeEvent) -> bool:
        summary = step_tools.check_task()
        if summary.get("pending_count", 0) > 0 or summary.get("undone_count", 0) > 0:
            return False
        prompt = (
            "Decide whether the full user task is truly complete.\n"
            'Return strict JSON: {"complete": <bool>, "reason": <string>, "step": {"goal":"...","instruction":"...","result":""}}\n'
            "If not complete, provide one additional executable step in `step`.\n\n"
            f"CurrentStepGoal:\n{self._current_step_goal()}\n"
        )
        out, _ = await self.ollama.generate_json(
            self.settings.brain_lm,
            prompt=prompt,
            fallback={"complete": True, "reason": "fallback", "steps": []},
            image_paths=[event.screenshot_path],
        )
        if bool(out.get("complete", False)):
            self.runtime.finished = True
            self.manager.log_info(f"Brain task complete reason={out.get('reason', '')}")
            return True
        raw_step = out.get("step", {})
        if isinstance(raw_step, dict) and raw_step:
            step_tools.create_new_step(target_path="", new_step=raw_step)
            self.manager.log_info("Brain check_task created additional step")
        return False

    def _normalize_tool_name(self, tool_name: str, arguments: dict | None = None) -> str:
        """
        Normalize model-emitted tool names to canonical callable names.

        Some model/tool providers return names like `action:click` while local
        tool registration uses `click`.
        """
        candidate = tool_name.strip()
        if candidate in HAND_TOOL_NAMES:
            return candidate
        if ":" in candidate:
            suffix = candidate.rsplit(":", 1)[-1].strip()
            if suffix in HAND_TOOL_NAMES:
                return suffix
        # Some providers return wrapper tokens (e.g. "type") and place the
        # real action in arguments.
        args = arguments if isinstance(arguments, dict) else {}
        for key in ("action", "tool", "name", "type"):
            value = args.get(key)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value in HAND_TOOL_NAMES:
                return value
            if ":" in value:
                suffix = value.rsplit(":", 1)[-1].strip()
                if suffix in HAND_TOOL_NAMES:
                    return suffix
        return candidate

    async def _decide_action(self, event: EyeEvent) -> tuple[str, list[ToolCommand], dict]:
        prompt = get_prompt("brain_decide_action")
        prev_action_text = self._previous_action_text(self.runtime.previous_action)
        memory = self.manager.require_paths().long_term_memory_txt.read_text(encoding="utf-8")
        step_path = self.runtime.active_step_path
        step = step_tools.get_step(step_path) if step_path else {"goal": self._current_step_goal(), "instruction": self._current_step_goal()}
        step_goal = self._current_step_goal(step)
        preferred_tool = str(step.get("tool", "")).strip()
        preferred_args = step.get("arguments", {})
        if preferred_tool in HAND_TOOL_NAMES:
            reason = f"Execute scripted step tool: {preferred_tool}"
            return reason, [
                ToolCommand(
                    action=preferred_tool,
                    args=preferred_args if isinstance(preferred_args, dict) else {},
                    screenshot_name=event.screenshot_name,
                    reason=reason,
                )
            ], {"message": reason, "scripted": True}

        full_prompt = f"{prompt}\n\nCurrentStepGoal:\n{step_goal}\n\n{self._step_prompt_block(step)}\n"
        if prev_action_text != "none":
            full_prompt += f"PreviousAction:\n{prev_action_text}\n\n"
        if memory:
            full_prompt += f"Memory:\n{memory}"

        assistant_message, tool_calls = await self.ollama.generate(
            self.settings.brain_lm,
            prompt=full_prompt,
            image_paths=[event.screenshot_path],
            use_tools=True,
            store_messages=True,
        )
        if not tool_calls:
            raise ValueError("No tool calls returned")

        commands: list[ToolCommand] = []
        for tool_call in tool_calls:
            arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            normalized_name = self._normalize_tool_name(tool_call.name, arguments)
            if normalized_name not in HAND_TOOL_NAMES:
                raise ValueError(f"Unknown tool call returned: {tool_call.name}")
            tool_instruction = arguments.get("instruction")
            if isinstance(tool_instruction, str) and tool_instruction.strip() and self.runtime.active_step_path:
                step_tools.set_step_instruction(self.runtime.active_step_path, tool_instruction.strip())
            commands.append(
                ToolCommand(
                    action=normalized_name,
                    args=arguments,
                    screenshot_name=event.screenshot_name,
                    reason=assistant_message,
                )
            )
        return assistant_message, commands, {"message": assistant_message}

    async def process_eye_event(self, event: EyeEvent) -> BrainCycleResult:
        self._ensure_seed_step()
        self.runtime.latest_event = event
        if self.runtime.finished:
            return BrainCycleResult(finished=True)
        self.runtime.processing = True

        pending_path = step_tools.get_pending_step_path()
        if pending_path and self.runtime.previous_action is not None:
            done = await self._verify_pending_step(pending_path, event)
            if done:
                step_tools.mark_step_result(pending_path, "Done", message="verified complete")
            else:
                failed_step = step_tools.get_step(pending_path)
                step_tools.mark_step_result(pending_path, "Failed", message="verification failed")
                for retry_step in await self._generate_retry_steps(failed_step, event):
                    step_tools.create_new_step(target_path=pending_path, new_step=retry_step)
            self.runtime.previous_action = None

        if await self._check_task_and_expand(event):
            self.runtime.processing = False
            return BrainCycleResult(finished=True)

        next_path = step_tools.get_next_actionable_step_path()
        if next_path is None:
            self.runtime.processing = False
            return BrainCycleResult()

        self.runtime.active_step_path = next_path
        step_tools.set_step_image(next_path, event.screenshot_name)
        step_tools.mark_step_result(next_path, "Pending")
        self.runtime.active = BrainTaskState(event=event)
        thought, commands, raw_response = await self._decide_action(event)
        if self.runtime.active_step_path:
            step_tools.set_step_tool_calls(
                self.runtime.active_step_path,
                [{"action": command.action, "args": command.args, "reason": command.reason} for command in commands],
            )
        self.manager.write_thinking_record(event.screenshot_name, thought, raw_response)
        self.manager.append_brain_memory(f"[{datetime.now(timezone.utc).isoformat()}] {thought}")
        self.runtime.active = None
        self.runtime.processing = False
        return BrainCycleResult(
            commands=commands,
            thought=thought,
            raw_response=raw_response,
            request_capture=bool(commands),
            finished=self.runtime.finished,
        )

    async def on_action_done(self, result: HandExecutionResult) -> None:
        self.runtime.previous_action = result
        self.manager.append_brain_memory(
            f"[{datetime.now(timezone.utc).isoformat()}] ActionDone: {result.action} ok={result.ok} message={result.message}"
        )

