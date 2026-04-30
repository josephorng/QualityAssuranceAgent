from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
        self.manager.log_info(f"Brain module initialized run_id={self.run_id}")

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

    def _current_goal(self) -> str:
        return self.script_lines[0] if self.script_lines else self.task_input

    def _previous_action_text(self, action: HandExecutionResult | None) -> str:
        return action.model_dump_json(exclude={"timestamp", "screenshot_name"}) if action else "none"

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
        goal = self._current_goal()
        full_prompt = f"{prompt}\n\nCurrentTaskGoal:\n{goal}\n\n"
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
        self.runtime.latest_event = event
        if self.runtime.finished:
            return BrainCycleResult(finished=True)
        self.runtime.processing = True
        self.runtime.active = BrainTaskState(event=event)
        thought, commands, raw_response = await self._decide_action(event)
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

