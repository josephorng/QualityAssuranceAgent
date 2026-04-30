from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cua_mcp.tools import mcp_server, TOOL_FUNCTIONS
from src.common.io_utils import write_json
from src.common.models import BrainTaskState, EyeEvent, ToolCommand
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import SCRIPT_LINES_ENV, get_runtime_env
from src.common.settings import load_settings

if TYPE_CHECKING:
    from src.hand.module import HandModule

_MAX_INNER_DECIDE_STEPS = 100


@dataclass
class BrainRuntime:
    active: BrainTaskState | None = None
    latest_event: EyeEvent | None = None
    finished: bool = False
    processing: bool = False


@dataclass
class BrainStepResult:
    reason: str = ""
    finished: bool = False


class BrainModule:
    def __init__(self, hand: HandModule | None = None) -> None:
        self.settings = load_settings()
        self.ollama = OllamaClient(self.settings.ollama_host)
        self.run_root, self.task_input, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.task_input, self.run_root.name)
        self.runtime = BrainRuntime()
        self.script_lines = self._script_seed_steps()
        self._script_step_index = 0
        self._hand = hand
        self._step_transcript_counter = 0
        self.manager.log_info(f"Brain module initialized run_id={self.run_id}")

    def _save_step_messages(self, messages: list[dict[str, Any]]) -> None:
        self._step_transcript_counter += 1
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / f"{self._step_transcript_counter}.json"
        write_json(out_path, {"messages": messages})

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
        if not self.script_lines:
            return self.task_input
        if self._script_step_index >= len(self.script_lines):
            return self.script_lines[-1]
        return self.script_lines[self._script_step_index]

    async def _normalize_tool_name(self, tool_name: str, arguments: dict | None = None) -> str:
        """
        Normalize model-emitted tool names to canonical callable names.

        Some model/tool providers return names like `action:click` while local
        tool registration uses `click`.
        """
        candidate = tool_name.strip()
        tools = await mcp_server.list_tools()        
        tool_names = [tool.name for tool in tools]
        if candidate in tool_names:
            return candidate
        if ":" in candidate:
            suffix = candidate.rsplit(":", 1)[-1].strip()
            if suffix in tool_names:
                return suffix
        # Some providers return wrapper tokens (e.g. "type") and place the
        # real action in arguments.
        args = arguments if isinstance(arguments, dict) else {}
        for key in ("action", "tool", "name", "type"):
            value = args.get(key)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value in tool_names:
                return value
            if ":" in value:
                suffix = value.rsplit(":", 1)[-1].strip()
                if suffix in tool_names:
                    return suffix
        return candidate

    async def loop(self, event: EyeEvent) -> bool:
        if self._hand is None:
            raise RuntimeError("BrainModule requires hand=HandModule(...) for the decide/execute loop")

        prompt = get_prompt("brain_decide_action")
        goal = self._current_goal()
        full_prompt = f"{prompt}\n\nCurrentTaskGoal:\n{goal}\n\n"

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": full_prompt, "images": [event.screenshot_path]},
        ]
        step_succeeded = False

        for _ in range(_MAX_INNER_DECIDE_STEPS):
            response_message = await self.ollama.chat_messages(
                self.settings.brain_lm,
                messages=messages,
                use_tools=True,
            )
            if not response_message:
                self.manager.log_error("Ollama returned empty response")
                break
            messages.append(response_message.model_dump())

            if not response_message.tool_calls:
                step_succeeded = True
                break

            for tool_call in response_message.tool_calls:
                arguments = dict(tool_call.function.arguments)
                try:
                    normalized_name = await self._normalize_tool_name(tool_call.function.name, arguments)
                except Exception as e:
                    self.manager.log_error(f"Error normalizing tool name: {e}")
                    step_succeeded = False
                    break
                result = await self._hand.execute_tool_command(ToolCommand(action=normalized_name, args=arguments))
                result_body = result.model_dump(mode="json")
                messages.append({
                    "role": "tool_response",
                    "content": json.dumps(result_body, ensure_ascii=False),
                })
                if not result.ok:
                    step_succeeded = False
                    break
            else:
                continue
            break
        else:
            self.manager.log_info(
                f"Brain inner loop reached max steps ({_MAX_INNER_DECIDE_STEPS}) without model completion"
            )
            step_succeeded = False

        self._save_step_messages(messages)
        return step_succeeded

    async def process_step(self, event: EyeEvent) -> BrainStepResult:
        _tools = await mcp_server.list_tools()
        print(f"_tools: {_tools}")
        print(f"_tools type: {type(_tools)}")
        print(f"_tools[0]: {_tools[0]}")
        tool_names = {tool.name for tool in await mcp_server.list_tools()}
        tool_functions = {tool_function.__name__ for tool_function in TOOL_FUNCTIONS}
        if tool_names != tool_functions:
            only_in_tool_names = tool_names - tool_functions
            only_in_tool_functions = tool_functions - tool_names
            self.manager.log_error(
                f"Tool names and tool functions do not match.\n"
                f"Only in tool names: {only_in_tool_names}\n"
                f"Only in tool functions: {only_in_tool_functions}"
            )
            raise RuntimeError("Tool names and tool functions do not match")
       
        
        self.runtime.latest_event = event
        if self.runtime.finished:
            return BrainStepResult(finished=True)
        self.runtime.processing = True
        self.runtime.active = BrainTaskState(event=event)
        step_succeeded = await self.loop(event)
        self.runtime.active = None
        self.runtime.processing = False
        if not step_succeeded:
            return BrainStepResult(
                reason=f"Script step {self._script_step_index + 1} failed",
                finished=False,
            )

        self._script_step_index += 1
        all_steps_done = self._script_step_index >= len(self.script_lines)
        self.runtime.finished = all_steps_done
        return BrainStepResult(
            reason=f"Completed script step {self._script_step_index}/{len(self.script_lines)}",
            finished=all_steps_done,
        )

