from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from cua_mcp.tool_module import HAND_TOOL_NAMES
from src.common.io_utils import write_json
from src.common.models import BrainTaskState, EyeEvent, HandExecutionResult, ToolCommand, ToolCommandAction
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
    def __init__(self, hand: HandModule | None = None) -> None:
        self.settings = load_settings()
        self.ollama = OllamaClient(self.settings.ollama_host)
        self.run_root, self.task_input, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.task_input, self.run_root.name)
        self.runtime = BrainRuntime()
        self.script_lines = self._script_seed_steps()
        self._hand = hand
        self._step_transcript_counter = 0
        self.manager.log_info(f"Brain module initialized run_id={self.run_id}")

    def _save_step_transcript(self, messages: list[dict[str, Any]]) -> None:
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

    async def _decide_action(self, event: EyeEvent) -> tuple[str, list[ToolCommand], dict[str, Any]]:
        if self._hand is None:
            raise RuntimeError("BrainModule requires hand=HandModule(...) for the decide/execute loop")

        prompt = get_prompt("brain_decide_action")
        prev_action_text = self._previous_action_text(self.runtime.previous_action)
        memory = self.manager.require_paths().long_term_memory_txt.read_text(encoding="utf-8")
        goal = self._current_goal()
        full_prompt = f"{prompt}\n\nCurrentTaskGoal:\n{goal}\n\n"
        if prev_action_text != "none":
            full_prompt += f"PreviousAction:\n{prev_action_text}\n\n"
        if memory:
            full_prompt += f"Memory:\n{memory}"

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": full_prompt, "images": [event.screenshot_path]},
        ]
        transcript: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": full_prompt,
                "images": [event.screenshot_name],
            },
        ]

        last_assistant = ""
        for _ in range(_MAX_INNER_DECIDE_STEPS):
            text, tool_calls, response_message = await self.ollama.chat_messages(
                self.settings.brain_lm,
                messages=messages,
                use_tools=True,
            )
            last_assistant = text
            asst_for_api = dict(response_message) if response_message else {"role": "assistant", "content": text}
            if asst_for_api.get("role") not in ("assistant",):
                asst_for_api = {**asst_for_api, "role": "assistant"}
            messages.append(asst_for_api)

            tc_serialized = [
                {"name": tc.name, "arguments": dict(tc.arguments) if isinstance(tc.arguments, dict) else {}}
                for tc in tool_calls
            ]
            transcript.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_use": tc_serialized,
                },
            )

            if not tool_calls:
                self.runtime.finished = True
                break

            for tool_call in tool_calls:
                arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                normalized_name = self._normalize_tool_name(tool_call.name, arguments)
                if normalized_name not in HAND_TOOL_NAMES:
                    raise ValueError(f"Unknown tool call returned: {tool_call.name}")
                cmd = ToolCommand(
                    action=cast(ToolCommandAction, normalized_name),
                    args=arguments,
                    screenshot_name=event.screenshot_name,
                    reason=text,
                )
                result = await self._hand.execute_tool_command(cmd)
                await self.on_action_done(result)
                result_body = result.model_dump(mode="json")
                transcript.append(
                    {
                        "role": "tool",
                        "tool_name": normalized_name,
                        "tool_arguments": arguments,
                        "tool_result": result_body,
                    },
                )
                tool_msg_api = {
                    "role": "tool",
                    "content": json.dumps(result_body, ensure_ascii=False),
                }
                messages.append(tool_msg_api)
        else:
            self.manager.log_info(
                f"Brain inner loop reached max steps ({_MAX_INNER_DECIDE_STEPS}) without model completion"
            )

        self._save_step_transcript(transcript)
        raw: dict[str, Any] = {"message": last_assistant, "messages": transcript}
        return last_assistant, [], raw

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
            request_capture=not self.runtime.finished,
            finished=self.runtime.finished,
        )

    async def on_action_done(self, result: HandExecutionResult) -> None:
        self.runtime.previous_action = result
        self.manager.append_brain_memory(
            f"[{datetime.now(timezone.utc).isoformat()}] ActionDone: {result.action} ok={result.ok} message={result.message}"
        )

