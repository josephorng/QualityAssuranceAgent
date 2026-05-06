from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cua_mcp.tools import mcp_server, TOOL_FUNCTIONS
from ollama import Message
from pydantic import ValidationError
from src.common.io_utils import write_json
from src.common.models import (
    BrainTaskState,
    ExecutionResult,
    ScriptStepVerifyResult,
    ToolCommand,
)
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import SCRIPT_LINES_ENV, get_runtime_env
from src.common.settings import load_settings
from time import sleep

ROLE_USER = "user"
ROLE_TOOL = "tool"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_THINKING = "thinking"

if TYPE_CHECKING:
    from src.eye.module import EyeModule
    from src.hand.module import HandModule

_MAX_INNER_DECIDE_STEPS = 100


@dataclass
class BrainRuntime:
    """Lightweight flags for brain task lifecycle (reserved for future use)."""

    active: BrainTaskState | None = None
    finished: bool = False
    processing: bool = False


@dataclass
class BrainStepResult:
    """Outcome of one `BrainModule.process_step()` call."""

    reason: str = ""
    step_finished: bool = False
    run_complete: bool = False


class BrainModule:
    """Orchestrates scripted tasks: vision + LLM tool loop, then vision verification and script branching."""

    def __init__(
        self,
        hand: HandModule | None = None,
        eye: EyeModule | None = None,
    ) -> None:
        """Initialize run state, load script lines from the environment, and retain hand/eye modules."""
        self.settings = load_settings()
        self.ollama = OllamaClient(self.settings.ollama_host)
        self.run_root, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.run_id, self.run_root.name)
        self.runtime = BrainRuntime()
        self.script_lines = self._script_seed_steps()
        self._script_step_index = 0
        self._hand = hand
        self._eye = eye
        self._step_transcript_counter = 0
        self.manager.log_info(f"Brain module initialized run_id={self.run_id}")

    def _save_step_messages(self, messages: list[dict[str, Any]]) -> None:
        """Append the current decide-loop transcript to `steps/<n>.json` under the run root."""        
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / f"{self._step_transcript_counter}_{self._script_step_index}.json"
        write_json(out_path, {"messages": messages})

    def _append_step_messages(
        self,
        messages: list[dict[str, Any]],
        transcript_counter: int,
        script_step_index: int,
        attribute_name: str = "messages",
    ) -> None:
        """Append messages under `attribute_name` in a step transcript file."""
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / f"{transcript_counter}_{script_step_index}.json"

        merged_messages: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
                    existing_messages = existing.get(attribute_name)
                    if isinstance(existing_messages, list):
                        merged_messages.extend(existing_messages)
            except (OSError, ValueError, json.JSONDecodeError):
                # If the existing transcript is malformed/unreadable, keep only new messages
                # under the requested attribute.
                merged_messages = []
                payload = {}

        merged_messages.extend(messages)
        payload[attribute_name] = merged_messages
        write_json(out_path, payload)

    def _script_seed_steps(self) -> list[str]:
        """Load non-empty script lines from `SCRIPT_LINES_ENV` (JSON array of strings)."""
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
            raise RuntimeError("No script steps found")
        lines: list[str] = []
        for item in payload:
            cleaned = item.strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    def _current_goal(self) -> str:
        """Return the goal text for `_script_step_index`, or the last line if the index is past the end."""
        if not self.script_lines:
            raise RuntimeError("No script steps found")
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

    async def _validate_tool_functions_match_mcp(self) -> None:
        """Ensure MCP-registered tool names match `TOOL_FUNCTIONS`; raises if they diverge."""
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

    def sanitize_execution_result(self, result: ExecutionResult) -> dict[str, Any]:
        """Strip noisy or redundant fields from tool execution results before logging into chat."""
        result_dict = result.model_dump()
        result_dict.pop("timestamp")
        result_dict.pop("screenshot_name")
        result_dict.pop("ok")
        pop_list = []
        for key, value in result_dict.items():
            # Keep numeric zeros (e.g. screen x/y) and False; only strip null-ish empties.
            if value is None or value == "":
                pop_list.append(key)
            elif isinstance(value, (list, dict, set)) and len(value) == 0:
                pop_list.append(key)
        for key in pop_list:
            result_dict.pop(key)
        return result_dict
    
    def sanitize_message(self, message: Message) -> dict[str, Any]:
        """Prepare an Ollama assistant message dict for persistence (drops empty keys and thinking role)."""
        message_dict = message.model_dump()
        message_dict.pop(ROLE_THINKING)
        pop_list = []
        for key, value in message_dict.items():
            if not value:
                pop_list.append(key)
        for key in pop_list:
                message_dict.pop(key)
        return message_dict

    @staticmethod
    def _parse_json_object_from_model_content(content: str) -> dict[str, Any]:
        """Parse a JSON object from model text, stripping optional markdown code fences."""
        text = (content or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)

    def _apply_verify_branch(self, result: ScriptStepVerifyResult) -> bool:
        """Apply verification `branch` to `_script_step_index`. Returns whether all script lines are done."""
        n = len(self.script_lines)
        idx = self._script_step_index

        if result.accomplished:
            if result.branch != "advance":
                self.manager.log_info(
                    f"Verify: accomplished with branch={result.branch}; advancing to next line. {result.reason}"
                )
            self._script_step_index = idx + 1
        elif result.branch == "retry":
            pass
        elif result.branch == "skip":
            self._script_step_index = idx + 1
        elif result.branch == "goto":
            assert result.target_step is not None
            target_0 = result.target_step - 1
            self._script_step_index = max(0, min(target_0, n - 1))
        elif result.branch == "advance":
            self.manager.log_info(
                f"Verify: not accomplished but branch=advance; holding step. {result.reason}"
            )

        run_complete = self._script_step_index >= n
        self.manager.log_info(
            f"Verify branch applied: index={self._script_step_index}/{n} "
            f"accomplished={result.accomplished} branch={result.branch} run_complete={run_complete} "
            f"reason={result.reason}"
        )
        return run_complete

    async def _verify_script_step(
        self,
        transcript_counter: int,
        script_step_index: int,
    ) -> ScriptStepVerifyResult | None:
        """Capture a fresh screenshot and ask the LLM (no tools) for `ScriptStepVerifyResult` JSON, or None on failure."""
        if self._eye is None:
            raise RuntimeError("BrainModule requires eye=EyeModule(...) for step verification")

        prompt = get_prompt("brain_verify_script_step")
        numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(self.script_lines, start=1))
        current_1based = min(self._script_step_index + 1, len(self.script_lines))
        goal = self._current_goal()
        body = (
            f"{prompt}\n\n"
            f"NumberedScript:\n{numbered}\n\n"
            f"CurrentStepNumber (1-based): {current_1based}\n"
            f"CurrentStepGoal:\n{goal}\n\n"
            f"All the monitor screenshot(s) are captured and will be provided to you.\n"
            "Respond with JSON only."
        )

        verification_image_paths = await self._eye.capture_separated_images()

        messages: list[dict[str, Any]] = [
            {
                "role": ROLE_USER,
                "content": body,
                "images": verification_image_paths,
            }
        ]
        response_message = await self.ollama.chat_messages(
            self.settings.brain_lm,
            messages=messages,
            use_tools=False,
        )
        if response_message:
            messages.append(response_message.model_dump())
        self._append_step_messages(
            messages,
            transcript_counter,
            script_step_index,
            attribute_name="verification",
        )
        if not response_message or not response_message.content:
            self.manager.log_error("Ollama verify step returned empty content")
            return None
        try:
            payload = self._parse_json_object_from_model_content(response_message.content)
            return ScriptStepVerifyResult.model_validate(payload)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            self.manager.log_error(f"Verify step JSON parse/validation failed: {e}")
            return None

    async def loop(self) -> bool:
        """Run the capture → LLM (with tools) → execute tools loop until the model returns no tool calls or cap is hit."""
        if self._hand is None:
            raise RuntimeError("BrainModule requires hand=HandModule(...) for the decide/execute loop")
        if self._eye is None:
            raise RuntimeError("BrainModule requires eye=EyeModule(...) to capture screenshots for the decide loop")

        goal = self._current_goal()
        first_prompt = get_prompt("brain_decide_action").format(task=goal)
        second_prompt = get_prompt("brain_decide_action_2").format(task=goal)

        messages: list[dict[str, Any]] = []
        step_succeeded = False

        for _ in range(_MAX_INNER_DECIDE_STEPS):
            try:
                all_image_paths = await self._eye.capture_separated_images()
                user_content = first_prompt if not messages else second_prompt
                messages.append(
                    {
                        "role": ROLE_USER,
                        "content": user_content,
                        "images": all_image_paths,
                    }
                )
                response_message = await self.ollama.chat_messages(
                    self.settings.brain_lm,
                    messages=messages,
                    use_tools=True,
                )
                if not response_message:
                    self.manager.log_error("Ollama returned empty response")
                    break
                response_message_dict = self.sanitize_message(response_message)
                messages.append(response_message_dict)

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
                    messages.append({
                        "role": ROLE_TOOL,
                        "content": json.dumps(self.sanitize_execution_result(result), ensure_ascii=False),
                    })
                    sleep(1)
                    if not result.ok:
                        step_succeeded = False
                        break
                else:
                    continue
                break
            finally:
                self._save_step_messages(messages)
        else:
            self.manager.log_info(
                f"Brain inner loop reached max steps ({_MAX_INNER_DECIDE_STEPS}) without model completion"
            )
            step_succeeded = False

        return step_succeeded

    async def process_step(self) -> BrainStepResult:
        """Run one script step: tool loop, then vision verification and index branching. Sets `run_complete` when the script is exhausted."""
        await self._validate_tool_functions_match_mcp()

        if self._script_step_index >= len(self.script_lines):
            return BrainStepResult(
                reason="All script steps complete",
                step_finished=True,
                run_complete=True,
            )

        step_succeeded = await self.loop()
        if not step_succeeded:
            return BrainStepResult(
                reason=f"Script step {self._script_step_index + 1} failed",
                step_finished=False,
            )

        verify_result = await self._verify_script_step(self._step_transcript_counter, self._script_step_index)
        self._step_transcript_counter += 1
        if verify_result is None:
            return BrainStepResult(
                reason="Script step verification failed (parse or empty response)",
                step_finished=False,
            )

        run_complete = self._apply_verify_branch(verify_result)
        return BrainStepResult(
            reason=f"Verify: {verify_result.reason}",
            step_finished=True,
            run_complete=run_complete,
        )

