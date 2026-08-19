from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any

from cua_mcp.tools import (
    TOOL_FUNCTIONS,
    VERIFICATION_TOOLS,
    get_mode_tool_functions,
    mcp_server,
)
from ollama import Message
from pydantic import ValidationError
from src.common.io_utils import write_json
from cua_mcp.llm_json import extract_json_object_string, parse_json_object
from src.common.models import (
    BrainStepOutcome,
    BrainTaskState,
    ExecutionResult,
    ScriptStepVerifyResult,
    ToolCommand,
)
from src.common.instruction_tool_cache import (
    extract_tool_calls_from_messages,
    lookup_tool_calls,
    upsert_tool_calls,
)
from src.common.llm_factory import get_llm_client
from src.common.nearby_side import enrich_tool_arguments_from_goal
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import (
    SCRIPT_LINES_ENV,
    SCRIPT_OUTCOMES_ENV,
    get_runtime_env,
    is_runtime_command_mode,
    is_smart_mode,
    use_tool_cache_enabled,
)
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

_MAX_INNER_DECIDE_STEPS = 10
# Models sometimes invent these as tool calls when asked to end the step with JSON.
_PSEUDO_END_TOOL_NAMES = frozenset({"finish", "done", "complete", "end", "end_step"})


# Remove key-value pairs with None values in each message
def prune_nulls(d):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if v is not None}


def stamp_message(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a chat message with `timestamp_utc` set if missing."""
    stamped = prune_nulls(message)
    if "timestamp_utc" not in stamped:
        stamped["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return stamped

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
    step_index: int | None = None


class BrainModule:
    """Orchestrates scripted tasks: vision + LLM tool loop, then vision verification and script branching."""

    def __init__(
        self,
        hand: HandModule | None = None,
        eye: EyeModule | None = None,
    ) -> None:
        """Initialize run state, load script lines from the environment, and retain hand/eye modules."""
        self.settings = load_settings()
        self.ollama = get_llm_client()
        self.run_root, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.run_id, self.run_root.name)
        self.runtime = BrainRuntime()
        self.script_lines = (
            []
            if is_runtime_command_mode() or is_smart_mode()
            else self._script_seed_steps()
        )
        self.script_expected_outcomes = (
            []
            if is_runtime_command_mode() or is_smart_mode()
            else self._script_seed_outcomes(len(self.script_lines))
        )
        self._script_step_index = 0
        self._hand = hand
        self._eye = eye
        self._step_transcript_counter = (
            self._resume_step_transcript_counter()
            if is_runtime_command_mode() or is_smart_mode()
            else 0
        )
        self.manager.log_info(f"Brain module initialized run_id={self.run_id}")

    def _resume_step_transcript_counter(self) -> int:
        """Continue numbering runtime steps after script steps in the same run folder."""
        steps_dir = self.manager.require_paths().root / "steps"
        if not steps_dir.is_dir():
            return 0
        max_tc = -1
        for path in steps_dir.iterdir():
            if path.suffix not in (".json", ".log"):
                continue
            stem = path.stem
            if "_" not in stem:
                continue
            try:
                max_tc = max(max_tc, int(stem.split("_", 1)[0]))
            except ValueError:
                continue
        return max_tc + 1

    def _save_step_messages(self, messages: list[dict[str, Any]]) -> None:
        """Save or update the decide-loop transcript under `steps/<n>.json`."""        
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / f"{self._step_transcript_counter}_{self._script_step_index}.json"
        payload: dict[str, Any] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
        payload["messages"] = [stamp_message(msg) for msg in messages]
        write_json(out_path, payload)

    def _update_step_metadata(
        self,
        transcript_counter: int,
        script_step_index: int,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert `step_timing` metadata for one step transcript file."""
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / f"{transcript_counter}_{script_step_index}.json"

        payload: dict[str, Any] = {}
        existing_metadata: dict[str, Any] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
                    existing_timing = existing.get("step_timing")
                    if isinstance(existing_timing, dict):
                        existing_metadata = dict(existing_timing)
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
                existing_metadata = {}

        existing_metadata.update(metadata)
        payload["step_timing"] = existing_metadata
        write_json(out_path, payload)

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

        merged_messages.extend([stamp_message(msg) for msg in messages])
 
        payload[attribute_name] = merged_messages
        write_json(out_path, payload)

    def _append_failed_tool_call(
        self,
        tool_name: str,
        transcript_counter: int,
        script_step_index: int,
    ) -> None:
        """Append a failed tool name to `failed_tool_calls` in a step transcript file."""
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / f"{transcript_counter}_{script_step_index}.json"

        payload: dict[str, Any] = {}
        failed_tool_calls: list[str] = []
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
                    existing_failed = existing.get("failed_tool_calls")
                    if isinstance(existing_failed, list):
                        failed_tool_calls.extend(
                            item for item in existing_failed if isinstance(item, str)
                        )
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
                failed_tool_calls = []

        failed_tool_calls.append(tool_name)
        payload["failed_tool_calls"] = failed_tool_calls
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

    def _script_seed_outcomes(self, step_count: int) -> list[str | None]:
        """Load optional expected outcomes aligned with script steps."""
        raw = os.environ.get(SCRIPT_OUTCOMES_ENV, "")
        outcomes: list[str | None] = [None] * step_count
        if not raw:
            return outcomes
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return outcomes
        if not isinstance(parsed, list):
            return outcomes
        for index in range(min(step_count, len(parsed))):
            item = parsed[index]
            if isinstance(item, str) and item.strip():
                outcomes[index] = item.strip()
        return outcomes

    def _current_expected_outcome(self) -> str:
        if not self.script_expected_outcomes:
            return ""
        if self._script_step_index >= len(self.script_expected_outcomes):
            return ""
        value = self.script_expected_outcomes[self._script_step_index]
        return value.strip() if isinstance(value, str) else ""

    def _format_numbered_script(self) -> str:
        """Numbered script lines with each step's recorded expected outcome."""
        lines: list[str] = []
        for index, line in enumerate(self.script_lines, start=1):
            outcome: str | None = None
            if index - 1 < len(self.script_expected_outcomes):
                outcome = self.script_expected_outcomes[index - 1]
            if isinstance(outcome, str) and outcome.strip():
                outcome_text = outcome.strip()
            else:
                outcome_text = "(none)"
            lines.append(f"{index}. {line}  | expected: {outcome_text}")
        return "\n".join(lines)

    def prepare_runtime_step(self, command: str) -> None:
        """Set a single script line for the next `process_step()` (runtime command mode)."""
        cleaned = command.strip()
        if not cleaned:
            raise ValueError("runtime step command must be non-empty")
        self.script_lines = [cleaned]
        self.script_expected_outcomes = [None]
        self._script_step_index = 0

    async def execute_instruction(self, instruction: str) -> bool:
        """
        Run the existing multi-tool decide/act loop against one bounded instruction.

        Used by smart mode as the Act step. Preserves surrounding script state and
        advances the step transcript counter after the actor finishes.
        """
        cleaned = instruction.strip()
        if not cleaned:
            raise ValueError("instruction must be non-empty")

        saved_lines = list(self.script_lines)
        saved_outcomes = list(self.script_expected_outcomes)
        saved_index = self._script_step_index
        transcript_counter = self._step_transcript_counter
        script_step_index = 0
        self.script_lines = [cleaned]
        self.script_expected_outcomes = [None]
        self._script_step_index = 0
        self.manager.set_step_log_context(transcript_counter, script_step_index)
        started_iso = datetime.now(timezone.utc).isoformat()
        started_at = perf_counter()
        try:
            step_succeeded = await self.loop()
            finished_iso = datetime.now(timezone.utc).isoformat()
            duration_seconds = round(perf_counter() - started_at, 3)
            self._update_step_metadata(
                transcript_counter,
                script_step_index,
                {
                    "started_at_utc": started_iso,
                    "finished_at_utc": finished_iso,
                    "duration_seconds": duration_seconds,
                    "status": "completed" if step_succeeded else "failed",
                    "step_index": script_step_index,
                    "goal": cleaned,
                },
            )
            self._step_transcript_counter += 1
            return step_succeeded
        finally:
            self.script_lines = saved_lines
            self.script_expected_outcomes = saved_outcomes
            self._script_step_index = saved_index
            self.manager.clear_step_log_context()

    def undo_last_runtime_step(self) -> bool:
        """Remove the last completed step transcript and rewind the step counter."""
        if self._step_transcript_counter <= 0:
            return False
        tc = self._step_transcript_counter - 1
        si = 0
        steps_dir = self.manager.require_paths().root / "steps"
        for ext in (".json", ".log"):
            path = steps_dir / f"{tc}_{si}{ext}"
            if path.exists():
                path.unlink()
        self._step_transcript_counter = tc
        self._script_step_index = 0
        return True

    def _current_goal(self) -> str:
        """Return the goal text for `_script_step_index`, or the last line if the index is past the end."""
        if not self.script_lines:
            raise RuntimeError("No script steps found")
        if self._script_step_index >= len(self.script_lines):
            return self.script_lines[-1]
        return self.script_lines[self._script_step_index]

    @staticmethod
    def _primary_decision_screenshot(image_paths: list[str]) -> str | None:
        if not image_paths:
            return None
        return image_paths[0]

    def _enrich_tool_arguments(
        self, tool_name: str, arguments: dict[str, Any], goal: str
    ) -> dict[str, Any]:
        """Restore directed nearby sides from the step goal when the model stripped them."""
        enriched = enrich_tool_arguments_from_goal(tool_name, arguments, goal)
        for key in (
            "nearby_objects",
            "start_nearby_objects",
            "destination_nearby_objects",
        ):
            before = arguments.get(key)
            after = enriched.get(key)
            if after is not None and after != before:
                self.manager.log_info(
                    f"{tool_name}: restored nearby sides from goal "
                    f"{key}={before!r} -> {after!r}"
                )
        return enriched

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
        result_dict.pop("screenshot_name", None)
        result_dict.pop("screenshot_before_path", None)
        result_dict.pop("screenshot_after_path", None)
        result_dict = prune_nulls(result_dict)       
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
        return parse_json_object(
            content,
            empty_error="Model returned empty content; expected a JSON object",
            decode_error_prefix="Model JSON decode failed",
        )

    @staticmethod
    def _recover_step_outcome_payload(content: str) -> dict[str, Any] | None:
        """Best-effort extract of status/reason when model JSON is malformed.

        Handles common LLM mistakes such as unescaped quotes inside ``reason``.
        """
        text = extract_json_object_string(content)
        status_match = re.search(r'"status"\s*:\s*"(completed|failed)"', text)
        if not status_match:
            return None
        reason = ""
        reason_match = re.search(r'"reason"\s*:\s*"(.*)"\s*}\s*$', text, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1)
        return {"status": status_match.group(1), "reason": reason}

    @staticmethod
    def _is_pseudo_end_tool_name(tool_name: str | None) -> bool:
        """True for invented end-of-step tool names (not registered MCP tools)."""
        if not tool_name:
            return False
        return tool_name.strip().lower() in _PSEUDO_END_TOOL_NAMES

    def _parse_step_outcome(self, content: str | None) -> BrainStepOutcome | None:
        """Parse a structured step finish reply, or None if content is missing/invalid."""
        if not (content or "").strip():
            return None
        try:
            payload = self._parse_json_object_from_model_content(content or "")
            return BrainStepOutcome.model_validate(payload)
        except (ValueError, ValidationError, TypeError) as e:
            recovered = self._recover_step_outcome_payload(content or "")
            if recovered is not None:
                try:
                    outcome = BrainStepOutcome.model_validate(recovered)
                    self.manager.log_info(
                        "Step outcome JSON was malformed; recovered status/reason via fallback parser"
                    )
                    return outcome
                except (ValidationError, TypeError):
                    pass
            self.manager.log_error(f"Step outcome JSON parse/validation failed: {e}")
            return None

    def _parse_step_outcome_from_arguments(
        self, arguments: dict[str, Any] | None
    ) -> BrainStepOutcome | None:
        """Parse step outcome from a pseudo end-tool's arguments (status/reason)."""
        if not isinstance(arguments, dict):
            return None
        try:
            return BrainStepOutcome.model_validate(arguments)
        except (ValidationError, TypeError):
            status = arguments.get("status")
            if status not in ("completed", "failed"):
                return None
            reason = arguments.get("reason", "")
            if reason is None:
                reason = ""
            try:
                return BrainStepOutcome(status=status, reason=str(reason))
            except (ValidationError, TypeError):
                return None

    def _resolve_step_success_from_outcome(
        self,
        outcome: BrainStepOutcome | None,
        unresolved_tool_failures: set[str],
        *,
        missing_outcome_message: str,
    ) -> bool:
        """Apply a parsed step outcome to success/failure, logging the decision."""
        if outcome is None:
            self.manager.log_error(missing_outcome_message)
            return False
        if outcome.status == "completed":
            if unresolved_tool_failures:
                failed = ", ".join(sorted(unresolved_tool_failures))
                self.manager.log_info(
                    f"Model marked completed but unresolved tool failure(s) remain "
                    f"({failed}); treating step as failed. Model reason: {outcome.reason}"
                )
                return False
            self.manager.log_info(f"Step marked completed by model: {outcome.reason}")
            return True
        self.manager.log_info(f"Step marked failed by model: {outcome.reason}")
        return False

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

        prompt = get_prompt("brain_verify_script_step").format(
            expected_outcome=self._current_expected_outcome() or "(none)",
        )
        numbered = self._format_numbered_script()
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
            stamp_message(
                {
                    "role": ROLE_USER,
                    "content": body,
                    "images": verification_image_paths,
                }
            )
        ]
        response_message = await self.ollama.chat_messages(
            self.settings.brain_lm,
            messages=messages,
            tools=VERIFICATION_TOOLS,
        )
        if response_message:
            messages.append(stamp_message(response_message.model_dump()))
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

    async def _try_replay_cached_tools(
        self,
        goal: str,
        cached_calls: list[dict[str, Any]],
    ) -> bool:
        """Execute cached tool calls in order. Returns True only if every tool succeeds."""
        all_image_paths = await self._eye.capture_separated_images()
        before_screenshot = self._primary_decision_screenshot(all_image_paths)
        messages: list[dict[str, Any]] = [
            stamp_message(
                {
                    "role": ROLE_USER,
                    "content": f"Cache replay for task: {goal}",
                    "images": all_image_paths,
                    "cache_replay": True,
                }
            ),
            stamp_message(
                {
                    "role": ROLE_ASSISTANT,
                    "tool_calls": [
                        {
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            }
                        }
                        for call in cached_calls
                    ],
                    "cache_replay": True,
                }
            ),
        ]

        for call in cached_calls:
            arguments = dict(call["arguments"])
            try:
                normalized_name = await self._normalize_tool_name(call["name"], arguments)
            except Exception as e:
                self.manager.log_error(f"Cache replay: error normalizing tool name: {e}")
                self._save_step_messages(messages)
                return False
            arguments = self._enrich_tool_arguments(normalized_name, arguments, goal)
            result = await self._hand.execute_tool_command(
                ToolCommand(
                    action=normalized_name,
                    args=arguments,
                    screenshot_before_path=before_screenshot,
                )
            )
            messages.append(
                stamp_message(
                    {
                        "role": ROLE_TOOL,
                        "content": json.dumps(
                            self.sanitize_execution_result(result),
                            ensure_ascii=False,
                        ),
                    }
                )
            )
            sleep(1)
            if not result.ok:
                self._append_failed_tool_call(
                    result.action,
                    self._step_transcript_counter,
                    self._script_step_index,
                )
                self._save_step_messages(messages)
                return False

        self._save_step_messages(messages)
        return True

    async def loop(self) -> bool:
        """Run the capture → LLM (with tools) → execute tools loop until the model returns no tool calls or cap is hit."""
        if self._hand is None:
            raise RuntimeError("BrainModule requires hand=HandModule(...) for the decide/execute loop")
        if self._eye is None:
            raise RuntimeError("BrainModule requires eye=EyeModule(...) to capture screenshots for the decide loop")

        goal = self._current_goal()
        first_prompt = get_prompt("brain_decide_action").format(task=goal)
        second_prompt = get_prompt("brain_decide_action_2").format(task=goal)
        tool_functions = get_mode_tool_functions()
        tool_names = {tool.__name__ for tool in tool_functions}
        if "move_mouse_visual" in tool_names and "move_mouse" not in tool_names:
            mode_tool_policy = (
                "\n\nMode tool policy: move_mouse is unavailable. Use move_mouse_visual "
                "for movement to a named on-screen target."
            )
        elif "move_mouse" in tool_names and "move_mouse_visual" not in tool_names:
            mode_tool_policy = (
                "\n\nMode tool policy: move_mouse_visual is unavailable. Use move_mouse "
                "for movement to a named on-screen target."
            )
        else:
            mode_tool_policy = ""
        first_prompt += mode_tool_policy
        second_prompt += mode_tool_policy

        if use_tool_cache_enabled():
            cached_calls = lookup_tool_calls(goal)
            if cached_calls:
                self.manager.log_info(
                    f"Instruction tool cache hit ({len(cached_calls)} tool call(s)); replaying"
                )
                if await self._try_replay_cached_tools(goal, cached_calls):
                    return True
                self.manager.log_info("Cache replay failed; falling back to LLM decide loop")

        messages: list[dict[str, Any]] = []
        step_succeeded = False
        llm_path_used = False
        # Actions that returned ok=false and have not succeeded on a later retry.
        unresolved_tool_failures: set[str] = set()

        for _ in range(_MAX_INNER_DECIDE_STEPS):
            try:
                llm_path_used = True
                all_image_paths = await self._eye.capture_separated_images()
                before_screenshot = self._primary_decision_screenshot(all_image_paths)
                user_content = first_prompt if not messages else second_prompt
                messages.append(
                    stamp_message(
                        {
                            "role": ROLE_USER,
                            "content": user_content,
                            "images": all_image_paths,
                        }
                    )
                )
                response_message = await self.ollama.chat_messages(
                    self.settings.brain_lm,
                    messages=messages,
                    tools=tool_functions,
                )
                if not response_message:
                    self.manager.log_error("Ollama returned empty response")
                    break
                response_message_dict = stamp_message(self.sanitize_message(response_message))
                messages.append(response_message_dict)

                raw_tool_calls = list(response_message.tool_calls or [])
                real_tool_calls = [
                    call
                    for call in raw_tool_calls
                    if not self._is_pseudo_end_tool_name(
                        getattr(getattr(call, "function", None), "name", None)
                    )
                ]
                pseudo_end_calls = [
                    call
                    for call in raw_tool_calls
                    if self._is_pseudo_end_tool_name(
                        getattr(getattr(call, "function", None), "name", None)
                    )
                ]

                if not raw_tool_calls:
                    outcome = self._parse_step_outcome(
                        getattr(response_message, "content", None)
                    )
                    step_succeeded = self._resolve_step_success_from_outcome(
                        outcome,
                        unresolved_tool_failures,
                        missing_outcome_message=(
                            "Decide loop ended without tools but step outcome JSON was "
                            "missing/invalid"
                        ),
                    )
                    break

                if not real_tool_calls and pseudo_end_calls:
                    end_call = pseudo_end_calls[0]
                    end_name = getattr(end_call.function, "name", "finish")
                    end_args = dict(getattr(end_call.function, "arguments", {}) or {})
                    outcome = self._parse_step_outcome_from_arguments(end_args)
                    if outcome is None:
                        outcome = self._parse_step_outcome(
                            getattr(response_message, "content", None)
                        )
                    self.manager.log_info(
                        f"Model emitted pseudo end tool '{end_name}'; treating arguments "
                        "as step outcome JSON (not executing)"
                    )
                    step_succeeded = self._resolve_step_success_from_outcome(
                        outcome,
                        unresolved_tool_failures,
                        missing_outcome_message=(
                            f"Pseudo end tool '{end_name}' lacked valid status/reason"
                        ),
                    )
                    break

                if pseudo_end_calls:
                    ignored = ", ".join(
                        sorted(
                            {
                                str(getattr(call.function, "name", "finish"))
                                for call in pseudo_end_calls
                            }
                        )
                    )
                    self.manager.log_info(
                        f"Ignoring pseudo end tool(s) mixed with real tools: {ignored}"
                    )

                abort_step = False
                for tool_call in real_tool_calls:
                    arguments = dict(tool_call.function.arguments)
                    try:
                        normalized_name = await self._normalize_tool_name(
                            tool_call.function.name, arguments
                        )
                    except Exception as e:
                        self.manager.log_error(f"Error normalizing tool name: {e}")
                        step_succeeded = False
                        abort_step = True
                        break
                    arguments = self._enrich_tool_arguments(
                        normalized_name, arguments, goal
                    )
                    result = await self._hand.execute_tool_command(
                        ToolCommand(
                            action=normalized_name,
                            args=arguments,
                            screenshot_before_path=before_screenshot,
                        )
                    )
                    messages.append(
                        stamp_message(
                            {
                                "role": ROLE_TOOL,
                                "content": json.dumps(
                                    self.sanitize_execution_result(result),
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    )
                    sleep(1)
                    if not result.ok:
                        self._append_failed_tool_call(
                            result.action,
                            self._step_transcript_counter,
                            self._script_step_index,
                        )
                        unresolved_tool_failures.add(result.action)
                        # Do not run dependent follow-ups (e.g. click after failed move_mouse).
                        # Continue the decide loop so the model can retry or end as failed.
                        break
                    unresolved_tool_failures.discard(result.action)
                if abort_step:
                    break
                continue
            finally:
                self._save_step_messages(messages)
        else:
            self.manager.log_info(
                f"Brain inner loop reached max steps ({_MAX_INNER_DECIDE_STEPS}) without model completion"
            )
            step_succeeded = False

        if step_succeeded and llm_path_used:
            tool_calls = extract_tool_calls_from_messages(messages)
            if tool_calls:
                upsert_tool_calls(goal, tool_calls, source_run_id=self.run_id)
                self.manager.log_info(
                    f"Instruction tool cache updated ({len(tool_calls)} tool call(s))"
                )

        return step_succeeded

    def _should_skip_vision_verify(self, step_succeeded: bool) -> bool:
        """True when the actor succeeded and there is no recorded visual success criterion.

        `step_succeeded` already means the model marked completed and every tool
        that failed was later retried successfully (no unresolved `ok=false`).
        """
        return bool(step_succeeded) and not self._current_expected_outcome()

    @staticmethod
    def _auto_advance_verify_result() -> ScriptStepVerifyResult:
        return ScriptStepVerifyResult(
            accomplished=True,
            branch="advance",
            target_step=None,
            reason="Actor completed the step with all tools ok; no recorded expected outcome.",
        )

    async def process_step(self) -> BrainStepResult:
        """Run one script step: tool loop, then verification and index branching.

        Happy path: empty expected outcome + actor success (all tools ok) auto-advances
        without a screenshot or verifier LLM. Recovery path: actor failure or a recorded
        expected outcome still uses screenshot verification for `goto`/`retry`/`skip`.
        Sets `run_complete` when the script is exhausted.
        """
        # await self._validate_tool_functions_match_mcp()

        if self._script_step_index >= len(self.script_lines):
            return BrainStepResult(
                reason="All script steps complete",
                step_finished=True,
                run_complete=True,
            )

        transcript_counter = self._step_transcript_counter
        script_step_index = self._script_step_index
        self.manager.set_step_log_context(transcript_counter, script_step_index)
        try:
            started_iso = datetime.now(timezone.utc).isoformat()
            started_at = perf_counter()

            step_succeeded = await self.loop()
            if self._should_skip_vision_verify(step_succeeded):
                self.manager.log_info(
                    f"Script step {script_step_index + 1} skipping vision verification "
                    "(empty expected outcome; actor tools succeeded)"
                )
                verify_result = self._auto_advance_verify_result()
            else:
                if not step_succeeded:
                    self.manager.log_info(
                        f"Script step {script_step_index + 1} actor failed; "
                        "running verification for recovery"
                    )
                verify_result = await self._verify_script_step(
                    transcript_counter,
                    script_step_index,
                )
            finished_iso = datetime.now(timezone.utc).isoformat()
            duration_seconds = round(perf_counter() - started_at, 3)
            self._step_transcript_counter += 1
            if verify_result is None:
                self._update_step_metadata(
                    transcript_counter,
                    script_step_index,
                    {
                        "started_at_utc": started_iso,
                        "finished_at_utc": finished_iso,
                        "duration_seconds": duration_seconds,
                        "status": "failed",
                        "step_index": script_step_index,
                        "goal": self._current_goal(),
                        "verify": None,
                    },
                )
                reason = (
                    f"Script step {script_step_index + 1} failed"
                    if not step_succeeded
                    else "Script step verification failed (parse or empty response)"
                )
                return BrainStepResult(
                    reason=reason,
                    step_finished=False,
                    step_index=script_step_index,
                )

            step_goal = self._current_goal()
            step_expected_outcome = self._current_expected_outcome() or None
            run_complete = self._apply_verify_branch(verify_result)
            if verify_result.accomplished:
                status = "completed"
            elif not step_succeeded:
                status = "failed"
            else:
                status = "verify_failed"
            self._update_step_metadata(
                transcript_counter,
                script_step_index,
                {
                    "started_at_utc": started_iso,
                    "finished_at_utc": finished_iso,
                    "duration_seconds": duration_seconds,
                    "status": status,
                    "step_index": script_step_index,
                    "goal": step_goal,
                    "expected_outcome": step_expected_outcome,
                    "verify": {
                        "accomplished": verify_result.accomplished,
                        "branch": verify_result.branch,
                        "target_step": verify_result.target_step,
                        "reason": verify_result.reason,
                    },
                },
            )
            return BrainStepResult(
                reason=f"Verify: {verify_result.reason}",
                step_finished=True,
                run_complete=run_complete,
                step_index=script_step_index,
            )
        finally:
            self.manager.clear_step_log_context()
