"""Smart-mode Plan → Act → Verify coordinator."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cua_mcp.llm_json import parse_json_object
from cua_mcp.screen_context import ScreenContext, capture_screen_context
from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.tools import get_mode_tool_names, mcp_server
from src.brain.module import BrainModule, stamp_message
from src.common.io_utils import append_text, write_json
from src.common.models import (
    SmartCheckpoint,
    SmartPlannerDecision,
    SmartRuntimeState,
    SmartVerifierDecision,
)
from src.common.prompting import get_prompt
from src.common.run_control import take_pause_log, wait_while_paused
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import get_smart_goal
from src.common.settings import load_settings
from src.eye.module import EyeModule
from src.hand.module import HandModule

_SMART_STATE_NAME = "smart_state.json"
_SMART_EVENTS_NAME = "smart_events.jsonl"
_ROLE_USER = "user"

SMART_PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["continue", "completed", "failed"]},
        "instruction": {"type": ["string", "null"]},
        "expected_outcome": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["status", "instruction", "expected_outcome", "rationale"],
    "additionalProperties": False,
}

SMART_VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["succeeded", "failed"]},
        "updated_state": {"type": "string"},
        "branch": {
            "type": "string",
            "enum": ["advance", "retry", "replan", "backtrack", "stop"],
        },
        "reason": {"type": "string"},
        "corrected_instruction": {"type": ["string", "null"]},
    },
    "required": [
        "outcome",
        "updated_state",
        "branch",
        "reason",
        "corrected_instruction",
    ],
    "additionalProperties": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_history(history: list[dict[str, Any]], *, limit: int = 8) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for item in history[-limit:]:
        phase = item.get("phase", "?")
        summary = item.get("summary", "")
        lines.append(f"- [{phase}] {summary}")
    return "\n".join(lines)


def _format_available_tools(tools: list[Any]) -> str:
    """Format current MCP tool metadata for planner capability awareness."""
    if not tools:
        return "(no tools available)"
    lines: list[str] = []
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").strip()
        if not name:
            continue
        schema = getattr(tool, "inputSchema", None)
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        params: list[str] = []
        if isinstance(properties, dict):
            for param_name in properties:
                suffix = "" if param_name in required else "?"
                params.append(f"{param_name}{suffix}")
        signature = f"{name}({', '.join(params)})"
        description = " ".join(
            str(getattr(tool, "description", "") or "").split()
        )
        lines.append(f"- {signature}: {description}" if description else f"- {signature}")
    return "\n".join(lines) if lines else "(no tools available)"


def _parse_planner(content: str | None) -> SmartPlannerDecision:
    if not (content or "").strip():
        raise ValueError("Planner returned empty content")
    payload = parse_json_object(
        content or "",
        empty_error="Planner returned empty content",
        decode_error_prefix="Planner JSON decode failed",
    )
    return SmartPlannerDecision.model_validate(payload)


def _parse_verifier(content: str | None) -> SmartVerifierDecision:
    if not (content or "").strip():
        raise ValueError("Verifier returned empty content")
    payload = parse_json_object(
        content or "",
        empty_error="Verifier returned empty content",
        decode_error_prefix="Verifier JSON decode failed",
    )
    return SmartVerifierDecision.model_validate(payload)


class SmartCoordinator:
    """Bounded multimodal Plan → Act → Verify loop for 智能模式."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.eye = EyeModule()
        self.hand = HandModule(eye=self.eye)
        self.brain = BrainModule(hand=self.hand, eye=self.eye)
        self.manager = get_run_state_manager()
        self.max_cycles = max(1, int(self.settings.smart_max_cycles))
        self.max_recovery = max(1, int(self.settings.smart_max_recovery_attempts))
        goal = get_smart_goal()
        if not goal:
            raise RuntimeError("Smart mode requires CUA_SMART_GOAL")
        self.state = SmartRuntimeState(goal=goal, current_state="(initial)")

    def _state_path(self) -> Path:
        return self.manager.require_paths().root / _SMART_STATE_NAME

    def _events_path(self) -> Path:
        return self.manager.require_paths().root / _SMART_EVENTS_NAME

    def _persist_state(self) -> None:
        write_json(self._state_path(), self.state.model_dump(mode="json"))

    def _append_event(self, event: dict[str, Any]) -> None:
        payload = {"timestamp_utc": _utc_now(), **event}
        append_text(self._events_path(), json.dumps(payload, ensure_ascii=False) + "\n")
        self.state.history.append(
            {
                "phase": str(event.get("phase", "")),
                "summary": str(event.get("summary", event.get("reason", "")))[:500],
            }
        )
        self._persist_state()

    async def _wait_if_paused(self) -> None:
        if take_pause_log():
            self.manager.log_info("Smart coordinator paused")
        await wait_while_paused()

    async def _capture_context(self) -> ScreenContext:
        await self._wait_if_paused()
        return await capture_screen_context()

    async def _plan(
        self, context: ScreenContext
    ) -> SmartPlannerDecision:
        await self._wait_if_paused()
        mode_tool_names = get_mode_tool_names()
        available_tools = _format_available_tools(
            [
                tool
                for tool in await mcp_server.list_tools()
                if tool.name in mode_tool_names
            ]
        )
        prompt = get_prompt("brain_smart_plan").format(
            goal=self.state.goal,
            current_state=self.state.current_state or "(empty)",
            history=_format_history(self.state.history),
            available_tools=available_tools,
            ocr_text=context.ocr_text or "(none)",
        )
        messages = [
            stamp_message(
                {
                    "role": _ROLE_USER,
                    "content": prompt,
                    "images": list(context.screenshot_paths),
                }
            )
        ]

        def parse_reply(content: str | None) -> SmartPlannerDecision:
            return _parse_planner(content)

        try:
            decision = await request_json_with_retry(
                messages=messages,
                response_schema=SMART_PLANNER_SCHEMA,
                parse_reply=parse_reply,
                retry_instruction=(
                    "Previous reply was invalid. Return strict JSON only matching the schema."
                ),
                log_info=self.manager.log_info,
            )
        except (ValueError, ValidationError, TypeError) as exc:
            self.manager.log_error(f"Smart planner failed: {exc}")
            raise
        return decision

    async def _verify(
        self,
        *,
        context: ScreenContext,
        instruction: str,
        expected_outcome: str,
        actor_ok: bool,
        actor_reason: str,
    ) -> SmartVerifierDecision:
        await self._wait_if_paused()
        actor_result = json.dumps(
            {"ok": actor_ok, "reason": actor_reason},
            ensure_ascii=False,
        )
        prompt = get_prompt("brain_smart_verify").format(
            goal=self.state.goal,
            current_state=self.state.current_state or "(empty)",
            instruction=instruction,
            expected_outcome=expected_outcome or "(none)",
            actor_result=actor_result,
            ocr_text=context.ocr_text or "(none)",
        )
        messages = [
            stamp_message(
                {
                    "role": _ROLE_USER,
                    "content": prompt,
                    "images": list(context.screenshot_paths),
                }
            )
        ]

        def parse_reply(content: str | None) -> SmartVerifierDecision:
            return _parse_verifier(content)

        try:
            decision = await request_json_with_retry(
                messages=messages,
                response_schema=SMART_VERIFIER_SCHEMA,
                parse_reply=parse_reply,
                retry_instruction=(
                    "Previous reply was invalid. Return strict JSON only matching the schema."
                ),
                log_info=self.manager.log_info,
            )
        except (ValueError, ValidationError, TypeError) as exc:
            self.manager.log_error(f"Smart verifier failed: {exc}")
            raise
        return decision

    def _push_checkpoint(self, instruction: str) -> None:
        self.state.checkpoints.append(
            SmartCheckpoint(
                cycle=self.state.cycle,
                state_summary=self.state.current_state,
                instruction=instruction,
                created_at_utc=_utc_now(),
            )
        )

    def _backtrack(self) -> bool:
        if not self.state.checkpoints:
            return False
        checkpoint = self.state.checkpoints.pop()
        self.state.current_state = checkpoint.state_summary
        self.state.pending_instruction = None
        self.state.last_instruction = None
        return True

    async def run(self) -> None:
        self.manager.log_info("Smart coordinator startup")
        self._persist_state()
        try:
            await self._run_loop()
        except asyncio.CancelledError:
            self.state.terminal_reason = "user_stopped"
            self.manager.set_session_end_reason("user_stopped")
            raise
        finally:
            self._persist_state()

    async def _run_loop(self) -> None:
        force_instruction: str | None = None
        force_expected: str = ""

        while self.state.cycle < self.max_cycles:
            await self._wait_if_paused()
            self.state.cycle += 1
            cycle = self.state.cycle

            context = await self._capture_context()
            if force_instruction:
                instruction = force_instruction
                expected = force_expected
                force_instruction = None
                force_expected = ""
                plan = SmartPlannerDecision(
                    status="continue",
                    instruction=instruction,
                    expected_outcome=expected,
                    rationale="Forced retry/replan instruction from verifier",
                )
            else:
                try:
                    plan = await self._plan(context)
                except (ValueError, ValidationError, TypeError):
                    self.state.terminal_reason = "planner_failed"
                    self.manager.set_session_end_reason("step_failed")
                    self._append_event(
                        {
                            "phase": "plan",
                            "cycle": cycle,
                            "summary": "Planner parse/validation failed",
                            "screen": context.to_log_dict(),
                        }
                    )
                    return

            self._append_event(
                {
                    "phase": "plan",
                    "cycle": cycle,
                    "status": plan.status,
                    "instruction": plan.instruction,
                    "expected_outcome": plan.expected_outcome,
                    "rationale": plan.rationale,
                    "summary": plan.rationale or plan.status,
                    "screen": context.to_log_dict(),
                }
            )

            if plan.status == "completed":
                self.state.terminal_reason = "completed"
                self.manager.set_session_end_reason("completed")
                self.manager.log_info(plan.rationale or "Smart goal completed")
                return
            if plan.status == "failed":
                self.state.terminal_reason = "failed"
                self.manager.set_session_end_reason("step_failed")
                self.manager.log_info(plan.rationale or "Smart goal failed")
                return

            instruction = (plan.instruction or "").strip()
            self.state.last_instruction = instruction
            self.state.last_expected_outcome = plan.expected_outcome
            self.state.pending_instruction = instruction
            self._persist_state()

            await self._wait_if_paused()
            actor_ok = await self.brain.execute_instruction(instruction)
            actor_reason = (
                "Actor reported instruction completed"
                if actor_ok
                else "Actor reported instruction failed"
            )
            self.state.last_actor_ok = actor_ok
            self.state.last_actor_reason = actor_reason
            self._append_event(
                {
                    "phase": "act",
                    "cycle": cycle,
                    "instruction": instruction,
                    "ok": actor_ok,
                    "reason": actor_reason,
                    "summary": actor_reason,
                }
            )

            verify_context = await self._capture_context()
            try:
                verify = await self._verify(
                    context=verify_context,
                    instruction=instruction,
                    expected_outcome=plan.expected_outcome,
                    actor_ok=actor_ok,
                    actor_reason=actor_reason,
                )
            except (ValueError, ValidationError, TypeError):
                self.state.terminal_reason = "verifier_failed"
                self.manager.set_session_end_reason("step_failed")
                self._append_event(
                    {
                        "phase": "verify",
                        "cycle": cycle,
                        "summary": "Verifier parse/validation failed",
                        "screen": verify_context.to_log_dict(),
                    }
                )
                return

            self._append_event(
                {
                    "phase": "verify",
                    "cycle": cycle,
                    "outcome": verify.outcome,
                    "branch": verify.branch,
                    "updated_state": verify.updated_state,
                    "reason": verify.reason,
                    "corrected_instruction": verify.corrected_instruction,
                    "summary": verify.reason or verify.branch,
                    "screen": verify_context.to_log_dict(),
                }
            )

            if verify.outcome == "succeeded" or verify.branch == "advance":
                if verify.updated_state.strip():
                    self.state.current_state = verify.updated_state.strip()
                self._push_checkpoint(instruction)
                self.state.recovery_attempts = 0
                self.state.pending_instruction = None
                continue

            if verify.branch == "stop":
                self.state.terminal_reason = "verifier_stop"
                self.manager.set_session_end_reason("step_failed")
                return

            self.state.recovery_attempts += 1
            if self.state.recovery_attempts > self.max_recovery:
                self.state.terminal_reason = "recovery_budget_exhausted"
                self.manager.set_session_end_reason("step_failed")
                self.manager.log_info(
                    f"Smart recovery budget exhausted "
                    f"({self.state.recovery_attempts}/{self.max_recovery})"
                )
                return

            if verify.branch == "retry":
                corrected = (verify.corrected_instruction or instruction).strip()
                force_instruction = corrected
                force_expected = plan.expected_outcome
                continue

            if verify.branch == "replan":
                self.state.pending_instruction = None
                continue

            if verify.branch == "backtrack":
                if self._backtrack():
                    self._append_event(
                        {
                            "phase": "backtrack",
                            "cycle": cycle,
                            "summary": f"Restored checkpoint; state={self.state.current_state}",
                        }
                    )
                else:
                    self.manager.log_info("Backtrack requested but no checkpoints remain")
                continue

            # Unknown / unexpected branch after failed outcome → replan.
            self.state.pending_instruction = None

        self.state.terminal_reason = "cycle_budget_exhausted"
        self.manager.set_session_end_reason("step_failed")
        self.manager.log_info(
            f"Smart cycle budget exhausted ({self.state.cycle}/{self.max_cycles})"
        )
