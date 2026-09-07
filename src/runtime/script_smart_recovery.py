"""Bounded Plan→Act→Verify recovery used when script verify returns branch=smart."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from cua_mcp.llm_json import parse_json_object
from cua_mcp.screen_context import ScreenContext, capture_screen_context
from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.tools import get_mode_tool_names, mcp_server
from src.brain.module import stamp_message
from src.common.models import SmartPlannerDecision, SmartVerifierDecision
from src.common.prompting import get_prompt
from src.runtime.smart_coordinator import (
    SMART_PLANNER_SCHEMA,
    SMART_VERIFIER_SCHEMA,
    _format_available_tools,
    _format_history,
)

if TYPE_CHECKING:
    from src.brain.module import BrainModule

_ROLE_USER = "user"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_planner(content: str | None) -> SmartPlannerDecision:
    if not (content or "").strip():
        raise ValueError("Recovery planner returned empty content")
    payload = parse_json_object(
        content or "",
        empty_error="Recovery planner returned empty content",
        decode_error_prefix="Recovery planner JSON decode failed",
    )
    return SmartPlannerDecision.model_validate(payload)


def _parse_verifier(content: str | None) -> SmartVerifierDecision:
    if not (content or "").strip():
        raise ValueError("Recovery verifier returned empty content")
    payload = parse_json_object(
        content or "",
        empty_error="Recovery verifier returned empty content",
        decode_error_prefix="Recovery verifier JSON decode failed",
    )
    return SmartVerifierDecision.model_validate(payload)


def build_recovery_goal(*, script_goal: str, verify_reason: str) -> str:
    goal = (script_goal or "").strip() or "(unknown script step)"
    reason = (verify_reason or "").strip() or "(no verifier reason)"
    return (
        "Clear any blocking popup, dialog, or unexpected overlay that is not part of "
        f"the recorded script, then leave the UI ready to complete this scripted step: "
        f"{goal}. Verifier noted: {reason}. "
        "Do not advance the recorded script yourself; only remove blockers."
    )


@dataclass
class ScriptSmartRecoveryResult:
    ok: bool
    reason: str
    events: list[dict[str, Any]] = field(default_factory=list)


async def run_script_smart_recovery(
    *,
    brain: BrainModule,
    script_goal: str,
    verify_reason: str,
    max_cycles: int,
) -> ScriptSmartRecoveryResult:
    """Run a short smart Plan→Act→Verify episode to clear off-script blockers."""
    max_cycles = max(1, int(max_cycles))
    recovery_goal = build_recovery_goal(
        script_goal=script_goal,
        verify_reason=verify_reason,
    )
    current_state = "Scripted step blocked; starting smart recovery"
    history: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    force_instruction: str | None = None
    force_expected = ""

    def _append(event: dict[str, Any]) -> None:
        payload = {"timestamp_utc": _utc_now(), **event}
        events.append(payload)
        history.append(
            {
                "phase": str(event.get("phase", "")),
                "summary": str(event.get("summary", event.get("reason", "")))[:500],
            }
        )

    brain.manager.log_info(
        f"Script smart recovery start cycles={max_cycles} goal={recovery_goal[:160]}"
    )

    for cycle in range(1, max_cycles + 1):
        context = await capture_screen_context()
        if force_instruction:
            instruction = force_instruction
            expected = force_expected
            force_instruction = None
            force_expected = ""
            plan = SmartPlannerDecision(
                status="continue",
                instruction=instruction,
                expected_outcome=expected,
                rationale="Forced retry/replan instruction from recovery verifier",
            )
        else:
            try:
                plan = await _plan(
                    brain=brain,
                    goal=recovery_goal,
                    current_state=current_state,
                    history=history,
                    context=context,
                )
            except (ValueError, ValidationError, TypeError) as exc:
                reason = f"Recovery planner failed: {exc}"
                brain.manager.log_error(reason)
                _append(
                    {
                        "phase": "plan",
                        "cycle": cycle,
                        "summary": reason,
                        "screen": context.to_log_dict(),
                    }
                )
                return ScriptSmartRecoveryResult(ok=False, reason=reason, events=events)

        _append(
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
            reason = plan.rationale or "Recovery goal completed"
            brain.manager.log_info(f"Script smart recovery succeeded: {reason}")
            return ScriptSmartRecoveryResult(ok=True, reason=reason, events=events)
        if plan.status == "failed":
            reason = plan.rationale or "Recovery goal failed"
            brain.manager.log_info(f"Script smart recovery failed: {reason}")
            return ScriptSmartRecoveryResult(ok=False, reason=reason, events=events)

        instruction = (plan.instruction or "").strip()
        if not instruction:
            reason = "Recovery planner returned empty instruction"
            _append({"phase": "plan", "cycle": cycle, "summary": reason})
            return ScriptSmartRecoveryResult(ok=False, reason=reason, events=events)

        actor_ok = await brain.execute_instruction(instruction)
        actor_reason = (
            "Actor reported instruction completed"
            if actor_ok
            else "Actor reported instruction failed"
        )
        _append(
            {
                "phase": "act",
                "cycle": cycle,
                "instruction": instruction,
                "ok": actor_ok,
                "reason": actor_reason,
                "summary": actor_reason,
            }
        )

        context = await capture_screen_context()
        try:
            verify = await _verify(
                brain=brain,
                goal=recovery_goal,
                current_state=current_state,
                instruction=instruction,
                expected_outcome=plan.expected_outcome,
                actor_ok=actor_ok,
                actor_reason=actor_reason,
                context=context,
            )
        except (ValueError, ValidationError, TypeError) as exc:
            reason = f"Recovery verifier failed: {exc}"
            brain.manager.log_error(reason)
            _append(
                {
                    "phase": "verify",
                    "cycle": cycle,
                    "summary": reason,
                    "screen": context.to_log_dict(),
                }
            )
            return ScriptSmartRecoveryResult(ok=False, reason=reason, events=events)

        _append(
            {
                "phase": "verify",
                "cycle": cycle,
                "outcome": verify.outcome,
                "branch": verify.branch,
                "updated_state": verify.updated_state,
                "reason": verify.reason,
                "corrected_instruction": verify.corrected_instruction,
                "summary": verify.reason or verify.branch,
                "screen": context.to_log_dict(),
            }
        )

        if verify.updated_state.strip():
            current_state = verify.updated_state.strip()

        if verify.outcome == "succeeded" or verify.branch == "advance":
            reason = verify.reason or "Blocker cleared; ready to retry script step"
            brain.manager.log_info(f"Script smart recovery succeeded: {reason}")
            return ScriptSmartRecoveryResult(ok=True, reason=reason, events=events)

        if verify.branch == "stop":
            reason = verify.reason or "Recovery verifier stopped"
            brain.manager.log_info(f"Script smart recovery failed: {reason}")
            return ScriptSmartRecoveryResult(ok=False, reason=reason, events=events)

        if verify.branch == "retry":
            corrected = (verify.corrected_instruction or instruction).strip()
            force_instruction = corrected
            force_expected = plan.expected_outcome
            continue

        # replan / backtrack / unknown → ask planner again next cycle
        continue

    reason = f"Script smart recovery exhausted after {max_cycles} cycle(s)"
    brain.manager.log_info(reason)
    return ScriptSmartRecoveryResult(ok=False, reason=reason, events=events)


async def _plan(
    *,
    brain: BrainModule,
    goal: str,
    current_state: str,
    history: list[dict[str, Any]],
    context: ScreenContext,
) -> SmartPlannerDecision:
    mode_tool_names = get_mode_tool_names()
    available_tools = _format_available_tools(
        [
            tool
            for tool in await mcp_server.list_tools()
            if tool.name in mode_tool_names
        ]
    )
    prompt = get_prompt("brain_smart_plan").format(
        goal=goal,
        current_state=current_state or "(empty)",
        history=_format_history(history),
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
    return await request_json_with_retry(
        messages=messages,
        response_schema=SMART_PLANNER_SCHEMA,
        parse_reply=_parse_planner,
        retry_instruction=(
            "Previous reply was invalid. Return strict JSON only matching the schema."
        ),
        log_info=brain.manager.log_info,
    )


async def _verify(
    *,
    brain: BrainModule,
    goal: str,
    current_state: str,
    instruction: str,
    expected_outcome: str,
    actor_ok: bool,
    actor_reason: str,
    context: ScreenContext,
) -> SmartVerifierDecision:
    actor_result = json.dumps(
        {"ok": actor_ok, "reason": actor_reason},
        ensure_ascii=False,
    )
    prompt = get_prompt("brain_smart_verify").format(
        goal=goal,
        current_state=current_state or "(empty)",
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
    return await request_json_with_retry(
        messages=messages,
        response_schema=SMART_VERIFIER_SCHEMA,
        parse_reply=_parse_verifier,
        retry_instruction=(
            "Previous reply was invalid. Return strict JSON only matching the schema."
        ),
        log_info=brain.manager.log_info,
    )
