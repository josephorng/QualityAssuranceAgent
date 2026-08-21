from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from datetime import timezone

class ToolCommand(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    screenshot_name: str | None = None
    screenshot_before_path: str | None = None
    screenshot_after_path: str | None = None
    reason: str = ""


class ExecutionResult(BaseModel):
    ok: bool
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    screenshot_name: str | None = None
    screenshot_before_path: str | None = None
    screenshot_after_path: str | None = None
    message: str = ""


class EyeEvent(BaseModel):
    screenshot_name: str
    screenshot_path: str
    similarity_to_previous: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InterruptionDecision(BaseModel):
    interruption: bool
    replace_state: bool = False
    reason: str = ""


class BrainDecision(BaseModel):
    command: ToolCommand
    needs_more_info: bool = False
    retrieval_request: str | None = None
    rationale: str = ""


class BrainStepOutcome(BaseModel):
    """Structured end-of-step reply when the decide loop issues no further tool calls."""

    status: Literal["completed", "failed"]
    reason: str = ""


class BrainTaskState(BaseModel):
    event: EyeEvent
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    thought: str = ""


class ScriptStepVerifyResult(BaseModel):
    """Vision LLM output for whether the current scripted step is satisfied, and how to branch."""

    accomplished: bool
    branch: Literal["advance", "retry", "skip", "goto"]
    target_step: int | None = Field(
        default=None,
        description="1-based line number in the script when branch is goto",
    )
    clearly_unmet: bool = Field(
        default=False,
        description=(
            "True only when ExpectedOutcome is visibly contradicted by the screenshot; "
            "false when uncertain or the outcome appears met"
        ),
    )
    reason: str = ""

    @model_validator(mode="after")
    def goto_requires_target_step(self) -> ScriptStepVerifyResult:
        if self.branch == "goto":
            if self.target_step is None:
                raise ValueError("target_step is required when branch is goto")
            if self.target_step < 1:
                raise ValueError("target_step must be >= 1 (1-based line number)")
        return self


class SmartPlannerDecision(BaseModel):
    """Planner output for one Plan→Act→Verify cycle in smart mode."""

    status: Literal["continue", "completed", "failed"]
    instruction: str | None = Field(
        default=None,
        description="One bounded sub-goal for the actor when status is continue",
    )
    expected_outcome: str = ""
    rationale: str = ""

    @model_validator(mode="after")
    def continue_requires_instruction(self) -> SmartPlannerDecision:
        if self.status == "continue":
            if not (self.instruction or "").strip():
                raise ValueError("instruction is required when status is continue")
        return self


class SmartVerifierDecision(BaseModel):
    """Verifier output after the actor finishes one smart-mode instruction."""

    outcome: Literal["succeeded", "failed"]
    updated_state: str = ""
    branch: Literal["advance", "retry", "replan", "backtrack", "stop"]
    reason: str = ""
    corrected_instruction: str | None = Field(
        default=None,
        description="Optional repaired instruction when branch is retry",
    )

    @model_validator(mode="after")
    def succeeded_defaults_to_advance(self) -> SmartVerifierDecision:
        if self.outcome == "succeeded" and self.branch not in ("advance", "stop"):
            # Treat success with an unexpected branch as advance for safety.
            object.__setattr__(self, "branch", "advance")
        return self


class SmartCheckpoint(BaseModel):
    """Logical checkpoint after a verified successful instruction (not a physical undo)."""

    cycle: int
    state_summary: str
    instruction: str
    created_at_utc: str = ""


class SmartRuntimeState(BaseModel):
    """Persisted smart-mode run state written to ``smart_state.json``."""

    goal: str
    current_state: str = ""
    cycle: int = 0
    recovery_attempts: int = 0
    checkpoints: list[SmartCheckpoint] = Field(default_factory=list)
    last_instruction: str | None = None
    last_expected_outcome: str = ""
    last_actor_ok: bool | None = None
    last_actor_reason: str = ""
    pending_instruction: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    terminal_reason: str | None = None
