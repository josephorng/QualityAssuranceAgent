from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from datetime import timezone

class ToolCommand(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    screenshot_name: str | None = None
    reason: str = ""


class ExecutionResult(BaseModel):
    ok: bool
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    screenshot_name: str | None = None
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
    reason: str = ""

    @model_validator(mode="after")
    def goto_requires_target_step(self) -> ScriptStepVerifyResult:
        if self.branch == "goto":
            if self.target_step is None:
                raise ValueError("target_step is required when branch is goto")
            if self.target_step < 1:
                raise ValueError("target_step must be >= 1 (1-based line number)")
        return self
