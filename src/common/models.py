from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCommand(BaseModel):
    action: Literal["click", "type_text", "hotkey", "move", "wait"]
    args: dict[str, Any] = Field(default_factory=dict)
    screenshot_name: str | None = None
    reason: str = ""


class HandExecutionResult(BaseModel):
    ok: bool
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    screenshot_name: str | None = None
    message: str = ""


class EyeEvent(BaseModel):
    screenshot_name: str
    screenshot_path: str
    similarity_to_previous: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    started_at: datetime = Field(default_factory=datetime.utcnow)
    thought: str = ""
