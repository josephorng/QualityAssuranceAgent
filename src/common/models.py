from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from datetime import timezone

ToolCommandAction = Literal[
    "click",
    "paste_text",
    "press_key",
    "hotkey",
    "move",
    "wait",
    "store_text",
    "store_image",
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "screenshot",
    "cursor_position",
    "left_mouse_down",
    "left_mouse_up",
    "scroll",
    "hold_key",
    "zoom",
]


class ToolCommand(BaseModel):
    action: ToolCommandAction
    args: dict[str, Any] = Field(default_factory=dict)
    screenshot_name: str | None = None
    reason: str = ""


class HandExecutionResult(BaseModel):
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
