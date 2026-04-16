from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"


@dataclass
class RuntimeState:
    eye_status: WorkerStatus = WorkerStatus.IDLE
    brain_status: WorkerStatus = WorkerStatus.IDLE
    hand_status: WorkerStatus = WorkerStatus.IDLE
    latest_image_name: str | None = None
    interrupted_reasoning: bool = False
