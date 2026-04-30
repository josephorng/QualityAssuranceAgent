from __future__ import annotations

import asyncio

from src.common.models import EyeEvent, ExecutionResult, ToolCommand
from src.runtime.coordinator import RuntimeCoordinator


class _FakeManager:
    def log_info(self, _message: str) -> None:
        return


class _FakeEye:
    def __init__(self) -> None:
        self.calls = 0

    async def capture_once(self) -> EyeEvent:
        self.calls += 1
        return EyeEvent(screenshot_name=f"{self.calls}.png", screenshot_path=f"/tmp/{self.calls}.png")


class _FakeBrain:
    def __init__(self) -> None:
        self.action_done_calls = 0
        self.process_calls = 0

    async def process_eye_event(self, _event: EyeEvent):
        self.process_calls += 1
        if self.process_calls == 1:
            return type("Cycle", (), {"finished": False, "commands": [ToolCommand(action="wait", args={"seconds": 0.0})], "request_capture": True})()
        return type("Cycle", (), {"finished": True, "commands": [], "request_capture": False})()

    async def on_action_done(self, _result: ExecutionResult) -> None:
        self.action_done_calls += 1


class _FakeHand:
    async def execute_tool_command(self, cmd: ToolCommand) -> ExecutionResult:
        return ExecutionResult(ok=True, action=cmd.action, args=cmd.args, message="ok")


def test_runtime_coordinator_basic_cycle() -> None:
    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.eye = _FakeEye()
    coordinator.brain = _FakeBrain()
    coordinator.hand = _FakeHand()
    coordinator.manager = _FakeManager()

    asyncio.run(coordinator.run())

    assert coordinator.eye.calls == 2
    assert coordinator.brain.action_done_calls == 1

