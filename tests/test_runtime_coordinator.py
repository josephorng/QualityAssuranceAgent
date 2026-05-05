from __future__ import annotations

import asyncio

from src.brain.module import BrainStepResult
from src.runtime.coordinator import RuntimeCoordinator


class _FakeManager:
    def log_info(self, _message: str) -> None:
        return


class _FakeBrain:
    def __init__(self) -> None:
        self.process_step_calls = 0

    async def process_step(self) -> BrainStepResult:
        self.process_step_calls += 1
        if self.process_step_calls == 1:
            return BrainStepResult(reason="mid-script", step_finished=True, run_complete=False)
        return BrainStepResult(reason="All script steps complete", step_finished=True, run_complete=True)


def test_runtime_coordinator_basic_cycle() -> None:
    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _FakeBrain()
    coordinator.manager = _FakeManager()

    asyncio.run(coordinator.run())

    assert coordinator.brain.process_step_calls == 2

