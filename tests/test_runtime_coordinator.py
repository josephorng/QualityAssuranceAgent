from __future__ import annotations

import asyncio
import threading
import time

from src.brain.module import BrainStepResult
from src.common import run_control
from src.runtime.coordinator import RuntimeCoordinator


class _FakeManager:
    def __init__(self) -> None:
        self.session_end_reason: str | None = None
        self.logs: list[str] = []

    def log_info(self, message: str) -> None:
        self.logs.append(message)

    def set_session_end_reason(self, reason: str) -> None:
        self.session_end_reason = reason


class _FakeBrain:
    def __init__(self) -> None:
        self.process_step_calls = 0

    async def process_step(self) -> BrainStepResult:
        self.process_step_calls += 1
        if self.process_step_calls == 1:
            return BrainStepResult(reason="mid-script", step_finished=True, run_complete=False)
        return BrainStepResult(reason="All script steps complete", step_finished=True, run_complete=True)


def test_runtime_coordinator_basic_cycle() -> None:
    run_control.reset_run_control()
    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _FakeBrain()
    coordinator.manager = _FakeManager()

    asyncio.run(coordinator.run())

    assert coordinator.brain.process_step_calls == 2


def test_runtime_coordinator_pauses_between_steps() -> None:
    run_control.reset_run_control()

    class _SlowBrain:
        def __init__(self) -> None:
            self.process_step_calls = 0
            self.after_first_started = asyncio.Event()

        async def process_step(self) -> BrainStepResult:
            self.process_step_calls += 1
            if self.process_step_calls == 1:
                self.after_first_started.set()
                # Hold the first step so the test can pause before the next loop iteration.
                await asyncio.sleep(0.2)
                return BrainStepResult(reason="mid-script", step_finished=True, run_complete=False)
            return BrainStepResult(
                reason="All script steps complete",
                step_finished=True,
                run_complete=True,
            )

    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _SlowBrain()
    coordinator.manager = _FakeManager()

    async def _run_and_pause() -> None:
        run_task = asyncio.create_task(coordinator.run())
        await coordinator.brain.after_first_started.wait()
        run_control.pause_run()
        # First step finishes, then coordinator should block before step 2.
        await asyncio.sleep(0.35)
        assert coordinator.brain.process_step_calls == 1
        assert "Coordinator paused" in coordinator.manager.logs
        run_control.resume_run()
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_run_and_pause())
    assert coordinator.brain.process_step_calls == 2
    run_control.reset_run_control()


def test_run_control_pause_resume_reset() -> None:
    run_control.reset_run_control()
    assert not run_control.is_paused()
    run_control.pause_run()
    assert run_control.is_paused()
    assert run_control.take_pause_log() is True
    assert run_control.take_pause_log() is False
    run_control.resume_run()
    assert not run_control.is_paused()
    run_control.pause_run()
    run_control.reset_run_control()
    assert not run_control.is_paused()


def test_wait_while_paused_blocking_unblocks_on_resume() -> None:
    run_control.reset_run_control()
    run_control.pause_run()
    done = threading.Event()

    def _worker() -> None:
        run_control.wait_while_paused_blocking()
        done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    time.sleep(0.08)
    assert not done.is_set()
    run_control.resume_run()
    assert done.wait(timeout=2.0)
    t.join(timeout=1.0)
