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
            return BrainStepResult(
                reason="mid-script",
                step_finished=True,
                run_complete=False,
                step_index=0,
            )
        return BrainStepResult(reason="All script steps complete", step_finished=True, run_complete=True)


def test_runtime_coordinator_basic_cycle() -> None:
    run_control.reset_run_control()
    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _FakeBrain()
    coordinator.manager = _FakeManager()

    asyncio.run(coordinator.run())

    assert coordinator.brain.process_step_calls == 2


def test_runtime_coordinator_notifies_step_status() -> None:
    run_control.reset_run_control()
    events: list[tuple[int, str]] = []
    run_control.set_step_status_callback(lambda index, status: events.append((index, status)))

    class _StatusBrain:
        def __init__(self) -> None:
            self.process_step_calls = 0

        async def process_step(self) -> BrainStepResult:
            self.process_step_calls += 1
            if self.process_step_calls == 1:
                return BrainStepResult(
                    reason="step ok",
                    step_finished=True,
                    run_complete=False,
                    step_index=0,
                )
            if self.process_step_calls == 2:
                return BrainStepResult(
                    reason="step failed",
                    step_finished=False,
                    step_index=1,
                )
            return BrainStepResult(reason="unused", step_finished=True, run_complete=True)

    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _StatusBrain()
    coordinator.manager = _FakeManager()

    asyncio.run(coordinator.run())

    assert events == [(0, "ok"), (1, "fail")]
    assert coordinator.manager.session_end_reason == "step_failed"
    run_control.reset_run_control()


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
                return BrainStepResult(
                    reason="mid-script",
                    step_finished=True,
                    run_complete=False,
                    step_index=0,
                )
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


def test_run_control_step_status_callback_reset() -> None:
    run_control.reset_run_control()
    events: list[tuple[int, str]] = []
    run_control.set_step_status_callback(lambda index, status: events.append((index, status)))
    run_control.notify_step_status(2, "ok")
    assert events == [(2, "ok")]
    run_control.reset_run_control()
    run_control.notify_step_status(3, "fail")
    assert events == [(2, "ok")]


def test_run_control_step_error_persists_across_retry_until_success() -> None:
    run_control.reset_run_control()
    events: list[tuple[int, str]] = []
    run_control.set_step_status_callback(lambda index, status: events.append((index, status)))
    run_control.notify_step_status(1, "ok", reason="  window missing  ", retry=True)
    assert events == [(1, "ok")]
    assert run_control.step_error(1) == "window missing"
    run_control.notify_active_step(1, instruction="open the dialog")
    assert run_control.step_error(1) == "window missing"
    run_control.notify_step_status(1, "ok", reason="Verify: still missing", retry=True)
    assert run_control.step_error(1) == "Verify: still missing"
    run_control.notify_step_status(1, "ok")
    assert run_control.step_error(1) is None
    run_control.notify_step_status(1, "fail", reason="  window missing  ")
    assert run_control.step_error(1) == "window missing"
    run_control.notify_step_status(1, "fail", reason="   ")
    assert run_control.step_error(1) == "window missing"
    run_control.notify_step_status(2, "fail", reason="tool failed")
    assert run_control.step_errors() == {1: "window missing", 2: "tool failed"}
    run_control.reset_run_control()
    assert run_control.step_errors() == {}
    assert events == [(1, "ok"), (1, "ok"), (1, "ok"), (1, "fail"), (1, "fail"), (2, "fail")]


def test_run_control_step_progress_callback() -> None:
    run_control.reset_run_control()
    phases: list[str] = []
    run_control.notify_step_progress("選取 · FindAll")
    assert run_control.step_progress_phase() == "選取 · FindAll"
    assert phases == []
    run_control.set_step_progress_callback(phases.append)
    run_control.notify_step_progress("選取 · OCR")
    assert run_control.step_progress_phase() == "選取 · OCR"
    assert phases == ["選取 · OCR"]
    run_control.notify_step_progress("  ")
    assert run_control.step_progress_phase() == ""
    assert phases == ["選取 · OCR", ""]
    run_control.notify_active_step(0)
    run_control.notify_step_progress("執行中")
    run_control.notify_active_step(None)
    assert run_control.step_progress_phase() == ""
    run_control.reset_run_control()
    run_control.notify_step_progress("驗證中")
    assert phases == ["選取 · OCR", "", "執行中"]
    assert run_control.step_progress_phase() == "驗證中"
    run_control.reset_run_control()
    assert run_control.step_progress_phase() == ""


def test_run_control_active_step_reset() -> None:
    run_control.reset_run_control()
    seen: list[int | None] = []
    run_control.notify_active_step(2)
    assert run_control.active_step_index() == 2
    assert seen == []
    run_control.set_active_step_callback(seen.append)
    run_control.notify_active_step(4, instruction="click save", event_index=7)
    assert run_control.active_step_index() == 4
    assert run_control.active_step_instruction() == "click save"
    assert run_control.active_step_event_index() == 7
    assert seen == [4]
    run_control.notify_active_step(None)
    assert run_control.active_step_index() is None
    assert run_control.active_step_instruction() is None
    assert run_control.active_step_event_index() is None
    assert seen == [4, None]
    run_control.reset_run_control()
    assert run_control.active_step_index() is None
    run_control.notify_active_step(1)
    assert seen == [4, None]
    assert run_control.active_step_index() == 1
    run_control.request_step_jump(3)
    assert run_control.peek_pending_step_jump() == 3
    assert run_control.take_pending_step_jump() == 3
    assert run_control.take_pending_step_jump() is None
    run_control.request_step_jump(1)
    run_control.reset_run_control()
    assert run_control.peek_pending_step_jump() is None


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


def test_run_control_pause_at_step_consume_and_reset() -> None:
    run_control.reset_run_control()
    seen: list[int] = []
    run_control.set_pause_at_step_callback(seen.append)
    run_control.set_pause_at_step(2)
    assert run_control.peek_pause_at_step() == 2
    assert run_control.consume_pause_at_step(1) is False
    assert run_control.peek_pause_at_step() == 2
    assert run_control.consume_pause_at_step(2) is True
    assert run_control.peek_pause_at_step() is None
    assert run_control.consume_pause_at_step(2) is False
    run_control.set_pause_at_step(0)
    run_control.notify_paused_at_step(0)
    assert seen == [0]
    run_control.reset_run_control()
    assert run_control.peek_pause_at_step() is None
    run_control.notify_paused_at_step(1)
    assert seen == [0]


def test_runtime_coordinator_pauses_before_pause_at_step() -> None:
    run_control.reset_run_control()
    paused_at: list[int] = []
    run_control.set_pause_at_step_callback(paused_at.append)
    run_control.set_pause_at_step(1)

    class _IndexedBrain:
        def __init__(self) -> None:
            self._script_step_index = 0
            self.process_step_calls = 0
            self.jumps: list[int] = []

        def jump_to_script_step(self, index: int) -> None:
            self.jumps.append(index)
            self._script_step_index = index

        async def process_step(self) -> BrainStepResult:
            self.process_step_calls += 1
            current = self._script_step_index
            self._script_step_index = current + 1
            if self.process_step_calls == 1:
                return BrainStepResult(
                    reason="step 1 ok",
                    step_finished=True,
                    run_complete=False,
                    step_index=current,
                )
            return BrainStepResult(
                reason="All script steps complete",
                step_finished=True,
                run_complete=True,
                step_index=current,
            )

    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _IndexedBrain()
    coordinator.manager = _FakeManager()

    async def _run_with_breakpoint() -> None:
        run_task = asyncio.create_task(coordinator.run())
        # First step should run; then pause before step index 1.
        for _ in range(40):
            if paused_at:
                break
            await asyncio.sleep(0.05)
        assert paused_at == [1]
        assert coordinator.brain.process_step_calls == 1
        assert run_control.is_paused()
        assert any(
            "paused before step 2" in message for message in coordinator.manager.logs
        )
        run_control.resume_run()
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_run_with_breakpoint())
    assert coordinator.brain.process_step_calls == 2
    assert run_control.peek_pause_at_step() is None
    run_control.reset_run_control()


def test_runtime_coordinator_jump_then_pause_skips_earlier_steps() -> None:
    """Start-at + pause-at the same index must not execute earlier steps."""
    run_control.reset_run_control()
    paused_at: list[int] = []
    run_control.set_pause_at_step_callback(paused_at.append)
    run_control.request_step_jump(2)
    run_control.set_pause_at_step(2)

    class _JumpBrain:
        def __init__(self) -> None:
            self._script_step_index = 0
            self.process_step_calls = 0
            self.processed: list[int] = []

        def jump_to_script_step(self, index: int) -> None:
            self._script_step_index = index

        async def process_step(self) -> BrainStepResult:
            self.process_step_calls += 1
            current = self._script_step_index
            self.processed.append(current)
            self._script_step_index = current + 1
            return BrainStepResult(
                reason="done",
                step_finished=True,
                run_complete=True,
                step_index=current,
            )

    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _JumpBrain()
    coordinator.manager = _FakeManager()

    async def _run() -> None:
        run_task = asyncio.create_task(coordinator.run())
        for _ in range(40):
            if paused_at:
                break
            await asyncio.sleep(0.05)
        assert paused_at == [2]
        assert coordinator.brain.process_step_calls == 0
        assert coordinator.brain._script_step_index == 2
        run_control.resume_run()
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_run())
    assert coordinator.brain.processed == [2]
    assert run_control.peek_pause_at_step() is None
    run_control.reset_run_control()


def test_runtime_coordinator_pause_at_step_zero_before_first_process() -> None:
    run_control.reset_run_control()
    paused_at: list[int] = []
    run_control.set_pause_at_step_callback(paused_at.append)
    run_control.set_pause_at_step(0)

    class _FirstStepBrain:
        def __init__(self) -> None:
            self._script_step_index = 0
            self.process_step_calls = 0

        async def process_step(self) -> BrainStepResult:
            self.process_step_calls += 1
            return BrainStepResult(
                reason="done",
                step_finished=True,
                run_complete=True,
                step_index=0,
            )

    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.brain = _FirstStepBrain()
    coordinator.manager = _FakeManager()

    async def _run() -> None:
        run_task = asyncio.create_task(coordinator.run())
        for _ in range(40):
            if paused_at:
                break
            await asyncio.sleep(0.05)
        assert paused_at == [0]
        assert coordinator.brain.process_step_calls == 0
        run_control.resume_run()
        await asyncio.wait_for(run_task, timeout=2.0)

    asyncio.run(_run())
    assert coordinator.brain.process_step_calls == 1
    run_control.reset_run_control()
