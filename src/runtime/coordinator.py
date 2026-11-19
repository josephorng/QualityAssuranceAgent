from __future__ import annotations

import asyncio
from pathlib import Path

from src.brain.module import BrainModule
from src.common.io_utils import append_text, pop_last_nonempty_line
from src.common.run_control import (
    consume_pause_at_step,
    notify_paused_at_step,
    notify_step_status,
    pause_run,
    set_step_busy,
    take_pause_log,
    take_pending_step_jump,
    wait_while_paused,
)
from src.common.run_state import get_run_state_manager
from src.common.runtime_command_dialog import (
    prompt_runtime_command_popup,
    set_runtime_step_undo_handler,
)
from src.common.runtime_context import is_runtime_command_mode
from src.eye.module import EyeModule
from src.hand.module import HandModule

_RUNTIME_COMMAND_SCRIPT_NAME = "runtime_commands.txt"


def _runtime_command_script_path(run_root: Path) -> Path:
    return run_root / _RUNTIME_COMMAND_SCRIPT_NAME


class RuntimeCoordinator:
    def __init__(self) -> None:
        self.eye = EyeModule()
        self.hand = HandModule(eye=self.eye)
        self.brain = BrainModule(hand=self.hand, eye=self.eye)
        self.manager = get_run_state_manager()

    def _undo_last_step(self) -> bool:
        if not is_runtime_command_mode():
            return False
        if not self.brain.undo_last_runtime_step():
            return False
        root = self.manager.require_paths().root
        pop_last_nonempty_line(_runtime_command_script_path(root))
        return True

    async def run(self) -> None:
        self.manager.log_info("Coordinator startup")
        set_runtime_step_undo_handler(self._undo_last_step)
        try:
            await self._run_loop()
        except asyncio.CancelledError:
            self.manager.set_session_end_reason("user_stopped")
            raise
        finally:
            set_runtime_step_undo_handler(None)

    def _apply_pending_step_jump(self) -> None:
        """Apply a queued overlay/hub jump before the next process_step."""
        if is_runtime_command_mode():
            return
        jump_target = take_pending_step_jump()
        jump_to_step = getattr(self.brain, "jump_to_script_step", None)
        if jump_target is not None and callable(jump_to_step):
            jump_to_step(jump_target)

    async def _maybe_pause_at_upcoming_step(self) -> None:
        """One-shot pause before tools when pause-at-step matches the next index."""
        if is_runtime_command_mode():
            return
        next_index = getattr(self.brain, "_script_step_index", None)
        if not isinstance(next_index, int):
            return
        if not consume_pause_at_step(next_index):
            return
        pause_run()
        notify_paused_at_step(next_index)
        if take_pause_log():
            self.manager.log_info(
                f"Coordinator paused before step {next_index + 1}"
            )
        await wait_while_paused()
        # Re-record / overlay nav may have queued a jump while paused.
        self._apply_pending_step_jump()

    async def _run_loop(self) -> None:
        while True:
            if take_pause_log():
                self.manager.log_info("Coordinator paused")
            await wait_while_paused()
            self._apply_pending_step_jump()
            await self._maybe_pause_at_upcoming_step()
            if is_runtime_command_mode():
                cmd = prompt_runtime_command_popup()
                if cmd is None:
                    self.manager.log_info("Runtime mode: user ended run")
                    self.manager.set_session_end_reason("user_ended")
                    break
                run_root = self.manager.require_paths().root
                append_text(_runtime_command_script_path(run_root), cmd + "\n")
                self.brain.prepare_runtime_step(cmd)
            set_step_busy(True)
            try:
                step_result = await self.brain.process_step()
            finally:
                set_step_busy(False)
            if step_result.step_index is not None:
                holding_retry = (
                    step_result.step_finished
                    and not step_result.run_complete
                    and getattr(self.brain, "_script_step_index", None) == step_result.step_index
                )
                notify_step_status(
                    step_result.step_index,
                    "ok" if step_result.step_finished else "fail",
                    None
                    if step_result.step_finished and not holding_retry
                    else (step_result.reason or "Coordinator failed to process step"),
                    retry=holding_retry,
                )
            if not step_result.step_finished:
                self.manager.log_info(step_result.reason or "Coordinator failed to process step")
                self.manager.set_session_end_reason("step_failed")
                break
            if step_result.run_complete:
                if is_runtime_command_mode():
                    self.manager.log_info(step_result.reason or "Runtime step complete")
                    continue
                self.manager.log_info(step_result.reason or "All script steps complete")
                self.manager.set_session_end_reason("completed")
                break
            self.manager.log_info("Coordinator finished one step cycle")