from __future__ import annotations

from pathlib import Path

from src.brain.module import BrainModule
from src.common.io_utils import append_text, pop_last_nonempty_line
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
        self.hand = HandModule()
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
        finally:
            set_runtime_step_undo_handler(None)

    async def _run_loop(self) -> None:
        while True:
            if is_runtime_command_mode():
                cmd = prompt_runtime_command_popup()
                if cmd is None:
                    self.manager.log_info("Runtime mode: user ended run")
                    break
                run_root = self.manager.require_paths().root
                append_text(_runtime_command_script_path(run_root), cmd + "\n")
                self.brain.prepare_runtime_step(cmd)
            step_result = await self.brain.process_step()
            if not step_result.step_finished:
                self.manager.log_info(step_result.reason or "Coordinator failed to process step")
                break
            if step_result.run_complete:
                if is_runtime_command_mode():
                    self.manager.log_info(step_result.reason or "Runtime step complete")
                    continue
                self.manager.log_info(step_result.reason or "All script steps complete")
                break
            self.manager.log_info("Coordinator finished one step cycle")