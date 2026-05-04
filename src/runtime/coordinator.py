from __future__ import annotations

from src.brain.module import BrainModule
from src.common.run_state import get_run_state_manager
from src.eye.module import EyeModule
from src.hand.module import HandModule


class RuntimeCoordinator:
    def __init__(self) -> None:
        self.eye = EyeModule()
        self.hand = HandModule()
        self.brain = BrainModule(hand=self.hand, eye=self.eye)
        self.manager = get_run_state_manager()

    async def run(self) -> None:
        self.manager.log_info("Coordinator startup")
        while True:
            step_result = await self.brain.process_step()
            if step_result.finished:
                self.manager.log_info("Coordinator detected finished task")
                break
            self.manager.log_info(step_result.reason or "Coordinator moving to next script step")