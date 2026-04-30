from __future__ import annotations

import asyncio

from src.brain.module import BrainModule
from src.common.run_state import get_run_state_manager
from src.eye.module import EyeModule
from src.hand.module import HandModule


class RuntimeCoordinator:
    def __init__(self) -> None:
        self.eye = EyeModule()
        self.hand = HandModule()
        self.brain = BrainModule(hand=self.hand)
        self.manager = get_run_state_manager()

    async def run(self) -> None:
        self.manager.log_info("Coordinator startup")
        while True:
            event = await self.eye.capture_once()
            step_result = await self.brain.process_step(event)
            if step_result.finished:
                self.manager.log_info("Coordinator detected finished task")
            else:
                break

            # if not cycle.request_capture and not cycle.commands:
            #     await asyncio.sleep(1.0)

