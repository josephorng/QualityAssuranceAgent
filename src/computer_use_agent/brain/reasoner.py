from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrainDecision:
    done: bool
    tool: str | None = None
    args: dict | None = None
    thought: str = ""


class BrainReasoner:
    def decide(
        self,
        task: str,
        screenshot_description: str,
        memory_context: str,
        cycle_index: int,
        max_cycles: int,
    ) -> BrainDecision:
        if cycle_index >= max_cycles - 1:
            return BrainDecision(
                done=True,
                thought="Stopping because max cycle limit reached.",
            )

        if "error" in screenshot_description.lower():
            return BrainDecision(
                done=False,
                tool="mock_ocr",
                args={"target": "active_window"},
                thought="Need more context from OCR before interaction.",
            )

        return BrainDecision(
            done=False,
            tool="type_text",
            args={"text": f"Working on task: {task[:40]}"},
            thought="Send a heartbeat action to keep task progressing in scaffold mode.",
        )
