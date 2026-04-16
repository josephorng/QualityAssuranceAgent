from __future__ import annotations

import csv
import json
from pathlib import Path

from computer_use_agent.core.session import utc_stamp
from computer_use_agent.mcp.registry import ToolRegistry


class HandExecutor:
    def __init__(self, registry: ToolRegistry, hand_csv_path: Path) -> None:
        self.registry = registry
        self.hand_csv_path = hand_csv_path

    def execute(self, image_name: str, tool_name: str, args: dict | None) -> dict:
        result = self.registry.run(tool_name, args)
        with self.hand_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "image_name", "tool", "arguments", "result"]
            )
            writer.writerow(
                {
                    "timestamp": utc_stamp(),
                    "image_name": image_name,
                    "tool": tool_name,
                    "arguments": json.dumps(args or {}, ensure_ascii=True),
                    "result": json.dumps(result, ensure_ascii=True),
                }
            )
        return result
