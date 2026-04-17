from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPTool:
    name: str
    category: str  # interaction | retrieval
    description: str


INTERACTION_TOOLS = [
    MCPTool("click", "interaction", "Click at target coordinates."),
    MCPTool("type", "interaction", "Type text into focused input."),
    MCPTool("hotkey", "interaction", "Press a hotkey combination."),
    MCPTool("move", "interaction", "Move pointer to target coordinates."),
]

RETRIEVAL_TOOLS = [
    MCPTool("ocr", "retrieval", "Read text from screenshot."),
    MCPTool("program_list", "retrieval", "List current running programs."),
]


def list_tools() -> list[dict[str, Any]]:
    return [t.__dict__ for t in INTERACTION_TOOLS + RETRIEVAL_TOOLS]
