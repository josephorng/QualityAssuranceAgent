from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


ToolFn = Callable[[dict], dict]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str  # interaction | retrieval
    fn: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def kind(self, tool_name: str) -> str:
        return self._tools[tool_name].kind

    def run(self, tool_name: str, args: dict | None) -> dict:
        payload = args or {}
        return self._tools[tool_name].fn(payload)
