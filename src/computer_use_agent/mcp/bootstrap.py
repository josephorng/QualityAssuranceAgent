from __future__ import annotations

from computer_use_agent.mcp.registry import ToolRegistry, ToolSpec
from computer_use_agent.mcp.tools.interaction import click, key_press, type_text
from computer_use_agent.mcp.tools.retrieval import get_running_programs, mock_ocr


def build_registry(dry_run: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(name="click", kind="interaction", fn=click))
    registry.register(ToolSpec(name="type_text", kind="interaction", fn=type_text))
    registry.register(ToolSpec(name="key_press", kind="interaction", fn=key_press))
    registry.register(
        ToolSpec(name="get_running_programs", kind="retrieval", fn=get_running_programs)
    )
    registry.register(ToolSpec(name="mock_ocr", kind="retrieval", fn=mock_ocr))

    if dry_run:
        for name in ("click", "type_text", "key_press"):
            registry.register(
                ToolSpec(
                    name=name,
                    kind="interaction",
                    fn=lambda args, tool=name: {
                        "ok": True,
                        "dry_run": True,
                        "tool": tool,
                        "args": args,
                    },
                )
            )
    return registry
