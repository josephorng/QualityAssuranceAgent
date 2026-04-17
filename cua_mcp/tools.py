from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

# 1. Initialize the MCP server
mcp = FastMCP("ComputerUseAgent")


# 2. Define tools
@mcp.tool()
def detect_objects(image_path: str) -> dict[str, Any]:
    """Run object detection on the given image path."""
    # Placeholder implementation until detector integration is added.
    return {"status": "success", "image_path": image_path, "detected": []}


@mcp.tool()
def read_screen_text(region: list[int]) -> str:
    """Run OCR on a screen region [x, y, w, h]."""
    if len(region) != 4:
        raise ValueError("region must be [x, y, w, h]")
    # Placeholder implementation until OCR integration is added.
    return ""


OLLAMA_TOOL_FUNCTIONS: list[Callable[..., Any]] = [
    detect_objects,
    read_screen_text,
]


def get_ollama_tools() -> list[Callable[..., Any]]:
    """Return all tool callables in Ollama chat() format."""
    return OLLAMA_TOOL_FUNCTIONS


def mcp_to_llm_tools(mcp_tools: Any) -> list[dict[str, Any]]:
    """Convert MCP tools metadata into LLM function-tool format."""
    tools: Iterable[Any] = getattr(mcp_tools, "tools", mcp_tools)
    llm_tools: list[dict[str, Any]] = []

    for tool in tools:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", "") or ""
        parameters = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {"type": "object", "properties": {}}
        )
        if not name:
            continue
        llm_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return llm_tools


if __name__ == "__main__":
    mcp.run(transport="stdio")
