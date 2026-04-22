from __future__ import annotations

from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from cua_mcp import hand_tools
from cua_mcp.read_screen_text.ocr_image import read_text_from_image_path
from cua_mcp.storage import store_image as _store_image
from cua_mcp.storage import store_text as _store_text

# 1. Initialize the MCP server
mcp = FastMCP("ComputerUseAgent")


# 2. Define tools
@mcp.tool()
def detect_objects(image_path: str) -> dict[str, Any]:
    """Run object detection on the given image path."""
    return hand_tools.detect_objects(image_path)


@mcp.tool()
def get_coordinates(image_path: str) -> str:
    """Run Yolo and OCR to get the coordinates and the contents of the detected objects in the given image path."""
    return read_text_from_image_path(image_path)


@mcp.tool()
def click(x: int, y: int, button: str = "left") -> dict[str, Any]:
    """Click a screen coordinate."""
    return hand_tools.click(x=x, y=y, button=button)


@mcp.tool()
def type_text(text: str, coordinate: list[int], interval: float = 0.0) -> dict[str, Any]:
    """Click a coordinate to focus, then type text with an optional key interval."""
    return hand_tools.type_text(text=text, coordinate=coordinate, interval=interval)


@mcp.tool()
def hotkey(keys: list[str] | str) -> dict[str, Any]:
    """Press a key combination."""
    return hand_tools.hotkey(keys=keys)


@mcp.tool()
def move(x: int, y: int, duration: float = 0.0) -> dict[str, Any]:
    """Move mouse to a screen coordinate."""
    return hand_tools.move(x=x, y=y, duration=duration)


@mcp.tool()
def wait(seconds: float) -> dict[str, Any]:
    """Pause execution for the specified number of seconds."""
    return hand_tools.wait(seconds=seconds)


@mcp.tool()
def store_text(
    text: str,
    title: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Store text context to this run's storage.json index."""
    return _store_text(text=text, title=title, tags=tags)


@mcp.tool()
def store_image(
    image_path: str,
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Copy an image to this run's storage folder and index it."""
    return _store_image(image_path=image_path, summary=summary, alias=alias, tags=tags)


OLLAMA_TOOL_FUNCTIONS: list[Callable[..., Any]] = [
    detect_objects,
    get_coordinates,
    store_text,
    store_image,
    click,
    type_text,
    hotkey,
    move,
    wait,
]

HAND_TOOL_NAMES: set[str] = {"click", "type_text", "hotkey", "move", "wait"}


def get_ollama_tools() -> list[Callable[..., Any]]:
    """Return all tool callables in Ollama chat() format."""
    return OLLAMA_TOOL_FUNCTIONS


def get_tool_function_map() -> dict[str, Callable[..., Any]]:
    """Return a name -> callable map for all registered tools."""
    return {tool.__name__: tool for tool in OLLAMA_TOOL_FUNCTIONS}


def execute_tool_call(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Execute a local MCP tool callable by name."""
    function_map = get_tool_function_map()
    function = function_map.get(tool_name)
    if function is None:
        raise ValueError(f"unknown tool: {tool_name}")
    return function(**arguments)


# def get_available_actions_text() -> str:
#     """Return the action list text used in the brain prompt."""
#     action_lines: list[str] = []
#     for tool in mcp_to_llm_tools(OLLAMA_TOOL_FUNCTIONS):
#         function = tool.get("function", {})
#         name = str(function.get("name", ""))
#         if name not in ACTION_TOOL_NAMES:
#             continue
#         parameters = function.get("parameters", {})
#         properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
#         required = set(parameters.get("required", [])) if isinstance(parameters, dict) else set()
#         args: list[str] = []
#         for prop_name, schema in properties.items():
#             if not isinstance(schema, dict):
#                 continue
#             json_type = str(schema.get("type", "any"))
#             is_required = prop_name in required
#             suffix = "" if is_required else "?"
#             if json_type == "array":
#                 item_type = "any"
#                 items = schema.get("items")
#                 if isinstance(items, dict):
#                     item_type = str(items.get("type", "any"))
#                 json_type = f"[{item_type},...]"
#             args.append(f"{prop_name}{suffix}:{json_type}")
#         action_lines.append(f"- {name}: {{{','.join(args)}}}")
#     return "\n".join(action_lines)


# def mcp_to_llm_tools(mcp_tools: Any) -> list[dict[str, Any]]:
#     """Convert MCP tools metadata into LLM function-tool format."""
#     tools: Iterable[Any] = getattr(mcp_tools, "tools", mcp_tools)
#     llm_tools: list[dict[str, Any]] = []

#     for tool in tools:
#         name = getattr(tool, "name", None)
#         description = getattr(tool, "description", "") or ""
#         parameters = (
#             getattr(tool, "inputSchema", None)
#             or getattr(tool, "input_schema", None)
#             or {"type": "object", "properties": {}}
#         )
#         if not name:
#             continue
#         llm_tools.append(
#             {
#                 "type": "function",
#                 "function": {
#                     "name": name,
#                     "description": description,
#                     "parameters": parameters,
#                 },
#             }
#         )
#     return llm_tools


if __name__ == "__main__":
    mcp.run(transport="stdio")
