from __future__ import annotations

import json
from typing import Any, Callable

import httpx
from mcp.server.fastmcp import FastMCP

from cua_mcp import hand_tools
from cua_mcp import steps as step_tools
from cua_mcp.read_screen_text.ocr_image import get_coordinates
from cua_mcp.storage import store_image as _store_image
from cua_mcp.storage import store_text as _store_text
from src.common.run_state import get_run_state_manager
from src.common.settings import load_settings

# 1. Initialize the MCP server
mcp = FastMCP("ComputerUseAgent")
settings = load_settings()
logger = get_run_state_manager()


def _get_active_capture_offset() -> tuple[int, int]:
    """Return (left, top) offset for the active Eye capture target."""
    url = f"http://127.0.0.1:{settings.eye_port}/capture_targets"
    try:
        response = httpx.get(url, timeout=2)
        response.raise_for_status()
        payload = response.json()
        active_index = int(payload.get("active_monitor_index", 0))
        monitors = payload.get("monitors", [])
        if not isinstance(monitors, list):
            return 0, 0
        for monitor in monitors:
            if not isinstance(monitor, dict):
                continue
            if int(monitor.get("index", -1)) == active_index:
                left = int(monitor.get("left", 0))
                top = int(monitor.get("top", 0))
                return left, top
    except Exception as exc:
        logger.log_info(
            f"Failed to fetch active capture offset url={url} err={type(exc).__name__}: {exc}"
        )
    return 0, 0


def _to_global_coordinate(local_x: int, local_y: int) -> tuple[int, int]:
    """Convert screenshot-local coordinates to desktop-global coordinates."""
    offset_left, offset_top = _get_active_capture_offset()
    global_x = local_x + offset_left
    global_y = local_y + offset_top
    logger.log_info(
        f"Coordinate converted local=({local_x},{local_y}) "
        f"offset=({offset_left},{offset_top}) global=({global_x},{global_y})"
    )
    return global_x, global_y


def _select_coordinate(instruction: str, coordinate_text: str) -> tuple[int, int]:
    """Select one [x, y] target using instruction and detected coordinate text."""
    logger.log_info(
        f"Selecting coordinate from OCR text (instruction_len={len(instruction)}, coordinate_text_len={len(coordinate_text)})"
    )
    prompt = (
        "Choose one target coordinate based on user instruction and screen coordinates text.\n"
        "Return JSON only in this exact shape: {\"x\": <int>, \"y\": <int>}.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"CoordinatesText:\n{coordinate_text}\n"
    )
    response = httpx.post(
        f"{settings.ollama_host.rstrip('/')}/api/chat",
        json={
            "model": settings.brain_lm,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": 4096},
        },
        timeout=30,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "").strip()
    try:
        out = json.loads(content)
        x, y = int(out["x"]), int(out["y"])
        logger.log_info(f"Selected coordinate x={x} y={y}")
        return x, y
    except Exception as exc:
        logger.log_info(f"Failed to parse coordinate selection response: {content}")
        raise ValueError(f"failed to parse coordinate selection: {content}") from exc


@mcp.tool()
def click(
    instruction: str,
    image_path: str,
    button: str = "left",
) -> dict[str, Any]:
    """Use instruction and Yolo model and OCR to find the target on the screenshot. Then click the target."""
    logger.log_info(f"Tool click start (button={button}), image_path={image_path}")
    coordinate_text = get_coordinates(image_path)
    local_x, local_y = _select_coordinate(instruction=instruction, coordinate_text=coordinate_text)
    x, y = _to_global_coordinate(local_x, local_y)
    result = hand_tools.click(x=x, y=y, button=button)
    logger.log_info("Tool click done")
    return result


@mcp.tool()
def type_text(
    text: str,
    instruction: str,
    image_path: str,
    interval: float = 0.0,
) -> dict[str, Any]:
    """Use instruction and Yolo model and OCR to find the target on the screenshot. Click to focus on the target and then type text into the target."""
    logger.log_info(
        f"Tool type_text start (text_len={len(text)}, interval={interval}), image_path={image_path}"
    )
    coordinate_text = get_coordinates(image_path)
    local_x, local_y = _select_coordinate(instruction=instruction, coordinate_text=coordinate_text)
    x, y = _to_global_coordinate(local_x, local_y)
    coordinate = [x, y]
    result = hand_tools.type_text(text=text, coordinate=coordinate, interval=interval)
    logger.log_info("Tool type_text done")
    return result


@mcp.tool()
def hotkey(keys: list[str] | str, instruction: str = "") -> dict[str, Any]:
    """Press a key combination."""
    logger.log_info("Tool hotkey start")
    result = hand_tools.hotkey(keys=keys)
    logger.log_info("Tool hotkey done")
    return result


@mcp.tool()
def move(
    instruction: str,
    image_path: str,
    duration: float = 0.0,
) -> dict[str, Any]:
    """Use instruction and Yolo model and OCR to find the target on the screenshot. Move mouse to the target."""
    logger.log_info(f"Tool move start (duration={duration}), image_path={image_path}")
    coordinate_text = get_coordinates(image_path)
    local_x, local_y = _select_coordinate(instruction=instruction, coordinate_text=coordinate_text)
    x, y = _to_global_coordinate(local_x, local_y)
    result = hand_tools.move(x=x, y=y, duration=duration)
    logger.log_info("Tool move done")
    return result


@mcp.tool()
def wait(seconds: float, instruction: str = "") -> dict[str, Any]:
    """Pause execution for the specified number of seconds."""
    logger.log_info(f"Tool wait start (seconds={seconds})")
    result = hand_tools.wait(seconds=seconds)
    logger.log_info("Tool wait done")
    return result


@mcp.tool()
def store_text(
    text: str,
    title: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Store text context to this run's storage.json index."""
    logger.log_info(
        f"Tool store_text start (text_len={len(text)}, title_present={bool(title)}, tags_count={len(tags or [])}"
    )
    result = _store_text(text=text, title=title, tags=tags)
    logger.log_info("Tool store_text done")
    return result


@mcp.tool()
def store_image(
    image_path: str,
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Copy an image to this run's storage folder and index it."""
    logger.log_info(
        f"Tool store_image start (summary_present={bool(summary)}, alias_present={bool(alias)}, tags_count={len(tags or [])}), image_path={image_path}"
    )
    result = _store_image(image_path=image_path, summary=summary, alias=alias, tags=tags)
    logger.log_info("Tool store_image done")
    return result


@mcp.tool()
def divide_step(path: str, new_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Split one step into nested child steps."""
    logger.log_info(f"Tool divide_step start path={path} count={len(new_steps)}")
    result = step_tools.divide_step(path=path, new_steps=new_steps)
    logger.log_info("Tool divide_step done")
    return result


@mcp.tool()
def create_new_steps(target_path: str, new_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Create one or more sibling steps at the target step level."""
    logger.log_info(f"Tool create_new_steps start target_path={target_path} count={len(new_steps)}")
    result = step_tools.create_new_steps(target_path=target_path, new_steps=new_steps)
    logger.log_info("Tool create_new_steps done")
    return result


OLLAMA_TOOL_FUNCTIONS: list[Callable[..., Any]] = [
    store_text,
    store_image,
    click,
    type_text,
    hotkey,
    move,
    wait,
]

HAND_TOOL_NAMES: set[str] = {tool.__name__ for tool in OLLAMA_TOOL_FUNCTIONS}


def get_ollama_tools() -> list[Callable[..., Any]]:
    """Return all tool callables in Ollama chat() format."""
    return OLLAMA_TOOL_FUNCTIONS


def get_tool_function_map() -> dict[str, Callable[..., Any]]:
    """Return a name -> callable map for all registered tools."""
    return {tool.__name__: tool for tool in OLLAMA_TOOL_FUNCTIONS}


def execute_tool_call(tool_name: str, arguments: dict[str, Any], image_path: str) -> Any:
    """Execute a local MCP tool callable by name."""
    logger.log_info(f"Executing tool call '{tool_name}' with argument keys: {sorted(arguments.keys())}")
    if "image_path" in arguments:
        arguments["image_path"] = image_path
    function_map = get_tool_function_map()
    function = function_map.get(tool_name)
    if function is None:
        logger.log_info(f"Unknown tool requested: {tool_name}")
        raise ValueError(f"unknown tool: {tool_name}")
    result = function(**arguments)
    logger.log_info(f"Completed tool call '{tool_name}'")
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
