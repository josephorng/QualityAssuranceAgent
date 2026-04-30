from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from cua_mcp import hand_tools
from cua_mcp.active_monitor_capture import active_monitor_offset
from cua_mcp.read_screen_text.ocr_image import get_coordinates
from cua_mcp.storage import store_image as _store_image
from cua_mcp.storage import store_text as _store_text
from src.common.run_state import get_run_state_manager
from src.common.settings import load_settings

settings = load_settings()
logger = get_run_state_manager()


def _get_active_capture_offset() -> tuple[int, int]:
    try:
        return active_monitor_offset()
    except Exception as exc:
        logger.log_info(f"Failed active_monitor_offset err={type(exc).__name__}: {exc}")
    return 0, 0


def _to_global_coordinate(local_x: int, local_y: int) -> tuple[int, int]:
    left, top = _get_active_capture_offset()
    return local_x + left, local_y + top


def _select_coordinate(instruction: str, coordinate_text: str) -> tuple[int, int]:
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
    out = json.loads(content)
    return int(out["x"]), int(out["y"])


def _resolve_point(instruction: str) -> tuple[int, int]:
    coordinate_text = get_coordinates()
    local_x, local_y = _select_coordinate(instruction=instruction, coordinate_text=coordinate_text)
    return _to_global_coordinate(local_x, local_y)


def click(instruction: str, button: str = "left") -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.click(x=x, y=y, button=button)


def paste_text(
    text: str,
    target_instruction: str,
    interval: float = 0.0,
) -> dict[str, Any]:
    x, y = _resolve_point(target_instruction)
    return hand_tools.type_text(text=text, coordinate=[x, y], interval=interval)


def press_key(key: str, instruction: str = "") -> dict[str, Any]:
    return hand_tools.hotkey(keys=key)


def hotkey(keys: list[str] | str, instruction: str = "") -> dict[str, Any]:
    return hand_tools.hotkey(keys=keys)


def move(instruction: str, duration: float = 0.0) -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.move(x=x, y=y, duration=duration)


def wait(seconds: float, instruction: str = "") -> dict[str, Any]:
    return hand_tools.wait(seconds=seconds)


def key(key: str, instruction: str = "") -> dict[str, Any]:
    return hand_tools.key_press(key)


def type_chars(
    text: str,
    instruction: str,
    interval: float = 0.0,
) -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.write_text(text=text, coordinate=[x, y], interval=interval)


def mouse_move(instruction: str, duration: float = 0.0) -> dict[str, Any]:
    return move(instruction=instruction, duration=duration)


def _click_at_instruction(instruction: str, **click_kw: Any) -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.click(x=x, y=y, **click_kw)


def left_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="left", clicks=1)


def right_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="right", clicks=1)


def middle_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="middle", clicks=1)


def double_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="left", clicks=2, interval=0.1)


def triple_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="left", clicks=3, interval=0.1)


def left_click_drag(
    instruction_start: str,
    instruction_end: str,
    duration: float = 0.5,
) -> dict[str, Any]:
    x1, y1 = _resolve_point(instruction_start)
    x2, y2 = _resolve_point(instruction_end)
    return hand_tools.drag(x1, y1, x2, y2, duration=duration, button="left")


def screenshot(path: str = "", instruction: str = "") -> dict[str, Any]:
    p = path.strip() if path else ""
    return hand_tools.screenshot_to_file(p or None)


def cursor_position(instruction: str = "") -> dict[str, Any]:
    return hand_tools.cursor_position()


def left_mouse_down(instruction: str) -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.mouse_down(x, y, button="left")


def left_mouse_up(instruction: str) -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.mouse_up(x, y, button="left")


def scroll(instruction: str, clicks: int) -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.scroll_at(clicks, x, y)


def hold_key(key: str, seconds: float, instruction: str = "") -> dict[str, Any]:
    return hand_tools.hold_key_down(key, seconds)


def zoom(instruction: str, scroll_clicks: int) -> dict[str, Any]:
    x, y = _resolve_point(instruction)
    return hand_tools.zoom_scroll(scroll_clicks, x, y)


def store_text(
    text: str,
    instruction: str = "",
    title: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return _store_text(text=text, title=title, tags=tags)


def store_image(
    image_path: str,
    instruction: str = "",
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return _store_image(image_path=image_path, summary=summary, alias=alias, tags=tags)


OLLAMA_TOOL_FUNCTIONS: list[Callable[..., Any]] = [
    store_text,
    store_image,
    click,
    paste_text,
    press_key,
    hotkey,
    move,
    wait,
    key,
    type_chars,
    mouse_move,
    left_click,
    left_click_drag,
    right_click,
    middle_click,
    double_click,
    triple_click,
    screenshot,
    cursor_position,
    left_mouse_down,
    left_mouse_up,
    scroll,
    hold_key,
    zoom,
]

TOOL_NAME_ALIASES: dict[str, str] = {"type": "type_chars"}

HAND_TOOL_NAMES: set[str] = {tool.__name__ for tool in OLLAMA_TOOL_FUNCTIONS} | set(TOOL_NAME_ALIASES.keys())


def get_ollama_tools() -> list[Callable[..., Any]]:
    return OLLAMA_TOOL_FUNCTIONS


def get_tool_function_map() -> dict[str, Callable[..., Any]]:
    mapping = {tool.__name__: tool for tool in OLLAMA_TOOL_FUNCTIONS}
    for alias, target_name in TOOL_NAME_ALIASES.items():
        mapping[alias] = mapping[target_name]
    return mapping


def execute_tool_call(tool_name: str, arguments: dict[str, Any], image_path: str) -> Any:
    args = dict(arguments)
    args.pop("image_path", None)
    args.pop("image_path_start", None)
    args.pop("image_path_end", None)
    function = get_tool_function_map().get(tool_name)
    if function is None:
        raise ValueError(f"unknown tool: {tool_name}")
    return function(**args)
