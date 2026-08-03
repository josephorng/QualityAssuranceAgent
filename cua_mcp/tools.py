from __future__ import annotations

import webbrowser
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.common.runtime_context import is_runtime_command_mode, is_smart_mode
from cua_mcp.tool_module import (
    _click,
    _check_object_exists,
    _cursor_position,
    _double_click,
    _drag,
    _hold_key,
    _hotkey,
    _key,
    _left_mouse_down,
    _left_mouse_up,
    _list_storage_files,
    _middle_click,
    _move_mouse,
    _move_mouse_visual,
    _type_text,
    _press_key,
    _read_storage_text,
    _right_click,
    _scroll,
    _store_image,
    _store_clipboard_text,
    _store_text,
    _triple_click,
    _wait,
    _zoom,
    _maximize_windows,
    _close_windows,
    _minimize_windows,
)

mcp_server = FastMCP("ComputerUseAgent")

@mcp_server.tool()
def click(
    button: str = "left",
    modifiers: list[str] | None = None,
    instruction: str = "",
):
    '''
    Single click at the current cursor (點擊 / 點選). Use after move_mouse.
    Do not use for double-click (連按兩下) — use double_click instead.
    For Ctrl+click or Shift+click, pass modifiers=["ctrl"] or modifiers=["shift"].
    '''
    return _click(button=button, modifiers=modifiers).update({"instruction": instruction})


@mcp_server.tool()
def type_text(
    text: str,
    instruction: str = "",
):
    '''
    Type or paste text into the focused input.
    '''
    return _type_text(text=text).update({"instruction": instruction})


@mcp_server.tool()
def press_key(
    key: str,
    instruction: str = "",
):
    '''
    Press a single keyboard key.
    '''
    return _press_key(key=key).update({"instruction": instruction})


@mcp_server.tool()
def hotkey(
    keys: list[str] | str,
    instruction: str = "",
):
    '''
    Press a keyboard shortcut combination.

    keys: key names as a list (e.g. ["win", "e"]) or string such as "win+e",
    "win,e", or "[win,e]".
    '''
    return _hotkey(keys=keys).update({"instruction": instruction})


@mcp_server.tool()
def wait(
    seconds: float,
    instruction: str = "",
):
    '''
    Pause execution for a number of seconds.
    '''
    return _wait(seconds=seconds).update({"instruction": instruction})


@mcp_server.tool()
def store_text(
    text: str,
    instruction: str = "",
    title: str = "",
    tags: list[str] | None = None,
):
    '''
    Save text to run storage.
    '''
    return _store_text(text=text, title=title, tags=tags).update({"instruction": instruction})


@mcp_server.tool()
def store_clipboard_text(
    instruction: str = "",
    title: str = "",
    tags: list[str] | None = None,
    file_name: str = "",
):
    '''
    Save current clipboard text to run storage.
    '''
    return _store_clipboard_text(title=title, tags=tags, file_name=file_name).update({"instruction": instruction})


@mcp_server.tool()
def store_image(
    image_path: str,
    instruction: str = "",
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
):
    '''
    Save an image reference to run storage.
    '''
    return _store_image(
        image_path=image_path,
        summary=summary,
        alias=alias,
        tags=tags,
    ).update({"instruction": instruction})


# --- CUA action vocabulary (see ToolCommand / agent schema) ---


@mcp_server.tool()
def key(
    key: str,
    instruction: str = "",
):
    '''
    Press and release one key.
    '''
    return _key(key=key).update({"instruction": instruction})


@mcp_server.tool()
async def move_mouse(
    instruction: str,
    nearby_objects: list[str] | None = None,
):
    '''
    Take a screenshot, detect UI targets (text, element, input, scrollbar), and move
    the mouse cursor to the best match for the instruction.

    instruction: primary target phrase (e.g. 「資料夾」圖示), including any relative
    pixel offsets in Traditional Chinese when needed (e.g. 「資料夾」圖示左方5個像素、下方55個像素的位置).
    nearby_objects: optional list of nearby landmark labels used to disambiguate the
    target. When the goal gives a side (左邊/右邊/上面/下面/左上方/右上方/左下方/右下方),
    keep it as a directed phrase (e.g. ["在「joseph」文字的下面", "在「確定」文字的上面"]);
    do not strip the side down to an undirected label like ["「joseph」文字"].
    Undirected labels are fine only when the goal has no side
    (e.g. ["「Edge」圖示", "「Copilot」圖示"]). Prefer this over embedding
    （附近有…） comments inside instruction.
    '''
    duration: float = 0.2
    result = await _move_mouse(
        instruction=instruction,
        nearby_objects=nearby_objects,
        duration=duration,
    )
    result["instruction"] = instruction
    return result


@mcp_server.tool()
async def move_mouse_visual(
    instruction: str,
):
    """
    Visual fallback for ambiguous targets. Capture fresh screenshot(s), run YOLO/OCR,
    show the full indexed candidate list and screenshot(s) to the multimodal LLM once,
    then move to the center of the candidate selected by that LLM.

    Use when normal move_mouse repeatedly fails to match a visible target. Skips
    target parsing, similarity filtering, and nearby-object filtering. When the
    chosen target has label-similar peers, runs similar_function_describe to
    re-rank among those peers before moving.
    """
    duration: float = 0.2
    return await _move_mouse_visual(
        instruction=instruction,
        duration=duration,
    )


@mcp_server.tool()
async def check_object_exists(
    instruction: str,
    nearby_objects: list[str] | None = None,
):
    '''
    Check whether the UI target matching instruction is currently on screen.
    Returns exists true/false; does not move or click.
    Use when the step depends on whether something is visible; the brain decides
    whether to continue based on exists and the full step wording (e.g. 如果畫面上有… / 如果畫面上沒有…).

    instruction: primary target phrase to look for (e.g. 「取代目的地中的檔案」文字).
    nearby_objects: optional list of nearby landmark labels used to disambiguate the
    target (e.g. ["「取消」文字", "「略過」文字"]). Prefer this over embedding
    （附近有…） comments inside instruction.
    '''
    return await _check_object_exists(
        instruction=instruction,
        nearby_objects=nearby_objects,
    )


@mcp_server.tool()
async def drag(
    start_instruction: str,
    destination_instruction: str,
    start_nearby_objects: list[str] | None = None,
    destination_nearby_objects: list[str] | None = None,
    duration: float = 0.5,
    button: str = "left",
):
    '''
    Drag from the UI target matching start_instruction to the target matching
    destination_instruction. It will move the mouse cursor to the start_instruction and then drag to the destination_instruction.
    start_instruction: the source phrase, including any relative pixel offsets in Traditional Chinese (e.g. 「OneNote」文字左方5個像素、下方55個像素的位置)
    destination_instruction: the destination phrase, including any relative pixel offsets in Traditional Chinese (e.g. 「OneNote」文字左方5個像素、下方55個像素的位置)
    start_nearby_objects: optional list of nearby landmark labels used to disambiguate the
    drag source (e.g. ["「Desktop」文字"]). Prefer this over embedding （起點附近有…）
    comments inside start_instruction.
    destination_nearby_objects: optional list of nearby landmark labels used to disambiguate
    the drag destination (e.g. ["「新增文字文件txt」文字"]). Prefer this over embedding
    （附近有…） / （終點附近有…） comments inside destination_instruction.
    '''
    return await _drag(
        start_instruction=start_instruction,
        destination_instruction=destination_instruction,
        start_nearby_objects=start_nearby_objects,
        destination_nearby_objects=destination_nearby_objects,
        duration=duration,
        button=button,
    )


@mcp_server.tool()
def right_click(
    instruction: str = "",
):
    '''
    Right-click at the current cursor position.
    '''
    return _right_click().update({"instruction": instruction})


@mcp_server.tool()
def middle_click(
    instruction: str = "",
):
    '''
    Middle-click at the current cursor position.
    '''
    return _middle_click().update({"instruction": instruction})


@mcp_server.tool()
def double_click(
    modifiers: list[str] | None = None,
    instruction: str = "",
):
    '''
    Double-click at the current cursor position (連按兩下).
    Do not use for a normal single click (點擊 / 點選) — use click instead.
    For Ctrl+double-click or Shift+double-click, pass modifiers=["ctrl"] or modifiers=["shift"].
    '''
    return _double_click(modifiers=modifiers).update({"instruction": instruction})


@mcp_server.tool()
def triple_click(
    instruction: str = "",
):
    '''
    Triple-click at the current cursor position.
    '''
    return _triple_click().update({"instruction": instruction})


@mcp_server.tool()
def cursor_position(
    instruction: str = "",
):
    '''
    Get the current mouse cursor coordinates.
    '''
    return _cursor_position(instruction=instruction)


@mcp_server.tool()
def left_mouse_down(
    instruction: str = "",
):
    '''
    Press and hold the left mouse button.
    '''
    return _left_mouse_down().update({"instruction": instruction})


@mcp_server.tool()
def left_mouse_up(
    instruction: str = "",
):
    '''
    Release the left mouse button.
    '''
    return _left_mouse_up().update({"instruction": instruction})


@mcp_server.tool()
def scroll(
    clicks: int,
    instruction: str = "",
):
    '''
    Scroll the mouse wheel at the current cursor. Positive ``clicks`` move the
    document downward (toward the bottom); negative move upward. Each unit is
    roughly one wheel detent. Hover the scrollable region before calling.
    '''
    return _scroll(clicks=clicks).update({"instruction": instruction})


@mcp_server.tool()
def hold_key(
    key: str,
    seconds: float,
    instruction: str = "",
):
    '''
    Hold a keyboard key for a duration.
    '''
    return _hold_key(key=key, seconds=seconds).update({"instruction": instruction})


@mcp_server.tool()
def zoom(
    scroll_clicks: int,
    instruction: str = "",
):
    '''
    Zoom in or out using mouse wheel input.
    '''
    return _zoom(scroll_clicks=scroll_clicks).update({"instruction": instruction})


@mcp_server.tool()
async def maximize_windows(
    window_title_contains: str,
    instruction: str = "",
):
    '''
    Maximize windows matching the title text. Set the window_title_contains
    to "all" to maximize all windows.
    '''
    return (await _maximize_windows(
        window_title_contains=window_title_contains,
        instruction=instruction,
    )).update({"instruction": instruction})


@mcp_server.tool()
async def close_windows(
    window_title_contains: str,
    instruction: str = "",
):
    '''
    Close windows matching the title text. Set the window_title_contains to "all" to close all windows.
    '''
    return (await _close_windows(
        window_title_contains=window_title_contains,
        instruction=instruction,
    )).update({"instruction": instruction})


@mcp_server.tool()
async def minimize_windows(
    window_title_contains: str,
    instruction: str = "",
):
    '''
    Minimize windows matching the title text. Set the window_title_contains
    to "all" to minimize all windows.
    '''
    return (await _minimize_windows(
        window_title_contains=window_title_contains,
        instruction=instruction,
    )).update({"instruction": instruction})


@mcp_server.tool()
def open_website(
    url: str,
    new: int = 0,
    autoraise: bool = True,
    instruction: str = "",
):
    '''
    Open a URL in the default web browser.
    '''
    normalized = (url or "").strip()
    if not normalized:
        raise ValueError("url must be a non-empty string")
    if "://" not in normalized:
        normalized = f"https://{normalized}"

    ok = webbrowser.open(normalized, new=new, autoraise=autoraise)
    return {
        "status": "opened" if ok else "not_opened",
        "url": normalized,
        "new": new,
        "autoraise": autoraise,
        "instruction": instruction,
    }


@mcp_server.tool()
def list_storage_files(
    pattern: str = "*",
    max_results: int = 200,
    instruction: str = "",
):
    '''
    List files in run storage.
    '''
    return _list_storage_files(pattern=pattern, max_results=max_results).update({"instruction": instruction})


@mcp_server.tool()
def open_storage_text(
    file_name: str,
    max_chars: int = 20000,
    encoding: str = "utf-8",
    instruction: str = "",
):
    '''
    Read a text file from run storage.
    '''
    return _read_storage_text(file_name=file_name, max_chars=max_chars, encoding=encoding).update(
        {"instruction": instruction}
    )


TOOL_FUNCTIONS: list[callable[..., Any]] = [
    click,
    type_text,
    press_key,
    hotkey,
    wait,
    store_text,
    store_clipboard_text,
    store_image,
    key,
    move_mouse,
    move_mouse_visual,
    check_object_exists,
    drag,
    right_click,
    cursor_position,
    left_mouse_down,
    left_mouse_up,
    scroll,
    hold_key,
    zoom,
    maximize_windows,
    close_windows,
    minimize_windows,
    triple_click,
    middle_click,
    double_click,
    open_website,
]


def get_mode_tool_functions() -> list[callable[..., Any]]:
    """Return the action tools exposed to the model for the active run mode."""
    if is_smart_mode():
        hidden_names = {"move_mouse"}
    elif not is_runtime_command_mode():
        hidden_names = {"move_mouse_visual"}
    else:
        hidden_names = set()
    return [tool for tool in TOOL_FUNCTIONS if tool.__name__ not in hidden_names]


def get_mode_tool_names() -> set[str]:
    """Return action tool names exposed to the model for the active run mode."""
    return {tool.__name__ for tool in get_mode_tool_functions()}


VERIFICATION_TOOLS: list[callable[..., Any]] = [
    list_storage_files,
    open_storage_text,
]

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
