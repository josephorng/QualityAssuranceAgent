from __future__ import annotations

import webbrowser
from typing import Any

from mcp.server.fastmcp import FastMCP

from cua_mcp.tool_module import (
    _click,
    _cursor_position,
    _double_click,
    _hold_key,
    _hotkey,
    _key,
    _left_click_drag,
    _left_mouse_down,
    _left_mouse_up,
    _list_storage_files,
    _middle_click,
    _move,
    _type_text,
    _press_key,
    _read_storage_text,
    _right_click,
    _screenshot,
    _scroll,
    _store_image,
    _store_clipboard_text,
    _store_text,
    _triple_click,
    _wait,
    _move_to_ui_element,
    _zoom,
    _maximize_windows,
    _close_windows,
    _minimize_windows,
)

mcp_server = FastMCP("ComputerUseAgent")

@mcp_server.tool()
def click(
    button: str = "left",
    instruction: str = "",
):
    '''
    Click the selected mouse button.
    '''
    return _click(button=button).update({"instruction": instruction})


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
async def mouse_move(
    target_type: str,
    instruction: str,
    target_text: str = "",
):
    '''
    Move the mouse cursor based on the instruction.
    target_type: "text" or "ui_element"
    target: the target text if the target_type is "text", leave it blank if the target_type is "ui_element"
    '''
    duration: float = 0.2
    if target_type == "text":
        return (await _move(target=target_text, instruction=instruction, duration=duration)).update({"instruction": instruction})
    elif target_type == "ui_element":
        return (await _move_to_ui_element(instruction=instruction, duration=duration)).update({"instruction": instruction})
    else:
        raise ValueError(f"Invalid target type: {target_type}")


@mcp_server.tool()
def left_click_drag(
    x2: int,
    y2: int,
    duration: float = 0.5,
    instruction: str = "",
):
    '''
    Drag from current cursor position to a target point.
    '''
    return _left_click_drag(x2=x2, y2=y2, duration=duration).update({"instruction": instruction})


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
    instruction: str = "",
):
    '''
    Double-click at the current cursor position.
    '''
    return _double_click().update({"instruction": instruction})


@mcp_server.tool()
def triple_click(
    instruction: str = "",
):
    '''
    Triple-click at the current cursor position.
    '''
    return _triple_click().update({"instruction": instruction})


@mcp_server.tool()
def screenshot(
    path: str = "",
    instruction: str = "",
):
    '''
    Capture a screenshot and save it.
    '''
    return _screenshot(path=path, instruction=instruction)


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
    mouse_move,
    left_click_drag,
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
    screenshot,
    open_website,
]

VERIFICATION_TOOLS: list[callable[..., Any]] = [
    list_storage_files,
    open_storage_text,
]

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
