from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from cua_mcp.tool_module import (
    _click,
    _cursor_position,
    _double_click,
    _hold_key,
    _hotkey,
    _key,
    _left_click,
    _left_click_drag,
    _left_mouse_down,
    _left_mouse_up,
    _middle_click,
    _mouse_move,
    _move,
    _type_text,
    _press_key,
    _right_click,
    _screenshot,
    _scroll,
    _store_image,
    _store_text,
    _triple_click,
    _wait,
    _zoom,
)

mcp_server = FastMCP("ComputerUseAgent")

@mcp_server.tool()
def click(
    instruction: str,
    button: str = "left",
):
    '''
    Click a UI element identified by natural-language instruction. Takes a fresh capture of the active monitor,
    runs OCR (artifacts under yolo_ocr/), and uses the brain LM to pick coordinates. Optional mouse button.
    
    Args:
        instruction: Natural-language description of what to click on the screen.
        button: Mouse button: typically 'left', 'right', or 'middle'.
    
    Returns:
        dict: A dictionary containing the clicked coordinates and the button used.
    '''
    return _click(instruction=instruction, button=button)


@mcp_server.tool()
def type_text(
    text: str,
    target_instruction: str,
):
    '''
    Focus a target region using natural-language instruction, then input text using keyboard automation.

    Args:
        text: Text content to input after the target is focused.
        target_instruction: Natural-language description of the field or region to focus first.

    Returns:
        dict: A dictionary containing typing execution details.
    '''
    return _type_text(text=text, target_instruction=target_instruction)


@mcp_server.tool()
def press_key(
    key: str,
    instruction: str = "",
):
    '''
    Send a single key press such as Enter, Tab, or Escape.

    Args:
        key: Key name to press.
        instruction: Optional context describing why the key is pressed.

    Returns:
        dict: A dictionary containing key-press execution details.
    '''
    return _press_key(key=key, instruction=instruction)


@mcp_server.tool()
def hotkey(
    keys: list[str] | str,
    instruction: str = "",
):
    '''
    Press multiple keys together as a shortcut, such as Ctrl+S or Alt+Tab.

    Args:
        keys: Key chord provided as a list of key names or supported string token.
        instruction: Optional context or rationale for the shortcut.

    Returns:
        dict: A dictionary containing hotkey execution details.
    '''
    return _hotkey(keys=keys, instruction=instruction)


@mcp_server.tool()
def move(
    instruction: str,
    duration: float = 0.0,
):
    '''
    Move the cursor to a target identified by natural-language instruction without clicking.

    Args:
        instruction: Natural-language description of where the cursor should move.
        duration: Seconds to animate the cursor movement.

    Returns:
        dict: A dictionary containing movement execution details.
    '''
    return _move(instruction=instruction, duration=duration)


@mcp_server.tool()
def wait(
    seconds: float,
    instruction: str = "",
):
    '''
    Pause execution for a fixed duration before continuing.

    Args:
        seconds: Delay duration in seconds.
        instruction: Optional context describing why the wait is needed.

    Returns:
        dict: A dictionary containing wait execution details.
    '''
    return _wait(seconds=seconds, instruction=instruction)


@mcp_server.tool()
def store_text(
    text: str,
    instruction: str = "",
    title: str = "",
    tags: list[str] | None = None,
):
    '''
    Persist text into run storage with optional metadata for later retrieval.

    Args:
        text: Body text to store.
        instruction: Optional context or provenance for the stored text.
        title: Short title or label for listing stored entries.
        tags: Optional tags used for categorization.

    Returns:
        dict: A dictionary containing storage execution details.
    '''
    return _store_text(text=text, instruction=instruction, title=title, tags=tags)


@mcp_server.tool()
def store_image(
    image_path: str,
    instruction: str = "",
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
):
    '''
    Register an image path in run storage with optional metadata.

    Args:
        image_path: Path to the image file to store.
        instruction: Optional context for why this image is stored.
        summary: Human-readable summary of the image contents or purpose.
        alias: Short memorable name to reference this image later.
        tags: Optional tags used for categorization.

    Returns:
        dict: A dictionary containing storage execution details.
    '''
    return _store_image(
        image_path=image_path,
        instruction=instruction,
        summary=summary,
        alias=alias,
        tags=tags,
    )


# --- CUA action vocabulary (see ToolCommand / agent schema) ---


@mcp_server.tool()
def key(
    key: str,
    instruction: str = "",
):
    '''
    Press and release one logical key, distinct from multi-key shortcuts.

    Args:
        key: Single key name to press.
        instruction: Optional context for logging.

    Returns:
        dict: A dictionary containing key execution details.
    '''
    return _key(key=key, instruction=instruction)


@mcp_server.tool()
def mouse_move(
    instruction: str,
    duration: float = 0.0,
):
    '''
    Move the cursor to a target described by natural-language instruction.

    Args:
        instruction: Natural-language description of where the cursor should move.
        duration: Seconds to animate the movement.

    Returns:
        dict: A dictionary containing movement execution details.
    '''
    return _mouse_move(instruction=instruction, duration=duration)


@mcp_server.tool()
def left_click(
    instruction: str,
):
    '''
    Left-click a UI element identified by natural-language instruction.

    Args:
        instruction: Natural-language description of the element to click.

    Returns:
        dict: A dictionary containing click execution details.
    '''
    return _left_click(instruction=instruction)


@mcp_server.tool()
def left_click_drag(
    instruction_start: str,
    instruction_end: str,
    duration: float = 0.5,
):
    '''
    Press and drag the left mouse button from a start target to an end target.

    Args:
        instruction_start: Natural-language instruction for the drag start point.
        instruction_end: Natural-language instruction for the drag end point.
        duration: Drag movement duration in seconds.

    Returns:
        dict: A dictionary containing drag execution details.
    '''
    return _left_click_drag(
        instruction_start=instruction_start,
        instruction_end=instruction_end,
        duration=duration,
    )


@mcp_server.tool()
def right_click(
    instruction: str,
):
    '''
    Right-click a UI element identified by natural-language instruction.

    Args:
        instruction: Natural-language description of where to right-click.

    Returns:
        dict: A dictionary containing click execution details.
    '''
    return _right_click(instruction=instruction)


@mcp_server.tool()
def middle_click(
    instruction: str,
):
    '''
    Middle-click a UI element identified by natural-language instruction.

    Args:
        instruction: Natural-language description of where to middle-click.

    Returns:
        dict: A dictionary containing click execution details.
    '''
    return _middle_click(instruction=instruction)


@mcp_server.tool()
def double_click(
    instruction: str,
):
    '''
    Double-click a UI element identified by natural-language instruction.

    Args:
        instruction: Natural-language description of what to double-click.

    Returns:
        dict: A dictionary containing click execution details.
    '''
    return _double_click(instruction=instruction)


@mcp_server.tool()
def triple_click(
    instruction: str,
):
    '''
    Triple-click a UI element identified by natural-language instruction.

    Args:
        instruction: Natural-language description of where to triple-click.

    Returns:
        dict: A dictionary containing click execution details.
    '''
    return _triple_click(instruction=instruction)


@mcp_server.tool()
def screenshot(
    path: str = "",
    instruction: str = "",
):
    '''
    Capture the current screen to a PNG path for evidence or downstream processing.

    Args:
        path: Output PNG path. Empty string lets the implementation choose a temp file.
        instruction: Optional note describing why the screenshot is taken.

    Returns:
        dict: A dictionary containing screenshot execution details.
    '''
    return _screenshot(path=path, instruction=instruction)


@mcp_server.tool()
def cursor_position(
    instruction: str = "",
):
    '''
    Read the current mouse cursor coordinates in screen space.

    Args:
        instruction: Optional context for logging.

    Returns:
        dict: A dictionary containing the current cursor coordinates.
    '''
    return _cursor_position(instruction=instruction)


@mcp_server.tool()
def left_mouse_down(
    instruction: str,
):
    '''
    Press the left mouse button down at a target without releasing it.

    Args:
        instruction: Natural-language description of where to press down.

    Returns:
        dict: A dictionary containing mouse-down execution details.
    '''
    return _left_mouse_down(instruction=instruction)


@mcp_server.tool()
def left_mouse_up(
    instruction: str,
):
    '''
    Release the left mouse button at a target location.

    Args:
        instruction: Natural-language description of where to release the button.

    Returns:
        dict: A dictionary containing mouse-up execution details.
    '''
    return _left_mouse_up(instruction=instruction)


@mcp_server.tool()
def scroll(
    instruction: str,
    clicks: int,
):
    '''
    Scroll at a target region identified by natural-language instruction.

    Args:
        instruction: Natural-language description of the scroll focus region or control.
        clicks: Wheel delta in clicks; sign controls scroll direction.

    Returns:
        dict: A dictionary containing scroll execution details.
    '''
    return _scroll(instruction=instruction, clicks=clicks)


@mcp_server.tool()
def hold_key(
    key: str,
    seconds: float,
    instruction: str = "",
):
    '''
    Hold a key down for a fixed duration, then release it.

    Args:
        key: Key name to hold.
        seconds: How long to keep the key depressed.
        instruction: Optional context for logging.

    Returns:
        dict: A dictionary containing hold-key execution details.
    '''
    return _hold_key(key=key, seconds=seconds, instruction=instruction)


@mcp_server.tool()
def zoom(
    instruction: str,
    scroll_clicks: int,
):
    '''
    Apply Ctrl+wheel zoom at a target identified by natural-language instruction.

    Args:
        instruction: Natural-language description of where to apply zoom.
        scroll_clicks: Wheel clicks while Ctrl is held; sign controls zoom in or out.

    Returns:
        dict: A dictionary containing zoom execution details.
    '''
    return _zoom(instruction=instruction, scroll_clicks=scroll_clicks)


TOOL_FUNCTIONS: list[callable[..., Any]] = [
    click,
    type_text,
    press_key,
    hotkey,
    move,
    wait,
    store_text,
    store_image,
    key,
    mouse_move,
    left_click,
    left_click_drag,
    right_click,
    cursor_position,
    left_mouse_down,
    left_mouse_up,
    scroll,
    hold_key,
    zoom,
    triple_click,
    middle_click,
    double_click,
    screenshot,
]

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
