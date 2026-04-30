from __future__ import annotations

from typing import Annotated, Any

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

@mcp_server.tool(
    description=(
        "Click a UI element identified by natural-language instruction. Takes a fresh capture of the active monitor, "
        "runs OCR (artifacts under yolo_ocr/), and uses the brain LM to pick coordinates. Optional mouse button."
    ),
)
def click(
    instruction: Annotated[str, "Natural-language description of what to click on the screen."],
    button: Annotated[str, "Mouse button: typically 'left', 'right', or 'middle'."] = "left",
):
    return _click(instruction=instruction, button=button)


@mcp_server.tool(
    description=(
        "Focus a field or region by clicking the resolved target, then input text via the keyboard."
    ),
)
def type_text(
    text: Annotated[str, "Text to paste after focusing the target."],
    target_instruction: Annotated[str, "Natural-language description of the target to focus before pasting."],
    interval: Annotated[float, "Delay between keystrokes when typing fallback is used (default 0)."] = 0.0,
):
    return _type_text(text=text, target_instruction=target_instruction, interval=interval)


@mcp_server.tool(
    description=(
        "Send a single key press (e.g. Enter, Tab, Escape). Use for confirmations, navigation, or dialogs "
        "without composing a full shortcut."
    ),
)
def press_key(
    key: Annotated[str, "The key to press, e.g., 'enter', 'tab', or 'esc'"],
    instruction: Annotated[str, "Optional context for why this key is being pressed"] = "",
):
    return _press_key(key=key, instruction=instruction)


@mcp_server.tool(
    description=(
        "Chord multiple keys together (e.g. Ctrl+S, Alt+Tab). Pass keys in press order as understood by the OS."
    ),
)
def hotkey(
    keys: Annotated[
        list[str] | str,
        "Key chord as a list of key names (e.g. ['ctrl','s']) or a single string token if your stack supports it.",
    ],
    instruction: Annotated[str, "Optional rationale or logging context for this shortcut."] = "",
):
    return _hotkey(keys=keys, instruction=instruction)


@mcp_server.tool(
    description=(
        "Move the mouse pointer to a target described by instruction using a fresh active-monitor capture and OCR "
        "under yolo_ocr/—without clicking; use before hover menus or tooltips."
    ),
)
def move(
    instruction: Annotated[str, "Natural-language description of where the cursor should end up."],
    duration: Annotated[float, "Seconds to animate the pointer move (0 = instant)."] = 0.0,
):
    return _move(instruction=instruction, duration=duration)


@mcp_server.tool(
    description=(
        "Pause execution for a fixed delay when the UI needs time to load, animate, or sync before the next action."
    ),
)
def wait(
    seconds: Annotated[float, "Non-negative delay in seconds before continuing."],
    instruction: Annotated[str, "Optional note explaining why the wait is needed."] = "",
):
    return _wait(seconds=seconds, instruction=instruction)


@mcp_server.tool(
    description=(
        "Persist arbitrary text into run storage for later retrieval—notes, extracted values, or labels with optional tags."
    ),
)
def store_text(
    text: Annotated[str, "Body text to persist."],
    instruction: Annotated[str, "Optional context or provenance for this stored text."] = "",
    title: Annotated[str, "Short title or label for listing stored items."] = "",
    tags: Annotated[list[str] | None, "Optional taxonomy tags for retrieval."] = None,
):
    return _store_text(text=text, instruction=instruction, title=title, tags=tags)


@mcp_server.tool(
    description=(
        "Register an image path with metadata (summary, alias, tags) so the agent can recall or cite it later."
    ),
)
def store_image(
    image_path: Annotated[str, "Path to the image file being registered."],
    instruction: Annotated[str, "Optional context for why this image is stored."] = "",
    summary: Annotated[str, "Human-readable summary of image contents or purpose."] = "",
    alias: Annotated[str, "Short memorable name to reference this image later."] = "",
    tags: Annotated[list[str] | None, "Optional tags for categorization."] = None,
):
    return _store_image(
        image_path=image_path,
        instruction=instruction,
        summary=summary,
        alias=alias,
        tags=tags,
    )


# --- CUA action vocabulary (see ToolCommand / agent schema) ---


@mcp_server.tool(
    description="Press and release one logical key via the keyboard driver—distinct from multi-key hotkeys.",
)
def key(
    key: Annotated[str, "Single key name as understood by pyautogui.press (e.g. 'a', 'enter')."],
    instruction: Annotated[str, "Optional context for logging."] = "",
):
    return _key(key=key, instruction=instruction)


@mcp_server.tool(
    description=(
        "Synonym for move: relocate the cursor using a fresh active-monitor capture and yolo_ocr/, optionally animated."
    ),
)
def mouse_move(
    instruction: Annotated[str, "Natural-language description of the hover/move target."],
    duration: Annotated[float, "Seconds to animate the move (0 = instant)."] = 0.0,
):
    return _mouse_move(instruction=instruction, duration=duration)


@mcp_server.tool(
    description=(
        "Primary click at the resolved target (fresh capture, yolo_ocr/)—default for buttons, links, and most controls."
    ),
)
def left_click(
    instruction: Annotated[str, "Natural-language description of the element to left-click."],
):
    return _left_click(instruction=instruction)


@mcp_server.tool(
    description=(
        "Press left at a start target and drag to an end target—each point uses a fresh capture (two yolo_ocr pairs)."
    ),
)
def left_click_drag(
    instruction_start: Annotated[str, "Instruction for the drag start point (e.g. scrollbar thumb)."],
    instruction_end: Annotated[str, "Instruction for the drag end point."],
    duration: Annotated[float, "Duration of the drag motion in seconds."] = 0.5,
):
    return _left_click_drag(
        instruction_start=instruction_start,
        instruction_end=instruction_end,
        duration=duration,
    )


@mcp_server.tool(
    description=(
        "Open context menus, alternate actions, or “more options” UI at the resolved location."
    ),
)
def right_click(
    instruction: Annotated[str, "Natural-language description of where to right-click."],
):
    return _right_click(instruction=instruction)


@mcp_server.tool(
    description=(
        "Middle-click to open links in a new tab, close tabs in many browsers, or other app-specific middle-button behavior."
    ),
)
def middle_click(
    instruction: Annotated[str, "Natural-language description of where to middle-click."],
):
    return _middle_click(instruction=instruction)


@mcp_server.tool(
    description=(
        "Double-click to open files, edit cells, or activate word selections depending on the target application."
    ),
)
def double_click(
    instruction: Annotated[str, "Natural-language description of what to double-click."],
):
    return _double_click(instruction=instruction)


@mcp_server.tool(
    description=(
        "Triple-clicking selects entire paragraphs in documents, full lines in code editors, or complete URLs in address bars."
    ),
)
def triple_click(
    instruction: Annotated[str, "Natural-language description of where to triple-click."],
):
    return _triple_click(instruction=instruction)


@mcp_server.tool(
    description=(
        "Capture the current screen to a PNG path (or a generated temp file) for evidence, diffing, or downstream OCR."
    ),
)
def screenshot(
    path: Annotated[str, "Output PNG path; empty string lets the implementation pick a temp file."] = "",
    instruction: Annotated[str, "Optional note for why the screenshot is taken."] = "",
):
    return _screenshot(path=path, instruction=instruction)


@mcp_server.tool(
    description=(
        "Read the current mouse pointer coordinates in screen space—useful for debugging or confirming hover location."
    ),
)
def cursor_position(
    instruction: Annotated[str, "Optional; reserved for future context—currently unused by the implementation."] = "",
):
    return _cursor_position(instruction=instruction)


@mcp_server.tool(
    description=(
        "Press the left button at the resolved point without releasing—start of a drag or marquee selection."
    ),
)
def left_mouse_down(
    instruction: Annotated[str, "Natural-language description of where to press the left button down."],
):
    return _left_mouse_down(instruction=instruction)


@mcp_server.tool(
    description=(
        "Release the left button at the resolved point—typically paired with left_mouse_down after a drag."
    ),
)
def left_mouse_up(
    instruction: Annotated[str, "Natural-language description of where to release the left button."],
):
    return _left_mouse_up(instruction=instruction)


@mcp_server.tool(
    description=(
        "Move the wheel at an instruction-resolved location—positive/negative clicks scroll direction per PyAutoGUI conventions."
    ),
)
def scroll(
    instruction: Annotated[str, "Natural-language description of the scroll focus region or control."],
    clicks: Annotated[int, "Wheel delta in clicks; sign maps to up/down per PyAutoGUI on this platform."],
):
    return _scroll(instruction=instruction, clicks=clicks)


@mcp_server.tool(
    description=(
        "Hold a key down for a duration then release—simulates long-press or gaming-style held inputs."
    ),
)
def hold_key(
    key: Annotated[str, "Key name to hold (e.g. 'shift', 'w')."],
    seconds: Annotated[float, "How long to keep the key depressed before key-up."],
    instruction: Annotated[str, "Optional context for logging."] = "",
):
    return _hold_key(key=key, seconds=seconds, instruction=instruction)


@mcp_server.tool(
    description=(
        "Ctrl+scroll zoom at the resolved point—common for browser/page zoom or canvas zoom where supported."
    ),
)
def zoom(
    instruction: Annotated[str, "Natural-language description of where to apply Ctrl+wheel zoom."],
    scroll_clicks: Annotated[int, "Wheel clicks while Ctrl is held; sign controls zoom in/out."],
):
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
