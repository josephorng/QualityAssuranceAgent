from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP

from cua_mcp.tool_module import (
    click,
    cursor_position,
    double_click,
    hold_key,
    hotkey,
    key as key_action,
    left_click,
    left_click_drag,
    left_mouse_down,
    left_mouse_up,
    middle_click,
    mouse_move,
    move,
    paste_text,
    press_key,
    right_click,
    screenshot,
    scroll,
    store_image,
    store_text,
    triple_click,
    type_chars,
    wait,
    zoom,
)

mcp = FastMCP("ComputerUseAgent")


@mcp.tool(
    name="click",
    description=(
        "Click a UI element identified by natural-language instruction. Takes a fresh capture of the active monitor, "
        "runs OCR (artifacts under yolo_ocr/), and uses the brain LM to pick coordinates. Optional mouse button."
    ),
)
def click_tool(
    instruction: Annotated[str, "Natural-language description of what to click on the screen."],
    button: Annotated[str, "Mouse button: typically 'left', 'right', or 'middle'."] = "left",
):
    return click(instruction=instruction, button=button)


@mcp.tool(
    name="paste_text",
    description=(
        "Focus a field or region by clicking the resolved target, then paste text via the clipboard (Ctrl+V). "
        "Use when typing must preserve Unicode or avoid key-by-key delays."
    ),
)
def paste_text_tool(
    text: Annotated[str, "Text to paste after focusing the target."],
    target_instruction: Annotated[str, "Natural-language description of the target to focus before pasting."],
    interval: Annotated[float, "Delay between keystrokes when typing fallback is used (default 0)."] = 0.0,
):
    return paste_text(text=text, target_instruction=target_instruction, interval=interval)


@mcp.tool(
    name="press_key",
    description=(
        "Send a single key press (e.g. Enter, Tab, Escape). Use for confirmations, navigation, or dialogs "
        "without composing a full shortcut."
    ),
)
def press_key_tool(
    key: Annotated[str, "The name of the key to press, e.g., 'enter', 'tab', or 'esc'"],
    instruction: Annotated[str, "Optional context for why this key is being pressed"] = "",
):
    return press_key(key=key, instruction=instruction)


@mcp.tool(
    name="hotkey",
    description=(
        "Chord multiple keys together (e.g. Ctrl+S, Alt+Tab). Pass keys in press order as understood by the OS."
    ),
)
def hotkey_tool(
    keys: Annotated[
        list[str] | str,
        "Key chord as a list of key names (e.g. ['ctrl','s']) or a single string token if your stack supports it.",
    ],
    instruction: Annotated[str, "Optional rationale or logging context for this shortcut."] = "",
):
    return hotkey(keys=keys, instruction=instruction)


@mcp.tool(
    name="move",
    description=(
        "Move the mouse pointer to a target described by instruction using a fresh active-monitor capture and OCR "
        "under yolo_ocr/—without clicking; use before hover menus or tooltips."
    ),
)
def move_tool(
    instruction: Annotated[str, "Natural-language description of where the cursor should end up."],
    duration: Annotated[float, "Seconds to animate the pointer move (0 = instant)."] = 0.0,
):
    return move(instruction=instruction, duration=duration)


@mcp.tool(
    name="wait",
    description=(
        "Pause execution for a fixed delay when the UI needs time to load, animate, or sync before the next action."
    ),
)
def wait_tool(
    seconds: Annotated[float, "Non-negative delay in seconds before continuing."],
    instruction: Annotated[str, "Optional note explaining why the wait is needed."] = "",
):
    return wait(seconds=seconds, instruction=instruction)


@mcp.tool(
    name="store_text",
    description=(
        "Persist arbitrary text into run storage for later retrieval—notes, extracted values, or labels with optional tags."
    ),
)
def store_text_tool(
    text: Annotated[str, "Body text to persist."],
    instruction: Annotated[str, "Optional context or provenance for this stored text."] = "",
    title: Annotated[str, "Short title or label for listing stored items."] = "",
    tags: Annotated[list[str] | None, "Optional taxonomy tags for retrieval."] = None,
):
    return store_text(text=text, instruction=instruction, title=title, tags=tags)


@mcp.tool(
    name="store_image",
    description=(
        "Register an image path with metadata (summary, alias, tags) so the agent can recall or cite it later."
    ),
)
def store_image_tool(
    image_path: Annotated[str, "Path to the image file being registered."],
    instruction: Annotated[str, "Optional context for why this image is stored."] = "",
    summary: Annotated[str, "Human-readable summary of image contents or purpose."] = "",
    alias: Annotated[str, "Short memorable name to reference this image later."] = "",
    tags: Annotated[list[str] | None, "Optional tags for categorization."] = None,
):
    return store_image(
        image_path=image_path,
        instruction=instruction,
        summary=summary,
        alias=alias,
        tags=tags,
    )


# --- CUA action vocabulary (see ToolCommand / agent schema) ---


@mcp.tool(
    name="key",
    description="Press and release one logical key via the keyboard driver—distinct from multi-key hotkeys.",
)
def key_tool(
    key: Annotated[str, "Single key name as understood by pyautogui.press (e.g. 'a', 'enter')."],
    instruction: Annotated[str, "Optional context for logging."] = "",
):
    return key_action(key=key, instruction=instruction)


@mcp.tool(
    name="type",
    description=(
        "Click to focus using a fresh capture + OCR (yolo_ocr/), then type raw characters with optional per-key interval."
    ),
)
def type_tool(
    text: Annotated[str, "Characters to type after focusing."],
    instruction: Annotated[str, "Natural-language description of where to click to focus."],
    interval: Annotated[float, "Seconds between key events when using pyautogui.write."] = 0.0,
):
    return type_chars(text=text, instruction=instruction, interval=interval)


@mcp.tool(
    name="mouse_move",
    description=(
        "Synonym for move: relocate the cursor using a fresh active-monitor capture and yolo_ocr/, optionally animated."
    ),
)
def mouse_move_tool(
    instruction: Annotated[str, "Natural-language description of the hover/move target."],
    duration: Annotated[float, "Seconds to animate the move (0 = instant)."] = 0.0,
):
    return mouse_move(instruction=instruction, duration=duration)


@mcp.tool(
    name="left_click",
    description=(
        "Primary click at the resolved target (fresh capture, yolo_ocr/)—default for buttons, links, and most controls."
    ),
)
def left_click_tool(
    instruction: Annotated[str, "Natural-language description of the element to left-click."],
):
    return left_click(instruction=instruction)


@mcp.tool(
    name="left_click_drag",
    description=(
        "Press left at a start target and drag to an end target—each point uses a fresh capture (two yolo_ocr pairs)."
    ),
)
def left_click_drag_tool(
    instruction_start: Annotated[str, "Instruction for the drag start point (e.g. scrollbar thumb)."],
    instruction_end: Annotated[str, "Instruction for the drag end point."],
    duration: Annotated[float, "Duration of the drag motion in seconds."] = 0.5,
):
    return left_click_drag(
        instruction_start=instruction_start,
        instruction_end=instruction_end,
        duration=duration,
    )


@mcp.tool(
    name="right_click",
    description=(
        "Open context menus, alternate actions, or “more options” UI at the resolved location."
    ),
)
def right_click_tool(
    instruction: Annotated[str, "Natural-language description of where to right-click."],
):
    return right_click(instruction=instruction)


@mcp.tool(
    name="middle_click",
    description=(
        "Middle-click to open links in a new tab, close tabs in many browsers, or other app-specific middle-button behavior."
    ),
)
def middle_click_tool(
    instruction: Annotated[str, "Natural-language description of where to middle-click."],
):
    return middle_click(instruction=instruction)


@mcp.tool(
    name="double_click",
    description=(
        "Double-click to open files, edit cells, or activate word selections depending on the target application."
    ),
)
def double_click_tool(
    instruction: Annotated[str, "Natural-language description of what to double-click."],
):
    return double_click(instruction=instruction)


@mcp.tool(
    name="triple_click",
    description=(
        "Triple-clicking selects entire paragraphs in documents, full lines in code editors, or complete URLs in address bars."
    ),
)
def triple_click_tool(
    instruction: Annotated[str, "Natural-language description of where to triple-click."],
):
    return triple_click(instruction=instruction)


@mcp.tool(
    name="screenshot",
    description=(
        "Capture the current screen to a PNG path (or a generated temp file) for evidence, diffing, or downstream OCR."
    ),
)
def screenshot_tool(
    path: Annotated[str, "Output PNG path; empty string lets the implementation pick a temp file."] = "",
    instruction: Annotated[str, "Optional note for why the screenshot is taken."] = "",
):
    return screenshot(path=path, instruction=instruction)


@mcp.tool(
    name="cursor_position",
    description=(
        "Read the current mouse pointer coordinates in screen space—useful for debugging or confirming hover location."
    ),
)
def cursor_position_tool(
    instruction: Annotated[str, "Optional; reserved for future context—currently unused by the implementation."] = "",
):
    return cursor_position(instruction=instruction)


@mcp.tool(
    name="left_mouse_down",
    description=(
        "Press the left button at the resolved point without releasing—start of a drag or marquee selection."
    ),
)
def left_mouse_down_tool(
    instruction: Annotated[str, "Natural-language description of where to press the left button down."],
):
    return left_mouse_down(instruction=instruction)


@mcp.tool(
    name="left_mouse_up",
    description=(
        "Release the left button at the resolved point—typically paired with left_mouse_down after a drag."
    ),
)
def left_mouse_up_tool(
    instruction: Annotated[str, "Natural-language description of where to release the left button."],
):
    return left_mouse_up(instruction=instruction)


@mcp.tool(
    name="scroll",
    description=(
        "Move the wheel at an instruction-resolved location—positive/negative clicks scroll direction per PyAutoGUI conventions."
    ),
)
def scroll_tool(
    instruction: Annotated[str, "Natural-language description of the scroll focus region or control."],
    clicks: Annotated[int, "Wheel delta in clicks; sign maps to up/down per PyAutoGUI on this platform."],
):
    return scroll(instruction=instruction, clicks=clicks)


@mcp.tool(
    name="hold_key",
    description=(
        "Hold a key down for a duration then release—simulates long-press or gaming-style held inputs."
    ),
)
def hold_key_tool(
    key: Annotated[str, "Key name to hold (e.g. 'shift', 'w')."],
    seconds: Annotated[float, "How long to keep the key depressed before key-up."],
    instruction: Annotated[str, "Optional context for logging."] = "",
):
    return hold_key(key=key, seconds=seconds, instruction=instruction)


@mcp.tool(
    name="zoom",
    description=(
        "Ctrl+scroll zoom at the resolved point—common for browser/page zoom or canvas zoom where supported."
    ),
)
def zoom_tool(
    instruction: Annotated[str, "Natural-language description of where to apply Ctrl+wheel zoom."],
    scroll_clicks: Annotated[int, "Wheel clicks while Ctrl is held; sign controls zoom in/out."],
):
    return zoom(instruction=instruction, scroll_clicks=scroll_clicks)


if __name__ == "__main__":
    mcp.run(transport="stdio")
