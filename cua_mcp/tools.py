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
    _zoom,
    _maximize_window,
    _minimize_all_windows,
)

mcp_server = FastMCP("ComputerUseAgent")

@mcp_server.tool()
def click(
    button: str = "left",
    instruction: str = "",
):
    '''
    Click the mouse at the current cursor position (after using mouse_move to aim).

    Args:
        button: Mouse button: typically 'left', 'right', or 'middle'.
        instruction: Optional note for logging only (does not affect targeting).
    '''
    return _click(button=button).update({"instruction": instruction})


@mcp_server.tool()
def type_text(
    text: str,
    instruction: str = "",
):
    '''
    Paste text at the current keyboard focus via clipboard (Ctrl+V).

    Does not move or click the mouse.

    Args:
        text: Text to paste.
        instruction: Optional note for logging only.
    '''
    return _type_text(text=text).update({"instruction": instruction})


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
    '''
    return _press_key(key=key).update({"instruction": instruction})


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
    '''
    return _hotkey(keys=keys).update({"instruction": instruction})


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
    Persist text into run storage with optional metadata for later retrieval.

    Args:
        text: Body text to store.
        instruction: Optional context or provenance for the stored text.
        title: Short title or label for listing stored entries.
        tags: Optional tags used for categorization.
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
    Read the current OS clipboard as text and save it to this run's storage/ folder plus storage.json.

    Args:
        instruction: Optional context for logging only.
        title: Short title or label for the stored entry.
        tags: Optional tags for categorization.
        file_name: Optional basename for the .txt file (defaults to clipboard_<utc_timestamp>.txt).
                   Only the basename is used; paths are stripped for safety.
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
    Register an image path in run storage with optional metadata.

    Args:
        image_path: Path to the image file to store.
        instruction: Optional context for why this image is stored.
        summary: Human-readable summary of the image contents or purpose.
        alias: Short memorable name to reference this image later.
        tags: Optional tags used for categorization.
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
    Press and release one logical key, distinct from multi-key shortcuts.

    Args:
        key: Single key name to press.
        instruction: Optional context for logging.
    '''
    return _key(key=key).update({"instruction": instruction})


@mcp_server.tool()
async def mouse_move(
    instruction: str,
    duration: float = 0.0,
):
    '''
    Capture the screen, run YOLO+OCR, use the brain LM to pick a target, then move the cursor there.

    This is the only tool that performs vision-based targeting; call it before click/scroll/etc.

    Args:
        instruction: Natural-language description of where the cursor should move.
        duration: Seconds to animate the movement.
    '''
    return (await _move(instruction=instruction, duration=duration)).update({"instruction": instruction})


@mcp_server.tool()
def left_click_drag(
    x2: int,
    y2: int,
    duration: float = 0.5,
    instruction: str = "",
):
    '''
    Left-button drag from the current cursor position to screen coordinates (x2, y2).

    Args:
        x2: End x in screen pixels.
        y2: End y in screen pixels.
        duration: Drag movement duration in seconds.
        instruction: Optional note for logging only.
    '''
    return _left_click_drag(x2=x2, y2=y2, duration=duration).update({"instruction": instruction})


@mcp_server.tool()
def right_click(
    instruction: str = "",
):
    '''
    Right-click at the current cursor position.

    Args:
        instruction: Optional note for logging only.
    '''
    return _right_click().update({"instruction": instruction})


@mcp_server.tool()
def middle_click(
    instruction: str = "",
):
    '''
    Middle-click at the current cursor position.

    Args:
        instruction: Optional note for logging only.
    '''
    return _middle_click().update({"instruction": instruction})


@mcp_server.tool()
def double_click(
    instruction: str = "",
):
    '''
    Double-click (left) at the current cursor position.

    Args:
        instruction: Optional note for logging only.
    '''
    return _double_click().update({"instruction": instruction})


@mcp_server.tool()
def triple_click(
    instruction: str = "",
):
    '''
    Triple-click (left) at the current cursor position.

    Args:
        instruction: Optional note for logging only.
    '''
    return _triple_click().update({"instruction": instruction})


@mcp_server.tool()
def screenshot(
    path: str = "",
    instruction: str = "",
):
    '''
    Capture the current screen to a PNG path for evidence or downstream processing.

    Args:
        path: Output PNG path. Empty string uses the current run's storage/ folder with a timestamped name.
              Relative paths are resolved under that storage folder (basename only).
        instruction: Optional note describing why the screenshot is taken.
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
    '''
    return _cursor_position(instruction=instruction)


@mcp_server.tool()
def left_mouse_down(
    instruction: str = "",
):
    '''
    Press the left mouse button down without releasing (at the current cursor).

    Args:
        instruction: Optional note for logging only.
    '''
    return _left_mouse_down().update({"instruction": instruction})


@mcp_server.tool()
def left_mouse_up(
    instruction: str = "",
):
    '''
    Release the left mouse button (at the current cursor).

    Args:
        instruction: Optional note for logging only.
    '''
    return _left_mouse_up().update({"instruction": instruction})


@mcp_server.tool()
def scroll(
    clicks: int,
    instruction: str = "",
):
    '''
    Scroll the mouse wheel at the current cursor position.

    Args:
        clicks: Wheel delta in clicks; sign controls scroll direction.
        instruction: Optional note for logging only.
    '''
    return _scroll(clicks=clicks).update({"instruction": instruction})


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
    '''
    return _hold_key(key=key, seconds=seconds).update({"instruction": instruction})


@mcp_server.tool()
def zoom(
    scroll_clicks: int,
    instruction: str = "",
):
    '''
    Apply Ctrl+wheel zoom at the current cursor position.

    Args:
        scroll_clicks: Wheel clicks while Ctrl is held; sign controls zoom in or out.
        instruction: Optional note for logging only.
    '''
    return _zoom(scroll_clicks=scroll_clicks).update({"instruction": instruction})


@mcp_server.tool()
async def maximize_window(
    window_title_contains: str,
    instruction: str = "",
):
    '''
    Maximize a top-level window whose title contains the given substring (case-insensitive).

    If no windows match the substring, or several do, Ollama chooses among candidates;
    supply instruction with extra natural-language context for that disambiguation.

    Args:
        window_title_contains: Non-empty substring to match against window titles.
        instruction: Extra description used to disambiguate (0 or multiple substring matches).
    '''
    return (await _maximize_window(
        window_title_contains=window_title_contains,
    )).update({"instruction": instruction})


@mcp_server.tool()
def minimize_all_windows(
    instruction: str = "",
):
    '''
    Minimize all top-level windows with visible titles.

    Args:
        instruction: Optional note for logging only.
    '''
    return _minimize_all_windows().update({"instruction": instruction})


@mcp_server.tool()
def open_website(
    url: str,
    new: int = 0,
    autoraise: bool = True,
    instruction: str = "",
):
    '''
    Open a website URL using the system default web browser.

    Args:
        url: The URL to open (e.g. "https://www.google.com").
        new: 0 = same window if possible, 1 = new window, 2 = new tab.
        autoraise: Whether to try to raise the browser window.
        instruction: Optional note for logging only.
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
    List files in the current run's storage directory.

    Args:
        pattern: Glob-style pattern matched against file basenames (e.g. "*.txt", "screenshot_*.png").
        max_results: Upper bound on number of results.
        instruction: Optional note for logging only.
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
    Open (read) a text file from the current run's storage directory.

    For safety, this tool only allows reading ``.txt`` files by basename and does not allow paths.

    Args:
        file_name: Basename of the text file in storage/ (e.g. "clipboard_20260101_000000_000000.txt").
        max_chars: Max characters returned in content (longer files are truncated).
        encoding: Text encoding used to decode the file.
        instruction: Optional note for logging only.
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
    maximize_window,
    minimize_all_windows,
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
