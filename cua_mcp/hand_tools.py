from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp
from time import sleep
from typing import Any
# import tkinter as tk
import pyperclip

import pyautogui
import pygetwindow as gw
from src.common.ollama_client import OllamaClient
from src.common.settings import load_settings
from src.eye.capture import capture_active_monitor_to_file

_settings = load_settings()
_ollama = OllamaClient(_settings.ollama_host, timeout_seconds=60)


def _normalize_hotkey_token(key: str) -> str:
    """Normalize odd wrappers that can appear in model output."""
    cleaned = key.strip()
    cleaned = cleaned.replace('<|"|>', "")
    return cleaned.strip("\"'")


def click(
    x: int | None = None,
    y: int | None = None,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.0,
) -> dict[str, Any]:
    """Click a screen coordinate, or the current cursor if x and y are omitted."""
    if x is not None and y is not None:
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
        rx, ry = x, y
    else:
        pyautogui.click(button=button, clicks=clicks, interval=interval)
        pos = pyautogui.position()
        rx, ry = int(pos.x), int(pos.y)
    return {"x": rx, "y": ry, "button": button, "clicks": clicks, "interval": interval}


def type_text(
    text: str,
) -> dict[str, Any]:
    """Paste text via clipboard (Ctrl+V) at the current keyboard focus.

    Does not move or click the mouse. Clears the clipboard afterward and does
    not restore previous contents.
    """
    # root = tk.Tk()
    pyperclip.copy(text)
    sleep(0.5)
    pyautogui.hotkey("ctrl", "v")
    sleep(0.05)
    return {
        "text": text,
        "effective_mode": "paste",
    }


def hotkey(keys: list[str] | str) -> dict[str, Any]:
    """Press a key combination."""
    if isinstance(keys, str):
        keys = [keys]
    normalized_keys = [_normalize_hotkey_token(key) for key in keys]
    normalized_keys = [key for key in normalized_keys if key]
    if not normalized_keys:
        raise ValueError("keys must contain at least one key")
    pyautogui.hotkey(*normalized_keys)
    return {"keys": normalized_keys}


def move(x: int, y: int, duration: float = 0.0) -> dict[str, Any]:
    """Move the mouse to a screen coordinate."""
    pyautogui.moveTo(x=x, y=y, duration=duration)
    return {"x": x, "y": y, "duration": duration}


def wait(seconds: float) -> dict[str, Any]:
    """Pause execution for a number of seconds."""
    sleep(seconds)
    return {"seconds": seconds}


keyboard_keys_map = {
    "Windows": "win",
}

def key_press(key: str) -> dict[str, Any]:
    """Press and release a single key."""
    token = _normalize_hotkey_token(key)
    token = keyboard_keys_map.get(token, token)
    if token not in pyautogui.KEYBOARD_KEYS:
        raise ValueError(f"Invalid key: {token}")
    pyautogui.press(token)
    return {"key": token}


def write_text(
    text: str,
    coordinate: list[int],
    interval: float = 0.0,
) -> dict[str, Any]:
    """Click to focus, then type raw characters with pyautogui.write."""
    if len(coordinate) != 2:
        raise ValueError("coordinate must be [x, y]")
    x, y = coordinate
    pyautogui.click(x=x, y=y, button="left")
    pyautogui.write(text, interval=interval)
    return {
        "text": text,
        "interval": interval,
        "clicked_coordinate": {"x": x, "y": y, "button": "left"},
        "effective_mode": "write",
    }


def drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float = 0.5,
    button: str = "left",
) -> dict[str, Any]:
    """Drag from (x1,y1) to (x2,y2)."""
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=duration, button=button)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "duration": duration,
        "button": button,
    }


def screenshot_to_file(path: str | None = None) -> dict[str, Any]:
    """Capture the active Eye monitor; saves PNG to path or a temp file."""
    if not path:
        dest = Path(mkdtemp()) / "screenshot.png"
    else:
        dest = Path(path)
        if not dest.suffix:
            dest = dest.with_suffix(".png")
    capture_active_monitor_to_file(dest)
    return {"path": str(dest)}


def cursor_position() -> dict[str, Any]:
    pos = pyautogui.position()
    return {"x": int(pos.x), "y": int(pos.y)}


def mouse_down(x: int | None = None, y: int | None = None, button: str = "left") -> dict[str, Any]:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.mouseDown(button=button)
    out: dict[str, Any] = {"button": button}
    if x is not None:
        out["x"] = x
    if y is not None:
        out["y"] = y
    return out


def mouse_up(x: int | None = None, y: int | None = None, button: str = "left") -> dict[str, Any]:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.mouseUp(button=button)
    out: dict[str, Any] = {"button": button}
    if x is not None:
        out["x"] = x
    if y is not None:
        out["y"] = y
    return out


def scroll_at(clicks: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)
    out: dict[str, Any] = {"clicks": clicks}
    if x is not None:
        out["x"] = x
    if y is not None:
        out["y"] = y
    return out


def hold_key_down(key: str, seconds: float) -> dict[str, Any]:
    token = _normalize_hotkey_token(key)
    pyautogui.keyDown(token)
    sleep(seconds)
    pyautogui.keyUp(token)
    return {"key": token, "seconds": seconds}


def _list_windows_with_titles() -> list[tuple[Any, str]]:
    out: list[tuple[Any, str]] = []
    for w in gw.getAllWindows():
        title = (w.title or "").strip()
        if title:
            out.append((w, title))
    return out


def _parse_json_object_from_llm(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


async def _ollama_pick_window_index(
    user_query: str,
    candidates: list[tuple[Any, str]],
    instruction: str = "",
) -> int:
    if not candidates:
        raise ValueError("no candidate windows to choose from")
    lines = [f"{i}: {title}" for i, (_, title) in enumerate(candidates)]
    extra = (instruction or "").strip()
    context_block = (
        f"Additional context from the operator (use to disambiguate):\n{extra}\n\n"
        if extra
        else ""
    )
    prompt = (
        "You pick exactly one desktop window to maximize.\n"
        "The user wants this window (natural-language or partial title):\n"
        f"{user_query!r}\n\n"
        f"{context_block}"
        "From the numbered list, choose the single best match. "
        "Prefer the main application window over tiny dialogs or tool windows when unclear.\n"
        "Return JSON only in this exact shape: {\"index\": <int>}\n"
        "Use the 0-based index from the list.\n\n"
        "Windows:\n"
        + "\n".join(lines)
    )
    msg = await _ollama.chat_messages(
        model=_settings.brain_lm,
        messages=[{"role": "user", "content": prompt}],
        use_tools=False,
    )
    content = (msg.content or "").strip()
    out = _parse_json_object_from_llm(content)
    idx = int(out["index"])
    if idx < 0 or idx >= len(candidates):
        raise ValueError(
            f"ollama returned index {idx} but valid range is 0..{len(candidates) - 1}"
        )
    return idx


async def maximize_window(
    window_title_contains: str,
    instruction: str = "",
) -> dict[str, Any]:
    """
    Bring a top-level window to the foreground and maximize it.

    First tries case-insensitive substring match on window titles. If exactly one
    window matches, it is used. If none or several match, asks Ollama (brain_lm)
    to pick the best index from the relevant candidate list. For those Ollama
    calls, ``instruction`` (if non-empty) is included in the prompt as extra
    disambiguation context.
    """
    needle = (window_title_contains or "").strip()
    if not needle:
        raise ValueError("window_title_contains must be a non-empty string")

    nlow = needle.lower()
    substring_matches: list[tuple[Any, str]] = []
    for w in gw.getAllWindows():
        title = (w.title or "").strip()
        if not title or nlow not in title.lower():
            continue
        substring_matches.append((w, title))

    selection_mode: str
    w: Any
    title: str

    if len(substring_matches) == 1:
        w, title = substring_matches[0]
        selection_mode = "substring_unique"
    elif len(substring_matches) == 0:
        candidates = _list_windows_with_titles()
        if not candidates:
            raise ValueError("no windows with non-empty titles found")
        idx = await _ollama_pick_window_index(needle, candidates, instruction=instruction)
        w, title = candidates[idx]
        selection_mode = "ollama_no_substring_match"
    else:
        idx = await _ollama_pick_window_index(needle, substring_matches, instruction=instruction)
        w, title = substring_matches[idx]
        selection_mode = "ollama_disambiguate"

    try:
        w.activate()
    except Exception:
        pass
    if w.isMinimized:
        w.restore()
    w.maximize()
    out: dict[str, Any] = {
        "window_title_contains": needle,
        "matched_title": title,
        "status": "maximized",
        "selection_mode": selection_mode,
    }
    if (instruction or "").strip():
        out["instruction"] = instruction.strip()
    return out


def minimize_all_windows() -> dict[str, Any]:
    """
    Minimize all top-level windows that are currently not minimized.
    """
    total_with_titles = 0
    minimized_titles: list[str] = []
    errors: list[str] = []

    for w in gw.getAllWindows():
        title = (w.title or "").strip()
        if not title:
            continue
        total_with_titles += 1
        if w.isMinimized:
            continue
        try:
            w.minimize()
            minimized_titles.append(title)
        except Exception as exc:
            errors.append(f"{title}: {type(exc).__name__}: {exc}")

    out: dict[str, Any] = {
        "status": "success" if not errors else "partial_success",
        "total_windows_with_titles": total_with_titles,
        "minimized_count": len(minimized_titles),
        "already_minimized_count": total_with_titles - len(minimized_titles) - len(errors),
    }
    if minimized_titles:
        out["minimized_titles"] = minimized_titles
    if errors:
        out["errors"] = errors
    return out


def zoom_scroll(scroll_clicks: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.keyDown("ctrl")
    try:
        pyautogui.scroll(scroll_clicks)
    finally:
        pyautogui.keyUp("ctrl")
    out: dict[str, Any] = {"scroll_clicks": scroll_clicks, "modifier": "ctrl"}
    if x is not None:
        out["x"] = x
    if y is not None:
        out["y"] = y
    return out


def detect_objects(image_path: str) -> dict[str, Any]:
    """
    Lightweight object-detection placeholder with image metadata.

    This validates the image path and returns dimensions so callers have
    actionable context even when a trained detector is not configured yet.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required for detect_objects") from exc

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unable to decode image: {image_path}")

    height, width = image.shape[:2]
    return {
        "status": "success",
        "image_path": image_path,
        "image_size": {"width": int(width), "height": int(height)},
        "detected": [],
    }
