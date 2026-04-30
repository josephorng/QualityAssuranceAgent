from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from time import sleep
from typing import Any
import tkinter as tk

import pyautogui
from src.eye.capture import capture_active_monitor_to_file


def _normalize_hotkey_token(key: str) -> str:
    """Normalize odd wrappers that can appear in model output."""
    cleaned = key.strip()
    cleaned = cleaned.replace('<|"|>', "")
    return cleaned.strip("\"'")


def click(
    x: int,
    y: int,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.0,
) -> dict[str, Any]:
    """Click a screen coordinate and return executed arguments."""
    pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
    return {"x": x, "y": y, "button": button, "clicks": clicks, "interval": interval}


def type_text(
    text: str,
    coordinate: list[int],
    interval: float = 0.0,
) -> dict[str, Any]:
    """Click a coordinate to focus, then paste text from clipboard (Ctrl+V)."""
    if len(coordinate) != 2:
        raise ValueError("coordinate must be [x, y]")
    x, y = coordinate
    pyautogui.click(x=x, y=y, button="left")
    click_result = {"x": x, "y": y, "button": "left"}

    clipboard_before: str | None = None
    root = tk.Tk()
    root.withdraw()
    try:
        try:
            clipboard_before = root.clipboard_get()
        except tk.TclError:
            clipboard_before = None
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        pyautogui.hotkey("ctrl", "v")
    finally:
        root.clipboard_clear()
        if clipboard_before is not None:
            root.clipboard_append(clipboard_before)
        root.update()
        root.destroy()
    return {
        "text": text,
        "interval": interval,
        "clicked_coordinate": click_result,
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


def key_press(key: str) -> dict[str, Any]:
    """Press and release a single key."""
    token = _normalize_hotkey_token(key)
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
        path = str(Path(mkdtemp()) / "screenshot.png")
    capture_active_monitor_to_file(Path(path))
    return {"path": path}


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
