from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Any

import pyautogui


def _normalize_hotkey_token(key: str) -> str:
    """Normalize odd wrappers that can appear in model output."""
    cleaned = key.strip()
    cleaned = cleaned.replace('<|"|>', "")
    return cleaned.strip("\"'")


def click(x: int, y: int, button: str = "left") -> dict[str, Any]:
    """Click a screen coordinate and return executed arguments."""
    pyautogui.click(x=x, y=y, button=button)
    return {"x": x, "y": y, "button": button}


def type_text(
    text: str,
    coordinate: list[int],
    interval: float = 0.0,
) -> dict[str, Any]:
    """Click a coordinate to focus, then type text."""
    if len(coordinate) != 2:
        raise ValueError("coordinate must be [x, y]")
    x, y = coordinate
    pyautogui.click(x=x, y=y, button="left")
    click_result = {"x": x, "y": y, "button": "left"}

    pyautogui.typewrite(text, interval=interval)
    return {"text": text, "interval": interval, "clicked_coordinate": click_result}


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
