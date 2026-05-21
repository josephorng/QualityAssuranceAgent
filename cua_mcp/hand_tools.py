from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import mkdtemp
from time import sleep
from typing import Any
# import tkinter as tk
import pyperclip

import pyautogui
import pygetwindow as gw
from src.common.llm_factory import get_llm_client
from src.common.settings import load_settings
from src.eye.capture import capture_active_monitor_to_file


def _normalize_hotkey_token(key: str) -> str:
    """Normalize odd wrappers that can appear in model output."""
    cleaned = key.strip()
    cleaned = cleaned.replace('<|"|>', "")
    cleaned = cleaned.strip("\"'").strip()
    # pyautogui key tokens are lowercase (e.g. "f6", "enter", "pagedown")
    return cleaned.lower()


KEY_ALIASES = {
    "control": "ctrl",
    "command": "win",
    "cmd": "win",
    "windows": "win",
    "option": "alt",
    "return": "enter",
    "esc": "escape",
}


def _canonicalize_key(key: str) -> str:
    token = _normalize_hotkey_token(key)
    return KEY_ALIASES.get(token, token)


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
        raw = keys.strip()
        parsed_keys: Any = None
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed_keys = json.loads(raw)
            except json.JSONDecodeError:
                parsed_keys = None
        if isinstance(parsed_keys, list):
            keys = [str(item) for item in parsed_keys]
        else:
            keys = [keys]
    normalized_keys: list[str] = []
    for key in keys:
        token = _canonicalize_key(key)
        if not token:
            continue
        # Accept "ctrl+a" and similar compact chord syntax.
        if "+" in token:
            parts = [_canonicalize_key(part) for part in token.split("+") if part.strip()]
            normalized_keys.extend(parts)
        else:
            normalized_keys.append(token)
    if not normalized_keys:
        raise ValueError("keys must contain at least one key")
    invalid = [k for k in normalized_keys if k not in pyautogui.KEYBOARD_KEYS]
    if invalid:
        raise ValueError(f"Invalid hotkey keys: {invalid}")
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


keyboard_keys_map = KEY_ALIASES

def key_press(key: str) -> dict[str, Any]:
    """Press and release a single key."""
    token = _canonicalize_key(key)
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


def _pyautogui_vertical_scroll_amount(clicks: int) -> int:
    """Convert logical wheel detents to the value PyAutoGUI passes to the OS.

    On Windows, vertical wheel ``dwData`` is in multiples of WHEEL_DELTA (120).
    PyAutoGUI forwards the integer unchanged, so small values like 4 are a tiny
    fraction of one notch and look like a no-op.
    """
    n = int(clicks)
    if sys.platform == "win32":
        return n * 120
    return n


def scroll_at(clicks: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    # Positive MCP `clicks` = scroll document/view downward (往下滑); PyAutoGUI uses
    # the opposite sign on Windows and X11.
    pyautogui.scroll(-_pyautogui_vertical_scroll_amount(clicks))
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


async def _ollama_pick_window_indices(
    user_query: str,
    candidates: list[tuple[Any, str]],
    instruction: str = "",
    action: str = "maximize",
) -> list[int]:
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
        f"You select one or more desktop windows to {action}.\n"
        "The user wants these windows (natural-language or partial title):\n"
        f"{user_query!r}\n\n"
        f"{context_block}"
        "From the numbered list, choose every window that matches the user's intent. "
        "Use a single-element list when only one window is appropriate. "
        "Prefer main application windows over tiny dialogs or tool windows when unclear.\n"
        "Return JSON only in this exact shape: {\"indices\": [<int>, ...]}\n"
        "Use 0-based indices from the list.\n\n"
        "Windows:\n"
        + "\n".join(lines)
    )
    msg = await get_llm_client().chat_messages(
        model=load_settings().brain_lm,
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )
    content = (msg.content or "").strip()
    out = _parse_json_object_from_llm(content)
    raw = out.get("indices")
    if not isinstance(raw, list):
        raise ValueError("ollama JSON must contain a non-empty list field \"indices\"")
    indices: list[int] = []
    seen: set[int] = set()
    for item in raw:
        idx = int(item)
        if idx < 0 or idx >= len(candidates):
            raise ValueError(
                f"ollama returned index {idx} but valid range is 0..{len(candidates) - 1}"
            )
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    if not indices:
        raise ValueError("ollama returned empty indices list")
    return indices


async def _select_target_windows(
    window_title_contains: str,
    instruction: str,
    action: str,
) -> tuple[list[tuple[Any, str]], str]:
    """
    Return windows to act on and how they were chosen.

    Single substring match -> no LLM. Zero or multiple substring matches ->
    Ollama returns one or more indices into the relevant candidate list.
    """
    needle = (window_title_contains or "").strip()
    if not needle:
        raise ValueError("window_title_contains must be a non-empty string")

    nlow = needle.lower()
    substring_matches: list[tuple[Any, str]] = []
    for w in gw.getAllWindows():
        title = (w.title or "").strip()
        if (not title or nlow not in title.lower()) and nlow != "all":
            continue
        substring_matches.append((w, title))

    if len(substring_matches) == 1:
        return [substring_matches[0]], "substring_unique"
    if len(substring_matches) == 0:
        candidates = _list_windows_with_titles()
        if not candidates:
            raise ValueError("no windows with non-empty titles found")
        idxs = await _ollama_pick_window_indices(
            needle,
            candidates,
            instruction=instruction,
            action=action,
        )
        return [candidates[i] for i in idxs], "ollama_no_substring_match"
    idxs = await _ollama_pick_window_indices(
        needle,
        substring_matches,
        instruction=instruction,
        action=action,
    )
    return [substring_matches[i] for i in idxs], "ollama_disambiguate"


async def maximize_windows(
    window_title_contains: str,
    instruction: str = "",
) -> dict[str, Any]:
    """
    Bring one or more top-level windows to the foreground and maximize them.

    First tries case-insensitive substring match on window titles. If exactly one
    window matches, it is used. If none or several match, asks Ollama (brain_lm)
    to pick one or more indices from the relevant candidate list. For those
    Ollama calls, ``instruction`` (if non-empty) is included in the prompt as extra
    disambiguation context.
    """
    needle = (window_title_contains or "").strip()
    if not needle:
        raise ValueError("window_title_contains must be a non-empty string")

    targets, selection_mode = await _select_target_windows(
        window_title_contains,
        instruction,
        action="maximize",
    )

    target_rows: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    for w, title in targets:
        try:
            try:
                w.activate()
            except Exception:
                pass
            if w.isMinimized:
                w.restore()
            w.maximize()
            target_rows.append({"matched_title": title, "status": "maximized"})
            succeeded += 1
        except Exception as exc:
            failed += 1
            target_rows.append(
                {
                    "matched_title": title,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    out: dict[str, Any] = {
        "window_title_contains": needle,
        "selection_mode": selection_mode,
        "matched_count": len(targets),
        "succeeded_count": succeeded,
        "failed_count": failed,
        "targets": target_rows,
        "status": "success" if failed == 0 else "partial_success",
    }
    if (instruction or "").strip():
        out["instruction"] = instruction.strip()
    return out


async def close_windows(
    window_title_contains: str,
    instruction: str = "",
) -> dict[str, Any]:
    """
    Close one or more top-level windows whose titles best match the query.

    First tries case-insensitive substring match on window titles. If exactly one
    window matches, it is used. If none or several match, asks Ollama (brain_lm)
    to pick one or more indices from the relevant candidate list.
    """
    needle = (window_title_contains or "").strip()
    if not needle:
        raise ValueError("window_title_contains must be a non-empty string")

    targets, selection_mode = await _select_target_windows(
        window_title_contains,
        instruction,
        action="close",
    )

    target_rows: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    for w, title in targets:
        try:
            try:
                w.activate()
            except Exception:
                pass
            if w.isMinimized:
                w.restore()
            w.close()
            target_rows.append({"matched_title": title, "status": "closed"})
            succeeded += 1
        except Exception as exc:
            failed += 1
            target_rows.append(
                {
                    "matched_title": title,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    out: dict[str, Any] = {
        "window_title_contains": needle,
        "selection_mode": selection_mode,
        "matched_count": len(targets),
        "succeeded_count": succeeded,
        "failed_count": failed,
        "targets": target_rows,
        "status": "success" if failed == 0 else "partial_success",
    }
    if (instruction or "").strip():
        out["instruction"] = instruction.strip()
    return out


async def minimize_windows(
    window_title_contains: str,
    instruction: str = "",
) -> dict[str, Any]:
    """
    Minimize one or more top-level windows selected like maximize/close.

    Windows already minimized are skipped and reported separately.
    """
    needle = (window_title_contains or "").strip()
    if not needle:
        raise ValueError("window_title_contains must be a non-empty string")

    targets, selection_mode = await _select_target_windows(
        window_title_contains,
        instruction,
        action="minimize",
    )

    target_rows: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    already_minimized = 0
    for w, title in targets:
        if w.isMinimized:
            already_minimized += 1
            target_rows.append({"matched_title": title, "status": "already_minimized"})
            continue
        try:
            try:
                w.activate()
            except Exception:
                pass
            w.minimize()
            target_rows.append({"matched_title": title, "status": "minimized"})
            succeeded += 1
        except Exception as exc:
            failed += 1
            target_rows.append(
                {
                    "matched_title": title,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    out: dict[str, Any] = {
        "window_title_contains": needle,
        "selection_mode": selection_mode,
        "matched_count": len(targets),
        "succeeded_count": succeeded,
        "failed_count": failed,
        "already_minimized_count": already_minimized,
        "targets": target_rows,
        "status": "success" if failed == 0 else "partial_success",
    }
    if (instruction or "").strip():
        out["instruction"] = instruction.strip()
    return out


def zoom_scroll(scroll_clicks: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.keyDown("ctrl")
    try:
        pyautogui.scroll(_pyautogui_vertical_scroll_amount(scroll_clicks))
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
