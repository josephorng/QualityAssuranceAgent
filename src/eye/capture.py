from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mss
import pyautogui
from PIL import Image, ImageDraw

from src.common.monitor_prompt import read_eye_monitor_index_from_env
from src.common.run_state import get_run_state_manager


def resolve_monitor_index(sct: mss.mss, requested_index: int) -> int:
    max_index = len(sct.monitors) - 1
    if max_index < 0:
        return 0
    if requested_index < 0:
        return 0
    if requested_index > max_index:
        return max_index
    return requested_index


def monitor_details() -> list[dict[str, int | str]]:
    with mss.mss() as sct:
        details: list[dict[str, int | str]] = []
        for idx in range(len(sct.monitors)):
            monitor = sct.monitors[idx]
            entry: dict[str, int | str] = {
                "index": idx,
                "left": int(monitor["left"]),
                "top": int(monitor["top"]),
                "width": int(monitor["width"]),
                "height": int(monitor["height"]),
            }
            if idx == 0:
                entry["name"] = "all_screens"
            details.append(entry)
        return details


def active_monitor_index(default: int = 1) -> int:
    return read_eye_monitor_index_from_env(default)


def active_monitor_offset(monitor_index: int | None = None) -> tuple[int, int]:
    with mss.mss() as sct:
        requested = active_monitor_index() if monitor_index is None else monitor_index
        idx = resolve_monitor_index(sct, requested)
        mon = sct.monitors[idx]
        return int(mon["left"]), int(mon["top"])


def overlay_mouse_cursor(img: Image.Image, origin_left: int = 0, origin_top: int = 0) -> Image.Image:
    """Draw the current mouse cursor onto a desktop screenshot."""
    try:
        pos = pyautogui.position()
    except Exception:
        return img

    local_x = int(pos.x) - int(origin_left)
    local_y = int(pos.y) - int(origin_top)
    if not (0 <= local_x < img.width and 0 <= local_y < img.height):
        return img

    draw = ImageDraw.Draw(img)
    x, y = local_x, local_y
    arrow = [
        (x, y),
        (x, y + 16),
        (x + 4, y + 12),
        (x + 7, y + 19),
        (x + 10, y + 18),
        (x + 7, y + 12),
        (x + 12, y + 11),
    ]
    draw.polygon(arrow, fill="white", outline="black")
    return img


def grab_monitor_image(monitor_index: int) -> Image.Image:
    with mss.mss() as sct:
        idx = resolve_monitor_index(sct, monitor_index)
        monitor = sct.monitors[idx]
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def grab_monitor_bgr(
    monitor_index: int,
    *,
    include_cursor: bool = False,
) -> tuple[int, Any]:
    """Grab one monitor as a BGR ``uint8`` array without a disk round-trip.

    Returns ``(resolved_monitor_index, bgr)``.
    """
    import cv2
    import numpy as np

    with mss.mss() as sct:
        idx = resolve_monitor_index(sct, monitor_index)
        monitor = sct.monitors[idx]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        if include_cursor:
            overlay_mouse_cursor(img, int(monitor["left"]), int(monitor["top"]))
    rgb = np.asarray(img, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return idx, bgr


def _log_eye_capture_saved(dest: Path, monitor_index: int) -> None:
    try:
        get_run_state_manager().log_info(
            f"eye capture saved path={dest} monitor_index={monitor_index}"
        )
    except RuntimeError:
        pass


def save_image_png(img: Image.Image, dest: Path, *, monitor_index: int) -> None:
    """Encode and write ``img`` as PNG, then log the save."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    _log_eye_capture_saved(dest, monitor_index)


def grab_all_screens_image() -> Image.Image:
    """Grab the full virtual desktop (mss monitor 0) with the mouse cursor overlaid."""
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        overlay_mouse_cursor(img, int(monitor["left"]), int(monitor["top"]))
        return img


def grab_monitor_image_to_save(
    monitor_index: int,
    *,
    include_cursor: bool = False,
) -> tuple[int, Image.Image]:
    """Grab one monitor as RGB, optionally with cursor. Returns ``(resolved_index, img)``."""
    with mss.mss() as sct:
        idx = resolve_monitor_index(sct, monitor_index)
        monitor = sct.monitors[idx]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        if include_cursor:
            overlay_mouse_cursor(img, int(monitor["left"]), int(monitor["top"]))
        return idx, img


def capture_all_screens_to_file(
    dest: Path,
    *,
    wait_for_save: bool = True,
) -> Callable[[], None] | None:
    """Capture the full virtual desktop into one image with the mouse cursor.

    When ``wait_for_save`` is True (default), writes PNG synchronously and returns
    None. When False, returns a zero-arg callable that performs the PNG save so
    callers can run it in the background.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = grab_all_screens_image()
    if wait_for_save:
        save_image_png(img, dest, monitor_index=0)
        return None

    def _save() -> None:
        save_image_png(img, dest, monitor_index=0)

    return _save


def capture_monitor_to_file(
    dest: Path,
    monitor_index: int,
    *,
    include_cursor: bool = False,
    wait_for_save: bool = True,
) -> int | tuple[int, Callable[[], None]]:
    """Capture one monitor to ``dest``.

    When ``wait_for_save`` is True (default), writes PNG synchronously and returns
    the resolved monitor index. When False, returns ``(resolved_index, save_fn)``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    idx, img = grab_monitor_image_to_save(monitor_index, include_cursor=include_cursor)
    if wait_for_save:
        save_image_png(img, dest, monitor_index=idx)
        return idx

    def _save() -> None:
        save_image_png(img, dest, monitor_index=idx)

    return idx, _save


def capture_active_monitor_to_file(dest: Path, default_monitor_index: int = 1) -> int:
    requested = active_monitor_index(default_monitor_index)
    result = capture_monitor_to_file(dest=dest, monitor_index=requested)
    assert isinstance(result, int)
    return result
