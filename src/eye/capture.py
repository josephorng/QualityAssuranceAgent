from __future__ import annotations

from pathlib import Path

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


def capture_all_screens_to_file(dest: Path) -> None:
    """Capture the full virtual desktop (all monitors) into one image with the mouse cursor."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        overlay_mouse_cursor(img, int(monitor["left"]), int(monitor["top"]))
    img.save(dest)
    try:
        get_run_state_manager().log_info(f"eye capture saved path={dest} monitor_index=0")
    except RuntimeError:
        pass


def capture_monitor_to_file(
    dest: Path,
    monitor_index: int,
    *,
    include_cursor: bool = False,
) -> int:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        idx = resolve_monitor_index(sct, monitor_index)
        monitor = sct.monitors[idx]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        if include_cursor:
            overlay_mouse_cursor(img, int(monitor["left"]), int(monitor["top"]))
    img.save(dest)
    try:
        get_run_state_manager().log_info(f"eye capture saved path={dest} monitor_index={idx}")
    except RuntimeError:
        pass
    return idx


def capture_active_monitor_to_file(dest: Path, default_monitor_index: int = 1) -> int:
    requested = active_monitor_index(default_monitor_index)
    return capture_monitor_to_file(dest=dest, monitor_index=requested)
