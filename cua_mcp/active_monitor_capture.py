"""Capture the active Eye monitor with mss (same semantics as EyeModule)."""

from __future__ import annotations

from pathlib import Path

import mss
from PIL import Image

from src.common.monitor_prompt import read_eye_monitor_index_from_env
from src.common.run_state import get_run_state_manager


def _resolve_monitor_index(sct: mss.mss, requested_index: int) -> int:
    max_index = len(sct.monitors) - 1
    if max_index < 0:
        return 0
    if requested_index < 0:
        return 0
    if requested_index > max_index:
        return max_index
    return requested_index


def active_monitor_index() -> int:
    """Monitor index from ``EYE_MONITOR_INDEX`` (same as EyeModule startup)."""
    return read_eye_monitor_index_from_env(1)


def active_monitor_offset() -> tuple[int, int]:
    """Global left/top of the monitor used for capture (matches ``capture_active_monitor_to_file``)."""
    with mss.mss() as sct:
        idx = _resolve_monitor_index(sct, active_monitor_index())
        mon = sct.monitors[idx]
        return int(mon["left"]), int(mon["top"])


def capture_active_monitor_to_file(dest: Path) -> None:
    """Grab the active monitor and save as PNG (same capture geometry as EyeModule._grab_screenshot)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        idx = _resolve_monitor_index(sct, active_monitor_index())
        monitor = sct.monitors[idx]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    img.save(dest)
    try:
        get_run_state_manager().log_info(f"active_monitor_capture saved path={dest} monitor_index={idx}")
    except RuntimeError:
        pass
