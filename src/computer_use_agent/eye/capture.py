from __future__ import annotations

from pathlib import Path

from PIL import ImageGrab

from computer_use_agent.core.session import utc_stamp


def capture_screenshot(eye_dir: Path) -> Path:
    image = ImageGrab.grab(all_screens=True)
    path = eye_dir / f"{utc_stamp()}.png"
    image.save(path)
    return path
