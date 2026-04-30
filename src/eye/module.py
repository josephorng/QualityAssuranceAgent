from __future__ import annotations

import mss
from PIL import Image

from src.common.models import EyeEvent
from src.common.monitor_prompt import read_eye_monitor_index_from_env
from src.common.run_state import get_run_state_manager, ts_name
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings


class EyeModule:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.run_root, self.task_input, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.task_input, self.run_root.name)
        self.active_monitor_index = read_eye_monitor_index_from_env(1)
        self.manager.log_info(f"Eye module initialized run_id={self.run_id}")

    def monitor_details(self) -> list[dict[str, int | str]]:
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

    def resolve_monitor_index(self, requested_index: int) -> int:
        with mss.mss() as sct:
            max_index = len(sct.monitors) - 1
            if max_index < 0:
                return 0
            if requested_index < 0:
                return 0
            if requested_index > max_index:
                return max_index
            return requested_index

    def set_capture_target(self, monitor_index: int) -> dict[str, object]:
        self.active_monitor_index = self.resolve_monitor_index(monitor_index)
        self.manager.log_info(f"Eye switched capture monitor={self.active_monitor_index}")
        return {
            "active_monitor_index": self.active_monitor_index,
            "monitors": self.monitor_details(),
        }

    def capture_targets(self) -> dict[str, object]:
        return {
            "active_monitor_index": self.active_monitor_index,
            "monitors": self.monitor_details(),
        }

    def _grab_screenshot(self) -> Image.Image:
        with mss.mss() as sct:
            self.active_monitor_index = self.resolve_monitor_index(self.active_monitor_index)
            monitor = sct.monitors[self.active_monitor_index]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            self.manager.log_info(
                f"Eye grabbed screenshot monitor={self.active_monitor_index} size={img.size}"
            )
            return img

    async def capture_once(self) -> EyeEvent:
        paths = self.manager.require_paths()
        image_name = f"{ts_name()}.png"
        image_path = paths.eye_dir / image_name
        screenshot_img = self._grab_screenshot()
        screenshot_img.save(image_path)
        event = EyeEvent(
            screenshot_name=image_name,
            screenshot_path=str(image_path),
            similarity_to_previous=None,
        )
        self.manager.log_info(f"Eye captured {image_name}")
        return event

