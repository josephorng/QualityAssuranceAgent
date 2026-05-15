from __future__ import annotations

from src.common.models import EyeEvent
from src.common.run_state import get_run_state_manager, ts_name
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings
from src.common.monitor_prompt import read_eye_monitor_indices_from_env
from src.eye.capture import (
    active_monitor_index,
    capture_monitor_to_file,
    grab_monitor_image,
    monitor_details as list_monitor_details,
    resolve_monitor_index,
)

import os


class EyeModule:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.run_root, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.run_id, self.run_root.name)
        self.active_monitor_index = active_monitor_index(1)
        self.manager.log_info(f"Eye module initialized run_id={self.run_id}")

    def monitor_details(self) -> list[dict[str, int | str]]:
        return list_monitor_details()

    def resolve_monitor_index(self, requested_index: int) -> int:
        import mss

        with mss.mss() as sct:
            return resolve_monitor_index(sct, requested_index)

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

    def _grab_screenshot(self):
        self.active_monitor_index = self.resolve_monitor_index(self.active_monitor_index)
        img = grab_monitor_image(self.active_monitor_index)
        self.manager.log_info(
            f"Eye grabbed screenshot monitor={self.active_monitor_index} size={img.size}"
        )
        return img

    async def capture_once(self) -> EyeEvent:
        paths = self.manager.require_paths()
        image_name = f"{ts_name()}.png"
        image_path = paths.eye_dir / image_name
        self.active_monitor_index = capture_monitor_to_file(
            dest=image_path,
            monitor_index=self.active_monitor_index,
        )
        event = EyeEvent(
            screenshot_name=image_name,
            screenshot_path=str(image_path),
            similarity_to_previous=None,
        )
        self.manager.log_info(f"Eye captured {image_name}")
        return event

    async def capture_separated_images(self) -> list[str]:
        """
        Capture verification screenshots and return image paths.

        If ``EYE_MONITOR_INDICES`` is set (comma-separated), capture each index in order.
        Else if ``EYE_MONITOR_INDEX`` is non-zero, capture a single target.
        Else (index 0): capture one screenshot per physical monitor (mss indexes > 0).
        """
        indices_override = read_eye_monitor_indices_from_env()
        if indices_override is not None:
            original_monitor_index = self.active_monitor_index
            image_paths: list[str] = []
            try:
                for monitor_index in indices_override:
                    self.set_capture_target(monitor_index)
                    monitor_event = await self.capture_once()
                    image_paths.append(monitor_event.screenshot_path)
            finally:
                self.set_capture_target(original_monitor_index)
            return image_paths

        eye_monitor_index = os.environ.get("EYE_MONITOR_INDEX", "0").strip()
        if eye_monitor_index != "0":
            self.set_capture_target(int(eye_monitor_index))
            event = await self.capture_once()
            return [event.screenshot_path]

        monitor_info = self.monitor_details()
        physical_monitor_indexes = [
            int(detail["index"])
            for detail in monitor_info
            if int(detail.get("index", 0)) > 0
        ]

        original_monitor_index = self.active_monitor_index
        image_paths: list[str] = []
        try:
            if physical_monitor_indexes:
                for monitor_index in physical_monitor_indexes:
                    self.set_capture_target(monitor_index)
                    monitor_event = await self.capture_once()
                    image_paths.append(monitor_event.screenshot_path)
            else:
                fallback_event = await self.capture_once()
                image_paths.append(fallback_event.screenshot_path)
        finally:
            self.set_capture_target(original_monitor_index)

        return image_paths

