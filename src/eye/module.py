from __future__ import annotations

from src.common.models import EyeEvent
from src.common.run_state import get_run_state_manager, ts_name
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings
from src.eye.capture import (
    active_monitor_index,
    capture_monitor_to_file,
    grab_monitor_image,
    monitor_details as list_monitor_details,
    resolve_monitor_index,
)


class EyeModule:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.run_root, self.task_input, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.task_input, self.run_root.name)
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

