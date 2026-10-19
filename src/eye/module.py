from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from src.common.models import EyeEvent
from src.common.monitor_prompt import selected_eye_monitor_indices
from src.common.run_state import get_run_state_manager, ts_name
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings
from src.eye.capture import (
    active_monitor_index,
    capture_all_screens_to_file,
    capture_monitor_to_file,
    grab_monitor_image,
    monitor_details as list_monitor_details,
    resolve_monitor_index,
)


class EyeModule:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.run_root, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.run_id, self.run_root.name)
        self.active_monitor_index = active_monitor_index(1)
        self._pending_saves: list[asyncio.Task[None]] = []
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

    def _schedule_png_save(self, save_fn: Callable[[], None]) -> None:
        """Run ``save_fn`` (PNG encode) on a worker thread; track until flushed."""

        async def _run() -> None:
            await asyncio.to_thread(save_fn)

        self._pending_saves.append(asyncio.create_task(_run()))

    async def flush_pending_saves(self) -> None:
        """Await all in-flight eye PNG writes. Safe to call when none are pending."""
        if not self._pending_saves:
            return
        pending = self._pending_saves
        self._pending_saves = []
        results = await asyncio.gather(*pending, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if not errors:
            return
        for exc in errors:
            self.manager.log_error(f"Eye PNG save failed: {exc}")
        raise errors[0]

    def _capture_selected_monitors_to_file(self, image_path: Path) -> None:
        """Grab the selected monitor(s) and schedule PNG encode in the background."""
        monitor_indices = selected_eye_monitor_indices(self.active_monitor_index)
        if len(monitor_indices) > 1:
            save_fn = capture_all_screens_to_file(image_path, wait_for_save=False)
            assert save_fn is not None
            self._schedule_png_save(save_fn)
            return
        result = capture_monitor_to_file(
            dest=image_path,
            monitor_index=monitor_indices[0],
            include_cursor=True,
            wait_for_save=False,
        )
        assert isinstance(result, tuple)
        resolved_idx, save_fn = result
        self.active_monitor_index = resolved_idx
        self._schedule_png_save(save_fn)

    async def capture_once(self) -> EyeEvent:
        await self.flush_pending_saves()
        paths = self.manager.require_paths()
        image_name = f"{ts_name()}.png"
        image_path = paths.eye_dir / image_name
        self._capture_selected_monitors_to_file(image_path)
        event = EyeEvent(
            screenshot_name=image_name,
            screenshot_path=str(image_path),
            similarity_to_previous=None,
        )
        self.manager.log_info(f"Eye captured {image_name}")
        return event

    async def capture_separated_images(self) -> list[str]:
        """
        Capture screenshot(s) for the Brain matching the user's monitor selection.

        One physical monitor → single-monitor image; multiple selected → combined
        virtual desktop. Per-monitor YOLO/OCR images are written separately (``yolo_ocr/``).

        PNG encode runs in the background; callers should ``await flush_pending_saves()``
        before relying on the file bytes (e.g. after tools / at step end).
        """
        await self.flush_pending_saves()
        paths = self.manager.require_paths()
        image_name = f"{ts_name()}.png"
        image_path = paths.eye_dir / image_name
        self._capture_selected_monitors_to_file(image_path)
        self.manager.log_info(f"Eye captured brain screenshot {image_name}")
        return [str(image_path)]
