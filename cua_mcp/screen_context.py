"""Capture screenshot + YOLO/OCR context for smart-mode planner/verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cua_mcp.select_mouse_target import _collect_monitor_detections
from cua_mcp.select_ui_element import (
    UiDetection,
    _format_ui_candidates_text,
    _sort_detections_reading_order,
)
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END
from src.common.io_utils import imread_bgr
from src.common.monitor_prompt import selected_eye_monitor_indices
from src.common.run_state import get_run_state_manager, ts_name
from src.eye.capture import capture_monitor_to_file


@dataclass
class ScreenContext:
    """Fresh per-monitor screenshots plus normalized OCR/element rows."""

    screenshot_paths: list[str] = field(default_factory=list)
    ocr_text: str = ""
    candidate_count: int = 0
    monitor_indices: list[int] = field(default_factory=list)
    candidates: list[UiDetection] = field(default_factory=list, repr=False)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "screenshot_paths": list(self.screenshot_paths),
            "candidate_count": self.candidate_count,
            "monitor_indices": list(self.monitor_indices),
            "ocr_text_chars": len(self.ocr_text),
        }


async def capture_screen_context(
    *,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    include_geometry: bool = True,
) -> ScreenContext:
    """
    Capture selected monitors into ``yolo_ocr/``, run YOLO+OCR, and format candidate text.

    Returns paths (not image bytes) plus a text summary suitable for planner/verifier prompts.
    """
    paths = get_run_state_manager().require_paths()
    monitor_indices = selected_eye_monitor_indices()
    stamp = ts_name()
    image_paths: list[str] = []
    captured: list[tuple[int, np.ndarray]] = []

    for monitor_index in monitor_indices:
        name = f"{stamp}_smart_mon{monitor_index}.png"
        out = paths.yolo_ocr_dir / name
        capture_monitor_to_file(out, monitor_index)
        image_path = str(out.resolve())
        image_paths.append(image_path)
        bgr = imread_bgr(image_path)
        if bgr is None:
            continue
        captured.append((monitor_index, bgr))

    detections = _sort_detections_reading_order(
        _collect_monitor_detections(
            captured,
            yolo_conf_threshold=yolo_conf_threshold,
        )
    )
    ocr_text = (
        _format_ui_candidates_text(detections, include_geometry=include_geometry)
        if detections
        else "(no OCR/YOLO candidates)"
    )
    return ScreenContext(
        screenshot_paths=image_paths,
        ocr_text=ocr_text,
        candidate_count=len(detections),
        monitor_indices=list(monitor_indices),
        candidates=detections,
    )
