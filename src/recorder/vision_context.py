from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from src.common.io_utils import write_json
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent

_NEAREST_CANDIDATE_LIMIT = 8


def _local_cursor(event: RecordedEvent) -> tuple[int, int] | None:
    if event.cursor_xy is None:
        return None
    gx, gy = event.cursor_xy
    if event.monitor_offset is not None:
        ox, oy = event.monitor_offset
        return gx - ox, gy - oy
    return gx, gy


def _point_to_bbox_distance_sq(
    local_x: int,
    local_y: int,
    bbox: tuple[int, int, int, int],
) -> float:
    """Squared distance from a point to the closest point on an axis-aligned bbox."""
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    closest_x = min(max(local_x, x), x2)
    closest_y = min(max(local_y, y), y2)
    dx = local_x - closest_x
    dy = local_y - closest_y
    return float(dx * dx + dy * dy)


def _nearest_candidates(
    detections: list[UiDetection],
    local_x: int,
    local_y: int,
    *,
    limit: int = _NEAREST_CANDIDATE_LIMIT,
) -> list[UiDetection]:
    """Return up to ``limit`` detections sorted by point-to-bbox distance (closest first)."""
    if not detections:
        return []
    scored = sorted(
        detections,
        key=lambda d: _point_to_bbox_distance_sq(local_x, local_y, d.bbox),
    )
    return scored[:limit]


def _detection_to_dict(det: UiDetection) -> dict[str, Any]:
    return {
        "bbox": list(det.bbox),
        "center": [det.cx, det.cy],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "text": det.text,
        "icons": det.icons,
    }


def build_vision_context(
    event: RecordedEvent,
    *,
    run_dir: Path,
    persist_debug: bool = True,
) -> dict[str, Any]:
    """Run YOLO+OCR for pointer events; return context for the LLM."""
    empty: dict[str, Any] = {
        "used_vision": False,
        "candidate_text": "",
        "local_cursor": None,
        "candidates": [],
        "detection_count": 0,
    }
    if event.kind not in POINTER_EVENT_KINDS:
        return empty

    image_path = event.screenshot_path
    if not image_path or not Path(image_path).is_file():
        empty["local_cursor"] = _local_cursor(event)
        return empty

    bgr = cv2.imread(image_path)
    if bgr is None:
        empty["local_cursor"] = _local_cursor(event)
        return empty

    try:
        all_detections = _build_candidates_from_bgr(bgr)
    except RuntimeError:
        all_detections = []

    local = _local_cursor(event)
    nearest: list[UiDetection] = []
    candidate_text = ""
    if local is not None and all_detections:
        nearest = _nearest_candidates(all_detections, local[0], local[1])
        candidate_text = _format_ui_candidates_text(nearest)

    candidate_dicts = [_detection_to_dict(d) for d in nearest]
    payload: dict[str, Any] = {
        "event_index": event.index,
        "image_path": image_path,
        "cursor_xy": list(event.cursor_xy) if event.cursor_xy else None,
        "local_cursor": list(local) if local else None,
        "candidate_text": candidate_text,
        "candidates": candidate_dicts,
        "detection_count": len(all_detections),
    }

    if persist_debug:
        debug_path = run_dir / "yolo_ocr" / f"event_{event.index:03d}.json"
        write_json(debug_path, payload)

    return {
        "used_vision": True,
        "candidate_text": candidate_text,
        "local_cursor": local,
        "candidates": candidate_dicts,
        "detection_count": len(all_detections),
    }
