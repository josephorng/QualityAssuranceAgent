from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT
from src.common.io_utils import write_json
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent

_NEAREST_CANDIDATE_LIMIT = 8


def _local_cursor(event: RecordedEvent) -> tuple[int, int] | None:
    if event.cursor_xy is None:
        return None
    return _global_to_local(event, event.cursor_xy)


def _global_to_local(event: RecordedEvent, global_xy: tuple[int, int]) -> tuple[int, int]:
    gx, gy = global_xy
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


def _bbox_center_inside(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    cx, cy = ix + iw // 2, iy + ih // 2
    return ox <= cx < ox + ow and oy <= cy < oy + oh


def format_field_context_hint(
    vision: dict[str, Any],
    *,
    typed_text: str | None = None,
) -> str:
    """Summarize visible text inside the nearest YOLO Input candidate for LLM naming."""
    candidates = vision.get("candidates") or []
    inputs = [c for c in candidates if c.get("class_name") == "input"]
    if not inputs:
        if typed_text and typed_text.strip():
            return f"Typed text: {typed_text.strip()!r}"
        return "(none)"

    local = vision.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        lx, ly = int(local[0]), int(local[1])
        inp = min(
            inputs,
            key=lambda c: _point_to_bbox_distance_sq(lx, ly, tuple(c["bbox"])),
        )
    else:
        inp = inputs[0]

    inp_bbox = tuple(inp["bbox"])
    inner_texts: list[str] = []
    for candidate in candidates:
        if candidate.get("class_name") != "text":
            continue
        text = (candidate.get("text") or "").strip()
        if text and _bbox_center_inside(inp_bbox, tuple(candidate["bbox"])):
            if text not in inner_texts:
                inner_texts.append(text)

    typed = (typed_text or "").strip()
    if typed:
        visible = typed
    elif inner_texts:
        visible = inner_texts[0]
    else:
        return "最近的輸入欄（無 OCR 可見文字）"

    return f"輸入欄內可見文字: 「{visible}」"


def _detection_to_dict(det: UiDetection) -> dict[str, Any]:
    return {
        "bbox": list(det.bbox),
        "center": [det.cx, det.cy],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "text": det.text,
        "icons": det.icons,
    }


def _ocr_input_bbox_text(bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    preds = _ocr_boxes_on_bgr(bgr, [bbox])
    if not preds:
        return ""
    return "".join(preds[0]).strip()


def extract_nearest_text(
    bgr: np.ndarray,
    detections: list[UiDetection],
    local_x: int,
    local_y: int,
) -> str | None:
    """Return OCR text from the nearest candidate at ``local_x``/``local_y``."""
    nearest = _nearest_candidates(detections, local_x, local_y, limit=len(detections))
    if not nearest:
        return None

    for det in nearest:
        text = (det.text or "").strip()
        if text:
            return text

    first = nearest[0]
    if first.class_id == YOLO_CLASS_INPUT:
        ocr_text = _ocr_input_bbox_text(bgr, first.bbox)
        if ocr_text:
            return ocr_text
    return None


def build_vision_context_at_point(
    event: RecordedEvent,
    *,
    local_x: int,
    local_y: int,
    run_dir: Path,
    persist_debug: bool = True,
    reference_xy: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run YOLO+OCR and rank candidates nearest to explicit screenshot-local coords."""
    empty: dict[str, Any] = {
        "used_vision": False,
        "candidate_text": "",
        "local_cursor": (local_x, local_y),
        "candidates": [],
        "detection_count": 0,
    }

    image_path = event.screenshot_path
    if not image_path or not Path(image_path).is_file():
        return empty

    bgr = cv2.imread(image_path)
    if bgr is None:
        return empty

    try:
        all_detections = _build_candidates_from_bgr(bgr)
    except RuntimeError:
        all_detections = []

    nearest: list[UiDetection] = []
    candidate_text = ""
    if all_detections:
        nearest = _nearest_candidates(all_detections, local_x, local_y)
        candidate_text = _format_ui_candidates_text(nearest)

    candidate_dicts = [_detection_to_dict(d) for d in nearest]
    payload: dict[str, Any] = {
        "event_index": event.index,
        "image_path": image_path,
        "cursor_xy": list(reference_xy) if reference_xy else None,
        "local_cursor": [local_x, local_y],
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
        "local_cursor": (local_x, local_y),
        "candidates": candidate_dicts,
        "detection_count": len(all_detections),
        "bgr": bgr,
        "all_detections": all_detections,
        "field_context": format_field_context_hint(
            {
                "candidates": candidate_dicts,
                "local_cursor": (local_x, local_y),
            }
        ),
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

    local = _local_cursor(event)
    if local is None:
        return empty

    result = build_vision_context_at_point(
        event,
        local_x=local[0],
        local_y=local[1],
        run_dir=run_dir,
        persist_debug=persist_debug,
        reference_xy=event.cursor_xy,
    )
    result.pop("bgr", None)
    result.pop("all_detections", None)
    result["field_context"] = format_field_context_hint(result)
    return result
