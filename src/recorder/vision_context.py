from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from cua_mcp.icon_map import is_pua_char
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT
from cua_mcp.selection_engine import request_json_with_retry
from src.common.io_utils import write_json
from src.common.prompting import get_prompt
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent

_NEAREST_CANDIDATE_LIMIT = 8
_DRAG_CLUSTER_MAX_DIST_PX = 60
_DRAG_DESTINATION_LLM_POOL = 12
_DRAG_OFFSET_THRESHOLD_PX = 5

_DRAG_OVERLAP_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "exclude_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["exclude_indices"],
}


def _local_cursor(event: RecordedEvent) -> tuple[int, int] | None:
    if event.cursor_xy is None:
        return None
    return _global_to_local(event, event.cursor_xy)


def _local_end_cursor(event: RecordedEvent) -> tuple[int, int] | None:
    if event.end_xy is None:
        return None
    gx, gy = event.end_xy
    offset = event.end_monitor_offset if event.end_monitor_offset is not None else event.monitor_offset
    if offset is not None:
        ox, oy = offset
        return gx - ox, gy - oy
    return gx, gy


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


def format_drag_relative_offset_phrase(dx: int, dy: int) -> str | None:
    """Return a Traditional Chinese offset phrase, or None when both axes are negligible."""
    parts: list[str] = []
    if abs(dx) >= _DRAG_OFFSET_THRESHOLD_PX:
        if dx > 0:
            parts.append(f"右方{dx}個像素")
        else:
            parts.append(f"左方{abs(dx)}個像素")
    if abs(dy) >= _DRAG_OFFSET_THRESHOLD_PX:
        if dy > 0:
            parts.append(f"下方{dy}個像素")
        else:
            parts.append(f"上方{abs(dy)}個像素")
    if not parts:
        return None
    return "、".join(parts)


def _candidate_display_label(candidate: dict[str, Any]) -> str:
    text = _visible_text(candidate.get("text"))
    if text:
        return f"「{text}」"
    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            return f"「{label}」圖示"
    class_name = candidate.get("class_name", "element")
    return f"({class_name})"


def _candidate_center(candidate: dict[str, Any]) -> tuple[int, int] | None:
    center = candidate.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return int(center[0]), int(center[1])
    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
    return None


def _candidate_matches_anchor(candidate: dict[str, Any], anchor: str) -> bool:
    text = _visible_text(candidate.get("text"))
    if text and text == anchor:
        return True
    raw = str(candidate.get("text") or "").strip()
    if raw and raw == anchor:
        return True
    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label == anchor:
            return True
    return False


def format_drag_destination_offset_hints(destination: dict[str, Any]) -> str:
    """Format per-candidate drop offsets for the drag destination LLM prompt."""
    local = destination.get("local_cursor")
    if not isinstance(local, (list, tuple)) or len(local) != 2:
        return "(none)"
    drop_x, drop_y = int(local[0]), int(local[1])
    candidates = destination.get("candidates") or []
    if not candidates:
        return "(none)"

    lines: list[str] = []
    for index, candidate in enumerate(candidates):
        center = _candidate_center(candidate)
        if center is None:
            continue
        dx = drop_x - center[0]
        dy = drop_y - center[1]
        phrase = format_drag_relative_offset_phrase(dx, dy)
        label = _candidate_display_label(candidate)
        if phrase is None:
            lines.append(f"[index {index}] {label}: (on anchor, offset negligible)")
        else:
            lines.append(f"[index {index}] {label}: {phrase}")
    return "\n".join(lines) if lines else "(none)"


def candidate_offset_for_instruction(
    destination: dict[str, Any],
    anchor_text: str,
) -> str | None:
    """Return the offset phrase for a destination anchor named in a drag instruction."""
    anchor = anchor_text.strip()
    if not anchor:
        return None
    local = destination.get("local_cursor")
    if not isinstance(local, (list, tuple)) or len(local) != 2:
        return None
    drop_x, drop_y = int(local[0]), int(local[1])
    for candidate in destination.get("candidates") or []:
        if not _candidate_matches_anchor(candidate, anchor):
            continue
        center = _candidate_center(candidate)
        if center is None:
            continue
        return format_drag_relative_offset_phrase(drop_x - center[0], drop_y - center[1])
    return None


def _bbox_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(aw * ah, 1)
    area_b = max(bw * bh, 1)
    return inter / min(area_a, area_b)


def _visible_text(text: str | None) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text if not is_pua_char(ch)).strip()


def _detection_identities(det: UiDetection) -> set[str]:
    """Return identity keys that may reappear when the same UI item is dragged."""
    keys: set[str] = set()
    raw = (det.text or "").strip()
    visible = _visible_text(det.text)
    if visible:
        keys.add(f"text:{visible}")
    if raw and not visible:
        keys.add(f"text:{raw}")
    for icon in det.icons or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            keys.add(f"icon:{label}")
            keys.add(f"text:{label}")
    if not keys:
        keys.add(f"{det.class_name}:element")
    return keys


def _candidate_dict_identities(candidate: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    raw = str(candidate.get("text") or "").strip()
    visible = _visible_text(candidate.get("text"))
    if visible:
        keys.add(f"text:{visible}")
    if raw and not visible:
        keys.add(f"text:{raw}")
    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            keys.add(f"icon:{label}")
            keys.add(f"text:{label}")
    if not keys:
        keys.add(f"{candidate.get('class_name', '')}:element")
    return keys


def _is_drag_cluster_member(det: UiDetection, anchor: UiDetection) -> bool:
    det_bbox = det.bbox
    anchor_bbox = anchor.bbox
    if _bbox_center_inside(anchor_bbox, det_bbox):
        return True
    if _bbox_center_inside(det_bbox, anchor_bbox):
        return True
    if _bbox_overlap_ratio(anchor_bbox, det_bbox) >= 0.15:
        return True
    dx = det.cx - anchor.cx
    dy = det.cy - anchor.cy
    return (dx * dx + dy * dy) <= _DRAG_CLUSTER_MAX_DIST_PX * _DRAG_CLUSTER_MAX_DIST_PX


def _collect_drag_source_cluster(
    all_detections: list[UiDetection],
    start_local: tuple[int, int],
) -> list[UiDetection]:
    """Return detections that belong to the UI cluster being dragged."""
    if not all_detections:
        return []

    nearest = _nearest_candidates(
        all_detections,
        start_local[0],
        start_local[1],
        limit=_NEAREST_CANDIDATE_LIMIT,
    )
    if not nearest:
        return []

    anchor = nearest[0]
    cluster: list[UiDetection] = []
    for det in all_detections:
        if not _is_drag_cluster_member(det, anchor):
            continue
        cluster.append(det)
    return cluster


def _collect_drag_source_identities(
    all_detections: list[UiDetection],
    start_local: tuple[int, int],
) -> set[str]:
    """Collect OCR/element identities for the widget cluster being dragged."""
    identities: set[str] = set()
    for det in _collect_drag_source_cluster(all_detections, start_local):
        identities.update(_detection_identities(det))
    return identities


def _parse_drag_overlap_exclude_indices(raw: str, *, pool_size: int) -> set[int]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    indices = data.get("exclude_indices")
    if not isinstance(indices, list):
        raise ValueError("exclude_indices must be a list")
    excluded: set[int] = set()
    for item in indices:
        if not isinstance(item, (int, float)):
            continue
        idx = int(item)
        if 0 <= idx < pool_size:
            excluded.add(idx)
    return excluded


async def _llm_exclude_dragged_destination_indices(
    start_cluster: list[UiDetection],
    destination_pool: list[UiDetection],
    *,
    log_info: Callable[[str], None] | None = None,
) -> set[int]:
    """Ask the LLM which destination candidates are the dragged object (OCR typo tolerant)."""
    if not start_cluster or not destination_pool:
        return set()

    prompt = get_prompt("recording_drag_destination_overlap").format(
        start_candidate_text=_format_ui_candidates_text(start_cluster),
        destination_candidate_text=_format_ui_candidates_text(destination_pool),
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        result = await request_json_with_retry(
            messages=messages,
            response_schema=_DRAG_OVERLAP_RESPONSE_SCHEMA,
            parse_reply=lambda raw: _parse_drag_overlap_exclude_indices(
                raw,
                pool_size=len(destination_pool),
            ),
            retry_instruction=get_prompt("recording_drag_destination_overlap_retry"),
            log_info=log_info,
            append_image_sizes=False,
        )
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"drag destination overlap LLM failed: {exc}")
        return set()


async def _filter_drag_destination_candidates(
    all_detections: list[UiDetection],
    end_local: tuple[int, int],
    start_cluster: list[UiDetection],
    *,
    limit: int = _NEAREST_CANDIDATE_LIMIT,
    log_info: Callable[[str], None] | None = None,
) -> tuple[list[UiDetection], set[int], set[str]]:
    """Filter destination detections using exact keys plus LLM fuzzy overlap."""
    if not all_detections:
        return [], set(), set()

    start_identities: set[str] = set()
    for det in start_cluster:
        start_identities.update(_detection_identities(det))

    scored = sorted(
        all_detections,
        key=lambda d: _point_to_bbox_distance_sq(end_local[0], end_local[1], d.bbox),
    )
    pool = scored[:_DRAG_DESTINATION_LLM_POOL]
    llm_exclude = await _llm_exclude_dragged_destination_indices(
        start_cluster,
        pool,
        log_info=log_info,
    )

    kept: list[UiDetection] = []
    for index, det in enumerate(pool):
        if index in llm_exclude:
            continue
        if _detection_identities(det) & start_identities:
            continue
        kept.append(det)
        if len(kept) >= limit:
            break
    return kept, llm_exclude, start_identities


def _filtered_nearest_candidates(
    all_detections: list[UiDetection],
    local_x: int,
    local_y: int,
    exclude_identities: set[str],
    *,
    limit: int = _NEAREST_CANDIDATE_LIMIT,
) -> list[UiDetection]:
    """Nearest detections at a point, skipping dragged-content identities."""
    if not all_detections:
        return []
    scored = sorted(
        all_detections,
        key=lambda d: _point_to_bbox_distance_sq(local_x, local_y, d.bbox),
    )
    kept: list[UiDetection] = []
    for det in scored:
        if _detection_identities(det) & exclude_identities:
            continue
        kept.append(det)
        if len(kept) >= limit:
            break
    return kept


def _candidate_dict_to_detection(candidate: dict[str, Any]) -> UiDetection:
    bbox = tuple(candidate["bbox"])
    x, y, w, h = bbox
    center = candidate.get("center") or [x + w // 2, y + h // 2]
    return UiDetection(
        bbox=bbox,
        cx=int(center[0]),
        cy=int(center[1]),
        class_id=int(candidate.get("class_id", 0)),
        class_name=str(candidate.get("class_name", "")),
        text=candidate.get("text"),
        icons=candidate.get("icons"),
    )


async def _build_filtered_destination_vision(
    end_result: dict[str, Any],
    *,
    end_local: tuple[int, int],
    start_cluster: list[UiDetection],
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    all_detections = end_result.get("all_detections")
    llm_exclude: set[int] = set()
    start_identities: set[str] = set()
    if isinstance(all_detections, list) and all_detections:
        filtered, llm_exclude, start_identities = await _filter_drag_destination_candidates(
            all_detections,
            end_local,
            start_cluster,
            log_info=log_info,
        )
        candidate_dicts = [_detection_to_dict(d) for d in filtered]
        candidate_text = (
            _format_ui_candidates_text(filtered)
            if filtered
            else "(no destination candidates after excluding dragged content)"
        )
    else:
        start_identities = set()
        for det in start_cluster:
            start_identities.update(_detection_identities(det))
        candidate_dicts = [
            c
            for c in end_result.get("candidates", [])
            if not (_candidate_dict_identities(c) & start_identities)
        ]
        filtered = [_candidate_dict_to_detection(c) for c in candidate_dicts]
        candidate_text = (
            _format_ui_candidates_text(filtered)
            if filtered
            else "(no destination candidates after excluding dragged content)"
        )

    vision = {
        "used_vision": bool(end_result.get("used_vision")),
        "candidate_text": candidate_text,
        "local_cursor": end_local,
        "candidates": candidate_dicts,
        "detection_count": end_result.get("detection_count", 0),
        "field_context": format_field_context_hint(
            {
                "candidates": candidate_dicts,
                "local_cursor": end_local,
            }
        ),
        "excluded_dragged_identities": sorted(start_identities),
        "llm_exclude_indices": sorted(llm_exclude),
    }
    vision["destination_offset_hints"] = format_drag_destination_offset_hints(vision)
    return vision


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
    image_path: str | None = None,
    debug_name: str | None = None,
) -> dict[str, Any]:
    """Run YOLO+OCR and rank candidates nearest to explicit screenshot-local coords."""
    empty: dict[str, Any] = {
        "used_vision": False,
        "candidate_text": "",
        "local_cursor": (local_x, local_y),
        "candidates": [],
        "detection_count": 0,
    }

    resolved_image_path = image_path or event.screenshot_path
    if not resolved_image_path or not Path(resolved_image_path).is_file():
        return empty

    bgr = cv2.imread(resolved_image_path)
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
        "image_path": resolved_image_path,
        "cursor_xy": list(reference_xy) if reference_xy else None,
        "local_cursor": [local_x, local_y],
        "candidate_text": candidate_text,
        "candidates": candidate_dicts,
        "detection_count": len(all_detections),
    }

    if persist_debug:
        suffix = debug_name or ""
        debug_path = run_dir / "yolo_ocr" / f"event_{event.index:03d}{suffix}.json"
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


def _compact_vision_point(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "used_vision": result.get("used_vision", False),
        "candidate_text": result.get("candidate_text", ""),
        "local_cursor": result.get("local_cursor"),
        "candidates": result.get("candidates", []),
        "detection_count": result.get("detection_count", 0),
        "field_context": format_field_context_hint(result),
    }
    return compact


async def build_vision_context(
    event: RecordedEvent,
    *,
    run_dir: Path,
    persist_debug: bool = True,
    log_info: Callable[[str], None] | None = None,
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

    if event.kind == "drag":
        start_local = _local_cursor(event)
        end_local = _local_end_cursor(event)
        if start_local is None or end_local is None:
            return empty

        start_result = build_vision_context_at_point(
            event,
            local_x=start_local[0],
            local_y=start_local[1],
            run_dir=run_dir,
            persist_debug=persist_debug,
            reference_xy=event.cursor_xy,
            image_path=event.screenshot_path,
            debug_name="",
        )
        end_image = event.end_screenshot_path or event.screenshot_path
        end_result = build_vision_context_at_point(
            event,
            local_x=end_local[0],
            local_y=end_local[1],
            run_dir=run_dir,
            persist_debug=persist_debug,
            reference_xy=event.end_xy,
            image_path=end_image,
            debug_name="_end",
        )
        start_compact = _compact_vision_point(start_result)
        start_cluster = _collect_drag_source_cluster(
            start_result.get("all_detections") or [],
            start_local,
        )
        end_compact = await _build_filtered_destination_vision(
            end_result,
            end_local=end_local,
            start_cluster=start_cluster,
            log_info=log_info,
        )
        if persist_debug:
            write_json(
                run_dir / "yolo_ocr" / f"event_{event.index:03d}_end_filtered.json",
                {
                    "event_index": event.index,
                    "local_cursor": list(end_local),
                    "excluded_dragged_identities": end_compact.get(
                        "excluded_dragged_identities",
                        [],
                    ),
                    "llm_exclude_indices": end_compact.get("llm_exclude_indices", []),
                    "candidate_text": end_compact["candidate_text"],
                    "candidates": end_compact["candidates"],
                },
            )
        return {
            **start_compact,
            "used_vision": bool(start_compact["used_vision"] or end_compact["used_vision"]),
            "destination": end_compact,
            "field_context": start_compact["field_context"],
            "destination_field_context": end_compact["field_context"],
        }

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
