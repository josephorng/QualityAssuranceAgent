from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from cua_mcp.icon_map import is_pua_char
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_INPUT, YOLO_CLASS_TEXT
from src.common.io_utils import write_json
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent

_NEAREST_CANDIDATE_LIMIT = 8
_DRAG_CLUSTER_MAX_DIST_PX = 60
_DRAG_OFFSET_THRESHOLD_PX = 5


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


def _point_inside_bbox(x: int, y: int, bbox: tuple[int, int, int, int]) -> bool:
    bx, by, bw, bh = bbox
    return bx <= x < bx + bw and by <= y < by + bh


def _drop_point_inside_candidate(
    drop_x: int,
    drop_y: int,
    candidate: dict[str, Any],
) -> bool:
    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return _point_inside_bbox(
            drop_x,
            drop_y,
            (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
        )
    return False


def _destination_target_at_point(
    all_detections: list[UiDetection],
    x: int,
    y: int,
) -> UiDetection | None:
    """Return the innermost text/element whose bbox contains the drop point."""
    hits = [
        det
        for det in all_detections
        if det.class_id in (YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT)
        and _point_inside_bbox(x, y, det.bbox)
    ]
    if not hits:
        return None
    return min(hits, key=lambda det: det.bbox[2] * det.bbox[3])


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


def format_drag_candidate_anchor(candidate: dict[str, Any]) -> str | None:
    """Return a hub-style drag anchor phrase like 「Chrome」圖示 or 「Desktop」文字."""
    class_name = str(candidate.get("class_name") or "").strip()

    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            return f"「{label}」圖示"

    visible = _visible_text(candidate.get("text"))
    if visible:
        if class_name == "text":
            return f"「{visible}」文字"
        if class_name == "element":
            return f"「{visible}」元素"
        if class_name == "input":
            return f"「{visible}」文字所在的輸入欄"
        return f"「{visible}」"

    raw = str(candidate.get("text") or "").strip()
    if raw:
        if class_name == "text":
            return f"「{raw}」文字"
        if class_name == "element":
            return f"「{raw}」元素"

    suffix_by_class = {
        "text": "文字",
        "element": "元素",
        "input": "輸入欄",
        "button": "按鈕",
        "scrollbar": "滾動條",
    }
    suffix = suffix_by_class.get(class_name)
    if suffix:
        return suffix
    return None


def _candidate_label_for_hint(candidate: dict[str, Any]) -> str | None:
    """Return a hub-style label for a nearby-context hint, or None if not meaningful."""
    anchor = format_drag_candidate_anchor(candidate)
    if anchor is None:
        return None
    generic_only = {"文字", "元素", "輸入欄", "按鈕", "滾動條"}
    if anchor in generic_only:
        return None
    return anchor


def _label_already_in_instruction(label: str, instruction: str) -> bool:
    """True when the label's quoted name or full phrase already appears in the instruction."""
    if label in instruction:
        return True
    if label.startswith("「") and "」" in label:
        inner = label.split("」", 1)[0][1:]
        if inner and f"「{inner}」" in instruction:
            return True
    return False


def collect_nearby_hint_labels(
    vision: dict[str, Any],
    *,
    instruction: str,
    max_count: int = 2,
) -> list[str]:
    """Collect up to max_count nearby candidate labels, skipping the primary target."""
    candidates = vision.get("candidates") or []
    if len(candidates) < 2:
        return []

    labels: list[str] = []
    seen: set[str] = set()
    for candidate in candidates[1:]:
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if _label_already_in_instruction(label, instruction):
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= max_count:
            break
    return labels


def format_nearby_context_comment(
    labels: list[str],
    *,
    location: str = "附近",
) -> str | None:
    """Format nearby labels as a trailing parenthetical comment."""
    if not labels:
        return None
    return f"（{location}有{'、'.join(labels)}）"


def append_nearby_context_comment(instruction: str, vision: dict[str, Any]) -> str:
    """Append a nearby-context parenthetical comment when vision data is available."""
    if not vision.get("used_vision"):
        return instruction
    comment = format_nearby_context_comment(
        collect_nearby_hint_labels(vision, instruction=instruction)
    )
    if comment is None:
        return instruction
    return instruction + comment


def append_drag_nearby_context_comments(
    instruction: str,
    vision: dict[str, Any],
    destination: dict[str, Any],
) -> str:
    """Insert start nearby comment before 拖到; append destination comment at end."""
    if not vision.get("used_vision"):
        return instruction

    result = instruction
    start_comment = format_nearby_context_comment(
        collect_nearby_hint_labels(vision, instruction=instruction),
        location="起點附近",
    )
    if start_comment and "拖到" in result:
        drag_at = result.index("拖到")
        result = result[:drag_at] + start_comment + result[drag_at:]

    dest_comment = format_nearby_context_comment(
        collect_nearby_hint_labels(destination, instruction=instruction),
        location="終點附近",
    )
    if dest_comment:
        result = result + dest_comment

    return result


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
        label = _candidate_display_label(candidate)
        if _drop_point_inside_candidate(drop_x, drop_y, candidate):
            lines.append(f"[index {index}] {label}: (on anchor, offset negligible)")
            continue
        center = _candidate_center(candidate)
        if center is None:
            continue
        dx = drop_x - center[0]
        dy = drop_y - center[1]
        phrase = format_drag_relative_offset_phrase(dx, dy)
        if phrase is None:
            lines.append(f"[index {index}] {label}: (on anchor, offset negligible)")
        else:
            lines.append(f"[index {index}] {label}: {phrase}")
    return "\n".join(lines) if lines else "(none)"


def candidate_anchor_name(candidate: dict[str, Any]) -> str | None:
    """Return the label used to match a candidate in drag offset lookup."""
    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            return label
    visible = _visible_text(candidate.get("text"))
    if visible:
        return visible
    raw = str(candidate.get("text") or "").strip()
    return raw or None


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
        if _drop_point_inside_candidate(drop_x, drop_y, candidate):
            return None
        center = _candidate_center(candidate)
        if center is None:
            continue
        return format_drag_relative_offset_phrase(drop_x - center[0], drop_y - center[1])
    return None


def _is_drag_cluster_member(det: UiDetection, anchor: UiDetection) -> bool:
    det_bbox = det.bbox
    anchor_bbox = anchor.bbox
    if _bbox_center_inside(anchor_bbox, det_bbox):
        return True
    if _bbox_center_inside(det_bbox, anchor_bbox):
        return True
    dx = det.cx - anchor.cx
    dy = det.cy - anchor.cy
    return (dx * dx + dy * dy) <= _DRAG_CLUSTER_MAX_DIST_PX * _DRAG_CLUSTER_MAX_DIST_PX


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


def _visible_text(text: str | None) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text if not is_pua_char(ch)).strip()


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


def _destination_candidates(
    all_detections: list[UiDetection],
    end_local: tuple[int, int],
    *,
    limit: int = _NEAREST_CANDIDATE_LIMIT,
) -> list[UiDetection]:
    """Return destination candidates at the drop point."""
    hit = _destination_target_at_point(all_detections, end_local[0], end_local[1])
    if hit is not None:
        return [hit]
    return _nearest_candidates(all_detections, end_local[0], end_local[1], limit=limit)


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


def _build_filtered_destination_vision(
    end_result: dict[str, Any],
    *,
    end_local: tuple[int, int],
) -> dict[str, Any]:
    all_detections = end_result.get("all_detections")
    if isinstance(all_detections, list) and all_detections:
        filtered = _destination_candidates(all_detections, end_local)
        candidate_dicts = [_detection_to_dict(d) for d in filtered]
        candidate_text = (
            _format_ui_candidates_text(filtered)
            if filtered
            else "(no destination candidates)"
        )
    else:
        candidate_dicts = list(end_result.get("candidates", []))
        filtered = [_candidate_dict_to_detection(c) for c in candidate_dicts]
        candidate_text = (
            _format_ui_candidates_text(filtered)
            if filtered
            else "(no destination candidates)"
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
        end_compact = _build_filtered_destination_vision(
            end_result,
            end_local=end_local,
        )
        if persist_debug:
            write_json(
                run_dir / "yolo_ocr" / f"event_{event.index:03d}_end_filtered.json",
                {
                    "event_index": event.index,
                    "local_cursor": list(end_local),
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
