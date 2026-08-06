from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    import numpy as np

from cua_mcp.icon_map import is_pua_char
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_INPUT, YOLO_CLASS_TEXT
from src.common.io_utils import write_json
from src.common.nearby_side import (
    NearbyHint,
    Side,
    format_nearby_context_comment,
    side_from_anchor_bbox,
    side_to_schema_value,
    side_to_zh,
)
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent

_MIN_NEARBY_TEXT_LANDMARKS = 2
_MIN_NEARBY_TEXT_CANDIDATES = 5
_MIN_NEARBY_ICON_CANDIDATES = 5
_DRAG_OFFSET_THRESHOLD_PX = 5


def _local_cursor(event: RecordedEvent) -> tuple[int, int] | None:
    """Convert the event cursor from global desktop coords to screenshot-local coords."""
    if event.cursor_xy is None:
        return None
    return _global_to_local(event, event.cursor_xy)


def _local_end_cursor(event: RecordedEvent) -> tuple[int, int] | None:
    """Convert the drag end point from global desktop coords to screenshot-local coords."""
    if event.end_xy is None:
        return None
    gx, gy = event.end_xy
    offset = event.end_monitor_offset if event.end_monitor_offset is not None else event.monitor_offset
    if offset is not None:
        ox, oy = offset
        return gx - ox, gy - oy
    return gx, gy


def _global_to_local(event: RecordedEvent, global_xy: tuple[int, int]) -> tuple[int, int]:
    """Subtract the event monitor offset to map a global point into screenshot space."""
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


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """Return width × height for an axis-aligned ``(x, y, w, h)`` bbox."""
    return int(bbox[2]) * int(bbox[3])


def _nearest_candidate_rank_key(
    bbox: tuple[int, int, int, int],
    local_x: int,
    local_y: int,
) -> tuple[float, int]:
    """Sort key: point-to-bbox distance, then smallest area (innermost hit)."""
    return (
        _point_to_bbox_distance_sq(local_x, local_y, bbox),
        _bbox_area(bbox),
    )


def _visible_text(text: str | None) -> str:
    """Strip Private Use Area icon glyphs and whitespace from OCR text."""
    if not text:
        return ""
    return "".join(ch for ch in text if not is_pua_char(ch)).strip()


def _is_multi_char_text_detection(det: UiDetection) -> bool:
    """True for text-class detections whose visible OCR has more than one character."""
    return det.class_id == YOLO_CLASS_TEXT and len(_visible_text(det.text)) > 1


def _is_icon_detection(det: UiDetection) -> bool:
    """True when a detection carries known icon metadata usable as a landmark."""
    return bool(det.icons)


def _is_multi_char_text_candidate(candidate: dict[str, Any], label: str) -> bool:
    """True when a candidate is a multi-character text landmark (not a 1-char icon miss)."""
    visible = _visible_text(candidate.get("text"))
    if len(visible) <= 1:
        return False
    class_name = str(candidate.get("class_name") or "").strip()
    return class_name == "text" or label.endswith("文字")


def _nearest_candidates(
    detections: list[UiDetection],
    local_x: int,
    local_y: int,
    *,
    limit: int | None = None,
    min_multi_char_text_neighbors: int | None = _MIN_NEARBY_TEXT_CANDIDATES,
    min_icon_neighbors: int | None = _MIN_NEARBY_ICON_CANDIDATES,
) -> list[UiDetection]:
    """Return detections sorted by point-to-bbox distance (closest first).

    When several boxes contain the cursor (distance 0), prefer the smallest bbox.

    By default, always includes the nearest detection as primary, then keeps
    appending neighbors until both quotas are met:

    - ``min_multi_char_text_neighbors`` multi-character text detections
    - ``min_icon_neighbors`` detections with icon metadata

    Icons misclassified as text with 0–1 visible characters do not count toward
    the text quota. There is no fixed candidate cap unless ``limit`` is set; if
    either quota cannot be filled from remaining detections, includes the rest.

    Pass ``min_multi_char_text_neighbors=None`` (or 0) to return the full
    distance-ranked list (optionally truncated by ``limit``), ignoring icon
    quotas as well.
    """
    if not detections:
        return []
    scored = sorted(
        detections,
        key=lambda d: _nearest_candidate_rank_key(d.bbox, local_x, local_y),
    )
    if not min_multi_char_text_neighbors:
        return scored if limit is None else scored[:limit]

    text_target = max(int(min_multi_char_text_neighbors), 0)
    icon_target = max(int(min_icon_neighbors or 0), 0)

    nearest: list[UiDetection] = [scored[0]]
    multi_char_texts = 1 if _is_multi_char_text_detection(scored[0]) else 0
    icon_count = 1 if _is_icon_detection(scored[0]) else 0
    for det in scored[1:]:
        if multi_char_texts >= text_target and icon_count >= icon_target:
            break
        nearest.append(det)
        if _is_multi_char_text_detection(det):
            multi_char_texts += 1
        if _is_icon_detection(det):
            icon_count += 1
        if limit is not None and len(nearest) >= limit:
            break
    return nearest


def _bbox_center_inside(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    """True when the center of ``inner`` lies inside the ``outer`` bbox."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    cx, cy = ix + iw // 2, iy + ih // 2
    return ox <= cx < ox + ow and oy <= cy < oy + oh


def _point_inside_bbox(x: int, y: int, bbox: tuple[int, int, int, int]) -> bool:
    """True when ``(x, y)`` lies inside an axis-aligned ``(x, y, w, h)`` bbox."""
    bx, by, bw, bh = bbox
    return bx <= x < bx + bw and by <= y < by + bh


def _drop_point_inside_candidate(
    drop_x: int,
    drop_y: int,
    candidate: dict[str, Any],
) -> bool:
    """True when the drop point falls inside a candidate's bbox dict."""
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


def _nearest_candidate_by_class(
    candidates: list[dict[str, Any]],
    class_name: str,
    local: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """Return the nearest candidate with ``class_name``, or the first when no cursor."""
    matches = [c for c in candidates if c.get("class_name") == class_name]
    if not matches:
        return None
    if local is not None:
        lx, ly = local
        return min(
            matches,
            key=lambda c: _nearest_candidate_rank_key(
                tuple(c["bbox"]),
                lx,
                ly,
            ),
        )
    return matches[0]


def _candidate_meaningful_content_label(candidate: dict[str, Any]) -> str | None:
    """Return OCR text or the first icon label from a candidate, when meaningful."""
    text = _visible_text(candidate.get("text"))
    if text:
        return text
    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            return label
    return None


def _scrollbar_content_region(
    scrollbar_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Expand a scrollbar bbox toward its scrollable content (left or above)."""
    x, y, w, h = scrollbar_bbox
    if h >= w:
        expand = min(max(w * 4, 120), 480)
        return max(0, x - expand), y, expand, h
    expand = min(max(h * 4, 120), 480)
    return x, max(0, y - expand), w, expand


def _scrollbar_adjacent_content_labels(
    candidates: list[dict[str, Any]],
    scrollbar_bbox: tuple[int, int, int, int],
) -> list[str]:
    """Return visible labels from text/element candidates beside a scrollbar."""
    region = _scrollbar_content_region(scrollbar_bbox)
    labels: list[str] = []
    for candidate in candidates:
        class_name = candidate.get("class_name")
        if class_name not in ("text", "element"):
            continue
        label = _candidate_meaningful_content_label(candidate)
        if not label:
            continue
        if _bbox_center_inside(region, tuple(candidate["bbox"])) and label not in labels:
            labels.append(label)
    return labels


def _input_field_context_hint(
    candidates: list[dict[str, Any]],
    local: tuple[int, int] | None,
    *,
    typed_text: str | None = None,
) -> str | None:
    """Summarize visible text inside the nearest input candidate."""
    inp = _nearest_candidate_by_class(candidates, "input", local)
    if inp is None:
        return None

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


def _scrollbar_field_context_hint(
    candidates: list[dict[str, Any]],
    local: tuple[int, int] | None,
) -> str | None:
    """Summarize scrollable content beside the nearest scrollbar candidate."""
    scroll = _nearest_candidate_by_class(candidates, "scrollbar", local)
    if scroll is None:
        return None

    adjacent = _scrollbar_adjacent_content_labels(candidates, tuple(scroll["bbox"]))
    if adjacent:
        return f"滾動條旁可見內容: 「{adjacent[0]}」"
    return "最近的滾動條（無可辨識內容）"


def format_input_context_hint(
    vision: dict[str, Any],
    *,
    typed_text: str | None = None,
) -> str | None:
    """Return an input context line for LLM naming, or None when no input nearby."""
    candidates = vision.get("candidates") or []
    local = vision.get("local_cursor")
    local_xy: tuple[int, int] | None = None
    if isinstance(local, (list, tuple)) and len(local) == 2:
        local_xy = int(local[0]), int(local[1])
    return _input_field_context_hint(candidates, local_xy, typed_text=typed_text)


def format_scrollbar_context_hint(vision: dict[str, Any]) -> str | None:
    """Return a scrollbar context line for LLM naming, or None when no scrollbar nearby."""
    candidates = vision.get("candidates") or []
    local = vision.get("local_cursor")
    local_xy: tuple[int, int] | None = None
    if isinstance(local, (list, tuple)) and len(local) == 2:
        local_xy = int(local[0]), int(local[1])
    return _scrollbar_field_context_hint(candidates, local_xy)


def format_field_context_hint(
    vision: dict[str, Any],
    *,
    typed_text: str | None = None,
) -> str:
    """Summarize the nearest input or scrollbar context for LLM naming."""
    candidates = vision.get("candidates") or []
    local = vision.get("local_cursor")
    local_xy: tuple[int, int] | None = None
    if isinstance(local, (list, tuple)) and len(local) == 2:
        local_xy = int(local[0]), int(local[1])

    input_hint = _input_field_context_hint(candidates, local_xy, typed_text=typed_text)
    scroll_hint = _scrollbar_field_context_hint(candidates, local_xy)

    if input_hint and scroll_hint and local_xy is not None:
        lx, ly = local_xy
        inp = _nearest_candidate_by_class(candidates, "input", local_xy)
        scroll = _nearest_candidate_by_class(candidates, "scrollbar", local_xy)
        if inp is not None and scroll is not None:
            d_input = _point_to_bbox_distance_sq(lx, ly, tuple(inp["bbox"]))
            d_scroll = _point_to_bbox_distance_sq(lx, ly, tuple(scroll["bbox"]))
            return input_hint if d_input <= d_scroll else scroll_hint

    if input_hint:
        return input_hint
    if scroll_hint:
        return scroll_hint
    if typed_text and typed_text.strip():
        return f"Typed text: {typed_text.strip()!r}"
    return "(none)"


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
    """Return a short display label for offset-hint lines (text, icon, or class)."""
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
    """Return a hub-style drag anchor phrase like 「Desktop」文字 or 「Chrome」圖示.

    Prefer visible OCR text (PUA stripped) over icon labels so mixed text+icon
    rows keep the readable caption (e.g. 「速的網域 (3)」文字, not 「下載」圖示).
    """
    class_name = str(candidate.get("class_name") or "").strip()

    visible = _visible_text(candidate.get("text"))
    if visible:
        if class_name == "text":
            return f"「{visible}」文字"
        if class_name == "element":
            return f"「{visible}」元素"
        if class_name == "unknown":
            return f"「{visible}」未知"
        if class_name == "input":
            return f"「{visible}」文字所在的輸入欄"
        return f"「{visible}」"

    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            return f"「{label}」圖示"

    raw = str(candidate.get("text") or "").strip()
    if raw:
        if class_name == "text":
            return f"「{raw}」文字"
        if class_name == "element":
            return f"「{raw}」元素"
        if class_name == "unknown":
            return f"「{raw}」未知"

    suffix_by_class = {
        "text": "文字",
        "element": "元素",
        "unknown": "未知",
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
    if str(candidate.get("class_name") or "").strip() == "unknown":
        return None
    anchor = format_drag_candidate_anchor(candidate)
    if anchor is None:
        return None
    generic_only = {"文字", "元素", "未知", "輸入欄", "按鈕", "滾動條"}
    if anchor in generic_only:
        return None
    if anchor.endswith("未知"):
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


def _nearby_hint_tier(candidate: dict[str, Any], label: str) -> int:
    """Lower tier is preferred when selecting nearby landmarks.

    Multi-character text landmarks rank first. Single-character text (often an
    icon misclassified as text) and other non-icon labels are middle; icons last.
    """
    if label.endswith("圖示"):
        return 2
    if _is_multi_char_text_candidate(candidate, label):
        return 0
    return 1


def collect_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
    max_count: int = _MIN_NEARBY_TEXT_LANDMARKS,
) -> list[NearbyHint]:
    """Collect nearby hints from candidates after the primary.

    Walks neighbors until at least ``max_count`` multi-character text landmarks
    are found. If fewer exist, fills remaining slots with other neighbors.
    Within a tier, keeps distance order from ``candidates``. Uses the primary
    candidate bbox and each neighbor center to assign an optional script side
    via the 9-section grid. Neighbors whose center falls in the CENTER cell
    stay undirected (``side=None``).
    """
    candidates = vision.get("candidates") or []
    if len(candidates) < 2:
        return []

    primary = candidates[0]
    if not isinstance(primary, dict):
        return []
    primary_bbox = primary.get("bbox")
    has_bbox = isinstance(primary_bbox, (list, tuple)) and len(primary_bbox) == 4
    bbox: tuple[int, int, int, int] | None = None
    if has_bbox:
        bbox = (
            int(primary_bbox[0]),
            int(primary_bbox[1]),
            int(primary_bbox[2]),
            int(primary_bbox[3]),
        )

    eligible: list[tuple[int, int, dict[str, Any], str]] = []
    seen: set[str] = set()
    for order, candidate in enumerate(candidates[1:]):
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if _label_already_in_instruction(label, instruction):
            continue
        seen.add(label)
        eligible.append((_nearby_hint_tier(candidate, label), order, candidate, label))

    eligible.sort(key=lambda item: (item[0], item[1]))

    selected: list[tuple[dict[str, Any], str]] = []
    text_count = 0
    for tier, _order, candidate, label in eligible:
        if tier == 0:
            selected.append((candidate, label))
            text_count += 1
            if text_count >= max_count:
                break
        elif text_count < max_count and len(selected) < max_count:
            # Fill only after texts are exhausted (eligible is text-first).
            selected.append((candidate, label))
            if len(selected) >= max_count:
                break

    hints: list[NearbyHint] = []
    for candidate, label in selected:
        side: Side | None = None
        if bbox is not None:
            center = _candidate_center(candidate)
            if center is not None:
                side = side_from_anchor_bbox(bbox, center[0], center[1])
        hints.append(NearbyHint(label=label, side=side))
    return hints


def collect_nearby_hint_labels(
    vision: dict[str, Any],
    *,
    instruction: str,
    max_count: int = _MIN_NEARBY_TEXT_LANDMARKS,
) -> list[str]:
    """Collect up to max_count nearby candidate labels, skipping the primary target."""
    return [
        hint.label
        for hint in collect_nearby_hints(
            vision, instruction=instruction, max_count=max_count
        )
    ]


def list_nearby_landmark_options(
    vision: dict[str, Any],
    *,
    instruction: str = "",
) -> list[dict[str, Any]]:
    """Return all labelable neighbor landmarks (no auto-pick cap).

    Each option is ``{"label", "side", "display"}`` where ``side`` is the schema
    string (e.g. ``lower_left``) or ``None``. Skips the primary candidate, unknown
    / generic labels, and labels already present in the base instruction (after
    nearby comments are stripped) so the click target itself is excluded.
    """
    from src.common.nearby_side import strip_nearby_context_comments

    candidates = vision.get("candidates") or []
    if len(candidates) < 2:
        return []

    primary = candidates[0]
    if not isinstance(primary, dict):
        return []
    primary_bbox = primary.get("bbox")
    bbox: tuple[int, int, int, int] | None = None
    if isinstance(primary_bbox, (list, tuple)) and len(primary_bbox) == 4:
        bbox = (
            int(primary_bbox[0]),
            int(primary_bbox[1]),
            int(primary_bbox[2]),
            int(primary_bbox[3]),
        )

    base_instruction = strip_nearby_context_comments(instruction) if instruction else ""

    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates[1:]:
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if base_instruction and _label_already_in_instruction(label, base_instruction):
            continue
        seen.add(label)
        side: Side | None = None
        if bbox is not None:
            center = _candidate_center(candidate)
            if center is not None:
                side = side_from_anchor_bbox(bbox, center[0], center[1])
        if side is not None:
            display = f"{label}（{side_to_zh(side)}）"
        else:
            display = label
        options.append(
            {
                "label": label,
                "side": side_to_schema_value(side),
                "display": display,
            }
        )
    return options


def load_yolo_ocr_payload(
    run_root: Path,
    event_index: int,
    *,
    suffix: str = "",
) -> dict[str, Any] | None:
    """Load a persisted ``yolo_ocr/event_NNN{suffix}.json`` payload, if present."""
    path = Path(run_root) / "yolo_ocr" / f"event_{int(event_index):03d}{suffix}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def vision_from_yolo_ocr(
    run_root: Path,
    event_index: int,
    *,
    suffix: str = "",
) -> dict[str, Any]:
    """Build a minimal vision dict from persisted YOLO/OCR debug JSON."""
    payload = load_yolo_ocr_payload(run_root, event_index, suffix=suffix)
    if payload is None:
        return {"used_vision": False, "candidates": []}
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    return {
        "used_vision": True,
        "candidate_text": str(payload.get("candidate_text") or ""),
        "candidates": candidates,
        "local_cursor": payload.get("local_cursor"),
        "detection_count": payload.get("detection_count", len(candidates)),
    }


def load_recording_landmark_options(
    run_root: Path,
    event_index: int,
    *,
    kind: str,
    instruction: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Load landmark option groups for a recording event.

    Returns ``{"start": [...]}`` for click/scroll, or ``{"start": [...], "end": [...]}``
    for drag (終點 from ``_end_filtered`` with ``_end`` fallback).
    """
    start_vision = vision_from_yolo_ocr(run_root, event_index, suffix="")
    start_options = list_nearby_landmark_options(
        start_vision, instruction=instruction
    )
    if kind != "drag":
        return {"start": start_options}

    end_vision = vision_from_yolo_ocr(run_root, event_index, suffix="_end_filtered")
    if not end_vision.get("candidates"):
        end_vision = vision_from_yolo_ocr(run_root, event_index, suffix="_end")
    end_options = list_nearby_landmark_options(end_vision, instruction=instruction)
    return {"start": start_options, "end": end_options}


def append_nearby_context_comment(instruction: str, vision: dict[str, Any]) -> str:
    """Append a nearby-context parenthetical comment when vision data is available."""
    if not vision.get("used_vision"):
        return instruction
    comment = format_nearby_context_comment(
        collect_nearby_hints(vision, instruction=instruction),
        location="附近",
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
        collect_nearby_hints(vision, instruction=instruction),
        location="起點",
    )
    if start_comment and "拖到" in result:
        drag_at = result.index("拖到")
        result = result[:drag_at] + start_comment + result[drag_at:]

    dest_comment = format_nearby_context_comment(
        collect_nearby_hints(destination, instruction=instruction),
        location="終點",
    )
    if dest_comment:
        result = result + dest_comment

    return result


def _candidate_center(candidate: dict[str, Any]) -> tuple[int, int] | None:
    """Return the candidate center from ``center`` or by deriving it from ``bbox``."""
    center = candidate.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return int(center[0]), int(center[1])
    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
    return None


def _candidate_matches_anchor(candidate: dict[str, Any], anchor: str) -> bool:
    """True when candidate visible text, raw text, or icon label equals ``anchor``."""
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
    """Return the label used to match a candidate in drag offset lookup.

    Prefer visible OCR text over icon labels (same order as format_drag_candidate_anchor).
    """
    visible = _visible_text(candidate.get("text"))
    if visible:
        return visible
    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            return label
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


def primary_candidate_offset(vision: dict[str, Any]) -> str | None:
    """Return relative-pixel offset from the nearest candidate, or None if on-target."""
    candidates = vision.get("candidates") or []
    if not candidates:
        return None
    local = vision.get("local_cursor")
    if not isinstance(local, (list, tuple)) or len(local) != 2:
        return None
    drop_x, drop_y = int(local[0]), int(local[1])
    primary = candidates[0]
    if _drop_point_inside_candidate(drop_x, drop_y, primary):
        return None
    center = _candidate_center(primary)
    if center is None:
        return None
    return format_drag_relative_offset_phrase(drop_x - center[0], drop_y - center[1])


def _destination_candidates(
    all_detections: list[UiDetection],
    end_local: tuple[int, int],
) -> list[UiDetection]:
    """Return destination candidates at the drop point.

    When the drop lands inside a text/element, pin that hit as index 0 and
    append the usual nearest neighbors (deduped).
    """
    nearest = _nearest_candidates(all_detections, end_local[0], end_local[1])
    hit = _destination_target_at_point(all_detections, end_local[0], end_local[1])
    if hit is None:
        return nearest
    others = [det for det in nearest if det != hit]
    return [hit, *others]


def _candidate_dict_to_detection(candidate: dict[str, Any]) -> UiDetection:
    """Rebuild a ``UiDetection`` from a serialized candidate dict."""
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
    """Build destination vision with drop hit pinned first, then nearest neighbors."""
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
    """Serialize a ``UiDetection`` into a JSON-friendly candidate dict."""
    return {
        "bbox": list(det.bbox),
        "center": [det.cx, det.cy],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "text": det.text,
        "icons": det.icons,
    }


def _ocr_input_bbox_text(bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """Run OCR on a single input bbox and return the joined text, or empty."""
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
    """Return OCR from the nearest input field, then fall back to nearby text."""
    nearest = _nearest_candidates(
        detections,
        local_x,
        local_y,
        min_multi_char_text_neighbors=None,
    )
    if not nearest:
        return None

    nearest_input = next(
        (det for det in nearest if det.class_id == YOLO_CLASS_INPUT),
        None,
    )
    if nearest_input is not None:
        input_texts: list[str] = []
        for det in sorted(detections, key=lambda item: (item.cy, item.cx)):
            text = (det.text or "").strip()
            if (
                det.class_id == YOLO_CLASS_TEXT
                and text
                and _bbox_center_inside(nearest_input.bbox, det.bbox)
                and text not in input_texts
            ):
                input_texts.append(text)
        if input_texts:
            return "".join(input_texts)

        ocr_text = _ocr_input_bbox_text(bgr, nearest_input.bbox)
        if ocr_text:
            return ocr_text

    for det in nearest:
        text = (det.text or "").strip()
        if text:
            return text

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
    """Drop heavy fields (bgr/detections) and keep LLM-facing vision payload keys."""
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
