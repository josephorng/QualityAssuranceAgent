from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    import numpy as np

from cua_mcp.char_target import detect_clicked_char, format_char_target_anchor
from cua_mcp.icon_map import is_pua_char
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_INPUT, YOLO_CLASS_TEXT
from src.common.io_utils import write_json
from src.common.prompting import get_prompt
from src.common.nearby_side import (
    LandmarkCell,
    NearbyHint,
    Side,
    format_nearby_context_comment,
    landmark_cell_from_anchor_bbox,
    side_from_anchor_bbox,
    side_to_schema_value,
    side_to_zh,
)
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent

_MIN_NEARBY_TEXT_LANDMARKS = 2
_MIN_NEARBY_TEXT_CANDIDATES = 5
_MIN_NEARBY_ICON_CANDIDATES = 5
_DRAG_OFFSET_THRESHOLD_PX = 5
_CONTAINER_LANDMARK_CLASSES = frozenset({"input", "scrollbar"})
# Orthogonal bands around the primary click target (not diagonals / center).
_CARDINAL_LANDMARK_CELLS = frozenset(
    {
        LandmarkCell.LEFT,
        LandmarkCell.RIGHT,
        LandmarkCell.ABOVE,
        LandmarkCell.BELOW,
    }
)
# Tier-0 text landmark preference by where the landmark sits relative to the target.
# Lower rank is preferred: left → top → bottom → right; diagonals/center last.
_TIER0_CELL_RANK: dict[LandmarkCell, int] = {
    LandmarkCell.LEFT: 0,
    LandmarkCell.ABOVE: 1,
    LandmarkCell.BELOW: 2,
    LandmarkCell.RIGHT: 3,
}
_TIER0_NON_CARDINAL_RANK = 4
_NEARBY_LANDMARK_SELECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["keep_indices"],
}


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


def _visible_text(text: str | None) -> str:
    """Strip Private Use Area icon glyphs and whitespace from OCR text."""
    if not text:
        return ""
    return "".join(ch for ch in text if not is_pua_char(ch)).strip()


def _is_multi_char_text_detection(det: UiDetection) -> bool:
    """True for text-class detections whose visible OCR has more than one character."""
    return det.class_id == YOLO_CLASS_TEXT and len(_visible_text(det.text)) > 1


def _is_single_char_text_detection(det: UiDetection) -> bool:
    """True for text-class detections whose visible OCR is exactly one character."""
    return det.class_id == YOLO_CLASS_TEXT and len(_visible_text(det.text)) == 1


def _is_icon_detection(det: UiDetection) -> bool:
    """True when a detection carries known icon metadata usable as a landmark."""
    return bool(det.icons)


def _hit_content_priority(det: UiDetection) -> int:
    """Lower is better when several boxes share the same click distance.

    Priority:
    1. text with visible length > 1
    2. icon object
    3. text with visible length == 1
    4. others (scrollbar, input, empty element, …)
    """
    if _is_multi_char_text_detection(det):
        return 0
    if _is_icon_detection(det):
        return 1
    if _is_single_char_text_detection(det):
        return 2
    return 3


def _nearest_candidate_rank_key(
    bbox: tuple[int, int, int, int],
    local_x: int,
    local_y: int,
    *,
    content_priority: int = 3,
) -> tuple[float, int, int]:
    """Sort key: distance, then content priority, then smallest area."""
    return (
        _point_to_bbox_distance_sq(local_x, local_y, bbox),
        content_priority,
        _bbox_area(bbox),
    )


def _nearest_detection_rank_key(
    det: UiDetection,
    local_x: int,
    local_y: int,
) -> tuple[float, int, int]:
    """Sort key for a detection at a click/drop point."""
    return _nearest_candidate_rank_key(
        det.bbox,
        local_x,
        local_y,
        content_priority=_hit_content_priority(det),
    )


def _is_multi_char_text_candidate(candidate: dict[str, Any], label: str) -> bool:
    """True when a candidate is a multi-character text landmark (not a 1-char icon miss)."""
    visible = _visible_text(candidate.get("text"))
    if len(visible) <= 1:
        return False
    class_name = str(candidate.get("class_name") or "").strip()
    return class_name == "text" or label.endswith("文字")


def _detection_cardinal_cell(
    primary_bbox: tuple[int, int, int, int],
    det: UiDetection,
) -> LandmarkCell | None:
    """Return LEFT/RIGHT/ABOVE/BELOW when ``det`` sits on a cardinal side of primary."""
    cell = landmark_cell_from_anchor_bbox(primary_bbox, det.cx, det.cy)
    return cell if cell in _CARDINAL_LANDMARK_CELLS else None


def _append_cardinal_side_neighbors(
    nearest: list[UiDetection],
    scored: list[UiDetection],
    *,
    limit: int | None,
) -> list[UiDetection]:
    """Append every remaining neighbor on the primary's left/right/above/below sides.

    Distance order from ``scored`` is preserved. Diagonals and center-band
    detections are left to the earlier quota pass. ``limit`` still caps the
    final list when set.
    """
    if len(nearest) < 1 or len(scored) < 2:
        return nearest
    if limit is not None and len(nearest) >= limit:
        return nearest

    primary_bbox = nearest[0].bbox
    kept_ids = {id(det) for det in nearest}
    for det in scored[1:]:
        if id(det) in kept_ids:
            continue
        if _detection_cardinal_cell(primary_bbox, det) is None:
            continue
        nearest.append(det)
        kept_ids.add(id(det))
        if limit is not None and len(nearest) >= limit:
            break
    return nearest


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

    When several boxes contain the cursor (distance 0), prefer by content:
    multi-char text, then icon, then single-char text, then others; within a
    tier, prefer the smallest bbox.

    By default, always includes the nearest detection as primary, then keeps
    appending neighbors until both quotas are met:

    - ``min_multi_char_text_neighbors`` multi-character text detections
    - ``min_icon_neighbors`` detections with icon metadata

    After the quotas, also keeps **all** remaining neighbors whose centers fall
    on the primary's left / right / above / below bands (9-grid cardinals), so
    recording HTML can offer every orthogonal nearby choice. Diagonals are not
    force-included beyond the distance quotas.

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
        key=lambda d: _nearest_detection_rank_key(d, local_x, local_y),
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
    return _append_cardinal_side_neighbors(nearest, scored, limit=limit)


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
    """Return the preferred text/element whose bbox contains the drop point.

    Uses the same content priority as click ranking (multi-char text, icon,
    single-char text, then others), with smallest area as the final tie-breaker.
    """
    hits = [
        det
        for det in all_detections
        if det.class_id in (YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT)
        and _point_inside_bbox(x, y, det.bbox)
    ]
    if not hits:
        return None
    return min(hits, key=lambda det: (_hit_content_priority(det), _bbox_area(det.bbox)))


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
    """Return a hub-style label for a nearby-context hint, or None if not meaningful.

    Bare class labels ``輸入欄`` and ``滾動條`` are kept so empty inputs/scrollbars
    can be selected as nearby landmarks. Generic ``文字`` / ``元素`` / ``未知`` /
    ``按鈕`` alone are still dropped.
    """
    if str(candidate.get("class_name") or "").strip() == "unknown":
        return None
    anchor = format_drag_candidate_anchor(candidate)
    if anchor is None:
        return None
    generic_only = {"文字", "元素", "未知", "按鈕"}
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


def _tier0_cell_rank(
    candidate: dict[str, Any],
    *,
    primary_bbox: tuple[int, int, int, int] | None,
) -> int:
    """Lower is preferred for Tier-0 text: left, top, bottom, then right of target."""
    if primary_bbox is None:
        return _TIER0_NON_CARDINAL_RANK
    center = _candidate_center(candidate)
    if center is None:
        return _TIER0_NON_CARDINAL_RANK
    cell = landmark_cell_from_anchor_bbox(primary_bbox, center[0], center[1])
    return _TIER0_CELL_RANK.get(cell, _TIER0_NON_CARDINAL_RANK)


def primary_candidate_char_target(vision: dict[str, Any]) -> tuple[str, int] | None:
    """Return ``(char, occurrence_0based)`` from the primary candidate, if annotated."""
    candidates = vision.get("candidates") or []
    if not candidates:
        return None
    primary = candidates[0]
    clicked_char = primary.get("clicked_char")
    if not isinstance(clicked_char, str) or not clicked_char:
        return None
    raw_index = primary.get("clicked_char_index", 0)
    try:
        occurrence = int(raw_index)
    except (TypeError, ValueError):
        occurrence = 0
    if occurrence < 0:
        occurrence = 0
    return clicked_char, occurrence


def _annotate_clicked_char_target(
    bgr: np.ndarray,
    candidates: list[dict[str, Any]],
    local_x: int,
    local_y: int,
) -> None:
    """Store clicked character metadata on the primary text candidate when applicable."""
    if not candidates:
        return
    primary = candidates[0]
    if str(primary.get("class_name") or "").strip() != "text":
        return
    visible = _visible_text(primary.get("text"))
    if len(visible) <= 1:
        return
    if not _drop_point_inside_candidate(local_x, local_y, primary):
        return

    bbox = _as_bbox_xywh(primary.get("bbox"))
    if bbox is None:
        return

    detected = detect_clicked_char(bgr, bbox, local_x, text=visible)
    if detected is None:
        return
    clicked_char, occurrence = detected
    primary["clicked_char"] = clicked_char
    primary["clicked_char_index"] = occurrence


def _as_bbox_xywh(value: Any) -> tuple[int, int, int, int] | None:
    """Parse a candidate bbox list/tuple into ``(x, y, w, h)``, or None."""
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return int(value[0]), int(value[1]), int(value[2]), int(value[3])
    return None


def _click_xy_from_vision(
    vision: dict[str, Any],
    *,
    primary: dict[str, Any] | None = None,
) -> tuple[int, int] | None:
    """Return screenshot-local click coords, falling back to primary center."""
    local = vision.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        return int(local[0]), int(local[1])
    if primary is not None:
        return _candidate_center(primary)
    return None


def _neighbor_side_for_candidate(
    candidate: dict[str, Any],
    *,
    primary_bbox: tuple[int, int, int, int] | None,
    click_xy: tuple[int, int] | None,
) -> Side | None:
    """Assign a script side for a neighbor landmark.

    When the click falls inside an ``input`` / ``scrollbar`` bbox, use
    ``Side.INSIDE``. Otherwise fall back to the 9-section grid (or None).
    """
    class_name = str(candidate.get("class_name") or "").strip()
    cand_bbox = _as_bbox_xywh(candidate.get("bbox"))
    if (
        class_name in _CONTAINER_LANDMARK_CLASSES
        and click_xy is not None
        and cand_bbox is not None
        and _point_inside_bbox(click_xy[0], click_xy[1], cand_bbox)
    ):
        return Side.INSIDE
    if primary_bbox is not None:
        center = _candidate_center(candidate)
        if center is not None:
            return side_from_anchor_bbox(primary_bbox, center[0], center[1])
    return None


def _collect_containing_container_hints(
    candidates: list[Any],
    *,
    instruction: str,
    click_xy: tuple[int, int] | None,
) -> list[NearbyHint]:
    """Force-include 輸入欄 / 滾動條 landmarks that contain the click point.

    Skips ``candidates[0]`` (the primary click target).
    """
    if click_xy is None or len(candidates) < 2:
        return []
    hints: list[NearbyHint] = []
    seen: set[str] = set()
    for candidate in candidates[1:]:
        if not isinstance(candidate, dict):
            continue
        class_name = str(candidate.get("class_name") or "").strip()
        if class_name not in _CONTAINER_LANDMARK_CLASSES:
            continue
        cand_bbox = _as_bbox_xywh(candidate.get("bbox"))
        if cand_bbox is None:
            continue
        if not _point_inside_bbox(click_xy[0], click_xy[1], cand_bbox):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if _label_already_in_instruction(label, instruction):
            continue
        seen.add(label)
        hints.append(NearbyHint(label=label, side=Side.INSIDE))
    return hints


def _prioritized_nearby_parts(
    vision: dict[str, Any],
    *,
    instruction: str,
) -> tuple[list[NearbyHint], list[NearbyHint]]:
    """Split containing-container hints from ranked eligible neighbors.

    Ranking matches ``collect_nearby_hints``: Tier 0 multi-char text first
    (left → top → bottom → right, then diagonals/center), then other labels,
    then icons. Within the same rank, keeps distance order from ``candidates``.
    """
    candidates = vision.get("candidates") or []
    if len(candidates) < 2:
        return [], []

    primary = candidates[0]
    if not isinstance(primary, dict):
        return [], []
    primary_bbox = _as_bbox_xywh(primary.get("bbox"))
    click_xy = _click_xy_from_vision(vision, primary=primary)

    forced = _collect_containing_container_hints(
        candidates,
        instruction=instruction,
        click_xy=click_xy,
    )
    forced_labels = {hint.label for hint in forced}

    eligible: list[tuple[int, int, int, dict[str, Any], str]] = []
    seen: set[str] = set(forced_labels)
    for order, candidate in enumerate(candidates[1:]):
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if _label_already_in_instruction(label, instruction):
            continue
        seen.add(label)
        tier = _nearby_hint_tier(candidate, label)
        cell_rank = (
            _tier0_cell_rank(candidate, primary_bbox=primary_bbox)
            if tier == 0
            else 0
        )
        eligible.append((tier, cell_rank, order, candidate, label))

    eligible.sort(key=lambda item: (item[0], item[1], item[2]))

    ranked: list[NearbyHint] = []
    for _tier, _cell_rank, _order, candidate, label in eligible:
        side = _neighbor_side_for_candidate(
            candidate,
            primary_bbox=primary_bbox,
            click_xy=click_xy,
        )
        ranked.append(NearbyHint(label=label, side=side))
    return forced, ranked


def list_prioritized_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
) -> list[NearbyHint]:
    """Return all nearby hints in rank order, with no auto-pick cap.

    Containing ``input`` / ``scrollbar`` landmarks come first (``side=inside``),
    then every other eligible neighbor in the same order ``collect_nearby_hints``
    uses before cutting at ``max_count``.
    """
    forced, ranked = _prioritized_nearby_parts(vision, instruction=instruction)
    return [*forced, *ranked]


def collect_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
    max_count: int = _MIN_NEARBY_TEXT_LANDMARKS,
) -> list[NearbyHint]:
    """Collect nearby hints from candidates after the primary.

    Walks neighbors until at least ``max_count`` multi-character text landmarks
    are found. If fewer exist, fills remaining slots with other neighbors.
    Within Tier 0 (multi-char text), prefers landmarks on the left, then top,
    then bottom, then right of the target; diagonals/center follow. Within the
    same cell rank (and for lower tiers), keeps distance order from
    ``candidates``. Uses the primary candidate bbox and each neighbor center to
    assign an optional script side via the 9-section grid. Neighbors whose
    center falls in the CENTER cell stay undirected (``side=None``).

    When the click lies inside a non-primary ``input`` / ``scrollbar``, that
    container is always prepended with ``side=inside`` (裡面), even if that
    exceeds ``max_count``.
    """
    forced, ranked = _prioritized_nearby_parts(vision, instruction=instruction)
    return [*forced, *ranked[:max_count]]


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


def _format_ranked_nearby_option_line(index: int, hint: NearbyHint) -> str:
    """Format one ranked neighbor as ``[index N] label（side）`` for the LLM."""
    if hint.side is not None:
        display = f"{hint.label}（{side_to_zh(hint.side)}）"
    else:
        display = hint.label
    return f"[index {index}] {display}"


def _parse_nearby_landmark_select_reply(raw: str) -> dict[str, Any]:
    """Parse ``{"keep_indices": [int, ...]}`` from the landmark-select LLM."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("nearby landmark select reply must be an object")
    indices = data.get("keep_indices")
    if not isinstance(indices, list):
        raise ValueError("keep_indices must be a list")
    cleaned: list[int] = []
    for item in indices:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("keep_indices must contain integers")
        cleaned.append(item)
    return {"keep_indices": cleaned}


def _map_keep_indices_to_hints(
    ranked: list[NearbyHint],
    keep_indices: list[int],
    *,
    max_count: int,
) -> list[NearbyHint]:
    """Keep ranked hints by LLM index order, dropping unknowns and duplicates."""
    selected: list[NearbyHint] = []
    seen: set[int] = set()
    for index in keep_indices:
        if index in seen or index < 0 or index >= len(ranked):
            continue
        seen.add(index)
        selected.append(ranked[index])
        if len(selected) >= max_count:
            break
    return selected


async def select_stable_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
    screenshot_path: str | None = None,
    log_info: Any = None,
    max_count: int = _MIN_NEARBY_TEXT_LANDMARKS,
) -> list[NearbyHint]:
    """Ask the LLM which ranked neighbors are stable landmarks.

    Containing ``input`` / ``scrollbar`` hints are always kept. Remaining
    neighbors are sent in rank order with the screenshot. On missing screenshot,
    empty remaining options, or LLM/parse failure, falls back to
    :func:`collect_nearby_hints`. A successful empty ``keep_indices`` keeps only
    the forced containers.
    """
    forced, ranked = _prioritized_nearby_parts(vision, instruction=instruction)
    fallback = [*forced, *ranked[:max_count]]
    if not ranked:
        return fallback

    shot = Path(screenshot_path) if screenshot_path else None
    if shot is None or not shot.is_file():
        return fallback

    options_lines = "\n".join(
        _format_ranked_nearby_option_line(index, hint)
        for index, hint in enumerate(ranked)
    )
    prompt = get_prompt("recording_nearby_landmark_select").format(
        instruction=instruction,
        options_lines=options_lines or "(none)",
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "images": [str(shot)],
        }
    ]
    try:
        result = await request_json_with_retry(
            messages=messages,
            response_schema=_NEARBY_LANDMARK_SELECT_SCHEMA,
            parse_reply=_parse_nearby_landmark_select_reply,
            retry_instruction=get_prompt("recording_nearby_landmark_select_retry"),
            log_info=log_info,
            append_image_sizes=True,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"select_stable_nearby_hints failed: {exc}")
        return fallback

    selected = _map_keep_indices_to_hints(
        ranked,
        result["keep_indices"],
        max_count=max_count,
    )
    return [*forced, *selected]


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
    When the click is inside a neighbor ``input`` / ``scrollbar``, ``side`` is
    ``inside`` (裡面).
    """
    from src.common.nearby_side import strip_nearby_context_comments

    candidates = vision.get("candidates") or []
    if len(candidates) < 2:
        return []

    primary = candidates[0]
    if not isinstance(primary, dict):
        return []
    primary_bbox = _as_bbox_xywh(primary.get("bbox"))
    click_xy = _click_xy_from_vision(vision, primary=primary)

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
        side = _neighbor_side_for_candidate(
            candidate,
            primary_bbox=primary_bbox,
            click_xy=click_xy,
        )
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

    end_vision = drag_end_vision(run_root, event_index)
    end_options = list_nearby_landmark_options(end_vision, instruction=instruction)
    return {"start": start_options, "end": end_options}


def drag_end_yolo_suffix(run_root: Path, event_index: int) -> str:
    """Prefer ``_end_filtered`` when it has candidates; otherwise ``_end``."""
    filtered = load_yolo_ocr_payload(
        run_root, event_index, suffix="_end_filtered"
    )
    if isinstance(filtered, dict) and filtered.get("candidates"):
        return "_end_filtered"
    return "_end"


def drag_end_vision(run_root: Path, event_index: int) -> dict[str, Any]:
    """Load destination vision for a drag event (filtered file when present)."""
    return vision_from_yolo_ocr(
        run_root,
        event_index,
        suffix=drag_end_yolo_suffix(run_root, event_index),
    )


def list_primary_target_options(
    vision: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return selectable primary click/drag targets from ranked candidates.

    Each option is ``{"index", "label", "display"}``. Index 0 is the current
    primary. Candidates without a meaningful hub-style label are omitted.
    """
    candidates = vision.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return []

    options: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label:
            continue
        display = f"{label}（目前）" if index == 0 else label
        options.append({"index": index, "label": label, "display": display})
    return options


def load_recording_primary_target_options(
    run_root: Path,
    event_index: int,
    *,
    kind: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load primary-target option groups for a recording event.

    Returns ``{"start": [...]}`` for click/scroll/hold, or
    ``{"start": [...], "end": [...]}`` for drag.
    """
    start_options = list_primary_target_options(
        vision_from_yolo_ocr(run_root, event_index, suffix="")
    )
    if kind != "drag":
        return {"start": start_options}
    end_options = list_primary_target_options(
        drag_end_vision(run_root, event_index)
    )
    return {"start": start_options, "end": end_options}


def _dict_to_detection(candidate: dict[str, Any]) -> UiDetection | None:
    """Convert a persisted candidate dict into ``UiDetection``, if bbox is valid."""
    bbox = candidate.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    center = candidate.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        cx, cy = int(center[0]), int(center[1])
    else:
        cx, cy = x + w // 2, y + h // 2
    class_id = candidate.get("class_id")
    icons = candidate.get("icons")
    return UiDetection(
        bbox=(x, y, w, h),
        cx=cx,
        cy=cy,
        class_id=int(class_id) if class_id is not None else 0,
        class_name=str(candidate.get("class_name") or "element"),
        text=candidate.get("text") if candidate.get("text") is not None else None,
        icons=list(icons) if isinstance(icons, list) else None,
    )


def _candidate_text_from_dicts(candidates: list[Any]) -> str:
    """Rebuild LLM-style candidate_text lines after reordering persisted candidates."""
    detections: list[UiDetection] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        det = _dict_to_detection(candidate)
        if det is not None:
            detections.append(det)
    if not detections:
        return ""
    return _format_ui_candidates_text(detections)


def reorder_yolo_ocr_primary(
    run_root: Path,
    event_index: int,
    primary_index: int,
    *,
    suffix: str = "",
) -> dict[str, Any]:
    """Move ``candidates[primary_index]`` to index 0 and persist ``yolo_ocr``.

    Regenerates ``candidate_text`` to match the new order. Returns the updated
    payload. Raises ``ValueError`` when the file/index is invalid. No-op write
    when ``primary_index`` is already 0 (still returns the payload).
    """
    if not isinstance(primary_index, int) or primary_index < 0:
        raise ValueError("invalid primary index")

    path = Path(run_root) / "yolo_ocr" / f"event_{int(event_index):03d}{suffix}.json"
    payload = load_yolo_ocr_payload(run_root, event_index, suffix=suffix)
    if payload is None:
        raise ValueError("yolo_ocr not found")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("no candidates")
    if primary_index >= len(candidates):
        raise ValueError("primary index out of range")

    if primary_index != 0:
        reordered = list(candidates)
        chosen = reordered.pop(primary_index)
        reordered.insert(0, chosen)
        payload = dict(payload)
        payload["candidates"] = reordered
        payload["candidate_text"] = _candidate_text_from_dicts(reordered)
        write_json(path, payload)
    return payload


def append_nearby_context_comment(
    instruction: str,
    vision: dict[str, Any],
    hints: list[NearbyHint] | None = None,
) -> str:
    """Append a nearby-context parenthetical comment when vision data is available."""
    if not vision.get("used_vision"):
        return instruction
    if hints is None:
        hints = collect_nearby_hints(vision, instruction=instruction)
    comment = format_nearby_context_comment(
        hints,
        location="附近",
    )
    if comment is None:
        return instruction
    return instruction + comment


def append_drag_nearby_context_comments(
    instruction: str,
    vision: dict[str, Any],
    destination: dict[str, Any],
    start_hints: list[NearbyHint] | None = None,
    end_hints: list[NearbyHint] | None = None,
) -> str:
    """Insert start nearby comment before 拖到; append destination comment at end."""
    if not vision.get("used_vision"):
        return instruction

    result = instruction
    if start_hints is None:
        start_hints = collect_nearby_hints(vision, instruction=instruction)
    start_comment = format_nearby_context_comment(
        start_hints,
        location="起點",
    )
    if start_comment and "拖到" in result:
        drag_at = result.index("拖到")
        result = result[:drag_at] + start_comment + result[drag_at:]

    if end_hints is None:
        end_hints = collect_nearby_hints(destination, instruction=instruction)
    dest_comment = format_nearby_context_comment(
        end_hints,
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
    if all_detections and candidate_dicts:
        _annotate_clicked_char_target(bgr, candidate_dicts, local_x, local_y)
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
