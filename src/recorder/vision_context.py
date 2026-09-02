from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from PIL import Image

if TYPE_CHECKING:
    import numpy as np

from cua_mcp.char_target import detect_clicked_char
from cua_mcp.color_spatial_segment import (
    ColorSegmentResult,
    SegmentDetection,
    color_segment_to_json_dict,
    load_color_segment_params,
    reorder_detections_for_landmark,
    segment_image_by_color,
    spatial_region_rank_for_detections,
)
from cua_mcp.icon_map import is_pua_char
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.scrollbar_arrows import (
    is_scrollbar_end_arrow_candidate,
    point_in_bbox,
    scrollbar_axis_percent,
    scrollbar_orientation,
)
from cua_mcp.select_mouse_target import (
    _MOUSE_FILTER_SIMILARITY_THRESHOLD,
    _detect_mouse_targets_from_bgr,
    _label_similarity,
)
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from cua_mcp.yolo_onnx import (
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
)
from src.common.io_utils import imread_bgr, write_json
from src.common.nearby_side import (
    LandmarkCell,
    NearbyHint,
    Side,
    anchor_satisfies_side,
    format_nearby_context_comment,
    landmark_cell_from_anchor_bbox,
    side_from_anchor_bbox,
    side_to_schema_value,
    side_to_zh,
)
from src.recorder.models import (
    POINTER_EVENT_KINDS,
    RecordedEvent,
    screenshot_path_for_event,
    screenshot_path_for_event_end,
)

_MIN_NEARBY_TEXT_LANDMARKS = 2
_MIN_NEARBY_TEXT_CANDIDATES = 8
_MIN_NEARBY_ICON_CANDIDATES = 5
# Recording HTML「點擊目標」radio list: closest labeled candidates only.
_MAX_PRIMARY_TARGET_OPTIONS = 10
# Recording HTML「附近地標」: max options per directed side (closest to click).
_MAX_NEARBY_LANDMARK_OPTIONS_PER_SIDE = 10
# Prefer at least this many multi-char text neighbors in each directional cell.
_MIN_MULTI_CHAR_TEXT_PER_SIDE = 2
_DRAG_OFFSET_THRESHOLD_PX = 5
# OCR boxes hug glyphs; pad so a near-miss click still counts as on-target.
_BBOX_HIT_TOLERANCE_PX = 4
_CONTAINER_LANDMARK_CLASSES = frozenset({"input", "scrollbar"})
_SIMILAR_CLASS_LABELS = frozenset({"input", "scrollbar"})
_CLASS_LABEL_BY_NAME = {"input": "輸入欄", "scrollbar": "滾動條"}
# All eight directed sides used for recording HTML landmark side groups.
_DIRECTIONAL_LANDMARK_CELLS = frozenset(
    {
        LandmarkCell.LEFT,
        LandmarkCell.RIGHT,
        LandmarkCell.ABOVE,
        LandmarkCell.BELOW,
        LandmarkCell.UPPER_LEFT,
        LandmarkCell.UPPER_RIGHT,
        LandmarkCell.LOWER_LEFT,
        LandmarkCell.LOWER_RIGHT,
    }
)
# Tier-0 text landmark preference by where the landmark sits relative to the target.
# Lower rank is preferred: left → right → top → bottom; diagonals/center last.
_TIER0_CELL_RANK: dict[LandmarkCell, int] = {
    LandmarkCell.LEFT: 0,
    LandmarkCell.RIGHT: 1,
    LandmarkCell.ABOVE: 2,
    LandmarkCell.BELOW: 3,
}
_TIER0_NON_CARDINAL_RANK = 4


def vision_source_fingerprint(event: RecordedEvent) -> str:
    """Stable hash of event fields that affect YOLO/OCR / text-resolve reuse."""
    payload = {
        "index": int(event.index),
        "kind": str(event.kind),
        "screenshot_path": str(event.screenshot_path or ""),
        "end_screenshot_path": str(event.end_screenshot_path or ""),
        "cursor_xy": list(event.cursor_xy) if event.cursor_xy is not None else None,
        "end_xy": list(event.end_xy) if event.end_xy is not None else None,
        "text": event.text,
        "click_count": event.click_count,
        "focus_rect": list(event.focus_rect) if event.focus_rect is not None else None,
        "anchor_click_xy": (
            list(event.anchor_click_xy) if event.anchor_click_xy is not None else None
        ),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def text_resolution_cache_path(run_dir: Path, event_index: int) -> Path:
    return Path(run_dir) / "yolo_ocr" / f"event_{int(event_index):03d}_text_resolution.json"


def save_text_resolution_cache(
    run_dir: Path,
    event: RecordedEvent,
    text_resolution: dict[str, Any],
) -> None:
    """Persist text-input resolve fields for mid-recording prefetch reuse."""
    write_json(
        text_resolution_cache_path(run_dir, event.index),
        {
            "source_fingerprint": vision_source_fingerprint(event),
            "text_resolution": dict(text_resolution),
        },
    )


def load_text_resolution_cache(
    event: RecordedEvent,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Return cached text_resolution when fingerprint still matches."""
    path = text_resolution_cache_path(run_dir, event.index)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("source_fingerprint") != vision_source_fingerprint(event):
        return None
    text_resolution = payload.get("text_resolution")
    return text_resolution if isinstance(text_resolution, dict) else None


def _payload_fingerprint_matches(payload: dict[str, Any] | None, fingerprint: str) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("source_fingerprint") == fingerprint


def _payload_worth_reusing(payload: dict[str, Any]) -> bool:
    """True when cached YOLO/OCR is usable (skip transient empty load failures)."""
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        return True
    if payload.get("yolo_error"):
        return False
    # Successful run with zero detections is still a cache hit.
    return "candidates" in payload


def _vision_from_yolo_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild compact LLM-facing vision from a persisted yolo_ocr payload."""
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    local = payload.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        local_cursor: tuple[int, int] | list[Any] | None = (int(local[0]), int(local[1]))
    else:
        local_cursor = local
    vision: dict[str, Any] = {
        "used_vision": True,
        "candidate_text": str(payload.get("candidate_text") or ""),
        "local_cursor": local_cursor,
        "candidates": candidates,
        "detection_count": payload.get("detection_count", len(candidates)),
        "field_context": format_field_context_hint(
            {
                "candidates": candidates,
                "local_cursor": local_cursor,
            }
        ),
    }
    if payload.get("yolo_error"):
        vision["yolo_error"] = payload.get("yolo_error")
    track = payload.get("scrollbar_track")
    if isinstance(track, dict):
        vision["scrollbar_track"] = track
    return vision


def try_rebuild_vision_from_cache(
    event: RecordedEvent,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Rebuild pointer-event vision from fingerprinted ``yolo_ocr`` files, or None."""
    if event.kind not in POINTER_EVENT_KINDS:
        return None
    fingerprint = vision_source_fingerprint(event)
    run_root = Path(run_dir)

    if event.kind == "drag":
        start_payload = load_yolo_ocr_payload(run_root, event.index, suffix="")
        end_payload = load_yolo_ocr_payload(run_root, event.index, suffix="_end")
        if not _payload_fingerprint_matches(start_payload, fingerprint):
            return None
        if not _payload_fingerprint_matches(end_payload, fingerprint):
            return None
        assert start_payload is not None and end_payload is not None
        if not _payload_worth_reusing(start_payload):
            return None
        if not _payload_worth_reusing(end_payload) and not load_yolo_ocr_payload(
            run_root, event.index, suffix="_end_filtered"
        ):
            return None

        start_compact = _vision_from_yolo_payload(start_payload)
        filtered_payload = load_yolo_ocr_payload(
            run_root, event.index, suffix="_end_filtered"
        )
        end_local = _local_end_cursor(event)
        if (
            isinstance(filtered_payload, dict)
            and _payload_fingerprint_matches(filtered_payload, fingerprint)
            and isinstance(filtered_payload.get("candidates"), list)
        ):
            local = filtered_payload.get("local_cursor")
            if isinstance(local, (list, tuple)) and len(local) == 2:
                dest_local = (int(local[0]), int(local[1]))
            elif end_local is not None:
                dest_local = end_local
            else:
                dest_local = (0, 0)
            end_compact = {
                "used_vision": True,
                "candidate_text": str(filtered_payload.get("candidate_text") or ""),
                "local_cursor": dest_local,
                "candidates": list(filtered_payload.get("candidates") or []),
                "detection_count": filtered_payload.get(
                    "detection_count",
                    len(filtered_payload.get("candidates") or []),
                ),
                "field_context": format_field_context_hint(
                    {
                        "candidates": list(filtered_payload.get("candidates") or []),
                        "local_cursor": dest_local,
                    }
                ),
            }
            filtered_track = filtered_payload.get("scrollbar_track")
            if isinstance(filtered_track, dict):
                end_compact["scrollbar_track"] = filtered_track
            end_compact["destination_offset_hints"] = format_drag_destination_offset_hints(
                end_compact
            )
        elif end_local is not None:
            end_result = _vision_from_yolo_payload(end_payload)
            end_compact = _build_filtered_destination_vision(
                end_result,
                end_local=end_local,
            )
        else:
            return None

        start_track = start_compact.get("scrollbar_track")
        if (
            isinstance(start_track, dict)
            and "bbox" in start_track
            and end_local is not None
            and not isinstance(end_compact.get("scrollbar_track"), dict)
        ):
            annotate_scrollbar_track(
                end_compact,
                local_x=end_local[0],
                local_y=end_local[1],
                require_point_inside=False,
                scrollbar_bbox=start_track["bbox"],
            )

        combined_error = start_compact.get("yolo_error") or end_payload.get("yolo_error")
        return {
            **start_compact,
            "used_vision": bool(
                start_compact.get("used_vision") or end_compact.get("used_vision")
            ),
            "destination": end_compact,
            "field_context": start_compact["field_context"],
            "destination_field_context": end_compact["field_context"],
            **({"yolo_error": combined_error} if combined_error else {}),
        }

    start_payload = load_yolo_ocr_payload(run_root, event.index, suffix="")
    if not _payload_fingerprint_matches(start_payload, fingerprint):
        return None
    assert start_payload is not None
    if not _payload_worth_reusing(start_payload):
        return None
    return _vision_from_yolo_payload(start_payload)


def try_rebuild_text_input_from_cache(
    event: RecordedEvent,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return ``(text_resolution, vision)`` from cache when fingerprints match."""
    if event.kind != "text_input":
        return None
    text_resolution = load_text_resolution_cache(event, run_dir)
    if text_resolution is None:
        return None
    fingerprint = vision_source_fingerprint(event)
    # Prefer the typing end frame payload when present.
    for suffix in ("_end", ""):
        payload = load_yolo_ocr_payload(Path(run_dir), event.index, suffix=suffix)
        if not _payload_fingerprint_matches(payload, fingerprint):
            continue
        assert payload is not None
        if not _payload_worth_reusing(payload) and not text_resolution.get("resolved_text"):
            continue
        vision = _vision_from_yolo_payload(payload)
        # Match text_resolve._vision_for_llm shape (no field_context required).
        vision_for_llm = {
            "used_vision": vision.get("used_vision"),
            "candidate_text": vision.get("candidate_text"),
            "local_cursor": vision.get("local_cursor"),
            "candidates": vision.get("candidates"),
            "detection_count": vision.get("detection_count"),
        }
        return text_resolution, vision_for_llm
    # Text resolution alone is enough when vision was unavailable.
    return text_resolution, {
        "used_vision": False,
        "candidate_text": "",
        "local_cursor": None,
        "candidates": [],
        "detection_count": 0,
    }


def resolve_event_screenshot_path(
    event: RecordedEvent,
    run_dir: Path,
    *,
    image_path: str | None = None,
    debug_name: str | None = None,
) -> Path | None:
    """Resolve a screenshot that may still point at a pre-rename recording folder."""
    end_shot = debug_name == "_end"
    raw = image_path
    if raw is None:
        raw = event.end_screenshot_path if end_shot and event.end_screenshot_path else event.screenshot_path
    fallback = (
        screenshot_path_for_event_end(run_dir, event.index)
        if end_shot
        else screenshot_path_for_event(run_dir, event.index)
    )
    candidates: list[Path] = []
    if raw:
        stored = Path(raw)
        candidates.append(stored)
        candidates.append(run_dir / "screenshots" / stored.name)
        if not stored.is_absolute():
            candidates.append(run_dir / stored)
    candidates.append(fallback)
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


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


def _global_to_local_end(event: RecordedEvent, global_xy: tuple[int, int]) -> tuple[int, int]:
    """Map a global point into the after-screenshot's local coordinate space."""
    gx, gy = global_xy
    offset = event.end_monitor_offset if event.end_monitor_offset is not None else event.monitor_offset
    if offset is not None:
        ox, oy = offset
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


def _detection_directional_cell(
    primary_bbox: tuple[int, int, int, int],
    det: UiDetection,
) -> LandmarkCell | None:
    """Return the 9-grid cell when ``det`` sits on a directed side of primary.

    Center-band neighbors return ``None`` (they stay undirected / 「其他」).
    """
    cell = landmark_cell_from_anchor_bbox(primary_bbox, det.cx, det.cy)
    return cell if cell in _DIRECTIONAL_LANDMARK_CELLS else None


def _append_directional_side_neighbors(
    nearest: list[UiDetection],
    scored: list[UiDetection],
    *,
    limit: int | None,
    min_multi_char_text_per_side: int = _MIN_MULTI_CHAR_TEXT_PER_SIDE,
) -> list[UiDetection]:
    """Grow ``nearest`` so recording HTML has rich per-side landmark choices.

    1. Prefer at least ``min_multi_char_text_per_side`` multi-character text
       neighbors in each of the eight directed cells (上/下/左/右/左上/…).
    2. Then append every remaining neighbor on those eight sides (any type),
       preserving distance order from ``scored``.

    Center-band detections are left to the earlier quota pass. ``limit`` still
    caps the final list when set.
    """
    if len(nearest) < 1 or len(scored) < 2:
        return nearest
    if limit is not None and len(nearest) >= limit:
        return nearest

    primary_bbox = nearest[0].bbox
    kept_ids = {id(det) for det in nearest}
    text_target = max(int(min_multi_char_text_per_side), 0)

    per_side_text: dict[LandmarkCell, int] = {
        cell: 0 for cell in _DIRECTIONAL_LANDMARK_CELLS
    }
    for det in nearest[1:]:
        cell = _detection_directional_cell(primary_bbox, det)
        if cell is not None and _is_multi_char_text_detection(det):
            per_side_text[cell] += 1

    if text_target > 0:
        for det in scored[1:]:
            if id(det) in kept_ids:
                continue
            if not _is_multi_char_text_detection(det):
                continue
            cell = _detection_directional_cell(primary_bbox, det)
            if cell is None or per_side_text[cell] >= text_target:
                continue
            nearest.append(det)
            kept_ids.add(id(det))
            per_side_text[cell] += 1
            if limit is not None and len(nearest) >= limit:
                return nearest

    for det in scored[1:]:
        if id(det) in kept_ids:
            continue
        if _detection_directional_cell(primary_bbox, det) is None:
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
    segment_result: ColorSegmentResult | None = None,
) -> list[UiDetection]:
    """Return detections sorted by point-to-bbox distance (closest first).

    When several boxes contain the cursor (distance 0), prefer by content:
    multi-char text, then icon, then single-char text, then others; within a
    tier, prefer the smallest bbox.

    By default, always includes the nearest detection as primary, then keeps
    appending neighbors until both quotas are met:

    - ``min_multi_char_text_neighbors`` multi-character text detections
    - ``min_icon_neighbors`` detections with icon metadata

    After the quotas, prefers at least
    ``_MIN_MULTI_CHAR_TEXT_PER_SIDE`` multi-character text neighbors in each of
    the eight directed 9-grid sides, then keeps **all** remaining neighbors on
    those sides (cardinals and diagonals) so recording HTML can offer every
    directed nearby choice. Center-band neighbors stay quota-only.

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
    if segment_result is not None and len(scored) > 1:
        primary = scored[0]
        neighbors = reorder_detections_for_landmark(
            tuple(int(v) for v in primary.bbox),
            segment_result,
            scored[1:],
            lambda det: tuple(int(v) for v in det.bbox),
        )
        scored = [primary, *neighbors]
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
    return _append_directional_side_neighbors(nearest, scored, limit=limit)


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
    """True when the drop point falls inside a candidate's bbox, with a small pad.

    OCR boxes hug glyphs tightly; a few pixels of padding treats near-miss clicks
    (row padding above/below a label) as on-target.
    """
    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        pad = _BBOX_HIT_TOLERANCE_PX
        bx, by, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        return _point_inside_bbox(
            drop_x,
            drop_y,
            (bx - pad, by - pad, bw + 2 * pad, bh + 2 * pad),
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

    Bare class labels ``輸入欄``, ``滾動條``, and ``未知`` are kept so empty
    inputs/scrollbars/ambiguous elements can be selected in recording HTML.
    Generic ``文字`` / ``元素`` / ``按鈕`` alone are still dropped.
    """
    anchor = format_drag_candidate_anchor(candidate)
    if anchor is None:
        return None
    generic_only = {"文字", "元素", "按鈕"}
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
    """Lower is preferred for Tier-0 text: left, right, top, then bottom of target."""
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


def recording_primary_char_target(
    run_root: Path,
    event_index: int,
) -> tuple[str, str, int] | None:
    """Return ``(visible_text, char, occurrence)`` for the primary click target."""
    vision = vision_from_yolo_ocr(run_root, event_index)
    parsed = primary_candidate_char_target(vision)
    if parsed is None:
        return None
    candidates = vision.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return None
    visible = _visible_text(candidates[0].get("text"))
    if not visible:
        return None
    clicked_char, occurrence = parsed
    return visible, clicked_char, occurrence


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


def _candidate_match_labels(candidate: dict[str, Any]) -> set[str]:
    """Labels used to find confusable peers (OCR, icons, class names for input/scrollbar)."""
    labels: set[str] = set()
    visible = _visible_text(candidate.get("text"))
    if visible:
        labels.add(visible)
    for icon in candidate.get("icons") or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            labels.add(label)
    class_name = str(candidate.get("class_name") or "").strip()
    if class_name in _SIMILAR_CLASS_LABELS:
        class_label = _CLASS_LABEL_BY_NAME.get(class_name, "").strip()
        if class_label:
            labels.add(class_label)
    return labels


def _candidates_label_similar(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> bool:
    """True when any label pair between two candidates scores at least ``threshold``."""
    labels_left = _candidate_match_labels(left)
    labels_right = _candidate_match_labels(right)
    if not labels_left or not labels_right:
        return False
    for a in labels_left:
        for b in labels_right:
            if _label_similarity(a, b) >= threshold:
                return True
    return False


def _find_confusable_peers(
    candidates: list[Any],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return neighbors label-similar to ``candidates[0]`` (excluding the primary)."""
    if not candidates or not isinstance(candidates[0], dict):
        return []
    primary = candidates[0]
    peers: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        if not isinstance(candidate, dict):
            continue
        if _candidates_label_similar(primary, candidate, threshold=threshold):
            peers.append(candidate)
    return peers


def _landmark_separates_primary_from_peers(
    landmark: dict[str, Any],
    *,
    primary_bbox: tuple[int, int, int, int],
    confusables: list[dict[str, Any]],
) -> Side | None:
    """Return the script side for primary when landmark uniquely separates all peers."""
    center = _candidate_center(landmark)
    if center is None:
        return None
    primary_side = side_from_anchor_bbox(primary_bbox, center[0], center[1])
    if primary_side is None:
        return None

    lm_bbox = _as_bbox_xywh(landmark.get("bbox"))
    for peer in confusables:
        peer_bbox = _as_bbox_xywh(peer.get("bbox"))
        if peer_bbox is None:
            continue
        if anchor_satisfies_side(
            peer_bbox,
            center[0],
            center[1],
            primary_side,
            landmark_bbox=lm_bbox,
        ):
            return None
    return primary_side


def _betweenness_score(
    landmark_center: tuple[int, int],
    primary_center: tuple[int, int],
    peer_center: tuple[int, int],
) -> float:
    """Higher when landmark lies between primary and peer on the separating axis."""
    lx, ly = landmark_center
    px, py = primary_center
    cx, cy = peer_center
    dx, dy = px - cx, py - cy
    if abs(dx) >= abs(dy):
        lo, hi = sorted((px, cx))
        if lo < lx < hi:
            mid = (lo + hi) / 2
            return 1.0 - abs(lx - mid) / max(hi - lo, 1)
    else:
        lo, hi = sorted((py, cy))
        if lo < ly < hi:
            mid = (lo + hi) / 2
            return 1.0 - abs(ly - mid) / max(hi - lo, 1)
    return 0.0


def _score_disambiguating_landmarks(
    candidates: list[Any],
    *,
    instruction: str,
) -> list[tuple[float, int, int, NearbyHint]]:
    """Score neighbor landmarks that separate the primary from label-similar peers."""
    if len(candidates) < 2 or not isinstance(candidates[0], dict):
        return []

    confusables = _find_confusable_peers(candidates)
    if not confusables:
        return []

    primary = candidates[0]
    primary_bbox = _as_bbox_xywh(primary.get("bbox"))
    primary_center = _candidate_center(primary)
    if primary_bbox is None or primary_center is None:
        return []

    confusable_ids = {id(peer) for peer in confusables}
    scored: list[tuple[float, int, int, NearbyHint]] = []

    for order, candidate in enumerate(candidates[1:]):
        if not isinstance(candidate, dict) or id(candidate) in confusable_ids:
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or _label_already_in_instruction(label, instruction):
            continue

        side = _landmark_separates_primary_from_peers(
            candidate,
            primary_bbox=primary_bbox,
            confusables=confusables,
        )
        if side is None:
            continue

        center = _candidate_center(candidate)
        if center is None:
            continue

        between = 0.0
        for peer in confusables:
            peer_center = _candidate_center(peer)
            if peer_center is None:
                continue
            between = max(between, _betweenness_score(center, primary_center, peer_center))

        tier = _nearby_hint_tier(candidate, label)
        scored.append((between, tier, order, NearbyHint(label=label, side=side)))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return scored


def _pick_disambiguating_hints(
    candidates: list[Any],
    *,
    instruction: str,
    max_count: int = _MIN_NEARBY_TEXT_LANDMARKS,
) -> list[NearbyHint]:
    """Pick landmarks whose directed side separates the primary from similar peers."""
    if max_count <= 0:
        return []
    scored = _score_disambiguating_landmarks(candidates, instruction=instruction)
    return [hint for _, _, _, hint in scored[:max_count]]


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
    (left → right → top → bottom, then diagonals/center), then other labels,
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
    disambiguating = _pick_disambiguating_hints(
        candidates,
        instruction=instruction,
        max_count=_MIN_NEARBY_TEXT_LANDMARKS,
    )
    forced_labels = {hint.label for hint in forced}
    disambiguating_labels = {hint.label for hint in disambiguating}
    reserved_labels = forced_labels | disambiguating_labels

    eligible: list[tuple[int, int, int, dict[str, Any], str]] = []
    seen: set[str] = set(reserved_labels)
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
        spatial_rank = int(candidate.get("spatial_region_rank", 0))
        eligible.append((spatial_rank, tier, cell_rank, order, candidate, label))

    eligible.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    ranked: list[NearbyHint] = []
    for _spatial, _tier, _cell_rank, _order, candidate, label in eligible:
        side = _neighbor_side_for_candidate(
            candidate,
            primary_bbox=primary_bbox,
            click_xy=click_xy,
        )
        ranked.append(NearbyHint(label=label, side=side))
    return [*forced, *disambiguating], ranked


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


def _pick_side_diverse_hints(
    ranked: list[NearbyHint],
    *,
    max_count: int,
) -> list[NearbyHint]:
    """Pick up to ``max_count`` hints, preferring distinct directed sides.

    Keeps rank order. A second (or later) candidate with the same non-None
    ``side`` as an already-picked hint is skipped while a different-side
    option remains. If diversity cannot fill ``max_count``, remaining slots
    are filled from unused ranked hints regardless of side.
    """
    if max_count <= 0 or not ranked:
        return []

    picked: list[NearbyHint] = []
    used_sides: set[Side] = set()
    for hint in ranked:
        if len(picked) >= max_count:
            break
        if hint.side is not None and hint.side in used_sides:
            continue
        picked.append(hint)
        if hint.side is not None:
            used_sides.add(hint.side)

    if len(picked) >= max_count:
        return picked

    picked_labels = {hint.label for hint in picked}
    for hint in ranked:
        if len(picked) >= max_count:
            break
        if hint.label in picked_labels:
            continue
        picked.append(hint)
        picked_labels.add(hint.label)
    return picked


def collect_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
    max_count: int = _MIN_NEARBY_TEXT_LANDMARKS,
) -> list[NearbyHint]:
    """Collect nearby hints from candidates after the primary.

    Walks neighbors until at least ``max_count`` multi-character text landmarks
    are found. If fewer exist, fills remaining slots with other neighbors.
    Within Tier 0 (multi-char text), prefers landmarks on the left, then right,
    then top, then bottom of the target; diagonals/center follow. Within the
    same cell rank (and for lower tiers), keeps distance order from
    ``candidates``. Auto-picks prefer landmarks on different directed sides
    (e.g. left+right rather than two on the right). Uses the primary candidate
    bbox and each neighbor center to assign an optional script side via the
    9-section grid. Neighbors whose center falls in the CENTER cell stay
    undirected (``side=None``).

    When the click lies inside a non-primary ``input`` / ``scrollbar``, that
    container is always prepended with ``side=inside`` (裡面), even if that
    exceeds ``max_count``.
    """
    forced, ranked = _prioritized_nearby_parts(vision, instruction=instruction)
    return [*forced, *_pick_side_diverse_hints(ranked, max_count=max_count)]


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
    """Return labelable neighbor landmarks for the recording HTML picker.

    Each option is ``{"label", "side", "display"}`` where ``side`` is the schema
    string (e.g. ``lower_left``) or ``None``. Skips the primary candidate, generic
    labels, and labels already present in the base instruction (after
    nearby comments are stripped) so the click target itself is excluded.
    When the click is inside a neighbor ``input`` / ``scrollbar``, ``side`` is
    ``inside`` (裡面).

    Per directed side (and ``inside`` / undirected), keeps at most
    ``_MAX_NEARBY_LANDMARK_OPTIONS_PER_SIDE`` options, preferring those closest
    to the click (``local_cursor``, else primary center).
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

    disambig_by_label = {
        hint.label: hint.side
        for hint in _pick_disambiguating_hints(
            candidates,
            instruction=base_instruction,
            max_count=len(candidates),
        )
    }

    # (disambig_boost, spatial_rank, distance_sq, order, option)
    pending: list[tuple[int, int, float, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for order, candidate in enumerate(candidates[1:]):
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if base_instruction and _label_already_in_instruction(label, base_instruction):
            continue
        seen.add(label)
        disambig_side = disambig_by_label.get(label)
        if disambig_side is not None:
            side = disambig_side
        else:
            side = _neighbor_side_for_candidate(
                candidate,
                primary_bbox=primary_bbox,
                click_xy=click_xy,
            )
        if side is not None:
            display = f"{label}（{side_to_zh(side)}）"
        else:
            display = label
        bbox = _as_bbox_xywh(candidate.get("bbox"))
        if click_xy is not None and bbox is not None:
            dist_sq = _point_to_bbox_distance_sq(click_xy[0], click_xy[1], bbox)
        else:
            dist_sq = float(order)
        spatial_rank = int(candidate.get("spatial_region_rank", 0))
        disambig_boost = 0 if label in disambig_by_label else 1
        pending.append(
            (
                disambig_boost,
                spatial_rank,
                dist_sq,
                order,
                {
                    "label": label,
                    "side": side_to_schema_value(side),
                    "display": display,
                },
            )
        )

    by_side: dict[str | None, list[tuple[int, int, float, int, dict[str, Any]]]] = {}
    for item in pending:
        side_key = item[4].get("side")
        if side_key is None or side_key == "":
            bucket: str | None = None
        else:
            bucket = str(side_key)
        by_side.setdefault(bucket, []).append(item)

    kept: list[tuple[int, int, float, int, dict[str, Any]]] = []
    for items in by_side.values():
        items.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        kept.extend(items[:_MAX_NEARBY_LANDMARK_OPTIONS_PER_SIDE])
    kept.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [option for _, _, _, _, option in kept]


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
    Only the ``_MAX_PRIMARY_TARGET_OPTIONS`` candidates nearest the click
    (``candidates`` order) are considered.
    """
    candidates = vision.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return []

    options: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:_MAX_PRIMARY_TARGET_OPTIONS]):
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


def _ui_detection_to_segment(det: UiDetection) -> SegmentDetection:
    return SegmentDetection(
        box=tuple(int(v) for v in det.bbox),
        class_id=int(det.class_id),
        class_name=str(det.class_name or ""),
        text=str(det.text or "").strip(),
    )


def _detection_to_dict(
    det: UiDetection,
    *,
    spatial_region_rank: int = 0,
) -> dict[str, Any]:
    """Serialize a ``UiDetection`` into a JSON-friendly candidate dict."""
    return {
        "bbox": list(det.bbox),
        "center": [det.cx, det.cy],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "text": det.text,
        "icons": det.icons,
        "spatial_region_rank": int(spatial_region_rank),
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
    source_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Run YOLO+OCR and rank candidates nearest to explicit screenshot-local coords."""
    empty: dict[str, Any] = {
        "used_vision": False,
        "candidate_text": "",
        "local_cursor": (local_x, local_y),
        "candidates": [],
        "detection_count": 0,
    }
    fingerprint = source_fingerprint or vision_source_fingerprint(event)

    resolved = resolve_event_screenshot_path(
        event,
        run_dir,
        image_path=image_path,
        debug_name=debug_name,
    )
    resolved_image_path = str(resolved) if resolved is not None else (image_path or event.screenshot_path or "")
    load_error: str | None = None
    bgr = None
    if resolved is None:
        load_error = "找不到截圖檔"
    else:
        bgr = imread_bgr(resolved)
        if bgr is None:
            load_error = "無法讀取截圖"

    if load_error is not None:
        empty["yolo_error"] = load_error
        if persist_debug:
            write_json(
                run_dir / "yolo_ocr" / f"event_{event.index:03d}{debug_name or ''}.json",
                {
                    "event_index": event.index,
                    "image_path": resolved_image_path,
                    "cursor_xy": list(reference_xy) if reference_xy else None,
                    "local_cursor": [local_x, local_y],
                    "candidate_text": "",
                    "candidates": [],
                    "detection_count": 0,
                    "yolo_error": load_error,
                    "source_fingerprint": fingerprint,
                },
            )
        return empty

    assert bgr is not None
    yolo_error: str | None = None
    try:
        all_detections = _detect_mouse_targets_from_bgr(bgr)
    except RuntimeError as exc:
        all_detections = []
        yolo_error = str(exc)

    nearest: list[UiDetection] = []
    candidate_text = ""
    segment_result: ColorSegmentResult | None = None
    if all_detections:
        segment_detections = [_ui_detection_to_segment(det) for det in all_detections]
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            segment_result = segment_image_by_color(
                Image.fromarray(rgb, mode="RGB"),
                params=load_color_segment_params(),
                detections=segment_detections,
            )
        except Exception:
            segment_result = None
        nearest = _nearest_candidates(
            all_detections,
            local_x,
            local_y,
            segment_result=segment_result,
        )
        candidate_text = _format_ui_candidates_text(nearest)

    spatial_ranks: dict[tuple[int, int, int, int], int] = {}
    if segment_result is not None and nearest:
        landmark_box = tuple(int(v) for v in nearest[0].bbox)
        spatial_ranks = spatial_region_rank_for_detections(
            landmark_box,
            segment_result,
            [_ui_detection_to_segment(det) for det in all_detections],
            cursor_xy=(local_x, local_y),
        )

    candidate_dicts = [
        _detection_to_dict(
            det,
            spatial_region_rank=spatial_ranks.get(tuple(int(v) for v in det.bbox), 0),
        )
        for det in nearest
    ]
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
        "source_fingerprint": fingerprint,
    }
    if segment_result is not None:
        landmark_box_for_segment: tuple[int, int, int, int] | None = None
        if nearest:
            landmark_box_for_segment = tuple(int(v) for v in nearest[0].bbox)
        else:
            landmark_box_for_segment = (local_x - 1, local_y - 1, 2, 2)
        payload["color_segment"] = color_segment_to_json_dict(
            segment_result,
            landmark_box=landmark_box_for_segment,
            cursor_xy=(local_x, local_y),
        )
    if yolo_error:
        payload["yolo_error"] = yolo_error

    vision_for_track: dict[str, Any] = {
        "candidates": candidate_dicts,
        "candidate_text": candidate_text,
    }
    track = annotate_scrollbar_track(
        vision_for_track,
        local_x=local_x,
        local_y=local_y,
        all_detections=all_detections or None,
        require_point_inside=True,
    )
    if track is not None:
        payload["scrollbar_track"] = track
        payload["candidates"] = vision_for_track["candidates"]
        payload["candidate_text"] = vision_for_track.get("candidate_text") or candidate_text
        candidate_dicts = payload["candidates"]
        candidate_text = payload["candidate_text"]

    if persist_debug:
        suffix = debug_name or ""
        debug_path = run_dir / "yolo_ocr" / f"event_{event.index:03d}{suffix}.json"
        write_json(debug_path, payload)

    result: dict[str, Any] = {
        "used_vision": True,
        "candidate_text": candidate_text,
        "local_cursor": (local_x, local_y),
        "candidates": candidate_dicts,
        "detection_count": len(all_detections),
        "bgr": bgr,
        "all_detections": all_detections,
        "yolo_error": yolo_error,
        "field_context": format_field_context_hint(
            {
                "candidates": candidate_dicts,
                "local_cursor": (local_x, local_y),
            }
        ),
    }
    if track is not None:
        result["scrollbar_track"] = track
    return result


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
    if result.get("yolo_error"):
        compact["yolo_error"] = result.get("yolo_error")
    track = result.get("scrollbar_track")
    if isinstance(track, dict):
        compact["scrollbar_track"] = track
    return compact


def _candidate_bbox_tuple(candidate: dict[str, Any] | UiDetection) -> tuple[int, int, int, int] | None:
    """Return ``(x, y, w, h)`` from a candidate dict or detection."""
    if isinstance(candidate, UiDetection):
        return tuple(int(v) for v in candidate.bbox)
    bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return None


def _find_containing_scrollbar(
    detections: list[UiDetection] | list[dict[str, Any]],
    local_x: int,
    local_y: int,
) -> UiDetection | dict[str, Any] | None:
    """Return the smallest scrollbar bbox containing ``(local_x, local_y)``."""
    hits: list[tuple[int, UiDetection | dict[str, Any]]] = []
    for det in detections:
        if isinstance(det, UiDetection):
            if det.class_name != "scrollbar":
                continue
            bbox = det.bbox
        else:
            if str(det.get("class_name") or "").strip() != "scrollbar":
                continue
            bbox_t = _candidate_bbox_tuple(det)
            if bbox_t is None:
                continue
            bbox = bbox_t
        if point_in_bbox(local_x, local_y, bbox):
            hits.append((int(bbox[2]) * int(bbox[3]), det))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    return hits[0][1]


def _point_hits_scrollbar_end_arrow(
    detections: list[UiDetection] | list[dict[str, Any]],
    local_x: int,
    local_y: int,
) -> bool:
    """True when the point lands inside a scrollbar end-arrow icon bbox."""
    for det in detections:
        bbox = _candidate_bbox_tuple(det)
        if bbox is None:
            continue
        if point_in_bbox(local_x, local_y, bbox) and is_scrollbar_end_arrow_candidate(det):
            return True
    return False


def _promote_scrollbar_in_candidates(
    candidates: list[dict[str, Any]],
    scrollbar: dict[str, Any],
) -> list[dict[str, Any]]:
    """Move ``scrollbar`` to index 0 (or insert it) so instructions name the track."""
    sb_bbox = _candidate_bbox_tuple(scrollbar)
    if sb_bbox is None:
        return candidates
    others: list[dict[str, Any]] = []
    matched: dict[str, Any] | None = None
    for cand in candidates:
        if _candidate_bbox_tuple(cand) == sb_bbox:
            matched = cand
            continue
        others.append(cand)
    primary = matched if matched is not None else scrollbar
    return [primary, *others]


def annotate_scrollbar_track(
    vision: dict[str, Any],
    *,
    local_x: int,
    local_y: int,
    all_detections: list[UiDetection] | None = None,
    require_point_inside: bool = True,
    scrollbar_bbox: tuple[int, int, int, int] | list[int] | None = None,
) -> dict[str, Any] | None:
    """Annotate ``vision`` with ``scrollbar_track`` when the press is on the track.

    Skips end-arrow hits. When ``require_point_inside`` is False (drag release),
    projects onto ``scrollbar_bbox`` even if the point is outside the bar.
    Promotes the scrollbar candidate to primary so thumb icons do not win wording.
    """
    pool: list[Any] = list(all_detections or [])
    if not pool:
        pool = list(vision.get("candidates") or [])

    if require_point_inside:
        if _point_hits_scrollbar_end_arrow(pool, local_x, local_y):
            vision.pop("scrollbar_track", None)
            return None
        found = _find_containing_scrollbar(pool, local_x, local_y)
        if found is None:
            vision.pop("scrollbar_track", None)
            return None
        bbox = _candidate_bbox_tuple(found)
        if bbox is None:
            vision.pop("scrollbar_track", None)
            return None
        scrollbar_dict = (
            _detection_to_dict(found) if isinstance(found, UiDetection) else dict(found)
        )
    else:
        if scrollbar_bbox is None:
            vision.pop("scrollbar_track", None)
            return None
        bbox = (int(scrollbar_bbox[0]), int(scrollbar_bbox[1]),
                int(scrollbar_bbox[2]), int(scrollbar_bbox[3]))
        scrollbar_dict = {
            "bbox": list(bbox),
            "center": [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2],
            "class_id": YOLO_CLASS_SCROLLBAR,
            "class_name": "scrollbar",
            "text": None,
            "icons": None,
        }
        # Prefer an existing scrollbar candidate with the same bbox when present.
        for cand in vision.get("candidates") or []:
            if (
                str(cand.get("class_name") or "").strip() == "scrollbar"
                and _candidate_bbox_tuple(cand) == bbox
            ):
                scrollbar_dict = dict(cand)
                break

    percent = scrollbar_axis_percent(local_x, local_y, bbox)
    track = {
        "bbox": list(bbox),
        "axis": scrollbar_orientation(bbox),
        "percent": percent,
        "anchor_class": "scrollbar",
    }
    vision["scrollbar_track"] = track
    candidates = list(vision.get("candidates") or [])
    vision["candidates"] = _promote_scrollbar_in_candidates(candidates, scrollbar_dict)
    if vision.get("candidates"):
        vision["candidate_text"] = _candidate_text_from_dicts(vision["candidates"])
    return track


def scrollbar_track_percent_phrase(vision: dict[str, Any]) -> str | None:
    """Return ``的N%處`` when vision has a scrollbar track percent, else None."""
    track = vision.get("scrollbar_track")
    if not isinstance(track, dict):
        return None
    percent = track.get("percent")
    if not isinstance(percent, int) or isinstance(percent, bool):
        return None
    if percent < 0 or percent > 100:
        return None
    return f"的{percent}%處"


def run_pointer_event_yolo_ocr(
    event: RecordedEvent,
    *,
    run_dir: Path,
    persist_debug: bool = True,
) -> dict[str, Any]:
    """Run YOLO+OCR for a pointer event and persist ``yolo_ocr`` JSON (no LLM)."""
    empty: dict[str, Any] = {
        "used_vision": False,
        "candidate_text": "",
        "local_cursor": None,
        "candidates": [],
        "detection_count": 0,
    }
    if event.kind not in POINTER_EVENT_KINDS:
        return empty

    fingerprint = vision_source_fingerprint(event)

    if event.kind == "drag":
        start_local = _local_cursor(event)
        end_local = _local_end_cursor(event)
        if start_local is None or end_local is None:
            return empty

        end_image = event.end_screenshot_path or event.screenshot_path
        # Start/end frames are independent images — run YOLO+OCR concurrently.
        with ThreadPoolExecutor(max_workers=2) as pool:
            start_future = pool.submit(
                build_vision_context_at_point,
                event,
                local_x=start_local[0],
                local_y=start_local[1],
                run_dir=run_dir,
                persist_debug=persist_debug,
                reference_xy=event.cursor_xy,
                image_path=event.screenshot_path,
                debug_name="",
                source_fingerprint=fingerprint,
            )
            end_future = pool.submit(
                build_vision_context_at_point,
                event,
                local_x=end_local[0],
                local_y=end_local[1],
                run_dir=run_dir,
                persist_debug=persist_debug,
                reference_xy=event.end_xy,
                image_path=end_image,
                debug_name="_end",
                source_fingerprint=fingerprint,
            )
            start_result = start_future.result()
            end_result = end_future.result()
        start_compact = _compact_vision_point(start_result)
        end_compact = _build_filtered_destination_vision(
            end_result,
            end_local=end_local,
        )
        start_track = start_compact.get("scrollbar_track")
        if isinstance(start_track, dict) and "bbox" in start_track:
            annotate_scrollbar_track(
                end_compact,
                local_x=end_local[0],
                local_y=end_local[1],
                all_detections=end_result.get("all_detections")
                if isinstance(end_result.get("all_detections"), list)
                else None,
                require_point_inside=False,
                scrollbar_bbox=start_track["bbox"],
            )
        if persist_debug:
            write_json(
                run_dir / "yolo_ocr" / f"event_{event.index:03d}_end_filtered.json",
                {
                    "event_index": event.index,
                    "local_cursor": list(end_local),
                    "candidate_text": end_compact["candidate_text"],
                    "candidates": end_compact["candidates"],
                    "detection_count": end_compact.get("detection_count", 0),
                    "source_fingerprint": fingerprint,
                    **(
                        {"scrollbar_track": end_compact["scrollbar_track"]}
                        if isinstance(end_compact.get("scrollbar_track"), dict)
                        else {}
                    ),
                },
            )
        combined_error = start_compact.get("yolo_error") or end_result.get("yolo_error")
        return {
            **start_compact,
            "used_vision": bool(start_compact["used_vision"] or end_compact["used_vision"]),
            "destination": end_compact,
            "field_context": start_compact["field_context"],
            "destination_field_context": end_compact["field_context"],
            **({"yolo_error": combined_error} if combined_error else {}),
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
        source_fingerprint=fingerprint,
    )
    result.pop("bgr", None)
    result.pop("all_detections", None)
    result["field_context"] = format_field_context_hint(result)
    return result


async def build_vision_context(
    event: RecordedEvent,
    *,
    run_dir: Path,
    persist_debug: bool = True,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run YOLO+OCR for pointer events; return context for the LLM.

    Offloads blocking Triton HTTP to a worker thread so the event loop can
    overlap LLM work and other events' vision.
    """
    _ = log_info
    return await asyncio.to_thread(
        run_pointer_event_yolo_ocr,
        event,
        run_dir=run_dir,
        persist_debug=persist_debug,
    )
