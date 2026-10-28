from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from PIL import Image

if TYPE_CHECKING:
    import numpy as np

from cua_mcp.char_target import detect_clicked_char
from cua_mcp.geometry import point_in_rect_xywh
from cua_mcp.color_spatial_segment import (
    ColorSegmentResult,
    SegmentDetection,
    color_segment_to_json_dict,
    load_color_segment_params,
    reorder_detections_for_landmark,
    segment_image_by_color,
    spatial_region_rank_for_detections,
)
from cua_mcp.icon_map import is_pua_char, is_unknown_icon_record
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.scrollbar_arrows import (
    is_scrollbar_end_arrow_candidate,
    point_in_bbox,
    project_point_onto_scrollbar,
    scrollbar_axis_percent,
    scrollbar_orientation,
)
from cua_mcp.select_mouse_target import (
    _MOUSE_FILTER_SIMILARITY_THRESHOLD,
    _box_edge_gap,
    _detect_mouse_targets_from_bgr,
    _detection_similarity_to_query,
    _detections_from_uia_prefetched,
    _detections_label_similar,
    _detections_same_place,
    _prefilter_anchors_by_nearby,
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
    extract_nearby_hints_by_location,
    extract_nearby_hints_from_instruction,
    format_nearby_context_comment,
    landmark_cell_from_anchor_bbox,
    merge_nearby_hints,
    normalize_nearby_hints,
    side_from_anchor_bbox,
    side_to_schema_value,
    side_to_zh,
    strip_nearby_context_comments,
)
from src.recorder.models import (
    POINTER_EVENT_KINDS,
    RecordedEvent,
    screenshot_path_for_event,
    screenshot_path_for_event_end,
)

# Closest unique multi-char text/icon on each directed script side
# (four cardinals + four diagonals). INSIDE is reserved for container landmarks.
_DIRECTED_LOCATE_SIDES = frozenset(
    {
        Side.LEFT,
        Side.RIGHT,
        Side.ABOVE,
        Side.BELOW,
        Side.UPPER_LEFT,
        Side.UPPER_RIGHT,
        Side.LOWER_LEFT,
        Side.LOWER_RIGHT,
    }
)
_DIRECTED_SIDE_ORDER = (
    Side.LEFT,
    Side.RIGHT,
    Side.ABOVE,
    Side.BELOW,
    Side.UPPER_LEFT,
    Side.UPPER_RIGHT,
    Side.LOWER_LEFT,
    Side.LOWER_RIGHT,
)
_MIN_NEARBY_TEXT_CANDIDATES = 8
_MIN_NEARBY_ICON_CANDIDATES = 5
# Recording HTML「點擊目標」radio list: closest labeled candidates only.
_MAX_PRIMARY_TARGET_OPTIONS = 10
# Manual steps with no click point need a wider gallery than proximity-ranked picks.
_MAX_PICK_TARGET_OPTIONS = 80
# Recording HTML「附近地標」: max options per directed side (closest to click).
_MAX_NEARBY_LANDMARK_OPTIONS_PER_SIDE = 10
# Prefer at least this many multi-char text neighbors in each directional cell.
_MIN_MULTI_CHAR_TEXT_PER_SIDE = 2
_DRAG_OFFSET_THRESHOLD_PX = 5
# OCR boxes hug glyphs; pad so a near-miss click still counts as on-target.
_BBOX_HIT_TOLERANCE_PX = 4
_CONTAINER_LANDMARK_CLASSES = frozenset({"input", "scrollbar"})
# Icons that are themselves the clickable control. Only these beat overlapping
# OCR text in click/drop primary ranking; decorative icons (頭像, Chrome, …)
# lose to any visible text so labels like 「是」 win over misread glyph icons.
_CONTROL_CHROME_ICON_IDS = frozenset(
    {
        "收合節點",
        "展開節點",
        "方框、矩形框線",
        "向下V箭頭",
        "向上V箭頭",
        "空心圓圈、單選鈕",
        "單選鈕核取",
        "已勾選方框",
        "向上雙V箭頭",
        "向下雙V箭頭",
        "加號",
        "網格檢視",
    }
)
# Decorative / branding icons lose to any visible OCR text (including 1-char).
_DECORATIVE_ICON_IDS = frozenset(
    {
        "頭像、使用者頭像",
        "Chrome",
    }
)
# UIA Name/AutomationId strings longer than this (or URL-like) do not count
# toward recording landmark sufficiency. The check lives in uia_input_boxes.
# Sentinel: build_vision_context_at_point uses the event press snapshot unless
# the caller passes a snapshot (including None to force a live FindAll).
_UIA_SNAPSHOT_UNSET = object()
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
    if event.end_click_window is not None:
        payload["end_click_window"] = event.end_click_window
    if event.uia_snapshot is not None:
        payload["uia_snapshot"] = event.uia_snapshot
    if event.end_uia_snapshot is not None:
        payload["end_uia_snapshot"] = event.end_uia_snapshot
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


def _yolo_ocr_fallback_vision_from_payload(
    block: dict[str, Any] | None,
    *,
    local_cursor: Any = None,
) -> dict[str, Any] | None:
    """Rebuild a compact YOLO+OCR twin vision from a persisted fallback block."""
    if not isinstance(block, dict):
        return None
    candidates = block.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    local = block.get("local_cursor", local_cursor)
    if isinstance(local, (list, tuple)) and len(local) == 2:
        cursor: tuple[int, int] | list[Any] | None = (int(local[0]), int(local[1]))
    else:
        cursor = local
    twin: dict[str, Any] = {
        "used_vision": True,
        "candidate_text": str(block.get("candidate_text") or ""),
        "local_cursor": cursor,
        "candidates": candidates,
        "detection_count": block.get("detection_count", len(candidates)),
        "field_context": format_field_context_hint(
            {
                "candidates": candidates,
                "local_cursor": cursor,
            }
        ),
        "vision_path": "yolo_ocr",
    }
    track = block.get("scrollbar_track")
    if isinstance(track, dict):
        twin["scrollbar_track"] = track
    return twin


def _persistable_yolo_ocr_fallback(vision: dict[str, Any] | None) -> dict[str, Any] | None:
    """JSON block stored beside primary candidates for a durable OCR twin."""
    if not isinstance(vision, dict):
        return None
    candidates = vision.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    block: dict[str, Any] = {
        "candidates": candidates,
        "candidate_text": str(vision.get("candidate_text") or ""),
        "detection_count": vision.get("detection_count", len(candidates)),
        "vision_path": "yolo_ocr",
    }
    local = vision.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        block["local_cursor"] = [int(local[0]), int(local[1])]
    track = vision.get("scrollbar_track")
    if isinstance(track, dict):
        block["scrollbar_track"] = track
    return block


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
    _copy_recorded_vision_mode(payload, vision)
    twin = _yolo_ocr_fallback_vision_from_payload(
        payload.get("yolo_ocr_fallback")
        if isinstance(payload.get("yolo_ocr_fallback"), dict)
        else None,
        local_cursor=local_cursor,
    )
    if twin is not None:
        vision["yolo_ocr_fallback_vision"] = twin
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
                end_compact,
                omit_when_inside=False,
            )
            # The filtered file is a candidate subset. The mode lives on the
            # full end payload, including a side later marked UIA by repair.
            _copy_recorded_vision_mode(end_payload, end_compact)
            twin = _yolo_ocr_fallback_vision_from_payload(
                filtered_payload.get("yolo_ocr_fallback")
                if isinstance(filtered_payload.get("yolo_ocr_fallback"), dict)
                else (
                    end_payload.get("yolo_ocr_fallback")
                    if isinstance(end_payload.get("yolo_ocr_fallback"), dict)
                    else None
                ),
                local_cursor=dest_local,
            )
            if twin is not None:
                end_compact["yolo_ocr_fallback_vision"] = twin
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


def _icons_include_control_chrome(icons: Any) -> bool:
    """True when any icon ``chinese_id`` / ``id`` is a whitelisted control affordance."""
    if not isinstance(icons, list):
        return False
    for icon in icons:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label in _CONTROL_CHROME_ICON_IDS:
            return True
    return False


def _icons_are_decorative_only(icons: Any) -> bool:
    """True when every known icon id is decorative/branding (no control chrome)."""
    if not isinstance(icons, list) or not icons:
        return False
    saw_known = False
    for icon in icons:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if not label:
            continue
        saw_known = True
        if label in _CONTROL_CHROME_ICON_IDS:
            return False
        if label not in _DECORATIVE_ICON_IDS:
            return False
    return saw_known


def _usable_uia_landmark_label(text: str | None) -> bool:
    """True when a UIA string is short/stable enough to seed recording landmarks.

    Rejects URL-like names and ultra-long AutomationIds that replay soft-match
    poorly and should not count toward the UIA-sufficiency gate.
    """
    from src.recorder.uia_input_boxes import is_usable_uia_landmark_label

    return is_usable_uia_landmark_label(_visible_text(text))


def _detection_source(det: UiDetection) -> str:
    """Normalize detection provenance for ranking (uia < ocr < other)."""
    return str(getattr(det, "source", None) or "").strip().lower()


def _source_rank_for_label(source: str | None) -> int:
    """Lower is better: prefer UIA labels over OCR over unlabeled YOLO boxes."""
    key = str(source or "").strip().lower()
    if key == "uia":
        return 0
    if key == "ocr":
        return 1
    return 2


def _is_usable_uia_text_detection(det: UiDetection) -> bool:
    """True for UIA-sourced text with a usable landmark label."""
    return (
        _detection_source(det) == "uia"
        and det.class_id == YOLO_CLASS_TEXT
        and _usable_uia_landmark_label(det.text)
    )


def _is_uia_textless_control(det: UiDetection) -> bool:
    """True for a UIA input or scrollbar. These controls need no label."""
    return _detection_source(det) == "uia" and det.class_id in (
        YOLO_CLASS_INPUT,
        YOLO_CLASS_SCROLLBAR,
    )


def _detection_contains_click(
    det: UiDetection,
    local_x: int,
    local_y: int,
) -> bool:
    """True when ``(local_x, local_y)`` lies in ``det.bbox`` with OCR hit padding."""
    pad = _BBOX_HIT_TOLERANCE_PX
    bx, by, bw, bh = det.bbox
    return _point_inside_bbox(
        local_x,
        local_y,
        (bx - pad, by - pad, bw + 2 * pad, bh + 2 * pad),
    )


def _hit_content_priority(det: UiDetection) -> int:
    """Lower is better when several boxes share the same click distance.

    Priority:
    1. text with visible length > 1
    2. control-chrome icon (checkbox, radio, tree toggle, chevron, 網格檢視, …)
    3. other non-decorative icons (beat single-char OCR hallucinations)
    4. any other text (including single-char)
    5. decorative icons
    6. others (scrollbar, input, empty element, …)
    """
    if _is_multi_char_text_detection(det):
        return 0
    if _icons_include_control_chrome(det.icons):
        return 1
    if _is_icon_detection(det) and not _icons_are_decorative_only(det.icons):
        return 2
    if _is_single_char_text_detection(det):
        return 3
    if _is_icon_detection(det):
        return 4
    return 5


def _candidate_hit_content_priority(candidate: dict[str, Any]) -> int:
    """Same ranking as ``_hit_content_priority`` for vision candidate dicts."""
    class_name = str(candidate.get("class_name") or "").strip()
    visible = _visible_text(candidate.get("text"))
    icons = candidate.get("icons")
    if class_name == "text" and len(visible) > 1:
        return 0
    if _icons_include_control_chrome(icons):
        return 1
    if icons and not _icons_are_decorative_only(icons):
        return 2
    if class_name == "text" and visible:
        return 3
    if icons:
        return 4
    return 5


def _nearest_candidate_rank_key(
    bbox: tuple[int, int, int, int],
    local_x: int,
    local_y: int,
    *,
    content_priority: int = 5,
    source_rank: int = 2,
) -> tuple[float, int, int, int]:
    """Sort key: distance, content priority, source (UIA first), then area."""
    return (
        _point_to_bbox_distance_sq(local_x, local_y, bbox),
        content_priority,
        source_rank,
        _bbox_area(bbox),
    )


def _nearest_detection_rank_key(
    det: UiDetection,
    local_x: int,
    local_y: int,
) -> tuple[float, int, int, int]:
    """Sort key for a detection at a click/drop point."""
    return _nearest_candidate_rank_key(
        det.bbox,
        local_x,
        local_y,
        content_priority=_hit_content_priority(det),
        source_rank=_source_rank_for_label(_detection_source(det)),
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
    cell = landmark_cell_from_anchor_bbox(
        primary_bbox,
        det.cx,
        det.cy,
        landmark_bbox=det.bbox,
    )
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
    multi-char text, then control-chrome icons, then other text, then other
    icons, then others; within a tier, prefer the smallest bbox.

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
    ix, iy, iw, ih = inner
    return _point_inside_bbox(ix + iw // 2, iy + ih // 2, outer)


def _point_inside_bbox(x: int, y: int, bbox: tuple[int, int, int, int]) -> bool:
    """True when ``(x, y)`` lies inside an axis-aligned ``(x, y, w, h)`` bbox."""
    return point_in_rect_xywh(x, y, bbox)


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

    Uses the same content priority as click ranking (multi-char text,
    control-chrome icons, other text, other icons, then others), with smallest
    area as the final tie-breaker.
    """
    hits = [
        det
        for det in all_detections
        if det.class_id in (YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT)
        and _point_inside_bbox(x, y, det.bbox)
    ]
    if not hits:
        return None
    return min(
        hits,
        key=lambda det: (
            _hit_content_priority(det),
            _source_rank_for_label(_detection_source(det)),
            _bbox_area(det.bbox),
        ),
    )


def _candidate_containing_click(
    candidates: list[dict[str, Any]],
    local_x: int,
    local_y: int,
) -> dict[str, Any] | None:
    """Return the best candidate whose padded bbox contains the click, if any.

    Prefers multi-char text, then control-chrome icons, then other non-decorative
    icons, then other text, then decorative icons, then others; UIA before OCR;
    smallest area breaks ties. Includes inputs/scrollbars so an on-field click
    keeps that container as the primary anchor.
    """
    hits = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and _drop_point_inside_candidate(local_x, local_y, candidate)
    ]
    if not hits:
        return None

    def _hit_key(candidate: dict[str, Any]) -> tuple[int, int, int]:
        bbox = _as_bbox_xywh(candidate.get("bbox"))
        area = _bbox_area(bbox) if bbox is not None else 0
        return (
            _candidate_hit_content_priority(candidate),
            _source_rank_for_label(
                str(candidate.get("source") or "").strip().lower() or None
            ),
            area,
        )

    return min(hits, key=_hit_key)


def _click_landmark_primary_rank_key(
    candidate: dict[str, Any],
    *,
    local_x: int,
    local_y: int,
    click_bbox: tuple[int, int, int, int],
) -> tuple[int, int, float, int] | None:
    """Sort key for landmark-style primary pick around the click point.

    Matches nearby-landmark preference: multi-char text first, then other
    labels, then icons; within Tier 0, prefer left/right/top/bottom cells
    relative to the click; then closer edge distance and smaller area.
    Returns ``None`` when the candidate has no meaningful hub label.
    """
    label = _candidate_label_for_hint(candidate)
    if not label:
        return None
    bbox = _as_bbox_xywh(candidate.get("bbox"))
    if bbox is None:
        return None
    tier = _nearby_hint_tier(candidate, label)
    cell_rank = (
        _tier0_cell_rank(candidate, primary_bbox=click_bbox) if tier == 0 else 0
    )
    dist_sq = _point_to_bbox_distance_sq(local_x, local_y, bbox)
    return tier, cell_rank, dist_sq, _bbox_area(bbox)


def select_click_primary_candidate(
    candidates: list[dict[str, Any]],
    local_x: int,
    local_y: int,
) -> dict[str, Any] | None:
    """Choose the click instruction primary using the click point as reference.

    1. If the click lies inside any candidate (with OCR hit padding), keep that
       containing target (content-priority + smallest area).
    2. Otherwise pick a single nearby-style landmark around the click point
       (multi-char text preferred, sides relative to the click).
    """
    if not candidates:
        return None
    containing = _candidate_containing_click(candidates, local_x, local_y)
    if containing is not None:
        return containing

    click_bbox = (local_x, local_y, 1, 1)
    best: dict[str, Any] | None = None
    best_key: tuple[int, int, float, int] | None = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = _click_landmark_primary_rank_key(
            candidate,
            local_x=local_x,
            local_y=local_y,
            click_bbox=click_bbox,
        )
        if key is None:
            continue
        if best_key is None or key < best_key:
            best = candidate
            best_key = key
    return best


def promote_click_primary_candidates(
    candidates: list[dict[str, Any]],
    local_x: int,
    local_y: int,
) -> list[dict[str, Any]]:
    """Return candidates with the click-anchor primary moved to index 0.

    Preserves relative order of the remaining entries. When no better primary
    is found, returns ``candidates`` unchanged (same object if already correct).
    """
    if not candidates:
        return candidates
    primary = select_click_primary_candidate(candidates, local_x, local_y)
    if primary is None:
        return candidates
    try:
        index = next(i for i, c in enumerate(candidates) if c is primary)
    except StopIteration:
        # Equality fallback when callers pass copies.
        primary_bbox = primary.get("bbox")
        index = next(
            (
                i
                for i, c in enumerate(candidates)
                if isinstance(c, dict) and c.get("bbox") == primary_bbox
            ),
            0,
        )
    if index == 0:
        return candidates
    return [candidates[index], *candidates[:index], *candidates[index + 1 :]]


def ensure_click_primary_candidate(vision: dict[str, Any]) -> dict[str, Any]:
    """Reorder ``vision["candidates"]`` so the click-landmark primary is first.

    Mutates ``vision`` in place when a local cursor and candidates are present.
    """
    local = vision.get("local_cursor")
    candidates = vision.get("candidates")
    if not isinstance(local, (list, tuple)) or len(local) != 2:
        return vision
    if not isinstance(candidates, list) or not candidates:
        return vision
    promoted = promote_click_primary_candidates(
        candidates,
        int(local[0]),
        int(local[1]),
    )
    if promoted is not candidates:
        vision["candidates"] = promoted
    return vision


def _promote_click_primary_detections(
    detections: list[UiDetection],
    local_x: int,
    local_y: int,
) -> list[UiDetection]:
    """Reorder YOLO detections with the click-landmark primary first."""
    if not detections:
        return detections
    as_dicts = [_detection_to_dict(det) for det in detections]
    promoted = promote_click_primary_candidates(as_dicts, local_x, local_y)
    if not promoted or promoted[0] is as_dicts[0]:
        return detections
    primary_bbox = tuple(int(v) for v in promoted[0]["bbox"])
    index = next(
        (i for i, det in enumerate(detections) if tuple(det.bbox) == primary_bbox),
        0,
    )
    if index == 0:
        return detections
    return [detections[index], *detections[:index], *detections[index + 1 :]]


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
    from src.recorder.uia_input_boxes import UIA_CONTROL_TYPE_SUFFIX

    uia_suffix = UIA_CONTROL_TYPE_SUFFIX.get(
        str(candidate.get("uia_control_type") or "").strip(),
        "",
    )

    visible = _visible_text(candidate.get("text"))
    if visible and uia_suffix:
        return f"「{visible}」{uia_suffix}"
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

    Bare class labels ``輸入欄`` and ``滾動條`` are kept so empty inputs and
    scrollbars can be selected in recording HTML. ``未知`` labels are kept here
    so an unknown click target can still be named; nearby-landmark pickers drop
    them via ``_is_unknown_nearby_landmark``. Generic ``文字`` / ``元素`` /
    ``按鈕`` alone are still dropped.
    """
    anchor = format_drag_candidate_anchor(candidate)
    if anchor is None:
        return None
    generic_only = {"文字", "元素", "按鈕"}
    if anchor in generic_only:
        return None
    return anchor


def _is_unknown_nearby_landmark(label: str, candidate: dict[str, Any]) -> bool:
    """True when a neighbor is an unknown-class detection and must not be a landmark.

    Unknown boxes (including PUA-only OCR such as 「」未知) are not stable
    references at playback, so they are excluded from auto-picked nearby comments
    and from the recording HTML landmark list.
    """
    if str(candidate.get("class_name") or "").strip() == "unknown":
        return True
    return not label or label == "未知" or label.endswith("未知")


def _is_stable_disambiguation_label(label: str, candidate: dict[str, Any]) -> bool:
    """True when ``label`` is stable enough to auto-pick as a nearby landmark.

    Drops unknown-class anchors, PUA-containing phrases, and quoted OCR that
    starts with underscore noise (e.g. 「_未分類黏署」).
    """
    if _is_unknown_nearby_landmark(label, candidate):
        return False
    if any(is_pua_char(ch) for ch in label):
        return False
    if label.startswith("「") and "」" in label:
        inner = label[1 : label.index("」")]
        if inner.startswith("_"):
            return False
    return True


def _label_already_in_instruction(label: str, instruction: str) -> bool:
    """True when the label's quoted name or full phrase already appears in the instruction."""
    if label in instruction:
        return True
    if label.startswith("「") and "」" in label:
        inner = label.split("」", 1)[0][1:]
        if inner and f"「{inner}」" in instruction:
            return True
    return False


def _label_frequencies(candidates: list[Any]) -> dict[str, int]:
    """Count how often each hint label appears among ``candidates``.

    A label occurring more than once cannot be pinned down at playback: the side
    check passes against *any* matching detection, so the hint stops identifying
    a specific landmark.
    """
    freq: dict[str, int] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if label:
            freq[label] = freq.get(label, 0) + 1
    return freq


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
    cell = landmark_cell_from_anchor_bbox(
        primary_bbox,
        center[0],
        center[1],
        landmark_bbox=_as_bbox_xywh(candidate.get("bbox")),
    )
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
            return side_from_anchor_bbox(
                primary_bbox,
                center[0],
                center[1],
                landmark_bbox=cand_bbox,
            )
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



def _is_known_icon_candidate(candidate: dict[str, Any], label: str) -> bool:
    """True when ``candidate`` is a known (non-unknown) icon landmark."""
    if not label.endswith("圖示"):
        return False
    icons = candidate.get("icons")
    if not isinstance(icons, list) or not icons:
        return False
    for icon in icons:
        if not isinstance(icon, dict):
            continue
        chinese_id = str(icon.get("chinese_id") or "").strip()
        if not chinese_id:
            continue
        if is_unknown_icon_record(icon):
            continue
        return True
    return False


@dataclass(frozen=True)
class _LocateLandmark:
    """One directed neighbor eligible as a closest-per-side landmark."""

    hint: NearbyHint
    gap: float
    is_text: bool


def _eligible_locate_landmarks(
    candidates: list[Any],
    *,
    instruction: str,
    primary: UiDetection,
    primary_bbox: tuple[int, int, int, int] | None,
    click_xy: tuple[int, int] | None,
    reserved_labels: set[str],
) -> list[_LocateLandmark]:
    """Unique directed neighbors stable enough to auto-pick as landmarks."""
    label_freq = _label_frequencies(candidates)
    seen: set[str] = set(reserved_labels)
    pool: list[_LocateLandmark] = []
    for candidate in candidates[1:]:
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if not _is_stable_disambiguation_label(label, candidate):
            continue
        if label_freq.get(label, 1) != 1:
            continue
        if _label_already_in_instruction(label, instruction):
            continue
        det = _dict_to_detection(candidate)
        if det is None or _detections_same_place(primary, det):
            continue
        is_text = _is_multi_char_text_candidate(candidate, label)
        is_icon = _is_known_icon_candidate(candidate, label)
        if not is_text and not is_icon:
            continue
        side = _neighbor_side_for_candidate(
            candidate,
            primary_bbox=primary_bbox,
            click_xy=click_xy,
        )
        if side is None:
            continue
        seen.add(label)
        pool.append(
            _LocateLandmark(
                hint=NearbyHint(label=label, side=side),
                gap=_box_edge_gap(primary.bbox, det.bbox),
                is_text=is_text,
            )
        )
    pool.sort(key=lambda item: (0 if item.is_text else 1, item.gap, item.hint.label))
    return pool


def _closest_directed_landmarks(
    pool: list[_LocateLandmark],
) -> list[_LocateLandmark]:
    """Closest unique multi-char text per side; else closest known icon on that side."""
    best: dict[Side, _LocateLandmark] = {}
    for item in pool:
        side = item.hint.side
        if side is None or side not in _DIRECTED_LOCATE_SIDES:
            continue
        prev = best.get(side)
        if prev is None:
            best[side] = item
            continue
        # Text always beats icon on the same side; among the same kind, closer wins.
        item_key = (0 if item.is_text else 1, item.gap, item.hint.label)
        prev_key = (0 if prev.is_text else 1, prev.gap, prev.hint.label)
        if item_key < prev_key:
            best[side] = item
    return [best[side] for side in _DIRECTED_SIDE_ORDER if side in best]


def _candidates_to_detections(candidates: list[Any]) -> list[UiDetection]:
    """Convert persisted candidate dicts to ``UiDetection`` rows (skip invalid)."""
    out: list[UiDetection] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        det = _dict_to_detection(candidate)
        if det is not None:
            out.append(det)
    return out


def _ambiguous_primary_peers(
    primary: UiDetection,
    detections: list[UiDetection],
) -> list[UiDetection]:
    """Same-class inputs/scrollbars or same-value text/icon peers of ``primary``."""
    peers: list[UiDetection] = []
    for det in detections:
        if det is primary or _detections_same_place(primary, det):
            continue
        if primary.class_name in ("input", "scrollbar"):
            if det.class_name == primary.class_name:
                peers.append(det)
            continue
        if _detections_label_similar(primary, det):
            peers.append(det)
    return peers


def _nearby_matches_for_hints(
    detections: list[UiDetection],
    hints: list[NearbyHint],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """Detections whose labels match any hint (replay nearby pool)."""
    if not detections or not hints:
        return []
    matches: list[UiDetection] = []
    seen: set[int] = set()
    for det in detections:
        if id(det) in seen:
            continue
        if any(
            _detection_similarity_to_query(det, hint.label) >= threshold for hint in hints
        ):
            seen.add(id(det))
            matches.append(det)
    return matches


def _prefilter_kept_among_peers(
    primary: UiDetection,
    peers: list[UiDetection],
    detections: list[UiDetection],
    hints: list[NearbyHint],
) -> tuple[int, bool]:
    """Return ``(kept_count, uniquely_primary)`` from replay nearby prefilter."""
    anchors = [primary, *peers]
    if not hints:
        return len(anchors), False
    nearby_matches = _nearby_matches_for_hints(detections, hints)
    kept = _prefilter_anchors_by_nearby(anchors, nearby_matches, hints)
    if not kept:
        return 0, False
    unique = len(kept) == 1 and (
        kept[0] is primary or _detections_same_place(kept[0], primary)
    )
    return len(kept), unique


def _hints_uniquely_prefilter_primary(
    primary: UiDetection,
    peers: list[UiDetection],
    detections: list[UiDetection],
    hints: list[NearbyHint],
) -> bool:
    """True when replay ``_prefilter_anchors_by_nearby`` keeps only ``primary``."""
    _kept, unique = _prefilter_kept_among_peers(primary, peers, detections, hints)
    return unique


def _grow_locating_hints_for_prefilter_uniqueness(
    *,
    primary: UiDetection,
    peers: list[UiDetection],
    detections: list[UiDetection],
    prefix_hints: list[NearbyHint],
    seed: list[NearbyHint],
    pool: list[_LocateLandmark],
) -> list[NearbyHint]:
    """Greedily add pool landmarks until prefilter keeps only ``primary``.

    Walks the full eligible ``pool`` (no addition cap). Returns the locating
    portion (not ``prefix_hints``). On failure, returns the best-effort locating
    set with the smallest kept peer set (seed if none improve).
    """
    used_labels = {hint.label for hint in prefix_hints} | {hint.label for hint in seed}
    current = list(seed)
    best = list(seed)
    best_kept, unique = _prefilter_kept_among_peers(
        primary, peers, detections, [*prefix_hints, *current]
    )
    if unique:
        return current
    if best_kept <= 0:
        best_kept = len(peers) + 1

    for item in pool:
        label = item.hint.label
        if not label or label in used_labels:
            continue
        trial = [*current, item.hint]
        kept, trial_unique = _prefilter_kept_among_peers(
            primary, peers, detections, [*prefix_hints, *trial]
        )
        if trial_unique:
            return trial
        current_kept, _ = _prefilter_kept_among_peers(
            primary, peers, detections, [*prefix_hints, *current]
        )
        if current_kept <= 0:
            current_kept = len(peers) + 1
        if kept <= 0 or kept >= current_kept:
            continue
        current = trial
        used_labels.add(label)
        if kept < best_kept:
            best = list(trial)
            best_kept = kept
    return best


def _pick_minimal_locating_hints(
    candidates: list[Any],
    *,
    instruction: str,
    reserved_labels: set[str],
    primary_bbox: tuple[int, int, int, int] | None,
    click_xy: tuple[int, int] | None,
    prefix_hints: list[NearbyHint] | None = None,
) -> list[NearbyHint]:
    """Closest unique multi-char text per side (else icon), plus spare neighbors.

    Spare budget starts at 1 and grows by one for each directed side that has
    neither a unique multi-char text nor a known icon. Spares prefer unused
    multi-char text (closest first); when none remain, unused known icons fill
    the rest.

    When the primary has ambiguous peers (same-value text/icon, or multiple
    inputs/scrollbars), grow the seed with more eligible landmarks until replay's
    ``_prefilter_anchors_by_nearby`` keeps only the primary. ``prefix_hints``
    (forced container / scrollbar-content) count toward that check but are not
    returned. If uniqueness cannot be proven after exhausting the eligible pool,
    keep the best-effort set (smallest kept peer set, else the original seed).
    """
    if len(candidates) < 2 or not isinstance(candidates[0], dict):
        return []
    primary = _dict_to_detection(candidates[0])
    if primary is None:
        return []
    pool = _eligible_locate_landmarks(
        candidates,
        instruction=instruction,
        primary=primary,
        primary_bbox=primary_bbox,
        click_xy=click_xy,
        reserved_labels=reserved_labels,
    )
    directed = _closest_directed_landmarks(pool)
    picked = [item.hint for item in directed]
    filled_sides = {item.hint.side for item in directed if item.hint.side is not None}
    empty_side_count = sum(
        1 for side in _DIRECTED_SIDE_ORDER if side not in filled_sides
    )
    spare_count = 1 + empty_side_count
    used_labels = {hint.label for hint in picked}
    # Pool is text-before-icon, closest-first; take unused texts then icons.
    for item in pool:
        if spare_count <= 0:
            break
        label = item.hint.label
        if not label or label in used_labels:
            continue
        picked.append(item.hint)
        used_labels.add(label)
        spare_count -= 1

    detections = _candidates_to_detections(candidates)
    # Re-bind primary to the detection instance from ``detections`` so peer
    # identity checks share object identity with the prefilter anchor list.
    primary_det = next(
        (
            det
            for det in detections
            if _detections_same_place(primary, det)
        ),
        primary,
    )
    peers = _ambiguous_primary_peers(primary_det, detections)
    if not peers:
        return picked

    prefix = list(prefix_hints or ())
    return _grow_locating_hints_for_prefilter_uniqueness(
        primary=primary_det,
        peers=peers,
        detections=detections,
        prefix_hints=prefix,
        seed=picked,
        pool=pool,
    )


def _ranked_eligible_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
    reserved_labels: set[str] | None = None,
) -> list[NearbyHint]:
    """All eligible directed neighbors ordered closest-first (text before icons)."""
    candidates = vision.get("candidates") or []
    if len(candidates) < 2 or not isinstance(candidates[0], dict):
        return []
    primary = candidates[0]
    primary_bbox = _as_bbox_xywh(primary.get("bbox"))
    click_xy = _click_xy_from_vision(vision, primary=primary)
    primary_det = _dict_to_detection(primary)
    if primary_det is None:
        return []
    pool = _eligible_locate_landmarks(
        candidates,
        instruction=instruction,
        primary=primary_det,
        primary_bbox=primary_bbox,
        click_xy=click_xy,
        reserved_labels=set(reserved_labels or ()),
    )
    # Also surface unique non-text/non-icon neighbors (e.g. bare 輸入欄 not containing
    # the click) for the uncapped HTML ranking list.
    label_freq = _label_frequencies(candidates)
    seen = {item.hint.label for item in pool} | set(reserved_labels or ())
    extras: list[tuple[float, NearbyHint]] = []
    for candidate in candidates[1:]:
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if _is_unknown_nearby_landmark(label, candidate):
            continue
        if _label_already_in_instruction(label, instruction):
            continue
        if label_freq.get(label, 1) > 1 and not label.endswith(("輸入欄", "滾動條")):
            continue
        det = _dict_to_detection(candidate)
        if det is None or _detections_same_place(primary_det, det):
            continue
        side = _neighbor_side_for_candidate(
            candidate,
            primary_bbox=primary_bbox,
            click_xy=click_xy,
        )
        gap = _box_edge_gap(primary_det.bbox, det.bbox)
        seen.add(label)
        extras.append((gap, NearbyHint(label=label, side=side)))
    extras.sort(key=lambda row: (row[0], row[1].label))
    return [item.hint for item in pool] + [hint for _gap, hint in extras]


def list_prioritized_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
) -> list[NearbyHint]:
    """Return all nearby hints in rank order, with no auto-pick cap.

    Containing ``input`` / ``scrollbar`` landmarks come first (``side=inside``),
    then every other eligible neighbor closest-first (multi-char text before
    known icons).
    """
    candidates = vision.get("candidates") or []
    if len(candidates) < 2 or not isinstance(candidates[0], dict):
        return []
    primary = candidates[0]
    click_xy = _click_xy_from_vision(vision, primary=primary)
    forced = _collect_containing_container_hints(
        candidates,
        instruction=instruction,
        click_xy=click_xy,
    )
    ranked = _ranked_eligible_nearby_hints(
        vision,
        instruction=instruction,
        reserved_labels={hint.label for hint in forced},
    )
    return [*forced, *ranked]


def _collect_scrollbar_adjacent_content_hints(
    candidates: list[Any],
    *,
    primary: dict[str, Any],
    primary_bbox: tuple[int, int, int, int] | None,
    click_xy: tuple[int, int] | None,
    instruction: str,
    reserved_labels: set[str],
) -> list[NearbyHint]:
    """Force-include scrollable content beside a primary scrollbar as a landmark.

    Content used to be baked into ``「X」文字區域的滾動條``; it now lives in the
    nearby parenthetical so the anchor stays bare ``滾動條``.
    """
    if primary.get("class_name") != "scrollbar" or primary_bbox is None:
        return []
    adjacent = _scrollbar_adjacent_content_labels(
        [c for c in candidates if isinstance(c, dict)],
        primary_bbox,
    )
    if not adjacent:
        return []
    target_raw = adjacent[0]
    for candidate in candidates[1:]:
        if not isinstance(candidate, dict):
            continue
        if _candidate_meaningful_content_label(candidate) != target_raw:
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in reserved_labels:
            continue
        if _is_unknown_nearby_landmark(label, candidate):
            continue
        if instruction and _label_already_in_instruction(label, instruction):
            continue
        side = _neighbor_side_for_candidate(
            candidate,
            primary_bbox=primary_bbox,
            click_xy=click_xy,
        )
        return [NearbyHint(label=label, side=side)]
    # Fallback: synthesize a text label when OCR sits in the content strip but
    # was not kept as a separate neighbor candidate.
    synth = f"「{target_raw}」文字"
    if synth in reserved_labels:
        return []
    if instruction and _label_already_in_instruction(synth, instruction):
        return []
    _x, _y, w, h = primary_bbox
    default_side = Side.RIGHT if h >= w else Side.BELOW
    return [NearbyHint(label=synth, side=default_side)]


def collect_nearby_hints(
    vision: dict[str, Any],
    *,
    instruction: str,
) -> list[NearbyHint]:
    """Collect directed landmarks near the click.

    Forced containing ``input`` / ``scrollbar`` landmarks (``side=inside``) come
    first. When the primary target is a scrollbar, adjacent content is forced
    next. Then the closest unique multi-character text on each of the eight
    directed sides, falling back to the closest known icon when a side has no
    unique text. Spare budget starts at one and grows by one per empty directed
    side (no text and no icon); spares prefer unused multi-character text, then
    unused known icons. When the primary is ambiguous among same-value text/icon
    peers or multiple inputs/scrollbars, locating hints grow until replay's
    nearby prefilter keeps only the primary (best-effort if still tied).
    """
    candidates = vision.get("candidates") or []
    if len(candidates) < 2 or not isinstance(candidates[0], dict):
        return []
    primary = candidates[0]
    primary_bbox = _as_bbox_xywh(primary.get("bbox"))
    click_xy = _click_xy_from_vision(vision, primary=primary)
    forced = _collect_containing_container_hints(
        candidates,
        instruction=instruction,
        click_xy=click_xy,
    )
    content_hints = _collect_scrollbar_adjacent_content_hints(
        candidates,
        primary=primary,
        primary_bbox=primary_bbox,
        click_xy=click_xy,
        instruction=instruction,
        reserved_labels={hint.label for hint in forced},
    )
    reserved = {hint.label for hint in forced} | {hint.label for hint in content_hints}
    prefix_hints = [*forced, *content_hints]
    locating = _pick_minimal_locating_hints(
        candidates,
        instruction=instruction,
        reserved_labels=reserved,
        primary_bbox=primary_bbox,
        click_xy=click_xy,
        prefix_hints=prefix_hints,
    )
    return [*prefix_hints, *locating]


def collect_nearby_hint_labels(
    vision: dict[str, Any],
    *,
    instruction: str,
) -> list[str]:
    """Collect auto-picked nearby candidate labels, skipping the primary target."""
    return [
        hint.label
        for hint in collect_nearby_hints(vision, instruction=instruction)
    ]


def bbox_xywh_to_ltrb(
    bbox: tuple[int, int, int, int] | list[int],
) -> tuple[int, int, int, int]:
    """Convert ``(x, y, w, h)`` to ``(left, top, right, bottom)``."""
    x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return x, y, x + w, y + h


def primary_target_rect_ltrb(vision: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
    """Return the primary click-target bbox as LTRB, or None when unavailable."""
    if not isinstance(vision, dict):
        return None
    candidates = vision.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return None
    bbox = _as_bbox_xywh(candidates[0].get("bbox"))
    if bbox is None:
        return None
    return bbox_xywh_to_ltrb(bbox)


def same_class_peer_rects_ltrb(
    vision: dict[str, Any] | None,
    *,
    class_name: str | None = None,
) -> list[tuple[int, int, int, int]]:
    """Return LTRB boxes for every other ``input`` / ``scrollbar`` peer.

    When the primary (or ``class_name``) is ``input`` or ``scrollbar``, every
    other candidate of that class is returned so HTML overlays can show the
    full ambiguous set. The primary itself is omitted (drawn separately).
    """
    if not isinstance(vision, dict):
        return []
    candidates = vision.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return []
    primary = candidates[0]
    kind = (class_name or str(primary.get("class_name") or "")).strip()
    if kind not in _CONTAINER_LANDMARK_CLASSES:
        return []
    primary_bbox = _as_bbox_xywh(primary.get("bbox"))
    peers: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("class_name") or "").strip() != kind:
            continue
        bbox = _as_bbox_xywh(candidate.get("bbox"))
        if bbox is None:
            continue
        if primary_bbox is not None and bbox == primary_bbox:
            continue
        ltrb = bbox_xywh_to_ltrb(bbox)
        if ltrb in seen:
            continue
        seen.add(ltrb)
        peers.append(ltrb)
    return peers


def _hints_for_overlay(
    instruction: str,
    *,
    location: str = "附近",
    nearby_objects: list[Any] | None = None,
) -> list[NearbyHint]:
    """Hints from the instruction (by location) plus optional tool nearby_objects."""
    buckets = extract_nearby_hints_by_location(instruction or "")
    hints = list(buckets.get(location) or [])
    if location == "附近" and not hints:
        hints = extract_nearby_hints_from_instruction(instruction or "")
    if nearby_objects:
        hints = merge_nearby_hints(hints, normalize_nearby_hints(nearby_objects))
    return hints


def _labels_match_for_overlay(candidate_label: str, hint_label: str) -> bool:
    if candidate_label == hint_label:
        return True
    # Allow bare quoted names to match typed labels: 「A1」 ↔ 「A1」文字.
    bare = hint_label if hint_label.startswith("「") and hint_label.endswith("」") else None
    if bare and candidate_label.startswith(bare):
        return True
    bare_c = (
        candidate_label
        if candidate_label.startswith("「") and candidate_label.endswith("」")
        else None
    )
    return bool(bare_c and hint_label.startswith(bare_c))


def match_nearby_landmark_overlays(
    vision: dict[str, Any] | None,
    instruction: str,
    *,
    location: str = "附近",
    nearby_objects: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve instruction landmarks to screenshot-local LTRB overlays.

    Each item is ``{"label", "side", "rect"}`` where ``rect`` is LTRB and
    ``side`` is the schema string (or None). Matching prefers candidates whose
    directed side relative to the primary matches the hint.
    """
    if not isinstance(vision, dict):
        return []
    candidates = vision.get("candidates") or []
    if len(candidates) < 2 or not isinstance(candidates[0], dict):
        return []
    primary = candidates[0]
    primary_bbox = _as_bbox_xywh(primary.get("bbox"))
    click_xy = _click_xy_from_vision(vision, primary=primary)
    hints = _hints_for_overlay(
        instruction,
        location=location,
        nearby_objects=nearby_objects,
    )
    if not hints:
        return []

    overlays: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for hint in hints:
        if not hint.label or hint.label in seen_labels:
            continue
        best: tuple[float, int, dict[str, Any]] | None = None
        for order, candidate in enumerate(candidates[1:]):
            if not isinstance(candidate, dict):
                continue
            label = _candidate_label_for_hint(candidate)
            if not label or not _labels_match_for_overlay(label, hint.label):
                continue
            bbox = _as_bbox_xywh(candidate.get("bbox"))
            if bbox is None:
                continue
            center = _candidate_center(candidate)
            if center is None:
                continue
            if hint.side is not None:
                if primary_bbox is None:
                    continue
                if not anchor_satisfies_side(
                    primary_bbox,
                    center[0],
                    center[1],
                    hint.side,
                    landmark_bbox=bbox,
                ):
                    continue
            if click_xy is not None:
                dist_sq = _point_to_bbox_distance_sq(click_xy[0], click_xy[1], bbox)
            elif primary_bbox is not None:
                pcx = primary_bbox[0] + primary_bbox[2] // 2
                pcy = primary_bbox[1] + primary_bbox[3] // 2
                dist_sq = _point_to_bbox_distance_sq(pcx, pcy, bbox)
            else:
                dist_sq = float(order)
            row = (
                dist_sq,
                order,
                {
                    "label": hint.label,
                    "side": side_to_schema_value(hint.side),
                    "rect": bbox_xywh_to_ltrb(bbox),
                },
            )
            if best is None or row[0] < best[0] or (
                row[0] == best[0] and row[1] < best[1]
            ):
                best = row
        if best is not None:
            seen_labels.add(hint.label)
            overlays.append(best[2])
    return overlays


def resolve_click_overlay_geometry(
    vision: dict[str, Any] | None,
    instruction: str,
    *,
    location: str = "附近",
    nearby_objects: list[Any] | None = None,
) -> tuple[
    tuple[int, int, int, int] | None,
    list[dict[str, Any]],
    list[tuple[int, int, int, int]],
]:
    """Return ``(target_ltrb, landmark_overlays, same_class_peers)`` for HTML baking.

    ``same_class_peers`` lists every other ``input`` / ``scrollbar`` when the
    primary target is one of those classes.
    """
    target = primary_target_rect_ltrb(vision)
    landmarks = match_nearby_landmark_overlays(
        vision,
        instruction,
        location=location,
        nearby_objects=nearby_objects,
    )
    peers = same_class_peer_rects_ltrb(vision)
    return target, landmarks, peers


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

    # (spatial_rank, distance_sq, order, option) — closest per side for the HTML picker.
    pending: list[tuple[int, float, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for order, candidate in enumerate(candidates[1:]):
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label or label in seen:
            continue
        if _is_unknown_nearby_landmark(label, candidate):
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
        bbox = _as_bbox_xywh(candidate.get("bbox"))
        if click_xy is not None and bbox is not None:
            dist_sq = _point_to_bbox_distance_sq(click_xy[0], click_xy[1], bbox)
        else:
            dist_sq = float(order)
        spatial_rank = int(candidate.get("spatial_region_rank", 0))
        pending.append(
            (
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

    by_side: dict[str | None, list[tuple[int, float, int, dict[str, Any]]]] = {}
    for item in pending:
        side_key = item[3].get("side")
        if side_key is None or side_key == "":
            bucket: str | None = None
        else:
            bucket = str(side_key)
        by_side.setdefault(bucket, []).append(item)

    kept: list[tuple[int, float, int, dict[str, Any]]] = []
    for items in by_side.values():
        items.sort(key=lambda row: (row[0], row[1], row[2]))
        kept.extend(items[:_MAX_NEARBY_LANDMARK_OPTIONS_PER_SIDE])
    kept.sort(key=lambda row: (row[0], row[1], row[2]))
    return [option for _, _, _, option in kept]


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


def yolo_ocr_payload_has_candidates(payload: dict[str, Any] | None) -> bool:
    """True when a persisted YOLO/OCR payload has usable detections."""
    if not isinstance(payload, dict):
        return False
    if payload.get("yolo_error"):
        return False
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        return True
    count = payload.get("detection_count")
    return isinstance(count, int) and count > 0


def recording_pointer_yolo_ocr_failed(
    run_root: Path,
    event_index: int,
    kind: str,
    *,
    require_payload: bool = False,
) -> bool:
    """True when a pointer step would show 「YOLO/OCR 未偵測到目標」.

    Drag steps fail only when both the start and end payloads lack detections.
    ``require_payload`` skips events that never wrote a ``yolo_ocr`` file, so an
    automatic retry does not invent a run for steps vision never persisted.
    """
    if kind not in POINTER_EVENT_KINDS:
        return False
    start = load_yolo_ocr_payload(run_root, event_index, suffix="")
    end = (
        load_yolo_ocr_payload(run_root, event_index, suffix="_end")
        if kind == "drag"
        else None
    )
    if require_payload and start is None and end is None:
        return False
    if yolo_ocr_payload_has_candidates(start):
        return False
    if kind == "drag":
        return not yolo_ocr_payload_has_candidates(end)
    return True


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
    *,
    max_options: int | None = _MAX_PRIMARY_TARGET_OPTIONS,
    mark_current: bool = True,
) -> list[dict[str, Any]]:
    """Return selectable primary click/drag targets from ranked candidates.

    Each option is ``{"index", "label", "display"}``. Index 0 is the current
    primary when ``mark_current`` is True. Candidates without a meaningful
    hub-style label are omitted. When ``max_options`` is set, only that many
    leading candidates (``candidates`` order) are considered.
    """
    candidates = vision.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return []

    limited = candidates if max_options is None else candidates[:max_options]
    options: list[dict[str, Any]] = []
    for index, candidate in enumerate(limited):
        if not isinstance(candidate, dict):
            continue
        label = _candidate_label_for_hint(candidate)
        if not label:
            continue
        display = f"{label}（目前）" if mark_current and index == 0 else label
        options.append({"index": index, "label": label, "display": display})
    return options


def list_pick_target_options(vision: dict[str, Any]) -> list[dict[str, Any]]:
    """Return labeled detections for choosing a click point when ``cursor_xy`` is unset."""
    return list_primary_target_options(
        vision,
        max_options=_MAX_PICK_TARGET_OPTIONS,
        mark_current=False,
    )


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


def load_recording_pick_target_options(
    run_root: Path,
    event_index: int,
) -> list[dict[str, Any]]:
    """Load pick-target options for a pointer event that still has no ``cursor_xy``."""
    return list_pick_target_options(
        vision_from_yolo_ocr(run_root, event_index, suffix="")
    )


def local_to_global_xy(
    event: RecordedEvent,
    local_xy: tuple[int, int],
) -> tuple[int, int]:
    """Map screenshot-local coords to global desktop coords via ``monitor_offset``."""
    lx, ly = local_xy
    if event.monitor_offset is not None:
        ox, oy = event.monitor_offset
        return lx + ox, ly + oy
    return lx, ly


def candidate_bbox_key(candidate: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Return a comparable ``(x, y, w, h)`` key for matching re-detected boxes."""
    return _candidate_bbox_tuple(candidate)


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
    source_raw = candidate.get("source")
    source = str(source_raw).strip() if source_raw is not None else None
    control_type = str(candidate.get("uia_control_type") or "").strip()
    return UiDetection(
        bbox=(x, y, w, h),
        cx=cx,
        cy=cy,
        class_id=int(class_id) if class_id is not None else 0,
        class_name=str(candidate.get("class_name") or "element"),
        text=candidate.get("text") if candidate.get("text") is not None else None,
        icons=list(icons) if isinstance(icons, list) else None,
        source=source or None,
        uia_control_type=control_type or None,
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


def _is_scrollbar_track_percent_drag(
    instruction: str,
    vision: dict[str, Any],
    destination: dict[str, Any],
) -> bool:
    """True for ``從滾動條的N%處拖到M%處`` style track drags."""
    if not scrollbar_track_percent_phrase(vision):
        return False
    if not scrollbar_track_percent_phrase(destination):
        return False
    cleaned = strip_nearby_context_comments(instruction)
    return "滾動條" in cleaned and "拖到" in cleaned


def append_drag_nearby_context_comments(
    instruction: str,
    vision: dict[str, Any],
    destination: dict[str, Any],
    start_hints: list[NearbyHint] | None = None,
    end_hints: list[NearbyHint] | None = None,
) -> str:
    """Insert start nearby comment before 拖到; append destination comment at end.

    Scrollbar track-percent drags use a single trailing ``（在…）`` comment
    (merged start+end landmarks) instead of split 起點/終點 parentheticals.
    """
    if not vision.get("used_vision"):
        return instruction

    if start_hints is None:
        start_hints = collect_nearby_hints(vision, instruction=instruction)
    if end_hints is None:
        end_hints = collect_nearby_hints(destination, instruction=instruction)

    if _is_scrollbar_track_percent_drag(instruction, vision, destination):
        merged = merge_nearby_hints(start_hints, end_hints)
        comment = format_nearby_context_comment(merged, location="附近")
        if comment is None:
            return instruction
        return instruction + comment

    result = instruction
    start_comment = format_nearby_context_comment(
        start_hints,
        location="起點",
    )
    if start_comment and "拖到" in result:
        drag_at = result.index("拖到")
        result = result[:drag_at] + start_comment + result[drag_at:]

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


def format_drag_destination_offset_hints(
    destination: dict[str, Any],
    *,
    omit_when_inside: bool = True,
) -> str:
    """Format per-candidate drop offsets for the drag destination LLM prompt.

    ``omit_when_inside`` keeps click hints quiet when the pointer is inside the
    anchor box. Drag recording passes False so an in-box point still reports
    its center offset.
    """
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
        if omit_when_inside and _drop_point_inside_candidate(drop_x, drop_y, candidate):
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
    *,
    omit_when_inside: bool = True,
) -> str | None:
    """Return the offset phrase for an anchor named in a pointer instruction.

    Clicks pass the default and skip the phrase when the pointer is inside the
    anchor box. Drag recording passes ``omit_when_inside=False`` so replay can
    aim at the recorded point inside that box.
    """
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
        if omit_when_inside and _drop_point_inside_candidate(drop_x, drop_y, candidate):
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
    control_type = str(candidate.get("uia_control_type") or "").strip()
    return UiDetection(
        bbox=bbox,
        cx=int(center[0]),
        cy=int(center[1]),
        class_id=int(candidate.get("class_id", 0)),
        class_name=str(candidate.get("class_name", "")),
        text=candidate.get("text"),
        icons=candidate.get("icons"),
        source=str(candidate.get("source") or "").strip() or None,
        uia_control_type=control_type or None,
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
    vision["destination_offset_hints"] = format_drag_destination_offset_hints(
        vision,
        omit_when_inside=False,
    )
    _copy_recorded_vision_mode(end_result, vision)
    fallback_detections = end_result.get("yolo_ocr_fallback_detections")
    if isinstance(fallback_detections, list) and fallback_detections:
        twin_filtered = _destination_candidates(fallback_detections, end_local)
        if twin_filtered:
            twin_dicts = [_detection_to_dict(d) for d in twin_filtered]
            twin: dict[str, Any] = {
                "used_vision": True,
                "candidate_text": _format_ui_candidates_text(twin_filtered),
                "local_cursor": end_local,
                "candidates": twin_dicts,
                "detection_count": len(fallback_detections),
                "field_context": format_field_context_hint(
                    {
                        "candidates": twin_dicts,
                        "local_cursor": end_local,
                    }
                ),
                "vision_path": "yolo_ocr",
            }
            twin["destination_offset_hints"] = format_drag_destination_offset_hints(
                twin,
                omit_when_inside=False,
            )
            vision["yolo_ocr_fallback_vision"] = twin
    elif isinstance(end_result.get("yolo_ocr_fallback_vision"), dict):
        vision["yolo_ocr_fallback_vision"] = end_result["yolo_ocr_fallback_vision"]
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
    payload: dict[str, Any] = {
        "bbox": list(det.bbox),
        "center": [det.cx, det.cy],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "text": det.text,
        "icons": det.icons,
        "spatial_region_rank": int(spatial_region_rank),
    }
    source = _detection_source(det)
    if source:
        payload["source"] = source
    if det.uia_control_type:
        payload["uia_control_type"] = det.uia_control_type
    return payload


VISION_PATH_REPAIR_REASON = "重播修復改為 UIA"
VISION_PATH_UIA_NOT_UNIQUE_REASON = "UIA 地標無法唯一鎖定目標"
_VISION_PATHS = frozenset({"uia", "yolo_ocr"})
_VISION_SECONDS_KEYS = (
    "yolo_s",
    "ocr_s",
    "line_s",
    "uia_input_s",
    "uia_scrollbar_s",
    "uia_text_s",
    "icon_ocr_retry_s",
    "total_s",
    "capture_s",
)


def _copy_recorded_vision_mode(source: dict[str, Any] | None, dest: dict[str, Any]) -> None:
    """Copy ``vision_path`` and its reason onto a compact vision dict."""
    if not isinstance(source, dict):
        return
    path = source.get("vision_path")
    if path in _VISION_PATHS:
        dest["vision_path"] = path
    reason = source.get("vision_path_reason")
    if isinstance(reason, str) and reason.strip():
        dest["vision_path_reason"] = reason.strip()


def _attach_vision_path_fields(payload: dict[str, Any], vision_timing: dict[str, Any]) -> None:
    """Persist the recording vision decision next to the YOLO/OCR payload."""
    path = vision_timing.get("vision_path")
    if path in _VISION_PATHS:
        payload["vision_path"] = path
    reason = vision_timing.get("vision_path_reason")
    if isinstance(reason, str) and reason.strip():
        payload["vision_path_reason"] = reason.strip()
    else:
        payload.pop("vision_path_reason", None)
    _attach_vision_timing_fields(payload, vision_timing)


def _attach_vision_timing_fields(
    payload: dict[str, Any],
    vision_timing: dict[str, Any],
) -> None:
    """Copy YOLO/OCR/UIA second timings onto the persisted yolo_ocr payload."""
    for key in _VISION_SECONDS_KEYS:
        raw = vision_timing.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        payload[key] = round(float(raw), 3)
    yolo_s = payload.get("yolo_s")
    if isinstance(yolo_s, (int, float)):
        payload["yolo_elapsed_ms"] = round(float(yolo_s) * 1000.0, 3)
    ocr_s = payload.get("ocr_s")
    if isinstance(ocr_s, (int, float)):
        payload["ocr_elapsed_ms"] = round(float(ocr_s) * 1000.0, 3)


def _screen_xyxy_contains_point(
    x: int,
    y: int,
    xyxy: tuple[int, int, int, int] | list[int],
    *,
    pad: int = _BBOX_HIT_TOLERANCE_PX,
) -> bool:
    """True when ``(x, y)`` lies in a screen ``(left, top, right, bottom)`` box."""
    left, top, right, bottom = (int(v) for v in xyxy[:4])
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return False
    return _point_inside_bbox(
        int(x),
        int(y),
        (left - pad, top - pad, width + 2 * pad, height + 2 * pad),
    )


def _drag_ends_share_monitor(event: RecordedEvent) -> bool:
    """True when the drag release screenshot is on the press monitor."""
    if event.monitor_index is None or event.end_monitor_index is None:
        return True
    return int(event.monitor_index) == int(event.end_monitor_index)


def _usable_snapshot_text_contains_point(
    snapshot: dict[str, Any],
    x: int,
    y: int,
) -> bool:
    """True when a usable UIA text in a screen-xyxy snapshot covers ``(x, y)``."""
    texts = snapshot.get("texts")
    if not isinstance(texts, list):
        return False
    for item in texts:
        if not isinstance(item, dict):
            continue
        if not _usable_uia_landmark_label(str(item.get("text") or "")):
            continue
        box = item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        if _screen_xyxy_contains_point(x, y, box):
            return True
    return False


def press_scrollbar_screen_xyxy(
    event: RecordedEvent,
) -> tuple[int, int, int, int] | None:
    """Screen xyxy of the scrollbar under the press, when no usable text is there.

    A press on usable text stays a text target, so the release is not forced
    onto a bar. The smallest containing scrollbar wins.
    """
    if event.cursor_xy is None:
        return None
    snapshot = event.uia_snapshot
    if not isinstance(snapshot, dict):
        return None
    x, y = int(event.cursor_xy[0]), int(event.cursor_xy[1])
    if _usable_snapshot_text_contains_point(snapshot, x, y):
        return None
    bars = snapshot.get("scrollbars")
    if not isinstance(bars, list):
        return None
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for item in bars:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        box = (int(item[0]), int(item[1]), int(item[2]), int(item[3]))
        if not _screen_xyxy_contains_point(x, y, box):
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if best is None or area < best[0]:
            best = (area, box)
    if best is None:
        return None
    return best[1]


def _screen_xyxy_to_local_xywh(
    xyxy: tuple[int, int, int, int],
    monitor_offset: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert a screen ``(left, top, right, bottom)`` box to image-local xywh."""
    left, top, right, bottom = xyxy
    origin_x, origin_y = int(monitor_offset[0]), int(monitor_offset[1])
    return (left - origin_x, top - origin_y, right - left, bottom - top)


def _forced_scrollbar_covers_release(
    local_x: int,
    local_y: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    """True when a drag release is assigned to the press scrollbar.

    The cross-axis is snapped onto the bar. The along-axis may fall outside
    the track and the release still counts as that bar.
    """
    bx, by, bw, bh = (int(v) for v in bbox[:4])
    if bw <= 0 or bh <= 0:
        return False
    snapped_x, snapped_y = project_point_onto_scrollbar(local_x, local_y, bbox)
    if scrollbar_orientation((bx, by, bw, bh)) == "vertical":
        return bx <= snapped_x < bx + bw
    return by <= snapped_y < by + bh


def _uia_forced_scrollbar_detection(
    bbox: tuple[int, int, int, int],
) -> UiDetection:
    """Synthetic UIA scrollbar for a press-bar release that missed the track."""
    bx, by, bw, bh = (int(v) for v in bbox[:4])
    return UiDetection(
        bbox=(bx, by, bw, bh),
        cx=bx + max(bw, 0) // 2,
        cy=by + max(bh, 0) // 2,
        class_id=YOLO_CLASS_SCROLLBAR,
        class_name="scrollbar",
        source="uia",
    )


def _pick_uia_click_primary(
    detections: list[UiDetection],
    *,
    point_x: int,
    point_y: int,
    forced_scrollbar_bbox: tuple[int, int, int, int] | None = None,
) -> UiDetection | None:
    """Best UIA detection covering the click, or the forced press scrollbar."""
    covering = [
        det
        for det in detections
        if (
            _is_usable_uia_text_detection(det) or _is_uia_textless_control(det)
        )
        and _detection_contains_click(det, point_x, point_y)
    ]
    if covering:
        return min(
            covering,
            key=lambda det: _nearest_detection_rank_key(det, point_x, point_y),
        )
    if forced_scrollbar_bbox is None or not _forced_scrollbar_covers_release(
        point_x,
        point_y,
        forced_scrollbar_bbox,
    ):
        return None
    forced = _uia_forced_scrollbar_detection(forced_scrollbar_bbox)
    for det in detections:
        if (
            _is_uia_textless_control(det)
            and det.class_id == YOLO_CLASS_SCROLLBAR
            and _detections_same_place(det, forced)
        ):
            return det
    return forced


def _uia_landmarks_uniquely_select_primary(
    primary: UiDetection,
    detections: list[UiDetection],
    *,
    click_xy: tuple[int, int],
) -> bool:
    """True when UIA nearby landmarks uniquely prefilter ``primary`` among peers."""
    working = list(detections)
    if not any(
        det is primary or _detections_same_place(primary, det) for det in working
    ):
        working = [primary, *working]
    primary_det = next(
        (
            det
            for det in working
            if det is primary or _detections_same_place(primary, det)
        ),
        primary,
    )
    peers = _ambiguous_primary_peers(primary_det, working)
    if not peers:
        return True

    candidates: list[dict[str, Any]] = [_detection_to_dict(primary_det)]
    for det in working:
        if det is primary_det or _detections_same_place(primary_det, det):
            continue
        candidates.append(_detection_to_dict(det))
    pool = _eligible_locate_landmarks(
        candidates,
        instruction="",
        primary=primary_det,
        primary_bbox=primary_det.bbox,
        click_xy=click_xy,
        reserved_labels=set(),
    )
    seed = [item.hint for item in _closest_directed_landmarks(pool)]
    if _hints_uniquely_prefilter_primary(primary_det, peers, working, seed):
        return True
    grown = _grow_locating_hints_for_prefilter_uniqueness(
        primary=primary_det,
        peers=peers,
        detections=working,
        prefix_hints=[],
        seed=seed,
        pool=pool,
    )
    return _hints_uniquely_prefilter_primary(primary_det, peers, working, grown)


def recording_uia_skip_reason(
    detections: list[UiDetection],
    *,
    local_x: int | None = None,
    local_y: int | None = None,
    forced_scrollbar_bbox: tuple[int, int, int, int] | None = None,
) -> str | None:
    """Why a UIA snapshot cannot name the click, or ``None`` when it can.

    ``None`` means UIA alone can select the click target. Callers that have no
    snapshot use ``沒有 UIA 快照`` themselves.

    A usable text box under the pointer wins. Otherwise a UIA input or
    scrollbar under the pointer is enough; those controls have no label.
    ``forced_scrollbar_bbox`` is the press scrollbar for a drag release: the
    release counts as that bar after the cross-axis snap, even past the track.

    When the click has ambiguous peers (same-label text/icon, or multiple
    inputs/scrollbars), nearby unique-label UIA landmarks must uniquely
    prefilter the primary the same way replay nearby matching does.
    """
    if local_x is None or local_y is None:
        if any(
            _is_usable_uia_text_detection(det) or _is_uia_textless_control(det)
            for det in detections
        ):
            return None
        return "點擊處沒有可用的 UIA 文字"

    point_x, point_y = int(local_x), int(local_y)
    primary = _pick_uia_click_primary(
        detections,
        point_x=point_x,
        point_y=point_y,
        forced_scrollbar_bbox=forced_scrollbar_bbox,
    )
    if primary is None:
        return "點擊處沒有可用的 UIA 文字"
    if _uia_landmarks_uniquely_select_primary(
        primary,
        detections,
        click_xy=(point_x, point_y),
    ):
        return None
    return VISION_PATH_UIA_NOT_UNIQUE_REASON


def recording_uia_detections_sufficient(
    detections: list[UiDetection],
    *,
    local_x: int | None = None,
    local_y: int | None = None,
    forced_scrollbar_bbox: tuple[int, int, int, int] | None = None,
) -> bool:
    """True when deferred UIA-only candidates uniquely select the click target.

    The click must land on usable UIA text, a UIA input, or a UIA scrollbar (or
    a drag release assigned to ``forced_scrollbar_bbox``). When peers make that
    target ambiguous, unique-label UIA landmarks must uniquely prefilter it.
    URL-like / ultra-long UIA names do not count as usable text.
    """
    return (
        recording_uia_skip_reason(
            detections,
            local_x=local_x,
            local_y=local_y,
            forced_scrollbar_bbox=forced_scrollbar_bbox,
        )
        is None
    )


def mark_recorded_side_repaired_to_uia(
    run_root: Path,
    event_index: int,
    side: str,
) -> None:
    """Remember that replay repair named this side from UIA.

    The next cache re-analysis reads this payload, so the instruction stays
    ``（UIA）`` for the repaired side instead of restoring ``（YOLO+OCR）``.
    """
    suffix = "_end" if side == "end" else ""
    payload = load_yolo_ocr_payload(run_root, event_index, suffix=suffix)
    if not isinstance(payload, dict):
        return
    payload["vision_path"] = "uia"
    payload["vision_path_reason"] = VISION_PATH_REPAIR_REASON
    path = Path(run_root) / "yolo_ocr" / f"event_{int(event_index):03d}{suffix}.json"
    write_json(path, payload)


def _prefetched_uia_from_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    monitor_offset: tuple[int, int],
    image_w: int,
    image_h: int,
    ocr_roi: tuple[int, int, int, int] | None,
    window_screen_xywh: tuple[int, int, int, int] | None,
) -> tuple[
    list[tuple[int, int, int, int]] | None,
    list[tuple[int, int, int, int]] | None,
    list[tuple[tuple[int, int, int, int], str]],
] | None:
    """Image-local UIA boxes from a click-time snapshot. ``None`` means query live."""
    if not isinstance(snapshot, dict):
        return None
    from src.recorder.uia_input_boxes import uia_snapshot_image_local

    return uia_snapshot_image_local(
        snapshot,
        monitor_offset=monitor_offset,
        image_w=image_w,
        image_h=image_h,
        ocr_roi=ocr_roi,
        window_screen_xywh=window_screen_xywh,
    )


def _attach_unused_uia_debug(
    vision_timing: dict[str, Any],
    uia_prefetched: tuple[
        list[tuple[int, int, int, int]] | None,
        list[tuple[int, int, int, int]] | None,
        list[tuple[tuple[int, int, int, int], str]],
    ]
    | None,
) -> None:
    """Keep the unused snapshot beside YOLO boxes in the debug overlay."""
    if uia_prefetched is None:
        return
    layers = vision_timing.get("debug_layers")
    if not isinstance(layers, dict):
        return
    from src.recorder.uia_input_boxes import unpack_uia_labeled

    inputs, scrollbars, labeled = uia_prefetched
    uia_boxes: list[dict[str, Any]] = []
    for item in labeled or []:
        bbox, label, control_type = unpack_uia_labeled(item)
        row: dict[str, Any] = {
            "bbox": [int(v) for v in bbox],
            "class_name": "uia_text",
            "text": str(label or ""),
        }
        if control_type:
            row["control_type"] = control_type
        uia_boxes.append(row)
    uia_boxes.extend(
        {"bbox": [int(v) for v in bbox], "class_name": "uia_input", "text": ""}
        for bbox in (inputs or [])
    )
    uia_boxes.extend(
        {"bbox": [int(v) for v in bbox], "class_name": "uia_scrollbar", "text": ""}
        for bbox in (scrollbars or [])
    )
    layers["uia"] = uia_boxes


def _detect_recording_mouse_targets(
    bgr: np.ndarray,
    *,
    coord_offset: tuple[int, int] = (0, 0),
    ocr_roi: tuple[int, int, int, int] | None = None,
    live_input_hwnd: int | None = None,
    live_input_monitor_offset: tuple[int, int] | None = None,
    live_input_window_screen_xywh: tuple[int, int, int, int] | None = None,
    timing_out: dict[str, Any] | None = None,
    local_x: int | None = None,
    local_y: int | None = None,
    uia_prefetched: tuple[
        list[tuple[int, int, int, int]] | None,
        list[tuple[int, int, int, int]] | None,
        list[tuple[tuple[int, int, int, int], str]],
    ]
    | None = None,
    forced_scrollbar_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[list[UiDetection], dict[str, Any]]:
    """UIA candidates when the usable-label gate passes; otherwise YOLO+OCR.

    When UIA wins, YOLO+OCR still runs and is stored on
    ``vision_timing["yolo_ocr_fallback_detections"]`` so analysis can keep a
    durable OCR twin under ``yolo_ocr_fallback``. On the YOLO primary path,
    ``uia_prefetched`` is still passed so UIA inputs and scrollbars can
    prefer-replace overlapping YOLO boxes. UIA labeled text is not mixed into
    OCR candidates.
    """
    _ = live_input_hwnd, live_input_monitor_offset, live_input_window_screen_xywh
    vision_timing: dict[str, Any] = timing_out if timing_out is not None else {}
    uia_detections = _detections_from_uia_prefetched(uia_prefetched)
    if uia_prefetched is None:
        skip_reason: str | None = "沒有 UIA 快照"
    else:
        skip_reason = recording_uia_skip_reason(
            uia_detections,
            local_x=local_x,
            local_y=local_y,
            forced_scrollbar_bbox=forced_scrollbar_bbox,
        )
    ocr_timing: dict[str, Any] = {}
    detections = _detect_mouse_targets_from_bgr(
        bgr,
        coord_offset=coord_offset,
        ocr_roi=ocr_roi,
        timing_out=ocr_timing,
        defer_ocr=False,
        refine_with_uia=True,
        # Prefer-replace UIA inputs/scrollbars into YOLO+OCR candidates; texts stay debug-only.
        uia_prefetched=uia_prefetched,
    )
    # Detector already writes UIA input/scrollbar/text into debug_layers when prefetched.
    if uia_prefetched is not None and "debug_layers" not in ocr_timing:
        _attach_unused_uia_debug(ocr_timing, uia_prefetched)
    # Copy detector timing/debug onto the shared timing dict (except path fields).
    for key, value in ocr_timing.items():
        if key in ("vision_path", "vision_path_reason"):
            continue
        vision_timing[key] = value
    vision_timing["recording_ocr_deferred"] = False

    if skip_reason is None:
        vision_timing["vision_path"] = "uia"
        vision_timing.pop("vision_path_reason", None)
        vision_timing["recording_uia_sufficient"] = True
        vision_timing["yolo_ocr_fallback_detections"] = detections
        return uia_detections, vision_timing

    vision_timing["vision_path"] = "yolo_ocr"
    vision_timing["vision_path_reason"] = skip_reason
    vision_timing["recording_uia_sufficient"] = False
    return detections, vision_timing


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


def build_vision_context_full_image(
    event: RecordedEvent,
    *,
    run_dir: Path,
    persist_debug: bool = True,
    image_path: str | None = None,
    debug_name: str | None = None,
    source_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Run YOLO+OCR on the full screenshot without a click point.

    Persists every detection (reading order) so the user can pick a click target.
    ``cursor_xy`` / ``local_cursor`` stay null until a target is chosen.
    """
    empty: dict[str, Any] = {
        "used_vision": False,
        "candidate_text": "",
        "local_cursor": None,
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
                    "cursor_xy": None,
                    "local_cursor": None,
                    "candidate_text": "",
                    "candidates": [],
                    "detection_count": 0,
                    "yolo_error": load_error,
                    "source_fingerprint": fingerprint,
                },
            )
        return empty

    assert bgr is not None
    offset = event.monitor_offset if event.monitor_offset is not None else (0, 0)
    yolo_error: str | None = None
    vision_timing: dict[str, Any] = {}
    img_h, img_w = bgr.shape[:2]
    uia_prefetched = _prefetched_uia_from_snapshot(
        event.uia_snapshot if isinstance(event.uia_snapshot, dict) else None,
        monitor_offset=offset,
        image_w=img_w,
        image_h=img_h,
        ocr_roi=None,
        window_screen_xywh=None,
    )
    try:
        all_detections, vision_timing = _detect_recording_mouse_targets(
            bgr,
            coord_offset=offset,
            uia_prefetched=uia_prefetched,
        )
    except RuntimeError as exc:
        all_detections = []
        yolo_error = str(exc)

    candidate_dicts = [_detection_to_dict(det) for det in all_detections]
    candidate_text = _format_ui_candidates_text(all_detections) if all_detections else ""
    payload: dict[str, Any] = {
        "event_index": event.index,
        "image_path": resolved_image_path,
        "cursor_xy": None,
        "local_cursor": None,
        "candidate_text": candidate_text,
        "candidates": candidate_dicts,
        "detection_count": len(all_detections),
        "source_fingerprint": fingerprint,
    }
    if yolo_error:
        payload["yolo_error"] = yolo_error
    if vision_timing.get("recording_ocr_deferred") is not None:
        payload["recording_ocr_deferred"] = bool(
            vision_timing.get("recording_ocr_deferred")
        )
        payload["recording_uia_sufficient"] = bool(
            vision_timing.get("recording_uia_sufficient")
        )
    _attach_vision_path_fields(payload, vision_timing)

    if persist_debug:
        suffix = debug_name or ""
        debug_path = run_dir / "yolo_ocr" / f"event_{event.index:03d}{suffix}.json"
        write_json(debug_path, payload)

    result: dict[str, Any] = {
        "used_vision": bool(candidate_dicts),
        "candidate_text": candidate_text,
        "local_cursor": None,
        "candidates": candidate_dicts,
        "detection_count": len(all_detections),
        "bgr": bgr,
        "all_detections": all_detections,
        "yolo_error": yolo_error,
        "field_context": "(none)",
        "recording_ocr_deferred": vision_timing.get("recording_ocr_deferred"),
        "recording_uia_sufficient": vision_timing.get("recording_uia_sufficient"),
        "vision_path": vision_timing.get("vision_path"),
    }
    reason = vision_timing.get("vision_path_reason")
    if isinstance(reason, str) and reason.strip():
        result["vision_path_reason"] = reason.strip()
    return result


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
    click_window: dict[str, Any] | None = None,
    monitor_offset: tuple[int, int] | None = None,
    uia_snapshot: dict[str, Any] | None | object = _UIA_SNAPSHOT_UNSET,
    press_scrollbar_screen_xyxy: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Run YOLO+OCR and rank candidates nearest to explicit screenshot-local coords.

    ``click_window`` overrides the event press-time window used as the OCR ROI.
    Pass the drag release window when ranking the drop point.

    ``monitor_offset`` is the origin of the screenshot being read. A live window
    match is in screen coordinates and is shifted by this origin. The stored
    ``rect`` is already image-local and is not shifted again. Defaults to the
    event press-time monitor.

    ``press_scrollbar_screen_xyxy`` is the press scrollbar in screen coordinates.
    A drag release counts as that bar after the cross-axis snap. ``local_cursor``
    stays the raw release point.
    """
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
    offset = event.monitor_offset if event.monitor_offset is not None else (0, 0)
    if monitor_offset is not None:
        screenshot_offset = (int(monitor_offset[0]), int(monitor_offset[1]))
    else:
        screenshot_offset = offset
    yolo_error: str | None = None
    # Live match uses the window's current screen rect for OCR ROI geometry and,
    # when the hwnd is still matchable, for UIA input/scrollbar/text refine.
    # Stored ``rect`` stays image-local when that match is missing.
    stored_window = click_window if click_window is not None else event.click_window
    click_window_payload = stored_window if isinstance(stored_window, dict) else None
    enhance_roi = None
    live_input_hwnd: int | None = None
    live_input_window_screen_xywh: tuple[int, int, int, int] | None = None
    if click_window_payload is not None:
        from src.recorder.window_snapshot import (
            find_matching_click_window,
            resolve_ocr_roi_local,
        )

        img_h, img_w = bgr.shape[:2]
        enhance_roi = resolve_ocr_roi_local(
            click_window_payload,
            image_w=img_w,
            image_h=img_h,
            monitor_offset=screenshot_offset,
        )
        live = find_matching_click_window(click_window_payload)
        if live is not None:
            live_input_hwnd = int(live.hwnd)
            live_input_window_screen_xywh = live.screen_rect()
    chosen_snapshot = (
        event.uia_snapshot if uia_snapshot is _UIA_SNAPSHOT_UNSET else uia_snapshot
    )
    uia_prefetched = None
    if isinstance(chosen_snapshot, dict):
        img_h, img_w = bgr.shape[:2]
        uia_prefetched = _prefetched_uia_from_snapshot(
            chosen_snapshot,
            monitor_offset=screenshot_offset,
            image_w=img_w,
            image_h=img_h,
            ocr_roi=enhance_roi,
            window_screen_xywh=live_input_window_screen_xywh,
        )
    forced_scrollbar_bbox: tuple[int, int, int, int] | None = None
    if press_scrollbar_screen_xyxy is not None:
        forced_scrollbar_bbox = _screen_xyxy_to_local_xywh(
            press_scrollbar_screen_xyxy,
            screenshot_offset,
        )
    vision_timing: dict = {}
    try:
        all_detections, vision_timing = _detect_recording_mouse_targets(
            bgr,
            coord_offset=offset,
            ocr_roi=enhance_roi,
            live_input_hwnd=None if uia_prefetched is not None else live_input_hwnd,
            live_input_monitor_offset=screenshot_offset,
            live_input_window_screen_xywh=live_input_window_screen_xywh,
            timing_out=vision_timing,
            local_x=local_x,
            local_y=local_y,
            uia_prefetched=uia_prefetched,
            forced_scrollbar_bbox=forced_scrollbar_bbox,
        )
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
        # Click point is the geometric reference; promote a containing target or
        # a nearby-style landmark to candidates[0] for instruction naming.
        nearest = _promote_click_primary_detections(nearest, local_x, local_y)
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
    if vision_timing.get("recording_ocr_deferred") is not None:
        payload["recording_ocr_deferred"] = bool(
            vision_timing.get("recording_ocr_deferred")
        )
        payload["recording_uia_sufficient"] = bool(
            vision_timing.get("recording_uia_sufficient")
        )
    _attach_vision_path_fields(payload, vision_timing)
    debug_layers = vision_timing.get("debug_layers")
    if isinstance(debug_layers, dict):
        payload["debug_layers"] = debug_layers
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

    fallback_detections = vision_timing.get("yolo_ocr_fallback_detections")
    yolo_ocr_fallback_vision: dict[str, Any] | None = None
    if isinstance(fallback_detections, list) and fallback_detections:
        yolo_ocr_fallback_vision = _yolo_ocr_fallback_vision_from_detections(
            bgr,
            fallback_detections,
            local_x,
            local_y,
        )
        persistable = _persistable_yolo_ocr_fallback(yolo_ocr_fallback_vision)
        if persistable is not None:
            payload["yolo_ocr_fallback"] = persistable

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
        "vision_path": vision_timing.get("vision_path"),
    }
    reason = vision_timing.get("vision_path_reason")
    if isinstance(reason, str) and reason.strip():
        result["vision_path_reason"] = reason.strip()
    if track is not None:
        result["scrollbar_track"] = track
    if isinstance(fallback_detections, list) and fallback_detections:
        result["yolo_ocr_fallback_detections"] = fallback_detections
    if yolo_ocr_fallback_vision is not None:
        result["yolo_ocr_fallback_vision"] = yolo_ocr_fallback_vision
    return result


def _rank_detections_near_point(
    bgr: np.ndarray,
    all_detections: list[UiDetection],
    local_x: int,
    local_y: int,
) -> tuple[list[dict[str, Any]], str]:
    """Rank detections near a click and return candidate dicts + text."""
    if not all_detections:
        return [], ""
    segment_result: ColorSegmentResult | None = None
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        segment_result = segment_image_by_color(
            Image.fromarray(rgb, mode="RGB"),
            params=load_color_segment_params(),
            detections=[_ui_detection_to_segment(det) for det in all_detections],
        )
    except Exception:
        segment_result = None
    nearest = _nearest_candidates(
        all_detections,
        local_x,
        local_y,
        segment_result=segment_result,
    )
    nearest = _promote_click_primary_detections(nearest, local_x, local_y)
    candidate_text = _format_ui_candidates_text(nearest) if nearest else ""
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
    if candidate_dicts:
        _annotate_clicked_char_target(bgr, candidate_dicts, local_x, local_y)
    return candidate_dicts, candidate_text


def _yolo_ocr_fallback_vision_from_detections(
    bgr: np.ndarray,
    detections: list[UiDetection],
    local_x: int,
    local_y: int,
) -> dict[str, Any] | None:
    """Build a compact YOLO+OCR twin vision ranked at the click point."""
    if not detections:
        return None
    candidate_dicts, candidate_text = _rank_detections_near_point(
        bgr, detections, local_x, local_y
    )
    if not candidate_dicts:
        return None
    twin: dict[str, Any] = {
        "used_vision": True,
        "candidate_text": candidate_text,
        "local_cursor": (local_x, local_y),
        "candidates": candidate_dicts,
        "detection_count": len(detections),
        "field_context": format_field_context_hint(
            {
                "candidates": candidate_dicts,
                "local_cursor": (local_x, local_y),
            }
        ),
        "vision_path": "yolo_ocr",
    }
    track = annotate_scrollbar_track(
        {
            "candidates": candidate_dicts,
            "candidate_text": candidate_text,
        },
        local_x=local_x,
        local_y=local_y,
        all_detections=detections,
        require_point_inside=True,
    )
    if track is not None:
        twin["scrollbar_track"] = track
        twin["candidates"] = candidate_dicts
        twin["candidate_text"] = candidate_text
    return twin


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
    _copy_recorded_vision_mode(result, compact)
    twin = result.get("yolo_ocr_fallback_vision")
    if isinstance(twin, dict) and isinstance(twin.get("candidates"), list) and twin["candidates"]:
        compact["yolo_ocr_fallback_vision"] = twin
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
        # A release off the bar still belongs to the press scrollbar when both
        # shots share a monitor. A cross-monitor drop keeps the raw end point.
        press_scrollbar = (
            press_scrollbar_screen_xyxy(event)
            if _drag_ends_share_monitor(event)
            else None
        )
        # Same-monitor drops store the mouse-down frame in both paths. A
        # cross-monitor drop uses that other monitor's mouse-down frame.
        # Run YOLO+OCR concurrently either way.
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
                uia_snapshot=event.uia_snapshot,
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
                uia_snapshot=event.end_uia_snapshot,
                click_window=event.end_click_window,
                monitor_offset=(
                    event.end_monitor_offset
                    if event.end_monitor_offset is not None
                    else event.monitor_offset
                ),
                press_scrollbar_screen_xyxy=press_scrollbar,
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
            filtered_payload: dict[str, Any] = {
                "event_index": event.index,
                "local_cursor": list(end_local),
                "candidate_text": end_compact["candidate_text"],
                "candidates": end_compact["candidates"],
                "detection_count": end_compact.get("detection_count", 0),
                "source_fingerprint": fingerprint,
            }
            if isinstance(end_compact.get("scrollbar_track"), dict):
                filtered_payload["scrollbar_track"] = end_compact["scrollbar_track"]
            twin_block = _persistable_yolo_ocr_fallback(
                end_compact.get("yolo_ocr_fallback_vision")
                if isinstance(end_compact.get("yolo_ocr_fallback_vision"), dict)
                else None
            )
            if twin_block is not None:
                filtered_payload["yolo_ocr_fallback"] = twin_block
            write_json(
                run_dir / "yolo_ocr" / f"event_{event.index:03d}_end_filtered.json",
                filtered_payload,
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
        result = build_vision_context_full_image(
            event,
            run_dir=run_dir,
            persist_debug=persist_debug,
            source_fingerprint=fingerprint,
        )
        result.pop("bgr", None)
        result.pop("all_detections", None)
        return result

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
