"""
Unified mouse target selection: YOLO (text, element, input, scrollbar) + OCR + LLM filter/pick.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import math
import re
import threading
import time
import unicodedata
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from cua_mcp.char_target import resolve_char_screen_point, screen_bbox_from_span
from cua_mcp.geometry import (
    boxes_overlap,
    clip_box,
    iou_xywh,
    merge_overlapping_boxes,
    merge_same_line_boxes,
    point_in_rect_xywh,
)
from cua_mcp.input_box_rectangles import merge_yolo_inputs_with_line_rectangles
from cua_mcp.instruction_offset import parse_mouse_target_instruction, split_vision_mode
from cua_mcp.scrollbar_arrows import point_from_scrollbar_percent
from cua_mcp.icon_map import (
    describe_text_icons,
    is_pua_char,
    is_unknown_icon_record,
    text_has_pua,
)
from cua_mcp.read_screen_text.ocr_image import (
    _expand_box,
    _ocr_boxes_on_bgr,
    ocr_box_with_spans,
    ocr_mode_for_yolo_class,
)
from cua_mcp.caption_buttons import (
    caption_button_cell_screen_rect,
    caption_role_from_instruction,
    relabel_caption_button_detections,
    resolve_caption_button_mouse_point,
)
from cua_mcp.scrollbar_arrows import (
    create_scrollbars_from_arrow_pairs,
    drop_scrollbars_without_arrow_ends,
    fit_scrollbar_bboxes_to_arrow_controls,
    merge_overlapping_scrollbars,
    scrollbar_orientation,
)
from cua_mcp.select_ui_element import (
    UiDetection,
    _ANCHOR_SUFFIX_BY_CLASS,
    _assign_exclusive_neighbors_to_anchors,
    _describe_ui_candidate_functions,
    _format_ui_candidates_text,
    _select_center_with_functions,
    _select_center_with_llm,
    _sort_detections_reading_order,
)
from src.common.io_utils import imread_bgr, imwrite_bgr, write_json
from src.common.nearby_side import (
    NearbyHint,
    anchor_satisfies_side,
    extract_nearby_hints_from_instruction,
    merge_nearby_hints,
    nearby_hints_to_labels,
    nearby_hints_to_phrases,
    normalize_nearby_hints,
)
from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    MOUSE_TARGET_CLASS_IDS,
    PICKER_CLASS_UNKNOWN,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
    run_yolo_onnx_end2end,
)
from src.common.monitor_prompt import selected_eye_monitor_indices
from src.common.run_state import RunStateManager, get_run_state_manager, ts_name
from src.eye.capture import active_monitor_offset, grab_monitor_bgr, monitor_details


DEFAULT_OCR_ROI_PAD = 0
# YOLO boxes must overlap at least this much before identical screen pixels
# can reuse the previous frame's OCR string.
OCR_REUSE_MIN_IOU = 0.85


def _run_manager() -> RunStateManager:
    """Always resolve the current singleton (never cache): ``reset_run_state_manager`` replaces it."""
    return get_run_state_manager()


def _log_info(text: str) -> None:
    """Write to the run log when run state is initialized; no-op otherwise."""
    try:
        _run_manager().log_info(text)
    except RuntimeError:
        pass


@dataclass(frozen=True)
class _MouseVisionPlan:
    """Instruction-guided shortcuts for capture-time YOLO/OCR work."""

    ocr_class_ids: frozenset[int]
    refine_with_uia: bool

    def describe(self) -> str:
        if not self.ocr_class_ids:
            ocr = "none"
        else:
            names = []
            for cls_id in sorted(self.ocr_class_ids):
                names.append(YOLO_CLASS_NAMES.get(cls_id, str(cls_id)))
            ocr = ",".join(names)
        return f"ocr_classes={ocr} refine_with_uia={self.refine_with_uia}"


def _mouse_vision_plan(
    instruction: str,
    nearby_objects: list[str] | None = None,
) -> _MouseVisionPlan:
    """Derive OCR plan from raw instruction text (no LLM wait).

    Hub scripts mark targets as ``…文字`` / ``…圖示`` / ``輸入欄`` / ``滾動條``.
    Ambiguous phrases (no class cue) keep both OCR streams. Class-only input /
    scrollbar targets skip OCR. UIA refine (inputs/scrollbars/text assist) is
    always on when a live hwnd is available.

    Directed nearby landmarks force both text and element OCR even when every
    phrase says only ``文字`` or only ``圖示``: the real control may be the
    other class at the same place (e.g. 「此」文字 vs 網格檢視).
    """
    phrases: list[str] = []
    _mode, raw = split_vision_mode((instruction or "").strip())
    raw = raw.strip()
    if raw:
        phrases.append(raw)
    if nearby_objects:
        phrases.extend(
            p.strip() for p in nearby_objects if isinstance(p, str) and p.strip()
        )
    nearby_hints = list(extract_nearby_hints_from_instruction(raw))
    if nearby_objects:
        nearby_hints.extend(normalize_nearby_hints(nearby_objects))
    for hint in nearby_hints:
        label = (hint.label or "").strip()
        if label:
            phrases.append(label)

    need_text = False
    need_icon = False
    ambiguous = False
    for phrase in phrases:
        if "文字" in phrase:
            need_text = True
        elif "圖示" in phrase or "元素" in phrase:
            need_icon = True
        elif "輸入欄" in phrase or "滾動條" in phrase:
            continue
        else:
            # No class cue in this phrase — may need either OCR stream.
            ambiguous = True

    if ambiguous or (need_text and need_icon):
        ocr_ids: frozenset[int] = frozenset({YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT})
    elif need_text:
        ocr_ids = frozenset({YOLO_CLASS_TEXT})
    elif need_icon:
        ocr_ids = frozenset({YOLO_CLASS_ELEMENT})
    else:
        ocr_ids = frozenset()

    directed_nearby = any(hint.side is not None for hint in nearby_hints)
    if directed_nearby and ocr_ids in (
        frozenset({YOLO_CLASS_TEXT}),
        frozenset({YOLO_CLASS_ELEMENT}),
    ):
        ocr_ids = frozenset({YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT})

    return _MouseVisionPlan(ocr_class_ids=ocr_ids, refine_with_uia=True)


def _xyxy_row_to_bbox(row: np.ndarray, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(img_w, int(x2)), min(img_h, int(y2))
    bw, bh = max(1, x2i - x1i), max(1, y2i - y1i)
    return clip_box(x1i, y1i, bw, bh, img_w, img_h)


def _known_icons_for_text(text: str) -> list[dict[str, Any]] | None:
    """Return icon metadata for mapped PUA codepoints only; omit unmapped/unknown icons."""
    icons = describe_text_icons(text)
    known = [icon for icon in icons if not is_unknown_icon_record(icon)]
    return known if known else None


def _text_is_pua_only(text: str) -> bool:
    """True when ``text`` has PUA codepoints and no other visible characters."""
    if not text:
        return False
    if not text_has_pua(text):
        return False
    non_pua = "".join(ch for ch in text if not is_pua_char(ch)).strip()
    return not non_pua


def _resolve_ocr_class_id(class_id: int, text_value: str) -> int:
    """
    Reclassify ambiguous OCR results as ``unknown``.

    - PUA-only OCR with no known ``icon_map`` labels (any YOLO class)
    - YOLO ``element`` with empty OCR (no glyph decoded)
    - YOLO ``element`` whose OCR decode is plain text (no PUA)

    Known-icon PUA keeps the YOLO class.
    """
    if text_value and _text_is_pua_only(text_value) and not _known_icons_for_text(text_value):
        return PICKER_CLASS_UNKNOWN
    if class_id != YOLO_CLASS_ELEMENT:
        return class_id
    if not text_value:
        return PICKER_CLASS_UNKNOWN
    if text_has_pua(text_value):
        return class_id
    return PICKER_CLASS_UNKNOWN


def _detection_from_bbox(
    bbox: tuple[int, int, int, int],
    class_id: int,
    *,
    text: str | None = None,
    icons: list[dict[str, Any]] | None = None,
    source: str | None = None,
    uia_control_type: str | None = None,
) -> UiDetection:
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    class_name = YOLO_CLASS_NAMES.get(class_id, str(class_id))
    if icons is None and text:
        icons = _known_icons_for_text(text)
    return UiDetection(
        bbox=bbox,
        cx=cx,
        cy=cy,
        class_id=class_id,
        class_name=class_name,
        text=text,
        icons=icons if icons else None,
        source=source,
        uia_control_type=uia_control_type or None,
    )


def _offset_detection(det: UiDetection, left: int, top: int) -> UiDetection:
    x, y, w, h = det.bbox
    return UiDetection(
        bbox=(x + left, y + top, w, h),
        cx=det.cx + left,
        cy=det.cy + top,
        class_id=det.class_id,
        class_name=det.class_name,
        text=det.text,
        icons=det.icons,
        source=det.source,
        uia_control_type=det.uia_control_type,
    )


# Drop near-duplicate YOLO/OCR candidates when boxes heavily overlap and share
# the same OCR text / icon labels (e.g. two 星號、我的最愛 boxes on one star).
DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD: float = 0.5


def _detection_content_key(det: UiDetection) -> tuple[str, str]:
    """Identity used for overlap dedupe: OCR text plus sorted icon chinese_ids."""
    text = (det.text or "").strip()
    icon_ids = ",".join(
        sorted(
            str(ii.get("chinese_id", ""))
            for ii in (det.icons or [])
            if ii.get("chinese_id")
        )
    )
    return text, icon_ids


def _is_unknown_detection(det: UiDetection) -> bool:
    return det.class_id == PICKER_CLASS_UNKNOWN or det.class_name == "unknown"


def _has_actual_icon_labels(det: UiDetection) -> bool:
    return bool(
        det.icons
        and any(str(ii.get("chinese_id", "")).strip() for ii in det.icons)
    )


def _has_actual_visible_text(det: UiDetection) -> bool:
    text = (det.text or "").strip()
    return bool(text and not _text_is_pua_only(text))


def _is_element_with_icon_labels(det: UiDetection) -> bool:
    return (
        det.class_id == YOLO_CLASS_ELEMENT or det.class_name == "element"
    ) and _has_actual_icon_labels(det)


def _is_text_with_ocr_text(det: UiDetection) -> bool:
    return (
        det.class_id == YOLO_CLASS_TEXT or det.class_name == "text"
    ) and _has_actual_visible_text(det)


def _detection_preference_score(det: UiDetection) -> tuple[int, int, int, int, int]:
    """Higher score is kept when two overlapping detections compete.

    Prefer informative boxes over ``unknown``: icons / visible OCR text / ``text``
    class rank above empty unknowns; larger area is the final tie-breaker.
    """
    not_unknown = 0 if _is_unknown_detection(det) else 1
    has_icons = 1 if _has_actual_icon_labels(det) else 0
    has_visible_text = 1 if _has_actual_visible_text(det) else 0
    is_text_class = 1 if det.class_id == YOLO_CLASS_TEXT or det.class_name == "text" else 0
    area = int(det.bbox[2]) * int(det.bbox[3])
    return (not_unknown, has_icons, has_visible_text, is_text_class, area)


def _detections_are_content_duplicates(a: UiDetection, b: UiDetection) -> bool:
    """True when ``a`` and ``b`` share the same text/icon identity (or blank same-class)."""
    key_a = _detection_content_key(a)
    key_b = _detection_content_key(b)
    if key_a != key_b:
        return False
    if key_a != ("", ""):
        return True
    return a.class_id == b.class_id


def _detections_compete_for_overlap_dedupe(a: UiDetection, b: UiDetection) -> bool:
    """True when overlap dedupe should keep only the more informative of ``a``/``b``."""
    # Element icons and OCR text boxes are complementary — keep both.
    if (
        _is_element_with_icon_labels(a) and _is_text_with_ocr_text(b)
    ) or (
        _is_text_with_ocr_text(a) and _is_element_with_icon_labels(b)
    ):
        return False
    if _detections_are_content_duplicates(a, b):
        return True
    # Ambiguous ``unknown`` under a richer OCR/text/element box → drop unknown.
    return _is_unknown_detection(a) != _is_unknown_detection(b)


def _dedupe_overlapping_detections(
    detections: list[UiDetection],
    *,
    iou_threshold: float = DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD,
) -> list[UiDetection]:
    """
    Keep one detection per heavily overlapping competing group.

    Candidates are compared by IoU of ``(x, y, w, h)`` boxes. When IoU exceeds
    ``iou_threshold`` and the pair competes (same OCR/icon content, blank
    same-class, or ``unknown`` vs a more informative class), only the preferred
    detection is kept — icons / visible text / ``text`` over ``unknown``.
    """
    if len(detections) < 2:
        return detections

    order = sorted(
        range(len(detections)),
        key=lambda i: _detection_preference_score(detections[i]),
        reverse=True,
    )
    kept: list[int] = []
    for i in order:
        det = detections[i]
        if any(
            iou_xywh(det.bbox, detections[j].bbox) > iou_threshold
            and _detections_compete_for_overlap_dedupe(det, detections[j])
            for j in kept
        ):
            continue
        kept.append(i)
    kept.sort()
    return [detections[i] for i in kept]


def _pua_char_count(text: str) -> int:
    return sum(1 for ch in text or "" if is_pua_char(ch))


def _split_multi_icon_element_detection(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    text_value: str,
) -> list[UiDetection] | None:
    """Split a PUA-only multi-icon element box into one detection per icon.

    Returns ``None`` when splitting does not apply or span OCR yields no usable
    PUA spans (caller should keep the combined detection).
    """
    if not _text_is_pua_only(text_value) or _pua_char_count(text_value) < 2:
        return None
    mode = ocr_mode_for_yolo_class(YOLO_CLASS_ELEMENT)
    _decoded, spans = ocr_box_with_spans(bgr, bbox, mode=mode)
    if not spans:
        return None
    img_h, img_w = bgr.shape[:2]
    split: list[UiDetection] = []
    for span in spans:
        if not is_pua_char(span.char):
            continue
        child_bbox = screen_bbox_from_span(bbox, span, img_w=img_w, img_h=img_h)
        resolved_cls = _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, span.char)
        split.append(
            _detection_from_bbox(child_bbox, resolved_cls, text=span.char)
        )
    return split if split else None


# Second-pass OCR for empty-OCR ``unknown`` icons: modest expand, stop before text.
_UNKNOWN_ICON_RETRY_EXTRA_MARGIN = 2
# Per-side pads ``(left, top, right, bottom)`` tried in order until one yields a
# single known-icon PUA. First: uniform +2; then +4 on exactly two sides.
_UNKNOWN_ICON_RETRY_SIDE_MARGINS: tuple[tuple[int, int, int, int], ...] = (
    (2, 2, 2, 2),  # +2 all sides
    (4, 0, 4, 0),  # +4 left+right
    (0, 4, 0, 4),  # +4 top+bottom
    (4, 4, 0, 0),  # +4 left+top
    (0, 0, 4, 4),  # +4 right+bottom
)


def _is_empty_unknown_detection(det: UiDetection) -> bool:
    """True for ``unknown`` detections with no OCR text and no icon metadata."""
    if not _is_unknown_detection(det):
        return False
    if (det.text or "").strip():
        return False
    if det.icons:
        return False
    return True


def _expand_bbox_avoiding_text(
    bbox: tuple[int, int, int, int],
    text_boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    *,
    extra_margin: int = _UNKNOWN_ICON_RETRY_EXTRA_MARGIN,
    side_margins: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Expand ``bbox`` by side pads, shrinking sides that would overlap text.

    ``side_margins`` is ``(left, top, right, bottom)``. When omitted, each side
    uses ``extra_margin`` (legacy uniform expand).

    Never shrinks below the original ``bbox``. Result is clipped to the image.
    Only the expanded portion of each side is retracted when that side's pad
    would intersect a text box (same-row text trims left/right only).
    """
    ox, oy, ow, oh = bbox
    ox2, oy2 = ox + ow, oy + oh
    if side_margins is None:
        left = top = right = bottom = int(extra_margin)
    else:
        left, top, right, bottom = (int(v) for v in side_margins)
    x1 = ox - left
    y1 = oy - top
    x2 = ox2 + right
    y2 = oy2 + bottom
    x1, y1, ew, eh = clip_box(x1, y1, x2 - x1, y2 - y1, img_w, img_h)
    x2, y2 = x1 + ew, y1 + eh

    for tx, ty, tw, th in text_boxes:
        if not boxes_overlap((x1, y1, x2 - x1, y2 - y1), (tx, ty, tw, th)):
            continue
        tx2, ty2 = tx + tw, ty + th
        # Retract only the expanded pad on each side when that pad hits text.
        if x2 > ox2 and tx < x2 and tx2 > ox2 and ty < y2 and ty2 > y1:
            x2 = min(x2, max(ox2, tx))
        if x1 < ox and tx2 > x1 and tx < ox and ty < y2 and ty2 > y1:
            x1 = max(x1, min(ox, tx2))
        if y2 > oy2 and ty < y2 and ty2 > oy2 and tx < x2 and tx2 > x1:
            y2 = min(y2, max(oy2, ty))
        if y1 < oy and ty2 > y1 and ty < oy and tx < x2 and tx2 > x1:
            y1 = max(y1, min(oy, ty2))

    x1 = min(x1, ox)
    y1 = min(y1, oy)
    x2 = max(x2, ox2)
    y2 = max(y2, oy2)
    return clip_box(x1, y1, x2 - x1, y2 - y1, img_w, img_h)


def _single_known_icon_pua(text_value: str) -> str | None:
    """Return the sole known-icon PUA in ``text_value``, else ``None``.

    Rejects empty, plain text, multi-PUA, unmapped PUA, and ``unknown_icon``.
    """
    text = (text_value or "").strip()
    if not text or not _text_is_pua_only(text) or _pua_char_count(text) != 1:
        return None
    if _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, text) == PICKER_CLASS_UNKNOWN:
        return None
    if not _known_icons_for_text(text):
        return None
    return text


def _retry_empty_unknown_icon_ocr(
    bgr: np.ndarray,
    candidates: list[UiDetection],
) -> list[UiDetection]:
    """
    Re-OCR empty ``unknown`` detections with several modestly expanded crops.

    Tries :data:`_UNKNOWN_ICON_RETRY_SIDE_MARGINS` in order (uniform +2, then
    four +4-on-two-sides variants). Uses icon decode and ``margin=0`` on
    already-clamped crops so the default OCR pad cannot bleed into neighboring
    text. Stored detection bboxes stay as the original YOLO boxes.

    Upgrades only when a variant yields exactly one known-icon PUA; otherwise
    leaves the empty unknown unchanged.
    """
    retry_indices = [
        i for i, det in enumerate(candidates) if _is_empty_unknown_detection(det)
    ]
    if not retry_indices:
        return candidates

    img_h, img_w = bgr.shape[:2]
    text_boxes = [
        det.bbox
        for det in candidates
        if (
            (det.class_id == YOLO_CLASS_TEXT or det.class_name == "text")
            and _has_actual_visible_text(det)
        )
    ]
    crop_boxes: list[tuple[int, int, int, int]] = []
    crop_owners: list[int] = []
    for idx in retry_indices:
        for side_margins in _UNKNOWN_ICON_RETRY_SIDE_MARGINS:
            crop_boxes.append(
                _expand_bbox_avoiding_text(
                    candidates[idx].bbox,
                    text_boxes,
                    img_w,
                    img_h,
                    side_margins=side_margins,
                )
            )
            crop_owners.append(idx)

    mode = ocr_mode_for_yolo_class(YOLO_CLASS_ELEMENT)
    ocr_preds = _ocr_boxes_on_bgr(bgr, crop_boxes, mode=mode, margin=0)

    preds_by_idx: dict[int, list[str]] = {idx: [] for idx in retry_indices}
    for owner, preds in zip(crop_owners, ocr_preds, strict=True):
        preds_by_idx[owner].append("".join(preds).strip())

    replacements: dict[int, list[UiDetection]] = {}
    for idx in retry_indices:
        chosen: str | None = None
        for text_value in preds_by_idx[idx]:
            chosen = _single_known_icon_pua(text_value)
            if chosen is not None:
                break
        if chosen is None:
            continue
        resolved_cls = _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, chosen)
        replacements[idx] = [
            _detection_from_bbox(candidates[idx].bbox, resolved_cls, text=chosen)
        ]

    _log_info(
        "move_mouse: unknown icon OCR retry "
        f"count={len(retry_indices)} variants={len(_UNKNOWN_ICON_RETRY_SIDE_MARGINS)} "
        f"upgraded={len(replacements)}"
    )
    if not replacements:
        return candidates

    result: list[UiDetection] = []
    for i, det in enumerate(candidates):
        if i in replacements:
            result.extend(replacements[i])
        else:
            result.append(det)
    return result


def _round_timing_s(value: float) -> float:
    return round(max(0.0, float(value)), 3)


def _normalize_ocr_roi_xywh(raw: Any) -> list[int] | None:
    """Return image-local ``[x, y, w, h]`` when ``raw`` is a 4-number ROI."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [int(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _vision_roi_fields(vision: dict[str, Any]) -> dict[str, Any]:
    """Pull ``ocr_roi`` / ``ocr_rois`` from a vision timing dict for tool results."""
    out: dict[str, Any] = {}
    single = _normalize_ocr_roi_xywh(vision.get("ocr_roi"))
    if single is not None:
        out["ocr_roi"] = single
    raw_list = vision.get("ocr_rois")
    if isinstance(raw_list, list):
        rois = [
            roi
            for item in raw_list
            for roi in (_normalize_ocr_roi_xywh(item),)
            if roi is not None
        ]
        if rois:
            out["ocr_rois"] = rois
            if "ocr_roi" not in out and len(rois) == 1:
                out["ocr_roi"] = rois[0]
    pad = vision.get("ocr_roi_pad")
    if isinstance(pad, (int, float)):
        out["ocr_roi_pad"] = int(pad)
    return out


def _ocr_reuse_count_fields(source: dict[str, Any]) -> dict[str, int]:
    """Copy OCR box reuse / UIA text-assist counts when present as integers."""
    out: dict[str, int] = {}
    for key in (
        "ocr_reused",
        "ocr_fresh",
        "uia_text_orphans",
        "uia_text_ocr_skipped",
    ):
        raw = source.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        out[key] = int(raw)
    return out


def _timing_payload_with_roi(
    *,
    yolo_s: float,
    line_s: float,
    ocr_s: float,
    total_s: float,
    ocr_roi: tuple[int, int, int, int] | None = None,
    ocr_roi_pad: int | None = None,
    ocr_reused: int | None = None,
    ocr_fresh: int | None = None,
    uia_text_orphans: int | None = None,
    uia_text_ocr_skipped: int | None = None,
    icon_ocr_retry_s: float = 0.0,
    uia_input_s: float = 0.0,
    uia_scrollbar_s: float = 0.0,
    uia_text_s: float = 0.0,
) -> dict[str, Any]:
    """Build a per-monitor vision timing dict, optionally including the OCR ROI."""
    payload: dict[str, Any] = {
        "yolo_s": yolo_s,
        "line_s": line_s,
        "uia_input_s": uia_input_s,
        "uia_scrollbar_s": uia_scrollbar_s,
        "uia_text_s": uia_text_s,
        "ocr_s": ocr_s,
        "icon_ocr_retry_s": icon_ocr_retry_s,
        "total_s": total_s,
    }
    roi = _normalize_ocr_roi_xywh(ocr_roi)
    if roi is not None:
        payload["ocr_roi"] = roi
    if ocr_roi_pad is not None:
        payload["ocr_roi_pad"] = int(ocr_roi_pad)
    if ocr_reused is not None:
        payload["ocr_reused"] = int(ocr_reused)
    if ocr_fresh is not None:
        payload["ocr_fresh"] = int(ocr_fresh)
    if uia_text_orphans is not None:
        payload["uia_text_orphans"] = int(uia_text_orphans)
    if uia_text_ocr_skipped is not None:
        payload["uia_text_ocr_skipped"] = int(uia_text_ocr_skipped)
    return payload


def _add_vision_phase_timing(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Accumulate YOLO/line/OCR phase seconds (and ROIs) from one monitor into ``dst``."""
    for key in (
        "yolo_s",
        "line_s",
        "uia_input_s",
        "uia_scrollbar_s",
        "uia_text_s",
        "ocr_s",
        "icon_ocr_retry_s",
        "total_s",
    ):
        raw = src.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            dst[key] = float(dst.get(key, 0.0)) + float(raw)
    for key in (
        "ocr_reused",
        "ocr_fresh",
        "uia_text_orphans",
        "uia_text_ocr_skipped",
    ):
        raw = src.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        previous = dst.get(key, 0)
        if isinstance(previous, bool) or not isinstance(previous, int):
            previous = 0
        dst[key] = int(previous) + int(raw)
    roi = _normalize_ocr_roi_xywh(src.get("ocr_roi"))
    if roi is not None:
        rois = dst.setdefault("ocr_rois", [])
        if isinstance(rois, list):
            rois.append(roi)
        if "ocr_roi" not in dst:
            dst["ocr_roi"] = roi
    if "ocr_roi_pad" not in dst:
        pad = src.get("ocr_roi_pad")
        if isinstance(pad, (int, float)):
            dst["ocr_roi_pad"] = int(pad)


def _build_move_mouse_timing(
    *,
    total_s: float,
    capture_s: float,
    vision: dict[str, Any],
    parse_s: float,
    select_s: float,
    select_phase: str = "select",
) -> dict[str, Any]:
    """Structured timing for move_mouse / check_object_exists tool results."""
    yolo_s = _round_timing_s(float(vision.get("yolo_s", 0.0)))
    line_s = _round_timing_s(float(vision.get("line_s", 0.0)))
    uia_input_s = _round_timing_s(float(vision.get("uia_input_s", 0.0)))
    uia_scrollbar_s = _round_timing_s(float(vision.get("uia_scrollbar_s", 0.0)))
    uia_text_s = _round_timing_s(float(vision.get("uia_text_s", 0.0)))
    ocr_s = _round_timing_s(float(vision.get("ocr_s", 0.0)))
    icon_ocr_retry_s = _round_timing_s(float(vision.get("icon_ocr_retry_s", 0.0)))
    capture_s = _round_timing_s(capture_s)
    parse_s = _round_timing_s(parse_s)
    select_s = _round_timing_s(select_s)
    total_s = _round_timing_s(total_s)
    roi_fields = _vision_roi_fields(vision)
    reuse_fields = _ocr_reuse_count_fields(vision)
    yolo_phase: dict[str, Any] = {"name": "yolo", "seconds": yolo_s, **roi_fields}
    ocr_phase: dict[str, Any] = {"name": "ocr", "seconds": ocr_s, **roi_fields, **reuse_fields}
    phases: list[dict[str, Any]] = [
        {"name": "capture", "seconds": capture_s},
        yolo_phase,
        {"name": "line_refine", "seconds": line_s, **roi_fields},
        {"name": "uia_input", "seconds": uia_input_s, **roi_fields},
        {"name": "uia_scrollbar", "seconds": uia_scrollbar_s, **roi_fields},
        {"name": "uia_text", "seconds": uia_text_s, **roi_fields},
        ocr_phase,
        {"name": "icon_ocr_retry", "seconds": icon_ocr_retry_s},
        {"name": "parse_instruction", "seconds": parse_s, "overlapped": False},
        {"name": select_phase, "seconds": select_s},
    ]
    result: dict[str, Any] = {
        "total_s": total_s,
        "capture_s": capture_s,
        "yolo_s": yolo_s,
        "line_s": line_s,
        "uia_input_s": uia_input_s,
        "uia_scrollbar_s": uia_scrollbar_s,
        "uia_text_s": uia_text_s,
        "ocr_s": ocr_s,
        "icon_ocr_retry_s": icon_ocr_retry_s,
        "parse_s": parse_s,
        "select_s": select_s,
        "phases": phases,
    }
    if "uia_first_hit" in vision:
        result["uia_first_hit"] = bool(vision.get("uia_first_hit"))
    if vision.get("vision_path"):
        result["vision_path"] = vision.get("vision_path")
    result.update(roi_fields)
    result.update(reuse_fields)
    return result


def _bbox_center_in_ocr_roi(
    bbox: tuple[int, int, int, int],
    ocr_roi: tuple[int, int, int, int] | None,
    *,
    pad: int = DEFAULT_OCR_ROI_PAD,
) -> bool:
    """True when bbox center is inside padded ``ocr_roi`` (or ROI is unset)."""
    if ocr_roi is None:
        return True
    x, y, w, h = (int(v) for v in bbox)
    return point_in_rect_xywh(x + w * 0.5, y + h * 0.5, ocr_roi, pad=pad)


def _point_outside_roi_xywh(
    x: float,
    y: float,
    roi: tuple[int, int, int, int],
) -> bool:
    """True when ``(x, y)`` is outside ``roi`` xywh (no pad)."""
    return not point_in_rect_xywh(x, y, roi, pad=0)


def _landmarks_clearly_outside_window(
    roi: tuple[int, int, int, int] | None,
    centers: list[tuple[float, float]] | None,
) -> bool:
    """True when any recorded landmark center falls outside the window ROI."""
    if roi is None or not centers:
        return False
    return any(_point_outside_roi_xywh(cx, cy, roi) for cx, cy in centers)


def _offset_xyxy_rows(xyxy: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """Translate ``xyxy`` rows by ``(ox, oy)``."""
    if xyxy.size == 0:
        return xyxy
    out = np.asarray(xyxy, dtype=np.float32).copy()
    out[:, 0] += float(ox)
    out[:, 1] += float(oy)
    out[:, 2] += float(ox)
    out[:, 3] += float(oy)
    return out


@dataclass(frozen=True)
class _OcrFrameEntry:
    """One OCR box kept until its pixels change or a new read replaces it."""

    bbox: tuple[int, int, int, int]
    mode: str
    text: tuple[str, ...]


@dataclass(frozen=True)
class _OcrFrameSlot:
    """Latest screenshot and still-valid OCR results for one monitor.

    Entries survive later captures whose ROI does not include them. A capture
    drops an entry when the expanded box pixels differ, and replaces it when a
    new box of the same mode overlaps it and is read again.
    """

    shape: tuple[int, ...]
    bgr: np.ndarray
    entries: tuple[_OcrFrameEntry, ...]


_OCR_FRAME_CACHE: dict[int, _OcrFrameSlot] = {}
_OCR_FRAME_CACHE_LOCK = threading.Lock()


def clear_ocr_frame_cache() -> None:
    """Drop remembered monitor screenshots. Tests use this for isolation."""
    with _OCR_FRAME_CACHE_LOCK:
        _OCR_FRAME_CACHE.clear()


def _store_ocr_frame_cache(
    cache_key: int,
    bgr: np.ndarray,
    entries: list[_OcrFrameEntry],
) -> None:
    slot = _OcrFrameSlot(
        shape=tuple(int(v) for v in bgr.shape),
        bgr=np.array(bgr, copy=True),
        entries=tuple(entries),
    )
    with _OCR_FRAME_CACHE_LOCK:
        _OCR_FRAME_CACHE[cache_key] = slot


def _take_ocr_frame_cache(cache_key: int, shape: tuple[int, ...]) -> _OcrFrameSlot | None:
    """Return the previous slot when its image shape matches, else drop it."""
    with _OCR_FRAME_CACHE_LOCK:
        slot = _OCR_FRAME_CACHE.get(cache_key)
        if slot is None:
            return None
        if slot.shape != shape:
            _OCR_FRAME_CACHE.pop(cache_key, None)
            return None
        return slot


def _ocr_overlap_pixels_equal(
    prev_bgr: np.ndarray,
    prev_box: tuple[int, int, int, int],
    new_bgr: np.ndarray,
    new_box: tuple[int, int, int, int],
) -> bool:
    """True when the expanded boxes share identical pixels at the same coordinates.

    Expansion matches the margin ``_ocr_boxes_on_bgr`` adds before CRNN. A 1-pixel
    box shift on a still screen still matches; a changed glyph does not.
    """
    img_h, img_w = int(new_bgr.shape[0]), int(new_bgr.shape[1])
    if prev_bgr.shape[:2] != (img_h, img_w):
        return False
    px, py, pw, ph = _expand_box(*prev_box, img_w, img_h)
    nx, ny, nw, nh = _expand_box(*new_box, img_w, img_h)
    ix1, iy1 = max(px, nx), max(py, ny)
    ix2, iy2 = min(px + pw, nx + nw), min(py + ph, ny + nh)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    return bool(
        np.array_equal(
            prev_bgr[iy1:iy2, ix1:ix2],
            new_bgr[iy1:iy2, ix1:ix2],
        )
    )


def _retain_unchanged_ocr_entries(
    prev_bgr: np.ndarray,
    entries: tuple[_OcrFrameEntry, ...] | list[_OcrFrameEntry],
    new_bgr: np.ndarray,
) -> list[_OcrFrameEntry]:
    """Drop cached boxes whose expanded pixels differ on ``new_bgr``.

    Checked for every entry, including ones outside this capture's ROI, so a
    label that changes off-screen is not served later from the old string.
    """
    if prev_bgr.shape[:2] != new_bgr.shape[:2]:
        return []
    return [
        entry
        for entry in entries
        if _ocr_overlap_pixels_equal(prev_bgr, entry.bbox, new_bgr, entry.bbox)
    ]


def _match_reused_ocr_entries(
    prev_bgr: np.ndarray,
    prev_entries: tuple[_OcrFrameEntry, ...] | list[_OcrFrameEntry],
    new_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    modes: list[str],
) -> tuple[list[list[str] | None], set[int]]:
    """Map new boxes to cached strings and to the cache indexes those strings used.

    One previous box is given to at most one new box. Higher IoU wins. The pair
    must share a decode mode, overlap by at least :data:`OCR_REUSE_MIN_IOU`, and
    have identical pixels on the intersection of the expanded rects. The returned
    index set contains only pairs that passed the pixel check.
    """
    reused: list[list[str] | None] = [None] * len(boxes)
    used_old: set[int] = set()
    if prev_bgr.shape[:2] != new_bgr.shape[:2] or not prev_entries or not boxes:
        return reused, used_old

    pairs: list[tuple[float, int, int]] = []
    for new_i, (box, mode) in enumerate(zip(boxes, modes, strict=True)):
        for old_i, entry in enumerate(prev_entries):
            if entry.mode != mode:
                continue
            score = iou_xywh(box, entry.bbox)
            if score >= OCR_REUSE_MIN_IOU:
                pairs.append((score, new_i, old_i))
    pairs.sort(key=lambda item: item[0], reverse=True)

    used_new: set[int] = set()
    assigned: dict[int, int] = {}
    for _score, new_i, old_i in pairs:
        if new_i in used_new or old_i in used_old:
            continue
        used_new.add(new_i)
        used_old.add(old_i)
        assigned[new_i] = old_i

    confirmed: set[int] = set()
    for new_i, old_i in assigned.items():
        entry = prev_entries[old_i]
        if _ocr_overlap_pixels_equal(prev_bgr, entry.bbox, new_bgr, boxes[new_i]):
            reused[new_i] = list(entry.text)
            confirmed.add(old_i)
    return reused, confirmed


def _assign_reused_ocr_texts(
    prev_bgr: np.ndarray,
    prev_entries: tuple[_OcrFrameEntry, ...] | list[_OcrFrameEntry],
    new_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    modes: list[str],
) -> list[list[str] | None]:
    """Map each new box to a previous OCR string, or None when it must be read."""
    reused, _used_old = _match_reused_ocr_entries(
        prev_bgr,
        prev_entries,
        new_bgr,
        boxes,
        modes,
    )
    return reused


def _drop_entries_replaced_by_fresh_ocr(
    entries: list[_OcrFrameEntry],
    fresh_boxes: list[tuple[int, int, int, int]],
    fresh_modes: list[str],
) -> list[_OcrFrameEntry]:
    """Drop a cached box when a newly read box of the same mode overlaps it."""
    if not entries or not fresh_boxes:
        return entries
    kept: list[_OcrFrameEntry] = []
    for entry in entries:
        replaced = False
        for box, mode in zip(fresh_boxes, fresh_modes, strict=True):
            if entry.mode == mode and iou_xywh(entry.bbox, box) >= OCR_REUSE_MIN_IOU:
                replaced = True
                break
        if not replaced:
            kept.append(entry)
    return kept


def _recognize_ocr_boxes(
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    modes: list[str],
    *,
    cache_key: int | None,
) -> tuple[list[list[str]], int, int]:
    """OCR ``boxes``, reusing identical regions when ``cache_key`` is set.

    Returns ``(predictions, reused_count, fresh_count)``. A missing key never
    reads or writes the monitor cache. A set key keeps prior entries whose
    pixels still match, including boxes outside this capture, and replaces an
    entry when this capture reads an overlapping box.
    """
    if len(modes) != len(boxes):
        raise ValueError(
            f"mode length {len(modes)} does not match box count {len(boxes)}"
        )
    prev_entries: list[_OcrFrameEntry] = []
    reused: list[list[str] | None] = [None] * len(boxes)
    used_old: set[int] = set()
    if cache_key is not None:
        slot = _take_ocr_frame_cache(cache_key, tuple(int(v) for v in bgr.shape))
        if slot is not None:
            prev_entries = _retain_unchanged_ocr_entries(slot.bgr, slot.entries, bgr)
            reused, used_old = _match_reused_ocr_entries(
                slot.bgr,
                prev_entries,
                bgr,
                boxes,
                modes,
            )

    fresh_indices = [i for i, pred in enumerate(reused) if pred is None]
    fresh_boxes = [boxes[i] for i in fresh_indices]
    fresh_modes = [modes[i] for i in fresh_indices]
    fresh_preds = (
        _ocr_boxes_on_bgr(bgr, fresh_boxes, mode=fresh_modes) if fresh_boxes else []
    )
    if len(fresh_preds) != len(fresh_indices):
        raise RuntimeError(
            f"OCR returned {len(fresh_preds)} results for {len(fresh_indices)} boxes"
        )

    preds: list[list[str]] = []
    frame_entries: list[_OcrFrameEntry] = []
    fresh_cursor = 0
    for i, box in enumerate(boxes):
        cached = reused[i]
        if cached is None:
            pred = list(fresh_preds[fresh_cursor])
            fresh_cursor += 1
        else:
            pred = cached
        preds.append(pred)
        frame_entries.append(
            _OcrFrameEntry(bbox=box, mode=modes[i], text=tuple(pred))
        )

    if cache_key is not None:
        retained = [
            entry for index, entry in enumerate(prev_entries) if index not in used_old
        ]
        retained = _drop_entries_replaced_by_fresh_ocr(
            retained,
            fresh_boxes,
            fresh_modes,
        )
        _store_ocr_frame_cache(cache_key, bgr, retained + frame_entries)
    return preds, len(boxes) - len(fresh_indices), len(fresh_indices)


def _vision_debug_layers(
    yolo_boxes: list[dict[str, Any]],
    uia_labeled_texts: list[tuple[tuple[int, int, int, int], str]],
    uia_inputs: list[tuple[int, int, int, int]],
    uia_scrollbars: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Image-local YOLO + UIA boxes for run/recording overlay debug."""
    from src.recorder.uia_input_boxes import unpack_uia_labeled

    uia_boxes: list[dict[str, Any]] = []
    for item in uia_labeled_texts:
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
        for bbox in uia_inputs
    )
    uia_boxes.extend(
        {"bbox": [int(v) for v in bbox], "class_name": "uia_scrollbar", "text": ""}
        for bbox in uia_scrollbars
    )
    return {"yolo": list(yolo_boxes), "uia": uia_boxes}


def _ui_detection_candidate_dict(det: UiDetection) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bbox": [int(v) for v in det.bbox],
        "center": [int(det.cx), int(det.cy)],
        "class_id": det.class_id,
        "class_name": det.class_name,
        "text": det.text,
        "icons": det.icons,
    }
    if det.source:
        payload["source"] = det.source
    if det.uia_control_type:
        payload["uia_control_type"] = det.uia_control_type
    return payload


@dataclass
class _MouseOcrPending:
    """Leftover OCR work after UIA associate, for soft UIA-first select."""

    bgr: np.ndarray
    ocr_boxes: list[tuple[int, int, int, int]]
    ocr_class_id_list: list[int]
    partial_candidates: list[UiDetection]
    used_uia_scrollbar: bool
    coord_offset: tuple[int, int]
    ocr_cache_key: int | None
    ocr_wanted: frozenset[int]
    yolo_s: float
    line_s: float
    uia_input_s: float
    uia_scrollbar_s: float
    uia_text_s: float
    vision_started: float
    uia_text_orphans: int
    uia_text_ocr_skipped: int
    ocr_roi: tuple[int, int, int, int] | None
    ocr_roi_pad: int
    refine_with_uia: bool
    yolo_box_count: int
    input_box_count: int
    debug_layers: dict[str, Any]
    original_scrollbar_bboxes_out: list[tuple[int, int, int, int]] | None = None


def _finalize_mouse_candidates(
    candidates: list[UiDetection],
    *,
    used_uia_scrollbar: bool,
    coord_offset: tuple[int, int],
    original_scrollbar_bboxes_out: list[tuple[int, int, int, int]] | None = None,
) -> list[UiDetection]:
    """Reading-order sort, scrollbar fit/create/drop, caption-button relabel."""
    candidates = _sort_detections_reading_order(candidates)
    pre_fit_scrollbars: list[tuple[int, tuple[int, int, int, int]]] = []
    if original_scrollbar_bboxes_out is not None:
        pre_fit_scrollbars = [
            (i, det.bbox)
            for i, det in enumerate(candidates)
            if det.class_id == YOLO_CLASS_SCROLLBAR or det.class_name == "scrollbar"
        ]
    candidates = fit_scrollbar_bboxes_to_arrow_controls(
        candidates, log_info=_log_info
    )
    if used_uia_scrollbar:
        # UIA tracks are authoritative; do not invent or drop for missing OCR arrows.
        candidates = merge_overlapping_scrollbars(
            candidates, log_info=_log_info
        )
    else:
        candidates = create_scrollbars_from_arrow_pairs(
            candidates, log_info=_log_info
        )
        candidates = drop_scrollbars_without_arrow_ends(
            candidates, log_info=_log_info
        )
        candidates = merge_overlapping_scrollbars(
            candidates, log_info=_log_info
        )
    candidates = relabel_caption_button_detections(
        candidates,
        coord_offset=coord_offset,
        log_info=_log_info,
    )
    if original_scrollbar_bboxes_out is not None:
        original_scrollbar_bboxes_out.extend(
            old_bbox for _i, old_bbox in pre_fit_scrollbars
        )
    return candidates


def _append_ocr_detections(
    bgr: np.ndarray,
    candidates: list[UiDetection],
    ocr_boxes: list[tuple[int, int, int, int]],
    ocr_class_id_list: list[int],
    ocr_preds: list[Any],
) -> None:
    """Append OCR (and multi-icon splits) onto ``candidates`` in place."""
    for bbox, cls_id, preds in zip(
        ocr_boxes, ocr_class_id_list, ocr_preds, strict=True
    ):
        text_value = "".join(preds).strip()
        if cls_id == YOLO_CLASS_TEXT and not text_value:
            continue
        if cls_id == YOLO_CLASS_ELEMENT:
            split = _split_multi_icon_element_detection(bgr, bbox, text_value)
            if split is not None:
                for det in split:
                    candidates.append(
                        _detection_from_bbox(
                            det.bbox,
                            det.class_id,
                            text=det.text,
                            icons=det.icons,
                            source="ocr",
                        )
                    )
                continue
        resolved_cls = _resolve_ocr_class_id(cls_id, text_value)
        candidates.append(
            _detection_from_bbox(
                bbox,
                resolved_cls,
                text=text_value or None,
                source="ocr",
            )
        )


def _finish_mouse_ocr_pending(
    pending: _MouseOcrPending,
    *,
    timing_out: dict[str, Any] | None = None,
) -> list[UiDetection]:
    """Run leftover OCR + icon retry, merge with partial candidates, finalize."""
    ocr_started = time.perf_counter()
    ocr_modes = [
        ocr_mode_for_yolo_class(cls_id) for cls_id in pending.ocr_class_id_list
    ]
    ocr_preds, ocr_reused, ocr_fresh = _recognize_ocr_boxes(
        pending.bgr,
        pending.ocr_boxes,
        ocr_modes,
        cache_key=pending.ocr_cache_key,
    )
    ocr_elapsed = time.perf_counter() - ocr_started

    candidates = list(pending.partial_candidates)
    _append_ocr_detections(
        pending.bgr,
        candidates,
        pending.ocr_boxes,
        pending.ocr_class_id_list,
        ocr_preds,
    )

    icon_ocr_retry_elapsed = 0.0
    if YOLO_CLASS_ELEMENT in pending.ocr_wanted:
        icon_retry_started = time.perf_counter()
        candidates = _retry_empty_unknown_icon_ocr(pending.bgr, candidates)
        icon_ocr_retry_elapsed = time.perf_counter() - icon_retry_started

    candidates = _finalize_mouse_candidates(
        candidates,
        used_uia_scrollbar=pending.used_uia_scrollbar,
        coord_offset=pending.coord_offset,
        original_scrollbar_bboxes_out=pending.original_scrollbar_bboxes_out,
    )
    total_elapsed = time.perf_counter() - pending.vision_started
    _log_info(
        "move_mouse vision profile "
        f"yolo_s={pending.yolo_s:.3f} line_s={pending.line_s:.3f} "
        f"uia_input_s={pending.uia_input_s:.3f} "
        f"uia_scrollbar_s={pending.uia_scrollbar_s:.3f} "
        f"uia_text_s={pending.uia_text_s:.3f} "
        f"ocr_s={ocr_elapsed:.3f} "
        f"icon_ocr_retry_s={icon_ocr_retry_elapsed:.3f} "
        f"ocr_reused={ocr_reused} ocr_fresh={ocr_fresh} "
        f"total_s={total_elapsed:.3f} "
        f"yolo_boxes={pending.yolo_box_count} "
        f"input_boxes={pending.input_box_count} "
        f"ocr_boxes={len(pending.ocr_boxes)} "
        f"candidates={len(candidates)} refine_with_uia={pending.refine_with_uia}"
        + (
            f" ocr_roi={tuple(int(v) for v in pending.ocr_roi)}"
            if pending.ocr_roi is not None
            else ""
        )
    )
    if timing_out is not None:
        timing_out.clear()
        timing_out.update(
            _timing_payload_with_roi(
                yolo_s=pending.yolo_s,
                line_s=pending.line_s,
                ocr_s=ocr_elapsed,
                total_s=total_elapsed,
                ocr_roi=pending.ocr_roi,
                ocr_roi_pad=pending.ocr_roi_pad,
                ocr_reused=ocr_reused,
                ocr_fresh=ocr_fresh,
                uia_text_orphans=pending.uia_text_orphans,
                uia_text_ocr_skipped=pending.uia_text_ocr_skipped,
                icon_ocr_retry_s=icon_ocr_retry_elapsed,
                uia_input_s=pending.uia_input_s,
                uia_scrollbar_s=pending.uia_scrollbar_s,
                uia_text_s=pending.uia_text_s,
            )
        )
        timing_out["debug_layers"] = pending.debug_layers
        timing_out["uia_first_hit"] = False
    return candidates


def _write_partial_vision_timing(
    *,
    timing_out: dict[str, Any] | None,
    yolo_s: float,
    line_s: float,
    uia_input_s: float,
    uia_scrollbar_s: float,
    uia_text_s: float,
    vision_started: float,
    ocr_roi: tuple[int, int, int, int] | None,
    ocr_roi_pad: int,
    uia_text_orphans: int,
    uia_text_ocr_skipped: int,
    debug_layers: dict[str, Any],
    uia_first_pending: bool,
) -> None:
    """Record pre-OCR vision timing (ocr_s / icon retry stay 0 until finish)."""
    if timing_out is None:
        return
    total_elapsed = time.perf_counter() - vision_started
    timing_out.clear()
    timing_out.update(
        _timing_payload_with_roi(
            yolo_s=yolo_s,
            line_s=line_s,
            ocr_s=0.0,
            total_s=total_elapsed,
            ocr_roi=ocr_roi,
            ocr_roi_pad=ocr_roi_pad,
            ocr_reused=0,
            ocr_fresh=0,
            uia_text_orphans=uia_text_orphans,
            uia_text_ocr_skipped=uia_text_ocr_skipped,
            icon_ocr_retry_s=0.0,
            uia_input_s=uia_input_s,
            uia_scrollbar_s=uia_scrollbar_s,
            uia_text_s=uia_text_s,
        )
    )
    timing_out["debug_layers"] = debug_layers
    if uia_first_pending:
        timing_out["uia_first_pending"] = True


def _detect_mouse_targets_from_bgr(
    bgr: np.ndarray,
    *,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    original_scrollbar_bboxes_out: list[tuple[int, int, int, int]] | None = None,
    original_input_bboxes_out: list[tuple[int, int, int, int]] | None = None,
    coord_offset: tuple[int, int] = (0, 0),
    timing_out: dict[str, Any] | None = None,
    ocr_class_ids: frozenset[int] | set[int] | None = None,
    refine_with_uia: bool = True,
    ocr_roi: tuple[int, int, int, int] | None = None,
    ocr_roi_pad: int = DEFAULT_OCR_ROI_PAD,
    ocr_cache_key: int | None = None,
    yolo_crop_roi: tuple[int, int, int, int] | None = None,
    live_input_hwnd: int | None = None,
    live_input_monitor_offset: tuple[int, int] | None = None,
    live_input_window_screen_xywh: tuple[int, int, int, int] | None = None,
    defer_ocr: bool = False,
    pending_out: list[_MouseOcrPending] | None = None,
    uia_prefetched: tuple[
        list[tuple[int, int, int, int]] | None,
        list[tuple[int, int, int, int]] | None,
        list[tuple[tuple[int, int, int, int], str]],
    ]
    | None = None,
) -> list[UiDetection]:
    """Detect mouse-target UI elements on ``bgr`` via YOLO + OCR.

    Returns local-coordinate ``UiDetection`` candidates (text, icons, inputs,
    scrollbars) for move_mouse / recording vision. Overlapping boxes are merged per
    class before OCR — ``text`` only within a line (see
    :func:`cua_mcp.geometry.merge_same_line_boxes`), so stacked list items that
    overlap by a pixel stay separate; results are sorted in reading order. PUA-only element OCR with 2+ icons is split into one
    single-icon detection per glyph (via character spans).

    After YOLO, when a live ``live_input_hwnd`` is provided, one UIA FindAll
    (Edit|Document|ScrollBar|Text|Hyperlink|ListItem|Button|MenuItem|TreeItem|
    DataItem) prefer-replaces YOLO inputs and scrollbars independently, and uses
    labeled controls as OCR assist (split glued YOLO text blobs, IoU fill,
    ROI/blob orphans).
    Non-empty UIA inputs skip Hough line-refine (no H-scrollbar overlap drop).
    Non-empty UIA scrollbars skip arrow-pair create/drop.
    Otherwise horizontal line-pair rectangles refine YOLO ``input`` boxes
    (IoU merge; still drops boxes overlapping horizontal scrollbars).

    When ``original_scrollbar_bboxes_out`` is provided, appends each pre-fit
    scrollbar bbox that differs from the post-fit box (for debug overlays).

    When ``original_input_bboxes_out`` is provided, appends each pre-merge
    YOLO ``input`` bbox that is absent from the post-merge input list.

    When ``timing_out`` is provided, writes ``yolo_s`` / ``line_s`` /
    ``uia_input_s`` / ``uia_scrollbar_s`` / ``uia_text_s`` / ``ocr_s`` /
    ``icon_ocr_retry_s`` / ``total_s`` for this image (replacing prior values
    in that dict). When ``ocr_roi`` is set, also records ``ocr_roi``
    (image-local xywh) and ``ocr_roi_pad``.

    ``ocr_class_ids`` limits which YOLO classes are OCR'd (default: text +
    element). ``refine_with_uia=False`` skips the input line-rectangle merge.
    UIA names are a separate vision path and are not mixed into these candidates.

    When ``yolo_crop_roi`` is set, Stage 1–4 YOLO runs on that image crop (no
    pad) and boxes are translated back to full-image coordinates. When unset,
    first YOLO stays full-frame and ``ocr_roi`` is passed as ``enhance_roi`` for
    Stage 3/4 only.

    When ``ocr_roi`` is set, Stage 3/4 enhance (full-frame path), input refine,
    OCR, and candidate admission are limited to boxes whose centers fall in the
    padded ROI.

    ``ocr_cache_key`` opts this capture into the per-monitor OCR frame cache
    (live ``move_mouse`` only). Entries from outside this capture's ROI stay
    cached while their pixels still match. Unset leaves recording, crops, and
    the viewer uncached.

    When ``defer_ocr=True`` and leftover OCR boxes remain, returns finalized
    UIA/class-only candidates without OCR and appends a ``_MouseOcrPending`` to
    ``pending_out`` (when provided) so the caller can soft-select then finish.
    """
    if ocr_class_ids is None:
        ocr_wanted = frozenset({YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT})
    else:
        ocr_wanted = frozenset(int(c) for c in ocr_class_ids)

    h, w = bgr.shape[:2]
    vision_started = time.perf_counter()
    yolo_started = time.perf_counter()
    yolo_crop_origin = (0, 0)
    yolo_input = bgr
    enhance_roi = ocr_roi
    enhance_pad = ocr_roi_pad
    if yolo_crop_roi is not None:
        rx, ry, rw, rh = (int(v) for v in yolo_crop_roi)
        rx2 = max(rx, min(w, rx + max(0, rw)))
        ry2 = max(ry, min(h, ry + max(0, rh)))
        rx = max(0, min(rx, w))
        ry = max(0, min(ry, h))
        if rx2 > rx and ry2 > ry:
            yolo_input = bgr[ry:ry2, rx:rx2]
            yolo_crop_origin = (rx, ry)
            # Crop is already the window; Stage 3/4 refine the whole crop.
            enhance_roi = None
            enhance_pad = 0
            _log_info(
                "move_mouse yolo crop "
                f"roi=({rx}, {ry}, {rx2 - rx}, {ry2 - ry}) "
                f"full={w}x{h}"
            )
        else:
            _log_info(
                "move_mouse yolo crop empty; falling back to full-frame "
                f"roi={tuple(int(v) for v in yolo_crop_roi)}"
            )
    try:
        xyxy, _scores, class_ids = run_yolo_onnx_end2end(
            yolo_input,
            class_ids=set(MOUSE_TARGET_CLASS_IDS),
            conf_threshold=yolo_conf_threshold,
            enhance_roi=enhance_roi,
            enhance_roi_pad=enhance_pad,
        )
    except Exception as exc:
        _log_info(f"move_mouse YOLO failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"move_mouse YOLO predict failed: {exc}") from exc
    if yolo_crop_origin != (0, 0) and xyxy.size != 0:
        xyxy = _offset_xyxy_rows(xyxy, yolo_crop_origin[0], yolo_crop_origin[1])
    yolo_elapsed = time.perf_counter() - yolo_started

    # Keep text vs element separate so dual-stream decode modes stay correct
    # after overlap merging.
    text_boxes: list[tuple[int, int, int, int]] = []
    element_boxes: list[tuple[int, int, int, int]] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    other_non_ocr: list[tuple[tuple[int, int, int, int], int]] = []

    if xyxy.size != 0:
        for row, cls_id in zip(xyxy, class_ids, strict=True):
            cls_id = int(cls_id)
            bbox = _xyxy_row_to_bbox(row, w, h)
            if cls_id == YOLO_CLASS_TEXT:
                text_boxes.append(bbox)
            elif cls_id == YOLO_CLASS_ELEMENT:
                element_boxes.append(bbox)
            elif cls_id == YOLO_CLASS_INPUT:
                input_boxes.append(bbox)
            else:
                other_non_ocr.append((bbox, cls_id))

    roi_pad = int(ocr_roi_pad)
    text_before_roi = len(text_boxes)
    element_before_roi = len(element_boxes)
    input_before_roi = len(input_boxes)
    other_before_roi = len(other_non_ocr)
    if ocr_roi is not None:
        text_boxes = [
            b for b in text_boxes if _bbox_center_in_ocr_roi(b, ocr_roi, pad=roi_pad)
        ]
        element_boxes = [
            b for b in element_boxes if _bbox_center_in_ocr_roi(b, ocr_roi, pad=roi_pad)
        ]
        input_boxes = [
            b for b in input_boxes if _bbox_center_in_ocr_roi(b, ocr_roi, pad=roi_pad)
        ]
        other_non_ocr = [
            (b, c)
            for b, c in other_non_ocr
            if _bbox_center_in_ocr_roi(b, ocr_roi, pad=roi_pad)
        ]
        skipped_boxes = (
            (text_before_roi - len(text_boxes))
            + (element_before_roi - len(element_boxes))
            + (input_before_roi - len(input_boxes))
            + (other_before_roi - len(other_non_ocr))
        )
        _log_info(
            "move_mouse ocr_roi gate "
            f"roi={tuple(int(v) for v in ocr_roi)} pad={roi_pad} "
            f"boxes_dropped={skipped_boxes}"
        )

    from src.recorder.uia_input_boxes import merge_nested_element_boxes

    element_before_nest = len(element_boxes)
    element_boxes = merge_nested_element_boxes(element_boxes)
    if len(element_boxes) < element_before_nest:
        _log_info(
            "move_mouse nested element merge "
            f"before={element_before_nest} after={len(element_boxes)}"
        )

    # Image-local YOLO boxes after ROI gate, before UIA prefer-replace / OCR.
    debug_yolo_boxes: list[dict[str, Any]] = [
        {"bbox": [int(v) for v in bbox], "class_name": "text"} for bbox in text_boxes
    ]
    debug_yolo_boxes.extend(
        {"bbox": [int(v) for v in bbox], "class_name": "element"}
        for bbox in element_boxes
    )
    debug_yolo_boxes.extend(
        {"bbox": [int(v) for v in bbox], "class_name": "input"} for bbox in input_boxes
    )
    debug_yolo_boxes.extend(
        {
            "bbox": [int(v) for v in bbox],
            "class_name": YOLO_CLASS_NAMES.get(int(cls_id), str(int(cls_id))),
        }
        for bbox, cls_id in other_non_ocr
    )

    line_elapsed = 0.0
    uia_input_elapsed = 0.0
    uia_scrollbar_elapsed = 0.0
    uia_text_elapsed = 0.0
    used_uia_scrollbar = False
    uia_labeled_texts: list[tuple[tuple[int, int, int, int], str]] = []
    debug_uia_inputs: list[tuple[int, int, int, int]] = []
    debug_uia_scrollbars: list[tuple[int, int, int, int]] = []
    # A supplied snapshot is debug-only. Candidates on this path are YOLO + OCR.
    if uia_prefetched is not None:
        uia_inputs, uia_scrollbars, uia_labeled_texts = uia_prefetched
        if uia_inputs:
            debug_uia_inputs = list(uia_inputs)
        if uia_scrollbars:
            debug_uia_scrollbars = list(uia_scrollbars)
        _log_info(
            "move_mouse uia snapshot ignored for candidates "
            f"uia_inputs={len(uia_inputs or [])} "
            f"uia_scrollbars={len(uia_scrollbars or [])} "
            f"uia_texts={len(uia_labeled_texts)}"
        )
    _ = live_input_hwnd, live_input_monitor_offset, live_input_window_screen_xywh
    if refine_with_uia:
        pre_merge_inputs = (
            list(input_boxes) if original_input_bboxes_out is not None else []
        )
        horizontal_scrollbar_boxes = [
            bbox
            for bbox, cls_id in other_non_ocr
            if cls_id == YOLO_CLASS_SCROLLBAR
            and scrollbar_orientation(bbox) == "horizontal"
        ]
        line_started = time.perf_counter()
        try:
            input_boxes = merge_yolo_inputs_with_line_rectangles(
                bgr,
                input_boxes,
                img_w=w,
                img_h=h,
                horizontal_scrollbar_boxes=horizontal_scrollbar_boxes,
            )
        except Exception as exc:
            _log_info(
                f"move_mouse input-box line refine failed: {type(exc).__name__}: {exc}"
            )
        line_elapsed = time.perf_counter() - line_started
        if original_input_bboxes_out is not None:
            merged_inputs = set(input_boxes)
            for bbox in pre_merge_inputs:
                if bbox not in merged_inputs:
                    original_input_bboxes_out.append(bbox)
        # Re-admit only in-ROI merged inputs when gating.
        if ocr_roi is not None:
            input_boxes = [
                b
                for b in input_boxes
                if _bbox_center_in_ocr_roi(b, ocr_roi, pad=roi_pad)
            ]
    non_ocr: list[tuple[tuple[int, int, int, int], int]] = [
        (bbox, YOLO_CLASS_INPUT) for bbox in input_boxes
    ]
    non_ocr.extend(other_non_ocr)

    # YOLO text and element boxes are OCR'd as a set. UIA names are not mixed in.
    merged_text_boxes = (
        merge_same_line_boxes(text_boxes) if text_boxes else []
    )
    uia_text_detections: list[tuple[tuple[int, int, int, int], str]] = []
    yolo_text_for_ocr = list(merged_text_boxes)
    yolo_element_for_ocr = list(element_boxes)
    uia_text_orphans = 0
    uia_text_ocr_skipped = 0

    debug_layers = _vision_debug_layers(
        debug_yolo_boxes,
        uia_labeled_texts,
        debug_uia_inputs,
        debug_uia_scrollbars,
    )
    if (
        not yolo_text_for_ocr
        and not yolo_element_for_ocr
        and not non_ocr
        and not uia_text_detections
    ):
        if ocr_cache_key is not None:
            _recognize_ocr_boxes(bgr, [], [], cache_key=ocr_cache_key)
        total_elapsed = time.perf_counter() - vision_started
        _log_info(
            f"move_mouse vision profile yolo_s={yolo_elapsed:.3f} "
            f"line_s={line_elapsed:.3f} "
            f"uia_input_s={uia_input_elapsed:.3f} "
            f"uia_scrollbar_s={uia_scrollbar_elapsed:.3f} "
            f"uia_text_s={uia_text_elapsed:.3f} ocr_s=0.000 "
            f"icon_ocr_retry_s=0.000 "
            f"ocr_reused=0 ocr_fresh=0 "
            f"total_s={total_elapsed:.3f} detections=0"
        )
        _write_partial_vision_timing(
            timing_out=timing_out,
            yolo_s=yolo_elapsed,
            line_s=line_elapsed,
            uia_input_s=uia_input_elapsed,
            uia_scrollbar_s=uia_scrollbar_elapsed,
            uia_text_s=uia_text_elapsed,
            vision_started=vision_started,
            ocr_roi=ocr_roi,
            ocr_roi_pad=roi_pad,
            uia_text_orphans=uia_text_orphans,
            uia_text_ocr_skipped=uia_text_ocr_skipped,
            debug_layers=debug_layers,
            uia_first_pending=False,
        )
        return []

    ocr_boxes: list[tuple[int, int, int, int]] = []
    ocr_class_id_list: list[int] = []
    if YOLO_CLASS_TEXT in ocr_wanted:
        for bbox in yolo_text_for_ocr:
            ocr_boxes.append(bbox)
            ocr_class_id_list.append(YOLO_CLASS_TEXT)
    if YOLO_CLASS_ELEMENT in ocr_wanted:
        for bbox in merge_overlapping_boxes(yolo_element_for_ocr):
            ocr_boxes.append(bbox)
            ocr_class_id_list.append(YOLO_CLASS_ELEMENT)

    partial_candidates: list[UiDetection] = []
    for bbox, cls_id in non_ocr:
        partial_candidates.append(
            _detection_from_bbox(bbox, cls_id, source="yolo")
        )
    for bbox, label in uia_text_detections:
        partial_candidates.append(
            _detection_from_bbox(
                bbox, YOLO_CLASS_TEXT, text=label, source="uia"
            )
        )

    yolo_box_count = 0 if xyxy.size == 0 else len(xyxy)
    pending = _MouseOcrPending(
        bgr=bgr,
        ocr_boxes=list(ocr_boxes),
        ocr_class_id_list=list(ocr_class_id_list),
        partial_candidates=list(partial_candidates),
        used_uia_scrollbar=used_uia_scrollbar,
        coord_offset=coord_offset,
        ocr_cache_key=ocr_cache_key,
        ocr_wanted=ocr_wanted,
        yolo_s=yolo_elapsed,
        line_s=line_elapsed,
        uia_input_s=uia_input_elapsed,
        uia_scrollbar_s=uia_scrollbar_elapsed,
        uia_text_s=uia_text_elapsed,
        vision_started=vision_started,
        uia_text_orphans=uia_text_orphans,
        uia_text_ocr_skipped=uia_text_ocr_skipped,
        ocr_roi=ocr_roi,
        ocr_roi_pad=roi_pad,
        refine_with_uia=refine_with_uia,
        yolo_box_count=yolo_box_count,
        input_box_count=len(input_boxes),
        debug_layers=debug_layers,
        original_scrollbar_bboxes_out=original_scrollbar_bboxes_out,
    )

    if defer_ocr and ocr_boxes:
        if pending_out is not None:
            pending_out.append(pending)
        candidates = _finalize_mouse_candidates(
            list(partial_candidates),
            used_uia_scrollbar=used_uia_scrollbar,
            coord_offset=coord_offset,
            # Scrollbar originals are recorded on finish / non-deferred path.
            original_scrollbar_bboxes_out=None,
        )
        total_elapsed = time.perf_counter() - vision_started
        _log_info(
            "move_mouse vision profile (ocr deferred) "
            f"yolo_s={yolo_elapsed:.3f} line_s={line_elapsed:.3f} "
            f"uia_input_s={uia_input_elapsed:.3f} "
            f"uia_scrollbar_s={uia_scrollbar_elapsed:.3f} "
            f"uia_text_s={uia_text_elapsed:.3f} "
            f"ocr_s=0.000 icon_ocr_retry_s=0.000 "
            f"ocr_reused=0 ocr_fresh=0 "
            f"total_s={total_elapsed:.3f} "
            f"yolo_boxes={yolo_box_count} "
            f"input_boxes={len(input_boxes)} ocr_boxes={len(ocr_boxes)} "
            f"candidates={len(candidates)} refine_with_uia={refine_with_uia}"
            + (
                f" ocr_roi={tuple(int(v) for v in ocr_roi)}"
                if ocr_roi is not None
                else ""
            )
        )
        _write_partial_vision_timing(
            timing_out=timing_out,
            yolo_s=yolo_elapsed,
            line_s=line_elapsed,
            uia_input_s=uia_input_elapsed,
            uia_scrollbar_s=uia_scrollbar_elapsed,
            uia_text_s=uia_text_elapsed,
            vision_started=vision_started,
            ocr_roi=ocr_roi,
            ocr_roi_pad=roi_pad,
            uia_text_orphans=uia_text_orphans,
            uia_text_ocr_skipped=uia_text_ocr_skipped,
            debug_layers=debug_layers,
            uia_first_pending=True,
        )
        return candidates

    return _finish_mouse_ocr_pending(pending, timing_out=timing_out)


# Class-only anchors (no OCR/icons) need these labels so similarity prefilter
# can keep them when the instruction is e.g. 輸入欄 or 滾動條.
_CLASS_MATCH_LABELS: frozenset[str] = frozenset({"input", "scrollbar"})


def _detection_match_labels(det: UiDetection) -> set[str]:
    """Labels used to find similar candidates: OCR text, icon ids, and class labels.

    For ``input`` / ``scrollbar``, also includes the Chinese class name (輸入欄 /
    滾動條) so class-only anchors survive the similarity prefilter.
    """
    labels: set[str] = set()
    text = (det.text or "").strip()
    if text:
        labels.add(text)
    for icon in det.icons or []:
        chinese_id = str(icon.get("chinese_id", "")).strip()
        if chinese_id:
            labels.add(chinese_id)
    if det.class_name in _CLASS_MATCH_LABELS:
        class_label = _ANCHOR_SUFFIX_BY_CLASS.get(det.class_name, "").strip()
        if class_label:
            labels.add(class_label)
    return labels


def _expand_keep_indices_with_similar(
    detections: list[UiDetection],
    keep_indices: list[int],
) -> list[int]:
    """
    Expand ``keep_indices`` with every detection that shares a text/icon label.

    If the model keeps one ``圖片`` text row, also keep other ``圖片`` text rows and
    element rows whose icon label is ``圖片``. Blank detections (no text/icons) are
    never used as similarity seeds and never matched by label overlap.
    """
    if not keep_indices:
        return []

    seed_labels: set[str] = set()
    for idx in keep_indices:
        if 0 <= idx < len(detections):
            seed_labels |= _detection_match_labels(detections[idx])

    if not seed_labels:
        return list(keep_indices)

    kept = set(keep_indices)
    expanded: list[int] = []
    for i, det in enumerate(detections):
        if i in kept or (_detection_match_labels(det) & seed_labels):
            expanded.append(i)
    return expanded


# Quote/bracket pairs that may wrap a hub target name (e.g. 「擷取」文字).
# Pair forms use distinct open/close; ASCII ``"…"`` uses the same glyph twice.
_SIMILARITY_WRAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("「", "」"),
    ("『", "』"),
    ("【", "】"),
    ("〔", "〕"),
    ("[", "]"),
    ('"', '"'),
)

# Whitespace variants stripped at edges before unwrap; pipe added after unwrap.
_SIMILARITY_WHITESPACE = " \t\n\r\f\v\u00a0\u3000"
_SIMILARITY_EDGE_CHARS = _SIMILARITY_WHITESPACE + "|"
_SIMILARITY_SPACE_FOLD = str.maketrans({"\u00a0": " ", "\u3000": " "})
_SIMILARITY_DASH_FOLD = str.maketrans(
    {
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\uff0d": "-",  # fullwidth hyphen-minus
        "\u2500": "-",  # box drawings light horizontal
    }
)
_SIMILARITY_CJK_PUNCT_FOLD = str.maketrans({"。": ".", "、": ","})
_SIMILARITY_LOOKALIKE_FOLD = str.maketrans({"I": "l", "1": "l", "O": "0"})


def _strip_similarity_edges(text: str, chars: str = _SIMILARITY_EDGE_CHARS) -> str:
    """Strip leading/trailing ``chars`` until stable."""
    while True:
        stripped = text.strip(chars)
        if stripped == text:
            return stripped
        text = stripped


def _canonicalize_similarity_text(text: str) -> str:
    """OCR/typography folds applied after hub-wrapper unwrap."""
    text = _strip_similarity_edges(text)
    if not text:
        return ""
    # Pre-NFKC: fold leaders that NFKC would otherwise turn into ``..`` / leave odd.
    text = text.replace("⋯", "…").replace("‥", "…")
    text = unicodedata.normalize("NFKC", text)
    # Post-NFKC: HORIZONTAL ELLIPSIS compatibility-decomposes to ASCII ``...``.
    text = text.replace("...", "…")
    text = text.translate(_SIMILARITY_SPACE_FOLD)
    text = _strip_similarity_edges(text)
    text = text.translate(_SIMILARITY_DASH_FOLD)
    text = text.translate(_SIMILARITY_CJK_PUNCT_FOLD)
    return text.translate(_SIMILARITY_LOOKALIKE_FOLD)


def _normalize_similarity_label(label: str) -> str:
    """Normalize a label for similarity matching against OCR / hub queries.

    Unwrap hub wrappers so ``「擷取」文字`` compares as ``擷取`` against raw OCR
    (leftmost pair among ``""`` / ``「」`` / ``『』`` / ``【】`` / ``〔〕`` / ``[]``).
    Then canonicalize: strip edge whitespace/``|``, fold ellipsis leaders, NFKC
    (then re-fold ASCII ``...``), space/dash/CJK punct folds, and lookalikes
    ``I``/``1``→``l``, ``O``→``0``. Class-only anchors such as ``輸入欄`` /
    ``滾動條`` are unchanged aside from those folds.
    """
    text = _strip_similarity_edges(label or "", _SIMILARITY_WHITESPACE)
    if not text:
        return ""

    best_start: int | None = None
    best_end: int | None = None
    for open_ch, close_ch in _SIMILARITY_WRAP_PAIRS:
        start = text.find(open_ch)
        if start < 0:
            continue
        end = text.find(close_ch, start + len(open_ch))
        if end <= start:
            continue
        if best_start is None or start < best_start:
            best_start, best_end = start, end

    if best_start is not None and best_end is not None:
        inner = _strip_similarity_edges(
            text[best_start + 1 : best_end], _SIMILARITY_WHITESPACE
        )
        if inner:
            text = inner
    return _canonicalize_similarity_text(text)


def _label_similarity(a: str, b: str) -> float:
    """Case-insensitive string similarity in ``[0, 1]``."""
    left = (a or "").strip()
    right = (b or "").strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_n = _normalize_similarity_label(left)
    right_n = _normalize_similarity_label(right)
    if left_n == right_n:
        return 1.0
    return SequenceMatcher(None, left_n.casefold(), right_n.casefold()).ratio()


def _detection_similarity_to_query(det: UiDetection, query: str) -> float:
    """Best similarity between ``query`` and a detection's OCR text / icon ids."""
    query = (query or "").strip()
    if not query:
        return 0.0
    return max(
        (
            _label_similarity(query, label)
            for label in _detection_match_labels(det)
            if label
        ),
        default=0.0,
    )


# Minimum SequenceMatcher score to keep a candidate before the LLM filter.
_MOUSE_FILTER_SIMILARITY_THRESHOLD = 0.5


def _detections_label_similar(
    left: UiDetection,
    right: UiDetection,
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> bool:
    """True when any OCR/icon/class label pair scores at least ``threshold``."""
    labels_left = _detection_match_labels(left)
    labels_right = _detection_match_labels(right)
    if not labels_left or not labels_right:
        return False
    for a in labels_left:
        for b in labels_right:
            if _label_similarity(a, b) >= threshold:
                return True
    return False


def _detections_similar_to(
    chosen: UiDetection,
    detections: list[UiDetection],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """
    Return YOLO detections label-similar to ``chosen``, in reading order.

    Peers are sorted top-to-bottom then left-to-right so overlay box numbers and
    LLM candidate ``[index N]`` rows match spatial ordinals (第一 / 第二 / …).
    ``chosen`` is always included when it has match labels. Blank detections
    never match.
    """
    if not _detection_match_labels(chosen):
        return [chosen]

    peers: list[UiDetection] = [chosen]
    seen: set[int] = {id(chosen)}
    for det in detections:
        if id(det) in seen:
            continue
        if not _detections_label_similar(chosen, det, threshold=threshold):
            continue
        peers.append(det)
        seen.add(id(det))
    return _sort_detections_reading_order(peers)


def _monitor_index_from_image_path(path: str) -> int | None:
    """Parse ``_mon{N}`` from a capture filename when present."""
    match = re.search(r"_mon(\d+)\.[^.]+$", path.replace("\\", "/"))
    if not match:
        return None
    return int(match.group(1))


def _monitor_geometry(monitor_index: int) -> tuple[int, int, int, int]:
    """Return ``(left, top, width, height)`` for ``monitor_index``."""
    for entry in monitor_details():
        if int(entry["index"]) == int(monitor_index):
            return (
                int(entry["left"]),
                int(entry["top"]),
                int(entry["width"]),
                int(entry["height"]),
            )
    left, top = active_monitor_offset(monitor_index)
    return left, top, 0, 0


def _local_bbox_on_monitor(
    bbox: tuple[int, int, int, int],
    *,
    left: int,
    top: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """Map a virtual-desktop ``(x,y,w,h)`` into image-local coords, or None if off-screen."""
    x, y, w, h = bbox
    lx = int(x) - int(left)
    ly = int(y) - int(top)
    rx2 = lx + int(w)
    ry2 = ly + int(h)
    if rx2 <= 0 or ry2 <= 0 or lx >= img_w or ly >= img_h:
        return None
    return clip_box(lx, ly, int(w), int(h), img_w, img_h)


def _draw_indexed_bbox(
    image_bgr: np.ndarray,
    local_bbox: tuple[int, int, int, int],
    index: int,
) -> None:
    """Draw a high-contrast box and index label onto ``image_bgr`` in-place."""
    x, y, w, h = local_bbox
    color = (0, 255, 255)  # yellow in BGR
    thickness = 2
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, thickness)

    label = str(index)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    pad = 3
    label_x = x
    label_y = max(th + pad * 2, y)
    # Prefer above the box when there is room; otherwise inside the top edge.
    if y - th - pad * 2 >= 0:
        bg_y1 = y - th - pad * 2
        bg_y2 = y
        text_org = (label_x + pad, y - pad - baseline)
    else:
        bg_y1 = y
        bg_y2 = min(image_bgr.shape[0], y + th + pad * 2)
        text_org = (label_x + pad, y + th + pad)
    bg_x2 = min(image_bgr.shape[1], label_x + tw + pad * 2)
    cv2.rectangle(image_bgr, (label_x, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.putText(
        image_bgr,
        label,
        text_org,
        font,
        font_scale,
        color,
        text_thickness,
        cv2.LINE_AA,
    )


def _write_indexed_bbox_overlay_images(
    candidates: list[UiDetection],
    image_paths: list[str],
    monitor_indices: list[int],
    output_dir: Path,
    *,
    stamp: str,
) -> list[str]:
    """
    Write per-monitor screenshots with candidate index boxes and return those paths.

    Candidate bboxes are virtual-desktop coordinates. Falls back to the original
    ``image_paths`` when no annotated image can be written.
    """
    if not candidates or not image_paths:
        return list(image_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated: list[str] = []
    drew_any = False

    for i, image_path in enumerate(image_paths):
        monitor_index = (
            monitor_indices[i]
            if i < len(monitor_indices)
            else _monitor_index_from_image_path(image_path)
        )
        if monitor_index is None:
            annotated.append(image_path)
            continue

        bgr = imread_bgr(image_path)
        if bgr is None:
            _log_info(
                "move_mouse indexed overlay could not read "
                f"path={image_path}"
            )
            annotated.append(image_path)
            continue

        left, top, _mw, _mh = _monitor_geometry(int(monitor_index))
        img_h, img_w = bgr.shape[:2]
        canvas = bgr.copy()
        drawn_here = 0
        for idx, det in enumerate(candidates):
            local = _local_bbox_on_monitor(
                det.bbox,
                left=left,
                top=top,
                img_w=img_w,
                img_h=img_h,
            )
            if local is None:
                continue
            _draw_indexed_bbox(canvas, local, idx)
            drawn_here += 1

        out_path = output_dir / f"{stamp}_indexed_mon{monitor_index}.png"
        if not imwrite_bgr(out_path, canvas):
            _log_info(
                "move_mouse indexed overlay write failed "
                f"path={out_path}"
            )
            annotated.append(image_path)
            continue

        annotated.append(str(out_path.resolve()))
        if drawn_here:
            drew_any = True
        _log_info(
            "move_mouse indexed overlay "
            f"monitor={monitor_index} boxes={drawn_here} path={out_path}"
        )

    if not drew_any:
        return list(image_paths)
    return annotated


def _anchor_indices_by_top_similarity(
    detections: list[UiDetection],
    anchor: str,
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[int]:
    """Keep detections whose anchor similarity equals the best score (ties only).

    Returns an empty list when the best score is below ``threshold``.
    """
    if not detections or not (anchor or "").strip():
        return []

    scored: list[tuple[int, float]] = [
        (i, _detection_similarity_to_query(det, anchor))
        for i, det in enumerate(detections)
    ]
    best = max(score for _, score in scored)
    if best < threshold:
        return []
    return [i for i, score in scored if score == best]


def _prefilter_detections_by_similarity(
    detections: list[UiDetection],
    anchor: str,
    nearby: list[str],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> tuple[list[int], list[int]]:
    """Return anchor/nearby index buckets using string similarity per query.

    Both anchor and each nearby label keep only the highest similarity score
    (ties kept). Lower-scoring partial matches above ``threshold`` are dropped
    so near-miss OCR cousins (e.g. 「新竹總部」 for 「新竹公司」) cannot enter
    the nearby landmark set. Indices that match the anchor are excluded from
    nearby (anchor wins).
    """
    anchor_indices = _anchor_indices_by_top_similarity(
        detections, anchor, threshold=threshold
    )
    anchor_set = set(anchor_indices)

    nearby_indices: list[int] = []
    nearby_seen: set[int] = set()
    for query in nearby:
        if not (query or "").strip():
            continue
        # Score against the full list so best is global; then skip anchors/dupes.
        top = _anchor_indices_by_top_similarity(
            detections, query, threshold=threshold
        )
        for i in top:
            if i in anchor_set or i in nearby_seen:
                continue
            nearby_indices.append(i)
            nearby_seen.add(i)

    return anchor_indices, nearby_indices


def _filter_mouse_candidates(
    detections: list[UiDetection],
    anchor: str,
    nearby: list[str],
) -> tuple[list[UiDetection], list[UiDetection]]:
    """Split detections into anchor vs nearby buckets via string similarity."""
    if not detections:
        return [], []

    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections, anchor, nearby
    )
    _log_info(
        "_filter_mouse_candidates: similarity split "
        f"threshold={_MOUSE_FILTER_SIMILARITY_THRESHOLD} "
        f"detections={len(detections)} anchor={len(anchor_indices)} "
        f"nearby={len(nearby_indices)}"
    )
    if not anchor_indices and not nearby_indices:
        _log_info(
            "_filter_mouse_candidates: no similarity matches; returning not found"
        )
        return [], []

    anchor_expanded = _expand_keep_indices_with_similar(detections, anchor_indices)
    nearby_expanded = _expand_keep_indices_with_similar(detections, nearby_indices)
    expanded_nearby_count = len(nearby_expanded)
    anchor_set = set(anchor_expanded)
    nearby_expanded = [i for i in nearby_expanded if i not in anchor_set]

    if (
        len(anchor_expanded) != len(anchor_indices)
        or expanded_nearby_count != len(nearby_indices)
    ):
        _log_info(
            "_filter_mouse_candidates: expanded similar "
            f"anchor={len(anchor_indices)} after_expand_anchor={len(anchor_expanded)} "
            f"nearby={len(nearby_indices)} after_expand_nearby={expanded_nearby_count} "
            f"nearby_after_dedupe={len(nearby_expanded)}"
        )

    return (
        [detections[i] for i in anchor_expanded],
        [detections[i] for i in nearby_expanded],
    )


# Blank/unknown peers admitted as icon candidates must be size-compatible with the
# labeled icon seed(s). Ratio is max(seed/cand, cand/seed) on either side.
_ICON_PEER_MAX_SIDE_RATIO = 2.5
# When no labeled icon seed size is available, keep only small glyph-like boxes.
_ICON_PEER_FALLBACK_MAX_SIDE_PX = 64


def _instruction_targets_icon(anchor: str, labeled_anchors: list[UiDetection]) -> bool:
    """True when the mouse target is an icon (instruction or labeled seeds)."""
    if "圖示" in (anchor or ""):
        return True
    return any(_has_actual_icon_labels(det) for det in labeled_anchors)


def _is_blank_unknown_icon_candidate(det: UiDetection) -> bool:
    """True for OCR-failed icon boxes: ``unknown`` with no usable text/icon ids."""
    if not _is_unknown_detection(det):
        return False
    if _has_actual_visible_text(det):
        return False
    if _has_actual_icon_labels(det):
        return False
    return True


def _icon_peer_size_compatible(
    candidate: UiDetection,
    seeds: list[UiDetection],
) -> bool:
    """True when ``candidate`` is roughly the same size as a labeled icon seed."""
    _, _, cw, ch = candidate.bbox
    if cw < 1 or ch < 1:
        return False
    icon_seeds = [s for s in seeds if _has_actual_icon_labels(s)]
    if not icon_seeds:
        return max(cw, ch) <= _ICON_PEER_FALLBACK_MAX_SIDE_PX
    for seed in icon_seeds:
        _, _, sw, sh = seed.bbox
        if sw < 1 or sh < 1:
            continue
        ratio = max(cw / sw, sw / cw, ch / sh, sh / ch)
        if ratio <= _ICON_PEER_MAX_SIDE_RATIO:
            return True
    return False


def _unknown_covers_directed_hints(
    candidate: UiDetection,
    nearby_matches: list[UiDetection],
    hints: list[NearbyHint],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> bool:
    """True when ``candidate`` satisfies every directed nearby side hint."""
    directed = [hint for hint in hints if hint.side is not None]
    if not directed or not nearby_matches:
        return False
    return all(
        _hint_covered_by_neighbors(
            candidate,
            nearby_matches,
            hint,
            threshold=threshold,
            require_side=True,
        )
        for hint in directed
    )


def _admit_unknown_icon_peers(
    detections: list[UiDetection],
    labeled_anchors: list[UiDetection],
    nearby_matches: list[UiDetection],
    nearby_labels: list[str] | list[NearbyHint] | None,
    *,
    anchor: str,
) -> list[UiDetection]:
    """
    Admit blank ``unknown`` detections that pass directed icon landmarks.

    When the target is an icon and OCR missed a peer glyph (class ``unknown``),
    similarity expansion cannot find it. If directed nearby hints are present,
    re-admit size-compatible unknowns that satisfy those sides so nearby
    prefilter / LLM pick can choose the correct peer.
    """
    hints = normalize_nearby_hints(nearby_labels)
    directed = [hint for hint in hints if hint.side is not None]
    if (
        not labeled_anchors
        or not detections
        or not directed
        or not nearby_matches
        or not _instruction_targets_icon(anchor, labeled_anchors)
    ):
        return []

    labeled_ids = {id(det) for det in labeled_anchors}
    admitted: list[UiDetection] = []
    for det in detections:
        if id(det) in labeled_ids:
            continue
        if not _is_blank_unknown_icon_candidate(det):
            continue
        if not _icon_peer_size_compatible(det, labeled_anchors):
            continue
        if not _unknown_covers_directed_hints(det, nearby_matches, hints):
            continue
        admitted.append(det)
    return admitted


def _merge_anchor_detections(
    labeled: list[UiDetection],
    admitted: list[UiDetection],
) -> list[UiDetection]:
    """Append admitted peers not already present (by object identity / bbox)."""
    if not admitted:
        return list(labeled)
    seen_ids = {id(det) for det in labeled}
    seen_bboxes = {det.bbox for det in labeled}
    merged = list(labeled)
    for det in admitted:
        if id(det) in seen_ids or det.bbox in seen_bboxes:
            continue
        merged.append(det)
        seen_ids.add(id(det))
        seen_bboxes.add(det.bbox)
    return merged


def _detections_for_captured_monitor(
    monitor_index: int,
    bgr: np.ndarray,
    *,
    yolo_conf_threshold: float,
    timing_out: dict[str, Any] | None = None,
    ocr_class_ids: frozenset[int] | set[int] | None = None,
    refine_with_uia: bool = True,
    ocr_roi: tuple[int, int, int, int] | None = None,
    ocr_roi_pad: int = DEFAULT_OCR_ROI_PAD,
    yolo_crop_roi: tuple[int, int, int, int] | None = None,
    live_input_hwnd: int | None = None,
    live_input_window_screen_xywh: tuple[int, int, int, int] | None = None,
    defer_ocr: bool = False,
    pending_out: list[_MouseOcrPending] | None = None,
) -> list[UiDetection]:
    """YOLO+OCR on one monitor image, then map boxes into virtual-desktop coords."""
    left, top = active_monitor_offset(monitor_index)
    local_candidates = _detect_mouse_targets_from_bgr(
        bgr,
        yolo_conf_threshold=yolo_conf_threshold,
        coord_offset=(left, top),
        timing_out=timing_out,
        ocr_class_ids=ocr_class_ids,
        refine_with_uia=refine_with_uia,
        ocr_roi=ocr_roi,
        ocr_roi_pad=ocr_roi_pad,
        ocr_cache_key=monitor_index,
        yolo_crop_roi=yolo_crop_roi,
        live_input_hwnd=live_input_hwnd,
        live_input_monitor_offset=(left, top),
        live_input_window_screen_xywh=live_input_window_screen_xywh,
        defer_ocr=defer_ocr,
        pending_out=pending_out,
    )
    return [_offset_detection(d, left, top) for d in local_candidates]


def _monitor_capture_for_global_point(
    captured: list[tuple[int, np.ndarray]],
    global_x: int,
    global_y: int,
) -> tuple[int, np.ndarray, int, int] | None:
    """Return ``(monitor_index, bgr, left, top)`` for a virtual-desktop point."""
    for monitor_index, bgr in captured:
        left, top = active_monitor_offset(monitor_index)
        img_h, img_w = bgr.shape[:2]
        if left <= global_x < left + img_w and top <= global_y < top + img_h:
            return monitor_index, bgr, left, top
    if captured:
        monitor_index, bgr = captured[0]
        left, top = active_monitor_offset(monitor_index)
        return monitor_index, bgr, left, top
    return None


def _resolve_char_target_point(
    chosen: UiDetection,
    captured: list[tuple[int, np.ndarray]],
    char_target: str,
    *,
    char_occurrence: int,
) -> tuple[int, int] | None:
    """Resolve a char-target instruction to global virtual-desktop coordinates."""
    monitor_info = _monitor_capture_for_global_point(captured, chosen.cx, chosen.cy)
    if monitor_info is None:
        return None

    _monitor_index, bgr, left, top = monitor_info
    img_h, img_w = bgr.shape[:2]
    bx, by, bw, bh = chosen.bbox
    local_bbox = (bx - left, by - top, bw, bh)
    mode = ocr_mode_for_yolo_class(chosen.class_id)
    _text, spans = ocr_box_with_spans(bgr, local_bbox, mode=mode)
    local_point = resolve_char_screen_point(
        local_bbox,
        spans,
        char_target,
        occurrence=char_occurrence,
        img_w=img_w,
        img_h=img_h,
    )
    if local_point is None:
        return None
    return local_point[0] + left, local_point[1] + top


def _screen_xyxy_to_image_roi(
    screen_xyxy: tuple[int, int, int, int],
    *,
    image_w: int,
    image_h: int,
    monitor_offset: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Clip a screen ``(left, top, right, bottom)`` rect to image-local xywh."""
    from src.recorder.window_snapshot import _clip_xywh_to_image

    left, top, right, bottom = (int(v) for v in screen_xyxy)
    ox, oy = int(monitor_offset[0]), int(monitor_offset[1])
    return _clip_xywh_to_image(
        (left - ox, top - oy, right - left, bottom - top),
        image_w=image_w,
        image_h=image_h,
    )


def _any_monitor_has_live_click_window(
    captured: list[tuple[int, np.ndarray]],
    click_window: dict[str, Any],
) -> bool:
    """True when a live match of ``click_window`` intersects a captured image."""
    from src.recorder.window_snapshot import resolve_live_ocr_roi_local

    for monitor_index, bgr in captured:
        img_h, img_w = bgr.shape[:2]
        live_roi = resolve_live_ocr_roi_local(
            click_window,
            image_w=img_w,
            image_h=img_h,
            monitor_offset=active_monitor_offset(monitor_index),
        )
        if live_roi is not None:
            return True
    return False


def _collect_monitor_detections(
    captured: list[tuple[int, np.ndarray]],
    *,
    yolo_conf_threshold: float,
    timing_out: dict[str, Any] | None = None,
    ocr_class_ids: frozenset[int] | set[int] | None = None,
    refine_with_uia: bool = True,
    click_window: dict[str, Any] | None = None,
    ocr_roi_pad: int = DEFAULT_OCR_ROI_PAD,
    caption_roi_screen: tuple[int, int, int, int] | None = None,
    landmark_centers_image: list[tuple[float, float]] | None = None,
    defer_ocr: bool = False,
) -> list[UiDetection]:
    """
    Run YOLO+OCR on captured monitors.

    Capture stays sequential (caller); inference runs in parallel when there is more
    than one monitor so Triton/ORT work can overlap.

    When ``timing_out`` is set, accumulates per-monitor ``yolo_s`` / ``line_s`` /
    ``ocr_s`` / ``icon_ocr_retry_s`` / ``total_s`` (sums across monitors) and
    records ``ocr_roi`` / ``ocr_rois`` when ROI gating was applied.

    When ``defer_ocr=True``, leftover OCR is skipped and any ``_MouseOcrPending``
    objects are stored on ``timing_out["_ocr_pending"]`` for a later finish pass.

    When ``click_window`` is set, each monitor resolves an OCR/gate ROI via
    :func:`resolve_ocr_roi_local` (live match, else recorded rect). Stage 1 YOLO
    crops only when a **live** match exists
    (:func:`resolve_live_ocr_roi_local`) and recorded landmark centers (if any)
    all lie inside the recorded window. A monitor with no live window is not
    searched. If no captured monitor intersects a live match, this raises
    :exc:`ValueError` (:data:`CLICK_WINDOW_MISS_MESSAGE`) instead of scanning
    other windows. Landmark centers that sit outside the recorded window still
    use a full-frame search, but only after a live window was found.

    ``caption_roi_screen`` replaces that window rect with one caption-button cell
    (screen ``left, top, right, bottom``). Monitors that miss the cell are skipped.
    If the cell misses every image, detection falls back to the window ROI.
    If any monitor has an ROI, monitors that do not intersect it are skipped.
    Full-frame detection runs when ``click_window`` is missing, or when a live
    window exists but no monitor ROI intersects the captures.
    """
    if not captured:
        if timing_out is not None:
            timing_out.clear()
            timing_out.update(
                {
                    "yolo_s": 0.0,
                    "line_s": 0.0,
                    "uia_input_s": 0.0,
                    "uia_scrollbar_s": 0.0,
                    "uia_text_s": 0.0,
                    "ocr_s": 0.0,
                    "icon_ocr_retry_s": 0.0,
                    "total_s": 0.0,
                }
            )
        return []

    # Lazy import: avoid cua_mcp ↔ recorder circular import at module load.
    from src.recorder.window_snapshot import (
        find_matching_click_window,
        resolve_live_ocr_roi_local,
        resolve_ocr_roi_local,
        resolve_recorded_ocr_roi_local,
    )

    landmark_centers = list(landmark_centers_image or [])
    live_click_window = (
        find_matching_click_window(click_window) if click_window is not None else None
    )
    live_input_hwnd = (
        int(live_click_window.hwnd) if live_click_window is not None else None
    )
    live_input_window_screen_xywh = (
        live_click_window.screen_rect() if live_click_window is not None else None
    )

    def _gate_roi(
        monitor_index: int,
        bgr: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        if click_window is None:
            return None
        img_h, img_w = bgr.shape[:2]
        return resolve_ocr_roi_local(
            click_window,
            image_w=img_w,
            image_h=img_h,
            monitor_offset=active_monitor_offset(monitor_index),
        )

    def _yolo_crop_for_window(
        monitor_index: int,
        bgr: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        """Live-window crop for Stage 1 YOLO, or None for full-frame."""
        if click_window is None:
            return None
        img_h, img_w = bgr.shape[:2]
        live_roi = resolve_live_ocr_roi_local(
            click_window,
            image_w=img_w,
            image_h=img_h,
            monitor_offset=active_monitor_offset(monitor_index),
        )
        if live_roi is None:
            _log_info(
                "move_mouse: no live click_window match "
                f"monitor={monitor_index}"
            )
            return None
        return live_roi

    def _live_hwnd_for_monitor(
        monitor_index: int,
        bgr: np.ndarray,
    ) -> tuple[int | None, tuple[int, int, int, int] | None]:
        """Pass live hwnd only when ``resolve_live_ocr_roi_local`` hits this monitor."""
        if live_input_hwnd is None or click_window is None:
            return None, None
        img_h, img_w = bgr.shape[:2]
        live_roi = resolve_live_ocr_roi_local(
            click_window,
            image_w=img_w,
            image_h=img_h,
            monitor_offset=active_monitor_offset(monitor_index),
        )
        if live_roi is None:
            return None, None
        return live_input_hwnd, live_input_window_screen_xywh

    # (monitor_index, bgr, gate_roi, yolo_crop_roi)
    planned: list[
        tuple[
            int,
            np.ndarray,
            tuple[int, int, int, int] | None,
            tuple[int, int, int, int] | None,
        ]
    ] = []
    if caption_roi_screen is not None:
        for monitor_index, bgr in captured:
            img_h, img_w = bgr.shape[:2]
            roi = _screen_xyxy_to_image_roi(
                caption_roi_screen,
                image_w=img_w,
                image_h=img_h,
                monitor_offset=active_monitor_offset(monitor_index),
            )
            # Caption cell is a live screen ROI — crop YOLO to it.
            planned.append((monitor_index, bgr, roi, roi))
        if any(gate is not None for _idx, _bgr, gate, _crop in planned):
            _log_info(
                "move_mouse caption button roi "
                f"screen={tuple(int(v) for v in caption_roi_screen)}"
            )
        else:
            _log_info(
                "move_mouse caption button roi missed every monitor; "
                "using window roi"
            )
            planned = []

    if not planned:
        if click_window is not None and not _any_monitor_has_live_click_window(
            captured, click_window
        ):
            _log_info("move_mouse fail: no live click_window match")
            raise ValueError(CLICK_WINDOW_MISS_MESSAGE)
        landmarks_outside = False
        for monitor_index, bgr in captured:
            if click_window is None or not landmark_centers:
                break
            img_h, img_w = bgr.shape[:2]
            recorded_roi = resolve_recorded_ocr_roi_local(
                click_window,
                image_w=img_w,
                image_h=img_h,
                monitor_offset=active_monitor_offset(monitor_index),
            )
            if _landmarks_clearly_outside_window(recorded_roi, landmark_centers):
                landmarks_outside = True
                _log_info(
                    "move_mouse yolo full-frame: landmarks outside recorded window "
                    f"monitor={monitor_index} roi={recorded_roi} "
                    f"landmarks={len(landmark_centers)}"
                )
                break
        if landmarks_outside:
            # Need desktop-wide detections; do not gate OCR to the window.
            planned = [
                (monitor_index, bgr, None, None) for monitor_index, bgr in captured
            ]
        else:
            for monitor_index, bgr in captured:
                gate = _gate_roi(monitor_index, bgr)
                crop = (
                    _yolo_crop_for_window(monitor_index, bgr)
                    if gate is not None
                    else None
                )
                # A recorded rect must not pull in a monitor the live window
                # does not intersect. Those monitors are dropped below.
                if click_window is not None and crop is None:
                    gate = None
                planned.append((monitor_index, bgr, gate, crop))

    if any(gate is not None for _idx, _bgr, gate, _crop in planned):
        skipped = [idx for idx, _bgr, gate, _crop in planned if gate is None]
        planned = [item for item in planned if item[2] is not None]
        if skipped:
            _log_info(
                "move_mouse skip monitors without ocr_roi "
                f"skipped={skipped} kept={[idx for idx, _bgr, _gate, _crop in planned]}"
            )

    pending_acc: list[_MouseOcrPending] = []

    if len(planned) == 1:
        monitor_index, bgr, gate, crop = planned[0]
        live_hwnd, live_win = _live_hwnd_for_monitor(monitor_index, bgr)
        dets = _detections_for_captured_monitor(
            monitor_index,
            bgr,
            yolo_conf_threshold=yolo_conf_threshold,
            timing_out=timing_out,
            ocr_class_ids=ocr_class_ids,
            refine_with_uia=refine_with_uia,
            ocr_roi=gate,
            ocr_roi_pad=ocr_roi_pad,
            yolo_crop_roi=crop,
            live_input_hwnd=live_hwnd,
            live_input_window_screen_xywh=live_win,
            defer_ocr=defer_ocr,
            pending_out=pending_acc if defer_ocr else None,
        )
        if timing_out is not None:
            layers = timing_out.get("debug_layers")
            if isinstance(layers, dict):
                timing_out["debug_layers_by_monitor"] = {int(monitor_index): layers}
            if defer_ocr and pending_acc:
                timing_out["_ocr_pending"] = pending_acc
        return dets

    all_detections: list[UiDetection] = []
    per_monitor_timings: list[tuple[int, dict[str, Any]]] = []

    def _detect_one(
        monitor_index: int,
        bgr: np.ndarray,
        ocr_roi: tuple[int, int, int, int] | None,
        yolo_crop_roi: tuple[int, int, int, int] | None,
    ) -> tuple[int, list[UiDetection], dict[str, Any], list[_MouseOcrPending]]:
        local_timing: dict[str, Any] = {}
        local_pending: list[_MouseOcrPending] = []
        live_hwnd, live_win = _live_hwnd_for_monitor(monitor_index, bgr)
        dets = _detections_for_captured_monitor(
            monitor_index,
            bgr,
            yolo_conf_threshold=yolo_conf_threshold,
            timing_out=local_timing,
            ocr_class_ids=ocr_class_ids,
            refine_with_uia=refine_with_uia,
            ocr_roi=ocr_roi,
            ocr_roi_pad=ocr_roi_pad,
            yolo_crop_roi=yolo_crop_roi,
            live_input_hwnd=live_hwnd,
            live_input_window_screen_xywh=live_win,
            defer_ocr=defer_ocr,
            pending_out=local_pending if defer_ocr else None,
        )
        return int(monitor_index), dets, local_timing, local_pending

    with ThreadPoolExecutor(max_workers=len(planned)) as pool:
        futures = [
            pool.submit(_detect_one, monitor_index, bgr, gate, crop)
            for monitor_index, bgr, gate, crop in planned
        ]
        for future in futures:
            monitor_index, dets, local_timing, local_pending = future.result()
            all_detections.extend(dets)
            per_monitor_timings.append((monitor_index, local_timing))
            pending_acc.extend(local_pending)

    if timing_out is not None:
        timing_out.clear()
        timing_out.update(
            {
                "yolo_s": 0.0,
                "line_s": 0.0,
                "uia_input_s": 0.0,
                "uia_scrollbar_s": 0.0,
                "uia_text_s": 0.0,
                "ocr_s": 0.0,
                "icon_ocr_retry_s": 0.0,
                "total_s": 0.0,
            }
        )
        layers_by_monitor: dict[int, Any] = {}
        any_pending = False
        for monitor_index, local_timing in per_monitor_timings:
            _add_vision_phase_timing(timing_out, local_timing)
            if local_timing.get("uia_first_pending"):
                any_pending = True
            layers = local_timing.get("debug_layers")
            if isinstance(layers, dict):
                layers_by_monitor[int(monitor_index)] = layers
        if layers_by_monitor:
            timing_out["debug_layers_by_monitor"] = layers_by_monitor
            # Keep a representative copy for callers that only read debug_layers.
            timing_out["debug_layers"] = next(iter(layers_by_monitor.values()))
        if defer_ocr and pending_acc:
            timing_out["_ocr_pending"] = pending_acc
            timing_out["uia_first_pending"] = True
        elif any_pending:
            timing_out["uia_first_pending"] = True
    return all_detections


def _normalize_nearby_labels(labels: list[str] | None) -> list[str]:
    """Strip, drop empties, and dedupe nearby labels while preserving order."""
    return nearby_hints_to_labels(normalize_nearby_hints(labels))


def _merge_nearby_labels(*sources: list[str] | None) -> list[str]:
    """Merge nearby label lists; earlier sources win on duplicates."""
    return nearby_hints_to_phrases(merge_nearby_hints(*sources))


def _hint_covered_by_neighbors(
    anchor: UiDetection,
    neighbors: list[UiDetection],
    hint: NearbyHint,
    *,
    threshold: float,
    require_side: bool,
) -> bool:
    """True when some neighbor matches ``hint`` (label, optional side)."""
    for neigh in neighbors:
        if _detection_similarity_to_query(neigh, hint.label) < threshold:
            continue
        if require_side and hint.side is not None:
            if not anchor_satisfies_side(
                anchor.bbox,
                neigh.cx,
                neigh.cy,
                hint.side,
                landmark_bbox=neigh.bbox,
            ):
                continue
        return True
    return False


# Overlay landmark matching is stricter than mouse prefilter so 「100%」does not
# claim a bare 「10」detection (SequenceMatcher ≈ 0.67 at the 0.5 prefilter bar).
_LIVE_OVERLAY_LABEL_SIMILARITY = 0.9


def _overlay_label_similarity(det: UiDetection, query: str) -> float:
    """Similarity for HTML landmark binding; short labels require an exact fold."""
    query = (query or "").strip()
    if not query:
        return 0.0
    sim = _detection_similarity_to_query(det, query)
    if sim <= 0.0:
        return 0.0
    normalized = _normalize_similarity_label(query)
    # "10" must not bind via partial ratio to "100%" / "10o%" noise.
    if len(normalized) <= 3 and sim < 1.0:
        return 0.0
    return sim


def _live_html_overlay_fields(
    chosen: UiDetection,
    detections: list[UiDetection],
    nearby_matches: list[UiDetection],
    nearby_hints: list[NearbyHint] | None,
    *,
    threshold: float = _LIVE_OVERLAY_LABEL_SIMILARITY,
) -> dict[str, Any]:
    """Peer / landmark geometry for session HTML (screen xywh / LTRB).

    Landmarks include every nearby hint that matches a live detection by label.
    Directed ``side`` is stored for the arrow overlay; assignment prefers a
    side-compatible detection but still keeps the hint when only off-side hits
    exist. Detections are claimed exclusively so 「100%」cannot steal 「10」.
    """
    peer_bboxes: list[dict[str, int]] = []
    if chosen.class_name in ("input", "scrollbar"):
        for det in detections:
            if det.class_name != chosen.class_name:
                continue
            if det.bbox == chosen.bbox:
                continue
            x, y, w, h = det.bbox
            peer_bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    # Prefer filter nearby hits, then fall back to the full detection list so a
    # label OCR'd on screen still overlays even when side prefilter dropped it.
    landmark_pool: list[UiDetection] = []
    seen_ids: set[int] = set()
    for det in [*nearby_matches, *detections]:
        det_id = id(det)
        if det_id in seen_ids:
            continue
        seen_ids.add(det_id)
        landmark_pool.append(det)

    # (hint_index, det, sim, side_miss, short_x, short_y, dist_sq)
    # short_x/y prefer leftmost exact short labels (row-header 「10」over mid-grid OCR).
    candidates: list[
        tuple[int, UiDetection, float, int, int, int, float]
    ] = []
    hints = list(nearby_hints or [])
    for hi, hint in enumerate(hints):
        label = (hint.label or "").strip()
        if not label:
            continue
        normalized = _normalize_similarity_label(label)
        short_exact = len(normalized) <= 3
        for det in landmark_pool:
            sim = _overlay_label_similarity(det, label)
            if sim < threshold:
                continue
            side_miss = 0
            if hint.side is not None and not anchor_satisfies_side(
                chosen.bbox,
                det.cx,
                det.cy,
                hint.side,
                landmark_bbox=det.bbox,
            ):
                side_miss = 1
            dist = float((det.cx - chosen.cx) ** 2 + (det.cy - chosen.cy) ** 2)
            short_x = int(det.cx) if short_exact and sim >= 1.0 else 0
            short_y = int(det.cy) if short_exact and sim >= 1.0 else 0
            candidates.append((hi, det, sim, side_miss, short_x, short_y, dist))

    # Best label match; side-compatible; leftmost short exact; then closer.
    candidates.sort(key=lambda row: (-row[2], row[3], row[4], row[5], row[6], row[0]))
    claimed_dets: set[int] = set()
    chosen_for_hint: dict[int, UiDetection] = {}
    for hi, det, _sim, _side_miss, _sx, _sy, _dist in candidates:
        if hi in chosen_for_hint:
            continue
        det_id = id(det)
        if det_id in claimed_dets:
            continue
        claimed_dets.add(det_id)
        chosen_for_hint[hi] = det

    landmarks: list[dict[str, Any]] = []
    for hi, hint in enumerate(hints):
        label = (hint.label or "").strip()
        det = chosen_for_hint.get(hi)
        if not label or det is None:
            continue
        x, y, w, h = det.bbox
        landmarks.append(
            {
                "label": label,
                "side": hint.side.value if hint.side is not None else None,
                "rect": [int(x), int(y), int(x + w), int(y + h)],
            }
        )
    return {"peer_bboxes": peer_bboxes, "landmarks": landmarks}


def _prefilter_anchors_by_nearby(
    anchors: list[UiDetection],
    nearby_matches: list[UiDetection],
    nearby_labels: list[str] | list[NearbyHint] | None,
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """Narrow anchors using exclusive nearby-landmark assignment.

    Keep only the anchors with **maximal** hint coverage (label and optional
    side); full coverage of every hint is just the best possible case. Ranking by
    coverage matters when a hint is uncoverable — e.g. a landmark the OCR
    mislabelled, leaving a same-named distractor elsewhere on screen: without
    ranking, an anchor matching 1 of 3 hints survives alongside the one matching
    2 of 3.

    Directed sides are checked against **all** ``nearby_matches`` for each anchor
    (so exclusive distance assignment cannot hide the correct side); distance
    never disqualifies a landmark. Undirected labels still use exclusive
    assignment. If directed sides wipe every anchor, retry with label-only
    coverage. If none match, return ``anchors`` unchanged.
    """
    hints = normalize_nearby_hints(nearby_labels)
    if not anchors or not nearby_matches or not hints:
        return anchors

    assigned = _assign_exclusive_neighbors_to_anchors(anchors, nearby_matches)

    def _select(require_side: bool) -> list[UiDetection]:
        coverage: list[set[int]] = []
        for i, neighbors in enumerate(assigned):
            covered: set[int] = set()
            for li, hint in enumerate(hints):
                pool = (
                    nearby_matches
                    if (require_side and hint.side is not None)
                    else neighbors
                )
                if _hint_covered_by_neighbors(
                    anchors[i],
                    pool,
                    hint,
                    threshold=threshold,
                    require_side=require_side,
                ):
                    covered.add(li)
            coverage.append(covered)

        best = max((len(covered) for covered in coverage), default=0)
        if not best:
            return []
        return [
            anchors[i] for i, covered in enumerate(coverage) if len(covered) == best
        ]

    selected = _select(require_side=True)
    if selected:
        return selected

    if any(hint.side is not None for hint in hints):
        selected = _select(require_side=False)
        if selected:
            return selected

    return anchors


# Gaps within this many pixels of the minimum count as the same closest tier.
_LANDMARK_CLOSEST_GAP_TIE_PX = 1.0


def _box_edge_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Euclidean gap between the closest edges of two ``(x, y, w, h)`` boxes.

    Overlapping or touching boxes have gap 0.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    if ax2 < bx:
        dx = float(bx - ax2)
    elif bx2 < ax:
        dx = float(ax - bx2)
    else:
        dx = 0.0
    if ay2 < by:
        dy = float(by - ay2)
    elif by2 < ay:
        dy = float(ay - by2)
    else:
        dy = 0.0
    return math.hypot(dx, dy)


def _detections_same_place(a: UiDetection, b: UiDetection) -> bool:
    """True when ``a`` and ``b`` are the same on-screen control.

    Heavy overlap counts, as does either center lying inside the other box.
    """
    if iou_xywh(a.bbox, b.bbox) > DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD:
        return True
    if point_in_rect_xywh(a.cx, a.cy, b.bbox):
        return True
    return point_in_rect_xywh(b.cx, b.cy, a.bbox)


# Class words that exist only as a UIA control suffix after 「…」.
_UIA_ONLY_ANCHOR_SUFFIXES = (
    "清單項目",
    "功能表項目",
    "樹狀項目",
    "資料項目",
    "超連結",
)
# Named UIA controls are on-object labels. 按鈕 is included so 「X」按鈕 is
# text-like for fallback policy, while the older scan still finds bare 按鈕.
_UIA_LABELED_ANCHOR_SUFFIXES = frozenset(
    _UIA_ONLY_ANCHOR_SUFFIXES + ("按鈕",)
)


def _anchor_type_suffix(anchor: str) -> str | None:
    """Return the class suffix in ``anchor`` (圖示, 文字, …), if any.

    UIA-only words are taken from the text after the closing quote so a name
    that contains 清單項目 does not hide a trailing 文字. The older scan still
    prefers 輸入欄 inside 「…」文字所在的輸入欄.
    """
    text = anchor or ""
    close = text.find("」")
    if close >= 0:
        tail = text[close + 1 :]
        for suffix in _UIA_ONLY_ANCHOR_SUFFIXES:
            if tail.startswith(suffix):
                return suffix
    for suffix in ("圖示", "輸入欄", "滾動條", "按鈕", "文字", "未知", "元素"):
        if suffix in text:
            return suffix
    return None


def _anchor_is_text(anchor: str) -> bool:
    """True when the anchor is an on-object named label (``…文字`` or a UIA type)."""
    suffix = _anchor_type_suffix(anchor)
    return suffix == "文字" or suffix in _UIA_LABELED_ANCHOR_SUFFIXES


def _is_relative_offset_position(instruction: str) -> bool:
    """True for empty-space clicks encoded with relative pixel offsets."""
    text = instruction or ""
    return "個像素" in text and "的位置" in text


def _matches_landmark_fallback_type(det: UiDetection, anchor: str) -> bool:
    """True when ``det`` is the kind of control ``anchor`` names.

    ``文字`` / ``圖示`` / ``元素`` accept either text or element boxes so a
    same-place OCR/icon swap (e.g. 「此」文字 vs 網格檢視) still locates. Blank
    unknown icon peers within :data:`_ICON_PEER_FALLBACK_MAX_SIDE_PX` stay
    eligible for icon-style anchors. ``輸入欄`` / ``滾動條`` stay on that class.
    An anchor with no suffix accepts any box.
    """
    suffix = _anchor_type_suffix(anchor)
    if suffix is None:
        return True
    if suffix in ("文字", "圖示", "元素") or suffix in _UIA_LABELED_ANCHOR_SUFFIXES:
        if (
            det.class_name in ("text", "element")
            or det.class_id in (YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT)
        ):
            return True
        if suffix == "按鈕" and det.class_name == "button":
            return True
        if suffix in ("圖示", "元素"):
            if not _is_blank_unknown_icon_candidate(det):
                return False
            return max(det.bbox[2], det.bbox[3]) <= _ICON_PEER_FALLBACK_MAX_SIDE_PX
        return False
    expected = {
        "輸入欄": "input",
        "滾動條": "scrollbar",
        "未知": "unknown",
    }[suffix]
    if det.class_name == expected:
        return True
    class_ids = {
        "input": YOLO_CLASS_INPUT,
        "scrollbar": YOLO_CLASS_SCROLLBAR,
        "unknown": PICKER_CLASS_UNKNOWN,
    }
    expected_id = class_ids.get(expected)
    return expected_id is not None and det.class_id == expected_id


def _directed_landmark_gap(
    candidate: UiDetection,
    landmark: UiDetection,
    hint: NearbyHint,
) -> float | None:
    """Gap when ``candidate`` sits on ``hint.side`` of ``landmark``, else None."""
    if hint.side is None:
        return None
    if not anchor_satisfies_side(
        candidate.bbox,
        landmark.cx,
        landmark.cy,
        hint.side,
        landmark_bbox=landmark.bbox,
    ):
        return None
    return _box_edge_gap(candidate.bbox, landmark.bbox)


def _hint_satisfied_by_nearby_landmark(
    candidate: UiDetection,
    landmarks: list[UiDetection],
    hint: NearbyHint,
    peers: list[UiDetection],
    *,
    threshold: float,
) -> bool:
    """True when ``candidate`` is a closest peer on ``hint``'s side.

    For each landmark whose label matches the hint, peers on the required side
    are ranked by box-edge gap. ``candidate`` matches when it ties the nearest
    peer of any such landmark.
    """
    if hint.side is None:
        return False
    matching = [
        neigh
        for neigh in landmarks
        if id(neigh) != id(candidate)
        and _detection_similarity_to_query(neigh, hint.label) >= threshold
    ]
    if not matching:
        return False

    for neigh in matching:
        candidate_gap: float | None = None
        nearest: float | None = None
        for peer in peers:
            gap = _directed_landmark_gap(peer, neigh, hint)
            if gap is None:
                continue
            if nearest is None or gap < nearest:
                nearest = gap
            if id(peer) == id(candidate):
                candidate_gap = gap
        if (
            candidate_gap is not None
            and nearest is not None
            and candidate_gap <= nearest + _LANDMARK_CLOSEST_GAP_TIE_PX
        ):
            return True
    return False


def _group_same_place(eligible: list[UiDetection]) -> list[UiDetection]:
    """Collapse co-located boxes to one representative per place."""
    count = len(eligible)
    if count <= 1:
        return list(eligible)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(count):
        for j in range(i + 1, count):
            if _detections_same_place(eligible[i], eligible[j]):
                union(i, j)

    groups: dict[int, list[UiDetection]] = {}
    for index, det in enumerate(eligible):
        groups.setdefault(find(index), []).append(det)
    return [
        max(group, key=_detection_preference_score) for group in groups.values()
    ]


def _landmark_fallback_places(
    detections: list[UiDetection],
    nearby_hints: list[str] | list[NearbyHint] | None,
    *,
    anchor: str,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """Places that satisfy every available directed nearby hint.

    Landmarks are detections whose label matches a directed hint. Those boxes,
    and any box that heavily overlaps one, are not selectable. Undirected hints
    are ignored. Directed hints with no matching landmark on screen are skipped
    so a partial landmark set can still uniquely locate the target. With no
    available directed hint, no place qualifies. A hint matches the closest
    selectable box on that side of the landmark, including a gap tie.
    """
    hints = normalize_nearby_hints(nearby_hints)
    directed = [hint for hint in hints if hint.side is not None]
    if not detections or not directed:
        return []

    landmarks: list[UiDetection] = []
    for det in detections:
        if any(
            _detection_similarity_to_query(det, hint.label) >= threshold
            for hint in directed
        ):
            landmarks.append(det)
    if not landmarks:
        return []

    available = [
        hint
        for hint in directed
        if any(
            _detection_similarity_to_query(neigh, hint.label) >= threshold
            for neigh in landmarks
        )
    ]
    if not available:
        return []

    landmark_ids = {id(det) for det in landmarks}
    selectable: list[UiDetection] = []
    for det in detections:
        if id(det) in landmark_ids:
            continue
        if any(
            iou_xywh(det.bbox, landmark.bbox) > DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD
            for landmark in landmarks
        ):
            continue
        if not _matches_landmark_fallback_type(det, anchor):
            continue
        selectable.append(det)
    eligible = [
        det
        for det in selectable
        if all(
            _hint_satisfied_by_nearby_landmark(
                det,
                landmarks,
                hint,
                selectable,
                threshold=threshold,
            )
            for hint in available
        )
    ]
    return _group_same_place(eligible)


def _unique_target_from_landmarks(
    detections: list[UiDetection],
    nearby_hints: list[str] | list[NearbyHint] | None,
    *,
    anchor: str,
) -> UiDetection | None:
    """Return the only place available directed landmarks describe, or None."""
    places = _landmark_fallback_places(detections, nearby_hints, anchor=anchor)
    if len(places) != 1:
        return None
    return places[0]


def _match_geometry_landmark_centers(
    detections: list[UiDetection],
    recorded: dict[str, Any],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> dict[str, tuple[float, float]]:
    """Greedy label→center map that preserves recorded inter-landmark vectors."""
    # Lazy import: ``src.recorder`` package init pulls tools → this module.
    from src.recorder.landmark_geometry import normalize_vector, vectors_agree

    labels = [
        str(item.get("label") or "").strip()
        for item in recorded.get("landmarks") or []
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    ]
    edge_map: dict[tuple[str, str], tuple[float, float]] = {}
    for item in recorded.get("landmark_to_landmark") or []:
        if not isinstance(item, dict):
            continue
        frm = str(item.get("from") or "").strip()
        to = str(item.get("to") or "").strip()
        if not frm or not to:
            continue
        try:
            edge_map[(frm, to)] = (float(item["dx"]), float(item["dy"]))
        except (KeyError, TypeError, ValueError):
            continue

    matched: dict[str, tuple[float, float]] = {}
    for label in labels:
        candidates = [
            det
            for det in detections
            if _detection_similarity_to_query(det, label) >= threshold
        ]
        if not candidates:
            continue
        if len(candidates) == 1 or not matched:
            best = max(candidates, key=_detection_preference_score)
            matched[label] = (float(best.cx), float(best.cy))
            continue

        try:
            recorded_scale = max(float(recorded.get("scale", 1.0)), 1.0)
        except (TypeError, ValueError):
            recorded_scale = 1.0

        def _score(det: UiDetection) -> tuple[int, float]:
            agree = 0
            err = 0.0
            center = (float(det.cx), float(det.cy))
            for other, other_xy in matched.items():
                recorded_vec = edge_map.get((label, other))
                if recorded_vec is None:
                    continue
                live_vec = normalize_vector(center, other_xy, recorded_scale)
                if vectors_agree(recorded_vec, live_vec):
                    agree += 1
                err += abs(live_vec[0] - recorded_vec[0]) + abs(
                    live_vec[1] - recorded_vec[1]
                )
            return (-agree, err)

        best = min(candidates, key=_score)
        matched[label] = (float(best.cx), float(best.cy))
    return matched


def _kept_nearby_hints_for_labels(
    nearby_hints: list[NearbyHint] | list[str] | None,
    kept_labels: set[str],
) -> list[NearbyHint]:
    """Restrict nearby hints to constellation-kept landmark labels."""
    hints = normalize_nearby_hints(nearby_hints)
    return [hint for hint in hints if hint.label in kept_labels]


def _resolve_with_landmark_geometry(
    detections: list[UiDetection],
    landmark_geometry: dict[str, Any] | None,
    *,
    anchor: str,
    nearby_hints: list[NearbyHint] | list[str] | None,
    synth_only: bool = False,
    candidate_pool: list[UiDetection] | None = None,
) -> tuple[UiDetection | None, tuple[int, int], str] | None:
    """Resolve a click via recorded constellation; ``None`` when geometry cannot help.

    Returns ``(detection_or_None, point, selection_method)``.

    When ``synth_only`` is True (empty-space / relative-offset clicks), skip
    detection picking and always return the synthesized point.

    When ``candidate_pool`` is set (e.g. similarity-matched inputs), pick only
    among that pool near the synthesized point; otherwise synthesize.
    """
    # Lazy import: ``src.recorder`` package init pulls tools → this module.
    from src.recorder.landmark_geometry import (
        filter_landmarks_by_constellation,
        parse_landmark_geometry,
        pick_detection_near_synthesized,
        replay_scale,
        synthesize_target_point,
        target_far_from_synthesized,
    )

    recorded = parse_landmark_geometry(landmark_geometry)
    if recorded is None or not detections:
        return None
    matched = _match_geometry_landmark_centers(detections, recorded)
    if len(matched) < 1:
        return None
    kept_labels = filter_landmarks_by_constellation(recorded, matched)
    kept_centers = {
        label: matched[label] for label in kept_labels if label in matched
    }
    if len(kept_centers) < 1:
        return None

    synth = synthesize_target_point(recorded, kept_centers)
    if synth is None:
        return None
    if synth_only:
        return None, synth, "landmark_geometry_synth"

    scale = replay_scale(recorded, kept_centers)

    if candidate_pool is not None:
        near = pick_detection_near_synthesized(candidate_pool, synth, scale)
        if near is not None:
            return near, (int(near.cx), int(near.cy)), "landmark_geometry_pick"
        return None, synth, "landmark_geometry_synth"

    kept_hints = _kept_nearby_hints_for_labels(nearby_hints, set(kept_centers))
    unique = _unique_target_from_landmarks(
        detections,
        kept_hints or nearby_hints,
        anchor=anchor,
    )
    if unique is not None and not target_far_from_synthesized(
        (float(unique.cx), float(unique.cy)),
        synth,
        scale,
    ):
        return unique, (int(unique.cx), int(unique.cy)), "landmark_geometry_pick"

    type_compatible = [
        det
        for det in detections
        if _matches_landmark_fallback_type(det, anchor)
    ]
    near = pick_detection_near_synthesized(type_compatible, synth, scale)
    if near is not None:
        return near, (int(near.cx), int(near.cy)), "landmark_geometry_pick"

    return None, synth, "landmark_geometry_synth"


def _landmark_geometry_synth_meta(
    *,
    point: tuple[int, int],
    offset_dx: int,
    offset_dy: int,
    track_percent: float | None,
    char_target: str | None,
    char_occurrence: int,
    instruction_text: str,
    nearby_phrases: list[str],
    image_paths: list[str],
    selection_method: str,
    timing: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Build ``(gx, gy, meta)`` for a constellation-synthesized click."""
    gx = int(point[0]) + int(offset_dx)
    gy = int(point[1]) + int(offset_dy)
    meta: dict[str, Any] = {
        "selected_index": -1,
        "class_name": "",
        "image_center": {"x": int(point[0]), "y": int(point[1])},
        "resolved_center": {"x": gx, "y": gy},
        "relative_offset": {"dx": offset_dx, "dy": offset_dy},
        "track_percent": track_percent,
        "char_target": char_target,
        "char_occurrence": char_occurrence,
        "anchor_instruction": instruction_text,
        "nearby_objects": nearby_phrases,
        "screenshot_path": image_paths[0] if image_paths else "",
        "screenshot_paths": image_paths,
        "target_kind": "landmark_geometry",
        "target_text": "",
        "target_icons": [],
        "target_bbox": None,
        "peer_bboxes": [],
        "landmarks": [],
        "selection_method": selection_method,
        "timing": timing,
    }
    return gx, gy, meta


def _landmark_hints_locate(
    detections: list[UiDetection],
    nearby_hints: list[str] | list[NearbyHint] | None,
    primary: UiDetection,
    *,
    anchor: str,
) -> bool:
    """True when the hints leave exactly one place and it is ``primary``.

    Uses playback's closest-on-side matching (see
    :func:`_landmark_fallback_places`).
    """
    places = _landmark_fallback_places(detections, nearby_hints, anchor=anchor)
    if len(places) != 1:
        return False
    chosen = places[0]
    return chosen is primary or _detections_same_place(chosen, primary)


def _hint_satisfied_by_landmark_side(
    candidate: UiDetection,
    landmarks: list[UiDetection],
    hint: NearbyHint,
    *,
    threshold: float,
) -> bool:
    """True when ``candidate`` sits on ``hint.side`` of any matching landmark.

    Unlike :func:`_hint_satisfied_by_nearby_landmark`, this ignores closest-gap:
    every box on the required side counts. Recording uses this so auto-picked
    landmarks are safe even when a sibling icon also lies on those sides.
    """
    if hint.side is None:
        return False
    matching = [
        neigh
        for neigh in landmarks
        if id(neigh) != id(candidate)
        and _detection_similarity_to_query(neigh, hint.label) >= threshold
    ]
    if not matching:
        return False
    return any(
        anchor_satisfies_side(
            candidate.bbox,
            neigh.cx,
            neigh.cy,
            hint.side,
            landmark_bbox=neigh.bbox,
        )
        for neigh in matching
    )


def _landmark_side_unique_places(
    detections: list[UiDetection],
    nearby_hints: list[str] | list[NearbyHint] | None,
    *,
    anchor: str,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """Places that sit on every directed hint side (no closest-gap filter).

    Same landmark / selectable construction as
    :func:`_landmark_fallback_places`, but a candidate matches a hint whenever
    it lies on that side of any label-matching landmark.
    """
    hints = normalize_nearby_hints(nearby_hints)
    directed = [hint for hint in hints if hint.side is not None]
    if not detections or not directed:
        return []

    landmarks: list[UiDetection] = []
    for det in detections:
        if any(
            _detection_similarity_to_query(det, hint.label) >= threshold
            for hint in directed
        ):
            landmarks.append(det)
    if not landmarks:
        return []

    landmark_ids = {id(det) for det in landmarks}
    selectable: list[UiDetection] = []
    for det in detections:
        if id(det) in landmark_ids:
            continue
        if any(
            iou_xywh(det.bbox, landmark.bbox) > DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD
            for landmark in landmarks
        ):
            continue
        if not _matches_landmark_fallback_type(det, anchor):
            continue
        selectable.append(det)
    eligible = [
        det
        for det in selectable
        if all(
            _hint_satisfied_by_landmark_side(
                det,
                landmarks,
                hint,
                threshold=threshold,
            )
            for hint in directed
        )
    ]
    return _group_same_place(eligible)


def _landmark_hints_locate_sides(
    detections: list[UiDetection],
    nearby_hints: list[str] | list[NearbyHint] | None,
    primary: UiDetection,
    *,
    anchor: str,
) -> bool:
    """True when only ``primary``'s place sits on every directed hint side.

    Recording auto-pick uses this stricter check so a sibling control on the
    same sides (e.g. 「側邊欄」 next to 「此」) blocks an unsafe landmark set.
    """
    places = _landmark_side_unique_places(detections, nearby_hints, anchor=anchor)
    if len(places) != 1:
        return False
    chosen = places[0]
    return chosen is primary or _detections_same_place(chosen, primary)


async def _maybe_disambiguate_similar_selection(
    *,
    chosen: UiDetection,
    initial_idx: int,
    selected_text: str | None,
    detections: list[UiDetection],
    image_paths: list[str],
    monitor_indices: list[int],
    overlay_dir: Path,
    overlay_stamp: str,
    anchor: str,
    nearby_phrases: list[str],
    nearby_matches: list[UiDetection],
) -> tuple[UiDetection, int, str | None, dict[str, Any]]:
    """
    When other YOLO detections share labels with ``chosen``, describe peers and re-pick.

    Returns ``(chosen, selected_index, selected_text, extra_meta)``. ``extra_meta`` is
    empty when disambiguation is skipped.
    """
    similar = _detections_similar_to(chosen, detections)
    if len(similar) <= 1:
        return chosen, initial_idx, selected_text, {}

    peer_centers = ", ".join(f"({d.cx},{d.cy})" for d in similar)
    _log_info(
        "similar_function_describe peers "
        f"count={len(similar)} centers=[{peer_centers}] "
        f"initial_center=({chosen.cx},{chosen.cy})"
    )
    annotated_paths = _write_indexed_bbox_overlay_images(
        similar,
        image_paths,
        monitor_indices,
        overlay_dir,
        stamp=overlay_stamp,
    )
    functions = await _describe_ui_candidate_functions(
        anchor,
        similar,
        annotated_paths,
    )
    pool_idx, new_text = await _select_center_with_functions(
        anchor,
        similar,
        functions,
        annotated_paths,
        neighbor_candidates=nearby_matches,
        nearby_labels=nearby_phrases,
    )
    new_chosen = similar[pool_idx]
    changed = (new_chosen.cx, new_chosen.cy) != (chosen.cx, chosen.cy)
    _log_info(
        "similar_function_describe re-pick "
        f"index={pool_idx} changed={changed} "
        f"center=[{new_chosen.cx},{new_chosen.cy}] "
        f"text={new_text!r}"
    )
    extra: dict[str, Any] = {
        "disambiguation": "similar_function_describe",
        "similar_count": len(similar),
        "function_descriptions": {
            str(i): functions[i] for i in range(len(functions))
        },
        "initial_selected_index": initial_idx,
        "initial_center": {"x": chosen.cx, "y": chosen.cy},
        "indexed_screenshot_paths": annotated_paths,
    }
    return new_chosen, pool_idx, new_text, extra


def _write_yolo_ocr_sidecars(
    captured: list[tuple[int, np.ndarray]],
    image_paths: list[str],
    detections: list[UiDetection],
    vision_timing: dict[str, Any],
) -> None:
    """Persist image-local YOLO/UIA debug layers next to each capture PNG."""
    layers_by_monitor = vision_timing.get("debug_layers_by_monitor")
    if not isinstance(layers_by_monitor, dict):
        layers_by_monitor = {}
        single = vision_timing.get("debug_layers")
        if isinstance(single, dict) and len(captured) == 1:
            layers_by_monitor = {int(captured[0][0]): single}
    for (monitor_index, bgr), image_path in zip(captured, image_paths, strict=False):
        layers = layers_by_monitor.get(int(monitor_index))
        if not isinstance(layers, dict):
            continue
        left, top = active_monitor_offset(monitor_index)
        img_h, img_w = bgr.shape[:2]
        local_dets = [
            _offset_detection(det, -left, -top)
            for det in detections
            if left <= int(det.cx) < left + img_w and top <= int(det.cy) < top + img_h
        ]
        sidecar = {
            "image_path": image_path,
            "monitor_index": int(monitor_index),
            "monitor_offset": [int(left), int(top)],
            "candidates": [_ui_detection_candidate_dict(det) for det in local_dets],
            "detection_count": len(local_dets),
            "debug_layers": layers,
            "source": "move_mouse_capture",
        }
        try:
            write_json(Path(image_path).with_suffix(".json"), sidecar)
        except Exception as exc:
            _log_info(
                f"move_mouse debug_layers write failed path={image_path} "
                f"{type(exc).__name__}: {exc}"
            )


def _finish_deferred_ocr_pendings(
    pendings: list[_MouseOcrPending],
    *,
    timing_base: dict[str, Any],
) -> tuple[list[UiDetection], dict[str, Any]]:
    """Complete deferred leftover OCR for each monitor and merge timing.

    Finish-pass timings replace the deferred pre-OCR vision phase values
    (they already include YOLO/UIA from the pending snapshot).
    """
    all_detections: list[UiDetection] = []
    merged_vision: dict[str, Any] = {
        "yolo_s": 0.0,
        "line_s": 0.0,
        "uia_input_s": 0.0,
        "uia_scrollbar_s": 0.0,
        "uia_text_s": 0.0,
        "ocr_s": 0.0,
        "icon_ocr_retry_s": 0.0,
        "total_s": 0.0,
    }
    layers_by_monitor: dict[int, Any] = {}
    for pending in pendings:
        local_timing: dict[str, Any] = {}
        local = _finish_mouse_ocr_pending(pending, timing_out=local_timing)
        left, top = pending.coord_offset
        all_detections.extend(_offset_detection(d, left, top) for d in local)
        _add_vision_phase_timing(merged_vision, local_timing)
        layers = local_timing.get("debug_layers")
        cache_key = pending.ocr_cache_key
        if isinstance(layers, dict) and cache_key is not None:
            layers_by_monitor[int(cache_key)] = layers
    if layers_by_monitor:
        merged_vision["debug_layers_by_monitor"] = layers_by_monitor
        merged_vision["debug_layers"] = next(iter(layers_by_monitor.values()))
    # Preserve ROI / UIA assist counts from the deferred pass when finish did
    # not re-emit them (finish always does via _timing_payload_with_roi).
    for key in (
        "ocr_roi",
        "ocr_rois",
        "ocr_roi_pad",
        "uia_text_orphans",
        "uia_text_ocr_skipped",
    ):
        if key not in merged_vision and key in timing_base:
            merged_vision[key] = timing_base[key]
    capture_s = float(timing_base.get("capture_s", 0.0))
    timing: dict[str, Any] = {
        "capture_s": capture_s,
        "yolo_s": float(merged_vision.get("yolo_s", 0.0)),
        "line_s": float(merged_vision.get("line_s", 0.0)),
        "uia_input_s": float(merged_vision.get("uia_input_s", 0.0)),
        "uia_scrollbar_s": float(merged_vision.get("uia_scrollbar_s", 0.0)),
        "uia_text_s": float(merged_vision.get("uia_text_s", 0.0)),
        "ocr_s": float(merged_vision.get("ocr_s", 0.0)),
        "icon_ocr_retry_s": float(merged_vision.get("icon_ocr_retry_s", 0.0)),
        "total_s": capture_s + float(merged_vision.get("total_s", 0.0)),
        "uia_first_hit": False,
    }
    timing.update(_vision_roi_fields(merged_vision))
    timing.update(_ocr_reuse_count_fields(merged_vision))
    if "debug_layers_by_monitor" in merged_vision:
        timing["debug_layers_by_monitor"] = merged_vision["debug_layers_by_monitor"]
    if "debug_layers" in merged_vision:
        timing["debug_layers"] = merged_vision["debug_layers"]
    return _sort_detections_reading_order(all_detections), timing


def _phrase_match_pool(
    detections: list[UiDetection],
    phrase: str,
) -> list[UiDetection]:
    """Keep detections of ``phrase``'s UIA type. Empty typed set falls back to all."""
    from src.recorder.uia_input_boxes import control_token_for_suffix

    token = control_token_for_suffix(_anchor_type_suffix(phrase))
    if not token:
        return detections
    typed = [det for det in detections if (det.uia_control_type or "") == token]
    if typed:
        return typed
    _log_info(
        "control_type_fallback "
        f"phrase={phrase!r} token={token} detections={len(detections)}"
    )
    return detections


def _matches_for_phrase(
    detections: list[UiDetection],
    phrase: str,
) -> list[UiDetection]:
    """Name matches for one phrase, inside its control-type pool when that pool exists."""
    pool = _phrase_match_pool(detections, phrase)
    indices = _anchor_indices_by_top_similarity(pool, phrase)
    if not indices:
        return []
    expanded = _expand_keep_indices_with_similar(pool, indices)
    return [pool[index] for index in expanded]


def _try_uia_first_unique_select(
    detections: list[UiDetection],
    *,
    anchor: str,
    nearby_hints: list[NearbyHint],
    nearby_labels: list[str],
    nearby_phrases: list[str],
    filter_by_control_type: bool = False,
) -> tuple[UiDetection | None, str]:
    """Soft UIA-first commit: unique after filter + nearby prefilter, else miss reason."""
    if not detections:
        return None, "no_detections"

    if filter_by_control_type:
        anchor_matches = _matches_for_phrase(detections, anchor)
        nearby_matches: list[UiDetection] = []
        seen = {id(det) for det in anchor_matches}
        for phrase in nearby_labels:
            for det in _matches_for_phrase(detections, phrase):
                if id(det) in seen:
                    continue
                seen.add(id(det))
                nearby_matches.append(det)
    else:
        anchor_matches, nearby_matches = _filter_mouse_candidates(
            detections, anchor, nearby_labels
        )
    _log_info(
        "move_mouse uia_first after_filter "
        f"anchor={len(anchor_matches)} nearby={len(nearby_matches)} "
        f"anchor={anchor!r} nearby_labels={nearby_phrases!r}"
    )
    if not anchor_matches:
        return None, "no_anchor"

    before_nearby = len(anchor_matches)
    # Blank unknown icon peers require OCR; skip admit on the partial set.
    anchor_matches = _prefilter_anchors_by_nearby(
        anchor_matches, nearby_matches, nearby_hints
    )
    if len(anchor_matches) != before_nearby:
        _log_info(
            "move_mouse uia_first nearby prefilter "
            f"anchors {before_nearby} -> {len(anchor_matches)} "
            f"nearby_labels={nearby_phrases!r}"
        )
    if len(anchor_matches) != 1:
        return None, "not_unique"

    chosen = anchor_matches[0]
    _log_info(
        "move_mouse uia_first hit "
        f"center=[{chosen.cx},{chosen.cy}] text={chosen.text!r} "
        f"source={chosen.source!r}"
    )
    return chosen, "hit"


def _detections_from_uia_prefetched(
    uia_prefetched: tuple[
        list[tuple[int, int, int, int]] | None,
        list[tuple[int, int, int, int]] | None,
        list[tuple[tuple[int, int, int, int], str]],
    ]
    | None,
) -> list[UiDetection]:
    """Text, input, and scrollbar candidates from a snapshot. URL names are omitted."""
    if uia_prefetched is None:
        return []
    from src.recorder.uia_input_boxes import (
        is_usable_uia_landmark_label,
        unpack_uia_labeled,
    )

    inputs, scrollbars, labeled = uia_prefetched
    detections: list[UiDetection] = []
    for item in labeled or []:
        box, label, control_type = unpack_uia_labeled(item)
        text = str(label or "").strip()
        if not is_usable_uia_landmark_label(text):
            continue
        detections.append(
            _detection_from_bbox(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                YOLO_CLASS_TEXT,
                text=text,
                source="uia",
                uia_control_type=control_type or None,
            )
        )
    for box in inputs or []:
        detections.append(
            _detection_from_bbox(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                YOLO_CLASS_INPUT,
                source="uia",
            )
        )
    for box in scrollbars or []:
        detections.append(
            _detection_from_bbox(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                YOLO_CLASS_SCROLLBAR,
                source="uia",
            )
        )
    return detections


def _labeled_text_boxes(
    detections: list[UiDetection],
    *,
    source: str,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Screen xywh and text for OCR or UIA text detections."""
    boxes: list[tuple[tuple[int, int, int, int], str]] = []
    for det in detections:
        text = str(det.text or "").strip()
        if not text or det.class_name != "text":
            continue
        det_source = str(det.source or "")
        if source == "uia" and det_source != "uia":
            continue
        if source == "ocr" and det_source == "uia":
            continue
        boxes.append((det.bbox, text))
    return boxes


def _uia_path_text_count(detections: list[UiDetection]) -> int:
    return sum(
        1
        for det in detections
        if det.class_id == YOLO_CLASS_TEXT and str(det.source or "") == "uia"
    )


def _live_uia_snapshot_for_click_window(
    click_window: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any]] | None:
    """FindAll for a repair walk. Does not grab the desktop, so it can overlap OCR.

    The recorded hwnd is often stale on a later replay. ``find_matching_click_window``
    keeps it when it still belongs to that window, and otherwise matches process and title.
    """
    if not isinstance(click_window, dict):
        return None
    from src.recorder.uia_input_boxes import snapshot_uia_screen
    from src.recorder.window_snapshot import find_matching_click_window

    live = find_matching_click_window(click_window)
    if live is None or not int(live.hwnd or 0):
        return None
    snapshot = snapshot_uia_screen(int(live.hwnd))
    if not isinstance(snapshot, dict):
        return None
    return live, snapshot


def _detections_from_live_uia_snapshot(
    captured: list[tuple[int, np.ndarray]],
    live: Any,
    snapshot: dict[str, Any],
) -> list[UiDetection]:
    """Map a finished FindAll onto the captured monitor. Screen-space boxes."""
    if not captured:
        return []
    from src.recorder.uia_input_boxes import uia_snapshot_image_local

    left, top = int(live.left), int(live.top)
    chosen: tuple[int, np.ndarray, tuple[int, int]] | None = None
    for monitor_index, bgr in captured:
        offset = active_monitor_offset(monitor_index)
        height, width = bgr.shape[:2]
        if offset[0] <= left < offset[0] + width and offset[1] <= top < offset[1] + height:
            chosen = (monitor_index, bgr, offset)
            break
    if chosen is None:
        monitor_index, bgr = captured[0]
        chosen = (monitor_index, bgr, active_monitor_offset(monitor_index))
    _monitor_index, bgr, offset = chosen
    height, width = bgr.shape[:2]
    prefetched = uia_snapshot_image_local(
        snapshot,
        monitor_offset=offset,
        image_w=width,
        image_h=height,
        ocr_roi=None,
        window_screen_xywh=None,
    )
    local = _detections_from_uia_prefetched(prefetched)
    return [_offset_detection(det, int(offset[0]), int(offset[1])) for det in local]


def _uia_only_detections_for_click_window(
    captured: list[tuple[int, np.ndarray]],
    click_window: dict[str, Any] | None,
) -> list[UiDetection]:
    """FindAll under the live click hwnd and return screen-space UIA candidates."""
    if not captured:
        return []
    found = _live_uia_snapshot_for_click_window(click_window)
    if found is None:
        return []
    live, snapshot = found
    return _detections_from_live_uia_snapshot(captured, live, snapshot)


def _grab_selected_monitors() -> tuple[
    list[int],
    list[str],
    list[tuple[int, np.ndarray]],
    float,
]:
    """Grab selected monitors once. Desktop grabbers stay on this thread."""
    paths = _run_manager().require_paths()
    monitor_indices = selected_eye_monitor_indices()
    stamp = ts_name()
    image_paths: list[str] = []
    captured: list[tuple[int, np.ndarray]] = []

    _log_info(f"move_mouse resolve monitors={monitor_indices}")
    capture_started = time.perf_counter()
    for monitor_index in monitor_indices:
        name = f"{stamp}_mon{monitor_index}.png"
        out = paths.yolo_ocr_dir / name
        try:
            _resolved_idx, bgr = grab_monitor_bgr(monitor_index)
        except Exception as exc:
            _log_info(
                f"move_mouse grab failed monitor={monitor_index} "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        image_path = str(out.resolve())
        if not imwrite_bgr(out, bgr):
            _log_info(f"move_mouse could not write captured image path={image_path}")
        image_paths.append(image_path)
        captured.append((monitor_index, bgr))
    return monitor_indices, image_paths, captured, time.perf_counter() - capture_started


def _finish_captured_mouse_candidates(
    monitor_indices: list[int],
    image_paths: list[str],
    captured: list[tuple[int, np.ndarray]],
    capture_s: float,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    *,
    ocr_class_ids: frozenset[int] | set[int] | None = None,
    refine_with_uia: bool = True,
    click_window: dict[str, Any] | None = None,
    caption_roi_screen: tuple[int, int, int, int] | None = None,
    landmark_geometry: dict[str, Any] | None = None,
    defer_ocr: bool = False,
) -> tuple[
    list[int],
    list[str],
    list[tuple[int, np.ndarray]],
    list[UiDetection],
    dict[str, Any],
]:
    """YOLO+OCR on frames already grabbed. Does not read UIA into the candidate list."""
    from src.recorder.landmark_geometry import landmark_geometry_centers

    vision_timing: dict[str, Any] = {}
    all_detections = _collect_monitor_detections(
        captured,
        yolo_conf_threshold=yolo_conf_threshold,
        timing_out=vision_timing,
        ocr_class_ids=ocr_class_ids,
        refine_with_uia=refine_with_uia,
        click_window=click_window,
        caption_roi_screen=caption_roi_screen,
        landmark_centers_image=landmark_geometry_centers(landmark_geometry),
        defer_ocr=defer_ocr,
    )
    detections = _sort_detections_reading_order(all_detections)
    _log_info(
        f"move_mouse yolo_candidates={len(detections)}"
        + (" (ocr deferred)" if vision_timing.get("uia_first_pending") else "")
    )
    if detections:
        _log_info(
            "move_mouse all_ocr_candidates:\n"
            + _format_ui_candidates_text(detections, include_geometry=True)
        )
    _write_yolo_ocr_sidecars(captured, image_paths, detections, vision_timing)
    timing: dict[str, Any] = {
        "capture_s": capture_s,
        "yolo_s": float(vision_timing.get("yolo_s", 0.0)),
        "line_s": float(vision_timing.get("line_s", 0.0)),
        "uia_input_s": float(vision_timing.get("uia_input_s", 0.0)),
        "uia_scrollbar_s": float(vision_timing.get("uia_scrollbar_s", 0.0)),
        "uia_text_s": float(vision_timing.get("uia_text_s", 0.0)),
        "ocr_s": float(vision_timing.get("ocr_s", 0.0)),
        "icon_ocr_retry_s": float(vision_timing.get("icon_ocr_retry_s", 0.0)),
        "total_s": capture_s + float(vision_timing.get("total_s", 0.0)),
    }
    timing.update(_vision_roi_fields(vision_timing))
    timing.update(_ocr_reuse_count_fields(vision_timing))
    pending = vision_timing.get("_ocr_pending")
    if isinstance(pending, list) and pending:
        timing["_ocr_pending"] = pending
        timing["uia_first_pending"] = True
    if "debug_layers_by_monitor" in vision_timing:
        timing["debug_layers_by_monitor"] = vision_timing["debug_layers_by_monitor"]
    if "debug_layers" in vision_timing:
        timing["debug_layers"] = vision_timing["debug_layers"]
    return monitor_indices, image_paths, captured, detections, timing


def _capture_and_detect_mouse_candidates(
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    *,
    ocr_class_ids: frozenset[int] | set[int] | None = None,
    refine_with_uia: bool = True,
    click_window: dict[str, Any] | None = None,
    caption_roi_screen: tuple[int, int, int, int] | None = None,
    landmark_geometry: dict[str, Any] | None = None,
    defer_ocr: bool = False,
) -> tuple[
    list[int],
    list[str],
    list[tuple[int, np.ndarray]],
    list[UiDetection],
    dict[str, Any],
]:
    """
    Capture selected monitor(s) and build YOLO+OCR candidates.

    Sync helper so it can overlap with instruction parsing via ``asyncio.to_thread``.
    Capture stays sequential — desktop grabbers are often not concurrent-safe.

    Screenshots are grabbed to BGR in memory for inference, then written once for
    meta / multimodal paths (no PNG decode round-trip).

    Returns ``(monitor_indices, image_paths, captured, detections, timing)`` where
    ``timing`` includes ``capture_s`` plus summed vision phases.

    When ``defer_ocr=True``, leftover OCR is skipped when class candidates
    exist; ``timing["_ocr_pending"]`` holds finish state for a later pass.
    """
    monitor_indices, image_paths, captured, capture_s = _grab_selected_monitors()
    return _finish_captured_mouse_candidates(
        monitor_indices,
        image_paths,
        captured,
        capture_s,
        yolo_conf_threshold,
        ocr_class_ids=ocr_class_ids,
        refine_with_uia=refine_with_uia,
        click_window=click_window,
        caption_roi_screen=caption_roi_screen,
        landmark_geometry=landmark_geometry,
        defer_ocr=defer_ocr,
    )


# Cache replay sets this so a recorded anchor must match by similarity.
# A visual one-pass pick can click a different control that only means the same thing.
_require_similarity_match: ContextVar[bool] = ContextVar(
    "require_similarity_match", default=False
)

RECORDED_ANCHOR_MISS_MESSAGE = (
    "No mouse target matched the instruction on selected monitor(s)."
)

# The anchor-miss sentence stays in the text. move_mouse replay treats this
# failure as a step retry (overlay countdown) instead of a visual re-pick.
CLICK_WINDOW_MISS_MESSAGE = (
    "Recorded click_window is not on screen. " + RECORDED_ANCHOR_MISS_MESSAGE
)


def similarity_match_required() -> bool:
    """True while a cache replay is executing recorded mouse targets."""
    return _require_similarity_match.get()


@contextmanager
def require_similarity_match() -> Iterator[None]:
    """Fail ``move_mouse`` when no detection is similar to the recorded anchor."""
    token = _require_similarity_match.set(True)
    try:
        yield
    finally:
        _require_similarity_match.reset(token)


async def find_mouse_point(
    instruction: str,
    *,
    nearby_objects: list[str] | None = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    click_window: dict[str, Any] | None = None,
    landmark_geometry: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any]] | None:
    """
    Capture selected monitor(s), build YOLO+OCR candidates, filter and pick via LLM.

    Instruction parsing overlaps with capture+YOLO+OCR; results merge before filtering.

    ``nearby_objects`` are optional landmark labels for spatial disambiguation.
    They are merged with any （附近有…） labels parsed from ``instruction``.

    ``landmark_geometry`` is an optional recorded constellation used during cache
    replay (``require_similarity_match``) for non-text anchors (e.g. empty
    ``輸入欄``) and for empty-space clicks with relative pixel offsets
    (``…個像素…的位置``), where constellation replaces raw pixel offsets.
    On-object text anchors (``…文字`` with no offset) never use constellation;
    a similarity miss fails the step.

    Returns ``(global_x, global_y, metadata)`` in virtual-desktop pixel space,
    or ``None`` when no YOLO candidates / no anchor match (soft miss).

    When ``click_window`` is set and the instruction is a system caption button
    (關閉視窗 / 最小化視窗 / 最大化視窗 / 縮小視窗), the point is taken from that
    window's DWM caption strip after WM_NCHITTEST confirms the button. That
    path skips capture, YOLO, and OCR. If the strip exists but the hit-test
    does not confirm the button, vision runs with the matching third of the
    strip as the ROI. A missing DWM strip keeps the window ROI.

    While ``require_similarity_match()`` is active (cache replay), a similarity
    miss returns ``None`` without the visual one-pass or Gemma ROI fallbacks,
    unless the directed nearby landmarks describe exactly one nearby place or
    recorded ``landmark_geometry`` can synthesize / pick a point.

    Successful metadata includes ``timing`` with capture / YOLO / OCR / parse /
    select phase seconds for session time profiles.
    """
    instruction_text = (instruction or "").strip()
    if not instruction_text:
        raise ValueError("instruction must be non-empty")

    total_started = time.perf_counter()
    caption_role = caption_role_from_instruction(instruction_text)
    if caption_role is not None and click_window:
        caption_hit = await asyncio.to_thread(
            resolve_caption_button_mouse_point,
            instruction_text,
            click_window,
        )
        if caption_hit is not None:
            gx, gy, meta = caption_hit
            if nearby_objects:
                meta["nearby_objects"] = [
                    item.strip()
                    for item in nearby_objects
                    if isinstance(item, str) and item.strip()
                ]
            elapsed = time.perf_counter() - total_started
            meta["timing"] = _build_move_mouse_timing(
                total_s=elapsed,
                capture_s=0.0,
                vision={},
                parse_s=0.0,
                select_s=elapsed,
                select_phase="caption_button",
            )
            _log_info(
                "move_mouse caption button "
                f"role={caption_role} point=[{gx},{gy}]"
            )
            return gx, gy, meta
        caption_roi_screen = await asyncio.to_thread(
            caption_button_cell_screen_rect,
            instruction_text,
            click_window,
        )
        if caption_roi_screen is not None:
            _log_info(
                "move_mouse caption button "
                f"role={caption_role} not confirmed; "
                f"vision roi={tuple(int(v) for v in caption_roi_screen)}"
            )
        else:
            _log_info(
                "move_mouse caption button "
                f"role={caption_role} not confirmed; using window vision"
            )
    else:
        caption_roi_screen = None

    vision_mode, _instruction_body = split_vision_mode(instruction_text)
    vision_plan = _mouse_vision_plan(instruction_text, nearby_objects)
    _log_info(
        f"move_mouse vision plan {vision_plan.describe()} vision_mode={vision_mode!r}"
    )

    parse_started = time.perf_counter()
    (
        anchor,
        offset_dx,
        offset_dy,
        nearby_from_instruction,
        char_target,
        char_occurrence,
        track_percent,
    ) = parse_mouse_target_instruction(instruction_text)
    parse_s = time.perf_counter() - parse_started
    nearby_hints = merge_nearby_hints(nearby_objects, nearby_from_instruction)
    nearby_labels = nearby_hints_to_labels(nearby_hints)
    nearby_phrases = nearby_hints_to_phrases(nearby_hints)

    uia_first_chosen: UiDetection | None = None
    grabbed_frames = False
    uia_dets: list[UiDetection] = []
    captured: list[tuple[int, np.ndarray]] = []
    click_hwnd = 0
    if isinstance(click_window, dict):
        try:
            click_hwnd = int(click_window.get("hwnd") or 0)
        except (TypeError, ValueError):
            click_hwnd = 0
    repair_request = None
    if similarity_match_required():
        from src.recorder.uia_replay_repair import take_uia_replay_repair

        repair_request = take_uia_replay_repair()
    # （YOLO+OCR）is the only side a recording asks replay to repair. FindAll
    # does not need the screenshot, so it overlaps capture and OCR. The click
    # still uses YOLO+OCR unless that walk names the same recorded box.
    repair_yolo = repair_request is not None and vision_mode == "yolo_ocr"
    uia_repair_task: asyncio.Task[Any] | None = None
    if repair_yolo and isinstance(click_window, dict):
        uia_repair_task = asyncio.create_task(
            asyncio.to_thread(_live_uia_snapshot_for_click_window, click_window)
        )

        def _retrieve_repair_task_error(task: asyncio.Task[Any]) -> None:
            if not task.cancelled():
                task.exception()

        uia_repair_task.add_done_callback(_retrieve_repair_task_error)
    allow_uia_select = vision_mode != "yolo_ocr"
    filter_by_control_type = vision_mode == "uia"
    if (
        allow_uia_select
        and vision_plan.refine_with_uia
        and isinstance(click_window, dict)
        and click_hwnd
    ):
        from src.recorder.uia_input_boxes import MIN_USABLE_UIA_LANDMARKS

        monitor_indices, image_paths, captured, capture_s = await asyncio.to_thread(
            _grab_selected_monitors
        )
        grabbed_frames = True
        uia_dets = await asyncio.to_thread(
            _uia_only_detections_for_click_window, captured, click_window
        )
        if _uia_path_text_count(uia_dets) >= MIN_USABLE_UIA_LANDMARKS:
            uia_first_chosen, uia_first_reason = _try_uia_first_unique_select(
                uia_dets,
                anchor=anchor,
                nearby_hints=nearby_hints,
                nearby_labels=nearby_labels,
                nearby_phrases=nearby_phrases,
                filter_by_control_type=filter_by_control_type,
            )
            if uia_first_chosen is None:
                _log_info(f"move_mouse uia_first miss reason={uia_first_reason}")
            else:
                detections = _sort_detections_reading_order(uia_dets)
                vision_timing = {
                    "capture_s": capture_s,
                    "yolo_s": 0.0,
                    "line_s": 0.0,
                    "uia_input_s": 0.0,
                    "uia_scrollbar_s": 0.0,
                    "uia_text_s": 0.0,
                    "ocr_s": 0.0,
                    "icon_ocr_retry_s": 0.0,
                    "total_s": capture_s,
                    "vision_path": "uia",
                    "uia_first_hit": True,
                }
                ocr_pendings = []
                _log_info("move_mouse uia_first=hit; skipping YOLO")

    if uia_first_chosen is None:
        if grabbed_frames:
            vision = await asyncio.to_thread(
                _finish_captured_mouse_candidates,
                monitor_indices,
                image_paths,
                captured,
                capture_s,
                yolo_conf_threshold,
                ocr_class_ids=vision_plan.ocr_class_ids,
                refine_with_uia=True,
                click_window=click_window,
                caption_roi_screen=caption_roi_screen,
                landmark_geometry=landmark_geometry,
                defer_ocr=False,
            )
        else:
            vision = await asyncio.to_thread(
                _capture_and_detect_mouse_candidates,
                yolo_conf_threshold,
                ocr_class_ids=vision_plan.ocr_class_ids,
                refine_with_uia=vision_plan.refine_with_uia,
                click_window=click_window,
                caption_roi_screen=caption_roi_screen,
                landmark_geometry=landmark_geometry,
                defer_ocr=True,
            )
        monitor_indices, image_paths, captured, detections, vision_timing = vision
        capture_s = float(vision_timing.get("capture_s", 0.0))
        if grabbed_frames:
            vision_timing["vision_path"] = "yolo_ocr"
            ocr_pendings = []
        else:
            ocr_pending_raw = vision_timing.pop("_ocr_pending", None)
            ocr_pendings = (
                list(ocr_pending_raw)
                if isinstance(ocr_pending_raw, list) and ocr_pending_raw
                else []
            )

    def _timing_meta(*, select_s: float, select_phase: str) -> dict[str, Any]:
        return _build_move_mouse_timing(
            total_s=time.perf_counter() - total_started,
            capture_s=capture_s,
            vision=vision_timing,
            parse_s=parse_s,
            select_s=select_s,
            select_phase=select_phase,
        )

    # No-window callers still finish a deferred OCR pass. A click-window UIA hit
    # already set ``uia_first_chosen`` and must not be replaced.
    if (
        allow_uia_select
        and uia_first_chosen is None
        and ocr_pendings
        and detections
    ):
        uia_first_chosen, uia_first_reason = _try_uia_first_unique_select(
            detections,
            anchor=anchor,
            nearby_hints=nearby_hints,
            nearby_labels=nearby_labels,
            nearby_phrases=nearby_phrases,
            filter_by_control_type=filter_by_control_type,
        )
        if uia_first_chosen is None:
            _log_info(f"move_mouse uia_first miss reason={uia_first_reason}")
            detections, vision_timing = await asyncio.to_thread(
                _finish_deferred_ocr_pendings,
                ocr_pendings,
                timing_base=vision_timing,
            )
            capture_s = float(vision_timing.get("capture_s", 0.0))
            _write_yolo_ocr_sidecars(
                captured, image_paths, detections, vision_timing
            )
            if detections:
                _log_info(
                    "move_mouse all_ocr_candidates (after deferred OCR):\n"
                    + _format_ui_candidates_text(detections, include_geometry=True)
                )
        else:
            vision_timing["uia_first_hit"] = True
            vision_timing.pop("uia_first_pending", None)
            ocr_pendings = []
            _log_info("move_mouse uia_first=hit; skipping leftover OCR")
    elif ocr_pendings and not detections:
        # Should not happen (pending implies partial candidates), but finish anyway.
        detections, vision_timing = await asyncio.to_thread(
            _finish_deferred_ocr_pendings,
            ocr_pendings,
            timing_base=vision_timing,
        )
        capture_s = float(vision_timing.get("capture_s", 0.0))
        ocr_pendings = []
    elif ocr_pendings and not allow_uia_select:
        # YOLO+OCR instructions skip the UIA commit, so the deferred OCR pass
        # still has to finish or the click has no text candidates.
        detections, vision_timing = await asyncio.to_thread(
            _finish_deferred_ocr_pendings,
            ocr_pendings,
            timing_base=vision_timing,
        )
        capture_s = float(vision_timing.get("capture_s", 0.0))
        ocr_pendings = []

    if repair_yolo:
        from src.recorder.uia_replay_repair import attempt_uia_replay_repair

        if ocr_pendings:
            detections, vision_timing = await asyncio.to_thread(
                _finish_deferred_ocr_pendings,
                ocr_pendings,
                timing_base=vision_timing,
            )
            capture_s = float(vision_timing.get("capture_s", 0.0))
            ocr_pendings = []
        if uia_repair_task is not None:
            walked: tuple[Any, dict[str, Any]] | None = None
            try:
                walked = await uia_repair_task
            except Exception as exc:
                _log_info(
                    "uia replay repair walk failed "
                    f"{type(exc).__name__}: {exc}"
                )
            if walked is not None:
                live, snapshot = walked
                uia_dets = _detections_from_live_uia_snapshot(
                    captured, live, snapshot
                )
        assert repair_request is not None
        repaired = attempt_uia_replay_repair(
            repair_request,
            click_window,
            _labeled_text_boxes(uia_dets, source="uia"),
            _labeled_text_boxes(detections, source="ocr"),
        )
        if repaired is not None:
            gx, gy, repair_meta = repaired
            image_path = image_paths[0] if image_paths else ""
            repair_meta["anchor_instruction"] = instruction_text
            repair_meta["nearby_objects"] = nearby_phrases
            repair_meta["screenshot_path"] = image_path
            repair_meta["screenshot_paths"] = image_paths
            repair_meta["timing"] = _timing_meta(
                select_s=0.0, select_phase="uia_replay_repair"
            )
            return gx, gy, repair_meta

    selection_method: str | None = None
    if uia_first_chosen is not None:
        select_started = time.perf_counter()
        chosen = uia_first_chosen
        idx = 0
        selected_text = chosen.text
        select_phase = "select_unique"
        select_s = time.perf_counter() - select_started
        nearby_matches: list[UiDetection] = []
        # Jump to resolve / meta construction (same as unique path below).
        resolved_x = chosen.cx + offset_dx
        resolved_y = chosen.cy + offset_dy
        char_center: dict[str, int] | None = None
        if char_target:
            char_point = _resolve_char_target_point(
                chosen,
                captured,
                char_target,
                char_occurrence=char_occurrence,
            )
            if char_point is not None:
                resolved_x, resolved_y = char_point
                char_center = {"x": resolved_x, "y": resolved_y}
            else:
                _log_info(
                    "move_mouse char target fallback to anchor center "
                    f"char={char_target!r} occurrence={char_occurrence} "
                    f"anchor=[{chosen.cx},{chosen.cy}]"
                )
        elif track_percent is not None and chosen.class_name == "scrollbar":
            resolved_x, resolved_y = point_from_scrollbar_percent(
                chosen.bbox, track_percent
            )
        image_path = image_paths[0] if image_paths else ""
        overlay = _live_html_overlay_fields(
            chosen,
            detections,
            nearby_matches,
            nearby_hints,
        )
        meta = {
            "selected_index": idx,
            "class_name": chosen.class_name,
            "image_center": {"x": chosen.cx, "y": chosen.cy},
            "resolved_center": {"x": resolved_x, "y": resolved_y},
            "relative_offset": {"dx": offset_dx, "dy": offset_dy},
            "track_percent": track_percent,
            "char_target": char_target,
            "char_occurrence": char_occurrence,
            "anchor_instruction": instruction_text,
            "nearby_objects": nearby_phrases,
            "screenshot_path": image_path,
            "screenshot_paths": image_paths,
            "target_kind": chosen.class_name,
            "target_text": chosen.text or "",
            "target_icons": chosen.icons or [],
            "target_bbox": {
                "x": chosen.bbox[0],
                "y": chosen.bbox[1],
                "w": chosen.bbox[2],
                "h": chosen.bbox[3],
            },
            "peer_bboxes": overlay["peer_bboxes"],
            "landmarks": overlay["landmarks"],
            "selection_method": "uia_first_unique",
            "timing": _timing_meta(
                select_s=select_s, select_phase=select_phase
            ),
        }
        if selected_text is not None:
            meta["selected_text"] = selected_text
        if char_center is not None:
            meta["char_center"] = char_center
        return resolved_x, resolved_y, meta

    if not detections:
        if similarity_match_required():
            _log_info(
                "move_mouse: no YOLO candidates; similarity match required, "
                "skipping Gemma ROI fallback"
            )
            return None
        # Stage B: Gemma ROI → YOLO/OCR on crop → index pick.
        _log_info(
            "move_mouse: no YOLO candidates; trying Gemma ROI + YOLO-on-crop fallback"
        )
        from cua_mcp.gemma_roi_refine import resolve_target_via_gemma_roi

        select_started = time.perf_counter()
        grounded = await resolve_target_via_gemma_roi(
            instruction_text,
            image_paths=image_paths,
            monitor_indices=monitor_indices,
            captured=captured,
            yolo_conf_threshold=yolo_conf_threshold,
        )
        select_s = time.perf_counter() - select_started
        if grounded is None:
            _log_info("move_mouse: Gemma ROI fallback found nothing")
            return None
        gx, gy, roi_meta = grounded
        gx = int(gx) + offset_dx
        gy = int(gy) + offset_dy
        meta = dict(roi_meta)
        meta["relative_offset"] = {"dx": offset_dx, "dy": offset_dy}
        meta["resolved_center"] = {"x": gx, "y": gy}
        meta["anchor_instruction"] = instruction_text
        meta["nearby_objects"] = nearby_phrases
        meta["char_target"] = char_target
        meta["char_occurrence"] = char_occurrence
        meta["track_percent"] = track_percent
        meta["timing"] = _timing_meta(select_s=select_s, select_phase="gemma_roi")
        return gx, gy, meta

    anchor_matches, nearby_matches = _filter_mouse_candidates(
        detections, anchor, nearby_labels
    )
    _log_info(
        f"move_mouse after_filter anchor={len(anchor_matches)} "
        f"nearby={len(nearby_matches)} "
        f"anchor={anchor!r} nearby_labels={nearby_phrases!r}"
    )

    selected_text: str | None = None
    select_phase = "select"
    select_started = time.perf_counter()
    is_offset_click = bool(offset_dx or offset_dy) or _is_relative_offset_position(
        instruction_text
    )

    # Empty-space / relative-offset clicks: prefer constellation over raw pixels.
    if (
        similarity_match_required()
        and landmark_geometry is not None
        and is_offset_click
    ):
        geom_hit = _resolve_with_landmark_geometry(
            detections,
            landmark_geometry,
            anchor=anchor,
            nearby_hints=nearby_hints,
            synth_only=True,
        )
        if geom_hit is not None:
            _geom_det, geom_point, geom_method = geom_hit
            select_s = time.perf_counter() - select_started
            _log_info(
                "move_mouse: landmark geometry synth for offset click "
                f"point=[{geom_point[0]},{geom_point[1]}] "
                f"(ignoring raw pixel offset dx={offset_dx} dy={offset_dy})"
            )
            return _landmark_geometry_synth_meta(
                point=geom_point,
                offset_dx=0,
                offset_dy=0,
                track_percent=track_percent,
                char_target=char_target,
                char_occurrence=char_occurrence,
                instruction_text=instruction_text,
                nearby_phrases=nearby_phrases,
                image_paths=image_paths,
                selection_method=geom_method,
                timing=_timing_meta(select_s=select_s, select_phase=geom_method),
            )

    if not anchor_matches:
        # On-object text anchors must match by OCR/similarity. Constellation /
        # landmark-unique fallbacks are for non-text targets (e.g. empty 輸入欄)
        # and for offset empty-space clicks (handled above).
        if (
            similarity_match_required()
            and _anchor_is_text(anchor)
            and not is_offset_click
        ):
            _log_info(
                "move_mouse: no similarity match for text anchor; "
                "failing step (no constellation fallback for 文字)"
            )
            return None
        if (
            similarity_match_required()
            and landmark_geometry is not None
            and not _anchor_is_text(anchor)
        ):
            geom_hit = _resolve_with_landmark_geometry(
                detections,
                landmark_geometry,
                anchor=anchor,
                nearby_hints=nearby_hints,
            )
            if geom_hit is not None:
                geom_det, geom_point, geom_method = geom_hit
                if geom_det is not None:
                    chosen = geom_det
                    idx = detections.index(geom_det)
                    selection_method = geom_method
                    select_phase = geom_method
                    _log_info(
                        "move_mouse: landmark geometry pick "
                        f"center=[{chosen.cx},{chosen.cy}] method={geom_method}"
                    )
                else:
                    select_s = time.perf_counter() - select_started
                    _log_info(
                        "move_mouse: landmark geometry synth "
                        f"point=[{geom_point[0]},{geom_point[1]}]"
                    )
                    return _landmark_geometry_synth_meta(
                        point=geom_point,
                        offset_dx=offset_dx,
                        offset_dy=offset_dy,
                        track_percent=track_percent,
                        char_target=char_target,
                        char_occurrence=char_occurrence,
                        instruction_text=instruction_text,
                        nearby_phrases=nearby_phrases,
                        image_paths=image_paths,
                        selection_method=geom_method,
                        timing=_timing_meta(
                            select_s=select_s, select_phase=geom_method
                        ),
                    )
        unique = _unique_target_from_landmarks(
            detections,
            nearby_hints,
            anchor=anchor,
        )
        if unique is not None and selection_method is None:
            chosen = unique
            idx = detections.index(unique)
            selection_method = "landmark_unique"
            select_phase = "landmark_unique"
            _log_info(
                "move_mouse: landmark-unique fallback "
                f"center=[{chosen.cx},{chosen.cy}] "
                f"nearby_labels={nearby_phrases!r}"
            )
        elif selection_method is not None:
            pass
        elif similarity_match_required():
            _log_info(
                "move_mouse: no similarity anchor matches; similarity match required, "
                "skipping visual one-pass fallback"
            )
            return None
        else:
            # Stage A: similarity miss but detections exist → Gemma pick among all.
            _log_info(
                "move_mouse: no similarity anchor matches; "
                "trying visual one-pass fallback over all detections"
            )
            from cua_mcp.gemma_roi_refine import pick_candidate_with_gemma

            try:
                idx, selected_text, chosen = await pick_candidate_with_gemma(
                    instruction_text,
                    detections,
                    image_paths,
                )
            except Exception as exc:
                _log_info(
                    "move_mouse: visual one-pass fallback failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return None
            selection_method = "visual_one_pass_fallback"
            select_phase = "visual_one_pass"
            _log_info(
                "move_mouse: visual one-pass fallback "
                f"index={idx} text={selected_text!r} center=[{chosen.cx},{chosen.cy}]"
            )
            nearby_matches = []
    else:
        before_nearby = len(anchor_matches)
        admitted_unknowns = _admit_unknown_icon_peers(
            detections,
            anchor_matches,
            nearby_matches,
            nearby_hints,
            anchor=anchor,
        )
        if admitted_unknowns:
            centers = ", ".join(f"({d.cx},{d.cy})" for d in admitted_unknowns)
            _log_info(
                "move_mouse: admitted blank unknown icon peers "
                f"count={len(admitted_unknowns)} centers=[{centers}] "
                f"labeled_anchors={before_nearby}"
            )
            anchor_matches = _merge_anchor_detections(
                anchor_matches, admitted_unknowns
            )
        after_admit = len(anchor_matches)
        anchor_matches = _prefilter_anchors_by_nearby(
            anchor_matches, nearby_matches, nearby_hints
        )
        if len(anchor_matches) != before_nearby or admitted_unknowns:
            _log_info(
                "move_mouse nearby prefilter "
                f"anchors {before_nearby} -> {after_admit} admitted -> "
                f"{len(anchor_matches)} nearby_labels={nearby_phrases!r}"
            )

        # Cache replay with recorded constellation: prefer geometry over LLM pick
        # (e.g. multiple 輸入欄 class matches). On-object text still uses unique/LLM.
        used_geometry = False
        if (
            similarity_match_required()
            and landmark_geometry is not None
            and not _anchor_is_text(anchor)
            and not is_offset_click
        ):
            geom_hit = _resolve_with_landmark_geometry(
                detections,
                landmark_geometry,
                anchor=anchor,
                nearby_hints=nearby_hints,
                candidate_pool=anchor_matches,
            )
            if geom_hit is not None:
                geom_det, geom_point, geom_method = geom_hit
                if geom_det is not None:
                    chosen = geom_det
                    idx = detections.index(geom_det)
                    selection_method = geom_method
                    select_phase = geom_method
                    used_geometry = True
                    _log_info(
                        "move_mouse: landmark geometry pick among anchors "
                        f"center=[{chosen.cx},{chosen.cy}] method={geom_method} "
                        f"pool={len(anchor_matches)}"
                    )
                else:
                    select_s = time.perf_counter() - select_started
                    _log_info(
                        "move_mouse: landmark geometry synth among anchors "
                        f"point=[{geom_point[0]},{geom_point[1]}] "
                        f"pool={len(anchor_matches)}"
                    )
                    return _landmark_geometry_synth_meta(
                        point=geom_point,
                        offset_dx=offset_dx,
                        offset_dy=offset_dy,
                        track_percent=track_percent,
                        char_target=char_target,
                        char_occurrence=char_occurrence,
                        instruction_text=instruction_text,
                        nearby_phrases=nearby_phrases,
                        image_paths=image_paths,
                        selection_method=geom_method,
                        timing=_timing_meta(
                            select_s=select_s, select_phase=geom_method
                        ),
                    )

        if used_geometry:
            pass
        elif len(anchor_matches) == 1:
            idx = 0
            chosen = anchor_matches[0]
            select_phase = "select_unique"
            _log_info(
                "move_mouse: single anchor candidate after filter; skipping LLM pick"
            )
        else:
            pool_idx, selected_text = await _select_center_with_llm(
                anchor,
                anchor_matches,
                image_paths,
                neighbor_candidates=nearby_matches,
                nearby_labels=nearby_phrases,
            )
            idx = pool_idx
            chosen = anchor_matches[pool_idx]
            select_phase = "llm_pick"
            _log_info(
                f"move_mouse: LLM picked index={pool_idx} "
                f"text={selected_text!r} center=[{chosen.cx},{chosen.cy}]"
            )
    select_s = time.perf_counter() - select_started

    # similar_function_describe runs on move_mouse_visual only (see visual_mouse.py).

    resolved_x = chosen.cx + offset_dx
    resolved_y = chosen.cy + offset_dy
    char_center: dict[str, int] | None = None
    if char_target:
        char_point = _resolve_char_target_point(
            chosen,
            captured,
            char_target,
            char_occurrence=char_occurrence,
        )
        if char_point is not None:
            resolved_x, resolved_y = char_point
            char_center = {"x": resolved_x, "y": resolved_y}
            _log_info(
                "move_mouse char target applied "
                f"char={char_target!r} occurrence={char_occurrence} "
                f"resolved=[{resolved_x},{resolved_y}]"
            )
        else:
            _log_info(
                "move_mouse char target fallback to anchor center "
                f"char={char_target!r} occurrence={char_occurrence} "
                f"anchor=[{chosen.cx},{chosen.cy}]"
            )
    elif track_percent is not None:
        scrollbar = chosen
        if chosen.class_name != "scrollbar":
            scrollbar = next(
                (det for det in anchor_matches if det.class_name == "scrollbar"),
                chosen,
            )
        if scrollbar.class_name == "scrollbar":
            resolved_x, resolved_y = point_from_scrollbar_percent(
                scrollbar.bbox, track_percent
            )
            chosen = scrollbar
            _log_info(
                "move_mouse scrollbar track percent applied "
                f"percent={track_percent} "
                f"bbox={list(scrollbar.bbox)} "
                f"resolved=[{resolved_x},{resolved_y}]"
            )
        elif offset_dx or offset_dy:
            _log_info(
                "move_mouse relative offset applied "
                f"dx={offset_dx} dy={offset_dy} "
                f"anchor=[{chosen.cx},{chosen.cy}] resolved=[{resolved_x},{resolved_y}]"
            )
    elif offset_dx or offset_dy:
        _log_info(
            "move_mouse relative offset applied "
            f"dx={offset_dx} dy={offset_dy} "
            f"anchor=[{chosen.cx},{chosen.cy}] resolved=[{resolved_x},{resolved_y}]"
        )

    image_path = image_paths[0] if image_paths else ""
    overlay = _live_html_overlay_fields(
        chosen,
        detections,
        nearby_matches,
        nearby_hints,
    )
    meta: dict[str, Any] = {
        "selected_index": idx,
        "class_name": chosen.class_name,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "resolved_center": {"x": resolved_x, "y": resolved_y},
        "relative_offset": {"dx": offset_dx, "dy": offset_dy},
        "track_percent": track_percent,
        "char_target": char_target,
        "char_occurrence": char_occurrence,
        "anchor_instruction": instruction_text,
        "nearby_objects": nearby_phrases,
        "screenshot_path": image_path,
        "screenshot_paths": image_paths,
        "target_kind": chosen.class_name,
        "target_text": chosen.text or "",
        "target_icons": chosen.icons or [],
        "target_bbox": {
            "x": chosen.bbox[0],
            "y": chosen.bbox[1],
            "w": chosen.bbox[2],
            "h": chosen.bbox[3],
        },
        "peer_bboxes": overlay["peer_bboxes"],
        "landmarks": overlay["landmarks"],
        "timing": _timing_meta(select_s=select_s, select_phase=select_phase),
    }
    if selected_text is not None:
        meta["selected_text"] = selected_text
    if char_center is not None:
        meta["resolved_char_center"] = char_center
    if selection_method is not None:
        meta["selection_method"] = selection_method
    return resolved_x, resolved_y, meta


async def resolve_mouse_point(
    instruction: str,
    *,
    nearby_objects: list[str] | None = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    click_window: dict[str, Any] | None = None,
    landmark_geometry: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    """
    Capture selected monitor(s), build YOLO+OCR candidates, filter and pick via LLM.

    Instruction parsing overlaps with capture+YOLO+OCR (see ``find_mouse_point``).

    ``nearby_objects`` are optional landmark labels for spatial disambiguation.
    They are merged with any （附近有…） labels parsed from ``instruction``.

    Returns ``(global_x, global_y, metadata)`` in virtual-desktop pixel space.
    Raises ``ValueError`` when the target is not found.
    """
    found = await find_mouse_point(
        instruction,
        nearby_objects=nearby_objects,
        yolo_conf_threshold=yolo_conf_threshold,
        click_window=click_window,
        landmark_geometry=landmark_geometry,
    )
    if found is None:
        raise ValueError(RECORDED_ANCHOR_MISS_MESSAGE)
    return found
