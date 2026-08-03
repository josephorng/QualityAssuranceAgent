"""
Unified mouse target selection: YOLO (text, element, input, scrollbar) + OCR + LLM filter/pick.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
import re
import time
from typing import Any

import cv2
import numpy as np

from cua_mcp.geometry import clip_box, iou_xywh
from cua_mcp.instruction_offset import parse_mouse_target_instruction
from cua_mcp.icon_map import (
    describe_text_icons,
    is_pua_char,
    is_unknown_icon_record,
    text_has_pua,
)
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.select_ui_element import (
    UiDetection,
    _ANCHOR_SUFFIX_BY_CLASS,
    _assign_exclusive_neighbors_to_anchors,
    _describe_ui_candidate_functions,
    _format_ui_candidates_text,
    _select_center_with_functions,
    _select_center_with_ollama,
    _sort_detections_reading_order,
)
from src.common.nearby_side import (
    NearbyHint,
    anchor_satisfies_side,
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
    YOLO_CLASS_NAMES,
    YOLO_CLASS_TEXT,
    run_yolo_onnx_end2end,
)
from src.common.monitor_prompt import selected_eye_monitor_indices
from src.common.run_state import RunStateManager, get_run_state_manager, ts_name
from src.eye.capture import active_monitor_offset, capture_monitor_to_file, monitor_details


def _run_manager() -> RunStateManager:
    """Always resolve the current singleton (never cache): ``reset_run_state_manager`` replaces it."""
    return get_run_state_manager()


def _log_info(text: str) -> None:
    """Write to the run log when run state is initialized; no-op otherwise."""
    try:
        _run_manager().log_info(text)
    except RuntimeError:
        pass


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
    - YOLO ``element`` whose OCR decode is plain text (no PUA)

    Known-icon PUA and empty element OCR keep their YOLO class.
    """
    if text_value and _text_is_pua_only(text_value) and not _known_icons_for_text(text_value):
        return PICKER_CLASS_UNKNOWN
    if class_id != YOLO_CLASS_ELEMENT:
        return class_id
    if not text_value:
        return class_id
    if text_has_pua(text_value):
        return class_id
    return PICKER_CLASS_UNKNOWN


def _detection_from_bbox(
    bbox: tuple[int, int, int, int],
    class_id: int,
    *,
    text: str | None = None,
    icons: list[dict[str, Any]] | None = None,
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


def _detection_preference_score(det: UiDetection) -> tuple[int, int, int]:
    """Higher score is kept when two overlapping detections share content."""
    has_icons = 1 if det.icons else 0
    has_text = 1 if (det.text or "").strip() else 0
    area = int(det.bbox[2]) * int(det.bbox[3])
    return (has_icons, has_text, area)


def _detections_are_content_duplicates(a: UiDetection, b: UiDetection) -> bool:
    """True when ``a`` and ``b`` share the same text/icon identity (or blank same-class)."""
    key_a = _detection_content_key(a)
    key_b = _detection_content_key(b)
    if key_a != key_b:
        return False
    if key_a != ("", ""):
        return True
    return a.class_id == b.class_id


def _dedupe_overlapping_detections(
    detections: list[UiDetection],
    *,
    iou_threshold: float = DEFAULT_DEDUPE_OVERLAP_IOU_THRESHOLD,
) -> list[UiDetection]:
    """
    Keep one detection per heavily overlapping same-content group.

    Candidates are compared by IoU of ``(x, y, w, h)`` boxes. When IoU exceeds
    ``iou_threshold`` and content matches (same OCR text and icon labels, or
    blank same-class boxes), only the preferred detection is kept.
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
            and _detections_are_content_duplicates(det, detections[j])
            for j in kept
        ):
            continue
        kept.append(i)
    kept.sort()
    return [detections[i] for i in kept]


def _build_candidates_from_bgr(
    bgr: np.ndarray,
    *,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[UiDetection]:
    """Run YOLO on ``bgr``, OCR text/element boxes, return local-coordinate candidates."""
    h, w = bgr.shape[:2]
    vision_started = time.perf_counter()
    yolo_started = time.perf_counter()
    try:
        xyxy, _scores, class_ids = run_yolo_onnx_end2end(
            bgr,
            class_ids=set(MOUSE_TARGET_CLASS_IDS),
            conf_threshold=yolo_conf_threshold,
        )
    except Exception as exc:
        _log_info(f"move_mouse YOLO failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"move_mouse YOLO predict failed: {exc}") from exc
    yolo_elapsed = time.perf_counter() - yolo_started

    if xyxy.size == 0:
        _log_info(
            f"move_mouse vision profile yolo_s={yolo_elapsed:.3f} ocr_s=0.000 "
            f"total_s={time.perf_counter() - vision_started:.3f} detections=0"
        )
        return []

    ocr_boxes: list[tuple[int, int, int, int]] = []
    ocr_class_ids: list[int] = []
    non_ocr: list[tuple[tuple[int, int, int, int], int]] = []

    for row, cls_id in zip(xyxy, class_ids, strict=True):
        cls_id = int(cls_id)
        bbox = _xyxy_row_to_bbox(row, w, h)
        if cls_id in (YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT):
            ocr_boxes.append(bbox)
            ocr_class_ids.append(cls_id)
        else:
            non_ocr.append((bbox, cls_id))

    ocr_started = time.perf_counter()
    ocr_preds = _ocr_boxes_on_bgr(bgr, ocr_boxes) if ocr_boxes else []
    ocr_elapsed = time.perf_counter() - ocr_started

    candidates: list[UiDetection] = []
    for bbox, cls_id in non_ocr:
        candidates.append(_detection_from_bbox(bbox, cls_id))

    for bbox, cls_id, preds in zip(ocr_boxes, ocr_class_ids, ocr_preds, strict=True):
        text_value = "".join(preds).strip()
        if cls_id == YOLO_CLASS_TEXT and not text_value:
            continue
        resolved_cls = _resolve_ocr_class_id(cls_id, text_value)
        candidates.append(
            _detection_from_bbox(bbox, resolved_cls, text=text_value or None)
        )

    before_dedupe = len(candidates)
    candidates = _dedupe_overlapping_detections(candidates)
    _log_info(
        "move_mouse vision profile "
        f"yolo_s={yolo_elapsed:.3f} ocr_s={ocr_elapsed:.3f} "
        f"total_s={time.perf_counter() - vision_started:.3f} "
        f"yolo_boxes={len(xyxy)} ocr_boxes={len(ocr_boxes)} "
        f"candidates={len(candidates)} deduped_from={before_dedupe}"
    )

    return candidates


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


def _normalize_similarity_label(label: str) -> str:
    """Strip hub wrappers so ``「擷取」文字`` compares as ``擷取`` against raw OCR.

    Prefer the content of the leftmost wrapper pair among ``""`` / ``「」`` /
    ``『』`` / ``【】`` / ``〔〕`` / ``[]`` when present; otherwise keep the trimmed
    label. Class-only anchors such as ``輸入欄`` / ``滾動條`` are unchanged.
    """
    text = (label or "").strip()
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
        inner = text[best_start + 1 : best_end].strip()
        if inner:
            return inner
    return text


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
# Max peers sent to the post-pick function-describe disambiguator.
_SIMILAR_PEER_CAP = 12


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
    cap: int = _SIMILAR_PEER_CAP,
) -> list[UiDetection]:
    """
    Return YOLO detections label-similar to ``chosen``.

    ``chosen`` is always first. Remaining peers follow ``detections`` order
    (typically reading order). Blank detections never match. Result length is
    capped at ``cap``.
    """
    if cap < 1:
        return []
    if not _detection_match_labels(chosen):
        return [chosen]

    ordered: list[UiDetection] = [chosen]
    seen: set[int] = {id(chosen)}
    for det in detections:
        if id(det) in seen:
            continue
        if not _detections_label_similar(chosen, det, threshold=threshold):
            continue
        ordered.append(det)
        seen.add(id(det))
        if len(ordered) >= cap:
            break
    return ordered


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

        bgr = cv2.imread(image_path)
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
        if not cv2.imwrite(str(out_path), canvas):
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

    Anchor detections use the highest similarity score only; lower-scoring
    partial matches are dropped unless they tie for first. Nearby buckets are
    filled by testing each nearby label independently at ``threshold``. Indices
    that match the anchor are excluded from nearby (anchor wins).
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
        for i, det in enumerate(detections):
            if i in anchor_set or i in nearby_seen:
                continue
            if _detection_similarity_to_query(det, query) >= threshold:
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


def _detections_for_captured_monitor(
    monitor_index: int,
    bgr: np.ndarray,
    *,
    yolo_conf_threshold: float,
) -> list[UiDetection]:
    """YOLO+OCR on one monitor image, then map boxes into virtual-desktop coords."""
    local_candidates = _build_candidates_from_bgr(
        bgr,
        yolo_conf_threshold=yolo_conf_threshold,
    )
    left, top = active_monitor_offset(monitor_index)
    return [_offset_detection(d, left, top) for d in local_candidates]


def _collect_monitor_detections(
    captured: list[tuple[int, np.ndarray]],
    *,
    yolo_conf_threshold: float,
) -> list[UiDetection]:
    """
    Run YOLO+OCR on captured monitors.

    Capture stays sequential (caller); inference runs in parallel when there is more
    than one monitor so Triton/ORT work can overlap.
    """
    if not captured:
        return []

    if len(captured) == 1:
        monitor_index, bgr = captured[0]
        return _detections_for_captured_monitor(
            monitor_index,
            bgr,
            yolo_conf_threshold=yolo_conf_threshold,
        )

    all_detections: list[UiDetection] = []
    with ThreadPoolExecutor(max_workers=len(captured)) as pool:
        futures = [
            pool.submit(
                _detections_for_captured_monitor,
                monitor_index,
                bgr,
                yolo_conf_threshold=yolo_conf_threshold,
            )
            for monitor_index, bgr in captured
        ]
        for future in futures:
            all_detections.extend(future.result())
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
            if not anchor_satisfies_side(anchor.bbox, neigh.cx, neigh.cy, hint.side):
                continue
        return True
    return False


def _prefilter_anchors_by_nearby(
    anchors: list[UiDetection],
    nearby_matches: list[UiDetection],
    nearby_labels: list[str] | list[NearbyHint] | None,
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """Narrow anchors using exclusive nearby-landmark assignment.

    Prefer anchors whose assigned neighbors cover **all** nearby hints (label and
    optional side). Directed sides are checked against **all** ``nearby_matches``
    for each anchor (so exclusive distance assignment cannot hide the correct
    side). Undirected labels still use exclusive assignment. If directed sides
    wipe every anchor, retry with label-only coverage. If none match, return
    ``anchors`` unchanged.
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

        n_hints = len(hints)
        full = [
            anchors[i] for i, covered in enumerate(coverage) if len(covered) == n_hints
        ]
        if full:
            return full
        partial = [anchors[i] for i, covered in enumerate(coverage) if covered]
        if partial:
            return partial
        return []

    selected = _select(require_side=True)
    if selected:
        return selected

    if any(hint.side is not None for hint in hints):
        selected = _select(require_side=False)
        if selected:
            return selected

    return anchors


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
        "move_mouse similar peers for function describe "
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
        "move_mouse function-describe re-pick "
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


async def find_mouse_point(
    instruction: str,
    *,
    nearby_objects: list[str] | None = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[int, int, dict[str, Any]] | None:
    """
    Capture selected monitor(s), build YOLO+OCR candidates, filter and pick via LLM.

    ``nearby_objects`` are optional landmark labels for spatial disambiguation.
    They are merged with any （附近有…） labels parsed from ``instruction``.

    Returns ``(global_x, global_y, metadata)`` in virtual-desktop pixel space,
    or ``None`` when no YOLO candidates / no anchor match (soft miss).
    """
    instruction_text = (instruction or "").strip()
    if not instruction_text:
        raise ValueError("instruction must be non-empty")

    # Parse anchor/offset/nearby once; filter splits anchor vs nearby; pick uses anchor only.
    anchor, offset_dx, offset_dy, nearby_from_instruction = await parse_mouse_target_instruction(
        instruction_text
    )
    nearby_hints = merge_nearby_hints(nearby_objects, nearby_from_instruction)
    nearby_labels = nearby_hints_to_labels(nearby_hints)
    nearby_phrases = nearby_hints_to_phrases(nearby_hints)

    paths = _run_manager().require_paths()
    monitor_indices = selected_eye_monitor_indices()
    stamp = ts_name()
    image_paths: list[str] = []
    captured: list[tuple[int, np.ndarray]] = []

    _log_info(f"move_mouse resolve monitors={monitor_indices}")
    # Capture sequentially — desktop grabbers are often not concurrent-safe.
    for monitor_index in monitor_indices:
        name = f"{stamp}_mon{monitor_index}.png"
        out = paths.yolo_ocr_dir / name
        capture_monitor_to_file(out, monitor_index)
        image_path = str(out.resolve())
        image_paths.append(image_path)

        bgr = cv2.imread(image_path)
        if bgr is None:
            _log_info(f"move_mouse could not read captured image path={image_path}")
            continue
        captured.append((monitor_index, bgr))

    all_detections = _collect_monitor_detections(
        captured,
        yolo_conf_threshold=yolo_conf_threshold,
    )
    detections = _sort_detections_reading_order(all_detections)
    _log_info(f"move_mouse yolo_candidates={len(detections)}")
    if detections:
        _log_info(
            "move_mouse all_ocr_candidates:\n"
            + _format_ui_candidates_text(detections, include_geometry=True)
        )

    if not detections:
        _log_info("move_mouse: no YOLO candidates found on selected monitor(s)")
        return None

    anchor_matches, nearby_matches = _filter_mouse_candidates(
        detections, anchor, nearby_labels
    )
    _log_info(
        f"move_mouse after_filter anchor={len(anchor_matches)} "
        f"nearby={len(nearby_matches)} "
        f"anchor={anchor!r} nearby_labels={nearby_phrases!r}"
    )

    if not anchor_matches:
        _log_info("move_mouse: no anchor candidates matched after LLM filtering")
        return None

    before_nearby = len(anchor_matches)
    anchor_matches = _prefilter_anchors_by_nearby(
        anchor_matches, nearby_matches, nearby_hints
    )
    if len(anchor_matches) != before_nearby:
        _log_info(
            "move_mouse nearby prefilter "
            f"anchors {before_nearby} -> {len(anchor_matches)} "
            f"nearby_labels={nearby_phrases!r}"
        )

    selected_text: str | None = None
    if len(anchor_matches) == 1:
        idx = 0
        chosen = anchor_matches[0]
        _log_info("move_mouse: single anchor candidate after filter; skipping Ollama pick")
    else:
        pool_idx, selected_text = await _select_center_with_ollama(
            anchor,
            anchor_matches,
            image_paths,
            neighbor_candidates=nearby_matches,
            nearby_labels=nearby_phrases,
        )
        idx = pool_idx
        chosen = anchor_matches[pool_idx]
        _log_info(
            f"move_mouse: Ollama picked index={pool_idx} "
            f"text={selected_text!r} center=[{chosen.cx},{chosen.cy}]"
        )

    chosen, idx, selected_text, disambiguation_meta = await _maybe_disambiguate_similar_selection(
        chosen=chosen,
        initial_idx=idx,
        selected_text=selected_text,
        detections=detections,
        image_paths=image_paths,
        monitor_indices=list(monitor_indices),
        overlay_dir=paths.yolo_ocr_dir,
        overlay_stamp=f"{stamp}_sim",
        anchor=anchor,
        nearby_phrases=nearby_phrases,
        nearby_matches=nearby_matches,
    )

    resolved_x = chosen.cx + offset_dx
    resolved_y = chosen.cy + offset_dy
    if offset_dx or offset_dy:
        _log_info(
            "move_mouse relative offset applied "
            f"dx={offset_dx} dy={offset_dy} "
            f"anchor=[{chosen.cx},{chosen.cy}] resolved=[{resolved_x},{resolved_y}]"
        )

    image_path = image_paths[0] if image_paths else ""
    meta: dict[str, Any] = {
        "selected_index": idx,
        "class_name": chosen.class_name,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "resolved_center": {"x": resolved_x, "y": resolved_y},
        "relative_offset": {"dx": offset_dx, "dy": offset_dy},
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
    }
    if selected_text is not None:
        meta["selected_text"] = selected_text
    if disambiguation_meta:
        meta.update(disambiguation_meta)
    return resolved_x, resolved_y, meta


async def resolve_mouse_point(
    instruction: str,
    *,
    nearby_objects: list[str] | None = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[int, int, dict[str, Any]]:
    """
    Capture selected monitor(s), build YOLO+OCR candidates, filter and pick via LLM.

    ``nearby_objects`` are optional landmark labels for spatial disambiguation.
    They are merged with any （附近有…） labels parsed from ``instruction``.

    Returns ``(global_x, global_y, metadata)`` in virtual-desktop pixel space.
    Raises ``ValueError`` when the target is not found.
    """
    found = await find_mouse_point(
        instruction,
        nearby_objects=nearby_objects,
        yolo_conf_threshold=yolo_conf_threshold,
    )
    if found is None:
        raise ValueError("No mouse target matched the instruction on selected monitor(s).")
    return found
