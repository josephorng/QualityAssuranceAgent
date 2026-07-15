"""
Unified mouse target selection: YOLO (text, element, input, scrollbar) + OCR + LLM filter/pick.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import time
from typing import Any

import cv2
import numpy as np

from cua_mcp.geometry import clip_box, iou_xywh
from cua_mcp.instruction_offset import parse_relative_pixel_offset
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
    _MOUSE_FILTER_JSON_SCHEMA,
    _assign_exclusive_neighbors_to_anchors,
    _detection_anchor_label,
    _format_ui_candidates_text,
    _parse_anchor_nearby_indices_from_llm,
    _select_center_with_ollama,
    _sort_detections_reading_order,
)
from cua_mcp.selection_engine import request_json_with_retry
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
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, get_run_state_manager, ts_name
from src.eye.capture import active_monitor_offset, capture_monitor_to_file


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
            on_session_created=lambda p: _log_info(f"move_mouse YOLO initializing model_path={p}"),
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
    Expand LLM ``keep_indices`` with every detection that shares a text/icon label.

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


def _label_similarity(a: str, b: str) -> float:
    """Case-insensitive string similarity in ``[0, 1]``."""
    left = (a or "").strip()
    right = (b or "").strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


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


def _detection_max_similarity_to_queries(
    det: UiDetection,
    queries: list[str],
) -> float:
    """Best similarity between ``det`` and any non-blank query in ``queries``."""
    return max(
        (_detection_similarity_to_query(det, q) for q in queries if (q or "").strip()),
        default=0.0,
    )


def _prefilter_detections_by_similarity(
    detections: list[UiDetection],
    anchor: str,
    nearby: list[str],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """Keep detections whose best score vs anchor/nearby is ``>= threshold``."""
    queries = [anchor, *nearby]
    return [
        det
        for det in detections
        if _detection_max_similarity_to_queries(det, queries) >= threshold
    ]


def _best_matching_index(detections: list[UiDetection], query: str) -> int | None:
    """Index of the detection most similar to ``query``, or None when query is blank."""
    query = (query or "").strip()
    if not query or not detections:
        return None
    best_idx = 0
    best_score = _detection_similarity_to_query(detections[0], query)
    for i in range(1, len(detections)):
        score = _detection_similarity_to_query(detections[i], query)
        if score > best_score:
            best_idx = i
            best_score = score
    return best_idx


def _fallback_filter_by_similarity(
    detections: list[UiDetection],
    anchor: str,
    nearby: list[str],
) -> tuple[list[UiDetection], list[UiDetection]]:
    """Return ``(anchor_matches, nearby_matches)`` via string similarity."""
    if not detections:
        return [], []

    anchor_matches: list[UiDetection] = []
    nearby_matches: list[UiDetection] = []
    seen: set[int] = set()

    anchor_idx = _best_matching_index(detections, anchor)
    if anchor_idx is not None:
        seen.add(anchor_idx)
        anchor_matches.append(detections[anchor_idx])
        _log_info(
            "_filter_mouse_candidates: similarity fallback "
            f"query={anchor!r} index={anchor_idx} "
            f"label={_detection_anchor_label(detections[anchor_idx])!r} "
            f"score={_detection_similarity_to_query(detections[anchor_idx], anchor):.3f} "
            f"bucket=anchor"
        )

    for query in nearby:
        if not (query or "").strip():
            continue
        idx = _best_matching_index(detections, query)
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        nearby_matches.append(detections[idx])
        _log_info(
            "_filter_mouse_candidates: similarity fallback "
            f"query={query!r} index={idx} "
            f"label={_detection_anchor_label(detections[idx])!r} "
            f"score={_detection_similarity_to_query(detections[idx], query):.3f} "
            f"bucket=nearby"
        )
    return anchor_matches, nearby_matches


async def _filter_mouse_candidates(
    detections: list[UiDetection],
    anchor: str,
    nearby: list[str],
) -> tuple[list[UiDetection], list[UiDetection]]:
    """Ask Ollama which candidates match the anchor vs nearby labels.

    Candidates are first narrowed by string similarity (``>= 0.5`` vs anchor or
    any nearby label). Only that shortlist is sent to the LLM. On empty prefilter
    or LLM failure, falls back to similarity matching for each bucket.
    """
    if not detections:
        return [], []

    prefiltered = _prefilter_detections_by_similarity(detections, anchor, nearby)
    _run_manager().log_info(
        "_filter_mouse_candidates: similarity prefilter "
        f"threshold={_MOUSE_FILTER_SIMILARITY_THRESHOLD} "
        f"before={len(detections)} after={len(prefiltered)}"
    )
    if not prefiltered:
        _run_manager().log_info(
            "_filter_mouse_candidates: prefilter empty; fallback similarity match"
        )
        return _fallback_filter_by_similarity(detections, anchor, nearby)

    nearby_text = ", ".join(label.strip() for label in nearby if label.strip()) or "(none)"
    candidates_text = _format_ui_candidates_text(prefiltered, include_geometry=False)
    prompt = get_prompt("mouse_target_filter").format(
        anchor=anchor,
        nearby_text=nearby_text,
        candidates_text=candidates_text,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        anchor_indices, nearby_indices = await request_json_with_retry(
            messages=messages,
            response_schema=_MOUSE_FILTER_JSON_SCHEMA,
            parse_reply=lambda raw: _parse_anchor_nearby_indices_from_llm(
                raw, len(prefiltered)
            ),
            retry_instruction=get_prompt("mouse_target_filter_retry"),
            log_info=lambda m: _run_manager().log_info(f"_filter_mouse_candidates: {m}"),
        )
    except ValueError as retry_exc:
        _run_manager().log_info(
            f"_filter_mouse_candidates: fallback similarity match ({retry_exc})"
        )
        return _fallback_filter_by_similarity(detections, anchor, nearby)

    anchor_expanded = _expand_keep_indices_with_similar(prefiltered, anchor_indices)
    nearby_expanded = _expand_keep_indices_with_similar(prefiltered, nearby_indices)
    expanded_nearby_count = len(nearby_expanded)
    anchor_set = set(anchor_expanded)
    nearby_expanded = [i for i in nearby_expanded if i not in anchor_set]

    if (
        len(anchor_expanded) != len(anchor_indices)
        or expanded_nearby_count != len(nearby_indices)
    ):
        _run_manager().log_info(
            "_filter_mouse_candidates: expanded similar "
            f"llm_anchor={len(anchor_indices)} after_expand_anchor={len(anchor_expanded)} "
            f"llm_nearby={len(nearby_indices)} after_expand_nearby={expanded_nearby_count} "
            f"nearby_after_dedupe={len(nearby_expanded)}"
        )

    return (
        [prefiltered[i] for i in anchor_expanded],
        [prefiltered[i] for i in nearby_expanded],
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
    if not labels:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in labels:
        if not isinstance(item, str):
            continue
        label = item.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _merge_nearby_labels(*sources: list[str] | None) -> list[str]:
    """Merge nearby label lists; earlier sources win on duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for label in _normalize_nearby_labels(source):
            if label in seen:
                continue
            seen.add(label)
            merged.append(label)
    return merged


def _prefilter_anchors_by_nearby(
    anchors: list[UiDetection],
    nearby_matches: list[UiDetection],
    nearby_labels: list[str],
    *,
    threshold: float = _MOUSE_FILTER_SIMILARITY_THRESHOLD,
) -> list[UiDetection]:
    """Narrow anchors using exclusive nearby-landmark assignment.

    Prefer anchors whose assigned neighbors cover **all** ``nearby_labels``.
    If none do, keep anchors with any covered nearby label. If still empty
    (or nearby is absent), return ``anchors`` unchanged.
    """
    labels = _normalize_nearby_labels(nearby_labels)
    if not anchors or not nearby_matches or not labels:
        return anchors

    assigned = _assign_exclusive_neighbors_to_anchors(anchors, nearby_matches)
    coverage: list[set[int]] = []
    for neighbors in assigned:
        covered: set[int] = set()
        for li, label in enumerate(labels):
            if any(
                _detection_similarity_to_query(neigh, label) >= threshold
                for neigh in neighbors
            ):
                covered.add(li)
        coverage.append(covered)

    n_labels = len(labels)
    full = [anchors[i] for i, covered in enumerate(coverage) if len(covered) == n_labels]
    if full:
        return full

    partial = [anchors[i] for i, covered in enumerate(coverage) if covered]
    if partial:
        return partial

    return anchors


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
    """
    instruction_text = (instruction or "").strip()
    if not instruction_text:
        raise ValueError("instruction must be non-empty")

    # Parse anchor/offset/nearby once; filter splits anchor vs nearby; pick uses anchor only.
    anchor, offset_dx, offset_dy, nearby_from_instruction = await parse_relative_pixel_offset(
        instruction_text
    )
    nearby = _merge_nearby_labels(nearby_objects, nearby_from_instruction)

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
        raise ValueError("No YOLO candidates found on selected monitor(s).")

    anchor_matches, nearby_matches = await _filter_mouse_candidates(
        detections, anchor, nearby
    )
    _log_info(
        f"move_mouse after_filter anchor={len(anchor_matches)} "
        f"nearby={len(nearby_matches)} "
        f"anchor={anchor!r} nearby_labels={nearby!r}"
    )

    if not anchor_matches:
        raise ValueError("No anchor candidates matched the instruction after LLM filtering.")

    before_nearby = len(anchor_matches)
    anchor_matches = _prefilter_anchors_by_nearby(
        anchor_matches, nearby_matches, nearby
    )
    if len(anchor_matches) != before_nearby:
        _log_info(
            "move_mouse nearby prefilter "
            f"anchors {before_nearby} -> {len(anchor_matches)} "
            f"nearby_labels={nearby!r}"
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
            nearby_labels=nearby,
        )
        idx = pool_idx
        chosen = anchor_matches[pool_idx]
        _log_info(
            f"move_mouse: Ollama picked index={pool_idx} "
            f"text={selected_text!r} center=[{chosen.cx},{chosen.cy}]"
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
        "nearby_objects": nearby,
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
    return resolved_x, resolved_y, meta
