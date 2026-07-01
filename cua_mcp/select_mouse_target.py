"""
Unified mouse target selection: YOLO (text, element, input, scrollbar) + OCR + LLM filter/pick.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from cua_mcp.geometry import clip_box
from cua_mcp.icon_map import (
    describe_text_icons,
    is_pua_char,
    is_unknown_icon_record,
    text_has_pua,
)
from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr
from cua_mcp.select_ui_element import (
    UiDetection,
    _TEXT_FILTER_JSON_SCHEMA,
    _format_ui_candidates_text,
    _parse_keep_indices_from_llm,
    _select_center_with_ollama,
    _sort_detections_reading_order,
)
from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    MOUSE_TARGET_CLASS_IDS,
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


def _should_skip_ocr_text_candidate(text_value: str, class_id: int) -> bool:
    """
    Skip OCR rows whose only PUA glyphs have no ``icon_map`` label.

    Plain text and element boxes with no OCR text are kept.
    """
    if not text_value:
        return False
    if not _text_is_pua_only(text_value):
        return False
    if _known_icons_for_text(text_value):
        return False
    return True


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
        if _should_skip_ocr_text_candidate(text_value, cls_id):
            continue
        candidates.append(_detection_from_bbox(bbox, cls_id, text=text_value or None))

    _log_info(
        "move_mouse vision profile "
        f"yolo_s={yolo_elapsed:.3f} ocr_s={ocr_elapsed:.3f} "
        f"total_s={time.perf_counter() - vision_started:.3f} "
        f"yolo_boxes={len(xyxy)} ocr_boxes={len(ocr_boxes)} candidates={len(candidates)}"
    )

    return candidates


async def _filter_mouse_candidates(
    detections: list[UiDetection],
    instruction: str,
) -> list[UiDetection]:
    """Ask Ollama which candidates match the instruction; on failure, keep all."""
    if not detections:
        return []

    candidates_text = _format_ui_candidates_text(detections)
    prompt = get_prompt("mouse_target_filter").format(
        instruction=instruction,
        candidates_text=candidates_text,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        keep_indices = await request_json_with_retry(
            messages=messages,
            response_schema=_TEXT_FILTER_JSON_SCHEMA,
            parse_reply=lambda raw: _parse_keep_indices_from_llm(raw, len(detections)),
            retry_instruction=get_prompt("mouse_target_filter_retry"),
            log_info=lambda m: _run_manager().log_info(f"_filter_mouse_candidates: {m}"),
        )
    except ValueError as retry_exc:
        _run_manager().log_info(f"_filter_mouse_candidates: fallback keep-all ({retry_exc})")
        return detections

    if not keep_indices:
        return []
    return [detections[i] for i in keep_indices]


async def resolve_mouse_point(
    instruction: str,
    *,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[int, int, dict[str, Any]]:
    """
    Capture selected monitor(s), build YOLO+OCR candidates, filter and pick via LLM.

    Returns ``(global_x, global_y, metadata)`` in virtual-desktop pixel space.
    """
    instruction_text = (instruction or "").strip()
    if not instruction_text:
        raise ValueError("instruction must be non-empty")

    paths = _run_manager().require_paths()
    monitor_indices = selected_eye_monitor_indices()
    stamp = ts_name()
    image_paths: list[str] = []
    all_detections: list[UiDetection] = []

    _log_info(f"move_mouse resolve monitors={monitor_indices}")
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

        local_candidates = _build_candidates_from_bgr(
            bgr,
            yolo_conf_threshold=yolo_conf_threshold,
        )
        left, top = active_monitor_offset(monitor_index)
        all_detections.extend(_offset_detection(d, left, top) for d in local_candidates)

    detections = _sort_detections_reading_order(all_detections)
    _log_info(f"move_mouse yolo_candidates={len(detections)}")

    if not detections:
        raise ValueError("No YOLO candidates found on selected monitor(s).")

    filtered = await _filter_mouse_candidates(detections, instruction_text)
    _log_info(f"move_mouse after_filter={len(filtered)}")

    if not filtered:
        raise ValueError("No candidates matched the instruction after LLM filtering.")

    if len(filtered) == 1:
        idx = 0
        chosen = filtered[0]
        _log_info("move_mouse: single candidate after filter; skipping Ollama pick")
    else:
        pool_idx = await _select_center_with_ollama(
            instruction_text,
            filtered,
            image_paths,
        )
        idx = pool_idx
        chosen = filtered[pool_idx]
        _log_info(
            f"move_mouse: Ollama picked index={pool_idx} center=[{chosen.cx},{chosen.cy}]"
        )

    image_path = image_paths[0] if image_paths else ""
    meta: dict[str, Any] = {
        "selected_index": idx,
        "class_name": chosen.class_name,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
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
    return chosen.cx, chosen.cy, meta
