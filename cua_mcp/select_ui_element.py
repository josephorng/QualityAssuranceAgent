"""
UI element selection from OCR regions (PUA icon tokens) and optional text anchors.

PUA recognition in ``get_coordinates_from_image_path`` regions becomes ``ocr_icon`` candidates.
When the instruction needs a text anchor, non-PUA regions are text candidates. Ollama
disambiguates by location (and ``chinese_id`` hints in the candidate list).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    PICKER_CLASS_OCR_ICON,
    PICKER_CLASS_TEXT,
    UI_DETECTION_CLASS_IDS,
    YOLO_CLASS_NAMES,
    run_yolo_onnx_end2end,
)
from cua_mcp.geometry import clip_box, sort_by_reading_order
from cua_mcp.icon_map import (
    describe_text_icons,
    is_unknown_icon_record,
    text_has_pua,
    unknown_icon_record,
)
from cua_mcp.llm_json import parse_json_object
from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.select_text import (
    _normalize_match_key,
    _sanitize_target_text,
    _to_simplified_chinese,
)
from cua_mcp.read_screen_text.ocr_image import get_coordinates_from_selected_monitors
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, get_run_state_manager
from src.common.settings import load_settings


def _run_manager() -> RunStateManager:
    """Always resolve the current singleton (never cache): ``reset_run_state_manager`` replaces it."""
    return get_run_state_manager()


_INDEX_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"index": {"type": "integer"}},
    "required": ["index"],
}
_INSTRUCTION_ANALYSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "need_text_anchor": {"type": "boolean"},
        "location_description": {"type": "string"},
    },
    "required": ["need_text_anchor", "location_description"],
}
_TEXT_FILTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep_indices": {
            "type": "array",
            "items": {"type": "integer"},
        }
    },
    "required": ["keep_indices"],
}


@dataclass(frozen=True)
class UiDetection:
    """One clickable candidate: bbox and center in screenshot pixels plus optional OCR metadata."""

    bbox: tuple[int, int, int, int]  # x, y, w, h in image pixels
    cx: int
    cy: int
    class_id: int
    class_name: str
    text: str | None = None
    icons: list[dict[str, Any]] | None = None


def _log_info(text: str) -> None:
    """Write to the run log when run state is initialized; no-op otherwise."""
    try:
        _run_manager().log_info(text)
    except RuntimeError:
        pass


def _parse_index_from_llm(raw: str, num_candidates: int) -> int:
    """Parse the picker LLM reply; returns the chosen candidate index (0-based)."""
    out = parse_json_object(
        raw,
        empty_error='Ollama UI picker returned empty content; expected {"index": <int>}',
        decode_error_prefix="invalid JSON",
    )
    preview = (raw or "")[:240]
    if not isinstance(out, dict) or "index" not in out:
        raise ValueError(f'must include "index"; preview={preview!r}')
    try:
        idx = int(out["index"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'"index" must be an integer; got {out.get("index")!r}') from exc
    if idx < 0 or idx >= num_candidates:
        raise ValueError(
            f'"index" out of range: {idx} (valid 0..{num_candidates - 1}); preview={preview!r}'
        )
    return idx


def _parse_instruction_analysis_from_llm(raw: str) -> tuple[bool, str]:
    """Parse Ollama reply: text-anchor flag and location description."""
    out = parse_json_object(
        raw,
        empty_error=(
            "Ollama instruction analysis returned empty content; expected "
            '{"need_text_anchor": bool, "location_description": str}'
        ),
        decode_error_prefix="invalid JSON",
    )
    preview = (raw or "")[:240]
    if "need_text_anchor" not in out or "location_description" not in out:
        raise ValueError(
            f'must include "need_text_anchor" and "location_description"; preview={preview!r}'
        )
    need = bool(out["need_text_anchor"])
    loc = str(out["location_description"] or "").strip()
    return need, loc


def _parse_keep_indices_from_llm(raw: str, max_len: int) -> list[int]:
    """Parse text-filter LLM reply; return deduplicated 0-based indices in ``[0, max_len)``."""
    out = parse_json_object(
        raw,
        empty_error='Ollama text filter returned empty content; expected {"keep_indices": [int, ...]}',
        decode_error_prefix="invalid JSON",
    )
    preview = (raw or "")[:240]
    if "keep_indices" not in out:
        raise ValueError(f'must include "keep_indices"; preview={preview!r}')
    value = out["keep_indices"]
    if not isinstance(value, list):
        raise ValueError(f'"keep_indices" must be a list; preview={preview!r}')
    keep: list[int] = []
    seen: set[int] = set()
    for item in value:
        idx = int(item)
        if idx < 0 or idx >= max_len or idx in seen:
            continue
        seen.add(idx)
        keep.append(idx)
    return keep


def _bgr_compress_long_axis_to_square(bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Resize BGR to a square by scaling the longer side down to the shorter side length.

    Returns ``(square_bgr, scale_x, scale_y)`` where original coordinates map as
    ``x_orig = x_square * scale_x``, ``y_orig = y_square * scale_y``.
    """
    h0, w0 = bgr.shape[:2]
    side = min(h0, w0)
    if side <= 0:
        raise ValueError("invalid image dimensions for UI YOLO")
    if h0 == w0:
        return bgr, 1.0, 1.0
    square = cv2.resize(bgr, (side, side), interpolation=cv2.INTER_LINEAR)
    return square, float(w0) / float(side), float(h0) / float(side)


def _scale_xyxy_square_to_original(
    xyxy: np.ndarray, scale_x: float, scale_y: float
) -> np.ndarray:
    """Map YOLO ``xyxy`` from square-compressed inference space back to original image pixels."""
    if xyxy.size == 0 or (scale_x == 1.0 and scale_y == 1.0):
        return xyxy
    out = xyxy.astype(np.float32, copy=True)
    out[:, 0] *= scale_x
    out[:, 2] *= scale_x
    out[:, 1] *= scale_y
    out[:, 3] *= scale_y
    return out


def _run_ui_yolo_inference(
    bgr: np.ndarray,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Letterbox BGR to 1280 square (RGB CHW /255), run UI YOLOv26 ONNX (end2end), return
    ``(xyxy, scores, class_ids)`` in the input image's pixel space. Keeps ``element``,
    ``input``, and ``scrollbar`` detections; excludes ``text``.
    """
    return run_yolo_onnx_end2end(
        bgr,
        class_ids=set(UI_DETECTION_CLASS_IDS),
        conf_threshold=conf_threshold,
        on_session_created=lambda p: _log_info(f"UI YOLO ONNX initializing model_path={p}"),
    )


def _predict_ui_elements_yolo(
    bgr: np.ndarray,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[UiDetection]:
    """
    Run UI YOLO on BGR (square-compress non-square inputs) and return ``element``,
    ``input``, and ``scrollbar`` detections.

    Used by debug viewers; ``resolve_ui_element_point`` uses OCR PUA regions instead.
    """
    h, w = bgr.shape[:2]
    predict_bgr, scale_x, scale_y = _bgr_compress_long_axis_to_square(bgr)
    if scale_x != 1.0 or scale_y != 1.0:
        side = predict_bgr.shape[0]
        _log_info(
            f"UI YOLO: compressed {w}x{h} -> {side}x{side} before predict "
            f"(scale_x={scale_x:.4f}, scale_y={scale_y:.4f})"
        )
    try:
        xyxy, _scores, class_ids = _run_ui_yolo_inference(
            predict_bgr, conf_threshold=conf_threshold
        )
    except Exception as exc:
        _log_info(f"UI YOLO ONNX predict failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"UI YOLO predict failed: {exc}") from exc

    if xyxy.size == 0:
        return []

    xyxy = _scale_xyxy_square_to_original(xyxy, scale_x, scale_y)

    out: list[UiDetection] = []
    for row, cls_id in zip(xyxy, class_ids, strict=True):
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        bw, bh = max(1, x2i - x1i), max(1, y2i - y1i)
        bx, by, bw, bh = clip_box(x1i, y1i, bw, bh, w, h)
        cx = bx + bw // 2
        cy = by + bh // 2
        cls_id = int(cls_id)
        class_name = YOLO_CLASS_NAMES.get(cls_id, str(cls_id))
        out.append(
            UiDetection(
                bbox=(bx, by, bw, bh),
                cx=cx,
                cy=cy,
                class_id=cls_id,
                class_name=class_name,
            )
        )
    return out


# ``get_coordinates_from_image_path`` region: (bbox, center, crnn token list).
_OcrRegion = tuple[tuple[int, int, int, int], tuple[int, int], list[str]]


def _icons_for_pua_text(text_value: str) -> list[dict[str, Any]]:
    """Map PUA tokens via ``icon_map``; unmapped PUA uses ``unknown_icon``."""
    icons = describe_text_icons(text_value)
    return icons if icons else [unknown_icon_record()]


def _detection_chinese_ids(detection: UiDetection) -> list[str]:
    return [
        str(icon.get("chinese_id", "")).strip()
        for icon in (detection.icons or [])
        if icon.get("chinese_id")
    ]


def _best_chinese_id_similarity_score(
    target_candidates: list[str],
    chinese_ids: list[str],
) -> float:
    if not target_candidates or not chinese_ids:
        return 0.0
    best = 0.0
    for chinese_id in chinese_ids:
        row_key = _normalize_match_key(_to_simplified_chinese(chinese_id))
        if not row_key:
            continue
        for candidate in target_candidates:
            cand_key = _normalize_match_key(_to_simplified_chinese(candidate))
            if not cand_key:
                continue
            score = SequenceMatcher(None, cand_key, row_key).ratio()
            if score > best:
                best = score
    return best


def _filter_ui_detections_by_icon_name(
    detections: list[UiDetection],
    ui_element_name: str,
    *,
    fallback_icon_text: str = "",
) -> list[UiDetection]:
    """
    Keep detections whose icon ``chinese_id`` values best-match ``ui_element_name``.

    When no detection scores above zero, return the original list unchanged.
    """
    candidates = _sanitize_target_text(ui_element_name)
    if not candidates and fallback_icon_text:
        candidates = _sanitize_target_text(fallback_icon_text)
    if not candidates:
        return detections

    matched = [
        d
        for d in detections
        if _best_chinese_id_similarity_score(candidates, _detection_chinese_ids(d)) > 0
    ]
    return matched if matched else detections


def _ocr_regions_to_candidates(
    regions: list[_OcrRegion],
    # *,
    # need_text_anchor: bool,
) -> tuple[list[UiDetection], list[UiDetection]]:
    """
    Convert OCR regions into picker candidates.

    PUA recognition always becomes an ``ocr_icon`` candidate. When ``need_text_anchor`` is
    true, non-PUA regions with text become ``text`` candidates; otherwise non-PUA regions
    are skipped.

    Returns:
        ``(text_detections, ocr_icon_detections)``.
    """
    text_detections: list[UiDetection] = []
    ocr_icon_detections: list[UiDetection] = []

    for bbox, _center, preds in regions:
        text_value = "".join(preds).strip()
        if not text_value:
            continue

        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2

        if text_has_pua(text_value):
            icons = _icons_for_pua_text(text_value)
            if not icons or all(is_unknown_icon_record(ii) for ii in icons):
                continue
            ocr_icon_detections.append(
                UiDetection(
                    bbox=bbox,
                    cx=cx,
                    cy=cy,
                    class_id=PICKER_CLASS_OCR_ICON,
                    class_name="ocr_icon",
                    text=text_value,
                    icons=icons,
                )
            )
            continue

        # if need_text_anchor:
        text_detections.append(
            UiDetection(
                bbox=bbox,
                cx=cx,
                cy=cy,
                class_id=PICKER_CLASS_TEXT,
                class_name="text",
                text=text_value,
                icons=describe_text_icons(text_value),
            )
        )

    return text_detections, ocr_icon_detections


def _sort_detections_reading_order(detections: list[UiDetection]) -> list[UiDetection]:
    """Top-to-bottom, then left-to-right (same spirit as OCR reading order)."""
    return sort_by_reading_order(
        detections,
        center_fn=lambda d: (d.cx, d.cy),
        row_height_fn=lambda d: d.bbox[3],
        x_fn=lambda d: d.cx,
    )


def _format_text_candidates_text(detections: list[UiDetection]) -> str:
    """Format text-anchor rows for the text-filter LLM (includes ``[index N]`` labels)."""
    lines: list[str] = []
    for i, d in enumerate(detections):
        text = f" text={d.text!r}" if d.text else ""
        # chinese_ids = ",".join(
        #     ii.get("chinese_id", "") for ii in (d.icons or []) if ii.get("chinese_id")
        # )
        # icon_text = f" icons={chinese_ids}" if chinese_ids else ""
        # _bx, _by, bw, bh = d.bbox
        # lines.append(f"[index {i}] center=[{d.cx},{d.cy}] w={bw} h={bh}{text}{icon_text}")
        lines.append(f"[index {i}] {text}")
    return "\n".join(lines)


def _format_ui_candidates_text(detections: list[UiDetection]) -> str:
    """Format candidate rows for the UI element picker LLM (``[N] center=...`` per line)."""
    lines: list[str] = []
    for i, d in enumerate(detections):
        # text = f" text={d.text!r}" if d.text else ""
        chinese_ids = ",".join(
            ii.get("chinese_id", "") for ii in (d.icons or []) if ii.get("chinese_id")
        )
        icon_text = f" icons={chinese_ids}" if chinese_ids else ""
        _bx, _by, bw, bh = d.bbox
        lines.append(f"[{i}] center=[{d.cx},{d.cy}] w={bw} h={bh}{icon_text}")
    return "\n".join(lines)


# async def _analyze_instruction(instruction: str) -> tuple[bool, str]:
#     """Classify text-anchor need and extract location hint in one Ollama call.

#     Returns ``(need_text_anchor, location_description)``.

#     On parse failure after retry, or on transport errors, returns ``(True, text)``
#     so downstream steps still run (conservative text-anchor path, full instruction
#     for location prompts).
#     """
#     text = (instruction or "").strip()
#     if not text:
#         return False, ""

#     prompt = get_prompt("ui_instruction_icon_location_extract").replace("{instruction}", text)
#     messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
#     try:
#         need, loc_llm = await request_json_with_retry(
#             messages=messages,
#             response_schema=_INSTRUCTION_ANALYSIS_JSON_SCHEMA,
#             parse_reply=_parse_instruction_analysis_from_llm,
#             retry_instruction=get_prompt("ui_instruction_icon_location_extract_retry"),
#             log_info=lambda m: _log_info(f"_need_text_anchors: {m}"),
#         )
#     except ValueError as retry_exc:
#         _log_info(f"_need_text_anchors: fallback full instruction ({retry_exc})")
#         return True, text
#     except Exception as exc:
#         _log_info(
#             f"_need_text_anchors: fallback full instruction ({type(exc).__name__}: {exc})"
#         )
#         return True, text

#     return need, loc_llm.strip()


async def _filter_text_detections(
    text_detections: list[UiDetection],
    instruction: str,
) -> list[UiDetection]:
    """Ask Ollama which text candidates match the instruction; on failure, keep all."""
    if not text_detections:
        return []

    candidates_text = _format_text_candidates_text(text_detections)
    prompt = get_prompt("ui_text_filter").format(
        instruction=instruction,
        candidates_text=candidates_text,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        keep_indices = await request_json_with_retry(
            messages=messages,
            response_schema=_TEXT_FILTER_JSON_SCHEMA,
            parse_reply=lambda raw: _parse_keep_indices_from_llm(raw, len(text_detections)),
            retry_instruction=get_prompt("ui_text_filter_retry"),
            log_info=lambda m: _run_manager().log_info(f"_filter_text_detections: {m}"),
        )
    except ValueError as retry_exc:
        _run_manager().log_info(f"_filter_text_detections: fallback keep-all ({retry_exc})")
        return text_detections

    if not keep_indices:
        return []
    return [text_detections[i] for i in keep_indices]


async def _select_center_with_ollama(
    instruction: str,
    detections: list[UiDetection],
    image_paths: list[str],
) -> int:
    """
    Ask Ollama for the best candidate index (0-based into ``detections``).

    Falls back to :func:`request_json_with_retry` if the first reply cannot be parsed.
    """
    if not detections:
        raise ValueError("no candidates to pick from")
    candidates_text = _format_ui_candidates_text(detections)
    screenshot_sizes: list[str] = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
            screenshot_sizes.append(f"{img_w}x{img_h}")
    screenshot_size_text = ", ".join(screenshot_sizes) if screenshot_sizes else "unknown"
    prompt = get_prompt("ui_element_selection").format(
        instruction=instruction,
        candidates_text=candidates_text,
        screenshot_sizes=screenshot_size_text,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "images": image_paths,
        },
    ]
    n = len(detections)
    reply1 = await get_llm_client().chat_messages(
        load_settings().brain_lm,
        messages=messages,
        tools=[],
        response_format=_INDEX_JSON_SCHEMA,
        think=True,
    )
    try:
        pool_idx = _parse_index_from_llm(reply1.content, n)
    except ValueError:
        pool_idx = None

    if pool_idx is not None:
        return pool_idx

    return await request_json_with_retry(
        messages=messages,
        response_schema=_INDEX_JSON_SCHEMA,
        parse_reply=lambda raw: _parse_index_from_llm(raw, n),
        retry_instruction=get_prompt("ui_element_selection_retry"),
        log_info=lambda m: _run_manager().log_info(f"_select_center_with_ollama: {m}"),
    )


async def resolve_ui_element_point(
    instruction: str,
    *,
    ui_element_name: str = "",
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[int, int, dict[str, Any]]:
    """
    Capture selected monitor(s), build UI candidates from OCR, and return a global click point.

    Candidates are PUA ``ocr_icon`` regions; when the instruction needs a text anchor, matching
    non-PUA text regions are included. Disambiguation uses Ollama on location hints from
    :func:`_analyze_instruction`.

    Args:
        instruction: Natural-language UI target (non-empty).
        ui_element_name: Optional label to pre-filter icon candidates by ``chinese_id`` similarity.
        yolo_conf_threshold: Confidence threshold passed to :func:`get_coordinates_from_selected_monitors`.

    Returns:
        ``(global_x, global_y, metadata)`` with bbox, class, icons, and screenshot path(s).

    Raises:
        ValueError: Empty instruction or no candidates after OCR/LLM filtering.
    """
    instruction_text = (instruction or "").strip()
    if not instruction_text:
        raise ValueError("instruction must be non-empty")

    # get ocr detections
    regions, image_paths = get_coordinates_from_selected_monitors(
        yolo_conf_threshold=yolo_conf_threshold,
    )
    image_path = image_paths[0] if image_paths else ""
    text_detections, ui_element_detections = _ocr_regions_to_candidates(
        regions,
    )
    
    # filter
    ui_element_detections = _filter_ui_detections_by_icon_name(
        ui_element_detections,
        ui_element_name,
        fallback_icon_text=instruction_text,
    )
    text_detections = await _filter_text_detections(text_detections, instruction)
    
    # sort
    detections = _sort_detections_reading_order(ui_element_detections + text_detections)
    _log_info(
        f"_resolve_ui_element: ocr_regions={len(regions)} "
        f"icon_similarity_candidates={len(ui_element_detections)} "
        f"text_candidates={len(text_detections)} "
        f"ui_element_name={ui_element_name!r}"
    )

    if not detections:
        raise ValueError("No matching text anchors or PUA icon regions for this instruction.")

    # location_instruction = (loc_desc or "").strip() or instruction_text

    if len(detections) == 1:
        idx = 0
        chosen = detections[0]
        _log_info("_resolve_ui_element: single candidate; skipping Ollama center pick")
    else:
        pool_idx = await _select_center_with_ollama(
            instruction_text,
            detections,
            image_paths,
        )
        idx = pool_idx
        chosen = detections[pool_idx]
        _log_info(
            f"_resolve_ui_element: Ollama returned index={pool_idx} "
            f"(chosen center=[{chosen.cx},{chosen.cy}])"
        )

    meta: dict[str, Any] = {
        "selected_index": idx,
        "class_name": chosen.class_name,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "screenshot_path": image_path,
        "screenshot_paths": image_paths,
        "target_kind": "ui_element",
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
