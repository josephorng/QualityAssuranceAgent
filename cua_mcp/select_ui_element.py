"""
UI element detection with YOLO (`cua_mcp/best.onnx`) + Ollama index selection.

Moves the cursor to the center of a detected non-text UI region; coordinate mapping
matches ``coordinate_selection._resolve_point`` (screenshot pixels → global screen).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_NAMES,
    run_best_onnx_end2end,
)
from cua_mcp.select_text import _to_global_coordinate
from cua_mcp.read_screen_text.ocr_image import get_coordinates_from_path, get_text_boxes_from_path
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, get_run_state_manager, ts_name
from src.common.settings import load_settings
from src.common.io_utils import write_json
from src.eye.capture import capture_active_monitor_to_file

settings = load_settings()


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
        "ui_icon_description": {"type": "string"},
        "location_description": {"type": "string"},
        "ui_shape_description": {"type": "string"},
    },
    "required": ["need_text_anchor", "ui_icon_description", "location_description"],
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
_CONFIRM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"confirmed": {"type": "boolean"}},
    "required": ["confirmed"],
}
# Max labeled icon crops attached to a single LLM request.
_MAX_ICON_CROPS_PER_REQUEST: int = 8


@dataclass(frozen=True)
class UiDetection:
    bbox: tuple[int, int, int, int]  # x, y, w, h in image pixels
    cx: int
    cy: int
    class_id: int
    class_name: str
    text: str | None = None


def _log_info(text: str) -> None:
    try:
        _run_manager().log_info(text)
    except RuntimeError:
        pass


def _llm_text_to_json_object_string(raw: str) -> str:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_index_from_llm(raw: str, num_candidates: int) -> int:
    """Parse the picker LLM reply; returns the chosen candidate index (0-based)."""
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            'Ollama UI picker returned empty content; expected {"index": <int>}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
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


def _parse_confirmed_from_llm(raw: str) -> bool:
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            'Ollama bbox confirm returned empty content; expected {"confirmed": bool}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "confirmed" not in out:
        raise ValueError(f'must include "confirmed"; preview={preview!r}')
    return bool(out["confirmed"])


def _parse_instruction_analysis_from_llm(raw: str) -> tuple[bool, str, str, str]:
    """Parse Ollama reply: text-anchor flag plus icon, location, and optional shape strings."""
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            "Ollama instruction analysis returned empty content; expected "
            '{"need_text_anchor": bool, "ui_icon_description": str, "location_description": str, '
            '"ui_shape_description": str (optional)}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict):
        raise ValueError(f"expected object; preview={preview!r}")
    if (
        "need_text_anchor" not in out
        or "ui_icon_description" not in out
        or "location_description" not in out
    ):
        raise ValueError(
            f'must include "need_text_anchor", "ui_icon_description", and '
            f'"location_description"; preview={preview!r}'
        )
    need = bool(out["need_text_anchor"])
    icon = str(out["ui_icon_description"] or "").strip()
    loc = str(out["location_description"] or "").strip()
    shape = str(out.get("ui_shape_description") or "").strip()
    if not icon and not loc:
        raise ValueError(f"both descriptions empty; preview={preview!r}")
    return need, icon, loc, shape


def _parse_keep_indices_from_llm(raw: str, max_len: int) -> list[int]:
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            'Ollama text filter returned empty content; expected {"keep_indices": [int, ...]}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "keep_indices" not in out:
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


def _run_ui_onnx_inference(
    bgr: np.ndarray,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess BGR (640, RGB CHW, /255), run UI YOLOv26 ONNX (end2end), return
    ``(xyxy, scores, class_ids)`` in original image pixels. Only ``element`` class detections are kept.
    """
    return run_best_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_ELEMENT},
        conf_threshold=conf_threshold,
        on_session_created=lambda p: _log_info(f"UI YOLO ONNX initializing model_path={p}"),
    )


def _clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _predict_ui_elements_raw(bgr: np.ndarray) -> list[UiDetection]:
    h, w = bgr.shape[:2]
    try:
        xyxy, _scores, class_ids = _run_ui_onnx_inference(bgr)
    except Exception as exc:
        _log_info(f"UI YOLO ONNX predict failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"UI YOLO predict failed: {exc}") from exc

    if xyxy.size == 0:
        return []

    out: list[UiDetection] = []
    for row, cls_id in zip(xyxy, class_ids, strict=True):
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        bw, bh = max(1, x2i - x1i), max(1, y2i - y1i)
        bx, by, bw, bh = _clip_box(x1i, y1i, bw, bh, w, h)
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


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = min(ax2, bx2) - max(ax, bx)
    inter_h = min(ay2, by2) - max(ay, by)
    return inter_w > 0 and inter_h > 0


def _area(box: tuple[int, int, int, int]) -> int:
    return box[2] * box[3]


def _keep_smallest_non_overlapping(detections: list[UiDetection]) -> list[UiDetection]:
    """
    If boxes overlap, keep only the smallest-area one among conflicting candidates.
    """
    if len(detections) <= 1:
        return detections

    # Smallest first so larger overlapping boxes are naturally dropped.
    ordered = sorted(detections, key=lambda d: _area(d.bbox))
    kept: list[UiDetection] = []
    for d in ordered:
        if any(_boxes_overlap(d.bbox, k.bbox) for k in kept):
            continue
        kept.append(d)
    return kept


def _remove_text_overlapping_detections(
    detections: list[UiDetection],
    text_boxes: list[tuple[int, int, int, int]],
) -> list[UiDetection]:
    if not detections or not text_boxes:
        return detections
    return [
        d for d in detections if not any(_boxes_overlap(d.bbox, tbox) for tbox in text_boxes)
    ]


def _persist_ui_result(
    image_path: str,
    raw: list[UiDetection],
    pruned: list[UiDetection],
    text_boxes: list[tuple[int, int, int, int]],
    final: list[UiDetection],
) -> None:
    paths = _run_manager().require_paths()
    out_path = paths.yolo_ui_dir / Path(image_path).with_suffix(".json").name
    result_image_path: Path | None = (
        paths.yolo_ui_dir / f"{Path(image_path).stem}_result{Path(image_path).suffix}"
    )
    img = cv2.imread(image_path)
    if img is not None:
        rendered = img.copy()
        for d in final:
            x, y, w, h = d.bbox
            cv2.rectangle(rendered, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(result_image_path), rendered)
    else:
        result_image_path = None
    write_json(
        out_path,
        {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "result_image_path": str(result_image_path) if result_image_path is not None else "",
            "result_image_name": result_image_path.name if result_image_path is not None else "",
            "text_boxes": [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in text_boxes],
            "counts": {
                "raw": len(raw),
                "pruned_smallest_overlap": len(pruned),
                "after_text_box_subtract": len(final),
            },
            "detections": [
                {
                    "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
                    "center": {"x": d.cx, "y": d.cy},
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
                for d in final
            ],
        },
    )
    _log_info(f"UI YOLO result persisted path={out_path}")
    if result_image_path is not None:
        _log_info(f"UI YOLO boxed image persisted path={result_image_path}")


def _sort_detections_reading_order(detections: list[UiDetection]) -> list[UiDetection]:
    """Top-to-bottom, then left-to-right (same spirit as OCR reading order)."""
    if not detections:
        return []
    items = sorted(detections, key=lambda d: (d.cy, d.cx))
    mean_h = sum(d.bbox[3] for d in detections) / len(detections)
    tol = max(10.0, mean_h * 0.5)
    rows: list[list[UiDetection]] = []
    row: list[UiDetection] = []
    row_y0: float | None = None
    for d in items:
        cy = float(d.cy)
        if row_y0 is None:
            row = [d]
            row_y0 = cy
            continue
        if abs(cy - row_y0) <= tol:
            row.append(d)
        else:
            rows.append(sorted(row, key=lambda x: x.cx))
            row = [d]
            row_y0 = cy
    if row:
        rows.append(sorted(row, key=lambda x: x.cx))
    ordered: list[UiDetection] = []
    for r in rows:
        ordered.extend(r)
    return ordered


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split ``items`` into consecutive sub-lists of length ``size``.

    The last chunk may be shorter. ``size`` is clamped to at least 1.
    """
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _format_text_candidates_text(detections: list[UiDetection]) -> str:
    lines: list[str] = []
    for i, d in enumerate(detections):
        text = f" text={d.text!r}" if d.text else ""
        _bx, _by, bw, bh = d.bbox
        lines.append(f"[index {i}] center=[{d.cx},{d.cy}] w={bw} h={bh}{text}")
    return "\n".join(lines)


def _format_ui_candidates_text(detections: list[UiDetection]) -> str:
    lines: list[str] = []
    for i, d in enumerate(detections):
        _bx, _by, bw, bh = d.bbox
        lines.append(f"[{i}] center=[{d.cx},{d.cy}] w={bw} h={bh}")
    return "\n".join(lines)


def _write_bbox_crop(image_path: str, bbox: tuple[int, int, int, int]) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"could not read image for bbox crop: {image_path}")
    x, y, w, h = bbox
    ih, iw = img.shape[:2]
    x, y, w, h = _clip_box(x, y, w, h, iw, ih)
    crop = img[y : y + h, x : x + w]
    paths = _run_manager().require_paths()
    out = paths.yolo_ui_dir / f"{Path(image_path).stem}_bbox_confirm_{ts_name()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    return str(out.resolve())


def _write_labeled_bbox_crop(
    image_path: str,
    bbox: tuple[int, int, int, int],
    label: str,
    *,
    pad_ratio: float = 0.25,
    min_side: int = 96,
) -> str:
    """Write a labeled crop of ``bbox`` with ``label`` burned into a top header.

    The crop is slightly padded to keep some surrounding context, upscaled when
    small so both the icon and the stamped label are legible to a vision LLM,
    and prefixed with a black header strip containing ``label`` (e.g. ``"[7]"``).
    This lets the model read the index directly from pixels instead of having to
    track image ordering or copy numbers out of a text list.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"could not read image for labeled bbox crop: {image_path}")
    ih, iw = img.shape[:2]
    x, y, w, h = bbox
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))
    x, y, w, h = _clip_box(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y, iw, ih)
    crop = img[y : y + h, x : x + w].copy()

    ch, cw = crop.shape[:2]
    scale = max(1.0, float(min_side) / float(max(1, min(ch, cw))))
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            (int(round(cw * scale)), int(round(ch * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    ch2, cw2 = crop.shape[:2]
    header_h = max(24, ch2 // 5)
    header = np.zeros((header_h, cw2, 3), dtype=crop.dtype)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, header_h / 32.0)
    thickness = max(1, int(round(font_scale * 1.5)))
    (text_w, text_h), _baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = max(4, (cw2 - text_w) // 2)
    text_y = (header_h + text_h) // 2
    cv2.putText(
        header,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    labeled = np.vstack([header, crop])

    paths = _run_manager().require_paths()
    safe_label = label.strip("[]").replace("/", "_").replace("\\", "_") or "x"
    out = paths.yolo_ui_dir / f"{Path(image_path).stem}_pick_{safe_label}_{ts_name()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), labeled)
    return str(out.resolve())


async def _confirm_selection_bbox_with_ollama(instruction: str, crop_image_path: str) -> bool:
    prompt = (
        "A UI candidate was chosen from a full screenshot. The attached image is a tight crop "
        "of that candidate's bounding box only.\n"
        "Does this crop clearly match what the user instruction asks to select or interact with?\n"
        'Return JSON only: {"confirmed": true|false}.\n'
        "Use true only if the crop is clearly the intended target; false if it is wrong, "
        "ambiguous, or mostly empty/irrelevant.\n\n"
        f"Instruction: {instruction}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": prompt, "images": [crop_image_path]},
    ]
    try:
        reply = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_CONFIRM_JSON_SCHEMA,
        )
        return _parse_confirmed_from_llm(reply.content)
    except ValueError as exc:
        _run_manager().log_info(f"_confirm_selection_bbox_with_ollama: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"confirmed": true|false}. '
            "No text before or after the JSON.\n"
        )
        reply = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
        )
        return _parse_confirmed_from_llm(reply.content)


async def _analyze_instruction(instruction: str) -> tuple[bool, str, str, str]:
    """Classify text-anchor need and split instruction for icon vs. location in one Ollama call.

    Returns ``(need_text_anchor, ui_icon_description, location_description, ui_shape_description)``.
    ``ui_shape_description`` may be empty when the model omits it or no shape hint applies.

    On parse failure after retry, or on transport errors, returns ``(True, text, text, "")``
    so downstream steps still run (conservative text-anchor path, full instruction
    for icon/location prompts).
    """
    text = (instruction or "").strip()
    if not text:
        return False, "", "", ""

    prompt = get_prompt("ui_instruction_icon_location_extract").replace("{instruction}", text)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        reply = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_INSTRUCTION_ANALYSIS_JSON_SCHEMA,
        )
        need, icon_llm, loc_llm, shape_llm = _parse_instruction_analysis_from_llm(reply.content)
    except ValueError as exc:
        _log_info(f"_need_text_anchors: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"need_text_anchor": true|false, '
            '"ui_icon_description": "...", "location_description": "...", '
            '"ui_shape_description": "..."}. '
            "need_text_anchor: true for visible words/labels/on-screen text; false for "
            "mostly non-text targets (icon, toggle, gear, unlabeled control). "
            "location_description: detailed positional language for picking among "
            "candidates, or empty when there is no spatial clue. "
            "ui_shape_description: optional short shape/size hint for the control, or empty. "
            "No text before or after the JSON.\n"
        )
        try:
            reply = await get_llm_client().chat_messages(
                settings.brain_lm,
                messages=messages,
                tools=[],
                response_format="json",
            )
            need, icon_llm, loc_llm, shape_llm = _parse_instruction_analysis_from_llm(
                reply.content
            )
        except ValueError as retry_exc:
            _log_info(f"_need_text_anchors: fallback full instruction ({retry_exc})")
            return True, text, text, ""
    except Exception as exc:
        _log_info(
            f"_need_text_anchors: fallback full instruction ({type(exc).__name__}: {exc})"
        )
        return True, text, text, ""

    icon_out = (icon_llm.strip() if icon_llm else "") or text
    loc_out = loc_llm.strip()
    shape_out = shape_llm.strip()
    return need, icon_out, loc_out, shape_out


async def _filter_text_detections(
    text_detections: list[UiDetection],
    instruction: str,
) -> list[UiDetection]:
    if not text_detections:
        return []

    candidates_text = _format_text_candidates_text(text_detections)
    prompt = (
        "Select ONLY text candidates that match the user instruction.\n"
        'Return JSON only: {"keep_indices": [<int>, ...]}.\n'
        "Use indices from the Candidates list. Keep an empty list when none match.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Candidates:\n{candidates_text}"
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        reply = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_TEXT_FILTER_JSON_SCHEMA,
        )
        keep_indices = _parse_keep_indices_from_llm(reply.content, len(text_detections))
    except ValueError as exc:
        _run_manager().log_info(f"_filter_text_detections: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"keep_indices": [<integer>, ...]}. '
            "No text before or after the JSON.\n"
        )
        try:
            reply = await get_llm_client().chat_messages(
                settings.brain_lm,
                messages=messages,
                tools=[],
                response_format="json",
            )
            keep_indices = _parse_keep_indices_from_llm(reply.content, len(text_detections))
        except ValueError as retry_exc:
            _run_manager().log_info(f"_filter_text_detections: fallback keep-all ({retry_exc})")
            return text_detections

    if not keep_indices:
        return []
    return [text_detections[i] for i in keep_indices]


async def _filter_icon_detections_with_ollama(
    instruction: str,
    detections: list[UiDetection],
    image_path: str,
) -> list[int]:
    """Filter candidates by icon crops only; returns kept local indices."""
    if not detections:
        return []

    labeled_crops: list[tuple[int, str]] = []
    for i, d in enumerate(detections):
        try:
            labeled_crops.append((i, _write_labeled_bbox_crop(image_path, d.bbox, f"[{i}]")))
        except (RuntimeError, OSError, cv2.error) as exc:
            _log_info(
                f"_filter_icon_detections_with_ollama: failed to write labeled crop "
                f"for index {i}: {exc}"
            )

    if not labeled_crops:
        _log_info(
            "_filter_icon_detections_with_ollama: no labeled crops written; keep all"
        )
        return list(range(len(detections)))

    keep_indices_set: set[int] = set()
    batch_size = max(1, _MAX_ICON_CROPS_PER_REQUEST)
    crop_groups = _chunked(labeled_crops, batch_size)
    _log_info(
        f"_filter_icon_detections_with_ollama: candidates={len(detections)} "
        f"written_crops={len(labeled_crops)} batches={len(crop_groups)} "
        f"batch_size_cap={batch_size}"
    )
    try:
        for batch_no, crop_group in enumerate(crop_groups, start=1):
            group_indices = [orig_idx for orig_idx, _path in crop_group]
            prompt = get_prompt("ui_icon_filter").format(
                instruction=instruction,
                candidate_count=len(group_indices),
            )
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [path for _orig_idx, path in crop_group],
                },
            ]
            try:
                reply = await get_llm_client().chat_messages(
                    settings.brain_lm,
                    messages=messages,
                    tools=[],
                    response_format=_TEXT_FILTER_JSON_SCHEMA,
                    append_image_sizes=False,
                )
                keep_indices = _parse_keep_indices_from_llm(reply.content, len(detections))
            except ValueError as exc:
                _log_info(
                    f"_filter_icon_detections_with_ollama: batch={batch_no} retry ({exc})"
                )
                messages[0]["content"] += (
                    '\nReply with ONLY: {"keep_indices": [<integer>, ...]}. '
                    "Use only indices visible in image headers [i]. "
                    "No text before or after the JSON.\n"
                )
                reply = await get_llm_client().chat_messages(
                    settings.brain_lm,
                    messages=messages,
                    tools=[],
                    response_format="json",
                    append_image_sizes=False,
                )
                keep_indices = _parse_keep_indices_from_llm(reply.content, len(detections))
            for idx in keep_indices:
                if idx in group_indices:
                    keep_indices_set.add(idx)
    finally:
        for _idx, path in labeled_crops:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    if not keep_indices_set:
        return []
    return sorted(keep_indices_set)


async def _select_center_with_ollama(
    instruction: str,
    detections: list[UiDetection],
    image_path: str,
    ui_shape_description: str = "",
) -> int:
    """Ask Ollama for the best candidate index; with thinking models, runs a second turn using ``thinking`` from the first reply."""
    if not detections:
        raise ValueError("no candidates to pick from")
    candidates_text = _format_ui_candidates_text(detections)
    base_instructions = get_prompt("ui_element_selection").format(
        instruction=instruction,
        candidates_text=candidates_text,
    )
    screenshot_size_text = "unknown"
    img = cv2.imread(image_path)
    if img is not None:
        img_h, img_w = img.shape[:2]
        screenshot_size_text = f"{img_w}x{img_h}"
    prompt = (
        f"{base_instructions}\n\n"
        f"Screenshot size: {screenshot_size_text} (width x height pixels)."
    )
    shape_hint = (ui_shape_description or "").strip()
    if shape_hint:
        prompt += (
            "\n\nTarget shape/size hint (from instruction analysis; use with location "
            f"to pick the best-matching candidate bbox proportions):\n{shape_hint}"
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    n = len(detections)
    reply1 = await get_llm_client().chat_messages(
        settings.brain_lm,
        messages=messages,
        tools=[],
        response_format=_INDEX_JSON_SCHEMA,
        think=True,
    )
    thinking = (getattr(reply1, "thinking", None) or "").strip()
    pool_idx: int | None = None

    if thinking:
        refine = (
            "Prior reasoning: "
            f"{thinking}\n\n"
            "Using your prior reasoning in context and the Candidates list below, output your "
            f"final choice as JSON only: a single object with key \"index\" (integer 0..{n - 1}). "
            "No markdown, no explanation.\n\n"
        )
        messages2 = [
            {"role": "user", "content": refine},
        ]
        reply2 = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages2,
            tools=[],
            response_format=_INDEX_JSON_SCHEMA,
        )
        try:
            pool_idx = _parse_index_from_llm(reply2.content, n)
        except ValueError as exc:
            _log_info(
                f"_select_center_with_ollama: thinking-refine parse failed ({exc}); "
                "trying first reply JSON"
            )
            try:
                pool_idx = _parse_index_from_llm(reply1.content, n)
            except ValueError:
                pool_idx = None
    else:
        try:
            pool_idx = _parse_index_from_llm(reply1.content, n)
        except ValueError:
            pool_idx = None

    if pool_idx is not None:
        return pool_idx

    _run_manager().log_info("_select_center_with_ollama: retry (invalid or missing index)")
    messages[0]["content"] += (
        '\nReply with ONLY: {"index": <integer>} — the [index] from the Candidates '
        "list row that best matches the location instruction (0-based). "
        "No other keys. No text before or after the JSON.\n"
    )
    reply = await get_llm_client().chat_messages(
        settings.brain_lm,
        messages=messages,
        tools=[],
        response_format="json",
    )
    return _parse_index_from_llm(reply.content, n)


async def resolve_ui_element_point(instruction: str) -> tuple[int, int, dict[str, Any]]:
    """
    Capture screen, run UI YOLO, pick one detection via Ollama, return global (x, y) and metadata.
    """
    text = (instruction or "").strip()
    if not text:
        raise ValueError("instruction must be non-empty")

    paths = _run_manager().require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ui_dir / name
    capture_active_monitor_to_file(out)
    image_path = str(out.resolve())

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"could not read capture image: {image_path}")

    raw = _predict_ui_elements_raw(bgr)
    pruned = _keep_smallest_non_overlapping(raw)
    need_text_anchor, icon_desc, loc_desc, shape_desc = await _analyze_instruction(text)
    _log_info(f"_resolve_ui_element: need_text_anchor={need_text_anchor}")
    if need_text_anchor:
        (_off_x, _off_y), regions = get_coordinates_from_path(image_path)
        text_detections: list[UiDetection] = []
        text_boxes: list[tuple[int, int, int, int]] = []
        for box, _center, preds in regions:
            x, y, w, h = box
            text_detections.append(UiDetection(bbox=(int(x), int(y), int(w), int(h)), cx=int(x) + int(w) // 2, cy=int(y) + int(h) // 2, class_id=0, class_name="text", text="".join(preds).strip()))
            text_boxes.append((int(x), int(y), int(w), int(h)))
        text_detections = await _filter_text_detections(text_detections, instruction)
        persisted_text_boxes = [d.bbox for d in text_detections]
        no_text_overlap = _remove_text_overlapping_detections(pruned, text_boxes)
        detections = _sort_detections_reading_order(no_text_overlap + text_detections)
    else:
        text_boxes = get_text_boxes_from_path(image_path)
        persisted_text_boxes = text_boxes
        no_text_overlap = _remove_text_overlapping_detections(pruned, text_boxes)
        detections = _sort_detections_reading_order(no_text_overlap)
    _persist_ui_result(
        image_path=image_path,
        raw=raw,
        pruned=pruned,
        text_boxes=persisted_text_boxes,
        final=detections,
    )

    if not detections:
        if raw:
            raise ValueError(
                "UI YOLO found only text-like regions; use mouse_move for text targets."
            )
        raise ValueError("UI YOLO detected no UI elements on the screen.")

    icon_instruction = (icon_desc or "").strip() or text
    location_instruction = (loc_desc or "").strip() or text

    candidate_pool: list[tuple[int, UiDetection]] = list(enumerate(detections))
    if len(candidate_pool) > 1:
        keep_local_indices = await _filter_icon_detections_with_ollama(
            icon_instruction, [d for _orig, d in candidate_pool], image_path
        )
        if not keep_local_indices:
            raise ValueError(
                "No icon candidate matched the instruction in crop-only filtering."
            )
        candidate_pool = [candidate_pool[i] for i in keep_local_indices]
        _log_info(
            f"_resolve_ui_element: icon filter kept {len(candidate_pool)} candidates, result: {candidate_pool}"
        )

    if len(candidate_pool) == 1:
        orig_idx, chosen = candidate_pool[0]
        idx = orig_idx
        _log_info("_resolve_ui_element: single candidate; skipping Ollama center pick")
    else:
        filtered_detections = [d for _orig, d in candidate_pool]
        pool_idx = await _select_center_with_ollama(
            location_instruction,
            filtered_detections,
            image_path,
            ui_shape_description=shape_desc,
        )
        orig_idx, chosen = candidate_pool[pool_idx]
        idx = orig_idx
        _log_info(
            f"_resolve_ui_element: Ollama returned index={pool_idx} "
            f"(chosen center=[{chosen.cx},{chosen.cy}])"
        )

    gx, gy = _to_global_coordinate(chosen.cx, chosen.cy)
    meta: dict[str, Any] = {
        "selected_index": idx,
        "class_name": chosen.class_name,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "screenshot_path": image_path,
    }
    return gx, gy, meta
