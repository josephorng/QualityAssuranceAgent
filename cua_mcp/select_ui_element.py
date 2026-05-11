"""
UI element detection with YOLO (`get_UI/model.pt`) + Ollama index selection.

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

from cua_mcp.select_text import _to_global_coordinate
from cua_mcp.read_screen_text.ocr_image import get_coordinates_from_path, get_text_boxes_from_path
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager, ts_name
from src.common.settings import load_settings
from src.common.io_utils import write_json
from src.eye.capture import capture_active_monitor_to_file

_PACKAGE_DIR = Path(__file__).resolve().parent
_UI_MODEL_PATH = _PACKAGE_DIR / "get_UI" / "model.pt"

_YOLO_UI_MODEL: object | None = None

settings = load_settings()
logger = get_run_state_manager()
_ollama = OllamaClient(settings.ollama_host)

_INDEX_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"index": {"type": "integer"}},
    "required": ["index"],
}
_TEXT_ANCHOR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"need_text_anchor": {"type": "boolean"}},
    "required": ["need_text_anchor"],
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
    bbox: tuple[int, int, int, int]  # x, y, w, h in image pixels
    cx: int
    cy: int
    class_id: int
    class_name: str
    text: str | None = None


def _log_info(text: str) -> None:
    try:
        logger.log_info(text)
    except RuntimeError:
        pass


def _llm_text_to_json_object_string(raw: str) -> str:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_index_from_llm(raw: str) -> int:
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError('Ollama UI picker returned empty content; expected {"index": int}')
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "index" not in out:
        raise ValueError(f'must include "index"; preview={preview!r}')
    return int(out["index"])


def _parse_need_text_anchor_from_llm(raw: str) -> bool:
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            'Ollama anchor classifier returned empty content; expected {"need_text_anchor": bool}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "need_text_anchor" not in out:
        raise ValueError(f'must include "need_text_anchor"; preview={preview!r}')
    return bool(out["need_text_anchor"])


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


def _get_ui_yolo() -> object:
    global _YOLO_UI_MODEL
    if _YOLO_UI_MODEL is None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is required for UI YOLO. Install with: pip install ultralytics"
            ) from e
        if not _UI_MODEL_PATH.is_file():
            raise FileNotFoundError(f"UI YOLO model not found: {_UI_MODEL_PATH}")
        _log_info(f"UI YOLO initializing model_path={_UI_MODEL_PATH}")
        _YOLO_UI_MODEL = YOLO(str(_UI_MODEL_PATH))
    return _YOLO_UI_MODEL


def _clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _predict_ui_elements_raw(bgr: np.ndarray) -> list[UiDetection]:
    model = _get_ui_yolo()
    h, w = bgr.shape[:2]
    m = max(h, w)
    imgsz = max(32, ((m + 31) // 32) * 32)
    try:
        results = model.predict(  # type: ignore[union-attr]
            bgr,
            verbose=False,
            conf=0.05,
            imgsz=imgsz,
            iou=0.7,
        )
    except Exception as exc:
        _log_info(f"UI YOLO predict failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"UI YOLO predict failed: {exc}") from exc

    if not results:
        return []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    names: dict[int, str] = getattr(model, "names", {}) or {}
    out: list[UiDetection] = []
    xyxy = res.boxes.xyxy.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy() if res.boxes.cls is not None else None
    for i in range(len(xyxy)):
        row = xyxy[i]
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        bw, bh = max(1, x2i - x1i), max(1, y2i - y1i)
        bx, by, bw, bh = _clip_box(x1i, y1i, bw, bh, w, h)
        cx = bx + bw // 2
        cy = by + bh // 2
        cls_id = int(clss[i]) if clss is not None else 0
        class_name = str(names.get(cls_id, str(cls_id)))
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
    paths = logger.require_paths()
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


def _format_candidates_text(detections: list[UiDetection]) -> str:
    lines: list[str] = []
    for i, d in enumerate(detections):
        text = f" text={d.text!r}" if d.text else ""
        lines.append(
            f"[{i}] center=[{d.cx},{d.cy}]{text}"
        )
    return "\n".join(lines)


async def _need_text_anchors(instruction: str) -> bool:
    prompt = (
        "You decide whether selecting a UI element requires text as an anchor.\n"
        "Return JSON only: {\"need_text_anchor\": true|false}.\n\n"
        "Set true when the instruction refers to visible words/labels/content "
        "(for example: click 'Sign in', click the item with text, select row by name).\n"
        "Set false when the instruction is mostly non-text visual targetting "
        "(for example: click icon, toggle, avatar, gear, unlabeled button, panel).\n\n"
        f"Instruction: {instruction}"
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_TEXT_ANCHOR_JSON_SCHEMA,
        )
        return _parse_need_text_anchor_from_llm(reply.content)
    except ValueError as exc:
        logger.log_info(f"_need_text_anchors: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"need_text_anchor": true|false}. '
            "No text before or after the JSON.\n"
        )
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
        )
        return _parse_need_text_anchor_from_llm(reply.content)


async def _filter_text_detections(
    text_detections: list[UiDetection],
    instruction: str,
) -> list[UiDetection]:
    if not text_detections:
        return []

    candidates_text = _format_candidates_text(text_detections)
    prompt = (
        "Select ONLY text candidates that match the user instruction.\n"
        'Return JSON only: {"keep_indices": [<int>, ...]}.\n'
        "Use indices from the Candidates list. Keep an empty list when none match.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Candidates:\n{candidates_text}"
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_TEXT_FILTER_JSON_SCHEMA,
        )
        keep_indices = _parse_keep_indices_from_llm(reply.content, len(text_detections))
    except ValueError as exc:
        logger.log_info(f"_filter_text_detections: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"keep_indices": [<integer>, ...]}. '
            "No text before or after the JSON.\n"
        )
        try:
            reply = await _ollama.chat_messages(
                settings.brain_lm,
                messages=messages,
                tools=[],
                response_format="json",
            )
            keep_indices = _parse_keep_indices_from_llm(reply.content, len(text_detections))
        except ValueError as retry_exc:
            logger.log_info(f"_filter_text_detections: fallback keep-all ({retry_exc})")
            return text_detections

    if not keep_indices:
        return []
    return [text_detections[i] for i in keep_indices]


async def _select_index_with_ollama(
    instruction: str,
    detections: list[UiDetection],
    image_path: str,
) -> int:
    candidates_text = _format_candidates_text(detections)
    base_instructions = get_prompt("ui_element_selection").format(
        instruction=instruction,
        candidates_text=candidates_text,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": base_instructions, "images": [image_path]},
    ]
    try:
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_INDEX_JSON_SCHEMA,
        )
        idx = _parse_index_from_llm(reply.content)
    except ValueError as exc:
        logger.log_info(f"_select_index_with_ollama: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"index": <integer>} using a valid index from the Candidates list. '
            "No text before or after the JSON.\n"
        )
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
        )
        idx = _parse_index_from_llm(reply.content)

    if idx < 0 or idx >= len(detections):
        raise ValueError(
            f"Ollama returned index {idx} but valid range is 0..{len(detections) - 1}"
        )
    return idx


async def resolve_ui_element_point(instruction: str) -> tuple[int, int, dict[str, Any]]:
    """
    Capture screen, run UI YOLO, pick one detection via Ollama, return global (x, y) and metadata.
    """
    text = (instruction or "").strip()
    if not text:
        raise ValueError("instruction must be non-empty")

    paths = logger.require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ui_dir / name
    capture_active_monitor_to_file(out)
    image_path = str(out.resolve())

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"could not read capture image: {image_path}")

    raw = _predict_ui_elements_raw(bgr)
    pruned = _keep_smallest_non_overlapping(raw)
    need_text_anchor = await _need_text_anchors(text)
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

    if len(detections) == 1:
        chosen = detections[0]
        idx = 0
        _log_info("_resolve_ui_element: single candidate; skipping Ollama index pick")
    else:
        idx = await _select_index_with_ollama(text, detections, image_path)
        chosen = detections[idx]

    gx, gy = _to_global_coordinate(chosen.cx, chosen.cy)
    meta: dict[str, Any] = {
        "selected_index": idx,
        "class_name": chosen.class_name,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "screenshot_path": image_path,
    }
    return gx, gy, meta
