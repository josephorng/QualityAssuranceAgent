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

from cua_mcp.coordinate_selection import _to_global_coordinate
from cua_mcp.read_screen_text.ocr_image import get_text_boxes_from_path
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

# Class names matching these substrings (case-insensitive) are treated as text regions.
_TEXT_CLASS_SUBSTRINGS: tuple[str, ...] = ("text", "ocr")


@dataclass(frozen=True)
class UiDetection:
    bbox: tuple[int, int, int, int]  # x, y, w, h in image pixels
    cx: int
    cy: int
    class_id: int
    class_name: str
    confidence: float


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
    confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
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
        conf = float(confs[i]) if confs is not None else 1.0
        cls_id = int(clss[i]) if clss is not None else 0
        class_name = str(names.get(cls_id, str(cls_id)))
        out.append(
            UiDetection(
                bbox=(bx, by, bw, bh),
                cx=cx,
                cy=cy,
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
            )
        )
    return out


def _is_text_like_class(name: str) -> bool:
    key = name.strip().lower()
    return any(sub in key for sub in _TEXT_CLASS_SUBSTRINGS)


def _filter_non_text(detections: list[UiDetection]) -> list[UiDetection]:
    return [d for d in detections if not _is_text_like_class(d.class_name)]


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
    filtered: list[UiDetection],
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
                "non_text": len(filtered),
                "pruned_smallest_overlap": len(pruned),
                "after_text_box_subtract": len(final),
            },
            "detections": [
                {
                    "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
                    "center": {"x": d.cx, "y": d.cy},
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
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
        lines.append(
            f"[{i}] {d.class_name} center=[{d.cx},{d.cy}] conf={d.confidence:.3f}"
        )
    return "\n".join(lines)


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
    filtered = _filter_non_text(raw)
    pruned = _keep_smallest_non_overlapping(filtered)
    text_boxes = get_text_boxes_from_path(image_path)
    no_text_overlap = _remove_text_overlapping_detections(pruned, text_boxes)
    detections = _sort_detections_reading_order(no_text_overlap)
    _persist_ui_result(
        image_path=image_path,
        raw=raw,
        filtered=filtered,
        pruned=pruned,
        text_boxes=text_boxes,
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
        "confidence": chosen.confidence,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "screenshot_path": image_path,
    }
    return gx, gy, meta
