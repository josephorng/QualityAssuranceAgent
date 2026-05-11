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
    "properties": {
        "index": {"type": ["integer", "null"]},
        "cx": {"type": ["integer", "null"]},
        "cy": {"type": ["integer", "null"]},
        "w": {"type": ["integer", "null"]},
        "h": {"type": ["integer", "null"]},
    },
    "required": ["index", "cx", "cy", "w", "h"],
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
_CONFIRM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"confirmed": {"type": "boolean"}},
    "required": ["confirmed"],
}

# Max labeled icon crops for tournament picker groups.
_MAX_LABELED_CROPS: int = 12
# Max labeled icon crops attached to a single LLM request.
_MAX_ICON_CROPS_PER_REQUEST: int = 12


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


def _parse_pick_from_llm(raw: str) -> dict[str, int] | None:
    """Parse the picker LLM reply.

    Returns ``None`` when the model explicitly signals "no candidate matches"
    (``index`` is ``null``). Otherwise returns a dict with ``index`` plus any
    parseable ``cx/cy/w/h`` echoed by the model.
    """
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            'Ollama UI picker returned empty content; expected '
            '{"index": int|null, "cx": int|null, "cy": int|null, '
            '"w": int|null, "h": int|null}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "index" not in out:
        raise ValueError(f'must include "index"; preview={preview!r}')
    raw_index = out["index"]
    if raw_index is None:
        return None
    try:
        picked: dict[str, int] = {"index": int(raw_index)}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'"index" must be integer or null; got {raw_index!r}'
        ) from exc
    for key in ("cx", "cy", "w", "h"):
        if key in out and out[key] is not None:
            try:
                picked[key] = int(out[key])
            except (TypeError, ValueError):
                pass
    return picked


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


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split ``items`` into consecutive sub-lists of length ``size``.

    The last chunk may be shorter. ``size`` is clamped to at least 1.
    """
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _format_candidates_text(detections: list[UiDetection]) -> str:
    lines: list[str] = []
    for i, d in enumerate(detections):
        text = f" text={d.text!r}" if d.text else ""
        _bx, _by, bw, bh = d.bbox
        lines.append(f"[{i}] center=[{d.cx},{d.cy}] w={bw} h={bh}{text}")
    return "\n".join(lines)


def _write_bbox_crop(image_path: str, bbox: tuple[int, int, int, int]) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"could not read image for bbox crop: {image_path}")
    x, y, w, h = bbox
    ih, iw = img.shape[:2]
    x, y, w, h = _clip_box(x, y, w, h, iw, ih)
    crop = img[y : y + h, x : x + w]
    paths = logger.require_paths()
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

    paths = logger.require_paths()
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
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_CONFIRM_JSON_SCHEMA,
        )
        return _parse_confirmed_from_llm(reply.content)
    except ValueError as exc:
        logger.log_info(f"_confirm_selection_bbox_with_ollama: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"confirmed": true|false}. '
            "No text before or after the JSON.\n"
        )
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
        )
        return _parse_confirmed_from_llm(reply.content)


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
                reply = await _ollama.chat_messages(
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
                reply = await _ollama.chat_messages(
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


async def _one_ollama_index_pick(
    instruction: str,
    detections: list[UiDetection],
    image_path: str,
) -> dict[str, int] | None:
    """Ask Ollama to pick a candidate and echo its bbox metadata.

    In addition to the full screenshot, this attaches one labeled crop per
    candidate (up to ``_MAX_LABELED_CROPS``). Each crop has its ``[index]``
    burned into a black header at the top, so the model can read the index
    directly from pixels — eliminating the failure mode where it confuses a
    coordinate with the index, or mis-attributes image ordering.

    Returns:
        ``None`` when the model signals that none of the candidates match the
        instruction (it emits ``{"index": null, ...}``). Otherwise a dict with
        keys ``index, cx, cy, w, h`` — echoing the bbox fields lets the caller
        cross-check that the model's ``index`` is consistent with the bbox it
        intended to choose.
    """
    candidates_text = _format_candidates_text(detections)
    base_instructions = get_prompt("ui_element_selection").format(
        instruction=instruction,
        candidates_text=candidates_text,
    )

    labeled_crops: list[str] = []
    if 0 < len(detections) <= _MAX_LABELED_CROPS:
        for i, d in enumerate(detections):
            try:
                labeled_crops.append(
                    _write_labeled_bbox_crop(image_path, d.bbox, f"[{i}]")
                )
            except (RuntimeError, OSError, cv2.error) as exc:
                _log_info(
                    f"_one_ollama_index_pick: failed to write labeled crop for "
                    f"index {i}: {exc}"
                )
    elif len(detections) > _MAX_LABELED_CROPS:
        _log_info(
            f"_one_ollama_index_pick: {len(detections)} candidates exceed "
            f"_MAX_LABELED_CROPS={_MAX_LABELED_CROPS}; sending screenshot only"
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": base_instructions,
            "images": [image_path, *labeled_crops],
        },
    ]
    try:
        try:
            reply = await _ollama.chat_messages(
                settings.brain_lm,
                messages=messages,
                tools=[],
                response_format=_INDEX_JSON_SCHEMA,
                append_image_sizes=False,
            )
            return _parse_pick_from_llm(reply.content)
        except ValueError as exc:
            logger.log_info(f"_one_ollama_index_pick: retry ({exc})")
            messages[0]["content"] += (
                '\nReply with ONLY: {"index": <integer>, "cx": <integer>, "cy": <integer>, '
                '"w": <integer>, "h": <integer>} where every value is copied verbatim from '
                "the same row of the Candidates list. "
                'If NONE of the candidates match the instruction, reply with '
                '{"index": null, "cx": null, "cy": null, "w": null, "h": null}. '
                "No text before or after the JSON.\n"
            )
            reply = await _ollama.chat_messages(
                settings.brain_lm,
                messages=messages,
                tools=[],
                response_format="json",
                append_image_sizes=False,
            )
            return _parse_pick_from_llm(reply.content)
    finally:
        for path in labeled_crops:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass


async def _tournament_pick(
    instruction: str,
    detections: list[UiDetection],
    image_path: str,
) -> int | None:
    """Tournament-style pick over arbitrarily many candidates.

    Splits ``detections`` into groups of size ``_MAX_LABELED_CROPS`` and runs
    ``_one_ollama_index_pick`` on each group. The winners from one round form
    the candidate set of the next round, and the process repeats until a single
    winner remains.

    Returns the winner's index into the original ``detections`` list, or
    ``None`` if the model signals that no candidate matches the instruction in
    any group of any round.

    Groups of size 1 are passed through automatically (no model call). The
    function requires at least one detection.
    """
    if not detections:
        raise ValueError("no candidates to pick from")

    current_indices: list[int] = list(range(len(detections)))
    round_no = 0
    while len(current_indices) > 1:
        round_no += 1
        next_indices: list[int] = []
        groups = _chunked(current_indices, _MAX_LABELED_CROPS)
        _log_info(
            f"_tournament_pick: round={round_no} candidates={len(current_indices)} "
            f"groups={len(groups)} group_size_cap={_MAX_LABELED_CROPS}"
        )
        for group_no, group in enumerate(groups, start=1):
            if len(group) == 1:
                next_indices.append(group[0])
                continue
            group_subset = [detections[i] for i in group]
            picked = await _one_ollama_index_pick(
                instruction, group_subset, image_path
            )
            if picked is None:
                _log_info(
                    f"_tournament_pick: round={round_no} group={group_no} "
                    f"size={len(group_subset)} no_match"
                )
                continue
            local_idx = picked["index"]
            if local_idx < 0 or local_idx >= len(group_subset):
                raise ValueError(
                    f"Ollama returned index {local_idx} but valid range is "
                    f"0..{len(group_subset) - 1} (picked={picked}, "
                    f"round={round_no}, group={group_no})"
                )
            winner_orig = group[local_idx]
            next_indices.append(winner_orig)
            _log_info(
                f"_tournament_pick: round={round_no} group={group_no} "
                f"size={len(group_subset)} local_idx={local_idx} "
                f"winner_orig={winner_orig}"
            )

        if not next_indices:
            _log_info(
                f"_tournament_pick: round={round_no} produced no winners; "
                "no candidate matched the instruction"
            )
            return None

        if len(next_indices) >= len(current_indices):
            # No reduction (e.g. cap=1 so every group is size 1). Bail out so we
            # don't loop forever; return the first survivor as a best-effort.
            _log_info(
                f"_tournament_pick: round={round_no} produced no reduction "
                f"({len(current_indices)} -> {len(next_indices)}); stopping"
            )
            return next_indices[0]
        current_indices = next_indices

    return current_indices[0]


async def _select_index_with_ollama(
    instruction: str,
    detections: list[UiDetection],
    image_path: str,
) -> int:
    candidates: list[tuple[int, UiDetection]] = list(enumerate(detections))
    max_rounds = max(1, len(detections) * 2 + 4)
    rounds = 0
    while candidates:
        rounds += 1
        if rounds > max_rounds:
            raise ValueError(
                "UI bbox confirmation exhausted retries; remaining candidates could not be confirmed."
            )
        subset = [d for _orig, d in candidates]
        idx = await _tournament_pick(instruction, subset, image_path)
        if idx is None:
            raise ValueError(
                "No UI candidate matched the instruction (tournament returned no winner)."
            )
        if idx < 0 or idx >= len(subset):
            raise ValueError(
                f"Tournament returned index {idx} but valid range is 0..{len(subset) - 1}"
            )
        crop_path = _write_bbox_crop(image_path, subset[idx].bbox)
        try:
            confirmed = await _confirm_selection_bbox_with_ollama(instruction, crop_path)
        finally:
            try:
                Path(crop_path).unlink(missing_ok=True)
            except OSError:
                pass

        if confirmed:
            orig_idx, _d = candidates[idx]
            return orig_idx

        _log_info(
            f"_select_index_with_ollama: bbox not confirmed; dropping candidate "
            f"orig_index={candidates[idx][0]} local_idx={idx}"
        )
        candidates.pop(idx)

    raise ValueError(
        "No UI candidate remained after bbox confirmation rejected all proposed picks."
    )


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

    candidate_pool: list[tuple[int, UiDetection]] = list(enumerate(detections))
    if len(candidate_pool) > 1:
        keep_local_indices = await _filter_icon_detections_with_ollama(
            text, [d for _orig, d in candidate_pool], image_path
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
        _log_info("_resolve_ui_element: single candidate; skipping Ollama index pick")
    else:
        filtered_detections = [d for _orig, d in candidate_pool]
        pick_idx = await _select_index_with_ollama(text, filtered_detections, image_path)
        orig_idx, chosen = candidate_pool[pick_idx]
        idx = orig_idx

    gx, gy = _to_global_coordinate(chosen.cx, chosen.cy)
    meta: dict[str, Any] = {
        "selected_index": idx,
        "class_name": chosen.class_name,
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "screenshot_path": image_path,
    }
    return gx, gy, meta
