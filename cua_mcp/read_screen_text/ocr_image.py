"""
OCR pipeline: YOLO text-region detection + CRNN recognition.

Reads an image from disk, detects text regions with ``cua_mcp/best.onnx`` (ONNX Runtime),
runs CRNN (ONNX) on each detected crop using ``ocr_model_finetuned.onnx``, and
returns an offset plus reading-order regions ``(bbox, (center_x, center_y), predict_images)``.
Use :func:`format_coordinate_text_from_regions` for ``[center_x,center_y] text`` hints.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    YOLO_CLASS_TEXT,
    run_best_onnx_end2end,
)
from src.eye.capture import capture_active_monitor_to_file
from .inference_onnx import TextPredictor
from src.common.io_utils import write_json
from src.common.run_state import get_run_state_manager, ts_name

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_CRNN_PREDICTOR: TextPredictor | None = None


def _log_info(text: str) -> None:
    """Write an info log when run state is available."""
    try:
        get_run_state_manager().log_info(text)
    except RuntimeError:
        # OCR helpers can run in isolation before run state is initialized.
        pass


def _persist_ocr_result(
    image_path: str,
    line_height: int,
    all_regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
    # formatted: list[str],
    yolo_elapsed_ms: float | None = None,
    ocr_elapsed_ms: float | None = None,
) -> None:
    """Persist OCR output under this run's yolo_ocr folder."""
    try:
        paths = get_run_state_manager().require_paths()
    except RuntimeError:
        return

    image_name = Path(image_path).name
    out_path = paths.yolo_ocr_dir / Path(image_name).with_suffix(".json").name
    write_json(
        out_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": image_path,
            "image_name": image_name,
            "line_height": line_height,
            "yolo_elapsed_ms": yolo_elapsed_ms,
            "ocr_elapsed_ms": ocr_elapsed_ms,
            "lines": all_regions,
        },
    )
    _log_info(f"OCR result persisted path={out_path}")


def _default_crnn_path() -> str:
    """Return the default ONNX CRNN model path inside this package."""
    return os.path.join(_PACKAGE_DIR, "ocr_model_finetuned.onnx")


def _get_crnn_predictor(model_path: Optional[str] = None) -> TextPredictor:
    """Lazily initialize and cache the ONNX CRNN predictor instance."""
    global _CRNN_PREDICTOR
    path = model_path or _default_crnn_path()
    if _CRNN_PREDICTOR is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ONNX CRNN model not found: {path}")
        _log_info(f"OCR initializing ONNX CRNN predictor model_path={path}")
        _CRNN_PREDICTOR = TextPredictor(path)
    return _CRNN_PREDICTOR


def _run_ocr_yolo_onnx_inference(
    bgr: np.ndarray,
    *,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> np.ndarray:
    """
    Letterbox to 640×640 (Ultralytics-style), RGB CHW normalize, run ``cua_mcp/best.onnx`` (YOLOv26 end2end).

    Returns ``N×4`` ``xyxy`` in original image pixel space after score filtering
    (NMS is in the ONNX graph). Only ``text`` class detections are kept.
    """
    xyxy, _scores, _cls = run_best_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT},
        conf_threshold=conf_threshold,
        on_session_created=lambda p: _log_info(
            f"OCR initializing YOLO ONNX detector model_path={p}"
        ),
    )
    return xyxy


def _clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Clamp a bounding box so it stays within image bounds."""
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _yolo_text_boxes(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) in image coordinates, or empty if unavailable."""
    try:
        xyxy = _run_ocr_yolo_onnx_inference(bgr)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        _log_info(f"OCR YOLO unavailable: {type(exc).__name__}: {exc}")
        return []
    except Exception as exc:
        _log_info(f"OCR YOLO ONNX predict failed: {type(exc).__name__}: {exc}")
        return []

    if xyxy.size == 0:
        return []

    h, w = bgr.shape[:2]
    out: list[tuple[int, int, int, int]] = []
    for row in xyxy:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        bw, bh = max(1, x2i - x1i), max(1, y2i - y1i)
        out.append(_clip_box(x1i, y1i, bw, bh, w, h))
    return out


def _sort_boxes_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Sort boxes top-to-bottom, then left-to-right within a row."""
    if not boxes:
        return []
    items = sorted(
        [(b, b[1] + b[3] / 2.0) for b in boxes],
        key=lambda t: (t[1], t[0][0]),
    )
    mean_h = sum(b[3] for b in boxes) / len(boxes)
    tol = max(10.0, mean_h * 0.5)
    rows: list[list[tuple[int, int, int, int]]] = []
    row: list[tuple[int, int, int, int]] = []
    row_y0: float | None = None
    for b, cy in items:
        if row_y0 is None:
            row = [b]
            row_y0 = cy
            continue
        if abs(cy - row_y0) <= tol:
            row.append(b)
        else:
            rows.append(sorted(row, key=lambda bb: bb[0]))
            row = [b]
            row_y0 = cy
    if row:
        rows.append(sorted(row, key=lambda bb: bb[0]))
    ordered: list[tuple[int, int, int, int]] = []
    for r in rows:
        ordered.extend(r)
    return ordered


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    """Return True when two (x, y, w, h) boxes overlap by area."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _merge_two_boxes(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the smallest box containing both boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    return x1, y1, x2 - x1, y2 - y1


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge all transitive overlaps into single bounding boxes."""
    if len(boxes) < 2:
        return boxes
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int]] = []
        while merged:
            current = merged.pop()
            merged_with_current = False
            for i, other in enumerate(merged):
                if _boxes_overlap(current, other):
                    current = _merge_two_boxes(current, other)
                    merged.pop(i)
                    merged.append(current)
                    changed = True
                    merged_with_current = True
                    break
            if not merged_with_current:
                next_boxes.append(current)
        merged = next_boxes
    return merged


def _ocr_crop_predicted_texts(
    bgr_crop: np.ndarray,
    predictor: TextPredictor,
    line_height: int,
) -> list[str]:
    """Run CRNN OCR on a single crop; return raw ``predict_images`` token strings (same as line 269)."""
    if bgr_crop.size == 0 or bgr_crop.shape[0] < 2 or bgr_crop.shape[1] < 2:
        return []
    if line_height < 2:
        line_height = 32

    if len(bgr_crop.shape) == 3:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        normalized = (gray - np.min(gray)) / (
                np.max(gray) - np.min(gray) + 1e-7
            )
    else:
        normalized = bgr_crop

    h, w = normalized.shape[:2]
    new_width = max(1, int((w / max(1, h)) * line_height))
    resized = cv2.resize(normalized, (new_width, line_height), interpolation=cv2.INTER_LINEAR)
    line_image = np.expand_dims(np.array(resized), axis=0)

    try:
        predicted_texts = predictor.predict_images(line_image)
        return list(predicted_texts) if predicted_texts else []
    except Exception as e:
        print(f"OCR _ocr_crop_predicted_texts error: {e}")
        return []


def format_coordinate_text_from_regions(
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
) -> str:
    """Build ``[cx,cy] text`` lines for the coordinate-picker LM (one line per region, reading order)."""
    lines: list[str] = []
    for _box, (cx, cy), preds in regions:
        t = "".join(preds).strip()
        lines.append(f"[{cx},{cy}] {t}")
    return "\n".join(lines)


def get_coordinates_from_path(
    image_path: str,
    *,
    line_height: int = 32,
    crnn_model_path: Optional[str] = None,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int], list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]]]:
    """
    Run YOLO + ONNX CRNN OCR on the image at ``image_path``.

    Optional ``crop_rect`` is ``(x, y, w, h)`` in image pixel coordinates; OCR runs only on
    that region. Box coordinates in ``regions`` are relative to the cropped OCR image; add
    the returned offset to map into the uncropped image.

    Returns ``((offset_x, offset_y), regions)`` where each region is
    ``((x, y, w, h), (center_x, center_y), predicted_texts)`` and ``predicted_texts`` is the raw list from
    ``TextPredictor.predict_images(..., beam_search=False)`` (reading order). On failure,
    returns ``((0, 0), [])``. Use :func:`format_coordinate_text_from_regions` when you need
    the ``[cx,cy] text`` hint string for an LM.
    """
    _log_info(
        f"OCR get_coordinates_from_path start image_path={image_path} line_height={line_height}"
        f" crop_rect={crop_rect}"
    )
    if not image_path or not isinstance(image_path, str):
        _log_info("OCR invalid image_path argument")
        return (0, 0), []
    if not os.path.isfile(image_path):
        _log_info(f"OCR image file not found path={image_path}")
        return (0, 0), []

    bgr = cv2.imread(image_path)
    if bgr is None:
        _log_info(f"OCR could not read image path={image_path}")
        return (0, 0), []

    img_h, img_w = bgr.shape[:2]
    offset_x, offset_y = 0, 0
    persist_path = image_path
    if crop_rect is not None:
        cx, cy, cw, ch = _clip_box(*crop_rect, img_w, img_h)
        if cw < 2 or ch < 2:
            _log_info("OCR crop_rect too small after clamp; using full image")
        else:
            offset_x, offset_y = cx, cy
            bgr = bgr[cy : cy + ch, cx : cx + cw]
            crop_name = f"{Path(image_path).stem}_crop.png"
            persist_path = str(Path(image_path).parent / crop_name)
            cv2.imwrite(persist_path, bgr)
            _log_info(f"OCR cropped region offset=({offset_x},{offset_y}) size=({cw},{ch}) path={persist_path}")

    try:
        predictor = _get_crnn_predictor(crnn_model_path)
    except FileNotFoundError as e:
        _log_info(f"OCR ONNX CRNN model missing: {e}")
        return (0, 0), []

    img_h, img_w = bgr.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    yolo_elapsed_ms: float | None = None
    yolo_start = time.perf_counter()
    boxes = _yolo_text_boxes(bgr)
    yolo_elapsed_ms = (time.perf_counter() - yolo_start) * 1000.0
    _log_info(f"OCR YOLO detected_boxes={len(boxes)}")

    if not boxes:
        # Full-frame fallback when detection is unavailable.
        _log_info("OCR using full-frame fallback box")
        boxes = [(0, 0, img_w, img_h)]

    boxes = _merge_overlapping_boxes(boxes)
    boxes = _sort_boxes_reading_order(boxes)

    all_regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]] = []
    ocr_elapsed_ms = 0.0
    for x, y, w, h in boxes:
        crop = bgr[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        ocr_start = time.perf_counter()
        preds = _ocr_crop_predicted_texts(crop, predictor, line_height)
        ocr_elapsed_ms += (time.perf_counter() - ocr_start) * 1000.0
        all_regions.append(((x, y, w, h), (x + w // 2, y + h // 2), preds))

    # Global reading order: top to bottom, left to right.
    all_regions.sort(key=lambda item: (item[1][1], item[1][0]))

    _persist_ocr_result(
        image_path=persist_path,
        line_height=line_height,
        all_regions=all_regions,
        yolo_elapsed_ms=yolo_elapsed_ms,
        ocr_elapsed_ms=ocr_elapsed_ms,
    )
    _log_info(f"OCR get_coordinates_from_path done regions={len(all_regions)}")
    _log_info(f"OCR get_coordinates_from_path regions={all_regions}")
    return (offset_x, offset_y), all_regions


def get_text_boxes_from_path(image_path: str) -> list[tuple[int, int, int, int]]:
    """
    Return OCR detector YOLO text boxes only (x, y, w, h) for an image.

    This skips CRNN text recognition and only runs the text-region detector.
    """
    _log_info(f"OCR get_text_boxes_from_path start image_path={image_path}")
    if not image_path or not isinstance(image_path, str):
        _log_info("OCR get_text_boxes_from_path invalid image_path argument")
        return []
    if not os.path.isfile(image_path):
        _log_info(f"OCR get_text_boxes_from_path file not found path={image_path}")
        return []

    bgr = cv2.imread(image_path)
    if bgr is None:
        _log_info(f"OCR get_text_boxes_from_path could not read image path={image_path}")
        return []

    boxes = _yolo_text_boxes(bgr)
    _log_info(f"OCR get_text_boxes_from_path boxes={len(boxes)}")
    return boxes


def get_coordinates(
    *,
    line_height: int = 32,
    crnn_model_path: Optional[str] = None,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int], list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]]]:
    """
    Capture the active monitor to this run's ``yolo_ocr/`` folder, then run YOLO + ONNX CRNN OCR.

    Writes ``<timestamp>.png`` and persists OCR JSON with the same basename beside it.
    If ``crop_rect`` is set, OCR runs on that (x, y, w, h) region of the capture; see
    :func:`get_coordinates_from_path` for offset and per-region ``(bbox, center, predicted_texts)`` tuples.
    """
    paths = get_run_state_manager().require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ocr_dir / name
    capture_active_monitor_to_file(out)
    return get_coordinates_from_path(
        str(out),
        line_height=line_height,
        crnn_model_path=crnn_model_path,
        crop_rect=crop_rect,
    )
