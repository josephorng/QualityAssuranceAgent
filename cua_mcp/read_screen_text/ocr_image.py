"""
OCR pipeline: YOLO text-region detection + CRNN recognition.

Reads an image from disk, detects text regions with ``yolo_best.pt``,
runs CRNN directly on each detected crop using ``crnn_cfc_model.pt``, and
returns reading-order lines formatted as ``[center_x,center_y] text``.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .inference import TextPredictor
from src.common.run_state import get_run_state_manager
from src.common.io_utils import write_json

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_YOLO_MODEL: object | None = None
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
    all_lines: list[tuple[tuple[int, int, int, int], str]],
    formatted: list[str],
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
            "lines": all_lines,
            "text": "\n".join(formatted),
        },
    )
    _log_info(f"OCR result persisted path={out_path}")


def _default_crnn_path() -> str:
    """Return the default CRNN model path inside this package."""
    return os.path.join(_PACKAGE_DIR, "crnn_cfc_model.pt")


def _default_yolo_path() -> str:
    """Return the default YOLO detector path inside this package."""
    return os.path.join(_PACKAGE_DIR, "yolo_best.pt")


def _get_crnn_predictor(model_path: Optional[str] = None) -> TextPredictor:
    """Lazily initialize and cache the CRNN predictor instance."""
    global _CRNN_PREDICTOR
    path = model_path or _default_crnn_path()
    if _CRNN_PREDICTOR is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CRNN model not found: {path}")
        _log_info(f"OCR initializing CRNN predictor model_path={path}")
        _CRNN_PREDICTOR = TextPredictor(path)
    return _CRNN_PREDICTOR


def _get_yolo() -> object:
    """Lazily initialize and cache the YOLO detector instance."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            ) from e
        yolo_path = _default_yolo_path()
        if not os.path.isfile(yolo_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        _log_info(f"OCR initializing YOLO detector model_path={yolo_path}")
        _YOLO_MODEL = YOLO(yolo_path)
    return _YOLO_MODEL


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
        model = _get_yolo()
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        _log_info(f"OCR YOLO unavailable: {type(exc).__name__}: {exc}")
        return []

    h, w = bgr.shape[:2]
    # Ultralytics expects imgsz to align with stride (32); avoid noisy warnings.
    m = max(h, w)
    imgsz = max(32, ((m + 31) // 32) * 32)
    try:
        results = model.predict(  # type: ignore[union-attr]
            bgr,
            verbose=False,
            conf=0.25,
            imgsz=imgsz,
        )
    except Exception as exc:
        _log_info(f"OCR YOLO predict failed: {type(exc).__name__}: {exc}")
        return []

    if not results:
        return []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    out: list[tuple[int, int, int, int]] = []
    xyxy = res.boxes.xyxy.cpu().numpy()
    for row in xyxy:
        x1, y1, x2, y2 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
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


def _ocr_crop(
    bgr_crop: np.ndarray,
    predictor: TextPredictor,
    line_height: int,
) -> str:
    """Run CRNN OCR on a single crop."""
    if bgr_crop.size == 0 or bgr_crop.shape[0] < 2 or bgr_crop.shape[1] < 2:
        return ""
    if line_height < 2:
        line_height = 32

    if len(bgr_crop.shape) == 3:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_crop

    h, w = gray.shape[:2]
    new_width = max(1, int((w / max(1, h)) * line_height))
    resized = cv2.resize(gray, (new_width, line_height), interpolation=cv2.INTER_LINEAR)
    line_image = np.expand_dims(np.array(resized), axis=0)

    try:
        predicted_texts, _pred_prob = predictor.predict_images(line_image, beam_search=False)
        return "".join(predicted_texts).strip()
    except Exception:
        return ""


def get_coordinates(
    image_path: str,
    *,
    line_height: int = 32,
    crnn_model_path: Optional[str] = None,
) -> str:
    """
    Run YOLO + CRNN OCR on the image at ``image_path``.

    Returns one string with one line per detection:
    ``[center_x,center_y] <recognized text>`` (reading order). On error, returns a line
    starting with ``[error]``.
    """
    _log_info(
        f"OCR get_coordinates start image_path={image_path} line_height={line_height}"
    )
    if not image_path or not isinstance(image_path, str):
        _log_info("OCR invalid image_path argument")
        return "[error] invalid image_path"
    if not os.path.isfile(image_path):
        _log_info(f"OCR image file not found path={image_path}")
        return f"[error] file not found: {image_path}"

    bgr = cv2.imread(image_path)
    if bgr is None:
        _log_info(f"OCR could not read image path={image_path}")
        return f"[error] could not read image: {image_path}"

    try:
        predictor = _get_crnn_predictor(crnn_model_path)
    except FileNotFoundError as e:
        _log_info(f"OCR CRNN model missing: {e}")
        return f"[error] {e}"

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

    boxes = _sort_boxes_reading_order(boxes)

    all_lines: list[tuple[tuple[int, int, int, int], str]] = []
    ocr_elapsed_ms = 0.0
    for x, y, w, h in boxes:
        crop = bgr[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        ocr_start = time.perf_counter()
        text = _ocr_crop(crop, predictor, line_height)
        ocr_elapsed_ms += (time.perf_counter() - ocr_start) * 1000.0
        all_lines.append(((x, y, w, h), text))

    # Global reading order: top to bottom, left to right.
    all_lines.sort(key=lambda item: (item[0][1], item[0][0]))

    formatted = [
        f"[{r[0] + (r[2] // 2)},{r[1] + (r[3] // 2)}] {txt}"
        for r, txt in all_lines
    ]
    _persist_ocr_result(
        image_path=image_path,
        line_height=line_height,
        all_lines=all_lines,
        formatted=formatted,
        yolo_elapsed_ms=yolo_elapsed_ms,
        ocr_elapsed_ms=ocr_elapsed_ms,
    )
    _log_info(f"OCR get_coordinates done lines={len(formatted)}")
    _log_info(f"OCR get_coordinates formatted={formatted}")
    return "\n".join(formatted)
