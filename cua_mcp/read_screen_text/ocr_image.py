"""
OCR pipeline: YOLO text/element detection + CRNN recognition.

Reads an image from disk, detects ``text`` and ``element`` regions with ``cua_mcp/best.onnx``
(ONNX Runtime), runs CRNN (ONNX) on each detected crop using ``ocr_model_finetuned.onnx``, and
returns reading-order regions ``(bbox, (center_x, center_y), predict_images)``.
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

from cua_mcp.geometry import boxes_overlap, clip_box, sort_by_reading_order
from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_TEXT,
    run_yolo_onnx_end2end,
)
from src.eye.capture import capture_active_monitor_to_file
from .inference_onnx import TextPredictor
from src.common.io_utils import write_json
from src.common.run_state import get_run_state_manager, ts_name

_OCR_DETECTION_CLASS_IDS = frozenset({YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT})

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


def _get_ocr_predictor(
    model_path: Optional[str] = None,
    *,
    quiet: bool = False,
) -> TextPredictor:
    """Lazily initialize and cache the ONNX CRNN predictor instance."""
    global _CRNN_PREDICTOR
    path = model_path or _default_crnn_path()
    if _CRNN_PREDICTOR is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ONNX CRNN model not found: {path}")
        if not quiet:
            _log_info(f"OCR initializing ONNX CRNN predictor model_path={path}")
        _CRNN_PREDICTOR = TextPredictor(path, quiet=quiet)
    return _CRNN_PREDICTOR


def warm_vision_models(*, quiet: bool = True) -> None:
    """Eagerly load YOLO and CRNN ONNX sessions (no-op when already cached)."""
    from cua_mcp.yolo_onnx import DEFAULT_YOLO_ONNX_PATH, get_cached_cpu_session

    get_cached_cpu_session(DEFAULT_YOLO_ONNX_PATH)
    _get_ocr_predictor(quiet=quiet)


def _run_ocr_yolo_onnx_inference(
    bgr: np.ndarray,
    *,
    class_ids: frozenset[int] | set[int] = _OCR_DETECTION_CLASS_IDS,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> np.ndarray:
    """
    Letterbox to 640×640 (Ultralytics-style), RGB CHW normalize, run ``cua_mcp/best.onnx`` (YOLOv26 end2end).

    Returns ``N×4`` ``xyxy`` in original image pixel space after score filtering
    (NMS is in the ONNX graph). Keeps only detections whose class is in ``class_ids``.
    """
    xyxy, _scores, _cls = run_yolo_onnx_end2end(
        bgr,
        class_ids=set(class_ids),
        conf_threshold=conf_threshold,
        on_session_created=lambda p: _log_info(
            f"OCR initializing YOLO ONNX detector model_path={p}"
        ),
    )
    return xyxy


def _expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    *,
    margin: int = 2,
) -> tuple[int, int, int, int]:
    """Expand a box by ``margin`` pixels on all four sides, clamped to image bounds."""
    return clip_box(x - margin, y - margin, w + 2 * margin, h + 2 * margin, img_w, img_h)


def _yolo_boxes(
    bgr: np.ndarray,
    *,
    class_ids: frozenset[int] | set[int] = _OCR_DETECTION_CLASS_IDS,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) in image coordinates, or empty if unavailable."""
    try:
        xyxy = _run_ocr_yolo_onnx_inference(
            bgr,
            class_ids=class_ids,
            conf_threshold=conf_threshold,
        )
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
        out.append(clip_box(x1i, y1i, bw, bh, w, h))
    return out


def _sort_boxes_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Sort boxes top-to-bottom, then left-to-right within a row."""
    return sort_by_reading_order(
        boxes,
        center_fn=lambda b: (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0),
        row_height_fn=lambda b: b[3],
        x_fn=lambda b: b[0],
    )


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
                if boxes_overlap(current, other):
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


def get_coordinates_from_image_path(
    image_path: str,
    *,
    line_height: int = 32,
    ocr_model_path: Optional[str] = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]]:
    """
    Run YOLO + OCR on the image at ``image_path``.

    YOLO detects both ``text`` and ``element`` classes; each box is cropped and passed through OCR.

    Returns a list of regions, each
    ``((x, y, w, h), (center_x, center_y), predicted_texts)`` where ``predicted_texts`` is the raw list from
    ``TextPredictor.predict_images(..., beam_search=False)`` (reading order). On failure,
    returns ``[]``. Use :func:`format_coordinate_text_from_regions` when you need
    the ``[cx,cy] text`` hint string for an LM.
    """
    _log_info(
        f"OCR get_coordinates_from_path start image_path={image_path} line_height={line_height}"
    )
    if not image_path or not isinstance(image_path, str):
        _log_info("OCR invalid image_path argument")
        return []
    if not os.path.isfile(image_path):
        _log_info(f"OCR image file not found path={image_path}")
        return []

    bgr = cv2.imread(image_path)
    if bgr is None:
        _log_info(f"OCR could not read image path={image_path}")
        return []

    img_h, img_w = bgr.shape[:2]

    try:
        predictor = _get_ocr_predictor(ocr_model_path)
    except FileNotFoundError as e:
        _log_info(f"OCR ONNX OCR model missing: {e}")
        return []

    img_h, img_w = bgr.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    yolo_elapsed_ms: float | None = None
    yolo_start = time.perf_counter()
    boxes = _yolo_boxes(bgr, conf_threshold=yolo_conf_threshold)
    yolo_elapsed_ms = (time.perf_counter() - yolo_start) * 1000.0
    _log_info(f"OCR YOLO detected_boxes={len(boxes)}")

    if not boxes:
        # Full-frame fallback when detection is unavailable.
        _log_info("OCR using full-frame fallback box")
        boxes = [(0, 0, img_w, img_h)]

    boxes = _merge_overlapping_boxes(boxes)
    boxes = _sort_boxes_reading_order(boxes)
    boxes = [_expand_box(x, y, w, h, img_w, img_h) for x, y, w, h in boxes]

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
        image_path=image_path,
        line_height=line_height,
        all_regions=all_regions,
        yolo_elapsed_ms=yolo_elapsed_ms,
        ocr_elapsed_ms=ocr_elapsed_ms,
    )
    _log_info(f"OCR get_coordinates_from_path done regions={len(all_regions)}")
    return all_regions


def get_text_boxes_from_path(
    image_path: str,
    *,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[tuple[int, int, int, int]]:
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

    boxes = _yolo_boxes(
        bgr,
        class_ids=frozenset({YOLO_CLASS_TEXT}),
        conf_threshold=yolo_conf_threshold,
    )
    _log_info(f"OCR get_text_boxes_from_path boxes={len(boxes)}")
    return boxes


def get_coordinates(
    *,
    line_height: int = 32,
    crnn_model_path: Optional[str] = None,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]]:
    """
    Capture the active monitor to this run's ``yolo_ocr/`` folder, then run YOLO + ONNX CRNN OCR.

    Writes ``<timestamp>.png`` and persists OCR JSON with the same basename beside it.
    See :func:`get_coordinates_from_image_path` for per-region tuples.
    """
    paths = get_run_state_manager().require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ocr_dir / name
    capture_active_monitor_to_file(out)
    return get_coordinates_from_image_path(
        str(out),
        line_height=line_height,
        ocr_model_path=crnn_model_path,
    )
