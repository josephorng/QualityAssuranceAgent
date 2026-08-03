"""
Dev/tooling OCR coordinate pipeline (not imported by the main agent runtime).

Run YOLO on disk images, selectively CRNN ``text``/``element`` boxes, and return reading-order
regions. Import this module only from viewers, debug scripts, and tests so Nuitka onefile builds
from ``main.py`` do not pull it in.
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
    MOUSE_TARGET_CLASS_IDS,
    OCR_DETECTION_CLASS_IDS,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_TEXT,
    run_yolo_onnx_end2end,
)
from src.common.io_utils import write_json
from src.common.monitor_prompt import selected_eye_monitor_indices
from src.common.run_state import get_run_state_manager, ts_name
from src.eye.capture import active_monitor_offset, capture_monitor_to_file

from .ocr_image import (
    _DEFAULT_CRNN_BATCH_SIZE,
    _expand_box,
    _get_ocr_predictor,
    _log_info,
    _ocr_crops_batched,
)

_OCR_DETECTION_CLASS_IDS = OCR_DETECTION_CLASS_IDS
_OCR_CRNN_CLASS_IDS = frozenset({YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT})

OcrRegion = tuple[tuple[int, int, int, int], tuple[int, int], list[str]]


def _persist_ocr_result(
    image_path: str,
    line_height: int,
    all_regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
    *,
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


def _yolo_classed_boxes(
    bgr: np.ndarray,
    *,
    class_ids: frozenset[int] | set[int] = MOUSE_TARGET_CLASS_IDS,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[tuple[tuple[int, int, int, int], int]]:
    """Return ``(bbox, class_id)`` pairs in image coordinates, or empty if unavailable."""
    h, w = bgr.shape[:2]
    try:
        xyxy, _scores, cls_arr = run_yolo_onnx_end2end(
            bgr,
            class_ids=set(class_ids),
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

    out: list[tuple[tuple[int, int, int, int], int]] = []
    for row, cls_id in zip(xyxy, cls_arr, strict=True):
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        bw, bh = max(1, x2i - x1i), max(1, y2i - y1i)
        out.append((clip_box(x1i, y1i, bw, bh, w, h), int(cls_id)))
    return out


def _yolo_boxes(
    bgr: np.ndarray,
    *,
    class_ids: frozenset[int] | set[int] = _OCR_DETECTION_CLASS_IDS,
    conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) in image coordinates, or empty if unavailable."""
    return [
        bbox
        for bbox, _cls_id in _yolo_classed_boxes(
            bgr,
            class_ids=class_ids,
            conf_threshold=conf_threshold,
        )
    ]


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


def _offset_region(region: OcrRegion, left: int, top: int) -> OcrRegion:
    """Shift a region's bbox and center into global desktop coordinates."""
    (x, y, w, h), (cx, cy), preds = region
    return ((x + left, y + top, w, h), (cx + left, cy + top), preds)


def format_coordinate_text_from_regions(
    regions: list[OcrRegion],
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
    batch_size: int = _DEFAULT_CRNN_BATCH_SIZE,
) -> list[OcrRegion]:
    """
    Run YOLO + selective OCR on the image at ``image_path``.

    YOLO detects ``text``, ``element``, ``input``, and ``scrollbar``. Only ``text`` and ``element``
    boxes are cropped and passed through CRNN; ``input`` and ``scrollbar`` are returned with empty
    ``predicted_texts``.

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

    yolo_elapsed_ms: float | None = None
    yolo_start = time.perf_counter()
    classed_boxes = _yolo_classed_boxes(bgr, conf_threshold=yolo_conf_threshold)
    yolo_elapsed_ms = (time.perf_counter() - yolo_start) * 1000.0
    _log_info(f"OCR YOLO detected_boxes={len(classed_boxes)}")

    ocr_boxes: list[tuple[int, int, int, int]] = []
    non_ocr_boxes: list[tuple[int, int, int, int]] = []
    for bbox, cls_id in classed_boxes:
        if cls_id in _OCR_CRNN_CLASS_IDS:
            ocr_boxes.append(bbox)
        else:
            non_ocr_boxes.append(bbox)

    if not classed_boxes:
        _log_info("OCR using full-frame fallback box")
        ocr_boxes = [(0, 0, img_w, img_h)]

    ocr_boxes = _merge_overlapping_boxes(ocr_boxes)
    ocr_boxes = _sort_boxes_reading_order(ocr_boxes)
    ocr_boxes = [_expand_box(x, y, w, h, img_w, img_h) for x, y, w, h in ocr_boxes]

    box_crops: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for x, y, w, h in ocr_boxes:
        crop = bgr[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        box_crops.append(((x, y, w, h), crop))

    ocr_elapsed_ms: float | None = None
    all_regions: list[OcrRegion] = []
    if box_crops:
        _log_info(f"OCR CRNN start boxes={len(box_crops)} batch_size={batch_size}")
        ocr_start = time.perf_counter()
        all_preds = _ocr_crops_batched(
            [crop for _, crop in box_crops],
            predictor,
            line_height,
            batch_size=batch_size,
        )
        ocr_elapsed_ms = (time.perf_counter() - ocr_start) * 1000.0
        for (x, y, w, h), preds in zip(
            (box for box, _crop in box_crops),
            all_preds,
            strict=True,
        ):
            all_regions.append(((x, y, w, h), (x + w // 2, y + h // 2), preds))

    for x, y, w, h in non_ocr_boxes:
        all_regions.append(((x, y, w, h), (x + w // 2, y + h // 2), []))

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


def get_coordinates_from_selected_monitors(
    *,
    line_height: int = 32,
    ocr_model_path: Optional[str] = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    batch_size: int = _DEFAULT_CRNN_BATCH_SIZE,
) -> tuple[list[OcrRegion], list[str]]:
    """
    Capture each selected monitor, run YOLO + OCR per image, and merge regions in global coords.

    Returns ``(merged_regions, image_paths)`` where centers and bboxes are in virtual-desktop
    pixel space (suitable for pyautogui). When only one monitor is selected, behavior matches
    single-monitor OCR with the appropriate desktop offset applied.
    """
    paths = get_run_state_manager().require_paths()
    monitor_indices = selected_eye_monitor_indices()
    stamp = ts_name()
    merged_regions: list[OcrRegion] = []
    image_paths: list[str] = []

    _log_info(f"OCR get_coordinates_from_selected_monitors monitors={monitor_indices}")
    for monitor_index in monitor_indices:
        name = f"{stamp}_mon{monitor_index}.png"
        out = paths.yolo_ocr_dir / name
        capture_monitor_to_file(out, monitor_index)
        image_path = str(out.resolve())
        image_paths.append(image_path)

        regions = get_coordinates_from_image_path(
            image_path,
            line_height=line_height,
            ocr_model_path=ocr_model_path,
            yolo_conf_threshold=yolo_conf_threshold,
            batch_size=batch_size,
        )
        left, top = active_monitor_offset(monitor_index)
        merged_regions.extend(_offset_region(region, left, top) for region in regions)

    merged_regions.sort(key=lambda item: (item[1][1], item[1][0]))
    _log_info(
        f"OCR get_coordinates_from_selected_monitors done monitors={len(monitor_indices)} "
        f"regions={len(merged_regions)}"
    )
    return merged_regions, image_paths


def get_coordinates(
    *,
    line_height: int = 32,
    crnn_model_path: Optional[str] = None,
    batch_size: int = _DEFAULT_CRNN_BATCH_SIZE,
) -> list[OcrRegion]:
    """
    Capture selected monitor(s) to this run's ``yolo_ocr/`` folder, then run YOLO + ONNX CRNN OCR.

    Writes ``<timestamp>_mon<N>.png`` per monitor and persists OCR JSON beside each image.
    See :func:`get_coordinates_from_image_path` for per-region tuples (local coords per image).
    """
    regions, _image_paths = get_coordinates_from_selected_monitors(
        line_height=line_height,
        ocr_model_path=crnn_model_path,
        batch_size=batch_size,
    )
    return regions
