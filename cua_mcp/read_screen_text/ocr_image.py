"""
Runtime OCR helpers: CRNN recognition on BGR crops and model warmup.

Used by the agent ``move_mouse`` path via :func:`_ocr_boxes_on_bgr`. For disk-image coordinate
pipelines (viewers, debug scripts), use :mod:`cua_mcp.read_screen_text.get_coordinates` instead.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from cua_mcp.geometry import clip_box
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT
from .constrained_decode import CharSpan, DecodeMode
from .inference_onnx import TextPredictor
from src.common.run_state import get_run_state_manager

_DEFAULT_CRNN_BATCH_SIZE = 64

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_CRNN_PREDICTOR: TextPredictor | None = None
_CRNN_PREDICTOR_LOCK = threading.Lock()


def _log_info(text: str) -> None:
    """Write an info log when run state is available."""
    try:
        get_run_state_manager().log_info(text)
    except RuntimeError:
        # OCR helpers can run in isolation before run state is initialized.
        pass


def _default_crnn_path() -> str:
    """Return the default ONNX CRNN model path inside this package."""
    return os.path.join(_PACKAGE_DIR, "ocr_model_finetuned.onnx")


def ocr_mode_for_yolo_class(class_id: int) -> DecodeMode:
    """Map YOLO class to CRNN decode stream: elements use icon_ids, else text_ids."""
    if int(class_id) == YOLO_CLASS_ELEMENT:
        return "icon"
    return "text"


def _normalize_crop_modes(
    mode: DecodeMode | Sequence[DecodeMode],
    count: int,
) -> list[DecodeMode]:
    if isinstance(mode, str):
        return [mode] * count
    modes = list(mode)
    if len(modes) != count:
        raise ValueError(f"mode length {len(modes)} does not match crop count {count}")
    return modes


def _get_ocr_predictor(
    model_path: Optional[str] = None,
    *,
    quiet: bool = False,
) -> TextPredictor:
    """Lazily initialize and cache the CRNN predictor instance (Triton-backed)."""
    global _CRNN_PREDICTOR
    path = model_path or _default_crnn_path()
    with _CRNN_PREDICTOR_LOCK:
        if _CRNN_PREDICTOR is None:
            if not quiet:
                _log_info(
                    f"OCR initializing CRNN predictor backend=triton model_path={path}"
                )
            _CRNN_PREDICTOR = TextPredictor(path, quiet=quiet)
        return _CRNN_PREDICTOR


def warm_vision_models(*, quiet: bool = True, timeout_seconds: float = 2.5) -> tuple[bool, str]:
    """Check whether Triton is ready. Returns ``(ok, message)``; never raises."""
    from cua_mcp.vision_triton import triton_ready

    started = time.perf_counter()
    if triton_ready(timeout_seconds=timeout_seconds):
        elapsed = time.perf_counter() - started
        message = f"Vision：Triton 就緒（{elapsed:.1f}s）"
        if not quiet:
            _log_info(f"Vision warmup: Triton server ready elapsed_s={elapsed:.3f}")
        return True, message
    elapsed = time.perf_counter() - started
    message = (
        f"Vision：Triton 無回應（{elapsed:.1f}s）— "
        "move_mouse / OCR 需要 Triton 連線"
    )
    if not quiet:
        _log_info(f"Vision warmup: Triton not ready elapsed_s={elapsed:.3f}")
    return False, message


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


def _ocr_boxes_on_bgr(
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    *,
    line_height: int = 32,
    ocr_model_path: Optional[str] = None,
    batch_size: int = _DEFAULT_CRNN_BATCH_SIZE,
    mode: DecodeMode | Sequence[DecodeMode] = "text",
) -> list[list[str]]:
    """
    Run CRNN OCR on each ``(x, y, w, h)`` crop in ``boxes``.

    ``mode`` selects the dual top-1 stream (``text_ids`` vs ``icon_ids``). Pass a
    single mode for all boxes, or one mode per box.

    Returns one prediction list per box (same order as ``boxes``). Empty boxes yield ``[]``.
    """
    if not boxes:
        return []

    img_h, img_w = bgr.shape[:2]
    try:
        predictor = _get_ocr_predictor(ocr_model_path)
    except FileNotFoundError as exc:
        _log_info(f"OCR ONNX OCR model missing: {exc}")
        return [[] for _ in boxes]

    expanded = [_expand_box(x, y, w, h, img_w, img_h) for x, y, w, h in boxes]
    crops: list[np.ndarray] = []
    for x, y, w, h in expanded:
        crop = bgr[y : y + h, x : x + w]
        crops.append(crop if crop.size > 0 else np.empty((0, 0), dtype=np.uint8))

    return _ocr_crops_batched(
        crops,
        predictor,
        line_height,
        batch_size=batch_size,
        mode=mode,
    )


def _effective_line_height(line_height: int) -> int:
    return 32 if line_height < 2 else line_height


def _prepare_crop_line_image(bgr_crop: np.ndarray, line_height: int) -> np.ndarray | None:
    """Normalize and resize a crop to shape ``(line_height, width)`` float32, or ``None`` if invalid."""
    if bgr_crop.size == 0 or bgr_crop.shape[0] < 2 or bgr_crop.shape[1] < 2:
        return None

    line_height = _effective_line_height(line_height)

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
    return np.asarray(resized, dtype=np.float32)


def _ocr_crops_batched(
    crops: list[np.ndarray],
    predictor: TextPredictor,
    line_height: int,
    *,
    batch_size: int = _DEFAULT_CRNN_BATCH_SIZE,
    mode: DecodeMode | Sequence[DecodeMode] = "text",
) -> list[list[str]]:
    """Run CRNN OCR on crops in width-sorted, zero-padded batches."""
    detailed = _ocr_crops_batched_detailed(
        crops,
        predictor,
        line_height,
        batch_size=batch_size,
        mode=mode,
    )
    results: list[list[str]] = [[] for _ in crops]
    for index, text, _spans in detailed:
        results[index] = [text] if text else []
    return results


def _ocr_crops_batched_detailed(
    crops: list[np.ndarray],
    predictor: TextPredictor,
    line_height: int,
    *,
    batch_size: int = _DEFAULT_CRNN_BATCH_SIZE,
    mode: DecodeMode | Sequence[DecodeMode] = "text",
) -> list[tuple[int, str, list[CharSpan]]]:
    """Run CRNN OCR; return ``(orig_index, text, char_spans)`` per valid crop."""
    if batch_size < 1:
        batch_size = _DEFAULT_CRNN_BATCH_SIZE

    line_height = _effective_line_height(line_height)
    crop_modes = _normalize_crop_modes(mode, len(crops))
    results: list[tuple[int, str, list[CharSpan]]] = []
    valid: list[tuple[int, np.ndarray, DecodeMode]] = []

    for index, crop in enumerate(crops):
        line_image = _prepare_crop_line_image(crop, line_height)
        if line_image is not None:
            valid.append((index, line_image, crop_modes[index]))

    valid.sort(key=lambda item: item[1].shape[1])

    batch_count = 0
    infer_total_s = 0.0
    started = time.perf_counter()

    for start in range(0, len(valid), batch_size):
        chunk = valid[start : start + batch_size]
        if not chunk:
            continue

        max_w = max(image.shape[1] for _, image, _ in chunk)
        batch = np.zeros((len(chunk), line_height, max_w), dtype=np.float32)
        chunk_modes = [row_mode for _, _, row_mode in chunk]
        row_widths = [int(image.shape[1]) for _, image, _ in chunk]
        for row, (_, image, _) in enumerate(chunk):
            h_img, w_img = image.shape[:2]
            batch[row, :h_img, :w_img] = image

        try:
            batch_started = time.perf_counter()
            predicted_texts, span_batches = predictor.predict_images(
                batch,
                widths=row_widths,
                mode=chunk_modes,
            )
            infer_total_s += time.perf_counter() - batch_started
            batch_count += 1
            for row, (orig_index, _, _) in enumerate(chunk):
                text = predicted_texts[row] if row < len(predicted_texts) else ""
                spans = span_batches[row] if row < len(span_batches) else []
                results.append((orig_index, text, spans))
        except Exception as exc:
            print(f"OCR _ocr_crops_batched error: {exc}")

    elapsed = time.perf_counter() - started
    if valid:
        _log_info(
            "CRNN profile "
            f"crops={len(crops)} valid={len(valid)} batches={batch_count} "
            f"infer_total_s={infer_total_s:.3f} total_s={elapsed:.3f}"
        )

    return results


def _ocr_crop_predicted_texts(
    bgr_crop: np.ndarray,
    predictor: TextPredictor,
    line_height: int,
    *,
    mode: DecodeMode = "text",
) -> list[str]:
    """Run CRNN OCR on a single crop; return raw ``predict_images`` token strings."""
    preds = _ocr_crops_batched(
        [bgr_crop], predictor, line_height, batch_size=1, mode=mode
    )
    return preds[0] if preds else []
