"""
Runtime OCR helpers: CRNN recognition on BGR crops and model warmup.

Used by the agent ``move_mouse`` path via :func:`_ocr_boxes_on_bgr`. For disk-image coordinate
pipelines (viewers, debug scripts), use :mod:`cua_mcp.read_screen_text.get_coordinates` instead.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from cua_mcp.geometry import clip_box
from cua_mcp.vision_backend import should_try_triton
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


def _get_ocr_predictor(
    model_path: Optional[str] = None,
    *,
    quiet: bool = False,
) -> TextPredictor:
    """Lazily initialize and cache the ONNX CRNN predictor instance."""
    global _CRNN_PREDICTOR
    path = model_path or _default_crnn_path()
    with _CRNN_PREDICTOR_LOCK:
        if _CRNN_PREDICTOR is None:
            if not should_try_triton() and not os.path.isfile(path):
                raise FileNotFoundError(f"ONNX CRNN model not found: {path}")
            if not quiet:
                backend = "triton" if should_try_triton() else "local"
                _log_info(
                    f"OCR initializing CRNN predictor backend={backend} model_path={path}"
                )
            _CRNN_PREDICTOR = TextPredictor(path, quiet=quiet)
        return _CRNN_PREDICTOR


def warm_vision_models(*, quiet: bool = True) -> None:
    """Eagerly warm YOLO + CRNN (Triton health check or local ONNX sessions)."""
    from cua_mcp.vision_backend import should_try_triton

    started = time.perf_counter()
    if should_try_triton():
        from cua_mcp.vision_triton import triton_ready

        if triton_ready():
            elapsed = time.perf_counter() - started
            _log_info(f"Vision warmup: Triton server ready elapsed_s={elapsed:.3f}")
            return
        elapsed = time.perf_counter() - started
        _log_info(
            f"Vision warmup: Triton not ready elapsed_s={elapsed:.3f}, "
            "loading local ONNX fallback"
        )

    from cua_mcp.yolo_onnx import DEFAULT_YOLO_ONNX_PATH, get_cached_cpu_session

    get_cached_cpu_session(DEFAULT_YOLO_ONNX_PATH)
    _get_ocr_predictor(quiet=quiet)
    if not quiet:
        elapsed = time.perf_counter() - started
        _log_info(f"Vision warmup: local ONNX loaded elapsed_s={elapsed:.3f}")


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
) -> list[list[str]]:
    """
    Run CRNN OCR on each ``(x, y, w, h)`` crop in ``boxes``.

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
) -> list[list[str]]:
    """Run CRNN OCR on crops in width-sorted, zero-padded batches."""
    if batch_size < 1:
        batch_size = _DEFAULT_CRNN_BATCH_SIZE

    line_height = _effective_line_height(line_height)
    results: list[list[str]] = [[] for _ in crops]
    valid: list[tuple[int, np.ndarray]] = []

    for index, crop in enumerate(crops):
        line_image = _prepare_crop_line_image(crop, line_height)
        if line_image is not None:
            valid.append((index, line_image))

    valid.sort(key=lambda item: item[1].shape[1])

    batch_count = 0
    infer_total_s = 0.0
    started = time.perf_counter()

    for start in range(0, len(valid), batch_size):
        chunk = valid[start : start + batch_size]
        if not chunk:
            continue

        max_w = max(image.shape[1] for _, image in chunk)
        batch = np.zeros((len(chunk), line_height, max_w), dtype=np.float32)
        for row, (_, image) in enumerate(chunk):
            h_img, w_img = image.shape[:2]
            batch[row, :h_img, :w_img] = image

        try:
            batch_started = time.perf_counter()
            predicted_texts = predictor.predict_images(batch)
            infer_total_s += time.perf_counter() - batch_started
            batch_count += 1
            for row, (orig_index, _) in enumerate(chunk):
                if predicted_texts and row < len(predicted_texts):
                    text = predicted_texts[row]
                    results[orig_index] = [text] if text else []
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
) -> list[str]:
    """Run CRNN OCR on a single crop; return raw ``predict_images`` token strings."""
    preds = _ocr_crops_batched([bgr_crop], predictor, line_height, batch_size=1)
    return preds[0] if preds else []
