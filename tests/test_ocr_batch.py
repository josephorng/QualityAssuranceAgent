from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import pytest

from cua_mcp.read_screen_text import ocr_image
from cua_mcp.read_screen_text.ocr_image import (
    DEFAULT_CONF_YOLOV26_END2END,
    _expand_box,
    _get_ocr_predictor,
    _merge_overlapping_boxes,
    _ocr_crop_predicted_texts,
    _ocr_crops_batched,
    _prepare_crop_line_image,
    _sort_boxes_reading_order,
    _yolo_boxes,
    get_coordinates_from_image_path,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRS = [
    ROOT / "cua_mcp" / "read_screen_text" / "images",
    ROOT / "cua_mcp" / "read_screen_text",
]


def _sample_images() -> list[Path]:
    images: list[Path] = []
    for image_dir in IMAGE_DIRS:
        if not image_dir.exists():
            continue
        images.extend(
            sorted(
                p
                for p in image_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        )
    return images


def _ocr_text_similarity(a: list[str], b: list[str]) -> float:
    text_a = "".join(a)
    text_b = "".join(b)
    if text_a == text_b:
        return 1.0
    return SequenceMatcher(None, text_a, text_b).ratio()


def _minor_ocr_diff_ok(a: list[str], b: list[str], *, min_ratio: float = 0.9) -> bool:
    text_a = "".join(a)
    text_b = "".join(b)
    if text_a == text_b:
        return True

    len_diff = abs(len(text_a) - len(text_b))
    max_len = max(len(text_a), len(text_b), 1)
    shorter, longer = (text_a, text_b) if len(text_a) <= len(text_b) else (text_b, text_a)
    if len_diff <= 2 and shorter and shorter in longer:
        return True

    ratio = _ocr_text_similarity(a, b)
    if len_diff > 2:
        return False
    if max_len <= 4:
        return ratio >= 0.65
    if max_len <= 8:
        return ratio >= 0.75
    return ratio >= min_ratio


def _make_text_crop(width: int, *, value: int = 180) -> np.ndarray:
    crop = np.full((24, width, 3), value, dtype=np.uint8)
    cv2.putText(
        crop,
        "A",
        (2, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return crop


def test_prepare_crop_line_image_invalid() -> None:
    assert _prepare_crop_line_image(np.zeros((0, 10, 3), dtype=np.uint8), 32) is None
    assert _prepare_crop_line_image(np.zeros((1, 10, 3), dtype=np.uint8), 32) is None
    assert _prepare_crop_line_image(np.zeros((10, 1, 3), dtype=np.uint8), 32) is None


def test_ocr_crops_batched_padding() -> None:
    predictor = _get_ocr_predictor(quiet=True)
    crops = [_make_text_crop(width) for width in (10, 50, 100)]

    serial_preds = [_ocr_crop_predicted_texts(crop, predictor, 32) for crop in crops]
    batch_preds = _ocr_crops_batched(crops, predictor, 32, batch_size=16)

    assert batch_preds == serial_preds


@pytest.mark.parametrize(
    "image_path",
    _sample_images(),
    ids=lambda p: Path(p).name,
)
def test_batch_matches_serial_on_sample_images(image_path: Path) -> None:
    serial_regions = get_coordinates_from_image_path(str(image_path), batch_size=1)
    batch_regions = get_coordinates_from_image_path(str(image_path), batch_size=16)

    assert len(batch_regions) == len(serial_regions)
    for batch_region, serial_region in zip(batch_regions, serial_regions, strict=True):
        assert batch_region[0] == serial_region[0]
        assert batch_region[1] == serial_region[1]


@pytest.mark.parametrize(
    "image_path",
    _sample_images(),
    ids=lambda p: Path(p).name,
)
def test_batch_ocr_text_near_serial_on_fixed_crops(image_path: Path) -> None:
    bgr = cv2.imread(str(image_path))
    assert bgr is not None

    img_h, img_w = bgr.shape[:2]
    boxes = _yolo_boxes(bgr, conf_threshold=DEFAULT_CONF_YOLOV26_END2END)
    boxes = _merge_overlapping_boxes(boxes)
    boxes = _sort_boxes_reading_order(boxes)
    boxes = [_expand_box(x, y, w, h, img_w, img_h) for x, y, w, h in boxes]
    crops = [
        bgr[y : y + h, x : x + w]
        for x, y, w, h in boxes
        if bgr[y : y + h, x : x + w].size
    ]
    if not crops:
        pytest.skip(f"No OCR crops for {image_path.name}")

    predictor = _get_ocr_predictor(quiet=True)
    serial_preds = _ocr_crops_batched(crops, predictor, 32, batch_size=1)
    batch_preds = _ocr_crops_batched(crops, predictor, 32, batch_size=16)

    mismatches = [
        (serial, batch)
        for serial, batch in zip(serial_preds, batch_preds, strict=True)
        if serial != batch
    ]
    mismatch_rate = len(mismatches) / len(crops)
    assert mismatch_rate <= 0.35, (
        f"{image_path.name}: batch OCR mismatch rate {mismatch_rate:.0%} too high"
    )
    for serial, batch in mismatches:
        assert _minor_ocr_diff_ok(serial, batch), (
            f"{image_path.name}: large OCR diff serial={serial!r} batch={batch!r}"
        )


def test_sample_images_available() -> None:
    images = _sample_images()
    assert images, "No OCR sample images found for batch regression tests"


def test_default_batch_size_constant() -> None:
    assert ocr_image._DEFAULT_CRNN_BATCH_SIZE == 16
