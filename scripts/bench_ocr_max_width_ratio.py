"""Sweep CRNN max_width_ratio on real move_mouse screenshots (local Triton)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from cua_mcp.read_screen_text.get_coordinates import (
    _merge_overlapping_boxes,
    _sort_boxes_reading_order,
    _yolo_boxes,
)
from cua_mcp.read_screen_text.ocr_image import (
    _expand_box,
    _get_ocr_predictor,
    _ocr_crops_batched_detailed,
    _partition_width_sorted_batches,
    _prepare_crop_line_image,
)
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END

ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "runs"
    / "recording_20260921_011432_920359_20260921_011546_961977"
    / "yolo_ocr"
)
IMAGES = sorted(RUN.glob("*_mon*.png"))
RATIOS = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 6.0, 8.0]
REPEATS = 5
WARMUP_ROUNDS = 2
LINE_H = 32
BATCH = 64


def _collect_crops(image_path: Path) -> list[np.ndarray]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"failed to read {image_path}")
    img_h, img_w = bgr.shape[:2]
    boxes = _yolo_boxes(bgr, conf_threshold=DEFAULT_CONF_YOLOV26_END2END)
    merged = _merge_overlapping_boxes(boxes)
    ordered = _sort_boxes_reading_order(merged)
    expanded = [_expand_box(x, y, w, h, img_w, img_h) for x, y, w, h in ordered]
    crops: list[np.ndarray] = []
    for x, y, w, h in expanded:
        crop = bgr[y : y + h, x : x + w]
        if crop.size:
            crops.append(crop)
    return crops


def _line_widths(crops: list[np.ndarray]) -> list[int]:
    widths: list[int] = []
    for crop in crops:
        line = _prepare_crop_line_image(crop, LINE_H)
        if line is not None:
            widths.append(int(line.shape[1]))
    return sorted(widths)


def _pad_waste(widths: list[int], ratio: float) -> tuple[int, int, float]:
    items = list(widths)
    chunks = _partition_width_sorted_batches(
        items,
        width_of=lambda w: w,
        batch_size=BATCH,
        max_width_ratio=ratio,
    )
    useful = sum(widths)
    padded = 0
    for chunk in chunks:
        padded += len(chunk) * max(chunk)
    waste = 0.0 if padded <= 0 else 1.0 - (useful / padded)
    return len(chunks), padded - useful, waste


def main() -> None:
    if not IMAGES:
        raise SystemExit(f"no images under {RUN}")

    print(f"Triton images ({len(IMAGES)}):")
    per_image_crops: list[tuple[str, list[np.ndarray]]] = []
    for path in IMAGES:
        crops = _collect_crops(path)
        widths = _line_widths(crops)
        print(
            f"  {path.name}: crops={len(crops)} "
            f"w_min={min(widths) if widths else 0} "
            f"w_max={max(widths) if widths else 0}"
        )
        per_image_crops.append((path.name, crops))

    predictor = _get_ocr_predictor(quiet=True)
    # Warmup (same work as a full pass so Triton/ORT stay hot).
    for _ in range(WARMUP_ROUNDS):
        for _name, crops in per_image_crops:
            if crops:
                _ocr_crops_batched_detailed(
                    crops, predictor, LINE_H, max_width_ratio=1.5
                )

    baseline_by_image: dict[str, dict[int, str]] = {}
    for name, crops in per_image_crops:
        detailed = _ocr_crops_batched_detailed(
            crops, predictor, LINE_H, max_width_ratio=1.5
        )
        baseline_by_image[name] = {idx: text for idx, text, _ in detailed}

    print(
        "\nratio | batches | pad_waste_px | waste% | "
        "median_s | mean_s | min_s | mismatch%"
    )
    print("-" * 88)
    results: list[tuple[float, float, int, float]] = []
    for ratio in RATIOS:
        all_widths: list[int] = []
        for _name, crops in per_image_crops:
            all_widths.extend(_line_widths(crops))
        batches, waste_px, waste_frac = _pad_waste(all_widths, ratio)

        times: list[float] = []
        mismatch = 0
        total_preds = 0
        for rep in range(REPEATS):
            t0 = time.perf_counter()
            for name, crops in per_image_crops:
                detailed = _ocr_crops_batched_detailed(
                    crops,
                    predictor,
                    LINE_H,
                    max_width_ratio=ratio,
                )
                if rep == REPEATS - 1:
                    base = baseline_by_image[name]
                    for idx, text, _spans in detailed:
                        total_preds += 1
                        if base.get(idx, "") != text:
                            mismatch += 1
            times.append(time.perf_counter() - t0)

        mismatch_pct = 100.0 * mismatch / max(total_preds, 1)
        mean_s = float(np.mean(times))
        median_s = float(np.median(times))
        min_s = float(np.min(times))
        label = f"{ratio:g}"
        print(
            f"{label:>5} | {batches:7d} | {waste_px:12d} | {waste_frac*100:5.1f} | "
            f"{median_s:8.3f} | {mean_s:6.3f} | {min_s:5.3f} | {mismatch_pct:6.2f}"
        )
        results.append((ratio, median_s, batches, mismatch_pct))

    # Prefer lowest median latency; allow tiny text drift vs 1.5 (<=2%).
    eligible = [r for r in results if r[3] <= 2.0]
    pool = eligible or results
    best = min(pool, key=lambda r: (r[1], r[0]))
    print(
        f"\nbest_ratio={best[0]:g} median_s={best[1]:.3f} "
        f"batches={best[2]} mismatch%={best[3]:.2f}"
    )


if __name__ == "__main__":
    main()
