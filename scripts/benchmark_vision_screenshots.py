"""Benchmark YOLO+OCR vision on recording screenshots: sequential vs parallel."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cua_mcp.select_mouse_target import _detect_mouse_targets_from_bgr
from src.common.io_utils import imread_bgr


@dataclass
class ImageResult:
    name: str
    elapsed_s: float
    detection_count: int
    error: str | None = None


def _image_paths(screenshots_dir: Path) -> list[Path]:
    paths = sorted(
        p
        for p in screenshots_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png", ".webp"}
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {screenshots_dir}")
    return paths


def _run_one(path: Path) -> ImageResult:
    started = time.perf_counter()
    try:
        bgr = imread_bgr(path)
        if bgr is None:
            raise RuntimeError("imread_bgr returned None")
        detections = _detect_mouse_targets_from_bgr(bgr)
        count = len(detections)
        error = None
    except Exception as exc:
        count = 0
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    return ImageResult(path.name, elapsed, count, error)


def _warmup(path: Path) -> None:
    result = _run_one(path)
    if result.error:
        raise RuntimeError(f"Warmup failed on {path.name}: {result.error}")


def _run_sequential(paths: list[Path]) -> tuple[list[ImageResult], float]:
    started = time.perf_counter()
    results = [_run_one(path) for path in paths]
    total = time.perf_counter() - started
    return results, total


def _run_parallel(paths: list[Path], *, max_workers: int) -> tuple[list[ImageResult], float]:
    started = time.perf_counter()
    results_by_name: dict[str, ImageResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, path): path for path in paths}
        for future in as_completed(futures):
            result = future.result()
            results_by_name[result.name] = result
    total = time.perf_counter() - started
    ordered = [results_by_name[path.name] for path in paths]
    return ordered, total


def _print_results(label: str, results: list[ImageResult], total_s: float) -> None:
    print(f"\n=== {label} ===")
    for row in results:
        status = "OK" if row.error is None else row.error
        print(
            f"  {row.name:22s}  {row.elapsed_s:6.2f}s  "
            f"detections={row.detection_count:4d}  {status}"
        )
    cpu_sum = sum(row.elapsed_s for row in results)
    print(f"  {'TOTAL wall':22s}  {total_s:6.2f}s")
    print(f"  {'SUM per-image CPU':22s}  {cpu_sum:6.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=ROOT / "recordings" / "控制面板" / "screenshots",
        help="Folder of screenshot images to benchmark",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker count (matches RECORDING_VISION_WORKERS default)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not run a warmup inference before timed runs",
    )
    args = parser.parse_args()

    screenshots_dir = args.screenshots_dir.resolve()
    paths = _image_paths(screenshots_dir)
    workers = max(1, min(args.workers, len(paths)))

    print(f"Screenshots: {screenshots_dir}")
    print(f"Images: {len(paths)}")
    print(f"Parallel workers: {workers}")

    if not args.skip_warmup:
        print("\nWarming up Triton/YOLO+OCR...")
        warmup_started = time.perf_counter()
        _warmup(paths[0])
        print(f"Warmup done in {time.perf_counter() - warmup_started:.2f}s ({paths[0].name})")

    seq_results, seq_total = _run_sequential(paths)
    par_results, par_total = _run_parallel(paths, max_workers=workers)

    _print_results("Run 1 — sequential (one image at a time)", seq_results, seq_total)
    _print_results(f"Run 2 — parallel ({workers} workers)", par_results, par_total)

    speedup = seq_total / par_total if par_total > 0 else float("inf")
    saved = seq_total - par_total
    print("\n=== Comparison ===")
    print(f"  Sequential wall time : {seq_total:.2f}s")
    print(f"  Parallel wall time   : {par_total:.2f}s")
    print(f"  Time saved           : {saved:.2f}s ({saved / seq_total * 100:.1f}% faster)" if seq_total else "")
    print(f"  Speedup              : {speedup:.2f}x")

    errors = [r for r in seq_results + par_results if r.error]
    if errors:
        print(f"\nWarning: {len(errors)} run(s) reported errors.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
