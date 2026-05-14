"""
Run OCR (YOLO + CRNN) on debug.png for manual inspection.

Usage (from repository root):
    python cua_mcp/read_screen_text/debug_ocr_image.py

Or with a custom image:
    python cua_mcp/read_screen_text/debug_ocr_image.py --image path/to/image.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cua_mcp.read_screen_text.ocr_image import (  # noqa: E402
    format_coordinate_text_from_regions,
    get_coordinates_from_path,
    get_text_boxes_from_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug OCR pipeline on a fixed test image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parent / "debug_0.png",
        help="Image path (default: debug.png next to this script)",
    )
    parser.add_argument(
        "--line-height",
        type=int,
        default=32,
        help="CRNN resize line height (default: 32)",
    )
    args = parser.parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    print(f"image: {image_path}")
    t0 = time.perf_counter()
    offset, regions = get_coordinates_from_path(str(image_path), line_height=args.line_height)
    elapsed = time.perf_counter() - t0
    print(f"offset: {offset}")
    print(f"regions: {len(regions)}")
    print(f"elapsed: {elapsed:.3f}s")
    print()

    boxes_only = get_text_boxes_from_path(str(image_path))
    print(f"yolo_boxes_only (no CRNN): {len(boxes_only)}")
    print()

    hint = format_coordinate_text_from_regions(regions)
    print("--- format_coordinate_text_from_regions ---")
    print(hint if hint else "(empty)")
    print()

    print("--- per-region detail ---")
    for i, (bbox, center, preds) in enumerate(regions):
        text = "".join(preds).strip()
        print(f"{i:3d}  bbox={bbox}  center={center}  text={text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
