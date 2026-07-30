"""Capture a physical monitor and prepare YOLO/OCR data for the OCR viewer.

Run from the repository root:

    python scripts/capture_monitors_for_ocr_viewer.py

The generated ``runs/fake_recording_<timestamp>`` directory can be opened
directly from ``app_ocr_viewer_tk.py``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mss
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cua_mcp.read_screen_text.ocr_image import _log_info
from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.select_ui_element import UiDetection
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END
from src.common.io_utils import write_json
from src.common.run_state import unique_run_folder_name


def _candidate_to_dict(candidate: UiDetection) -> dict[str, Any]:
    return {
        "bbox": list(candidate.bbox),
        "center": [candidate.cx, candidate.cy],
        "class_id": candidate.class_id,
        "class_name": candidate.class_name,
        "text": candidate.text,
        "icons": candidate.icons,
    }


def _select_monitor(monitors: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt for the leftmost or rightmost monitor when multiple are present."""
    if len(monitors) == 1:
        return monitors[0]

    left_monitor = min(monitors, key=lambda monitor: (monitor["left"], monitor["top"]))
    right_monitor = max(monitors, key=lambda monitor: (monitor["left"], monitor["top"]))
    print("Multiple monitors detected:")
    print(
        f"  [L] Left:  monitor {left_monitor['monitor_index']} "
        f"({left_monitor['width']}x{left_monitor['height']} at "
        f"{left_monitor['left']},{left_monitor['top']})"
    )
    print(
        f"  [R] Right: monitor {right_monitor['monitor_index']} "
        f"({right_monitor['width']}x{right_monitor['height']} at "
        f"{right_monitor['left']},{right_monitor['top']})"
    )

    while True:
        choice = input("Select monitor [L/R]: ").strip().lower()
        if choice in {"l", "left"}:
            return left_monitor
        if choice in {"r", "right"}:
            return right_monitor
        print("Please enter L for the left monitor or R for the right monitor.")


def _capture_physical_monitors(screenshots_dir: Path) -> list[dict[str, Any]]:
    """Capture the selected MSS physical monitor, excluding virtual desktop 0."""
    captured: list[dict[str, Any]] = []
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    with mss.mss() as sct:
        monitors = [
            {
                "monitor_index": monitor_index,
                "left": int(monitor["left"]),
                "top": int(monitor["top"]),
                "width": int(monitor["width"]),
                "height": int(monitor["height"]),
            }
            for monitor_index, monitor in enumerate(sct.monitors[1:], start=1)
        ]
        if not monitors:
            raise RuntimeError("MSS did not report any physical monitors")

        selected = _select_monitor(monitors)
        monitor_index = selected["monitor_index"]
        image_path = screenshots_dir / f"monitor_{monitor_index:03d}.png"
        shot = sct.grab(sct.monitors[monitor_index])
        Image.frombytes("RGB", shot.size, shot.rgb).save(image_path)
        selected["image_path"] = image_path
        captured.append(selected)

    if not captured:
        raise RuntimeError("MSS did not report any physical monitors")
    return captured


def _analyze_capture(
    capture: dict[str, Any],
    yolo_ocr_dir: Path,
    *,
    yolo_conf_threshold: float,
) -> Path:
    image_path = Path(capture["image_path"])
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Could not read captured image: {image_path}")

    candidates = _build_candidates_from_bgr(
        bgr,
        yolo_conf_threshold=yolo_conf_threshold,
    )
    candidate_dicts = [_candidate_to_dict(candidate) for candidate in candidates]
    output_path = yolo_ocr_dir / f"{image_path.stem}.json"
    write_json(
        output_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "fake_recording_monitor_capture",
            "image_path": str(image_path.resolve()),
            "image_name": image_path.name,
            "monitor_index": capture["monitor_index"],
            "monitor_offset": [capture["left"], capture["top"]],
            "monitor_size": [capture["width"], capture["height"]],
            "yolo_conf_threshold": yolo_conf_threshold,
            "detection_count": len(candidate_dicts),
            "candidates": candidate_dicts,
            "lines": [
                [candidate["bbox"], candidate.get("text") or ""]
                for candidate in candidate_dicts
            ],
            "text": "\n".join(
                str(candidate["text"]).strip()
                for candidate in candidate_dicts
                if candidate.get("text")
            ),
        },
    )
    return output_path


def create_fake_recording(
    runs_root: Path = ROOT_DIR / "runs",
    *,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> Path:
    """Capture and analyze the selected physical monitor, returning the run folder."""
    run_dir = Path(runs_root) / unique_run_folder_name("fake_recording")
    screenshots_dir = run_dir / "screenshots"
    yolo_ocr_dir = run_dir / "yolo_ocr"
    yolo_ocr_dir.mkdir(parents=True, exist_ok=True)

    captured = _capture_physical_monitors(screenshots_dir)
    output_files: list[Path] = []
    for capture in captured:
        image_path = Path(capture["image_path"])
        print(f"Running YOLO + OCR for monitor {capture['monitor_index']}: {image_path.name}")
        output_files.append(
            _analyze_capture(
                capture,
                yolo_ocr_dir,
                yolo_conf_threshold=yolo_conf_threshold,
            )
        )

    write_json(
        run_dir / "session.json",
        {
            "run_id": run_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "fake_recording_monitor_capture",
            "monitor_count": len(captured),
            "screenshots": [
                str(Path(capture["image_path"]).relative_to(run_dir))
                for capture in captured
            ],
            "yolo_ocr_results": [
                str(output_path.relative_to(run_dir)) for output_path in output_files
            ],
        },
    )
    _log_info(f"Fake recording prepared for OCR viewer: {run_dir}")
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and capture a monitor, then create YOLO/OCR results."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT_DIR / "runs",
        help="Parent directory for the generated fake recording.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONF_YOLOV26_END2END,
        help="YOLO confidence threshold from 0 to 1.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    return args


def main() -> int:
    args = _parse_args()
    try:
        run_dir = create_fake_recording(
            args.runs_root,
            yolo_conf_threshold=args.confidence,
        )
    except Exception as exc:
        print(f"Capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Ready: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
