from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_ocr_viewer_tk import (
    OcrLine,
    _discover_run_images,
    _smallest_box_hit_index,
    load_ocr_lines,
    load_yolo_lines,
    resolve_image_lines,
)


def test_smallest_box_hit_index_prefers_smallest_overlap() -> None:
    lines = [
        OcrLine(box=(0, 0, 100, 100), text="large"),
        OcrLine(box=(40, 40, 10, 10), text="small"),
        OcrLine(box=(80, 80, 10, 10), text="other"),
    ]
    assert _smallest_box_hit_index(lines, 45, 45) == 1
    assert _smallest_box_hit_index(lines, 85, 85) == 2
    assert _smallest_box_hit_index(lines, 5, 5) == 0
    assert _smallest_box_hit_index(lines, 200, 200) is None


def test_resolve_prefers_json_over_yolo(tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"png")
    json_path = image_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"lines": [[[1, 2, 3, 4], "hello"]]}),
        encoding="utf-8",
    )

    with patch("app_ocr_viewer_tk.load_yolo_lines") as load_yolo:
        lines, status = resolve_image_lines(
            image_path,
            yolo_conf_threshold=0.5,
            allow_yolo=True,
        )

    load_yolo.assert_not_called()
    assert len(lines) == 1
    assert lines[0].text == "hello"
    assert "OCR lines" in status


def test_resolve_without_json_skips_yolo_when_not_allowed(tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"png")

    with patch("app_ocr_viewer_tk.load_yolo_lines") as load_yolo:
        lines, status = resolve_image_lines(
            image_path,
            yolo_conf_threshold=0.5,
            allow_yolo=False,
        )

    load_yolo.assert_not_called()
    assert lines == []
    assert "Reload YOLO" in status


def test_resolve_uses_yolo_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"png")
    cache: dict[tuple[str, float], tuple[list[OcrLine], str]] = {}
    yolo_result = ([OcrLine(box=(0, 0, 1, 1), text="cached")], "Loaded 1 YOLO detections")

    with patch("app_ocr_viewer_tk.load_yolo_lines", return_value=yolo_result) as load_yolo:
        first = resolve_image_lines(
            image_path,
            yolo_conf_threshold=0.25,
            allow_yolo=True,
            yolo_cache=cache,
        )
        second = resolve_image_lines(
            image_path,
            yolo_conf_threshold=0.25,
            allow_yolo=True,
            yolo_cache=cache,
        )

    load_yolo.assert_called_once()
    assert first == second
    assert first[0][0].text == "cached"


def test_resolve_force_yolo_bypasses_json(tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"png")
    json_path = image_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"lines": [[[1, 2, 3, 4], "from json"]]}),
        encoding="utf-8",
    )
    yolo_result = ([OcrLine(box=(5, 6, 7, 8), text="from yolo")], "Loaded 1 YOLO detections")

    with patch("app_ocr_viewer_tk.load_yolo_lines", return_value=yolo_result) as load_yolo:
        lines, status = resolve_image_lines(
            image_path,
            yolo_conf_threshold=0.5,
            allow_yolo=True,
            force_yolo=True,
        )

    load_yolo.assert_called_once()
    assert lines[0].text == "from yolo"
    assert "YOLO" in status


def test_discover_run_images_finds_screenshot_recording_shots(tmp_path: Path) -> None:
    run_dir = tmp_path / "screen_record_test"
    shots = run_dir / "screenshots"
    yolo = run_dir / "yolo_ocr"
    shots.mkdir(parents=True)
    yolo.mkdir(parents=True)
    shot = shots / "event_001.jpeg"
    shot.write_bytes(b"jpeg")
    (yolo / "event_001.json").write_text(
        json.dumps(
            {
                "image_path": str(shot),
                "candidates": [
                    {"bbox": [10, 20, 30, 40], "class_name": "text", "text": "hello"},
                ],
            }
        ),
        encoding="utf-8",
    )

    images = _discover_run_images(run_dir)
    assert len(images) == 1
    assert images[0] == shot


def test_resolve_loads_yolo_ocr_candidates_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "screen_record_test"
    shots = run_dir / "screenshots"
    yolo = run_dir / "yolo_ocr"
    shots.mkdir(parents=True)
    yolo.mkdir(parents=True)
    shot = shots / "event_001.jpeg"
    shot.write_bytes(b"jpeg")
    (yolo / "event_001.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"bbox": [1, 2, 3, 4], "class_name": "text", "text": "間間Gemini"},
                    {"bbox": [5, 6, 7, 8], "class_name": "input", "text": None},
                ],
            }
        ),
        encoding="utf-8",
    )

    lines, status = resolve_image_lines(
        shot,
        yolo_conf_threshold=0.5,
        allow_yolo=False,
        run_dir=run_dir,
    )

    assert len(lines) == 2
    assert lines[0].text == "間間Gemini"
    assert lines[1].class_name == "input"
    assert "vision candidates" in status


def test_load_ocr_lines_candidates_format(tmp_path: Path) -> None:
    json_path = tmp_path / "event_001.json"
    json_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {"bbox": [0, 0, 10, 10], "class_name": "element", "text": "e", "icons": [{"chinese_id": "加號"}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    lines, status = load_ocr_lines(json_path)
    assert len(lines) == 1
    assert lines[0].chinese_ids == ("加號",)
    assert "vision candidates" in status


def test_load_yolo_lines_reads_unicode_path(tmp_path: Path) -> None:
    from PIL import Image

    folder = tmp_path / "中文目录"
    folder.mkdir()
    image_path = folder / "截图.png"
    Image.new("RGB", (8, 8), (0, 128, 255)).save(image_path)

    with patch("app_ocr_viewer_tk._build_candidates_from_bgr", return_value=[]) as build:
        lines, status = load_yolo_lines(image_path, yolo_conf_threshold=0.5)

    build.assert_called_once()
    assert lines == []
    assert "Could not read" not in status
    assert "YOLO detections" in status
