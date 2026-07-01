from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_ocr_viewer_tk import OcrLine, _smallest_box_hit_index, resolve_image_lines


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
