from __future__ import annotations

import re
from pathlib import Path

import pytest

from cua_mcp.read_screen_text.ocr_image import (
    format_coordinate_text_from_regions,
    get_coordinates_from_path,
)

ROOT = Path(__file__).resolve().parents[1]


# Extend this registry as new function folders are added under `cua_mcp/`.
FUNCTION_FOLDERS: dict[str, dict[str, object]] = {
    "get_coordinates": {
        "path": ROOT / "cua_mcp" / "read_screen_text",
        "required_files": [
            "ocr_image.py",
            "inference.py",
            "yolo_best.pt",
            "crnn_cfc_model.pt",
            "char_dict.json",
            "char_decode_dict.json",
            "model_config.json",
        ],
        "image_dir": ROOT / "cua_mcp" / "read_screen_text" / "images",
    }
}

OCR_LINE_RE = re.compile(r"^\[(\d+),(\d+)(?:,(\d+),(\d+))?\]\s?.*$")


def _sample_images(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


@pytest.mark.parametrize("folder_name,meta", FUNCTION_FOLDERS.items())
def test_function_folder_required_files_exist(folder_name: str, meta: dict[str, object]) -> None:
    base_path = meta["path"]
    assert isinstance(base_path, Path), f"{folder_name}: invalid path metadata"
    assert base_path.exists(), f"{folder_name}: folder missing: {base_path}"

    required_files = meta["required_files"]
    assert isinstance(required_files, list), f"{folder_name}: required_files must be a list"
    for rel in required_files:
        assert (base_path / str(rel)).exists(), f"{folder_name}: missing required file: {rel}"


def test_get_coordinates_has_sample_images() -> None:
    image_dir = FUNCTION_FOLDERS["get_coordinates"]["image_dir"]
    assert isinstance(image_dir, Path)
    assert image_dir.exists(), f"images folder missing: {image_dir}"
    images = _sample_images(image_dir)
    assert images, "No sample images found for get_coordinates tests"


@pytest.mark.parametrize(
    "image_path",
    _sample_images(FUNCTION_FOLDERS["get_coordinates"]["image_dir"]),  # type: ignore[arg-type]
    ids=lambda p: Path(p).name,
)
def test_get_coordinates_tool_returns_bbox_lines(image_path: Path) -> None:
    _offset, regions = get_coordinates_from_path(str(image_path))
    assert isinstance(regions, list)
    assert regions, f"OCR returned no regions for {image_path.name}"
    output = format_coordinate_text_from_regions(regions)
    assert output, f"OCR returned empty hint for {image_path.name}"

    lines = [line for line in output.splitlines() if line.strip()]
    assert lines, f"No OCR lines returned for {image_path.name}"
    for line in lines:
        assert OCR_LINE_RE.match(line), f"Line does not match expected coordinate format: {line}"


@pytest.mark.parametrize(
    "image_path",
    _sample_images(FUNCTION_FOLDERS["get_coordinates"]["image_dir"]),  # type: ignore[arg-type]
    ids=lambda p: Path(p).name,
)
def test_get_coordinates_helper_matches_tool_type(image_path: Path) -> None:
    _offset, regions = get_coordinates_from_path(str(image_path))
    helper_output = format_coordinate_text_from_regions(regions)
    assert isinstance(helper_output, str)

