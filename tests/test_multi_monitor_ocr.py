from __future__ import annotations

from pathlib import Path

import pytest

from cua_mcp.read_screen_text import ocr_image
from src.common import monitor_prompt
from src.common.run_state import RunStateManager


def test_selected_eye_monitor_indices_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYE_MONITOR_INDICES", "1,2")
    monkeypatch.setenv("EYE_MONITOR_INDEX", "1")
    assert monitor_prompt.selected_eye_monitor_indices() == [1, 2]


def test_selected_eye_monitor_indices_single(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYE_MONITOR_INDICES", raising=False)
    monkeypatch.setenv("EYE_MONITOR_INDEX", "3")
    assert monitor_prompt.selected_eye_monitor_indices() == [3]


def test_offset_region_shifts_bbox_and_center() -> None:
    region = ((10, 20, 30, 40), (25, 40), ["hello"])
    offset = ocr_image._offset_region(region, 1920, 0)
    assert offset[0] == (1930, 20, 30, 40)
    assert offset[1] == (1945, 40)
    assert offset[2] == ["hello"]


def test_get_coordinates_from_selected_monitors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = RunStateManager(runs_root=tmp_path)
    paths = manager.init_run("multi-monitor-ocr-test", "test_multi_monitor_ocr")

    captured: list[tuple[Path, int]] = []
    ocr_calls: list[str] = []

    def _fake_capture(dest: Path, monitor_index: int) -> int:
        captured.append((dest, monitor_index))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-png")
        return monitor_index

    def _fake_ocr(image_path: str, **kwargs: object) -> list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]]:
        ocr_calls.append(image_path)
        if image_path.endswith("_mon1.png"):
            return [((5, 6, 10, 12), (10, 12), ["left"])]
        if image_path.endswith("_mon2.png"):
            return [((7, 8, 11, 13), (12, 14), ["right"])]
        return []

    offsets = {1: (0, 0), 2: (1920, 0)}

    monkeypatch.setattr(ocr_image, "selected_eye_monitor_indices", lambda: [1, 2])
    monkeypatch.setattr(ocr_image, "capture_monitor_to_file", _fake_capture)
    monkeypatch.setattr(ocr_image, "get_coordinates_from_image_path", _fake_ocr)
    monkeypatch.setattr(ocr_image, "active_monitor_offset", lambda idx: offsets[idx])
    monkeypatch.setattr(ocr_image, "ts_name", lambda: "20260610_test")
    monkeypatch.setattr(ocr_image, "get_run_state_manager", lambda: manager)

    regions, image_paths = ocr_image.get_coordinates_from_selected_monitors()

    assert len(captured) == 2
    assert captured[0][1] == 1
    assert captured[1][1] == 2
    assert captured[0][0].parent == paths.yolo_ocr_dir
    assert captured[1][0].name.endswith("_mon2.png")

    assert len(ocr_calls) == 2
    assert len(image_paths) == 2

    assert len(regions) == 2
    assert regions[0][1] == (10, 12)
    assert regions[0][2] == ["left"]
    assert regions[1][1] == (1932, 14)
    assert regions[1][2] == ["right"]
