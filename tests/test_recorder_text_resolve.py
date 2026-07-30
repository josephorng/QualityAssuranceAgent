from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT, YOLO_CLASS_TEXT
from src.recorder.models import RecordedEvent
from src.recorder.text_resolve import event_with_resolved_text, resolve_text_input_text
from src.recorder.vision_context import extract_nearest_text


def test_extract_nearest_text_prefers_nearest_with_text() -> None:
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        _detection_from_bbox((40, 40, 20, 20), YOLO_CLASS_TEXT, text="Far"),
        _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="Near"),
    ]
    assert extract_nearest_text(bgr, detections, 10, 10) == "Near"


def test_extract_nearest_text_ocrs_input_bbox_when_no_text() -> None:
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        _detection_from_bbox((5, 5, 40, 20), YOLO_CLASS_INPUT),
    ]
    with patch(
        "src.recorder.vision_context._ocr_boxes_on_bgr",
        return_value=[["你", "好"]],
    ):
        assert extract_nearest_text(bgr, detections, 10, 10) == "你好"


def test_extract_nearest_text_prefers_input_ocr_over_nearby_page_text() -> None:
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="Page text"),
        _detection_from_bbox((5, 40, 80, 20), YOLO_CLASS_INPUT),
        _detection_from_bbox((10, 45, 60, 10), YOLO_CLASS_TEXT, text="Visible input"),
    ]
    with patch(
        "src.recorder.vision_context._ocr_boxes_on_bgr",
    ) as ocr_mock:
        assert extract_nearest_text(bgr, detections, 10, 10) == "Visible input"
    ocr_mock.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_text_input_always_uses_vision_when_ocr_finds_text(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="chrome",
        cursor_xy=(10, 10),
        screenshot_path="",
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (10, 10),
        "candidates": [],
        "detection_count": 1,
        "bgr": np.zeros((100, 100, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="Chrome"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "Chrome"
    assert resolved["source"] == "ocr"
    assert resolved["meaningful"] is None


@pytest.mark.asyncio
async def test_resolve_text_input_uses_ocr_with_anchor(
    tmp_path,
) -> None:
    shot = tmp_path / "event.jpeg"
    shot.write_bytes(b"x")
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="nihao",
        anchor_click_xy=(110, 210),
        monitor_offset=(100, 200),
        screenshot_path=str(shot),
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (10, 10),
        "candidates": [],
        "detection_count": 1,
        "bgr": np.zeros((100, 100, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="你好"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="你好",
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "你好"
    assert resolved["source"] == "ocr"
    assert resolved["meaningful"] is None


@pytest.mark.asyncio
async def test_resolve_text_input_keeps_recorded_without_vision_coordinates(
    tmp_path,
) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="nihao",
        screenshot_path="",
    )
    resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "nihao"
    assert resolved["source"] == "recorded"
    assert resolved["meaningful"] is None
    assert resolved["vision"] is None


def test_event_with_resolved_text_replaces_text_field() -> None:
    event = RecordedEvent(index=1, timestamp_utc="t", kind="text_input", text="nihao")
    updated = event_with_resolved_text(event, {"text": "你好"})
    assert updated.text == "你好"
    assert event.text == "nihao"
