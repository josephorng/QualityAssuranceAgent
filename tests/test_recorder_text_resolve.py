from __future__ import annotations

from unittest.mock import AsyncMock, patch

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


@pytest.mark.asyncio
async def test_resolve_text_input_keeps_meaningful_text(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="chrome",
        screenshot_path="",
    )
    with patch(
        "src.recorder.text_resolve._check_text_meaningful",
        new=AsyncMock(return_value={"meaningful": True, "reason": "english word"}),
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "chrome"
    assert resolved["source"] == "recorded"
    assert resolved["meaningful"] is True


@pytest.mark.asyncio
async def test_resolve_text_input_ocr_fallback_when_not_meaningful_and_anchor_set(
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
        "src.recorder.text_resolve._check_text_meaningful",
        new=AsyncMock(return_value={"meaningful": False, "reason": "ime pinyin"}),
    ), patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="你好",
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "你好"
    assert resolved["source"] == "ocr"
    assert resolved["meaningful"] is False


@pytest.mark.asyncio
async def test_resolve_text_input_keeps_recorded_when_not_meaningful_without_anchor(
    tmp_path,
) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="nihao",
        screenshot_path="",
    )
    with patch(
        "src.recorder.text_resolve._check_text_meaningful",
        new=AsyncMock(return_value={"meaningful": False, "reason": "ime pinyin"}),
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "nihao"
    assert resolved["source"] == "recorded"
    assert resolved["vision"] is None


def test_event_with_resolved_text_replaces_text_field() -> None:
    event = RecordedEvent(index=1, timestamp_utc="t", kind="text_input", text="nihao")
    updated = event_with_resolved_text(event, {"text": "你好"})
    assert updated.text == "你好"
    assert event.text == "nihao"
