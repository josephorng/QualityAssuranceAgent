from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT, YOLO_CLASS_TEXT
from src.recorder.models import RecordedEvent
from src.recorder.text_resolve import (
    _is_caret_thin_rect,
    _strip_ocr_caret,
    event_with_resolved_text,
    resolve_text_input_text,
)
from src.recorder.vision_context import extract_nearest_text


def test_strip_ocr_caret_removes_trailing_pipe() -> None:
    assert _strip_ocr_caret("office|") == "office"
    assert _strip_ocr_caret("office |") == "office"
    assert _strip_ocr_caret("office") == "office"
    assert _strip_ocr_caret("|") == ""
    assert _strip_ocr_caret("a|b") == "a|b"


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
async def test_resolve_text_input_prefers_recorded_when_ocr_finds_text(tmp_path) -> None:
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

    assert resolved["text"] == "chrome"
    assert resolved["source"] == "recorded"
    assert resolved["ocr_text"] == "Chrome"
    assert resolved["recorded_text"] == "chrome"
    assert resolved["meaningful"] is None


@pytest.mark.asyncio
async def test_resolve_text_input_strips_trailing_caret_from_ocr(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="office",
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
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="office|"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "office"
    assert resolved["ocr_text"] == "office"
    assert resolved["source"] == "recorded"


@pytest.mark.asyncio
async def test_resolve_text_input_uses_after_screenshot_for_ocr(tmp_path) -> None:
    before = tmp_path / "screenshots" / "event_001.jpeg"
    after = tmp_path / "screenshots" / "event_001_end.jpeg"
    before.parent.mkdir(parents=True)
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="ooffice",
        anchor_click_xy=(110, 210),
        monitor_offset=(100, 200),
        end_monitor_offset=(100, 200),
        screenshot_path=str(before),
        end_screenshot_path=str(after),
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (10, 10),
        "candidates": [],
        "detection_count": 1,
        "bgr": np.zeros((100, 100, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="office"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ) as build_mock, patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="office",
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "ooffice"
    assert resolved["source"] == "recorded"
    assert resolved["ocr_text"] == "office"
    assert "after-screenshot OCR" in resolved["reason"]
    build_mock.assert_called_once()
    assert build_mock.call_args.kwargs["debug_name"] == "_end"


@pytest.mark.asyncio
async def test_resolve_text_input_prefers_cursor_xy_over_anchor(
    tmp_path,
) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="typed",
        cursor_xy=(50, 60),
        anchor_click_xy=(110, 210),
        monitor_offset=(0, 0),
        screenshot_path="",
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (50, 60),
        "candidates": [],
        "detection_count": 1,
        "bgr": np.zeros((100, 100, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="typed"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ) as build_mock, patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="typed",
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "typed"
    assert build_mock.call_args.kwargs["reference_xy"] == (50, 60)
    assert build_mock.call_args.kwargs["local_x"] == 50
    assert build_mock.call_args.kwargs["local_y"] == 60


@pytest.mark.asyncio
async def test_resolve_text_input_keeps_ocr_alternate_with_anchor(
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

    assert resolved["text"] == "nihao"
    assert resolved["source"] == "recorded"
    assert resolved["ocr_text"] == "你好"
    assert resolved["meaningful"] is None


@pytest.mark.asyncio
async def test_resolve_text_input_falls_back_to_ocr_when_recorded_empty(
    tmp_path,
) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="",
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
    assert resolved["ocr_text"] == "Chrome"


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
    assert resolved["ocr_text"] is None
    assert resolved["meaningful"] is None
    assert resolved["vision"] is None


@pytest.mark.asyncio
async def test_resolve_text_input_lists_ocr_options_inside_focus_rect(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="gk6ak7g4wl4xu4?",
        cursor_xy=(960, 540),
        monitor_offset=(0, 0),
        focus_rect=(0, 0, 1920, 1080),
        screenshot_path="",
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (960, 540),
        "candidates": [],
        "detection_count": 3,
        "bgr": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((900, 80, 200, 24), YOLO_CLASS_TEXT, text="什麼是套利?"),
            _detection_from_bbox((500, 1050, 40, 16), YOLO_CLASS_TEXT, text="搜尋"),
            _detection_from_bbox((40, 40, 80, 20), YOLO_CLASS_TEXT, text="Google"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["source"] == "recorded"
    assert resolved["ocr_options"] == ["Google", "什麼是套利?", "搜尋"]
    # Nearest in-rect text to caret (960,540) is the omnibox query, not taskbar 搜尋.
    assert resolved["ocr_text"] == "什麼是套利?"


@pytest.mark.asyncio
async def test_resolve_text_input_ignores_ocr_outside_focus_rect(tmp_path) -> None:
    """Omnibox strip focus must not import taskbar / nearest-screen OCR."""
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="gk6ka7g4wl4xu4",
        cursor_xy=(1497, 84),
        monitor_offset=(0, 0),
        focus_rect=(226, 67, 1561, 100),
        screenshot_path="",
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (1497, 84),
        "candidates": [],
        "detection_count": 3,
        "bgr": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((228, 74, 91, 19), YOLO_CLASS_TEXT, text="什麼是套利"),
            _detection_from_bbox((490, 1049, 34, 16), YOLO_CLASS_TEXT, text="搜尋"),
            _detection_from_bbox((1601, 77, 58, 15), YOLO_CLASS_TEXT, text="AI 模式"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="搜尋",
    ) as nearest_mock:
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_not_called()
    assert resolved["ocr_options"] == ["什麼是套利"]
    assert resolved["ocr_text"] == "什麼是套利"


def test_is_caret_thin_rect_detects_caret_strip() -> None:
    assert _is_caret_thin_rect((321, 383, 322, 403))
    assert not _is_caret_thin_rect((226, 67, 1561, 100))


@pytest.mark.asyncio
async def test_resolve_text_input_caret_thin_uses_input_bbox_clip(tmp_path) -> None:
    """Caret-sized focus must expand via YOLO Input, not stay empty."""
    # Mirrors Save As filename: caret 1×20px, typed text left of caret, type
    # dropdown slightly below, taskbar search elsewhere.
    event = RecordedEvent(
        index=15,
        timestamp_utc="t",
        kind="text_input",
        text="wl6m06au/6wu0 2k7wu0 fu4",
        cursor_xy=(321, 393),
        monitor_offset=(0, 0),
        focus_rect=(321, 383, 322, 403),
        screenshot_path="",
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (321, 393),
        "candidates": [],
        "detection_count": 5,
        "bgr": np.zeros((1080, 960, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((43, 409, 904, 26), YOLO_CLASS_INPUT),
            _detection_from_bbox((139, 386, 93, 15), YOLO_CLASS_TEXT, text="桃園明天的天"),
            _detection_from_bbox((38, 387, 89, 16), YOLO_CLASS_TEXT, text="檔案名稱(N):"),
            _detection_from_bbox((138, 416, 106, 18), YOLO_CLASS_TEXT, text="文字文件 (*.txt)"),
            _detection_from_bbox((490, 1049, 34, 16), YOLO_CLASS_TEXT, text="搜尋"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="搜尋",
    ) as nearest_mock:
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_not_called()
    assert resolved["source"] == "recorded"
    assert resolved["ocr_text"] == "桃園明天的天"
    assert "桃園明天的天" in resolved["ocr_options"]
    assert "搜尋" not in resolved["ocr_options"]


@pytest.mark.asyncio
async def test_resolve_text_input_caret_thin_without_input_falls_back(
    tmp_path,
) -> None:
    """Caret-thin with no Input detection falls back to nearest-text helper."""
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="abc",
        cursor_xy=(100, 100),
        monitor_offset=(0, 0),
        focus_rect=(100, 90, 101, 110),
        screenshot_path="",
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (100, 100),
        "candidates": [],
        "detection_count": 1,
        "bgr": np.zeros((200, 200, 3), dtype=np.uint8),
        "all_detections": [
            _detection_from_bbox((80, 90, 40, 16), YOLO_CLASS_TEXT, text="fallback"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="fallback",
    ) as nearest_mock:
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_called_once()
    assert resolved["ocr_text"] == "fallback"
    assert resolved["ocr_options"] == ["fallback"]


def test_event_with_resolved_text_replaces_text_field() -> None:
    event = RecordedEvent(index=1, timestamp_utc="t", kind="text_input", text="nihao")
    updated = event_with_resolved_text(event, {"text": "你好"})
    assert updated.text == "你好"
    assert event.text == "nihao"
