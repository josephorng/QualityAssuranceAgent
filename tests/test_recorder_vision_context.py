from __future__ import annotations

from unittest.mock import patch

import numpy as np

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT
from src.recorder.models import RecordedEvent
from src.recorder.vision_context import (
    _local_cursor,
    _nearest_candidates,
    build_vision_context,
    build_vision_context_at_point,
    format_field_context_hint,
)


def test_local_cursor_uses_monitor_offset() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(150, 250),
        monitor_offset=(100, 200),
        screenshot_path="",
    )
    assert _local_cursor(event) == (50, 50)


def test_nearest_candidates_sorted_by_distance() -> None:
    detections = [
        _detection_from_bbox((40, 40, 20, 20), YOLO_CLASS_TEXT, text="Far"),
        _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="Near"),
    ]
    nearest = _nearest_candidates(detections, 10, 10, limit=2)
    assert nearest[0].text == "Near"
    assert nearest[1].text == "Far"


def test_nearest_candidates_uses_bbox_distance_not_center() -> None:
    """A tall control clicked on its edge ranks above text with a nearer center."""
    detections = [
        _detection_from_bbox((30, 45, 40, 10), YOLO_CLASS_TEXT, text="Side label"),
        _detection_from_bbox((10, 0, 12, 200), YOLO_CLASS_ELEMENT, text="Scrollbar"),
    ]
    nearest = _nearest_candidates(detections, 15, 50, limit=2)
    assert nearest[0].text == "Scrollbar"
    assert nearest[1].text == "Side label"


def test_nearest_candidates_respects_limit() -> None:
    detections = [
        _detection_from_bbox((i * 30, 0, 10, 10), YOLO_CLASS_ELEMENT)
        for i in range(12)
    ]
    nearest = _nearest_candidates(detections, 0, 0, limit=8)
    assert len(nearest) == 8


def test_build_vision_context_formats_yolo_candidates(tmp_path) -> None:
    shot = tmp_path / "event.jpeg"
    shot.write_bytes(b"not-a-real-jpeg")
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(110, 210),
        monitor_offset=(100, 200),
        screenshot_path=str(shot),
    )
    pua = "\ue002"
    fake_detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_ELEMENT, text=pua),
        _detection_from_bbox((30, 0, 80, 20), YOLO_CLASS_TEXT, text="AWS"),
    ]

    with patch("src.recorder.vision_context.cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        return_value=fake_detections,
    ):
        vision = build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    assert vision["used_vision"] is True
    assert vision["local_cursor"] == (10, 10)
    assert vision["field_context"] == "(none)"
    text = vision["candidate_text"]
    assert "[index 0]" in text
    assert "class=元素(Element)" in text
    assert "icons=向下V箭頭" in text
    assert "text='AWS'" in text
    assert len(vision["candidates"]) == 2
    assert (tmp_path / "yolo_ocr" / "event_001.json").is_file()


def test_build_vision_context_skips_vision_for_key_events(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="enter",
        screenshot_path="",
    )
    vision = build_vision_context(event, run_dir=tmp_path, persist_debug=False)
    assert vision["used_vision"] is False
    assert vision["candidate_text"] == ""


def test_build_vision_context_at_point_uses_explicit_coords(tmp_path) -> None:
    shot = tmp_path / "event.jpeg"
    shot.write_bytes(b"not-a-real-jpeg")
    event = RecordedEvent(
        index=2,
        timestamp_utc="t",
        kind="text_input",
        text="nihao",
        screenshot_path=str(shot),
    )
    fake_detections = [
        _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="你好"),
    ]

    with patch("src.recorder.vision_context.cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        return_value=fake_detections,
    ):
        vision = build_vision_context_at_point(
            event,
            local_x=10,
            local_y=10,
            run_dir=tmp_path,
            reference_xy=(110, 210),
        )

    assert vision["used_vision"] is True
    assert vision["local_cursor"] == (10, 10)
    assert "text='你好'" in vision["candidate_text"]
    assert "bgr" in vision
    assert "all_detections" in vision


def test_format_field_context_hint_combines_input_and_inner_text() -> None:
    vision = {
        "local_cursor": (888, 921),
        "candidates": [
            {
                "bbox": [714, 874, 652, 96],
                "class_name": "input",
                "text": None,
            },
            {
                "bbox": [788, 912, 119, 19],
                "class_name": "text",
                "text": "間間Gemini",
            },
        ],
    }
    hint = format_field_context_hint(vision)
    assert hint == "輸入欄內可見文字: 「間間Gemini」"


def test_format_field_context_hint_prefers_typed_text_for_text_input() -> None:
    vision = {
        "local_cursor": (10, 10),
        "candidates": [
            {"bbox": [0, 0, 100, 30], "class_name": "input", "text": None},
            {"bbox": [5, 5, 40, 20], "class_name": "text", "text": "間間Gemini"},
        ],
    }
    hint = format_field_context_hint(vision, typed_text="hello")
    assert hint == "輸入欄內可見文字: 「hello」"
