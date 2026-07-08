from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT
from src.recorder.models import RecordedEvent
from src.recorder.vision_context import (
    _collect_drag_source_identities,
    _detection_identities,
    _local_cursor,
    _nearest_candidates,
    build_vision_context,
    build_vision_context_at_point,
    candidate_offset_for_instruction,
    format_drag_candidate_anchor,
    format_drag_destination_offset_hints,
    format_drag_relative_offset_phrase,
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


@pytest.mark.asyncio
async def test_build_vision_context_formats_yolo_candidates(tmp_path) -> None:
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
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

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


@pytest.mark.asyncio
async def test_build_vision_context_drag_includes_start_and_destination(tmp_path) -> None:
    start_shot = tmp_path / "event_start.jpeg"
    end_shot = tmp_path / "event_end.jpeg"
    start_shot.write_bytes(b"not-a-real-jpeg")
    end_shot.write_bytes(b"not-a-real-jpeg")
    event = RecordedEvent(
        index=3,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(110, 210),
        end_xy=(310, 210),
        monitor_offset=(100, 200),
        end_monitor_offset=(100, 200),
        screenshot_path=str(start_shot),
        end_screenshot_path=str(end_shot),
    )
    start_detections = [
        _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="音量"),
    ]
    end_detections = [
        _detection_from_bbox((200, 8, 40, 10), YOLO_CLASS_TEXT, text="100%"),
    ]

    call_count = {"n": 0}

    def _fake_build(_bgr, **_kwargs):
        call_count["n"] += 1
        return start_detections if call_count["n"] == 1 else end_detections

    with patch(
        "src.recorder.vision_context.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        side_effect=_fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    assert vision["used_vision"] is True
    assert vision["local_cursor"] == (10, 10)
    assert "text='音量'" in vision["candidate_text"]
    destination = vision["destination"]
    assert destination["local_cursor"] == (210, 10)
    assert "text='100%'" in destination["candidate_text"]
    assert (tmp_path / "yolo_ocr" / "event_003.json").is_file()
    assert (tmp_path / "yolo_ocr" / "event_003_end.json").is_file()
    assert (tmp_path / "yolo_ocr" / "event_003_end_filtered.json").is_file()


@pytest.mark.asyncio
async def test_drag_destination_keeps_nearest_candidates_without_exclusion(tmp_path) -> None:
    start_shot = tmp_path / "event_start.jpeg"
    end_shot = tmp_path / "event_end.jpeg"
    start_shot.write_bytes(b"not-a-real-jpeg")
    end_shot.write_bytes(b"not-a-real-jpeg")
    event = RecordedEvent(
        index=4,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(110, 210),
        end_xy=(310, 210),
        monitor_offset=(100, 200),
        end_monitor_offset=(100, 200),
        screenshot_path=str(start_shot),
        end_screenshot_path=str(end_shot),
    )
    start_detections = [
        _detection_from_bbox((8, 8, 60, 20), YOLO_CLASS_TEXT, text="報告.pdf"),
        _detection_from_bbox((8, 30, 60, 20), YOLO_CLASS_TEXT, text="內嵌標籤"),
    ]
    end_detections = [
        _detection_from_bbox((8, 8, 60, 20), YOLO_CLASS_TEXT, text="報告.pdf"),
        _detection_from_bbox((200, 8, 80, 20), YOLO_CLASS_TEXT, text="文件"),
    ]

    call_count = {"n": 0}

    def _fake_build(_bgr, **_kwargs):
        call_count["n"] += 1
        return start_detections if call_count["n"] == 1 else end_detections

    with patch(
        "src.recorder.vision_context.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        side_effect=_fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    destination = vision["destination"]
    assert "text='文件'" in destination["candidate_text"]


def test_collect_drag_source_identities_includes_cluster_text() -> None:
    detections = [
        _detection_from_bbox((0, 0, 100, 80), YOLO_CLASS_ELEMENT),
        _detection_from_bbox((10, 10, 60, 20), YOLO_CLASS_TEXT, text="報告.pdf"),
        _detection_from_bbox((10, 40, 60, 20), YOLO_CLASS_TEXT, text="內嵌標籤"),
        _detection_from_bbox((300, 300, 40, 20), YOLO_CLASS_TEXT, text="遠處"),
    ]
    identities = _collect_drag_source_identities(detections, (20, 20))
    assert "text:報告.pdf" in identities
    assert "text:內嵌標籤" in identities
    assert "text:遠處" not in identities


def test_detection_identities_links_icon_label_and_visible_text() -> None:
    chrome_icon = _detection_from_bbox(
        (13, 607, 52, 50),
        YOLO_CLASS_ELEMENT,
        text="\ue007",
        icons=[{"chinese_id": "Chrome"}],
    )
    chrome_label = _detection_from_bbox((10, 680, 58, 14), YOLO_CLASS_TEXT, text="Chrome")

    icon_keys = _detection_identities(chrome_icon)
    label_keys = _detection_identities(chrome_label)

    assert "icon:Chrome" in icon_keys
    assert "text:Chrome" in icon_keys
    assert "text:Chrome" in label_keys
    assert icon_keys & label_keys


@pytest.mark.asyncio
async def test_drag_destination_lists_nearest_candidates_at_drop(tmp_path) -> None:
    start_shot = tmp_path / "event_start.jpeg"
    end_shot = tmp_path / "event_end.jpeg"
    start_shot.write_bytes(b"x")
    end_shot.write_bytes(b"x")
    event = RecordedEvent(
        index=5,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(33, 637),
        end_xy=(2100, 638),
        monitor_offset=(0, 0),
        end_monitor_offset=(1920, 0),
        screenshot_path=str(start_shot),
        end_screenshot_path=str(end_shot),
    )
    start_detections = [
        _detection_from_bbox(
            (13, 607, 52, 50),
            YOLO_CLASS_ELEMENT,
            text="\ue007",
            icons=[{"chinese_id": "Chrome"}],
        ),
        _detection_from_bbox((10, 680, 58, 14), YOLO_CLASS_TEXT, text="Chrome"),
        _detection_from_bbox((8, 560, 61, 18), YOLO_CLASS_TEXT, text="Inkscape"),
    ]
    end_detections = [
        _detection_from_bbox(
            (159, 609, 51, 49),
            YOLO_CLASS_ELEMENT,
            text="\ue007",
            icons=[{"chinese_id": "Chrome"}],
        ),
        _detection_from_bbox((157, 681, 55, 13), YOLO_CLASS_TEXT, text="Chrome"),
        _detection_from_bbox((162, 580, 56, 15), YOLO_CLASS_TEXT, text="Desktop"),
    ]

    call_count = {"n": 0}

    def _fake_build(_bgr, **_kwargs):
        call_count["n"] += 1
        return start_detections if call_count["n"] == 1 else end_detections

    with patch(
        "src.recorder.vision_context.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        side_effect=_fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=False)

    destination = vision["destination"]
    assert "Chrome" in destination["candidate_text"]


@pytest.mark.asyncio
async def test_drag_destination_uses_nearest_when_drop_not_inside_text(tmp_path) -> None:
    start_shot = tmp_path / "event_start.jpeg"
    end_shot = tmp_path / "event_end.jpeg"
    start_shot.write_bytes(b"x")
    end_shot.write_bytes(b"x")
    event = RecordedEvent(
        index=6,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(43, 638),
        end_xy=(2100, 638),
        monitor_offset=(0, 0),
        end_monitor_offset=(1920, 0),
        screenshot_path=str(start_shot),
        end_screenshot_path=str(end_shot),
    )
    start_detections = [
        _detection_from_bbox((12, 607, 51, 50), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Chrome"}]),
        _detection_from_bbox((16, 659, 33, 16), YOLO_CLASS_TEXT, text="帳多"),
        _detection_from_bbox((10, 680, 58, 14), YOLO_CLASS_TEXT, text="Cbhrome"),
    ]
    end_detections = [
        _detection_from_bbox((171, 665, 35, 15), YOLO_CLASS_TEXT, text="振銓"),
        _detection_from_bbox((162, 580, 56, 15), YOLO_CLASS_TEXT, text="Desktop"),
    ]

    call_count = {"n": 0}

    def _fake_build(_bgr, **_kwargs):
        call_count["n"] += 1
        return start_detections if call_count["n"] == 1 else end_detections

    with patch(
        "src.recorder.vision_context.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        side_effect=_fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=False)

    destination = vision["destination"]
    assert "振銓" in destination["candidate_text"]
    assert "text='Desktop'" in destination["candidate_text"]


@pytest.mark.asyncio
async def test_build_vision_context_skips_vision_for_key_events(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="enter",
        screenshot_path="",
    )
    vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=False)
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


def test_format_drag_relative_offset_phrase_below() -> None:
    assert format_drag_relative_offset_phrase(-2, 49) == "下方49個像素"


def test_format_drag_relative_offset_phrase_right_and_below() -> None:
    assert format_drag_relative_offset_phrase(12, 49) == "右方12個像素、下方49個像素"


def test_format_drag_relative_offset_phrase_negligible() -> None:
    assert format_drag_relative_offset_phrase(2, -3) is None


def test_format_drag_candidate_anchor_text() -> None:
    assert format_drag_candidate_anchor({"class_name": "text", "text": "Desktop"}) == "「Desktop」文字"


def test_format_drag_candidate_anchor_icon() -> None:
    candidate = {
        "class_name": "element",
        "text": "pua",
        "icons": [{"chinese_id": "Chrome"}],
    }
    assert format_drag_candidate_anchor(candidate) == "「Chrome」圖示"


def test_format_drag_candidate_anchor_element_text() -> None:
    assert format_drag_candidate_anchor({"class_name": "element", "text": "a"}) == "「a」元素"


def test_format_drag_destination_offset_hints_desktop_like() -> None:
    destination = {
        "local_cursor": (189, 638),
        "candidates": [
            {
                "bbox": [166, 661, 37, 16],
                "center": [184, 669],
                "class_name": "text",
                "text": "拖銓",
            },
            {
                "bbox": [163, 581, 56, 16],
                "center": [191, 589],
                "class_name": "text",
                "text": "Desktop",
            },
        ],
    }
    hints = format_drag_destination_offset_hints(destination)
    assert "[index 1] 「Desktop」: 下方49個像素" in hints


def test_candidate_offset_for_instruction_matches_desktop() -> None:
    destination = {
        "local_cursor": (189, 638),
        "candidates": [
            {
                "center": [191, 589],
                "class_name": "text",
                "text": "Desktop",
            },
        ],
    }
    assert candidate_offset_for_instruction(destination, "Desktop") == "下方49個像素"


def test_format_drag_destination_offset_hints_when_drop_inside_anchor() -> None:
    destination = {
        "local_cursor": (188, 672),
        "candidates": [
            {
                "bbox": [171, 665, 35, 15],
                "center": [188, 672],
                "class_name": "text",
                "text": "振銓",
            },
        ],
    }
    hints = format_drag_destination_offset_hints(destination)
    assert "[index 0] 「振銓」: (on anchor, offset negligible)" in hints


def test_candidate_offset_for_instruction_when_drop_inside_anchor() -> None:
    destination = {
        "local_cursor": (188, 672),
        "candidates": [
            {
                "bbox": [171, 665, 35, 15],
                "center": [188, 672],
                "class_name": "text",
                "text": "振銓",
            },
        ],
    }
    assert candidate_offset_for_instruction(destination, "振銓") is None


@pytest.mark.asyncio
async def test_drag_destination_uses_hit_target_when_drop_inside_text(tmp_path) -> None:
    start_shot = tmp_path / "event_start.jpeg"
    end_shot = tmp_path / "event_end.jpeg"
    start_shot.write_bytes(b"x")
    end_shot.write_bytes(b"x")
    event = RecordedEvent(
        index=7,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(43, 638),
        end_xy=(2108, 672),
        monitor_offset=(0, 0),
        end_monitor_offset=(1920, 0),
        screenshot_path=str(start_shot),
        end_screenshot_path=str(end_shot),
    )
    start_detections = [
        _detection_from_bbox((12, 607, 51, 50), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Chrome"}]),
        _detection_from_bbox((10, 680, 58, 14), YOLO_CLASS_TEXT, text="Cbhrome"),
    ]
    end_detections = [
        _detection_from_bbox((171, 665, 35, 15), YOLO_CLASS_TEXT, text="振銓"),
        _detection_from_bbox((162, 580, 56, 15), YOLO_CLASS_TEXT, text="Desktop"),
    ]

    call_count = {"n": 0}

    def _fake_build(_bgr, **_kwargs):
        call_count["n"] += 1
        return start_detections if call_count["n"] == 1 else end_detections

    with patch(
        "src.recorder.vision_context.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        side_effect=_fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=False)

    destination = vision["destination"]
    assert "text='振銓'" in destination["candidate_text"]
    assert "text='Desktop'" not in destination["candidate_text"]


@pytest.mark.asyncio
async def test_build_vision_context_drag_includes_destination_offset_hints(tmp_path) -> None:
    start_shot = tmp_path / "event_start.jpeg"
    end_shot = tmp_path / "event_end.jpeg"
    start_shot.write_bytes(b"x")
    end_shot.write_bytes(b"x")
    event = RecordedEvent(
        index=5,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(33, 637),
        end_xy=(2100, 638),
        monitor_offset=(0, 0),
        end_monitor_offset=(1920, 0),
        screenshot_path=str(start_shot),
        end_screenshot_path=str(end_shot),
    )
    start_detections = [
        _detection_from_bbox(
            (13, 607, 52, 50),
            YOLO_CLASS_ELEMENT,
            text="\ue007",
            icons=[{"chinese_id": "Chrome"}],
        ),
        _detection_from_bbox((10, 680, 58, 14), YOLO_CLASS_TEXT, text="Chrome"),
    ]
    end_detections = [
        _detection_from_bbox((162, 580, 56, 15), YOLO_CLASS_TEXT, text="Desktop"),
    ]

    call_count = {"n": 0}

    def _fake_build(_bgr, **_kwargs):
        call_count["n"] += 1
        return start_detections if call_count["n"] == 1 else end_detections

    with patch(
        "src.recorder.vision_context.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._build_candidates_from_bgr",
        side_effect=_fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=False)

    destination = vision["destination"]
    assert "destination_offset_hints" in destination
    assert "「Desktop」" in destination["destination_offset_hints"]
