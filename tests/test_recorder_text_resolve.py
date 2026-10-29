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
    draw_focus_rect_bgr,
    event_with_resolved_text,
    focus_rect_debug_image_path,
    focused_uia_text_from_event,
    resolve_text_input_text,
)
from src.common.io_utils import imread_bgr
from src.recorder.vision_context import extract_nearest_text


def test_strip_ocr_caret_removes_trailing_pipe() -> None:
    assert _strip_ocr_caret("office|") == "office"
    assert _strip_ocr_caret("office |") == "office"
    assert _strip_ocr_caret("office") == "office"
    assert _strip_ocr_caret("|") == ""
    assert _strip_ocr_caret("a|b") == "a|b"


def test_focused_uia_text_from_event_reads_signals_after() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="offioffice",
        window_snapshot_debug={
            "signals_after": {"focused": {"name": "搜尋方塊", "value": "office"}},
        },
    )
    assert focused_uia_text_from_event(event) == "office"
    empty = RecordedEvent(
        index=2,
        timestamp_utc="t",
        kind="text_input",
        text="x",
        window_snapshot_debug={
            "signals_after": {"focused": {"name": "搜尋方塊", "value": ""}},
        },
    )
    assert focused_uia_text_from_event(empty) is None


@pytest.mark.asyncio
async def test_resolve_text_input_prefers_uia_focused_value(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="offioffice",
        cursor_xy=(10, 10),
        screenshot_path="",
        window_snapshot_debug={
            "signals_after": {"focused": {"name": "搜尋方塊", "value": "office"}},
        },
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
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "office"
    assert resolved["source"] == "uia"
    assert resolved["uia_text"] == "office"
    assert resolved["recorded_text"] == "offioffice"
    assert resolved["ocr_text"] == "office"


@pytest.mark.asyncio
async def test_resolve_text_input_skips_empty_uia_and_uses_english_recorded(
    tmp_path,
) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="offioffice",
        cursor_xy=(10, 10),
        screenshot_path="",
        window_snapshot_debug={
            "signals_after": {"focused": {"name": "搜尋方塊", "value": ""}},
        },
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
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "offioffice"
    assert resolved["source"] == "recorded"
    assert "English" in resolved["reason"]


@pytest.mark.asyncio
async def test_resolve_text_input_prefers_ocr_for_chinese(tmp_path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="nihao",
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
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="你好"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    assert resolved["text"] == "你好"
    assert resolved["source"] == "ocr"
    assert "Chinese" in resolved["reason"]


def test_draw_focus_rect_bgr_paints_magenta_box() -> None:
    bgr = np.zeros((40, 60, 3), dtype=np.uint8)
    out = draw_focus_rect_bgr(bgr, (10, 8, 40, 28), cursor_local=(25, 18))
    assert out is not bgr
    # Magenta BGR on the rectangle edge.
    assert tuple(int(v) for v in out[8, 10]) == (255, 0, 255)
    assert tuple(int(v) for v in out[18, 25]) == (255, 0, 255)


def test_draw_focus_rect_bgr_paints_cyan_foreground_window() -> None:
    bgr = np.zeros((40, 60, 3), dtype=np.uint8)
    out = draw_focus_rect_bgr(
        bgr,
        (12, 10, 30, 22),
        foreground_rect_local=(4, 4, 56, 36),
    )
    # Cyan BGR on the foreground window edge; magenta on the focus box.
    assert tuple(int(v) for v in out[4, 4]) == (0, 255, 255)
    assert tuple(int(v) for v in out[10, 12]) == (255, 0, 255)


def test_draw_focus_rect_bgr_paints_target_and_landmark_arrow() -> None:
    bgr = np.zeros((80, 120, 3), dtype=np.uint8)
    out = draw_focus_rect_bgr(
        bgr,
        None,
        target_rect_local=(70, 30, 100, 55),
        peer_rects_local=[(50, 10, 65, 25)],
        landmarks=[
            {
                "label": "「A1」文字",
                "side": "right",
                "rect": (10, 30, 30, 50),
            }
        ],
    )
    # Lime target corner, yellow peer corner, orange landmark corner.
    assert tuple(int(v) for v in out[30, 70]) == (0, 220, 0)
    assert tuple(int(v) for v in out[10, 50]) == (0, 215, 255)
    assert tuple(int(v) for v in out[30, 10]) == (0, 140, 255)
    # Small fixed right-pointing arrow just outside the landmark's right edge.
    arrow = out[40, 40]
    assert int(arrow[2]) >= 100  # orange has strong red channel in BGR
    # Must not draw a long line all the way toward the distant target.
    assert int(out[40, 60][2]) < 50


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
    assert "English" in resolved["reason"]
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

    assert resolved["text"] == "你好"
    assert resolved["source"] == "ocr"
    assert resolved["ocr_text"] == "你好"
    assert resolved["recorded_text"] == "nihao"
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

    # Chinese OCR hard rule picks CJK over IME Latin keystrokes.
    assert resolved["source"] == "ocr"
    assert resolved["text"] == "什麼是套利?"
    assert resolved["ocr_options"] == ["Google", "什麼是套利?", "搜尋"]
    # Nearest in-rect text to caret (960,540) is the omnibox query, not taskbar 搜尋.
    assert resolved["ocr_text"] == "什麼是套利?"
    debug_path = focus_rect_debug_image_path(tmp_path, event.index, suffix="")
    assert debug_path.is_file()
    annotated = imread_bgr(debug_path)
    assert annotated is not None
    assert tuple(int(v) for v in annotated[0, 0]) == (255, 0, 255)


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
    """Caret-sized focus expands only via a YOLO Input that contains the caret."""
    # Mirrors Save As filename: caret 1×20px inside the filename Input, typed
    # text left of caret, type dropdown slightly below, taskbar search elsewhere.
    event = RecordedEvent(
        index=15,
        timestamp_utc="t",
        kind="text_input",
        text="wl6m06au/6wu0 2k7wu0 fu4",
        cursor_xy=(321, 393),
        monitor_offset=(0, 0),
        focus_rect=(321, 383, 322, 403),
        screenshot_path="",
        ime={"chinese": True, "lang_id": 0x0804},
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (321, 393),
        "candidates": [],
        "detection_count": 5,
        "bgr": np.zeros((1080, 960, 3), dtype=np.uint8),
        "all_detections": [
            # Input must contain caret (321, 393); pad covers filename row.
            _detection_from_bbox((43, 380, 904, 40), YOLO_CLASS_INPUT),
            _detection_from_bbox((139, 386, 93, 15), YOLO_CLASS_TEXT, text="桃園明天的天"),
            _detection_from_bbox((38, 387, 89, 16), YOLO_CLASS_TEXT, text="檔案名稱(N):"),
            _detection_from_bbox((138, 430, 106, 18), YOLO_CLASS_TEXT, text="文字文件 (*.txt)"),
            _detection_from_bbox((490, 1049, 34, 16), YOLO_CLASS_TEXT, text="搜尋"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="搜尋",
    ) as nearest_mock, patch(
        "src.recorder.text_resolve._resolve_or_sample_ocr_clip_rect",
        return_value=None,
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_not_called()
    # Chinese IME + focus-local CJK OCR selects filename text over Latin keys.
    assert resolved["source"] == "ocr"
    assert resolved["text"] == "桃園明天的天"
    assert resolved["ocr_text"] == "桃園明天的天"
    assert "桃園明天的天" in resolved["ocr_options"]
    assert "搜尋" not in resolved["ocr_options"]
    assert "文字文件 (*.txt)" not in resolved["ocr_options"]


@pytest.mark.asyncio
async def test_resolve_text_input_caret_thin_ignores_distant_input(
    tmp_path,
) -> None:
    """Distant nearest Input must not AABB-expand caret into UI chrome OCR."""
    event = RecordedEvent(
        index=4,
        timestamp_utc="t",
        kind="text_input",
        text="123456",
        cursor_xy=(222, 236),
        monitor_offset=(0, 0),
        focus_rect=(222, 226, 223, 246),
        screenshot_path="",
        ime={"chinese": False, "lang_id": 0x0409},
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (222, 236),
        "candidates": [],
        "detection_count": 4,
        "bgr": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "all_detections": [
            # Excel search box — nearest but does not contain the formula-bar caret.
            _detection_from_bbox((224, 410, 582, 39), YOLO_CLASS_INPUT),
            _detection_from_bbox((505, 225, 32, 17), YOLO_CLASS_TEXT, text="導覽"),
            _detection_from_bbox((743, 224, 32, 16), YOLO_CLASS_TEXT, text="公式"),
            _detection_from_bbox((270, 231, 7, 10), YOLO_CLASS_TEXT, text="3"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="3",
    ) as nearest_mock, patch(
        "src.recorder.text_resolve._resolve_or_sample_ocr_clip_rect",
        return_value=None,
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_called_once()
    assert "公式" not in resolved["ocr_options"]
    assert "導覽" not in resolved["ocr_options"]
    assert resolved["text"] == "123456"
    assert resolved["source"] == "recorded"
    assert resolved["ime"] == {"chinese": False, "lang_id": 0x0409}


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
    ) as nearest_mock, patch(
        "src.recorder.text_resolve._resolve_or_sample_ocr_clip_rect",
        return_value=None,
    ):
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_called_once()
    assert resolved["ocr_text"] == "fallback"
    assert resolved["ocr_options"] == ["fallback"]


@pytest.mark.asyncio
async def test_resolve_text_input_caret_thin_uses_uia_ocr_clip_rect(
    tmp_path,
) -> None:
    """Recorded UIA field clip expands caret-thin focus before YOLO Input."""
    event = RecordedEvent(
        index=4,
        timestamp_utc="t",
        kind="text_input",
        text="123456",
        cursor_xy=(222, 236),
        monitor_offset=(0, 0),
        focus_rect=(222, 226, 223, 246),
        ocr_clip_rect=(200, 220, 360, 250),
        screenshot_path="",
        ime={"chinese": False, "lang_id": 0x0409},
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (222, 236),
        "candidates": [],
        "detection_count": 3,
        "bgr": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "all_detections": [
            # No YOLO Input containing caret — UIA clip alone must gate OCR.
            _detection_from_bbox((210, 225, 80, 18), YOLO_CLASS_TEXT, text="123456"),
            _detection_from_bbox((220, 314, 76, 20), YOLO_CLASS_TEXT, text="a日腦n"),
            _detection_from_bbox((490, 1049, 34, 16), YOLO_CLASS_TEXT, text="搜尋"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="a日腦n",
    ) as nearest_mock:
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_not_called()
    assert resolved["ocr_text"] == "123456"
    assert resolved["ocr_options"] == ["123456"]
    assert "a日腦n" not in resolved["ocr_options"]
    assert resolved["text"] == "123456"
    assert resolved["source"] == "recorded"


@pytest.mark.asyncio
async def test_resolve_text_input_uia_clip_preferred_over_yolo_input(
    tmp_path,
) -> None:
    """When both UIA clip and YOLO Input exist, UIA field clip wins."""
    event = RecordedEvent(
        index=15,
        timestamp_utc="t",
        kind="text_input",
        text="hello",
        cursor_xy=(100, 100),
        monitor_offset=(0, 0),
        focus_rect=(100, 90, 101, 110),
        ocr_clip_rect=(40, 80, 200, 120),
        screenshot_path="",
        ime={"chinese": False, "lang_id": 0x0409},
    )
    fake_vision = {
        "used_vision": True,
        "candidate_text": "candidate",
        "local_cursor": (100, 100),
        "candidates": [],
        "detection_count": 3,
        "bgr": np.zeros((400, 400, 3), dtype=np.uint8),
        "all_detections": [
            # YOLO Input is wider and would include chrome text below.
            _detection_from_bbox((20, 70, 300, 80), YOLO_CLASS_INPUT),
            _detection_from_bbox((50, 90, 60, 16), YOLO_CLASS_TEXT, text="hello"),
            _detection_from_bbox((50, 130, 60, 16), YOLO_CLASS_TEXT, text="chrome"),
        ],
    }
    with patch(
        "src.recorder.text_resolve.build_vision_context_at_point",
        return_value=fake_vision,
    ), patch(
        "src.recorder.text_resolve.extract_nearest_text",
        return_value="chrome",
    ) as nearest_mock:
        resolved = await resolve_text_input_text(event, run_dir=tmp_path)

    nearest_mock.assert_not_called()
    assert resolved["ocr_text"] == "hello"
    assert resolved["ocr_options"] == ["hello"]
    assert "chrome" not in resolved["ocr_options"]


def test_event_with_resolved_text_replaces_text_field() -> None:
    event = RecordedEvent(index=1, timestamp_utc="t", kind="text_input", text="nihao")
    updated = event_with_resolved_text(event, {"text": "你好"})
    assert updated.text == "你好"
    assert event.text == "nihao"
