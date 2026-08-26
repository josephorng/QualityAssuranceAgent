from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_SCROLLBAR, YOLO_CLASS_TEXT
from src.recorder.models import RecordedEvent
from src.recorder.vision_context import (
    _BBOX_HIT_TOLERANCE_PX,
    _drop_point_inside_candidate,
    _local_cursor,
    _nearest_candidates,
    build_vision_context,
    build_vision_context_at_point,
    candidate_offset_for_instruction,
    format_drag_candidate_anchor,
    format_drag_destination_offset_hints,
    format_drag_relative_offset_phrase,
    format_field_context_hint,
    format_input_context_hint,
    format_scrollbar_context_hint,
    primary_candidate_offset,
    resolve_event_screenshot_path,
)


def _parallel_safe_drag_fakes(start_detections, end_detections):
    """Distinguish start vs end frames without depending on call order (parallel drag)."""
    start_img = np.zeros((100, 100, 3), dtype=np.uint8)
    end_img = np.full((100, 100, 3), 1, dtype=np.uint8)

    def fake_imread(path):
        name = Path(path).name.lower()
        return end_img if "end" in name else start_img

    def fake_build(bgr, **_kwargs):
        return end_detections if int(bgr[0, 0, 0]) == 1 else start_detections

    return fake_imread, fake_build


def test_resolve_event_screenshot_path_falls_back_to_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "選擇文字測試"
    (run_dir / "screenshots").mkdir(parents=True)
    shot = run_dir / "screenshots" / "event_001.jpeg"
    shot.write_bytes(b"jpeg")
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(10, 20),
        screenshot_path=str(tmp_path / "recording_old" / "screenshots" / "event_001.jpeg"),
    )
    assert resolve_event_screenshot_path(event, run_dir) == shot


def test_build_vision_context_at_point_uses_run_dir_screenshot_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "選擇文字測試"
    (run_dir / "screenshots").mkdir(parents=True)
    (run_dir / "screenshots" / "event_001.jpeg").write_bytes(b"x")
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(10, 20),
        screenshot_path=str(tmp_path / "recording_old" / "screenshots" / "event_001.jpeg"),
    )
    fake_detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="搜尋"),
    ]
    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        return_value=fake_detections,
    ):
        vision = build_vision_context_at_point(
            event,
            local_x=10,
            local_y=10,
            run_dir=run_dir,
        )
    assert vision["used_vision"] is True
    assert vision["candidates"]
    assert vision["candidates"][0]["text"] == "搜尋"


def test_build_vision_context_at_point_records_missing_screenshot(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(10, 20),
        screenshot_path=str(tmp_path / "missing.jpeg"),
    )
    vision = build_vision_context_at_point(
        event,
        local_x=10,
        local_y=10,
        run_dir=tmp_path,
    )
    assert vision["used_vision"] is False
    assert vision["yolo_error"] == "找不到截圖檔"


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
    nearest = _nearest_candidates(detections, 10, 10)
    assert nearest[0].text == "Near"
    assert nearest[1].text == "Far"


def test_nearest_candidates_uses_bbox_distance_not_center() -> None:
    """A tall control clicked on its edge ranks above text with a nearer center."""
    detections = [
        _detection_from_bbox((30, 45, 40, 10), YOLO_CLASS_TEXT, text="Side label"),
        _detection_from_bbox((10, 0, 12, 200), YOLO_CLASS_ELEMENT, text="Scrollbar"),
    ]
    nearest = _nearest_candidates(
        detections, 15, 50, min_multi_char_text_neighbors=None, limit=2
    )
    assert nearest[0].text == "Scrollbar"
    assert nearest[1].text == "Side label"


def test_nearest_candidates_collects_all_when_no_multi_char_text() -> None:
    detections = [
        _detection_from_bbox((i * 30, 0, 10, 10), YOLO_CLASS_ELEMENT)
        for i in range(12)
    ]
    nearest = _nearest_candidates(detections, 0, 0)
    assert len(nearest) == 12


def test_nearest_candidates_grows_until_eight_text_and_five_icon() -> None:
    """Keep collecting until both text and icon quotas are filled."""
    primary = _detection_from_bbox(
        (0, 0, 10, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "Chrome"}],
    )
    icons = [
        _detection_from_bbox(
            (20 + i * 20, 0, 10, 10),
            YOLO_CLASS_ELEMENT,
            icons=[{"chinese_id": f"Icon{i}"}],
        )
        for i in range(6)
    ]
    texts = [
        _detection_from_bbox(
            (200 + i * 40, 0, 40, 10),
            YOLO_CLASS_TEXT,
            text=f"Text{i}",
        )
        for i in range(8)
    ]
    detections = [primary, *icons, *texts]
    nearest = _nearest_candidates(detections, 5, 5)
    assert nearest[0] is primary
    text_count = sum(
        1 for d in nearest if d.class_id == YOLO_CLASS_TEXT and len(d.text or "") > 1
    )
    icon_count = sum(1 for d in nearest if d.icons)
    assert text_count >= 8
    assert icon_count >= 5
    # All same-row (right-band) icons/texts are kept for HTML side choices.
    assert all(det in nearest for det in icons)
    assert all(det in nearest for det in texts)


def test_nearest_candidates_stops_after_two_multi_char_texts_when_configured() -> None:
    """Legacy text-only quota still works when icon min is zero."""
    primary = _detection_from_bbox(
        (0, 0, 10, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "Chrome"}],
    )
    close_icons = [
        _detection_from_bbox(
            (20 + i * 20, 0, 10, 10),
            YOLO_CLASS_ELEMENT,
            icons=[{"chinese_id": f"Icon{i}"}],
        )
        for i in range(3)
    ]
    single = _detection_from_bbox((200, 0, 10, 10), YOLO_CLASS_TEXT, text="中")
    text_a = _detection_from_bbox((300, 0, 40, 10), YOLO_CLASS_TEXT, text="OneNote")
    text_b = _detection_from_bbox((400, 0, 40, 10), YOLO_CLASS_TEXT, text="Slack")
    detections = [primary, *close_icons, single, text_a, text_b]
    nearest = _nearest_candidates(
        detections,
        5,
        5,
        min_multi_char_text_neighbors=2,
        min_icon_neighbors=0,
    )
    assert nearest[0] is primary
    assert single in nearest
    assert [d.text for d in nearest if d.class_id == YOLO_CLASS_TEXT] == [
        "中",
        "OneNote",
        "Slack",
    ]
    # Cardinal right-band icons beyond the quota are still retained.
    assert all(det in nearest for det in close_icons)


def test_nearest_candidates_keeps_all_directional_side_neighbors() -> None:
    """All eight directed sides of the primary stay available after quota fill."""
    primary = _detection_from_bbox(
        (100, 100, 20, 20),
        YOLO_CLASS_TEXT,
        text="Target",
    )
    left = _detection_from_bbox((40, 105, 30, 10), YOLO_CLASS_TEXT, text="LeftLabel")
    right = _detection_from_bbox((160, 105, 30, 10), YOLO_CLASS_TEXT, text="RightLabel")
    above = _detection_from_bbox((105, 40, 30, 10), YOLO_CLASS_TEXT, text="AboveLabel")
    below = _detection_from_bbox((105, 160, 30, 10), YOLO_CLASS_TEXT, text="BelowLabel")
    upper_left = _detection_from_bbox(
        (40, 40, 30, 10), YOLO_CLASS_TEXT, text="UpperLeftA"
    )
    upper_right = _detection_from_bbox(
        (160, 40, 30, 10), YOLO_CLASS_TEXT, text="UpperRightA"
    )
    lower_left = _detection_from_bbox(
        (40, 160, 30, 10), YOLO_CLASS_TEXT, text="LowerLeftA"
    )
    lower_right = _detection_from_bbox(
        (160, 160, 30, 10), YOLO_CLASS_TEXT, text="LowerRightA"
    )
    # Nested center-cell icons (away from the click) fill the global icon quota.
    fillers = [
        _detection_from_bbox(
            (101 + (i % 3) * 3, 101 + (i // 3) * 3, 2, 2),
            YOLO_CLASS_ELEMENT,
            icons=[{"chinese_id": f"Mid{i}"}],
        )
        for i in range(6)
    ]
    detections = [
        primary,
        left,
        right,
        above,
        below,
        upper_left,
        upper_right,
        lower_left,
        lower_right,
        *fillers,
    ]
    nearest = _nearest_candidates(detections, 110, 110)
    assert nearest[0] is primary
    for side_det in (
        left,
        right,
        above,
        below,
        upper_left,
        upper_right,
        lower_left,
        lower_right,
    ):
        assert side_det in nearest


def test_nearest_candidates_keeps_two_multi_char_texts_per_side() -> None:
    """Prefer at least two multi-char text landmarks on each directed side."""
    primary = _detection_from_bbox(
        (100, 100, 20, 20),
        YOLO_CLASS_TEXT,
        text="Target",
    )
    # Two multi-char texts on every directed side, plus a third on lower-right.
    sides = {
        "Left": [(20, 105), (50, 108)],
        "Right": [(160, 105), (190, 108)],
        "Above": [(105, 20), (108, 50)],
        "Below": [(105, 160), (108, 190)],
        "UL": [(20, 20), (50, 50)],
        "UR": [(160, 20), (190, 50)],
        "LL": [(20, 160), (50, 190)],
        "LR": [(160, 160), (190, 190), (220, 220)],
    }
    side_texts: list = []
    for prefix, centers in sides.items():
        for i, (cx, cy) in enumerate(centers):
            side_texts.append(
                _detection_from_bbox(
                    (cx - 15, cy - 5, 30, 10),
                    YOLO_CLASS_TEXT,
                    text=f"{prefix}{i}",
                )
            )
    # Nested center-cell icons that would otherwise fill the global quota first.
    fillers = [
        _detection_from_bbox(
            (101 + (i % 3) * 3, 101 + (i // 3) * 3, 2, 2),
            YOLO_CLASS_ELEMENT,
            icons=[{"chinese_id": f"Mid{i}"}],
        )
        for i in range(8)
    ]
    nearest = _nearest_candidates(
        [primary, *fillers, *side_texts],
        110,
        110,
    )
    assert nearest[0] is primary
    from src.common.nearby_side import LandmarkCell, landmark_cell_from_anchor_bbox

    primary_bbox = primary.bbox
    per_side: dict[LandmarkCell, int] = {}
    for det in nearest[1:]:
        if det.class_id != YOLO_CLASS_TEXT or len(det.text or "") <= 1:
            continue
        cell = landmark_cell_from_anchor_bbox(primary_bbox, det.cx, det.cy)
        per_side[cell] = per_side.get(cell, 0) + 1
    for cell in (
        LandmarkCell.LEFT,
        LandmarkCell.RIGHT,
        LandmarkCell.ABOVE,
        LandmarkCell.BELOW,
        LandmarkCell.UPPER_LEFT,
        LandmarkCell.UPPER_RIGHT,
        LandmarkCell.LOWER_LEFT,
        LandmarkCell.LOWER_RIGHT,
    ):
        assert per_side.get(cell, 0) >= 2, f"{cell} has {per_side.get(cell, 0)}"
    # Extra third lower-right text is also kept (all directed sides retained).
    assert sum(1 for d in nearest if (d.text or "").startswith("LR")) == 3


def test_nearest_candidates_skips_extra_center_cell_beyond_quota() -> None:
    """Center-band neighbors are not force-kept after global quotas are met."""
    primary = _detection_from_bbox(
        (100, 100, 80, 80),
        YOLO_CLASS_ELEMENT,
    )
    # Close nested multi-char texts + icons fill both quotas near the click.
    # Keep them off the click point so primary stays rank-0.
    close_texts = [
        _detection_from_bbox(
            (110 + (i % 4) * 4, 110 + (i // 4) * 4, 3, 3),
            YOLO_CLASS_TEXT,
            text=f"Tx{i}",
        )
        for i in range(8)
    ]
    close_icons = [
        _detection_from_bbox(
            (130 + i * 3, 110, 2, 2),
            YOLO_CLASS_ELEMENT,
            icons=[{"chinese_id": f"Near{i}"}],
        )
        for i in range(5)
    ]
    # Still inside the primary (center cell), but farther from the click.
    far_center_icon = _detection_from_bbox(
        (170, 170, 4, 4),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "FarCen"}],
    )
    # Directed-side texts must remain available for HTML.
    side_text = _detection_from_bbox(
        (40, 135, 30, 10), YOLO_CLASS_TEXT, text="LeftSide"
    )
    nearest = _nearest_candidates(
        [primary, *close_texts, *close_icons, far_center_icon, side_text],
        120,
        120,
    )
    assert nearest[0] is primary
    assert all(det in nearest for det in close_texts)
    assert all(det in nearest for det in close_icons)
    assert side_text in nearest
    assert far_center_icon not in nearest


def test_nearest_candidates_prefers_icon_over_scrollbar() -> None:
    """Nested icon on a scrollbar thumb ranks above the scrollbar."""
    scrollbar = _detection_from_bbox((1411, 386, 24, 563), YOLO_CLASS_SCROLLBAR)
    thumb = _detection_from_bbox(
        (1415, 945, 14, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下三角"}],
    )
    detections = [scrollbar, thumb]
    nearest = _nearest_candidates(
        detections, 1421, 947, min_multi_char_text_neighbors=None, limit=2
    )
    assert nearest[0] is thumb
    assert nearest[1] is scrollbar


def test_nearest_candidates_prefers_content_priority_on_overlap() -> None:
    """Overlapping hits: multi-char text > icon > single-char text > others."""
    scrollbar = _detection_from_bbox((0, 0, 100, 100), YOLO_CLASS_SCROLLBAR)
    single = _detection_from_bbox((10, 10, 20, 20), YOLO_CLASS_TEXT, text="中")
    icon = _detection_from_bbox(
        (15, 15, 30, 30),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "Chrome"}],
    )
    multi = _detection_from_bbox((5, 5, 80, 80), YOLO_CLASS_TEXT, text="Submit")
    detections = [scrollbar, single, icon, multi]
    nearest = _nearest_candidates(
        detections, 25, 25, min_multi_char_text_neighbors=None, limit=4
    )
    assert [d for d in nearest] == [multi, icon, single, scrollbar]


def test_nearest_candidates_prefers_icon_over_single_char_text() -> None:
    icon = _detection_from_bbox(
        (0, 0, 40, 40),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "設定"}],
    )
    single = _detection_from_bbox((5, 5, 10, 10), YOLO_CLASS_TEXT, text="×")
    nearest = _nearest_candidates(
        [single, icon], 10, 10, min_multi_char_text_neighbors=None, limit=2
    )
    assert nearest[0] is icon
    assert nearest[1] is single


def test_nearest_candidates_prefers_multi_char_text_over_icon() -> None:
    icon = _detection_from_bbox(
        (5, 5, 10, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "下載"}],
    )
    multi = _detection_from_bbox((0, 0, 40, 40), YOLO_CLASS_TEXT, text="下載檔案")
    nearest = _nearest_candidates(
        [icon, multi], 10, 10, min_multi_char_text_neighbors=None, limit=2
    )
    assert nearest[0] is multi
    assert nearest[1] is icon


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

    with patch("src.recorder.vision_context.imread_bgr", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
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
    fake_imread, fake_build = _parallel_safe_drag_fakes(start_detections, end_detections)

    with patch(
        "src.recorder.vision_context.imread_bgr",
        side_effect=fake_imread,
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_build,
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
    fake_imread, fake_build = _parallel_safe_drag_fakes(start_detections, end_detections)

    with patch(
        "src.recorder.vision_context.imread_bgr",
        side_effect=fake_imread,
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    destination = vision["destination"]
    assert "text='文件'" in destination["candidate_text"]





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
    fake_imread, fake_build = _parallel_safe_drag_fakes(start_detections, end_detections)

    with patch(
        "src.recorder.vision_context.imread_bgr",
        side_effect=fake_imread,
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_build,
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
    fake_imread, fake_build = _parallel_safe_drag_fakes(start_detections, end_detections)

    with patch(
        "src.recorder.vision_context.imread_bgr",
        side_effect=fake_imread,
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_build,
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

    with patch("src.recorder.vision_context.imread_bgr", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
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


def test_format_scrollbar_context_hint_with_adjacent_text() -> None:
    vision = {
        "local_cursor": (3611, 358),
        "candidates": [
            {
                "bbox": [3600, 272, 21, 172],
                "class_name": "scrollbar",
                "text": None,
            },
            {
                "bbox": [3500, 320, 60, 14],
                "class_name": "text",
                "text": "資產總覽",
            },
        ],
    }
    hint = format_scrollbar_context_hint(vision)
    assert hint == "滾動條旁可見內容: 「資產總覽」"
    assert format_field_context_hint(vision) == hint


def test_format_scrollbar_context_hint_without_adjacent_text() -> None:
    vision = {
        "local_cursor": (2115, 577),
        "candidates": [
            {
                "bbox": [2104, 156, 22, 842],
                "class_name": "scrollbar",
                "text": None,
            },
        ],
    }
    hint = format_scrollbar_context_hint(vision)
    assert hint == "最近的滾動條（無可辨識內容）"
    assert format_field_context_hint(vision) == hint


def test_format_field_context_hint_prefers_closer_input_over_scrollbar() -> None:
    vision = {
        "local_cursor": (700, 1056),
        "candidates": [
            {
                "bbox": [564, 1040, 221, 31],
                "class_name": "input",
                "text": None,
            },
            {
                "bbox": [602, 1049, 33, 16],
                "class_name": "text",
                "text": "搜尋",
            },
            {
                "bbox": [3600, 272, 21, 172],
                "class_name": "scrollbar",
                "text": None,
            },
            {
                "bbox": [3500, 320, 60, 14],
                "class_name": "text",
                "text": "資產總覽",
            },
        ],
    }
    assert (
        format_input_context_hint(vision)
        == "輸入欄內可見文字: 「搜尋」"
    )
    assert (
        format_field_context_hint(vision)
        == "輸入欄內可見文字: 「搜尋」"
    )


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
        "text": "",
        "icons": [{"chinese_id": "Chrome"}],
    }
    assert format_drag_candidate_anchor(candidate) == "「Chrome」圖示"


def test_format_drag_candidate_anchor_prefers_text_over_icon() -> None:
    candidate = {
        "class_name": "text",
        "text": "\ue024速的網域 (3)",
        "icons": [{"chinese_id": "下載"}],
    }
    assert format_drag_candidate_anchor(candidate) == "「速的網域 (3)」文字"


def test_format_drag_candidate_anchor_element_text() -> None:
    assert format_drag_candidate_anchor({"class_name": "element", "text": "a"}) == "「a」元素"


def test_format_drag_candidate_anchor_unknown_text() -> None:
    assert format_drag_candidate_anchor({"class_name": "unknown", "text": "搜"}) == "「搜」未知"


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


def test_drop_point_inside_candidate_allows_small_tolerance() -> None:
    candidate = {"bbox": [57, 272, 27, 12], "center": [70, 278], "text": "下載"}
    assert _drop_point_inside_candidate(66, 271, candidate)
    assert _drop_point_inside_candidate(57, 272 - _BBOX_HIT_TOLERANCE_PX, candidate)
    assert not _drop_point_inside_candidate(
        57, 272 - _BBOX_HIT_TOLERANCE_PX - 1, candidate
    )


def test_primary_candidate_offset_none_when_click_within_bbox_tolerance() -> None:
    vision = {
        "local_cursor": (66, 271),
        "candidates": [
            {
                "bbox": [57, 272, 27, 12],
                "center": [70, 278],
                "class_name": "text",
                "text": "下載",
            },
        ],
    }
    assert primary_candidate_offset(vision) is None


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
    fake_imread, fake_build = _parallel_safe_drag_fakes(start_detections, end_detections)

    with patch(
        "src.recorder.vision_context.imread_bgr",
        side_effect=fake_imread,
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=False)

    destination = vision["destination"]
    assert destination["candidates"][0]["text"] == "振銓"
    assert "text='振銓'" in destination["candidate_text"]
    assert "text='Desktop'" in destination["candidate_text"]


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
    fake_imread, fake_build = _parallel_safe_drag_fakes(start_detections, end_detections)

    with patch(
        "src.recorder.vision_context.imread_bgr",
        side_effect=fake_imread,
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_build,
    ):
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=False)

    destination = vision["destination"]
    assert "destination_offset_hints" in destination
    assert "「Desktop」" in destination["destination_offset_hints"]
