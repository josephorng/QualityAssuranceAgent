from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.recorder.analyze import (
    after_screenshot_for_outcome,
    analyze_event_to_cache,
    before_screenshot_for_outcome,
    enrich_click_instruction_offset,
    enrich_drag_instruction,
    enrich_drag_instruction_offset,
    enrich_drag_instruction_source,
    infer_expected_outcome,
    instruction_for_click,
    instruction_for_drag,
    instruction_for_key,
    instruction_for_scroll,
)
from src.recorder.models import RecordedEvent
from src.recorder.orchestrator import (
    _elapsed_seconds,
    _wait_instruction,
    analyze_recording_session,
)
from src.recorder.vision_context import (
    append_drag_nearby_context_comments,
    append_nearby_context_comment,
    collect_nearby_hint_labels,
    format_nearby_context_comment,
)

_VISION_WITH_NEARBY = {
    "used_vision": True,
    "local_cursor": (38, 636),
    "candidates": [
        {
            "bbox": [28, 626, 20, 20],
            "center": [38, 636],
            "class_name": "element",
            "text": "",
            "icons": [{"chinese_id": "Chrome"}],
        },
        {"class_name": "text", "text": "OneNote"},
        {
            "class_name": "element",
            "text": "",
            "icons": [{"chinese_id": "Docker"}],
        },
    ],
}


def test_collect_nearby_hint_labels_skips_primary_and_instruction_duplicates() -> None:
    labels = collect_nearby_hint_labels(
        _VISION_WITH_NEARBY,
        instruction="點擊「Chrome」圖示",
    )
    assert labels == ["「OneNote」文字", "「Docker」圖示"]


def test_collect_nearby_hint_labels_prefers_text_over_icons() -> None:
    """Text landmarks win even when icon neighbors are closer (earlier in list)."""
    vision = {
        "used_vision": True,
        "candidates": [
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Docker"}],
            },
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Edge"}],
            },
            {"class_name": "text", "text": "OneNote"},
            {"class_name": "text", "text": "Slack"},
        ],
    }
    labels = collect_nearby_hint_labels(vision, instruction="點擊「Chrome」圖示")
    assert labels == ["「OneNote」文字", "「Slack」文字"]


def test_collect_nearby_hint_labels_keeps_collecting_until_two_texts() -> None:
    """Keep taking multi-char text landmarks until two are found, skipping closer icons."""
    vision = {
        "used_vision": True,
        "candidates": [
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Docker"}],
            },
            {"class_name": "text", "text": "OneNote"},
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Edge"}],
            },
            {"class_name": "text", "text": "Slack"},
            {"class_name": "text", "text": "Teams"},
        ],
    }
    labels = collect_nearby_hint_labels(vision, instruction="點擊「Chrome」圖示")
    assert labels == ["「OneNote」文字", "「Slack」文字"]


def test_collect_nearby_hint_labels_skips_single_char_text() -> None:
    """Single-character text (often an icon miss) does not count as a text landmark."""
    vision = {
        "used_vision": True,
        "candidates": [
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "展開節點"}],
            },
            {"class_name": "text", "text": "中"},
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "資料夾"}],
            },
            {"class_name": "text", "text": "報表"},
            {"class_name": "text", "text": "設定"},
        ],
    }
    labels = collect_nearby_hint_labels(vision, instruction="點擊「展開節點」圖示")
    assert labels == ["「報表」文字", "「設定」文字"]


def test_collect_nearby_hint_labels_skips_unknown_class() -> None:
    vision = {
        "used_vision": True,
        "candidates": [
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
            {"class_name": "unknown", "text": "v"},
            {"class_name": "text", "text": "OneNote"},
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Docker"}],
            },
        ],
    }
    labels = collect_nearby_hint_labels(vision, instruction="點擊「Chrome」圖示")
    assert labels == ["「OneNote」文字", "「Docker」圖示"]
    assert not any("未知" in label for label in labels)


def test_format_nearby_context_comment() -> None:
    assert format_nearby_context_comment(["「OneNote」文字", "「Docker」圖示"]) == (
        "（附近有「OneNote」文字、「Docker」圖示）"
    )
    assert format_nearby_context_comment([]) is None


def test_append_nearby_context_comment() -> None:
    result = append_nearby_context_comment("點擊「Chrome」圖示", _VISION_WITH_NEARBY)
    assert result == "點擊「Chrome」圖示（附近有「OneNote」文字、「Docker」圖示）"


def test_append_nearby_context_comment_directed_from_geometry() -> None:
    from src.common.nearby_side import NearbyHint, Side
    from src.recorder.vision_context import collect_nearby_hints

    vision = {
        "used_vision": True,
        "candidates": [
            {
                "bbox": [40, 40, 20, 20],
                "center": [50, 50],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "矩形框線"}],
            },
            {
                "bbox": [90, 40, 40, 20],
                "center": [110, 50],
                "class_name": "text",
                "text": "顯示已授權電腦",
            },
            {
                "bbox": [200, 200, 20, 20],
                "center": [210, 210],
                "class_name": "text",
                "text": "其他",
            },
        ],
    }
    hints = collect_nearby_hints(vision, instruction="點擊「矩形框線」圖示")
    assert hints[0] == NearbyHint("「顯示已授權電腦」文字", Side.LEFT)
    assert hints[1] == NearbyHint("「其他」文字", Side.UPPER_LEFT)
    result = append_nearby_context_comment("點擊「矩形框線」圖示", vision)
    assert result == (
        "點擊「矩形框線」圖示（在「顯示已授權電腦」文字的左邊、在「其他」文字的左上方）"
    )


def test_list_nearby_landmark_options_returns_all_neighbors() -> None:
    from src.recorder.vision_context import list_nearby_landmark_options

    vision = {
        "used_vision": True,
        "candidates": [
            {
                "bbox": [40, 40, 20, 20],
                "center": [50, 50],
                "class_name": "text",
                "text": "搜尋",
            },
            {
                "bbox": [90, 40, 40, 20],
                "center": [110, 50],
                "class_name": "text",
                "text": "已選取 2 個項目",
            },
            {
                "bbox": [10, 80, 30, 20],
                "center": [25, 90],
                "class_name": "text",
                "text": "45 個項目",
            },
            {
                "bbox": [200, 40, 20, 20],
                "center": [210, 50],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
            {
                "bbox": [300, 40, 10, 10],
                "center": [305, 45],
                "class_name": "unknown",
                "text": None,
            },
        ],
    }
    options = list_nearby_landmark_options(
        vision, instruction="將滑鼠移到「搜尋」文字"
    )
    labels = [item["label"] for item in options]
    assert labels == [
        "「已選取 2 個項目」文字",
        "「45 個項目」文字",
        "「Chrome」圖示",
    ]
    assert len(options) == 3
    assert options[0]["side"] == "left"
    assert "（左邊）" in options[0]["display"]


def test_list_nearby_landmark_options_includes_input_and_scrollbar() -> None:
    from src.recorder.vision_context import list_nearby_landmark_options

    vision = {
        "used_vision": True,
        "candidates": [
            {
                "bbox": [40, 40, 20, 20],
                "center": [50, 50],
                "class_name": "text",
                "text": "搜尋",
            },
            {
                "bbox": [80, 40, 200, 30],
                "center": [180, 55],
                "class_name": "input",
                "text": None,
            },
            {
                "bbox": [300, 0, 16, 200],
                "center": [308, 100],
                "class_name": "scrollbar",
                "text": None,
            },
            {
                "bbox": [90, 90, 40, 14],
                "center": [110, 97],
                "class_name": "text",
                "text": "標題",
            },
        ],
    }
    options = list_nearby_landmark_options(
        vision, instruction="將滑鼠移到「搜尋」文字"
    )
    labels = [item["label"] for item in options]
    assert "輸入欄" in labels
    assert "滾動條" in labels
    assert "「標題」文字" in labels


def test_collect_nearby_hints_force_includes_containing_input() -> None:
    from src.common.nearby_side import NearbyHint, Side
    from src.recorder.vision_context import (
        collect_nearby_hints,
        list_nearby_landmark_options,
    )

    vision = {
        "used_vision": True,
        "local_cursor": [100, 50],
        "candidates": [
            {
                "bbox": [90, 45, 30, 14],
                "center": [105, 52],
                "class_name": "text",
                "text": "搜尋",
            },
            {
                "bbox": [40, 30, 200, 50],
                "center": [140, 55],
                "class_name": "input",
                "text": None,
            },
            {
                "bbox": [300, 0, 40, 14],
                "center": [320, 7],
                "class_name": "text",
                "text": "標題",
            },
            {
                "bbox": [300, 40, 40, 14],
                "center": [320, 47],
                "class_name": "text",
                "text": "副標",
            },
        ],
    }
    hints = collect_nearby_hints(vision, instruction="將滑鼠移到「搜尋」文字")
    assert hints[0] == NearbyHint("輸入欄", Side.INSIDE)
    assert [h.label for h in hints[1:]] == ["「標題」文字", "「副標」文字"]

    options = list_nearby_landmark_options(
        vision, instruction="將滑鼠移到「搜尋」文字"
    )
    by_label = {item["label"]: item for item in options}
    assert by_label["輸入欄"]["side"] == "inside"
    assert "（裡面）" in by_label["輸入欄"]["display"]


def test_collect_nearby_hints_force_includes_containing_scrollbar() -> None:
    from src.common.nearby_side import NearbyHint, Side
    from src.recorder.vision_context import collect_nearby_hints

    vision = {
        "used_vision": True,
        "local_cursor": [308, 80],
        "candidates": [
            {
                "bbox": [300, 70, 16, 20],
                "center": [308, 80],
                "class_name": "text",
                "text": "握柄",
            },
            {
                "bbox": [300, 0, 16, 200],
                "center": [308, 100],
                "class_name": "scrollbar",
                "text": None,
            },
            {
                "bbox": [40, 40, 40, 14],
                "center": [60, 47],
                "class_name": "text",
                "text": "旁標",
            },
        ],
    }
    hints = collect_nearby_hints(vision, instruction="將滑鼠移到「握柄」文字")
    assert hints[0] == NearbyHint("滾動條", Side.INSIDE)


def test_collect_nearby_hints_skips_container_when_it_is_primary() -> None:
    from src.recorder.vision_context import collect_nearby_hints

    vision = {
        "used_vision": True,
        "local_cursor": [140, 55],
        "candidates": [
            {
                "bbox": [40, 30, 200, 50],
                "center": [140, 55],
                "class_name": "input",
                "text": None,
            },
            {
                "bbox": [40, 100, 40, 14],
                "center": [60, 107],
                "class_name": "text",
                "text": "旁標",
            },
        ],
    }
    hints = collect_nearby_hints(vision, instruction="將滑鼠移到輸入欄")
    assert all(h.label != "輸入欄" for h in hints)


def test_append_drag_nearby_context_comments() -> None:
    vision = {
        "used_vision": True,
        "candidates": [
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
            {"class_name": "text", "text": "OneNote"},
        ],
    }
    destination = {
        "candidates": [
            {"class_name": "text", "text": "Desktop"},
            {
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Recycle Bin"}],
            },
        ],
    }
    instruction = "從「Chrome」圖示拖到「Desktop」文字下方49個像素的位置"
    result = append_drag_nearby_context_comments(instruction, vision, destination)
    assert result == (
        "從「Chrome」圖示（起點附近有「OneNote」文字）拖到「Desktop」文字下方49個像素的位置"
        "（終點附近有「Recycle Bin」圖示）"
    )


@pytest.mark.asyncio
async def test_analyze_event_to_cache_key_press_is_deterministic(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="enter",
        screenshot_path="",
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision={"used_vision": False, "candidate_text": "", "local_cursor": None},
        )

    assert result is not None
    assert result["instruction"] == "按下 Enter 鍵"
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_text_input_is_deterministic(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=7,
        timestamp_utc="t",
        kind="text_input",
        text="打電話的時候",
        screenshot_path="",
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(event, run_dir=tmp_path)

    assert result == {"instruction": "輸入「打電話的時候」"}
    llm_mock.assert_not_called()


def test_typed_text_from_instruction_round_trips() -> None:
    from src.recorder.analyze import instruction_for_text_input, typed_text_from_instruction

    instruction = instruction_for_text_input("hello」world")
    assert instruction == "輸入「hello」world」"
    assert typed_text_from_instruction(instruction) == "hello」world"
    assert typed_text_from_instruction("點擊「搜尋」按鈕") is None
    assert typed_text_from_instruction("輸入「」") is None


@pytest.mark.asyncio
async def test_analyze_event_to_cache_window_change_is_deterministic(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=2,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(400, 120),
        button="left",
        screenshot_path="",
        window_change={"action": "minimize", "title": "Google Chrome", "confidence": "high"},
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision={"used_vision": False, "candidate_text": "", "local_cursor": None},
        )

    assert result == {"instruction": "最小化「Google Chrome」視窗"}
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_medium_close_is_deterministic(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=3,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(640, 70),
        button="left",
        screenshot_path="",
        window_change={
            "action": "close",
            "title": "連線資訊.txt - 記事本",
            "confidence": "medium",
            "from_title_bar_close": True,
        },
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision={"used_vision": False, "candidate_text": "", "local_cursor": None},
        )

    assert result == {"instruction": "關閉「連線資訊.txt - 記事本」視窗"}
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_non_caption_close_keeps_click(tmp_path: Path) -> None:
    """儲存/取消 dismissals still detect close, but emit the click instruction."""
    event = RecordedEvent(
        index=4,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(280, 420),
        button="left",
        screenshot_path="",
        window_change={
            "action": "close",
            "title": "另存新檔",
            "confidence": "high",
            "from_title_bar_close": False,
        },
    )
    vision = {
        "used_vision": True,
        "local_cursor": (280, 420),
        "candidate_text": "儲存",
        "candidates": [
            {
                "bbox": [250, 410, 60, 24],
                "center": [280, 422],
                "class_name": "text",
                "text": "儲存",
            },
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == "將滑鼠移到「儲存」文字，並點擊滑鼠一下。"
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_cancel_close_keeps_click(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=5,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(360, 420),
        button="left",
        screenshot_path="",
        window_change={
            "action": "close",
            "title": "確認",
            "confidence": "high",
            "from_title_bar_close": False,
        },
    )
    vision = {
        "used_vision": True,
        "local_cursor": (360, 420),
        "candidate_text": "取消",
        "candidates": [
            {
                "bbox": [330, 410, 60, 24],
                "center": [360, 422],
                "class_name": "text",
                "text": "取消",
            },
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == "將滑鼠移到「取消」文字，並點擊滑鼠一下。"
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_ignores_shell_host_close(tmp_path: Path) -> None:
    """Closing 快顯主機 is a side effect; keep the click instruction instead."""
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(624, 1059),
        button="left",
        screenshot_path="",
        window_change={"action": "close", "title": "快顯主機", "confidence": "medium"},
    )
    vision = {
        "used_vision": True,
        "local_cursor": (619, 1057),
        "candidate_text": "",
        "candidates": [
            {
                "bbox": [603, 1049, 32, 16],
                "center": [619, 1057],
                "class_name": "text",
                "text": "搜尋",
            },
            {
                "bbox": [577, 1048, 14, 15],
                "center": [584, 1056],
                "class_name": "element",
                "icons": [{"chinese_id": "搜尋"}],
            },
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result == {"instruction": "將滑鼠移到「搜尋」文字，並點擊滑鼠一下。"}
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_recording_session_binds_run_state(tmp_path: Path) -> None:
    from src.common.run_state import get_run_state_manager, reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_bind_test"
    (run_dir / "events").mkdir(parents=True)
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="enter",
        screenshot_path="",
    )
    (run_dir / "events" / "event_001.json").write_text(
        json.dumps(event.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": "screen_record_bind_test",
                "started_at_utc": "t",
                "stopped_at_utc": "t",
                "event_count": 1,
                "events": ["events/event_001.json"],
            }
        ),
        encoding="utf-8",
    )

    llm_payload = {"instruction": "按下 Enter 鍵"}

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ) as llm_mock:
        report = await analyze_recording_session(run_dir)

    assert report["cached"] == 1
    assert (run_dir / "run.log").is_file()
    llm_mock.assert_not_called()
    reset_run_state_manager()
    with pytest.raises(RuntimeError, match="Run state not initialized"):
        get_run_state_manager().require_paths()


@pytest.mark.asyncio
async def test_analyze_recording_session_writes_instructions(tmp_path: Path) -> None:
    from src.common.run_state import reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_test"
    (run_dir / "events").mkdir(parents=True)
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="2832u04cj842k7g6c.4",
        screenshot_path="",
    )
    (run_dir / "events" / "event_001.json").write_text(
        json.dumps(event.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": "screen_record_test",
                "started_at_utc": "t",
                "stopped_at_utc": "t",
                "event_count": 1,
                "events": ["events/event_001.json"],
            }
        ),
        encoding="utf-8",
    )

    llm_payload = {"instruction": "should not be used"}

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ), patch(
        "src.recorder.orchestrator.resolve_text_input_text",
        new=AsyncMock(
            return_value={
                "text": "打電話的時候",
                "recorded_text": "2832u04cj842k7g6c.4",
                "source": "ocr",
                "meaningful": False,
                "reason": "ime",
                "vision": None,
            }
        ),
    ):
        report = await analyze_recording_session(run_dir)

    assert report["cached"] == 1
    assert report["skipped"] == 0
    assert report["instructions"] == ["輸入「打電話的時候」"]
    analysis = json.loads((run_dir / "analysis" / "event_001.json").read_text(encoding="utf-8"))
    assert analysis["instruction"] == "輸入「打電話的時候」"
    assert "tool_calls" not in analysis
    assert analysis["text_resolution"]["resolved_text"] == "打電話的時候"
    recording_html = run_dir / "recording_steps.html"
    assert recording_html.is_file()
    assert "輸入「打電話的時候」" in recording_html.read_text(encoding="utf-8")


def test_elapsed_seconds_requires_valid_ordered_timezone_aware_timestamps() -> None:
    assert (
        _elapsed_seconds(
            "2026-07-30T03:00:00+00:00",
            "2026-07-30T03:00:03.250+00:00",
        )
        == 3.25
    )
    assert _elapsed_seconds("invalid", "2026-07-30T03:00:04+00:00") is None
    assert _elapsed_seconds("2026-07-30T03:00:00", "2026-07-30T03:00:04") is None
    assert (
        _elapsed_seconds(
            "2026-07-30T03:00:04+00:00",
            "2026-07-30T03:00:00+00:00",
        )
        is None
    )


def test_wait_instruction_ceilings_to_integer_seconds() -> None:
    assert _wait_instruction(4.0) == "等待 4 秒"
    assert _wait_instruction(3.0001) == "等待 4 秒"
    assert _wait_instruction(3.1254) == "等待 4 秒"


@pytest.mark.asyncio
async def test_analyze_recording_session_inserts_wait_only_over_threshold_seconds(
    tmp_path: Path,
) -> None:
    from src.common.run_state import reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_wait_test"
    (run_dir / "events").mkdir(parents=True)
    events = [
        RecordedEvent(
            index=1,
            timestamp_utc="2026-07-30T03:00:00+00:00",
            kind="key_press",
            key="enter",
            screenshot_path="",
        ),
        RecordedEvent(
            index=2,
            timestamp_utc="2026-07-30T03:00:10+00:00",
            kind="key_press",
            key="tab",
            screenshot_path="",
        ),
        RecordedEvent(
            index=3,
            timestamp_utc="2026-07-30T03:00:20.250+00:00",
            kind="key_press",
            key="esc",
            screenshot_path="",
        ),
    ]
    event_paths: list[str] = []
    for event in events:
        relative_path = f"events/event_{event.index:03d}.json"
        event_paths.append(relative_path)
        (run_dir / relative_path).write_text(
            json.dumps(event.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "started_at_utc": events[0].timestamp_utc,
                "stopped_at_utc": events[-1].timestamp_utc,
                "event_count": len(events),
                "events": event_paths,
            }
        ),
        encoding="utf-8",
    )

    no_vision = {"used_vision": False, "candidate_text": "", "local_cursor": None}
    with patch(
        "src.recorder.orchestrator.build_vision_context",
        new=AsyncMock(return_value=no_vision),
    ):
        report = await analyze_recording_session(run_dir)

    assert report["instructions"] == [
        "按下 Enter 鍵",
        "按下 Tab 鍵",
        "等待 11 秒",
        "按下 Esc 鍵",
    ]
    assert report["expected_outcomes"] == [None, None, None, None]
    second_analysis = json.loads(
        (run_dir / "analysis" / "event_002.json").read_text(encoding="utf-8")
    )
    third_analysis = json.loads(
        (run_dir / "analysis" / "event_003.json").read_text(encoding="utf-8")
    )
    assert second_analysis["elapsed_since_previous_seconds"] == 10.0
    assert "wait_instruction" not in second_analysis
    assert third_analysis["elapsed_since_previous_seconds"] == 10.25
    assert third_analysis["wait_instruction"] == "等待 11 秒"


@pytest.mark.asyncio
async def test_analyze_recording_session_drops_trailing_agent_restore(
    tmp_path: Path,
) -> None:
    from src.common.run_state import reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_drop_agent_restore"
    (run_dir / "events").mkdir(parents=True)
    events = [
        RecordedEvent(
            index=1,
            timestamp_utc="2026-07-30T03:00:00+00:00",
            kind="key_press",
            key="enter",
            screenshot_path="",
        ),
        RecordedEvent(
            index=2,
            timestamp_utc="2026-07-30T03:00:01+00:00",
            kind="click",
            cursor_xy=(612, 894),
            button="left",
            screenshot_path="",
            window_change={
                "action": "restored",
                "title": "電腦使用代理",
                "confidence": "medium",
            },
        ),
    ]
    event_paths: list[str] = []
    for event in events:
        relative_path = f"events/event_{event.index:03d}.json"
        event_paths.append(relative_path)
        (run_dir / relative_path).write_text(
            json.dumps(event.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "started_at_utc": events[0].timestamp_utc,
                "stopped_at_utc": events[-1].timestamp_utc,
                "event_count": len(events),
                "events": event_paths,
            }
        ),
        encoding="utf-8",
    )

    no_vision = {"used_vision": False, "candidate_text": "", "local_cursor": None}
    with patch(
        "src.recorder.orchestrator.build_vision_context",
        new=AsyncMock(return_value=no_vision),
    ):
        report = await analyze_recording_session(run_dir)

    assert report["recorded"] == 1
    assert report["cached"] == 1
    assert report["instructions"] == ["按下 Enter 鍵"]
    assert (run_dir / "analysis" / "event_001.json").is_file()
    assert not (run_dir / "analysis" / "event_002.json").is_file()
    assert not (run_dir / "events" / "event_002.json").exists()
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert session["event_count"] == 1
    assert session["events"] == ["events/event_001.json"]
    import_log = (run_dir / "import.log").read_text(encoding="utf-8")
    assert "dropping trailing agent restore event index=2" in import_log
    assert "purged trailing agent restore event index=2 remaining=1" in import_log


@pytest.mark.asyncio
async def test_analyze_recording_session_persists_coalesced_clicks(tmp_path: Path) -> None:
    from src.common.run_state import reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_persist_coalesce"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "screenshots").mkdir()
    events = [
        RecordedEvent(
            index=1,
            timestamp_utc="2026-08-13T08:00:00+00:00",
            kind="click",
            cursor_xy=(100, 200),
            button="left",
            screenshot_path=str(run_dir / "screenshots" / "event_001.jpeg"),
        ),
        RecordedEvent(
            index=2,
            timestamp_utc="2026-08-13T08:00:00.300000+00:00",
            kind="click",
            cursor_xy=(101, 200),
            button="left",
            screenshot_path=str(run_dir / "screenshots" / "event_002.jpeg"),
        ),
        RecordedEvent(
            index=3,
            timestamp_utc="2026-08-13T08:00:00.600000+00:00",
            kind="click",
            cursor_xy=(102, 201),
            button="left",
            screenshot_path=str(run_dir / "screenshots" / "event_003.jpeg"),
        ),
        RecordedEvent(
            index=4,
            timestamp_utc="2026-08-13T08:00:05+00:00",
            kind="click",
            cursor_xy=(400, 400),
            button="left",
            screenshot_path=str(run_dir / "screenshots" / "event_004.jpeg"),
        ),
    ]
    event_paths: list[str] = []
    for event in events:
        shot = Path(event.screenshot_path)
        shot.write_bytes(b"jpeg")
        relative_path = f"events/event_{event.index:03d}.json"
        event_paths.append(relative_path)
        (run_dir / relative_path).write_text(
            json.dumps(event.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "started_at_utc": events[0].timestamp_utc,
                "stopped_at_utc": events[-1].timestamp_utc,
                "event_count": len(events),
                "events": event_paths,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    async def _fake_analyze(event, **_kwargs):
        if event.kind == "triple_click":
            return {"instruction": "將滑鼠移到目標，並連按3下。"}
        return {"instruction": "將滑鼠移到另一目標，並點擊滑鼠一下。"}

    no_vision = {"used_vision": False, "candidate_text": "", "local_cursor": None}
    with patch(
        "src.recorder.orchestrator.build_vision_context",
        new=AsyncMock(return_value=no_vision),
    ), patch(
        "src.recorder.orchestrator.analyze_event_to_cache",
        new=AsyncMock(side_effect=_fake_analyze),
    ), patch(
        "src.recorder.orchestrator.infer_expected_outcome",
        new=AsyncMock(return_value=None),
    ):
        report = await analyze_recording_session(run_dir)

    assert report["recorded"] == 2
    assert report["cached"] == 2
    assert report["instructions"] == [
        "將滑鼠移到目標，並連按3下。",
        "將滑鼠移到另一目標，並點擊滑鼠一下。",
    ]
    assert (run_dir / "events" / "event_001.json").is_file()
    assert not (run_dir / "events" / "event_002.json").exists()
    assert not (run_dir / "events" / "event_003.json").exists()
    assert (run_dir / "events" / "event_004.json").is_file()
    kept = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert kept["kind"] == "triple_click"
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert session["event_count"] == 2
    assert session["events"] == ["events/event_001.json", "events/event_004.json"]
    html = (run_dir / "recording_steps.html").read_text(encoding="utf-8")
    assert "將滑鼠移到目標，並連按3下。" in html
    assert "將滑鼠移到另一目標，並點擊滑鼠一下。" in html
    assert html.count('instruction-title">點擊<') == 0
    assert "連按3下" in html
    import_log = (run_dir / "import.log").read_text(encoding="utf-8")
    assert "persisted coalesced events kept=2 purged=[2, 3]" in import_log


@pytest.mark.asyncio
async def test_analyze_recording_session_drops_trailing_agent_restore_from_snapshot_debug(
    tmp_path: Path,
) -> None:
    """Older captures may lack window_change; re-diff snapshot debug instead."""
    from src.common.run_state import reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_drop_agent_restore_debug"
    (run_dir / "events").mkdir(parents=True)
    taskbar = {
        "hwnd": 65714,
        "title": "",
        "pid": 6324,
        "left": 0,
        "top": 880,
        "width": 1918,
        "height": 40,
        "is_minimized": False,
        "is_maximized": False,
    }
    events = [
        RecordedEvent(
            index=1,
            timestamp_utc="2026-07-30T03:00:00+00:00",
            kind="key_press",
            key="tab",
            screenshot_path="",
        ),
        RecordedEvent(
            index=2,
            timestamp_utc="2026-07-30T03:00:01+00:00",
            kind="click",
            cursor_xy=(612, 894),
            button="left",
            screenshot_path="",
            window_snapshot_debug={
                "windows_before": [
                    {
                        "hwnd": 459206,
                        "title": "電腦使用代理",
                        "pid": 9284,
                        "left": -32000,
                        "top": -32000,
                        "width": 160,
                        "height": 28,
                        "is_minimized": True,
                        "is_maximized": False,
                    },
                    taskbar,
                ],
                "windows_after": [
                    {
                        "hwnd": 459206,
                        "title": "電腦使用代理",
                        "pid": 9284,
                        "left": 156,
                        "top": 156,
                        "width": 976,
                        "height": 719,
                        "is_minimized": False,
                        "is_maximized": False,
                    },
                    taskbar,
                ],
            },
        ),
    ]
    event_paths: list[str] = []
    for event in events:
        relative_path = f"events/event_{event.index:03d}.json"
        event_paths.append(relative_path)
        (run_dir / relative_path).write_text(
            json.dumps(event.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "started_at_utc": events[0].timestamp_utc,
                "stopped_at_utc": events[-1].timestamp_utc,
                "event_count": len(events),
                "events": event_paths,
            }
        ),
        encoding="utf-8",
    )

    no_vision = {"used_vision": False, "candidate_text": "", "local_cursor": None}
    with patch(
        "src.recorder.orchestrator.build_vision_context",
        new=AsyncMock(return_value=no_vision),
    ):
        report = await analyze_recording_session(run_dir)

    assert report["instructions"] == ["按下 Tab 鍵"]
    assert not (run_dir / "analysis" / "event_002.json").is_file()
    assert not (run_dir / "events" / "event_002.json").exists()
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert session["event_count"] == 1
    assert session["events"] == ["events/event_001.json"]


def test_enrich_drag_instruction_offset_appends_exact_pixels() -> None:
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
    instruction = "從「Chrome」圖示拖到「Desktop」文字"
    enriched = enrich_drag_instruction_offset(instruction, destination)
    assert enriched == "從「Chrome」圖示拖到「Desktop」文字下方49個像素的位置"


def test_enrich_click_instruction_offset_replaces_nearby_with_pixels() -> None:
    vision = {
        "local_cursor": (419, 875),
        "candidates": [
            {
                "bbox": [380, 829, 106, 14],
                "center": [433, 836],
                "class_name": "text",
                "text": "自訂Office 範本",
            },
            {
                "bbox": [356, 828, 17, 15],
                "center": [364, 835],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "資料夾"}],
            },
        ],
    }
    instruction = "在「快顯主機」視窗中的「自訂Office 範本」文字附近按右鍵"
    enriched = enrich_click_instruction_offset(instruction, vision)
    assert enriched == (
        "在「快顯主機」視窗中的「自訂Office 範本」文字"
        "左方14個像素、下方39個像素的位置按右鍵"
    )


def test_enrich_click_instruction_offset_leaves_inside_bbox_unchanged() -> None:
    vision = {
        "local_cursor": (433, 836),
        "candidates": [
            {
                "bbox": [380, 829, 106, 14],
                "center": [433, 836],
                "class_name": "text",
                "text": "自訂Office 範本",
            },
        ],
    }
    instruction = "點擊「自訂Office 範本」文字附近"
    assert enrich_click_instruction_offset(instruction, vision) == instruction


def test_enrich_click_instruction_offset_inserts_when_no_nearby() -> None:
    vision = {
        "local_cursor": (419, 875),
        "candidates": [
            {
                "bbox": [380, 829, 106, 14],
                "center": [433, 836],
                "class_name": "text",
                "text": "自訂Office 範本",
            },
        ],
    }
    instruction = "點擊「自訂Office 範本」文字"
    enriched = enrich_click_instruction_offset(instruction, vision)
    assert enriched == "點擊「自訂Office 範本」文字左方14個像素、下方39個像素的位置"


def test_enrich_drag_instruction_source_replaces_wrong_llm_pick() -> None:
    vision = {
        "candidates": [
            {
                "center": [190, 567],
                "class_name": "text",
                "text": "DDocker",
            },
            {
                "center": [111, 531],
                "class_name": "element",
                "text": "a",
            },
        ],
    }
    instruction = "從「a」元素拖到「Edge」圖示"
    enriched = enrich_drag_instruction_source(instruction, vision)
    assert enriched == "從「DDocker」文字拖到「Edge」圖示"


def test_instruction_for_drag_builds_from_nearest_candidates() -> None:
    vision = {
        "candidates": [
            {
                "center": [38, 636],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
        ],
    }
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
    assert instruction_for_drag(vision, destination) == (
        "從「Chrome」圖示拖到「Desktop」文字下方49個像素的位置"
    )


def test_instruction_for_drag_event_six_like_case() -> None:
    vision = {
        "candidates": [
            {
                "center": [190, 567],
                "class_name": "text",
                "text": "DDocker",
            },
        ],
    }
    destination = {
        "local_cursor": (192, 39),
        "candidates": [
            {
                "center": [114, 30],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Edge"}],
            },
        ],
    }
    assert instruction_for_drag(vision, destination) == (
        "從「DDocker」文字拖到「Edge」圖示右方78個像素、下方9個像素的位置"
    )


def test_instruction_for_drag_returns_none_without_candidates() -> None:
    assert instruction_for_drag({"candidates": []}, {"candidates": [{"text": "x"}]}) is None
    assert instruction_for_drag({"candidates": [{"text": "x"}]}, {"candidates": []}) is None


def test_instruction_for_click_builds_from_nearest_candidate() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(38, 636),
        button="left",
        screenshot_path="",
    )
    assert instruction_for_click(event, _VISION_WITH_NEARBY) == "將滑鼠移到「Chrome」圖示"


def test_instruction_for_click_right_click_with_offset() -> None:
    event = RecordedEvent(
        index=6,
        timestamp_utc="t",
        kind="right_click",
        cursor_xy=(419, 875),
        button="right",
        screenshot_path="",
    )
    vision = {
        "local_cursor": (419, 875),
        "candidates": [
            {
                "bbox": [380, 829, 106, 14],
                "center": [433, 836],
                "class_name": "text",
                "text": "自訂Office 範本",
            },
        ],
    }
    assert instruction_for_click(event, vision) == (
        "將滑鼠移到「自訂Office 範本」文字左方14個像素、下方39個像素的位置"
    )


def test_instruction_for_click_double_click() -> None:
    event = RecordedEvent(
        index=5,
        timestamp_utc="t",
        kind="double_click",
        cursor_xy=(1073, 184),
        button="left",
        screenshot_path="",
    )
    vision = {
        "local_cursor": (1073, 184),
        "candidates": [
            {
                "bbox": [1057, 176, 31, 15],
                "center": [1073, 184],
                "class_name": "text",
                "text": "文件",
            },
        ],
    }
    assert instruction_for_click(event, vision) == "將滑鼠移到「文件」文字"


def test_instruction_for_click_input_field_with_visible_text() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(675, 1056),
        button="left",
        screenshot_path="",
    )
    vision = {
        "local_cursor": (675, 1056),
        "candidates": [
            {
                "bbox": [564, 1040, 221, 31],
                "center": [675, 1056],
                "class_name": "input",
                "text": None,
            },
            {
                "bbox": [602, 1049, 33, 16],
                "center": [619, 1057],
                "class_name": "text",
                "text": "搜尋",
            },
        ],
    }
    assert instruction_for_click(event, vision) == "將滑鼠移到「搜尋」文字所在的輸入欄"


def test_instruction_for_click_empty_input_field() -> None:
    event = RecordedEvent(
        index=3,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(408, 56),
        button="left",
        screenshot_path="",
    )
    vision = {
        "local_cursor": (408, 56),
        "candidates": [
            {
                "bbox": [170, 34, 1243, 43],
                "center": [792, 56],
                "class_name": "input",
                "text": None,
            },
            {
                "bbox": [413, 100, 17, 15],
                "center": [421, 108],
                "class_name": "element",
                "text": None,
                "icons": [{"chinese_id": "排序或同步"}],
            },
        ],
    }
    assert instruction_for_click(event, vision) == "將滑鼠移到輸入欄"


def test_instruction_for_click_scrollbar_with_adjacent_text() -> None:
    event = RecordedEvent(
        index=2,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(3611, 358),
        button="left",
        screenshot_path="",
    )
    vision = {
        "local_cursor": (3611, 358),
        "candidates": [
            {
                "bbox": [3600, 272, 21, 172],
                "center": [3611, 358],
                "class_name": "scrollbar",
                "text": None,
            },
            {
                "bbox": [3500, 320, 60, 14],
                "center": [3530, 327],
                "class_name": "text",
                "text": "資產總覽",
            },
        ],
    }
    assert (
        instruction_for_click(event, vision)
        == "將滑鼠移到「資產總覽」文字區域的滾動條"
    )


def test_instruction_for_click_empty_scrollbar() -> None:
    event = RecordedEvent(
        index=3,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(2115, 577),
        button="left",
        screenshot_path="",
    )
    vision = {
        "local_cursor": (2115, 577),
        "candidates": [
            {
                "bbox": [2104, 156, 22, 842],
                "center": [2115, 577],
                "class_name": "scrollbar",
                "text": None,
            },
            {
                "bbox": [413, 100, 17, 15],
                "center": [421, 108],
                "class_name": "element",
                "text": None,
                "icons": [{"chinese_id": "排序或同步"}],
            },
        ],
    }
    assert instruction_for_click(event, vision) == "將滑鼠移到滾動條"


def test_instruction_for_scroll_on_scrollbar() -> None:
    event = RecordedEvent(
        index=4,
        timestamp_utc="t",
        kind="scroll",
        cursor_xy=(2115, 577),
        scroll_delta=-3,
        screenshot_path="",
    )
    vision = {
        "local_cursor": (2115, 577),
        "candidates": [
            {
                "bbox": [2104, 156, 22, 842],
                "center": [2115, 577],
                "class_name": "scrollbar",
                "text": None,
            },
        ],
    }
    assert instruction_for_scroll(event, vision) == "在滾動條附近向下捲動"


def test_instruction_for_click_returns_none_for_generic_anchor() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(10, 10),
        button="left",
        screenshot_path="",
    )
    vision = {
        "local_cursor": (10, 10),
        "candidates": [
            {"bbox": [0, 0, 20, 20], "center": [10, 10], "class_name": "element", "text": None},
        ],
    }
    assert instruction_for_click(event, vision) is None


def test_instruction_for_key_enter() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="enter",
        screenshot_path="",
    )
    assert instruction_for_key(event) == "按下 Enter 鍵"


def test_instruction_for_key_hotkey_ctrl_c() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="hotkey",
        keys=["ctrl", "c"],
        screenshot_path="",
    )
    assert instruction_for_key(event) == "按下 Ctrl+C"


def test_instruction_for_key_hotkey_legacy_ctrl_a_control_char() -> None:
    """Older recordings stored Ctrl+A as keys=['ctrl', '\\u0001']."""
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="hotkey",
        keys=["ctrl", "\u0001"],
        screenshot_path="",
    )
    assert instruction_for_key(event) == "按下 Ctrl+A"


def test_instruction_for_key_hotkey_orders_modifiers() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="hotkey",
        keys=["shift", "ctrl", "s"],
        screenshot_path="",
    )
    assert instruction_for_key(event) == "按下 Ctrl+Shift+S"


def test_instruction_for_key_returns_none_for_unknown_vk() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="vk_123",
        screenshot_path="",
    )
    assert instruction_for_key(event) is None


def test_instruction_for_scroll_with_named_target() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="scroll",
        cursor_xy=(100, 200),
        scroll_delta=-3,
        screenshot_path="",
    )
    vision = {
        "local_cursor": (100, 200),
        "candidates": [
            {
                "bbox": [80, 180, 40, 20],
                "center": [100, 190],
                "class_name": "text",
                "text": "檔案清單",
            },
        ],
    }
    assert instruction_for_scroll(event, vision) == "在「檔案清單」文字附近向下捲動"


def test_instruction_for_scroll_without_named_target() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="scroll",
        cursor_xy=(100, 200),
        scroll_delta=2,
        screenshot_path="",
    )
    vision = {
        "local_cursor": (100, 200),
        "candidates": [
            {"bbox": [0, 0, 20, 20], "center": [10, 10], "class_name": "element", "text": None},
        ],
    }
    assert instruction_for_scroll(event, vision) == "向上捲動"


def test_instruction_for_scroll_returns_none_without_delta() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="scroll",
        cursor_xy=(100, 200),
        scroll_delta=None,
        screenshot_path="",
    )
    assert instruction_for_scroll(event, {"candidates": []}) is None


def test_enrich_drag_instruction_normalizes_source_destination_and_offset() -> None:
    vision = {
        "candidates": [
            {
                "center": [190, 567],
                "class_name": "text",
                "text": "DDocker",
            },
        ],
    }
    destination = {
        "local_cursor": (192, 39),
        "candidates": [
            {
                "center": [114, 30],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Edge"}],
            },
        ],
    }
    instruction = "從「a」元素拖到「Edge」圖示下方9個像素的位置"
    enriched = enrich_drag_instruction(
        instruction,
        vision=vision,
        destination=destination,
    )
    assert enriched == "從「DDocker」文字拖到「Edge」圖示右方78個像素、下方9個像素的位置"


def test_enrich_drag_instruction_offset_replaces_partial_llm_offset() -> None:
    destination = {
        "local_cursor": (192, 39),
        "candidates": [
            {
                "center": [114, 30],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Edge"}],
            },
        ],
    }
    instruction = "從「a」元素拖到「Edge」圖示下方9個像素的位置"
    enriched = enrich_drag_instruction_offset(instruction, destination)
    assert enriched == "從「a」元素拖到「Edge」圖示右方78個像素、下方9個像素的位置"


def test_enrich_drag_instruction_offset_replaces_incorrect_llm_offset() -> None:
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
    instruction = "從「Chrome」圖示拖到「Desktop」文字右方12個像素的位置"
    enriched = enrich_drag_instruction_offset(instruction, destination)
    assert enriched == "從「Chrome」圖示拖到「Desktop」文字下方49個像素的位置"


@pytest.mark.asyncio
async def test_analyze_event_to_cache_drag_is_deterministic(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=2,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(38, 636),
        end_xy=(2109, 637),
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "candidate_text": "[index 0] icons=Chrome",
        "local_cursor": (38, 636),
        "candidates": [
            {
                "center": [38, 636],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
        ],
        "destination": {
            "local_cursor": (189, 638),
            "candidate_text": "[index 1] text='Desktop'",
            "candidates": [
                {
                    "center": [191, 589],
                    "class_name": "text",
                    "text": "Desktop",
                },
            ],
            "destination_offset_hints": "[index 0] 「Desktop」: 下方49個像素",
        },
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == "從「Chrome」圖示拖到「Desktop」文字下方49個像素的位置"
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_hotkey_is_deterministic(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="hotkey",
        keys=["ctrl", "c"],
        screenshot_path="",
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(event, run_dir=tmp_path)

    assert result == {"instruction": "按下 Ctrl+C"}
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_scroll_is_deterministic(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="scroll",
        cursor_xy=(100, 200),
        scroll_delta=-1,
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "local_cursor": (100, 200),
        "candidates": [
            {
                "bbox": [80, 180, 40, 20],
                "center": [100, 190],
                "class_name": "text",
                "text": "檔案清單",
            },
            {"class_name": "text", "text": "其他"},
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == (
        "在「檔案清單」文字附近向下捲動（附近有「其他」文字）"
    )
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_key_falls_back_to_llm_for_unknown_vk(
    tmp_path: Path,
) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="vk_999",
        screenshot_path="",
    )
    llm_payload = {"instruction": "按下未知按鍵"}

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision={"used_vision": False, "candidate_text": "", "local_cursor": None},
        )

    assert result is not None
    assert result["instruction"] == "按下未知按鍵"
    llm_mock.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_click_appends_nearby_context(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(38, 636),
        button="left",
        screenshot_path="",
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=_VISION_WITH_NEARBY,
        )

    assert result is not None
    assert result["instruction"] == (
        "將滑鼠移到「Chrome」圖示（附近有「OneNote」文字、「Docker」圖示），並點擊滑鼠一下。"
    )
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_ctrl_click_suffix(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(38, 636),
        button="left",
        modifiers=["ctrl"],
        screenshot_path="",
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=_VISION_WITH_NEARBY,
        )

    assert result is not None
    assert result["instruction"] == (
        "將滑鼠移到「Chrome」圖示（附近有「OneNote」文字、「Docker」圖示），並Ctrl+點擊。"
    )
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_hold_suffix(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="hold",
        cursor_xy=(38, 636),
        button="left",
        duration_seconds=1.2,
        screenshot_path="",
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=_VISION_WITH_NEARBY,
        )

    assert result is not None
    assert result["instruction"] == (
        "將滑鼠移到「Chrome」圖示（附近有「OneNote」文字、「Docker」圖示），並按住約1.2秒。"
    )
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_right_hold_suffix(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="hold",
        cursor_xy=(38, 636),
        button="right",
        duration_seconds=2.0,
        screenshot_path="",
    )

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=_VISION_WITH_NEARBY,
        )

    assert result is not None
    assert result["instruction"] == (
        "將滑鼠移到「Chrome」圖示（附近有「OneNote」文字、「Docker」圖示），並用右鍵按住約2秒。"
    )
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_shift_double_click_suffix(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="double_click",
        cursor_xy=(38, 636),
        button="left",
        modifiers=["shift"],
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "local_cursor": (38, 636),
        "candidates": [
            {
                "bbox": [28, 626, 20, 20],
                "center": [38, 636],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == "將滑鼠移到「Chrome」圖示，並Shift+連按2下。"
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_triple_click_suffix(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="triple_click",
        cursor_xy=(38, 636),
        button="left",
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "local_cursor": (38, 636),
        "candidates": [
            {
                "bbox": [28, 626, 20, 20],
                "center": [38, 636],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == "將滑鼠移到「Chrome」圖示，並連按3下。"
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_multi_click_count_suffix(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(38, 636),
        button="left",
        click_count=4,
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "local_cursor": (38, 636),
        "candidates": [
            {
                "bbox": [28, 626, 20, 20],
                "center": [38, 636],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == "將滑鼠移到「Chrome」圖示，並連按4下。"
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_click_enriches_offset_then_nearby(
    tmp_path: Path,
) -> None:
    event = RecordedEvent(
        index=6,
        timestamp_utc="t",
        kind="right_click",
        cursor_xy=(419, 875),
        button="right",
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "local_cursor": (419, 875),
        "candidates": [
            {
                "bbox": [380, 829, 106, 14],
                "center": [433, 836],
                "class_name": "text",
                "text": "自訂Office 範本",
            },
            {
                "bbox": [356, 828, 17, 15],
                "center": [364, 835],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "資料夾"}],
            },
            {
                "bbox": [378, 800, 143, 12],
                "center": [449, 806],
                "class_name": "text",
                "text": "WindowsPowerShell",
            },
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == (
        "將滑鼠移到「自訂Office 範本」文字左方14個像素、下方39個像素的位置"
        "（在「WindowsPowerShell」文字的下面、在「資料夾」圖示的右邊），用右鍵點選。"
    )
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_click_falls_back_to_llm_when_generic(
    tmp_path: Path,
) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(10, 10),
        button="left",
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "local_cursor": (10, 10),
        "candidates": [
            {"bbox": [0, 0, 20, 20], "center": [10, 10], "class_name": "element", "text": None},
        ],
    }
    llm_payload = {"instruction": "將滑鼠移到空白區域"}

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == "將滑鼠移到空白區域，並點擊滑鼠一下。"
    llm_mock.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_event_to_cache_drag_appends_nearby_after_enrichment(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=2,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(38, 636),
        end_xy=(189, 638),
        screenshot_path="",
    )
    vision = {
        "used_vision": True,
        "candidate_text": "[index 0] icons=Chrome",
        "local_cursor": (38, 636),
        "candidates": [
            {
                "center": [38, 636],
                "class_name": "element",
                "text": "",
                "icons": [{"chinese_id": "Chrome"}],
            },
            {"center": [100, 600], "class_name": "text", "text": "OneNote"},
        ],
        "destination": {
            "local_cursor": (189, 638),
            "candidate_text": "[index 1] text='Desktop'",
            "candidates": [
                {
                    "center": [191, 589],
                    "class_name": "text",
                    "text": "Desktop",
                },
                {
                    "center": [250, 620],
                    "class_name": "element",
                    "text": "",
                    "icons": [{"chinese_id": "Recycle Bin"}],
                },
            ],
            "destination_offset_hints": "[index 0] 「Desktop」: 下方49個像素",
        },
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=vision,
        )

    assert result is not None
    assert result["instruction"] == (
        "從「Chrome」圖示（起點附近有「OneNote」文字）拖到「Desktop」文字下方49個像素的位置"
        "（終點附近有「Recycle Bin」圖示）"
    )
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_infer_expected_outcome_uses_before_after_screenshots(tmp_path: Path) -> None:
    before = tmp_path / "before.jpeg"
    after = tmp_path / "after.jpeg"
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value={"expected_outcome": "對話框已開啟"}),
    ) as llm_mock:
        outcome = await infer_expected_outcome(
            instruction="將滑鼠移到「開啟」按鈕，並點擊滑鼠一下。",
            before_screenshot=str(before),
            after_screenshot=str(after),
        )

    assert outcome == "對話框已開啟"
    assert llm_mock.await_count == 1
    messages = llm_mock.await_args.kwargs["messages"]
    assert messages[0]["images"] == [str(before), str(after)]


@pytest.mark.asyncio
async def test_infer_expected_outcome_skips_identical_paths(tmp_path: Path) -> None:
    shot = tmp_path / "same.jpeg"
    shot.write_bytes(b"x")
    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(),
    ) as llm_mock:
        outcome = await infer_expected_outcome(
            instruction="點擊確定",
            before_screenshot=str(shot),
            after_screenshot=str(shot),
        )
    assert outcome is None
    llm_mock.assert_not_called()


def test_after_screenshot_prefers_next_event_before_shot(tmp_path: Path) -> None:
    before = tmp_path / "event_001.jpeg"
    next_before = tmp_path / "event_002.jpeg"
    before.write_bytes(b"a")
    next_before.write_bytes(b"b")
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        screenshot_path=str(before),
    )
    next_event = RecordedEvent(
        index=2,
        timestamp_utc="t2",
        kind="click",
        screenshot_path=str(next_before),
    )
    assert before_screenshot_for_outcome(event) == str(before)
    assert after_screenshot_for_outcome(event, next_event) == str(next_before)


@pytest.mark.asyncio
async def test_analyze_recording_session_writes_expected_outcome(tmp_path: Path) -> None:
    from src.common.run_state import reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_expected_outcome"
    shots = run_dir / "screenshots"
    shots.mkdir(parents=True)
    before = shots / "event_001.jpeg"
    after = shots / "event_002.jpeg"
    before.write_bytes(b"before-bytes")
    after.write_bytes(b"after-bytes")
    (run_dir / "events").mkdir(parents=True)
    events = [
        RecordedEvent(
            index=1,
            timestamp_utc="2026-07-30T03:00:00+00:00",
            kind="key_press",
            key="enter",
            screenshot_path=str(before),
        ),
        RecordedEvent(
            index=2,
            timestamp_utc="2026-07-30T03:00:01+00:00",
            kind="key_press",
            key="tab",
            screenshot_path=str(after),
        ),
    ]
    event_paths: list[str] = []
    for event in events:
        relative_path = f"events/event_{event.index:03d}.json"
        event_paths.append(relative_path)
        (run_dir / relative_path).write_text(
            json.dumps(event.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "started_at_utc": events[0].timestamp_utc,
                "stopped_at_utc": events[-1].timestamp_utc,
                "event_count": len(events),
                "events": event_paths,
            }
        ),
        encoding="utf-8",
    )

    no_vision = {"used_vision": False, "candidate_text": "", "local_cursor": None}
    with patch(
        "src.recorder.orchestrator.build_vision_context",
        new=AsyncMock(return_value=no_vision),
    ), patch(
        "src.recorder.orchestrator.infer_expected_outcome",
        new=AsyncMock(return_value="對話框已開啟"),
    ) as outcome_mock:
        report = await analyze_recording_session(run_dir)

    assert report["instructions"] == ["按下 Enter 鍵", "按下 Tab 鍵"]
    assert report["expected_outcomes"] == ["對話框已開啟", None]
    first_analysis = json.loads(
        (run_dir / "analysis" / "event_001.json").read_text(encoding="utf-8")
    )
    assert first_analysis["expected_outcome"] == "對話框已開啟"
    assert outcome_mock.await_count == 1
    assert outcome_mock.await_args.kwargs["before_screenshot"] == str(before)
    assert outcome_mock.await_args.kwargs["after_screenshot"] == str(after)
