from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.recorder.analyze import (
    analyze_event_to_cache,
    enrich_drag_instruction,
    enrich_drag_instruction_offset,
    enrich_drag_instruction_source,
    instruction_for_drag,
)
from src.recorder.models import RecordedEvent
from src.recorder.orchestrator import analyze_recording_session
from src.recorder.vision_context import (
    append_drag_nearby_context_comments,
    append_nearby_context_comment,
    collect_nearby_hint_labels,
    format_nearby_context_comment,
)

_VISION_WITH_NEARBY = {
    "used_vision": True,
    "candidates": [
        {
            "class_name": "element",
            "text": "chrome",
            "icons": [{"chinese_id": "Chrome"}],
        },
        {"class_name": "text", "text": "OneNote"},
        {
            "class_name": "element",
            "text": "docker",
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


def test_format_nearby_context_comment() -> None:
    assert format_nearby_context_comment(["「OneNote」文字", "「Docker」圖示"]) == (
        "（附近有「OneNote」文字、「Docker」圖示）"
    )
    assert format_nearby_context_comment([]) is None


def test_append_nearby_context_comment() -> None:
    result = append_nearby_context_comment("點擊「Chrome」圖示", _VISION_WITH_NEARBY)
    assert result == "點擊「Chrome」圖示（附近有「OneNote」文字、「Docker」圖示）"


def test_append_drag_nearby_context_comments() -> None:
    vision = {
        "used_vision": True,
        "candidates": [
            {
                "class_name": "element",
                "text": "chrome",
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
                "text": "recycle",
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
async def test_analyze_event_to_cache_parses_llm_json(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="enter",
        screenshot_path="",
    )
    llm_payload = {"instruction": "按下 Enter 鍵。"}

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ):
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision={"used_vision": False, "candidate_text": "", "local_cursor": None},
        )

    assert result is not None
    assert result["instruction"] == "按下 Enter 鍵。"


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
        window_change={"action": "close", "title": "連線資訊.txt - 記事本", "confidence": "medium"},
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

    llm_payload = {"instruction": "按下 Enter 鍵。"}

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ):
        report = await analyze_recording_session(run_dir)

    assert report["cached"] == 1
    assert (run_dir / "run.log").is_file()
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
                "text": "chrome",
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
                "text": "edge-icon",
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
                "text": "edge-icon",
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
                "text": "edge-icon",
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
                "text": "chrome",
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
async def test_analyze_event_to_cache_click_appends_nearby_context(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(38, 636),
        button="left",
        screenshot_path="",
    )
    llm_payload = {"instruction": "點擊「Chrome」圖示"}

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ):
        result = await analyze_event_to_cache(
            event,
            run_dir=tmp_path,
            vision=_VISION_WITH_NEARBY,
        )

    assert result is not None
    assert result["instruction"] == (
        "點擊「Chrome」圖示（附近有「OneNote」文字、「Docker」圖示）"
    )


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
                "text": "chrome",
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
                    "text": "recycle",
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
