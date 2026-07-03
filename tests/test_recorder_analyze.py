from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.recorder.analyze import analyze_event_to_cache
from src.recorder.models import RecordedEvent
from src.recorder.orchestrator import analyze_recording_session


@pytest.mark.asyncio
async def test_analyze_event_to_cache_parses_llm_json(tmp_path: Path) -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="key_press",
        key="enter",
        screenshot_path="",
    )
    llm_payload = {
        "instruction": "按下 Enter 鍵。",
        "tool_calls": [
            {"name": "press_key", "arguments": {"instruction": "按下 Enter", "key": "enter"}},
        ],
    }

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
    assert result["tool_calls"][0]["name"] == "press_key"


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

    llm_payload = {
        "instruction": "按下 Enter 鍵。",
        "tool_calls": [
            {"name": "press_key", "arguments": {"instruction": "按下 Enter", "key": "enter"}},
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ), patch("src.recorder.orchestrator.upsert_tool_calls"):
        report = await analyze_recording_session(run_dir)

    assert report["cached"] == 1
    assert (run_dir / "run.log").is_file()
    reset_run_state_manager()
    with pytest.raises(RuntimeError, match="Run state not initialized"):
        get_run_state_manager().require_paths()


@pytest.mark.asyncio
async def test_analyze_recording_session_writes_cache(tmp_path: Path) -> None:
    from src.common.run_state import reset_run_state_manager

    reset_run_state_manager()
    run_dir = tmp_path / "screen_record_test"
    (run_dir / "events").mkdir(parents=True)
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="a",
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

    llm_payload = {
        "instruction": "輸入字母 a。",
        "tool_calls": [
            {"name": "type_text", "arguments": {"instruction": "輸入 a", "text": "a"}},
        ],
    }

    with patch(
        "src.recorder.analyze.request_json_with_retry",
        new=AsyncMock(return_value=llm_payload),
    ), patch(
        "src.recorder.orchestrator.upsert_tool_calls",
    ) as upsert_mock:
        report = await analyze_recording_session(run_dir)

    assert report["cached"] == 1
    assert report["skipped"] == 0
    upsert_mock.assert_called_once()
    args = upsert_mock.call_args
    assert args[0][0] == "輸入字母 a。"
    assert args[0][1][0]["name"] == "type_text"
