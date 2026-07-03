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
