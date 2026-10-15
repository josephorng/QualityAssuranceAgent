"""Refine-phase UIA hydration when vision cached before window-step patch."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io_utils import write_json
from src.recorder.models import RecordedEvent, event_json_path
from src.recorder.orchestrator import (
    _PreparedEvent,
    _hydrate_uia_text_from_event,
    _refine_all_text_resolutions,
)


def test_hydrate_uia_text_from_event_fills_empty_cache() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="jt",
        window_snapshot_debug={
            "signals_after": {"focused": {"name": "搜尋方塊", "value": "今天"}},
        },
    )
    updated = _hydrate_uia_text_from_event(
        {
            "recorded_text": "jt",
            "uia_text": None,
            "ocr_text": "今天是|作麼日子",
            "resolved_text": "今天是|作麼日子",
            "source": "ocr",
        },
        event,
    )
    assert updated["uia_text"] == "今天"


def test_hydrate_uia_text_from_event_keeps_existing() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="jt",
        window_snapshot_debug={
            "signals_after": {"focused": {"name": "搜尋方塊", "value": "今天"}},
        },
    )
    updated = _hydrate_uia_text_from_event(
        {
            "recorded_text": "jt",
            "uia_text": "office",
            "ocr_text": "今天",
            "source": "uia",
        },
        event,
    )
    assert updated["uia_text"] == "office"


@pytest.mark.asyncio
async def test_refine_hydrates_uia_from_disk_and_prefers_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "recording"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "yolo_ocr").mkdir(parents=True)

    on_disk = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="jt",
        cursor_xy=(10, 10),
        window_snapshot_debug={
            "signals_after": {"focused": {"name": "搜尋方塊", "value": "今天"}},
        },
    )
    write_json(event_json_path(run_dir, 1), on_disk.to_dict())

    stale = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="text_input",
        text="jt",
        cursor_xy=(10, 10),
    )
    prepared = _PreparedEvent(
        event=stale,
        event_for_llm=stale,
        vision={},
        text_resolution={
            "recorded_text": "jt",
            "uia_text": None,
            "ocr_text": "今天是|作麼日子",
            "ocr_options": ["今天是|作麼日子"],
            "resolved_text": "今天是|作麼日子",
            "source": "ocr",
            "reason": "Chinese typing; prefer OCR",
            "meaningful": True,
        },
    )

    cancelled = await _refine_all_text_resolutions(
        [prepared],
        run_dir=run_dir,
        log_info=lambda _msg: None,
        should_cancel=None,
        max_workers=1,
    )
    assert cancelled is False
    assert prepared.text_resolution is not None
    assert prepared.text_resolution["uia_text"] == "今天"
    assert prepared.text_resolution["resolved_text"] == "今天"
    assert prepared.text_resolution["source"] == "uia"
    assert prepared.event_for_llm.text == "今天"
