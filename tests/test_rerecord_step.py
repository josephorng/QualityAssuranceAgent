"""Mid-replay single-step re-record: overwrite event N and reanalyze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.common.io_utils import read_json
from src.common.run_control import take_recording_replay_dirty
from src.common.script_helper import (
    invalidate_recording_replay_cache,
    load_recording_replay_steps,
    merge_unstarted_recording_steps,
)
from src.recorder.rerecord_step import (
    RERECORDABLE_KINDS,
    event_kind_is_rerecordable,
    overwrite_pointer_event,
    purge_rerecord_analysis_artifacts,
    reanalyze_recording_event,
    replace_and_reanalyze_pointer_event,
)


def _write_recording(run_dir: Path, *, event_count: int = 2) -> None:
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "screenshots").mkdir(exist_ok=True)
    (run_dir / "yolo_ocr").mkdir(exist_ok=True)
    (run_dir / "signals").mkdir(exist_ok=True)
    events: list[str] = []
    for index in range(1, event_count + 1):
        rel = f"events/event_{index:03d}.json"
        events.append(rel)
        shot = f"screenshots/event_{index:03d}.jpeg"
        (run_dir / shot).write_bytes(b"jpeg")
        (run_dir / rel).write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": f"2026-09-07T00:00:0{index}+00:00",
                    "kind": "click",
                    "button": "left",
                    "click_count": 1,
                    "cursor_xy": [10 * index, 20 * index],
                    "screenshot_path": shot,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {
                    "event_index": index,
                    "instruction": f"click {index}",
                    "expected_outcome": f"result {index}",
                    "use_expected_outcome": True,
                    "settle_after_seconds": 1.5,
                    "window_verify": {"appeared": [{"title": "A"}]},
                    "window_verify_disabled": ["appeared:0"],
                    "tool_calls": [{"name": "click", "arguments": {}}],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "yolo_ocr" / f"event_{index:03d}.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (run_dir / "signals" / f"event_{index:03d}_before.json").write_text(
            "{}",
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps({"run_id": run_dir.name, "events": events, "event_count": event_count}),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "instructions": [f"click {i}" for i in range(1, event_count + 1)],
                "expected_outcomes": [f"result {i}" for i in range(1, event_count + 1)],
            }
        ),
        encoding="utf-8",
    )


def test_event_kind_is_rerecordable() -> None:
    assert RERECORDABLE_KINDS == {"click", "double_click", "right_click"}
    assert event_kind_is_rerecordable("click")
    assert event_kind_is_rerecordable("double_click")
    assert event_kind_is_rerecordable("right_click")
    assert not event_kind_is_rerecordable("drag")
    assert not event_kind_is_rerecordable("text_input")
    assert not event_kind_is_rerecordable(None)


def test_purge_removes_yolo_and_signals_keeps_analysis(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)
    end_shot = run_dir / "screenshots" / "event_002_end.jpeg"
    end_shot.write_bytes(b"end")

    purge_rerecord_analysis_artifacts(run_dir, 2)

    assert (run_dir / "analysis" / "event_002.json").is_file()
    assert (run_dir / "events" / "event_002.json").is_file()
    assert (run_dir / "screenshots" / "event_002.jpeg").is_file()
    assert not (run_dir / "yolo_ocr" / "event_002.json").is_file()
    assert not (run_dir / "signals" / "event_002_before.json").is_file()
    assert not end_shot.is_file()
    assert (run_dir / "yolo_ocr" / "event_001.json").is_file()


def test_overwrite_keeps_event_index_and_session_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)
    monkeypatch.setattr(
        "src.recorder.rerecord_step._WINDOW_AFTER_SETTLE_S",
        0.0,
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.snapshot_top_level_windows",
        lambda: [],
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_step_signals",
        lambda **_kwargs: {"foreground": {"title": "after"}},
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_fast_signals",
        lambda **_kwargs: {},
    )

    shot = run_dir / "screenshots" / "event_002.jpeg"
    shot.write_bytes(b"new-jpeg")
    event = overwrite_pointer_event(
        run_dir,
        2,
        kind="double_click",
        cursor_xy=(100, 200),
        button="left",
        timestamp_utc="2026-10-30T00:00:00+00:00",
        screenshot_path=str(shot),
        monitor_index=1,
        monitor_offset=(0, 0),
        click_window=None,
        windows_before=(),
        signals_before={"foreground": {"title": "before"}},
        click_count=2,
    )

    assert event.index == 2
    assert event.kind == "double_click"
    raw = read_json(run_dir / "events" / "event_002.json", {})
    assert raw["index"] == 2
    assert raw["kind"] == "double_click"
    assert raw["cursor_xy"] == [100, 200]
    assert raw["click_count"] == 2
    session = read_json(run_dir / "session.json", {})
    assert session["events"] == [
        "events/event_001.json",
        "events/event_002.json",
    ]
    assert (run_dir / "events" / "event_001.json").is_file()
    assert not (run_dir / "events" / "event_003.json").is_file()


@pytest.mark.asyncio
async def test_reanalyze_rewrites_instruction_and_preserves_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)

    async def fake_prepare(event, *, run_dir, log_info):
        from src.recorder.orchestrator import _PreparedEvent

        return _PreparedEvent(
            event=event,
            event_for_llm=event,
            vision={
                "used_vision": True,
                "candidate_text": "儲存",
                "candidates": [],
                "local_cursor": [10, 20],
            },
            text_resolution=None,
            vision_seconds=0.01,
            vision_cache_hit=False,
        )

    async def fake_analyze(event, *, run_dir, vision=None, log_info=None):
        return {"instruction": "點擊「儲存」"}

    monkeypatch.setattr(
        "src.recorder.orchestrator.prepare_event_vision",
        fake_prepare,
    )
    monkeypatch.setattr(
        "src.recorder.analyze.analyze_event_to_cache",
        fake_analyze,
    )
    monkeypatch.setattr(
        "src.recorder.compile_tool_calls.compile_tool_calls",
        lambda *_args, **_kwargs: [{"name": "click", "arguments": {"text": "儲存"}}],
    )
    monkeypatch.setattr(
        "src.recorder.compile_tool_calls.rebuild_recording_instruction_tool_cache",
        lambda _run_dir: _run_dir / "instruction_tool_cache.json",
    )
    monkeypatch.setattr(
        "src.common.session_html.write_recording_html_from_run",
        lambda run_dir, update_index=True: run_dir / "recording_steps.html",
    )
    monkeypatch.setattr(
        "src.recorder.orchestrator._bind_run_state_for_analysis",
        lambda *_args, **_kwargs: None,
    )
    take_recording_replay_dirty()

    result = await reanalyze_recording_event(run_dir, 2)
    assert result["event_index"] == 2
    assert result["instruction"] == "點擊「儲存」"

    analysis = read_json(run_dir / "analysis" / "event_002.json", {})
    assert analysis["instruction"] == "點擊「儲存」"
    assert analysis["expected_outcome"] == "result 2"
    assert analysis["use_expected_outcome"] is True
    assert analysis["window_verify_disabled"] == ["appeared:0"]
    assert analysis["settle_after_seconds"] == 1.5
    # Re-recording event 2 must not rewrite event 1's settle from the timestamp gap.
    previous = read_json(run_dir / "analysis" / "event_001.json", {})
    assert previous["settle_after_seconds"] == 1.5

    report = read_json(run_dir / "report.json", {})
    assert report["instructions"][1] == "點擊「儲存」"
    assert take_recording_replay_dirty() is True


def test_replace_and_reanalyze_keeps_index_for_replay_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)
    monkeypatch.setattr("src.recorder.rerecord_step._WINDOW_AFTER_SETTLE_S", 0.0)
    monkeypatch.setattr(
        "src.recorder.rerecord_step.snapshot_top_level_windows",
        lambda: [],
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_step_signals",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_fast_signals",
        lambda **_kwargs: {},
    )

    async def fake_reanalyze(run_dir, event_index, *, log_info=None):
        analysis_path = Path(run_dir) / "analysis" / f"event_{event_index:03d}.json"
        analysis_path.write_text(
            json.dumps(
                {
                    "event_index": event_index,
                    "instruction": "點擊「新目標」",
                    "use_expected_outcome": False,
                }
            ),
            encoding="utf-8",
        )
        report = read_json(run_dir / "report.json", {})
        report["instructions"] = ["click 1", "點擊「新目標」"]
        (run_dir / "report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        from src.common.run_control import mark_recording_replay_dirty

        invalidate_recording_replay_cache(run_dir)
        mark_recording_replay_dirty()
        return {
            "event_index": event_index,
            "instruction": "點擊「新目標」",
            "kind": "click",
        }

    monkeypatch.setattr(
        "src.recorder.rerecord_step.reanalyze_recording_event",
        fake_reanalyze,
    )

    shot = run_dir / "screenshots" / "event_002.jpeg"
    shot.write_bytes(b"replaced")
    result = replace_and_reanalyze_pointer_event(
        run_dir,
        2,
        kind="click",
        cursor_xy=(50, 60),
        button="left",
        timestamp_utc="2026-10-30T01:00:00+00:00",
        screenshot_path=str(shot),
        monitor_index=1,
        monitor_offset=(0, 0),
        click_window=None,
        windows_before=(),
        signals_before={},
    )
    assert result["event_index"] == 2
    assert result["instruction"] == "點擊「新目標」"

    invalidate_recording_replay_cache(run_dir)
    fresh = load_recording_replay_steps(run_dir)
    merged = merge_unstarted_recording_steps(
        step_index=1,
        current_lines=["click 1", "click 2"],
        current_outcomes=[None, None],
        current_baselines=[None, None],
        current_settles=[None, None],
        current_verifies=[{}, {}],
        current_event_indices=[1, 2],
        fresh_lines=fresh[0],
        fresh_outcomes=fresh[1],
        fresh_baselines=fresh[2],
        fresh_settles=fresh[3],
        fresh_verifies=fresh[4],
        fresh_event_indices=fresh[5],
    )
    assert merged[0] == ["click 1", "點擊「新目標」"]
    assert merged[5] == [1, 2]
