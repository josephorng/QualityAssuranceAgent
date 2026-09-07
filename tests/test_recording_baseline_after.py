from __future__ import annotations

import json
import os
from pathlib import Path

from main import prepare_run_session
from src.common.runtime_context import (
    RUNTIME_COMMAND_MODE_ENV,
    SCRIPT_BASELINE_AFTER_ENV,
    SCRIPT_LINES_ENV,
    SCRIPT_OUTCOMES_ENV,
    SCRIPT_PATH_ENV,
    SMART_GOAL_ENV,
    SMART_MODE_ENV,
)
from src.common.script_helper import (
    collect_recording_baseline_after_paths,
    collect_recording_instructions,
)


def _write_recording(tmp_path: Path) -> Path:
    run_dir = tmp_path / "rec_demo"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir()
    (run_dir / "screenshots").mkdir()

    shot0 = run_dir / "screenshots" / "event_000.jpeg"
    shot1 = run_dir / "screenshots" / "event_001.jpeg"
    final_after = run_dir / "screenshots" / "final_after.jpeg"
    shot0.write_bytes(b"before0")
    shot1.write_bytes(b"before1")
    final_after.write_bytes(b"final")

    events = []
    for index, shot in ((0, shot0), (1, shot1)):
        rel = f"events/event_{index:03d}.json"
        events.append(rel)
        (run_dir / rel).write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": "2026-09-07T00:00:00+00:00",
                    "kind": "click",
                    "screenshot_path": str(shot),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        analysis = {
            "instruction": f"click step {index}",
            "expected_outcome": None,
            "use_expected_outcome": False,
        }
        if index == 0:
            analysis["wait_instruction"] = "等待 1 秒"
        (run_dir / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps(analysis, ensure_ascii=False),
            encoding="utf-8",
        )

    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": "rec_demo",
                "started_at_utc": "2026-09-07T00:00:00+00:00",
                "events": events,
                "final_after_screenshot": str(final_after),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_collect_recording_baseline_after_paths_aligns_with_wait_lines(tmp_path: Path) -> None:
    run_dir = _write_recording(tmp_path)
    instructions, outcomes = collect_recording_instructions(run_dir)
    baselines = collect_recording_baseline_after_paths(run_dir)

    assert instructions == ["等待 1 秒", "click step 0", "click step 1"]
    assert outcomes == [None, None, None]
    assert len(baselines) == len(instructions)
    assert baselines[0] is None  # wait line
    # step 0 after = next event before shot
    assert baselines[1] is not None
    assert baselines[1].endswith("event_001.jpeg")
    # last step after = final_after
    assert baselines[2] is not None
    assert baselines[2].endswith("final_after.jpeg")


def test_prepare_run_session_seeds_baseline_after_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)
    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    monkeypatch.delenv(SMART_GOAL_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_PATH_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_LINES_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_OUTCOMES_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_BASELINE_AFTER_ENV, raising=False)

    run_dir = _write_recording(tmp_path)
    instructions, _ = collect_recording_instructions(run_dir)
    prepare_run_session(
        runs_root=tmp_path / "runs",
        task=instructions[0],
        runtime_mode=False,
        selected_script_path=run_dir,
        script_steps=instructions,
        eye_monitor_indices=[1],
        clear_runs_root=False,
    )

    raw = os.environ[SCRIPT_BASELINE_AFTER_ENV]
    baselines = json.loads(raw)
    assert len(baselines) == len(instructions)
    assert baselines[0] is None
    assert isinstance(baselines[1], str) and baselines[1].endswith("event_001.jpeg")
    assert isinstance(baselines[2], str) and baselines[2].endswith("final_after.jpeg")


def test_prepare_run_session_drops_baselines_when_script_edited(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(SCRIPT_BASELINE_AFTER_ENV, raising=False)
    run_dir = _write_recording(tmp_path)
    # Edited hub script: fewer lines than collected baselines
    prepare_run_session(
        runs_root=tmp_path / "runs2",
        task="only one",
        runtime_mode=False,
        selected_script_path=run_dir,
        script_steps=["only one edited line"],
        eye_monitor_indices=[1],
        clear_runs_root=False,
    )
    baselines = json.loads(os.environ[SCRIPT_BASELINE_AFTER_ENV])
    assert baselines == [None]
