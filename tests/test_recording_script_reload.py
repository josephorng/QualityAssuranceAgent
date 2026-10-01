from __future__ import annotations

import json
from pathlib import Path

from src.brain.module import BrainModule
from src.common.runtime_context import SCRIPT_PATH_ENV
from src.common.script_helper import merge_unstarted_recording_steps


def _merge(
    *,
    step_index: int,
    lines: list[str],
    event_indices: list[int | None],
    fresh_lines: list[str],
    fresh_event_indices: list[int | None],
    outcomes: list[str | None] | None = None,
    fresh_outcomes: list[str | None] | None = None,
) -> tuple[list[str], list[str | None], list[int | None]]:
    merged = merge_unstarted_recording_steps(
        step_index=step_index,
        current_lines=lines,
        current_outcomes=outcomes or [None] * len(lines),
        current_baselines=[None] * len(lines),
        current_settles=[None] * len(lines),
        current_verifies=[{} for _ in lines],
        current_event_indices=event_indices,
        fresh_lines=fresh_lines,
        fresh_outcomes=fresh_outcomes or [None] * len(fresh_lines),
        fresh_baselines=[None] * len(fresh_lines),
        fresh_settles=[None] * len(fresh_lines),
        fresh_verifies=[{} for _ in fresh_lines],
        fresh_event_indices=fresh_event_indices,
    )
    return merged[0], merged[1], merged[5]


def test_merge_keeps_started_steps_and_updates_the_rest() -> None:
    lines, outcomes, event_indices = _merge(
        step_index=1,
        lines=["open", "click save"],
        event_indices=[0, 1],
        outcomes=[None, "old result"],
        fresh_lines=["open edited", "click save now", "close"],
        fresh_event_indices=[0, 1, 2],
        fresh_outcomes=["ignored", "new result", None],
    )
    assert lines == ["open", "click save now", "close"]
    assert outcomes == [None, "new result", None]
    assert event_indices == [0, 1, 2]


def test_merge_drops_deleted_unstarted_steps() -> None:
    lines, _outcomes, event_indices = _merge(
        step_index=1,
        lines=["open", "click save", "close"],
        event_indices=[0, 1, 2],
        fresh_lines=["open", "close"],
        fresh_event_indices=[0, 2],
    )
    assert lines == ["open", "close"]
    assert event_indices == [0, 2]


def test_merge_runs_an_inserted_unstarted_step_next() -> None:
    lines, _outcomes, event_indices = _merge(
        step_index=1,
        lines=["open", "close"],
        event_indices=[0, 2],
        fresh_lines=["open", "wait", "close"],
        fresh_event_indices=[0, 1, 2],
    )
    assert lines == ["open", "wait", "close"]
    assert event_indices == [0, 1, 2]


def test_reload_pending_recording_steps_reads_disk(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "rec"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir()
    (run_dir / "screenshots").mkdir()
    events = []
    for index, instruction in enumerate(("open", "click save")):
        rel = f"events/event_{index:03d}.json"
        events.append(rel)
        (run_dir / rel).write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": "2026-09-07T00:00:00+00:00",
                    "kind": "click",
                    "screenshot_path": "",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps({"instruction": instruction, "use_expected_outcome": False}),
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps({"run_id": "rec", "events": events}),
        encoding="utf-8",
    )
    monkeypatch.setenv(SCRIPT_PATH_ENV, str(run_dir))
    monkeypatch.delenv("CUA_RUNTIME_COMMAND_MODE", raising=False)
    monkeypatch.delenv("CUA_SMART_MODE", raising=False)

    (run_dir / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "instruction": "click save now",
                "use_expected_outcome": True,
                "expected_outcome": "dialog closed",
            }
        ),
        encoding="utf-8",
    )

    logs: list[str] = []
    brain = BrainModule.__new__(BrainModule)
    brain.manager = type("Manager", (), {"log_info": lambda self, message: logs.append(message)})()
    brain._script_step_index = 1
    brain.script_lines = ["open", "click save"]
    brain.script_expected_outcomes = [None, None]
    brain.script_baseline_after_paths = [None, None]
    brain.script_settle_after_seconds = [None, None]
    brain.script_window_verifies = [{}, {}]
    brain._script_event_indices = [0, 1]

    brain.reload_pending_recording_steps()

    assert brain._script_step_index == 1
    assert brain.script_lines == ["open", "click save now"]
    assert brain.script_expected_outcomes[1] == "dialog closed"
    assert brain._script_event_indices == [0, 1]
    assert any("Applied recording edits" in message for message in logs)


def test_jump_to_script_step_clamps_and_moves_the_cursor() -> None:
    brain = BrainModule.__new__(BrainModule)
    logs: list[str] = []
    brain.manager = type("Manager", (), {"log_info": lambda self, message: logs.append(message)})()
    brain.script_lines = ["one", "two", "three"]
    brain._script_step_index = 2
    brain._pending_settle_deadline_perf = 1.0

    brain.jump_to_script_step(-4)
    assert brain._script_step_index == 0
    assert brain._pending_settle_deadline_perf is None

    brain.jump_to_script_step(99)
    assert brain._script_step_index == 2
    assert any("Replay jump" in message for message in logs)
