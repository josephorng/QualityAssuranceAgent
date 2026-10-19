from __future__ import annotations

import json
from pathlib import Path

from src.brain.module import BrainModule
from src.common.run_control import (
    mark_recording_replay_dirty,
    take_recording_replay_dirty,
)
from src.common.runtime_context import SCRIPT_PATH_ENV
from src.common.script_helper import (
    invalidate_recording_replay_cache,
    load_recording_replay_steps,
    merge_unstarted_recording_steps,
)


def _write_recording(run_dir: Path, instructions: tuple[str, ...]) -> None:
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "screenshots").mkdir(exist_ok=True)
    events = []
    for index, instruction in enumerate(instructions):
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


def _brain_for_reload(run_dir: Path, monkeypatch) -> BrainModule:
    monkeypatch.setenv(SCRIPT_PATH_ENV, str(run_dir))
    monkeypatch.delenv("CUA_RUNTIME_COMMAND_MODE", raising=False)
    monkeypatch.delenv("CUA_SMART_MODE", raising=False)
    brain = BrainModule.__new__(BrainModule)
    brain.manager = type("Manager", (), {"log_info": lambda self, message: None})()
    brain._script_step_index = 1
    brain.script_lines = ["open", "click save"]
    brain.script_expected_outcomes = [None, None]
    brain.script_baseline_after_paths = [None, None]
    brain.script_settle_after_seconds = [None, None]
    brain.script_window_verifies = [{}, {}]
    brain._script_event_indices = [0, 1]
    return brain


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
    _write_recording(run_dir, ("open", "click save"))
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
    brain = _brain_for_reload(run_dir, monkeypatch)
    brain.manager = type("Manager", (), {"log_info": lambda self, message: logs.append(message)})()
    mark_recording_replay_dirty()
    invalidate_recording_replay_cache(run_dir)

    brain.reload_pending_recording_steps()

    assert brain._script_step_index == 1
    assert brain.script_lines == ["open", "click save now"]
    assert brain.script_expected_outcomes[1] == "dialog closed"
    assert brain._script_event_indices == [0, 1]
    assert any("Applied recording edits" in message for message in logs)


def test_reload_pending_skips_disk_when_clean(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, ("open", "click save"))
    brain = _brain_for_reload(run_dir, monkeypatch)
    mark_recording_replay_dirty()
    invalidate_recording_replay_cache(run_dir)
    brain.reload_pending_recording_steps()
    assert brain.script_lines == ["open", "click save"]

    (run_dir / "analysis" / "event_001.json").write_text(
        json.dumps({"instruction": "edited while clean", "use_expected_outcome": False}),
        encoding="utf-8",
    )
    invalidate_recording_replay_cache(run_dir)
    brain.reload_pending_recording_steps()
    assert brain.script_lines == ["open", "click save"]

    mark_recording_replay_dirty()
    brain.reload_pending_recording_steps()
    assert brain.script_lines == ["open", "edited while clean"]


def test_take_recording_replay_dirty_clears_on_read() -> None:
    mark_recording_replay_dirty()
    assert take_recording_replay_dirty() is True
    assert take_recording_replay_dirty() is False
    mark_recording_replay_dirty()
    assert take_recording_replay_dirty() is True


def test_load_recording_replay_steps_mtime_cache(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, ("open", "click save"))
    invalidate_recording_replay_cache()
    calls = {"n": 0}
    import src.common.script_helper as script_helper

    real = script_helper._load_instruction_event_rows_uncached

    def counting(path: Path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(script_helper, "_load_instruction_event_rows_uncached", counting)
    first = load_recording_replay_steps(run_dir)
    second = load_recording_replay_steps(run_dir)
    assert calls["n"] == 1
    assert first[0] == second[0] == ["open", "click save"]

    analysis = run_dir / "analysis" / "event_001.json"
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    payload["instruction"] = "click save now"
    analysis.write_text(json.dumps(payload), encoding="utf-8")
    # Bump mtime past fingerprint resolution on coarse filesystems.
    stamp = analysis.stat().st_mtime + 2.0
    import os

    os.utime(analysis, (stamp, stamp))
    third = load_recording_replay_steps(run_dir)
    assert calls["n"] == 2
    assert third[0] == ["open", "click save now"]


def test_jump_to_script_step_clamps_and_moves_the_cursor() -> None:
    brain = BrainModule.__new__(BrainModule)
    logs: list[str] = []
    brain.manager = type("Manager", (), {"log_info": lambda self, message: logs.append(message)})()
    brain.script_lines = ["one", "two", "three"]
    brain._script_step_index = 2

    brain.jump_to_script_step(-4)
    assert brain._script_step_index == 0

    brain.jump_to_script_step(99)
    assert brain._script_step_index == 2
    assert any("Replay jump" in message for message in logs)
