from __future__ import annotations

from pathlib import Path

from src.brain.module import BrainModule
from src.common.run_state import RunStateManager


def test_undo_last_runtime_step_removes_files_and_decrements_counter(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    paths = mgr.init_run("test", "test_run")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = mgr
    brain._step_transcript_counter = 2
    brain._script_step_index = 0

    steps_dir = paths.root / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    (steps_dir / "1_0.json").write_text("{}", encoding="utf-8")
    (steps_dir / "1_0.log").write_text("log\n", encoding="utf-8")

    assert brain.undo_last_runtime_step() is True
    assert brain._step_transcript_counter == 1
    assert brain._script_step_index == 0
    assert not (steps_dir / "1_0.json").exists()
    assert not (steps_dir / "1_0.log").exists()


def test_undo_last_runtime_step_returns_false_when_empty(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    mgr.init_run("test", "test_run")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = mgr
    brain._step_transcript_counter = 0
    brain._script_step_index = 0

    assert brain.undo_last_runtime_step() is False
