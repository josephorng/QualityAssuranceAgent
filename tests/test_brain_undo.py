from __future__ import annotations

import json
from pathlib import Path

from src.brain.module import BrainModule, stamp_message
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


def test_resume_step_transcript_counter_continues_after_script_steps(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    paths = mgr.init_run("test", "test_run")
    steps_dir = paths.root / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    (steps_dir / "0_0.json").write_text("{}", encoding="utf-8")
    (steps_dir / "1_1.json").write_text("{}", encoding="utf-8")
    (steps_dir / "2_2.json").write_text("{}", encoding="utf-8")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = mgr

    assert brain._resume_step_transcript_counter() == 3


def test_append_failed_tool_call_records_tool_name(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    paths = mgr.init_run("test", "test_run")
    steps_dir = paths.root / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    step_path = steps_dir / "1_1.json"
    step_path.write_text(json.dumps({"messages": []}), encoding="utf-8")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = mgr

    brain._append_failed_tool_call("hotkey", 1, 1)
    brain._append_failed_tool_call("hotkey", 1, 1)

    payload = json.loads(step_path.read_text(encoding="utf-8"))
    assert payload["failed_tool_calls"] == ["hotkey", "hotkey"]


def test_stamp_message_adds_timestamp_utc_once() -> None:
    first = stamp_message({"role": "user", "content": "hello"})
    second = stamp_message(first)

    assert "timestamp_utc" in first
    assert second["timestamp_utc"] == first["timestamp_utc"]


def test_append_step_messages_stamps_each_message(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    paths = mgr.init_run("test", "test_run")
    steps_dir = paths.root / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    brain = BrainModule.__new__(BrainModule)
    brain.manager = mgr

    brain._append_step_messages(
        [{"role": "assistant", "content": "done"}],
        transcript_counter=0,
        script_step_index=0,
    )

    payload = json.loads((steps_dir / "0_0.json").read_text(encoding="utf-8"))
    assert "timestamp_utc" in payload["messages"][0]


def test_undo_last_runtime_step_returns_false_when_empty(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    mgr.init_run("test", "test_run")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = mgr
    brain._step_transcript_counter = 0
    brain._script_step_index = 0

    assert brain.undo_last_runtime_step() is False


def test_parse_step_outcome_completed() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = RunStateManager.__new__(RunStateManager)

    outcome = brain._parse_step_outcome(
        '{"status":"completed","reason":"clicked the button"}'
    )
    assert outcome is not None
    assert outcome.status == "completed"
    assert outcome.reason == "clicked the button"


def test_parse_step_outcome_failed_with_fence() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = RunStateManager.__new__(RunStateManager)

    outcome = brain._parse_step_outcome(
        '```json\n{"status":"failed","reason":"target not on screen"}\n```'
    )
    assert outcome is not None
    assert outcome.status == "failed"
    assert "not on screen" in outcome.reason


def test_parse_step_outcome_invalid_returns_none() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = type("Mgr", (), {"log_error": lambda self, msg: None})()

    assert brain._parse_step_outcome("The task is completed because...") is None
    assert brain._parse_step_outcome("") is None
    assert brain._parse_step_outcome('{"status":"maybe","reason":"x"}') is None


def test_parse_step_outcome_recovers_unescaped_quotes_in_reason() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = type(
        "Mgr",
        (),
        {
            "log_error": lambda self, msg: None,
            "log_info": lambda self, msg: None,
        },
    )()

    # Same failure mode as task_20260805_032319_689937 step 8: nested "click" breaks json.loads.
    raw = (
        '{"status":"completed","reason":"The user\'s instruction was to "click", '
        'and the click action was executed successfully at the current cursor position. '
        'Since no specific target was provided, the single click satisfies the goal."}'
    )
    outcome = brain._parse_step_outcome(raw)
    assert outcome is not None
    assert outcome.status == "completed"
    assert "click" in outcome.reason
    assert "satisfies the goal" in outcome.reason


def test_is_pseudo_end_tool_name() -> None:
    assert BrainModule._is_pseudo_end_tool_name("finish") is True
    assert BrainModule._is_pseudo_end_tool_name("Finish") is True
    assert BrainModule._is_pseudo_end_tool_name("done") is True
    assert BrainModule._is_pseudo_end_tool_name("click") is False
    assert BrainModule._is_pseudo_end_tool_name(None) is False


def test_parse_step_outcome_from_arguments() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = type("Mgr", (), {"log_error": lambda self, msg: None})()

    outcome = brain._parse_step_outcome_from_arguments(
        {"status": "completed", "reason": "clicked successfully"}
    )
    assert outcome is not None
    assert outcome.status == "completed"
    assert outcome.reason == "clicked successfully"
    assert brain._parse_step_outcome_from_arguments({"reason": "no status"}) is None
    assert brain._parse_step_outcome_from_arguments(None) is None
