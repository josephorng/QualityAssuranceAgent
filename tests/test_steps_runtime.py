from pathlib import Path

from cua_mcp import steps
from src.common import run_state
from src.common.run_state import RunStateManager


def _init_manager(tmp_path: Path) -> RunStateManager:
    manager = RunStateManager(tmp_path, memory_max_chars=100)
    manager.init_run("parent goal", "run")
    run_state._manager = manager
    return manager


def test_create_new_step_and_get_next_actionable(tmp_path: Path) -> None:
    _init_manager(tmp_path)
    created_first = steps.create_new_step(
        target_path="",
        new_step={"goal": "step 1", "instruction": "do one", "result": ""},
    )
    created_second = steps.create_new_step(
        target_path="",
        new_step={"goal": "step 2", "instruction": "do two", "result": ""},
    )
    assert created_first["created_path"] == "0"
    assert created_second["created_path"] == "1"
    assert steps.get_next_actionable_step_path() == "0"


def test_divide_step_and_pending_lookup(tmp_path: Path) -> None:
    _init_manager(tmp_path)
    steps.create_new_step(
        target_path="",
        new_step={"goal": "parent", "instruction": "split me", "result": ""},
    )
    split = steps.divide_step(
        path="0",
        new_steps=[
            {"goal": "child 1", "instruction": "do child 1", "result": ""},
            {"goal": "child 2", "instruction": "do child 2", "result": ""},
        ],
    )
    assert split["first_child_path"] == "0.0"
    assert steps.get_next_actionable_step_path() == "0.0"
    steps.mark_step_result("0.0", "Pending")
    assert steps.get_pending_step_path() == "0.0"


def test_check_task_complete_and_incomplete(tmp_path: Path) -> None:
    _init_manager(tmp_path)
    steps.create_new_step(
        target_path="",
        new_step={"goal": "a", "instruction": "a", "result": "Done"},
    )
    steps.create_new_step(
        target_path="",
        new_step={"goal": "b", "instruction": "b", "result": ""},
    )
    summary = steps.check_task()
    assert summary["complete"] is False
    assert summary["undone_count"] == 1

    steps.mark_step_result("1", "Done")
    summary_done = steps.check_task()
    assert summary_done["complete"] is True
    assert summary_done["pending_count"] == 0
    assert summary_done["undone_count"] == 0


def test_set_step_image(tmp_path: Path) -> None:
    _init_manager(tmp_path)
    steps.create_new_step(
        target_path="",
        new_step={"goal": "step", "instruction": "do step", "result": ""},
    )
    updated = steps.set_step_image("0", "20260424_000000_000000.png")
    assert updated["image"] == "20260424_000000_000000.png"
    assert steps.get_step("0")["image"] == "20260424_000000_000000.png"
