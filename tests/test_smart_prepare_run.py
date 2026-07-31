"""Tests for prepare_run_session smart mode contract."""

from __future__ import annotations

import os
from pathlib import Path

from main import prepare_run_session
from src.common.runtime_context import (
    RUNTIME_COMMAND_MODE_ENV,
    SMART_GOAL_ENV,
    SMART_MODE_ENV,
    is_smart_mode,
)


def test_prepare_run_session_smart_mode(tmp_path: Path, monkeypatch) -> None:
    from src.common.runtime_context import SCRIPT_PATH_ENV

    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)
    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    monkeypatch.delenv(SMART_GOAL_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_PATH_ENV, raising=False)

    goal_file = tmp_path / "open_notepad.txt"
    goal_file.write_text("Open notepad and type hello\n", encoding="utf-8")

    manager, paths, run_id = prepare_run_session(
        runs_root=tmp_path,
        task="Open notepad",
        runtime_mode=False,
        selected_script_path=goal_file,
        script_steps=None,
        eye_monitor_indices=[1],
        clear_runs_root=False,
        smart_mode=True,
        smart_goal="Open notepad and type hello",
    )
    assert paths.root.is_dir()
    assert run_id
    assert is_smart_mode() is True
    assert os.environ[SMART_GOAL_ENV] == "Open notepad and type hello"
    assert os.environ[SCRIPT_PATH_ENV] == str(goal_file)
    assert RUNTIME_COMMAND_MODE_ENV not in os.environ or os.environ.get(RUNTIME_COMMAND_MODE_ENV) != "1"
    manager.log_info("ok")
