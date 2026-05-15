"""Tests for monitor list helper and unique run folder naming."""

from __future__ import annotations

from src.common.monitor_prompt import EyeMonitorChoice, list_eye_monitor_choices
from src.common.run_state import unique_run_folder_name


def test_list_eye_monitor_choices_no_physical(monkeypatch) -> None:
    monkeypatch.setattr("src.common.monitor_prompt._physical_monitors", lambda: [])
    rows = list_eye_monitor_choices()
    assert rows == []


def test_list_eye_monitor_choices_with_physical(monkeypatch) -> None:
    physical = [
        {"index": 1, "left": 0, "top": 0, "width": 800, "height": 600},
        {"index": 2, "left": 800, "top": 0, "width": 800, "height": 600},
    ]
    monkeypatch.setattr("src.common.monitor_prompt._physical_monitors", lambda: physical)
    rows = list_eye_monitor_choices()
    assert len(rows) == 2
    assert rows[0].index == 1
    assert rows[1].index == 2
    assert isinstance(rows[0], EyeMonitorChoice)


def test_unique_run_folder_name(monkeypatch) -> None:
    monkeypatch.setattr("src.common.run_state.ts_name", lambda: "fixed_ts")
    assert unique_run_folder_name("Hello!! World") == "hello_world_fixed_ts"
