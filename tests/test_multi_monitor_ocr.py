from __future__ import annotations

import pytest

from src.common import monitor_prompt


def test_selected_eye_monitor_indices_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYE_MONITOR_INDICES", "1,2")
    monkeypatch.setenv("EYE_MONITOR_INDEX", "1")
    assert monitor_prompt.selected_eye_monitor_indices() == [1, 2]


def test_selected_eye_monitor_indices_single(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYE_MONITOR_INDICES", raising=False)
    monkeypatch.setenv("EYE_MONITOR_INDEX", "3")
    assert monitor_prompt.selected_eye_monitor_indices() == [3]
