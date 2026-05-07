"""Tests for runtime command mode env helpers."""

import pytest

from src.common.runtime_context import (
    RUNTIME_COMMAND_MODE_ENV,
    is_runtime_command_mode,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("YES", True),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
    ],
)
def test_is_runtime_command_mode(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)
    if value:
        monkeypatch.setenv(RUNTIME_COMMAND_MODE_ENV, value)
    assert is_runtime_command_mode() is expected
