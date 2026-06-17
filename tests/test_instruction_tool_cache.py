from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from ollama import Message

from src.brain.module import BrainModule
from src.common.instruction_tool_cache import (
    extract_tool_calls_from_messages,
    load_cache,
    lookup_tool_calls,
    normalize_instruction,
    upsert_tool_calls,
)
from src.common.models import ExecutionResult
from src.common.runtime_context import USE_TOOL_CACHE_ENV


STEP_0_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "decide"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "minimize_windows",
                    "arguments": {
                        "instruction": "將所有視窗最小化。",
                        "window_title_contains": "all",
                    },
                }
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"ok": true, "action": "minimize_windows"}',
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "type_text",
                    "arguments": {"instruction": "type hello", "text": "hello"},
                }
            }
        ],
    },
]


def test_normalize_instruction_strips_whitespace() -> None:
    assert normalize_instruction("  hello  ") == "hello"


def test_extract_tool_calls_from_messages_preserves_order() -> None:
    calls = extract_tool_calls_from_messages(STEP_0_MESSAGES)
    assert calls == [
        {
            "name": "minimize_windows",
            "arguments": {
                "instruction": "將所有視窗最小化。",
                "window_title_contains": "all",
            },
        },
        {
            "name": "type_text",
            "arguments": {"instruction": "type hello", "text": "hello"},
        },
    ]


def test_lookup_and_upsert_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "instruction_tool_cache.json"
    instruction = "將所有視窗最小化。"
    tool_calls = [
        {
            "name": "minimize_windows",
            "arguments": {"instruction": instruction, "window_title_contains": "all"},
        }
    ]

    assert lookup_tool_calls(instruction, cache_path) is None
    upsert_tool_calls(instruction, tool_calls, source_run_id="run_a", path=cache_path)

    loaded = lookup_tool_calls(instruction, cache_path)
    assert loaded == tool_calls

    cache = load_cache(cache_path)
    entry = cache["entries"][instruction]
    assert entry["source_run_id"] == "run_a"
    assert entry["instruction"] == instruction


def test_upsert_overwrites_existing_entry(tmp_path: Path) -> None:
    cache_path = tmp_path / "instruction_tool_cache.json"
    instruction = "open site"
    upsert_tool_calls(
        instruction,
        [{"name": "open_website", "arguments": {"url": "https://a.test"}}],
        source_run_id="run_old",
        path=cache_path,
    )
    upsert_tool_calls(
        instruction,
        [{"name": "open_website", "arguments": {"url": "https://b.test"}}],
        source_run_id="run_new",
        path=cache_path,
    )

    loaded = lookup_tool_calls(instruction, cache_path)
    assert loaded == [{"name": "open_website", "arguments": {"url": "https://b.test"}}]
    cache = load_cache(cache_path)
    assert cache["entries"][instruction]["source_run_id"] == "run_new"


@pytest.mark.asyncio
async def test_loop_uses_cache_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USE_TOOL_CACHE_ENV, "1")
    cached = [{"name": "minimize_windows", "arguments": {"instruction": "goal"}}]

    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain._step_transcript_counter = 0
    brain._script_step_index = 0
    brain.run_id = "test_run"
    brain._hand = MagicMock()
    brain._eye = MagicMock()
    brain._eye.capture_separated_images = AsyncMock(return_value=["shot.png"])
    brain._normalize_tool_name = AsyncMock(return_value="minimize_windows")
    brain._hand.execute_tool_command = AsyncMock(
        return_value=ExecutionResult(
            ok=True,
            action="minimize_windows",
            args={"instruction": "goal"},
            message="executed",
        )
    )
    brain.sanitize_execution_result = BrainModule.sanitize_execution_result.__get__(brain, BrainModule)
    brain._append_failed_tool_call = MagicMock()
    brain._save_step_messages = MagicMock()
    brain._current_goal = MagicMock(return_value="goal")

    monkeypatch.setattr("src.brain.module.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "src.brain.module.lookup_tool_calls",
        lambda instruction: cached if instruction == "goal" else None,
    )
    chat_messages = AsyncMock()
    monkeypatch.setattr("src.brain.module.get_llm_client", lambda: MagicMock(chat_messages=chat_messages))

    assert await brain.loop() is True
    chat_messages.assert_not_called()
    brain._hand.execute_tool_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_loop_falls_back_to_llm_when_cache_replay_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USE_TOOL_CACHE_ENV, "1")
    cached = [{"name": "minimize_windows", "arguments": {"instruction": "goal"}}]

    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain._step_transcript_counter = 0
    brain._script_step_index = 0
    brain.run_id = "test_run"
    brain.settings = MagicMock()
    brain.settings.brain_lm = "test-model"
    brain._hand = MagicMock()
    brain._eye = MagicMock()
    brain._eye.capture_separated_images = AsyncMock(return_value=["shot.png"])
    brain._normalize_tool_name = AsyncMock(return_value="minimize_windows")
    brain._hand.execute_tool_command = AsyncMock(
        return_value=ExecutionResult(
            ok=False,
            action="minimize_windows",
            args={"instruction": "goal"},
            message="failed",
        )
    )
    brain.sanitize_execution_result = BrainModule.sanitize_execution_result.__get__(brain, BrainModule)
    brain.sanitize_message = BrainModule.sanitize_message.__get__(brain, BrainModule)
    brain._append_failed_tool_call = MagicMock()
    brain._save_step_messages = MagicMock()
    brain._current_goal = MagicMock(return_value="goal")

    monkeypatch.setattr("src.brain.module.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "src.brain.module.lookup_tool_calls",
        lambda instruction: cached if instruction == "goal" else None,
    )

    response = Message(role="assistant", content="done")
    chat_messages = AsyncMock(return_value=response)
    brain.ollama = MagicMock(chat_messages=chat_messages)
    monkeypatch.setattr("src.brain.module.get_prompt", lambda name: "prompt {task}")
    monkeypatch.setattr("src.brain.module.upsert_tool_calls", MagicMock())

    assert await brain.loop() is True
    chat_messages.assert_awaited()
