from __future__ import annotations

from pathlib import Path

from src.common.instruction_tool_cache import load_cache, lookup_tool_calls
from src.recorder.to_cache import validate_tool_calls


def test_validate_tool_calls_rejects_unknown_tool() -> None:
    err = validate_tool_calls([{"name": "not_a_real_tool", "arguments": {}}])
    assert err is not None
    assert "invalid tool name" in err


def test_validate_tool_calls_requires_type_text_text() -> None:
    err = validate_tool_calls([{"name": "type_text", "arguments": {"instruction": "x"}}])
    assert err is not None
    assert "text" in err


def test_validate_tool_calls_accepts_move_mouse_and_click() -> None:
    calls = [
        {"name": "move_mouse", "arguments": {"instruction": "移到 Submit"}},
        {"name": "click", "arguments": {"button": "left", "instruction": "點擊"}},
    ]
    assert validate_tool_calls(calls) is None


def test_upsert_from_recorder_shape(tmp_path: Path) -> None:
    from src.common.instruction_tool_cache import upsert_tool_calls

    cache_path = tmp_path / "instruction_tool_cache.json"
    instruction = "按下 Enter 鍵。"
    tool_calls = [{"name": "press_key", "arguments": {"instruction": "按下 Enter", "key": "enter"}}]
    upsert_tool_calls(instruction, tool_calls, source_run_id="screen_record_test", path=cache_path)
    assert lookup_tool_calls(instruction, path=cache_path) == tool_calls
    cache = load_cache(cache_path)
    entry = cache["entries"][instruction]
    assert entry["source_run_id"] == "screen_record_test"
