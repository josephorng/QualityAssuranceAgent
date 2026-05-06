from __future__ import annotations

from typing import Any

import pytest

from cua_mcp import tools as cua_tools

# Mirrors tool_calls in a windows_chrome_enter run step (e.g. steps/0_0.json).
WINDOWS_CHROME_ENTER_TOOL_CALLS: list[dict[str, Any]] = [
    {
        "function": {
            "name": "press_key",
            "arguments": {
                "key": "win",
                "instruction": "Press the Windows key to open the Start Menu search bar.",
            },
        }
    },
    {
        "function": {
            "name": "type_text",
            "arguments": {
                "instruction": 'Type "Chrome" into the search bar.',
                "text": "Chrome",
            },
        }
    },
    {
        "function": {
            "name": "press_key",
            "arguments": {
                "key": "Enter",
                "instruction": "Press Enter to launch Google Chrome.",
            },
        }
    },
]


def _dispatch_tool_call(spec: dict[str, Any]) -> Any:
    fn_spec = spec["function"]
    name = fn_spec["name"]
    arguments = dict(fn_spec["arguments"])
    tool_fn = getattr(cua_tools, name)
    return tool_fn(**arguments)


def test_windows_chrome_enter_sequence_invokes_hand_tools_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs press_key(win) → type_text(Chrome) → press_key(Enter) without driving the real desktop."""
    calls: list[tuple[str, Any]] = []

    def fake_hotkey(keys: list[str] | str) -> dict[str, Any]:
        calls.append(("hotkey", keys))
        if isinstance(keys, str):
            normalized = [keys]
        else:
            normalized = list(keys)
        return {"keys": normalized}

    def fake_type_text(text: str) -> dict[str, Any]:
        calls.append(("type_text", text))
        return {"text": text, "effective_mode": "paste"}

    monkeypatch.setattr("cua_mcp.hand_tools.hotkey", fake_hotkey)
    monkeypatch.setattr("cua_mcp.hand_tools.type_text", fake_type_text)

    for spec in WINDOWS_CHROME_ENTER_TOOL_CALLS:
        _dispatch_tool_call(spec)

    assert calls == [
        ("hotkey", "win"),
        ("type_text", "Chrome"),
        ("hotkey", "Enter"),
    ]
