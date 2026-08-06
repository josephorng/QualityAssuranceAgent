from __future__ import annotations

import pytest

from cua_mcp import hand_tools


def test_type_text_restores_previous_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = {"value": "previously copied"}

    def fake_paste() -> str:
        return clipboard["value"]

    def fake_copy(text: str) -> None:
        clipboard["value"] = text

    hotkeys: list[tuple[str, str]] = []

    def fake_hotkey(*keys: str) -> None:
        hotkeys.append(tuple(keys))

    monkeypatch.setattr(hand_tools, "pyperclip", type("P", (), {"paste": staticmethod(fake_paste), "copy": staticmethod(fake_copy)})())
    monkeypatch.setattr(hand_tools.pyautogui, "hotkey", fake_hotkey)
    monkeypatch.setattr(hand_tools, "sleep", lambda _s: None)

    result = hand_tools.type_text("typed value")

    assert hotkeys == [("ctrl", "v")]
    assert clipboard["value"] == "previously copied"
    assert result["clipboard_restored"] is True
    assert result["effective_mode"] == "paste"


def test_type_text_still_pastes_when_clipboard_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = {"value": "untouched"}

    def boom_paste() -> str:
        raise RuntimeError("clipboard locked")

    def fake_copy(text: str) -> None:
        clipboard["value"] = text

    monkeypatch.setattr(
        hand_tools,
        "pyperclip",
        type("P", (), {"paste": staticmethod(boom_paste), "copy": staticmethod(fake_copy)})(),
    )
    monkeypatch.setattr(hand_tools.pyautogui, "hotkey", lambda *keys: None)
    monkeypatch.setattr(hand_tools, "sleep", lambda _s: None)

    result = hand_tools.type_text("typed value")

    assert clipboard["value"] == "typed value"
    assert result["clipboard_restored"] is False
