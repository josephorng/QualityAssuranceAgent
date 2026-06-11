from __future__ import annotations

import pytest

from cua_mcp import hand_tools


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["win", "e"], ["win", "e"]),
        ('["win", "e"]', ["win", "e"]),
        ("[win,e]", ["win", "e"]),
        ("[win, e]", ["win", "e"]),
        ("['win', 'e']", ["win", "e"]),
        ("win+e", ["win", "e"]),
        ("win,e", ["win", "e"]),
        ("windows+e", ["win", "e"]),
        ("ctrl+shift+s", ["ctrl", "shift", "s"]),
        (["ctrl", "a"], ["ctrl", "a"]),
    ],
)
def test_hotkey_accepts_common_model_formats(
    monkeypatch: pytest.MonkeyPatch,
    keys: list[str] | str,
    expected: list[str],
) -> None:
    captured: list[str] = []

    def _fake_hotkey(*args: str) -> None:
        captured.extend(args)

    monkeypatch.setattr(hand_tools.pyautogui, "hotkey", _fake_hotkey)

    result = hand_tools.hotkey(keys)

    assert captured == expected
    assert result["keys"] == expected


def test_parse_hotkey_keys_bracket_fallback() -> None:
    assert hand_tools._parse_hotkey_keys("[win,e]") == ["win", "e"]
    assert hand_tools._parse_hotkey_keys('["win", "e"]') == ["win", "e"]


def test_hotkey_invalid_keys_includes_format_hint() -> None:
    with pytest.raises(ValueError, match=r'Use a JSON array like \["win", "e"\]'):
        hand_tools.hotkey("[notakey, foo]")
