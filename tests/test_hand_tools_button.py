from __future__ import annotations

import pytest

from cua_mcp import hand_tools


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("left", "left"),
        ("RIGHT", "right"),
        ('<|"|>left<|"|>', "left"),
        ("'middle'", "middle"),
        (1, 1),
        ("2", 2),
        ('<|"|>3<|"|>', 3),
    ],
)
def test_normalize_button(raw: str | int, expected: str | int) -> None:
    assert hand_tools._normalize_button(raw) == expected


def test_click_strips_model_button_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_click(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(hand_tools.pyautogui, "click", _fake_click)
    monkeypatch.setattr(
        hand_tools.pyautogui,
        "position",
        lambda: type("P", (), {"x": 10, "y": 20})(),
    )

    result = hand_tools.click(button='<|"|>left<|"|>', clicks=1)

    assert captured["button"] == "left"
    assert result["button"] == "left"
