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


def test_click_reads_root_window_before_the_button_goes_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def _fake_click(**_kwargs: object) -> None:
        order.append("click")

    def _point_window(_x: int, _y: int) -> dict[str, str]:
        order.append("point")
        return {
            "class_name": "Windows.UI.Core.CoreWindow",
            "title": "搜尋",
            "process_name": "SearchHost.exe",
        }

    monkeypatch.setattr(hand_tools.pyautogui, "click", _fake_click)
    monkeypatch.setattr(
        hand_tools.pyautogui,
        "position",
        lambda: type("P", (), {"x": 704, "y": 286})(),
    )
    monkeypatch.setattr(hand_tools, "_press_time_point_window", _point_window)

    result = hand_tools.click()

    assert order == ["point", "click"]
    assert result["point_window"]["title"] == "搜尋"
    assert result["x"] == 704
    assert result["y"] == 286


def test_mcp_click_keeps_point_window_when_attaching_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp import tools as mcp_tools

    monkeypatch.setattr(
        mcp_tools,
        "_click",
        lambda **_kwargs: {
            "x": 641,
            "y": 1056,
            "button": "left",
            "clicks": 1,
            "interval": 0.0,
            "modifiers": [],
            "point_window": {
                "class_name": "Shell_TrayWnd",
                "title": "",
                "process_name": "explorer.exe",
            },
        },
    )

    result = mcp_tools.click(instruction="點擊搜尋")

    assert result["instruction"] == "點擊搜尋"
    assert result["point_window"]["class_name"] == "Shell_TrayWnd"
    assert result["x"] == 641


def test_merged_tool_args_keeps_point_window_from_fastmcp_text() -> None:
    from mcp.types import TextContent

    from src.hand.module import _merged_tool_args

    tool_output = [
        TextContent(
            type="text",
            text=(
                '{"x": 641, "y": 1056, "button": "left", "clicks": 1, '
                '"point_window": {"class_name": "Shell_TrayWnd", '
                '"title": "", "process_name": "explorer.exe"}, '
                '"instruction": "點擊搜尋"}'
            ),
        )
    ]
    merged = _merged_tool_args(
        {"button": "left", "clicks": 1, "instruction": "點擊搜尋"},
        tool_output,
    )
    assert merged["point_window"]["class_name"] == "Shell_TrayWnd"
    assert merged["x"] == 641
