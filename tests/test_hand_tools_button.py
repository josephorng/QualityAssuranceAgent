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


def test_move_refuses_failsafe_corner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hand_tools, "_primary_screen_size", lambda: (1920, 1080))
    moved: list[tuple[int, int]] = []

    def _fake_moveTo(*, x: int, y: int, duration: float = 0.0) -> None:
        moved.append((int(x), int(y)))

    monkeypatch.setattr(hand_tools.pyautogui, "moveTo", _fake_moveTo)

    with pytest.raises(ValueError, match="fail-safe corner"):
        hand_tools.move(1919, 1079)
    assert moved == []

    result = hand_tools.move(100, 200)
    assert result == {"x": 100, "y": 200, "duration": 0.0}
    assert moved == [(100, 200)]


def test_move_recovers_from_failsafe_when_cursor_already_in_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hand_tools, "_primary_screen_size", lambda: (1920, 1080))
    moves: list[tuple[int, int]] = []
    cursor = {"x": 1919, "y": 1079}

    def _fake_moveTo(*, x: int, y: int, duration: float = 0.0) -> None:
        if hand_tools.pyautogui.FAILSAFE and hand_tools._is_pyautogui_failsafe_corner(
            cursor["x"], cursor["y"]
        ):
            raise hand_tools.pyautogui.FailSafeException(
                "PyAutoGUI fail-safe triggered from mouse moving to a corner of the screen."
            )
        cursor["x"], cursor["y"] = int(x), int(y)
        moves.append((int(x), int(y)))

    monkeypatch.setattr(hand_tools.pyautogui, "moveTo", _fake_moveTo)
    monkeypatch.setattr(
        hand_tools.pyautogui,
        "position",
        lambda: type("P", (), {"x": cursor["x"], "y": cursor["y"]})(),
    )
    monkeypatch.setattr(hand_tools.pyautogui, "FAILSAFE", True)

    result = hand_tools.move(500, 400)
    assert result["x"] == 500
    assert result["y"] == 400
    # Nudge off corner, then the intended move.
    assert moves == [(1918, 1078), (500, 400)]


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
