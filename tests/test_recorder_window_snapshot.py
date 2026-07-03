from __future__ import annotations

from unittest.mock import patch

import pytest

from src.recorder.window_snapshot import (
    WindowInfo,
    WindowStateChange,
    diff_snapshots,
    instruction_for_window_change,
    settle_delay_for_click,
)


@pytest.fixture(autouse=True)
def _no_live_window_at_point():
    with patch("src.recorder.window_snapshot.window_at_point", return_value=None):
        yield


def _win(
    hwnd: int,
    title: str,
    *,
    left: int = 0,
    top: int = 0,
    width: int = 800,
    height: int = 600,
    is_minimized: bool = False,
    is_maximized: bool = False,
    pid: int | None = 1,
) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        pid=pid,
        left=left,
        top=top,
        width=width,
        height=height,
        is_minimized=is_minimized,
        is_maximized=is_maximized,
    )


def test_diff_detects_minimize_at_click_target() -> None:
    before = [_win(100, "Google Chrome", left=100, top=100, width=800, height=600)]
    after = [
        _win(
            100,
            "Google Chrome",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=True,
        )
    ]
    change = diff_snapshots(before, after, click_xy=(400, 120))
    assert change == WindowStateChange(action="minimize", title="Google Chrome", confidence="high")


def test_diff_detects_close_when_window_missing() -> None:
    before = [_win(200, "Notepad", left=50, top=50, width=600, height=400)]
    after: list[WindowInfo] = []
    change = diff_snapshots(before, after, click_xy=(300, 70))
    assert change == WindowStateChange(action="close", title="Notepad", confidence="high")


def test_diff_detects_maximize_by_flag() -> None:
    before = [_win(300, "Excel", left=100, top=100, width=600, height=400)]
    after = [_win(300, "Excel", left=0, top=0, width=1920, height=1040, is_maximized=True)]
    change = diff_snapshots(before, after, click_xy=(150, 110))
    assert change == WindowStateChange(action="maximize", title="Excel", confidence="high")


def test_diff_detects_maximize_by_area_growth() -> None:
    before = [_win(400, "Word", left=200, top=200, width=500, height=400)]
    after = [_win(400, "Word", left=0, top=0, width=1800, height=1000)]
    change = diff_snapshots(before, after, click_xy=(250, 210))
    assert change == WindowStateChange(action="maximize", title="Word", confidence="high")


def test_diff_matches_by_pid_and_title_when_hwnd_changes() -> None:
    before = [_win(500, "Slack", left=10, top=10, width=700, height=500, pid=42)]
    after = [
        _win(
            999,
            "Slack",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=True,
            pid=42,
        )
    ]
    change = diff_snapshots(before, after, click_xy=(100, 20))
    assert change == WindowStateChange(action="minimize", title="Slack", confidence="high")


def test_diff_medium_confidence_single_removed_window() -> None:
    before = [
        _win(1, "App A", left=0, top=0, width=400, height=300),
        _win(2, "App B", left=500, top=0, width=400, height=300),
    ]
    after = [_win(1, "App A", left=0, top=0, width=400, height=300)]
    change = diff_snapshots(before, after, click_xy=(900, 900))
    assert change == WindowStateChange(action="close", title="App B", confidence="medium")


def test_diff_detects_minimize_from_iconic_rect_without_flag() -> None:
    before = [_win(100, "Google Chrome", left=400, top=100, width=900, height=700)]
    after = [
        _win(
            100,
            "Google Chrome",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=False,
        )
    ]
    change = diff_snapshots(before, after, click_xy=(500, 120))
    assert change == WindowStateChange(action="minimize", title="Google Chrome", confidence="medium")


def test_diff_title_bar_prefers_window_at_point_over_larger_window() -> None:
    vscode = _win(1, "VS Code", left=0, top=0, width=1920, height=1080)
    chrome = _win(2, "Google Chrome", left=400, top=100, width=900, height=700)
    before = [vscode, chrome]
    after = [
        _win(1, "VS Code", left=0, top=0, width=1920, height=1080),
        _win(
            2,
            "Google Chrome",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=True,
        ),
    ]
    with patch("src.recorder.window_snapshot.window_at_point", return_value=chrome):
        change = diff_snapshots(before, after, click_xy=(1531, 46))
    assert change == WindowStateChange(action="minimize", title="Google Chrome", confidence="high")


def test_diff_global_minimize_fallback_when_target_not_at_click() -> None:
    before = [
        _win(1, "VS Code", left=0, top=0, width=1920, height=1080),
        _win(2, "Google Chrome", left=400, top=100, width=900, height=700),
    ]
    after = [
        _win(1, "VS Code", left=0, top=0, width=1920, height=1080),
        _win(
            2,
            "Google Chrome",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=True,
        ),
    ]
    change = diff_snapshots(before, after, click_xy=(10, 10))
    assert change == WindowStateChange(action="minimize", title="Google Chrome", confidence="medium")


def test_instruction_for_high_confidence_window_change() -> None:
    change = WindowStateChange(action="minimize", title="Google Chrome", confidence="high")
    assert instruction_for_window_change(change) == "最小化「Google Chrome」視窗"
    assert instruction_for_window_change({"action": "close", "title": "Notepad", "confidence": "high"}) == "關閉「Notepad」視窗"
    assert instruction_for_window_change({"action": "close", "title": "Notepad", "confidence": "medium"}) == "關閉「Notepad」視窗"
    assert instruction_for_window_change({"action": "minimize", "title": "Chrome", "confidence": "medium"}) == "最小化「Chrome」視窗"
    assert instruction_for_window_change({"action": "opened", "title": "X", "confidence": "medium"}) is None


def test_settle_delay_is_longer_for_title_bar_clicks() -> None:
    assert settle_delay_for_click((100, 40)) > settle_delay_for_click((100, 200))
