from __future__ import annotations

from unittest.mock import patch

import pytest

from src.recorder.window_snapshot import (
    WindowInfo,
    WindowStateChange,
    click_hits_caption_buttons,
    diff_snapshots,
    diff_snapshots_with_debug,
    expected_outcome_for_window_change,
    format_window_change_hint,
    instruction_for_window_change,
    is_agent_app_restore,
    resolve_window_change,
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
    caption_button_bounds: tuple[int, int, int, int] | None = None,
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
        caption_button_bounds=caption_button_bounds,
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
    assert change == WindowStateChange(
        action="close",
        title="Notepad",
        confidence="high",
        from_title_bar_close=False,
    )


def test_diff_close_marks_title_bar_caption_hit() -> None:
    before = [
        _win(
            200,
            "Notepad",
            left=50,
            top=50,
            width=600,
            height=400,
            caption_button_bounds=(512, 50, 650, 82),
        )
    ]
    after: list[WindowInfo] = []
    change = diff_snapshots(before, after, click_xy=(630, 60))
    assert change == WindowStateChange(
        action="close",
        title="Notepad",
        confidence="high",
        from_title_bar_close=True,
    )


def test_diff_close_save_button_is_not_title_bar_close() -> None:
    before = [
        _win(
            200,
            "另存新檔",
            left=100,
            top=200,
            width=500,
            height=300,
            caption_button_bounds=(462, 200, 600, 232),
        )
    ]
    after: list[WindowInfo] = []
    # Center/bottom dialog button area (儲存), not caption X
    change = diff_snapshots(before, after, click_xy=(280, 420))
    assert change == WindowStateChange(
        action="close",
        title="另存新檔",
        confidence="high",
        from_title_bar_close=False,
    )


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
    result = diff_snapshots_with_debug(before, after, click_xy=(900, 900))
    assert result.change == WindowStateChange(
        action="close",
        title="App B",
        confidence="medium",
        from_title_bar_close=False,
    )
    assert result.debug["detection_path"] == "identity_close"
    assert [w["title"] for w in result.debug["windows_before"]] == ["App A", "App B"]
    assert [w["title"] for w in result.debug["windows_after"]] == ["App A"]


def test_diff_ignores_title_flicker_when_hwnd_still_present() -> None:
    """Explorer-style navigation can clear/change the title while hwnd stays alive."""
    before = [
        _win(1, "VS Code", left=0, top=0, width=400, height=300),
        _win(7934096, "常用 - 檔案總管", left=0, top=0, width=1920, height=1040, pid=42),
    ]
    after = [
        _win(1, "VS Code", left=0, top=0, width=400, height=300),
        # Same hwnd/pid; title briefly empty or changed during navigation.
        _win(7934096, "", left=0, top=0, width=1920, height=1040, pid=42),
    ]
    result = diff_snapshots_with_debug(before, after, click_xy=(72, 307))
    assert result.change is None
    assert result.debug["detection_path"] is None
    assert result.debug["windows_before_count"] == 2
    assert result.debug["windows_after_count"] == 2
    assert any(w["hwnd"] == 7934096 and w["title"] == "常用 - 檔案總管" for w in result.debug["windows_before"])
    assert any(w["hwnd"] == 7934096 and w["title"] == "" for w in result.debug["windows_after"])


def test_diff_ignores_title_rename_when_hwnd_still_present() -> None:
    before = [_win(100, "常用 - 檔案總管", left=0, top=0, width=800, height=600, pid=7)]
    after = [_win(100, "文件 - 檔案總管", left=0, top=0, width=800, height=600, pid=7)]
    change = diff_snapshots(before, after, click_xy=(72, 307))
    assert change is None


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


def test_diff_global_restore_from_taskbar_click() -> None:
    """Taskbar click hits the shell strip, not the app; restore is off-target."""
    taskbar = _win(65714, "", left=0, top=880, width=1918, height=40, pid=6324)
    before = [
        _win(1, "神網7", left=-8, top=-8, width=1934, height=896, is_maximized=True, pid=10424),
        _win(
            459206,
            "電腦使用代理",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=True,
            pid=9284,
        ),
        taskbar,
    ]
    after = [
        _win(459206, "電腦使用代理", left=156, top=156, width=976, height=719, pid=9284),
        _win(1, "神網7", left=-8, top=-8, width=1934, height=896, is_maximized=True, pid=10424),
        taskbar,
    ]
    result = diff_snapshots_with_debug(before, after, click_xy=(612, 894))
    assert result.change == WindowStateChange(
        action="restored", title="電腦使用代理", confidence="medium"
    )
    assert result.debug["detection_path"] == "global_restore"
    assert result.debug["target_hwnd"] == 65714


def test_diff_global_restore_from_iconic_rect_without_flag() -> None:
    taskbar = _win(10, "", left=0, top=880, width=1920, height=40)
    before = [
        _win(
            2,
            "Google Chrome",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=False,
        ),
        taskbar,
    ]
    after = [
        _win(2, "Google Chrome", left=100, top=80, width=900, height=700),
        taskbar,
    ]
    result = diff_snapshots_with_debug(before, after, click_xy=(100, 900))
    assert result.change == WindowStateChange(
        action="restored", title="Google Chrome", confidence="medium"
    )
    assert result.debug["detection_path"] == "global_restore"


def test_diff_global_restore_to_maximized() -> None:
    """Taskbar restore often returns a previously-maximized window as maximized."""
    taskbar = _win(10, "", left=0, top=880, width=1920, height=40)
    before = [
        _win(
            2,
            "Excel",
            left=-32000,
            top=-32000,
            width=160,
            height=28,
            is_minimized=True,
        ),
        taskbar,
    ]
    after = [
        _win(2, "Excel", left=-8, top=-8, width=1936, height=1056, is_maximized=True),
        taskbar,
    ]
    result = diff_snapshots_with_debug(before, after, click_xy=(200, 900))
    assert result.change == WindowStateChange(action="restored", title="Excel", confidence="medium")
    assert result.debug["detection_path"] == "global_restore"


def test_diff_global_restore_skips_when_ambiguous() -> None:
    taskbar = _win(10, "", left=0, top=880, width=1920, height=40)
    before = [
        _win(1, "App A", left=-32000, top=-32000, width=160, height=28, is_minimized=True),
        _win(2, "App B", left=-32000, top=-32000, width=160, height=28, is_minimized=True),
        taskbar,
    ]
    after = [
        _win(1, "App A", left=50, top=50, width=400, height=300),
        _win(2, "App B", left=500, top=50, width=400, height=300),
        taskbar,
    ]
    result = diff_snapshots_with_debug(before, after, click_xy=(100, 900))
    assert result.change is None
    assert result.debug["detection_path"] is None


def test_instruction_for_high_confidence_window_change() -> None:
    change = WindowStateChange(action="minimize", title="Google Chrome", confidence="high")
    assert instruction_for_window_change(change) == "最小化「Google Chrome」視窗"
    assert (
        instruction_for_window_change(
            {
                "action": "close",
                "title": "Notepad",
                "confidence": "high",
                "from_title_bar_close": True,
            }
        )
        == "關閉「Notepad」視窗"
    )
    assert (
        instruction_for_window_change(
            {
                "action": "close",
                "title": "Notepad",
                "confidence": "medium",
                "from_title_bar_close": True,
            }
        )
        == "關閉「Notepad」視窗"
    )
    assert (
        instruction_for_window_change(
            {
                "action": "close",
                "title": "Notepad",
                "confidence": "high",
                "from_title_bar_close": False,
            }
        )
        is None
    )
    assert instruction_for_window_change(
        {"action": "close", "title": "Notepad", "confidence": "high"}
    ) is None
    assert instruction_for_window_change({"action": "minimize", "title": "Chrome", "confidence": "medium"}) == "最小化「Chrome」視窗"
    assert instruction_for_window_change({"action": "restored", "title": "電腦使用代理", "confidence": "medium"}) == "還原「電腦使用代理」視窗"
    assert instruction_for_window_change({"action": "opened", "title": "X", "confidence": "medium"}) is None


def test_expected_outcome_for_window_change() -> None:
    assert (
        expected_outcome_for_window_change(
            {"action": "opened", "title": "常用 - 檔案總管", "confidence": "medium"}
        )
        == "「常用 - 檔案總管」視窗已開啟"
    )
    assert (
        expected_outcome_for_window_change(
            WindowStateChange(action="restored", title="檔案總管", confidence="high")
        )
        == "「檔案總管」視窗已顯示"
    )
    assert (
        expected_outcome_for_window_change(
            {"action": "maximize", "title": "常用 - 檔案總管", "confidence": "high"}
        )
        == "「常用 - 檔案總管」視窗已最大化並佔滿螢幕"
    )
    assert (
        expected_outcome_for_window_change(
            {"action": "close", "title": "下載 - 檔案總管", "confidence": "high"}
        )
        == "「下載 - 檔案總管」視窗已關閉"
    )
    assert expected_outcome_for_window_change(None) is None
    assert expected_outcome_for_window_change(
        {"action": "opened", "title": "電腦使用代理", "confidence": "medium"}
    ) is None
    assert expected_outcome_for_window_change(
        {"action": "restored", "title": "電腦使用代理", "confidence": "medium"}
    ) is None
    assert expected_outcome_for_window_change(
        {"action": "opened", "title": "X", "confidence": "low"}
    ) is None
    # Synthetic hwnd titles are not checkable at replay (taskbar shell strips).
    assert expected_outcome_for_window_change(
        {"action": "close", "title": "hwnd:65934", "confidence": "high"}
    ) is None


def test_diff_ignores_untitled_taskbar_strip_close_on_search_click() -> None:
    """Search/Start clicks can make an untitled taskbar hwnd vanish; not a close."""
    before = [
        _win(196998, "NVIDIA GeForce Overlay", left=0, top=0, width=1920, height=1080),
        _win(66018, "Program Manager", left=0, top=-1, width=3840, height=1081),
        _win(65934, "", left=0, top=1032, width=1920, height=48, pid=15728),
    ]
    after = [
        _win(196998, "NVIDIA GeForce Overlay", left=0, top=0, width=1920, height=1080),
        _win(66018, "Program Manager", left=0, top=-1, width=3840, height=1081),
    ]
    live = _win(65934, "", left=0, top=1032, width=1920, height=48, pid=15728)
    with patch("src.recorder.window_snapshot.window_at_point", return_value=live):
        result = diff_snapshots_with_debug(before, after, click_xy=(558, 1070))
    assert result.change is None
    assert expected_outcome_for_window_change(
        {"action": "close", "title": "hwnd:65934", "confidence": "high"}
    ) is None


def test_click_hits_caption_buttons_uses_stored_and_fallback_bounds() -> None:
    win = _win(
        1,
        "Dialog",
        left=100,
        top=200,
        width=400,
        height=300,
        caption_button_bounds=(362, 200, 500, 232),
    )
    assert click_hits_caption_buttons((400, 210), win)
    assert not click_hits_caption_buttons((200, 350), win)

    fallback = _win(2, "App", left=0, top=0, width=800, height=600)
    # Right-edge caption strip via geometry fallback
    assert click_hits_caption_buttons((780, 10), fallback)
    assert not click_hits_caption_buttons((100, 10), fallback)


def test_click_hits_caption_buttons_includes_bottom_right_edge() -> None:
    """Clicks on DWM rect edges must count (exclusive < bottom missed real X clicks)."""
    win = _win(
        1773306,
        "OANDA Lab - Google Chrome",
        left=1912,
        top=-9,
        width=1936,
        height=1048,
        is_maximized=True,
        caption_button_bounds=(3693, -9, 3839, 21),
    )
    assert click_hits_caption_buttons((3814, 21), win)  # y == bottom
    assert click_hits_caption_buttons((3839, 21), win)  # right+bottom corner
    assert not click_hits_caption_buttons((3814, 22), win)
    assert not click_hits_caption_buttons((3840, 21), win)


def test_format_hint_notes_non_caption_close() -> None:
    assert format_window_change_hint(
        {
            "action": "close",
            "title": "另存新檔",
            "confidence": "high",
            "from_title_bar_close": False,
        }
    ) == (
        "action=close (not title-bar X; prefer click label), "
        "title='另存新檔', confidence=high"
    )
    assert format_window_change_hint(
        {
            "action": "close",
            "title": "Notepad",
            "confidence": "high",
            "from_title_bar_close": True,
        }
    ) == "action=close, title='Notepad', confidence=high"


def test_is_agent_app_restore() -> None:
    assert is_agent_app_restore(
        {"action": "restored", "title": "電腦使用代理", "confidence": "medium"}
    )
    assert not is_agent_app_restore(
        {"action": "restored", "title": "Google Chrome", "confidence": "medium"}
    )
    assert not is_agent_app_restore(
        {"action": "minimize", "title": "電腦使用代理", "confidence": "high"}
    )
    assert not is_agent_app_restore(None)


def test_resolve_window_change_prefers_captured_then_rediffs_debug() -> None:
    captured = {"action": "minimize", "title": "Chrome", "confidence": "high"}
    assert resolve_window_change(captured, None, (1, 1)) == captured

    taskbar = {
        "hwnd": 10,
        "title": "",
        "pid": 1,
        "left": 0,
        "top": 880,
        "width": 1920,
        "height": 40,
        "is_minimized": False,
        "is_maximized": False,
    }
    debug = {
        "windows_before": [
            {
                "hwnd": 2,
                "title": "電腦使用代理",
                "pid": 9,
                "left": -32000,
                "top": -32000,
                "width": 160,
                "height": 28,
                "is_minimized": True,
                "is_maximized": False,
            },
            taskbar,
        ],
        "windows_after": [
            {
                "hwnd": 2,
                "title": "電腦使用代理",
                "pid": 9,
                "left": 100,
                "top": 100,
                "width": 800,
                "height": 600,
                "is_minimized": False,
                "is_maximized": False,
            },
            taskbar,
        ],
    }
    resolved = resolve_window_change(None, debug, (100, 900))
    assert resolved == {
        "action": "restored",
        "title": "電腦使用代理",
        "confidence": "medium",
    }


def test_instruction_ignores_shell_experience_host_window() -> None:
    assert (
        instruction_for_window_change(
            {"action": "close", "title": "快顯主機", "confidence": "medium"}
        )
        is None
    )
    assert (
        instruction_for_window_change(
            {"action": "close", "title": "快顯主機", "confidence": "high"}
        )
        is None
    )
    assert format_window_change_hint(
        {"action": "close", "title": "快顯主機", "confidence": "medium"}
    ) == "(none)"


def test_settle_delay_is_longer_for_title_bar_clicks() -> None:
    assert settle_delay_for_click((100, 40)) > settle_delay_for_click((100, 200))
    win = _win(1, "App", left=0, top=400, width=800, height=400)
    # Relative to window top (y=410), not absolute screen y<=80
    assert settle_delay_for_click((100, 410), [win]) > settle_delay_for_click((100, 600), [win])
