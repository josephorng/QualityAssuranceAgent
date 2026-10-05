from __future__ import annotations

from unittest.mock import patch

import pytest

from src.recorder.window_snapshot import (
    WindowInfo,
    WindowStateChange,
    build_window_verify_predicate,
    click_hits_caption_buttons,
    diff_snapshots,
    diff_snapshots_with_debug,
    expected_outcome_for_window_change,
    format_window_change_hint,
    instruction_for_window_change,
    is_agent_app_restore,
    resolve_window_change,
    settle_delay_for_click,
    filter_disabled_window_verify,
    normalize_window_verify_disabled,
    window_verify_condition_enabled,
    window_verify_satisfied,
    window_verify_from_debug,
    snapshot_top_level_windows,
    window_at_point as _window_at_point_impl,
)


@pytest.fixture(autouse=True)
def _no_live_window_at_point():
    with patch("src.recorder.window_snapshot.window_at_point", return_value=None):
        yield


def test_window_at_point_reads_hwnd_without_full_scan() -> None:
    """Press-time lookup must not call getAllWindows (that scan stalls the mouse hook)."""
    import builtins

    info = _win(9, "Excel")
    user32 = type("User32", (), {})()
    user32.WindowFromPoint = lambda pt: 7
    user32.GetAncestor = lambda hwnd, ga: 9
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pygetwindow":
            raise AssertionError("window_at_point enumerated every top-level window")
        return real_import(name, globals, locals, fromlist, level)

    with patch("src.recorder.window_snapshot.ctypes.windll.user32", user32), patch(
        "src.recorder.window_snapshot._window_info_from_hwnd",
        return_value=info,
    ) as from_hwnd, patch("builtins.__import__", side_effect=_import):
        got = _window_at_point_impl(3, 4)

    assert got == info
    from_hwnd.assert_called_once_with(9)


def test_snapshot_top_level_windows_skips_library_scan_and_caption_query() -> None:
    """Enumeration must not call getAllWindows or DWM once per window."""
    import builtins

    dwm_calls = {"n": 0}
    read_hwnds: list[int] = []

    def _enum(proc, lparam) -> bool:
        proc(11, lparam)
        proc(22, lparam)
        return True

    def _visible(hwnd: int) -> bool:
        return int(hwnd) != 22

    def _from_hwnd(hwnd: int) -> WindowInfo:
        read_hwnds.append(int(hwnd))
        return _win(int(hwnd), "App")

    def _dwm(*_args, **_kwargs):
        dwm_calls["n"] += 1
        return None

    user32 = type(
        "User32",
        (),
        {"EnumWindows": staticmethod(_enum), "IsWindowVisible": staticmethod(_visible)},
    )()
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pygetwindow":
            raise AssertionError("snapshot enumerated windows via pygetwindow")
        return real_import(name, globals, locals, fromlist, level)

    with patch("src.recorder.window_snapshot.ctypes.windll.user32", user32), patch(
        "src.recorder.window_snapshot._window_info_from_hwnd",
        side_effect=_from_hwnd,
    ), patch(
        "src.recorder.window_snapshot._dwm_caption_button_bounds_screen",
        side_effect=_dwm,
    ), patch("builtins.__import__", side_effect=_import):
        windows = snapshot_top_level_windows()

    assert [win.hwnd for win in windows] == [11]
    assert read_hwnds == [11]
    assert dwm_calls["n"] == 0
    assert windows[0].caption_button_bounds is None


def test_caption_bounds_query_dwm_only_when_missing() -> None:
    from src.recorder.window_snapshot import caption_button_bounds_for_window

    missing = _win(5, "App", left=0, top=0, width=800, height=600)
    with patch(
        "src.recorder.window_snapshot._hwnd_still_valid",
        return_value=True,
    ), patch(
        "src.recorder.window_snapshot._dwm_caption_button_bounds_screen",
        return_value=(700, 0, 800, 32),
    ) as dwm:
        assert caption_button_bounds_for_window(missing) == (700, 0, 800, 32)
    dwm.assert_called_once_with(5, 0, 0)

    stored = _win(5, "App", caption_button_bounds=(1, 2, 3, 4))
    with patch(
        "src.recorder.window_snapshot._dwm_caption_button_bounds_screen",
        side_effect=AssertionError("stored bounds should skip DWM"),
    ):
        assert caption_button_bounds_for_window(stored) == (1, 2, 3, 4)


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
    class_name: str = "",
    process_name: str | None = None,
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
        class_name=class_name,
        process_name=process_name,
    )


def test_attach_process_names_fills_executable_for_every_window(monkeypatch) -> None:
    from src.recorder.window_snapshot import _attach_process_names

    chrome = _win(2, "Google Chrome", class_name="Chrome_WidgetWin_1", pid=42)
    notepad = _win(3, "Untitled - Notepad", class_name="Notepad", pid=9)
    named = _win(
        4,
        "快顯主機",
        class_name="Popup",
        pid=7,
        process_name="explorer.exe",
    )
    monkeypatch.setattr(
        "src.recorder.verify_signals._process_names_by_pid",
        lambda: {42: "chrome.exe"},
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot._process_name_for_pid",
        lambda pid: "notepad.exe" if pid == 9 else None,
    )
    filled = _attach_process_names([chrome, notepad, named])
    assert [win.process_name for win in filled] == [
        "chrome.exe",
        "notepad.exe",
        "explorer.exe",
    ]
    predicate = build_window_verify_predicate([], [filled[0]])
    assert predicate["appeared"] == [
        {
            "class_name": "Chrome_WidgetWin_1",
            "title": "Google Chrome",
            "process_name": "chrome.exe",
        }
    ]


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


def test_diff_close_does_not_pair_vanished_window_with_minimized_sibling() -> None:
    """Closing one Explorer must not become a minimize of another with the same title.

    WindowFromPoint after the close hits the window that was behind it.
    """
    visible = _win(
        261229422,
        "圖片 - 檔案總管",
        left=-8,
        top=-8,
        width=1936,
        height=1048,
        is_maximized=True,
        pid=52892,
        class_name="CabinetWClass",
        caption_button_bounds=(1790, 0, 1928, 28),
    )
    already_minimized = _win(
        2557630,
        "圖片 - 檔案總管",
        left=-32000,
        top=-32000,
        width=160,
        height=28,
        is_minimized=True,
        pid=52892,
        class_name="CabinetWClass",
    )
    behind = _win(
        264204,
        "main.py - QualityAssuranceAgent - Cursor",
        left=-8,
        top=-8,
        width=1936,
        height=1048,
        is_maximized=True,
        pid=16048,
        class_name="Chrome_WidgetWin_1",
    )
    before = [visible, already_minimized, behind]
    after = [already_minimized, behind]
    click_xy = (1898, 14)
    with patch("src.recorder.window_snapshot.window_at_point", return_value=behind):
        result = diff_snapshots_with_debug(before, after, click_xy=click_xy)
    assert result.change == WindowStateChange(
        action="close",
        title="圖片 - 檔案總管",
        confidence="high",
        from_title_bar_close=True,
    )
    assert result.debug["detection_path"] == "closed_at_click"
    assert result.debug["target_hwnd"] == 261229422
    assert instruction_for_window_change(result.change) == "關閉「圖片 - 檔案總管」視窗"


def test_diff_sibling_close_when_click_misses_the_vanished_window() -> None:
    visible = _win(10, "圖片 - 檔案總管", left=0, top=0, width=800, height=600, pid=5)
    sibling = _win(
        11,
        "圖片 - 檔案總管",
        left=-32000,
        top=-32000,
        width=160,
        height=28,
        is_minimized=True,
        pid=5,
    )
    result = diff_snapshots_with_debug(
        [visible, sibling],
        [sibling],
        click_xy=(2000, 2000),
    )
    assert result.change == WindowStateChange(
        action="close",
        title="圖片 - 檔案總管",
        confidence="medium",
        from_title_bar_close=False,
    )
    assert result.debug["detection_path"] == "hwnd_close"


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
    """DWM caption rects are tight; edge and slightly-below clicks still count."""
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
    # Maximized Chrome: click can land a few px below DWM bottom (real X).
    assert click_hits_caption_buttons((3810, 23), win)
    assert not click_hits_caption_buttons((3814, 40), win)  # well below caption
    assert not click_hits_caption_buttons((3600, 21), win)  # left of caption strip


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


def test_reconcile_promotes_foreground_opened_when_list_lags() -> None:
    from src.recorder.window_snapshot import reconcile_window_change_with_signals

    debug = {
        "windows_before": [
            {
                "hwnd": 1,
                "title": "Git - 檔案總管",
                "class_name": "CabinetWClass",
                "process_name": "explorer.exe",
            }
        ],
        "windows_after": [
            {
                "hwnd": 1,
                "title": "Git - 檔案總管",
                "class_name": "CabinetWClass",
                "process_name": "explorer.exe",
            }
        ],
        "detection_path": None,
    }
    change, out = reconcile_window_change_with_signals(
        None,
        debug,
        {
            "foreground": {
                "class_name": "Windows.UI.Core.CoreWindow",
                "title": "搜尋",
                "process_name": "SearchHost.exe",
            }
        },
        {
            "foreground": {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
                "process_name": "explorer.exe",
            }
        },
    )
    assert change == {
        "action": "opened",
        "title": "常用 - 檔案總管",
        "confidence": "medium",
    }
    assert out["detection_path"] == "foreground_opened"


def test_reconcile_suppresses_opened_when_already_foreground() -> None:
    from src.recorder.window_snapshot import reconcile_window_change_with_signals

    opened = {
        "action": "opened",
        "title": "常用 - 檔案總管",
        "confidence": "medium",
    }
    change, out = reconcile_window_change_with_signals(
        opened,
        {"detection_path": "opened_at_click"},
        {
            "foreground": {
                "class_name": "CabinetWClass",
                "title": "檔案總管",
                "process_name": "explorer.exe",
            }
        },
        {
            "foreground": {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
                "process_name": "explorer.exe",
            }
        },
    )
    assert change is None
    assert out["detection_path"] == "opened_suppressed_already_foreground"
    assert out["suppressed_opened"] == opened


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


def test_flyout_dismiss_is_not_a_close_action_but_stays_in_verify() -> None:
    """Menu dismiss must not become window_change close; verify still sees disappeared."""
    explorer = _win(
        1,
        "常用 - 檔案總管",
        class_name="CabinetWClass",
        process_name="explorer.exe",
        left=0,
        top=0,
        width=1920,
        height=1040,
    )
    flyout = _win(
        9,
        "快顯主機",
        class_name="Microsoft.UI.Content.PopupWindowSiteBridge",
        process_name="explorer.exe",
        left=438,
        top=131,
        width=213,
        height=217,
        # Cap strip overlaps menu body; must not become from_title_bar_close.
        caption_button_bounds=(550, 131, 650, 160),
    )
    # Click on 「並排」 inside the flyout (would hit caption slack without flyout guard).
    result = diff_snapshots_with_debug(
        [explorer, flyout],
        [explorer],
        click_xy=(560, 150),
    )
    assert result.change is None

    predicate = build_window_verify_predicate([explorer, flyout], [explorer])
    assert predicate.get("disappeared") == [
        {
            "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
            "title": "快顯主機",
            "process_name": "explorer.exe",
        }
    ]
    assert expected_outcome_for_window_change(
        {
            "action": "close",
            "title": "快顯主機",
            "confidence": "high",
            "from_title_bar_close": True,
        }
    ) is None


def test_from_title_bar_close_never_true_for_flyout() -> None:
    from src.recorder.window_snapshot import _from_title_bar_close

    flyout = _win(
        9,
        "快顯主機",
        class_name="Microsoft.UI.Content.PopupWindowSiteBridge",
        left=100,
        top=100,
        width=200,
        height=200,
        caption_button_bounds=(150, 100, 300, 140),
    )
    assert _from_title_bar_close((200, 120), flyout) is False
    normal = _win(
        1,
        "Notepad",
        left=100,
        top=100,
        width=800,
        height=600,
        caption_button_bounds=(700, 100, 900, 140),
    )
    assert _from_title_bar_close((800, 120), normal) is True


def test_settle_delay_is_longer_for_title_bar_clicks() -> None:
    from src.recorder.window_snapshot import (
        WINDOW_SETTLE_DELAY_S,
        WINDOW_SETTLE_TITLE_BAR_DELAY_S,
    )

    assert WINDOW_SETTLE_DELAY_S == 0.25
    assert WINDOW_SETTLE_TITLE_BAR_DELAY_S == 0.45
    assert settle_delay_for_click((100, 40)) > settle_delay_for_click((100, 200))
    win = _win(1, "App", left=0, top=400, width=800, height=400)
    # Relative to window top (y=410), not absolute screen y<=80
    assert settle_delay_for_click((100, 410), [win]) > settle_delay_for_click((100, 600), [win])


def test_resolve_ocr_roi_local_keeps_rect_when_maximized() -> None:
    from src.recorder.window_snapshot import resolve_ocr_roi_local

    payload = {
        "hwnd": 1,
        "title": "App",
        "rect": [10, 10, 200, 150],
        "is_maximized": True,
    }
    with patch(
        "src.recorder.window_snapshot.find_matching_click_window",
        return_value=None,
    ):
        assert resolve_ocr_roi_local(payload, image_w=1000, image_h=800) == (
            10,
            10,
            200,
            150,
        )


def test_resolve_ocr_roi_local_keeps_rect_when_coverage_high() -> None:
    from src.recorder.window_snapshot import resolve_ocr_roi_local

    payload = {
        "hwnd": 1,
        "title": "App",
        "rect": [0, 0, 950, 750],
        "is_maximized": False,
    }
    with patch(
        "src.recorder.window_snapshot.find_matching_click_window",
        return_value=None,
    ):
        assert resolve_ocr_roi_local(payload, image_w=1000, image_h=800) == (
            0,
            0,
            950,
            750,
        )


def test_resolve_ocr_roi_local_clips_maximized_window_to_image() -> None:
    from src.recorder.window_snapshot import resolve_ocr_roi_local

    payload = {
        "hwnd": 1,
        "title": "常用 - 檔案總管",
        "rect": [-8, -8, 1936, 1048],
        "is_maximized": True,
    }
    with patch(
        "src.recorder.window_snapshot.find_matching_click_window",
        return_value=None,
    ):
        assert resolve_ocr_roi_local(payload, image_w=1920, image_h=1080) == (
            0,
            0,
            1920,
            1040,
        )


def test_resolve_ocr_roi_local_returns_clipped_rect_for_small_window() -> None:
    from src.recorder.window_snapshot import resolve_ocr_roi_local

    payload = {
        "hwnd": 1,
        "title": "Taskbar",
        "rect": [0, 700, 1000, 80],
        "is_maximized": False,
        "is_taskbar": True,
    }
    with patch(
        "src.recorder.window_snapshot.find_matching_click_window",
        return_value=None,
    ):
        roi = resolve_ocr_roi_local(payload, image_w=1000, image_h=800)
    assert roi == (0, 700, 1000, 80)


def test_resolve_ocr_roi_local_live_window_uses_monitor_offset() -> None:
    """A maximized window on monitor 2 must cover that screenshot, not an 8px strip."""
    from src.recorder.window_snapshot import ClickWindowInfo, resolve_ocr_roi_local

    payload = {
        "hwnd": 7,
        "title": "Chrome",
        "rect": [-8, -8, 1936, 1048],
        "is_maximized": True,
    }
    live = ClickWindowInfo(
        hwnd=7,
        title="Chrome",
        process_name="chrome.exe",
        left=1912,
        top=-9,
        width=1936,
        height=1048,
        is_maximized=True,
    )
    with patch(
        "src.recorder.window_snapshot.find_matching_click_window",
        return_value=live,
    ):
        roi = resolve_ocr_roi_local(
            payload,
            image_w=1920,
            image_h=1080,
            monitor_offset=(1920, -1),
        )
    assert roi == (0, 0, 1920, 1040)


def test_resolve_ocr_roi_local_none_on_other_monitor_offset() -> None:
    from src.recorder.window_snapshot import ClickWindowInfo, resolve_ocr_roi_local

    info = ClickWindowInfo(
        hwnd=1,
        title="App",
        process_name="app.exe",
        left=1920,
        top=0,
        width=400,
        height=300,
        is_maximized=False,
    )
    with patch(
        "src.recorder.window_snapshot.find_matching_click_window",
        return_value=info,
    ):
        # Image is monitor 0; window lives on monitor 1 → no intersection.
        assert (
            resolve_ocr_roi_local(
                info.to_dict(),
                image_w=1920,
                image_h=1080,
                monitor_offset=(0, 0),
            )
            is None
        )


def test_click_window_to_local_payload_subtracts_monitor_offset() -> None:
    from src.recorder.window_snapshot import ClickWindowInfo

    info = ClickWindowInfo(
        hwnd=7,
        title="Flyout",
        process_name="explorer.exe",
        left=100,
        top=200,
        width=300,
        height=400,
        is_maximized=False,
        is_flyout=True,
        class_name="Windows.UI.Core.CoreWindow",
    )
    payload = info.to_local_payload((50, 80))
    assert payload["rect"] == [50, 120, 300, 400]
    assert payload["is_flyout"] is True


def test_nchittest_lparam_packs_signed_screen_coordinates() -> None:
    import os

    from src.recorder.window_snapshot import (
        _nchittest_coord_fits,
        _nchittest_lparam,
        sample_caption_nchittest,
    )

    assert _nchittest_lparam(3815, 13) == ((13 & 0xFFFF) << 16) | (3815 & 0xFFFF)
    assert _nchittest_lparam(-8, -8) == ((-8 & 0xFFFF) << 16) | (-8 & 0xFFFF)
    assert _nchittest_coord_fits(3815, 13)
    assert not _nchittest_coord_fits(40000, 0)
    if os.name != "nt":
        assert sample_caption_nchittest(1, (0, 0, 40, 20)) is None


def test_sample_caption_nchittest_stops_when_window_does_not_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    if os.name != "nt":
        pytest.skip("WM_NCHITTEST sampling is Windows-only")

    from src.recorder.window_snapshot import sample_caption_nchittest

    monkeypatch.setattr(
        "src.recorder.window_snapshot._nchittest_sender",
        lambda: object(),
    )
    calls = {"n": 0}

    def _send(_send, _hwnd: int, x: int, y: int) -> int | None:
        calls["n"] += 1
        if calls["n"] > 2:
            return None
        return 20

    monkeypatch.setattr(
        "src.recorder.window_snapshot._send_wm_nchittest",
        _send,
    )
    samples = sample_caption_nchittest(1, (0, 0, 80, 30), step=10, slack=0)
    assert samples is not None
    assert [(x, y, code) for x, y, code in samples] == [(0, 15, 20), (10, 15, 20)]


def test_window_verify_keeps_flyout_disappear() -> None:
    before = [
        _win(
            9,
            "快顯主機",
            class_name="Microsoft.UI.Content.PopupWindowSiteBridge",
            process_name="explorer.exe",
            left=446,
            top=139,
            width=213,
            height=217,
        )
    ]
    predicate = build_window_verify_predicate(before, [])
    assert predicate == {
        "disappeared": [
            {
                "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
                "title": "快顯主機",
                "process_name": "explorer.exe",
            }
        ]
    }
    ok, reason = window_verify_satisfied(predicate, {}, live_windows_after=before)
    assert ok is False
    assert "快顯主機" in reason
    ok, _reason = window_verify_satisfied(predicate, {}, live_windows_after=[])
    assert ok is True


def test_window_verify_disappeared_passes_when_flyout_was_never_open() -> None:
    recorded = {
        "disappeared": [
            {
                "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
                "title": "快顯主機",
                "process_name": "explorer.exe",
            }
        ]
    }
    ok, _reason = window_verify_satisfied(recorded, {}, live_windows_after=[])
    assert ok is True


def test_window_verify_appeared_checks_live_after_not_the_delta() -> None:
    recorded = {
        "appeared": [
            {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
                "process_name": "explorer.exe",
            }
        ]
    }
    explorer = _win(
        3,
        "常用 - 檔案總管",
        class_name="CabinetWClass",
        process_name="explorer.exe",
    )
    ok, reason = window_verify_satisfied(recorded, {}, live_windows_after=[])
    assert ok is False
    assert "檔案總管" in reason
    ok, _reason = window_verify_satisfied(
        recorded,
        {},
        live_windows_after=[explorer],
    )
    assert ok is True


def test_window_verify_state_checks_live_after_flags_not_the_delta() -> None:
    recorded = {
        "state": [
            {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
                "change": "maximized",
            }
        ]
    }
    snapped = _win(
        3,
        "常用 - 檔案總管",
        class_name="CabinetWClass",
        is_maximized=False,
        left=0,
        top=0,
        width=960,
        height=1040,
    )
    maximized = _win(
        3,
        "常用 - 檔案總管",
        class_name="CabinetWClass",
        is_maximized=True,
        left=-8,
        top=-8,
        width=1936,
        height=1048,
    )
    # Already maximized before the step: no live delta, but after flags hold.
    ok, _reason = window_verify_satisfied(
        recorded,
        {"state": []},
        live_windows_after=[maximized],
    )
    assert ok is True
    ok, reason = window_verify_satisfied(
        recorded,
        {"state": []},
        live_windows_after=[snapped],
    )
    assert ok is False
    assert "常用 - 檔案總管" in reason
    # Without live_windows_after, still require the before→after delta.
    ok, reason = window_verify_satisfied(recorded, {"state": []})
    assert ok is False
    assert "常用 - 檔案總管" in reason
    ok, _reason = window_verify_satisfied(
        recorded,
        {
            "state": [
                {
                    "class_name": "CabinetWClass",
                    "title": "常用 - 檔案總管",
                    "change": "maximized",
                }
            ]
        },
    )
    assert ok is True


def test_window_verify_state_restored_and_minimized_use_after_flags() -> None:
    restored_recorded = {
        "state": [
            {
                "class_name": "Chrome_WidgetWin_1",
                "title": "Google Chrome",
                "change": "restored",
            }
        ]
    }
    minimized_recorded = {
        "state": [
            {
                "class_name": "Chrome_WidgetWin_1",
                "title": "Google Chrome",
                "change": "minimized",
            }
        ]
    }
    open_win = _win(
        4,
        "Google Chrome",
        class_name="Chrome_WidgetWin_1",
        is_minimized=False,
    )
    min_win = _win(
        4,
        "Google Chrome",
        class_name="Chrome_WidgetWin_1",
        is_minimized=True,
        left=-32000,
        top=-32000,
        width=160,
        height=28,
    )
    ok, _reason = window_verify_satisfied(
        restored_recorded,
        {},
        live_windows_after=[open_win],
    )
    assert ok is True
    ok, reason = window_verify_satisfied(
        restored_recorded,
        {},
        live_windows_after=[min_win],
    )
    assert ok is False
    assert "Google Chrome" in reason
    ok, _reason = window_verify_satisfied(
        minimized_recorded,
        {},
        live_windows_after=[min_win],
    )
    assert ok is True
    ok, reason = window_verify_satisfied(
        minimized_recorded,
        {},
        live_windows_after=[open_win],
    )
    assert ok is False
    assert "Google Chrome" in reason


def test_window_verify_ignores_title_when_class_and_process_match() -> None:
    recorded_appeared = {
        "appeared": [
            {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
                "process_name": "explorer.exe",
            }
        ]
    }
    renamed = _win(
        3,
        "下載 - 檔案總管",
        class_name="CabinetWClass",
        process_name="explorer.exe",
    )
    ok, _reason = window_verify_satisfied(
        recorded_appeared,
        {},
        live_windows_after=[renamed],
    )
    assert ok is True
    other_process = _win(
        4,
        "常用 - 檔案總管",
        class_name="CabinetWClass",
        process_name="chrome.exe",
    )
    ok, reason = window_verify_satisfied(
        recorded_appeared,
        {},
        live_windows_after=[other_process],
    )
    assert ok is False
    assert "檔案總管" in reason

    recorded_state = {
        "state": [
            {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
                "process_name": "explorer.exe",
                "change": "maximized",
            }
        ]
    }
    maximized = _win(
        3,
        "下載 - 檔案總管",
        class_name="CabinetWClass",
        process_name="explorer.exe",
        is_maximized=True,
    )
    ok, _reason = window_verify_satisfied(
        recorded_state,
        {},
        live_windows_after=[maximized],
    )
    assert ok is True

    recorded_foreground = {
        "foreground": {
            "class_name": "Chrome_WidgetWin_1",
            "title": "新分頁 - Google Chrome",
            "process_name": "chrome.exe",
        }
    }
    ok, _reason = window_verify_satisfied(
        recorded_foreground,
        {},
        live_after={
            "foreground": {
                "class_name": "Chrome_WidgetWin_1",
                "title": "YouTube - Google Chrome",
                "process_name": "chrome.exe",
            }
        },
    )
    assert ok is True
    ok, reason = window_verify_satisfied(
        recorded_foreground,
        {},
        live_after={
            "foreground": {
                "class_name": "Chrome_WidgetWin_1",
                "title": "新分頁 - Google Chrome",
                "process_name": "msedge.exe",
            }
        },
    )
    assert ok is False
    assert "foreground" in reason

    recorded_click = {
        "click_window": {
            "class_name": "CabinetWClass",
            "title": "常用 - 檔案總管",
            "process_name": "explorer.exe",
        }
    }
    ok, _reason = window_verify_satisfied(
        recorded_click,
        {},
        live_after={
            "point_window": {
                "class_name": "CabinetWClass",
                "title": "下載 - 檔案總管",
                "process_name": "explorer.exe",
            }
        },
    )
    assert ok is True


def test_limit_appeared_to_windows_that_become_foreground() -> None:
    from src.recorder.window_snapshot import limit_appeared_to_later_foreground

    predicate = {
        "appeared": [
            {
                "class_name": "Chrome_WidgetWin_1",
                "title": "Google Chrome",
                "process_name": "chrome.exe",
            },
            {
                "class_name": "Windows.UI.Core.CoreWindow",
                "title": "搜尋",
                "process_name": "SearchHost.exe",
            },
        ],
        "disappeared": [
            {"class_name": "Notepad", "title": "Untitled", "process_name": "notepad.exe"}
        ],
    }
    chrome_later = {
        "class_name": "Chrome_WidgetWin_1",
        "title": "新分頁 - Google Chrome",
        "process_name": "chrome.exe",
    }
    limited = limit_appeared_to_later_foreground(predicate, [chrome_later])
    assert limited["appeared"] == [predicate["appeared"][0]]
    assert limited["disappeared"] == predicate["disappeared"]
    assert "appeared" not in limit_appeared_to_later_foreground(predicate, [])


def test_window_verify_title_alone_does_not_match() -> None:
    """A recorded entry with no class and no process is not identified by title."""
    recorded = {"disappeared": [{"class_name": "", "title": "快顯主機"}]}
    ok, _reason = window_verify_satisfied(recorded, {}, live_windows_after=[])
    assert ok is True
    still_open = [
        _win(
            9,
            "快顯主機",
            class_name="Microsoft.UI.Content.PopupWindowSiteBridge",
            process_name="explorer.exe",
        )
    ]
    ok, _reason = window_verify_satisfied(
        recorded,
        {},
        live_windows_after=still_open,
    )
    assert ok is True


def test_window_verify_ignores_input_pane_appear_disappear() -> None:
    """A flickering EdgeUi strip must not block replay while a sibling remains."""
    strip = dict(class_name="EdgeUiInputTopWndClass", height=4, width=1905)
    before = [
        _win(1, "", left=15, top=0, **strip),
        _win(2, "", left=1935, top=-1, **strip),
    ]
    after = [
        _win(2, "", left=1935, top=-1, **strip),
        _win(3, "Google Chrome", class_name="Chrome_WidgetWin_1"),
    ]
    predicate = build_window_verify_predicate(before, after)
    assert predicate == {
        "appeared": [{"class_name": "Chrome_WidgetWin_1", "title": "Google Chrome"}]
    }
    recorded = {
        "appeared": [{"class_name": "Chrome_WidgetWin_1", "title": "Google Chrome"}],
        "disappeared": [{"class_name": "EdgeUiInputTopWndClass", "title": ""}],
    }
    ok, _reason = window_verify_satisfied(recorded, {}, live_windows_after=after)
    assert ok is True
    ok, reason = window_verify_satisfied(
        recorded,
        {},
        live_windows_after=[_win(2, "", left=1935, top=-1, **strip)],
    )
    assert ok is False
    assert "Google Chrome" in reason


def test_window_verify_ignores_taskbar_appear_disappear() -> None:
    before = [_win(1, "", class_name="Shell_TrayWnd", process_name="explorer.exe")]
    after = [_win(2, "", class_name="Shell_TrayWnd", process_name="explorer.exe")]
    assert build_window_verify_predicate(before, after) == {}
    recorded = {
        "disappeared": [{"class_name": "Shell_TrayWnd", "title": ""}],
    }
    ok, _reason = window_verify_satisfied(
        recorded,
        {},
        live_windows_after=after,
    )
    assert ok is True


def test_window_verify_drops_agent_hub_restore() -> None:
    before = [
        _win(
            1,
            "電腦使用代理",
            is_minimized=True,
            left=-32000,
            top=-32000,
            width=160,
            height=28,
        )
    ]
    after = [_win(1, "電腦使用代理", is_minimized=False)]
    assert build_window_verify_predicate(before, after) == {}


def test_window_verify_drops_same_hwnd_title_flicker() -> None:
    before = [_win(5, "資料夾", class_name="CabinetWClass")]
    after = [_win(5, "資料夾 - 檔案總管", class_name="CabinetWClass")]
    assert build_window_verify_predicate(before, after) == {}


def _explorer_foreground() -> dict[str, str]:
    return {
        "class_name": "CabinetWClass",
        "title": "檔案總管",
        "process_name": "explorer.exe",
    }


def _chrome_foreground() -> dict[str, str]:
    return {
        "class_name": "Chrome_WidgetWin_1",
        "title": "Google Chrome",
        "process_name": "chrome.exe",
    }


def test_window_verify_stores_foreground_only_when_it_changes() -> None:
    from src.recorder.verify_signals import signal_verify_fields

    explorer = _explorer_foreground()
    chrome = _chrome_foreground()
    assert "foreground" not in signal_verify_fields(
        {"foreground": explorer},
        {"foreground": explorer},
        kind="click",
    )
    changed = signal_verify_fields(
        {"foreground": explorer},
        {"foreground": chrome},
        kind="click",
    )
    assert changed["foreground"] == chrome
    agent = {
        "class_name": "Chrome_WidgetWin_1",
        "title": "電腦使用代理",
        "process_name": "python.exe",
    }
    assert "foreground" not in signal_verify_fields(
        {"foreground": explorer},
        {"foreground": agent},
        kind="click",
    )
    ok, reason = window_verify_satisfied(
        changed,
        {},
        live_after={"foreground": explorer},
    )
    assert ok is False
    assert "foreground" in reason
    ok, _reason = window_verify_satisfied(
        changed,
        {},
        live_after={"foreground": chrome},
    )
    assert ok is True


def test_window_verify_flyout_disappear_keeps_stable_foreground() -> None:
    explorer = _win(1, "檔案總管", class_name="CabinetWClass", process_name="explorer.exe")
    flyout = _win(
        9,
        "快顯主機",
        class_name="Microsoft.UI.Content.PopupWindowSiteBridge",
        process_name="explorer.exe",
    )
    foreground = _explorer_foreground()
    predicate = window_verify_from_debug(
        {
            "windows_before": [explorer.to_dict(), flyout.to_dict()],
            "windows_after": [explorer.to_dict()],
            "signals_before": {"foreground": foreground},
            "signals_after": {"foreground": foreground},
            "signal_kind": "click",
        }
    )
    assert "foreground" not in predicate
    assert predicate["disappeared"][0]["title"] == "快顯主機"


def test_window_verify_omits_foreground_when_disappeared_present() -> None:
    explorer = _win(1, "常用 - 檔案總管", class_name="CabinetWClass", process_name="explorer.exe")
    chrome = _win(2, "Google Chrome", class_name="Chrome_WidgetWin_1", process_name="chrome.exe")
    predicate = window_verify_from_debug(
        {
            "windows_before": [explorer.to_dict(), chrome.to_dict()],
            "windows_after": [chrome.to_dict()],
            "signals_before": {"foreground": _explorer_foreground()},
            "signals_after": {"foreground": _chrome_foreground()},
            "signal_kind": "click",
        }
    )
    assert predicate["disappeared"][0]["title"] == "常用 - 檔案總管"
    assert "foreground" not in predicate


def test_window_verify_omits_foreground_when_appeared_present() -> None:
    chrome = _win(2, "Google Chrome", class_name="Chrome_WidgetWin_1", process_name="chrome.exe")
    explorer = _win(1, "檔案總管", class_name="CabinetWClass", process_name="explorer.exe")
    predicate = window_verify_from_debug(
        {
            "windows_before": [chrome.to_dict()],
            "windows_after": [chrome.to_dict(), explorer.to_dict()],
            "signals_before": {"foreground": _chrome_foreground()},
            "signals_after": {"foreground": _explorer_foreground()},
            "signal_kind": "click",
        }
    )
    assert predicate["appeared"][0]["title"] == "檔案總管"
    assert "foreground" not in predicate


def test_window_verify_omits_foreground_when_state_present() -> None:
    before = _win(
        1,
        "檔案總管",
        class_name="CabinetWClass",
        process_name="explorer.exe",
        is_maximized=False,
        width=800,
        height=600,
    )
    after = _win(
        1,
        "檔案總管",
        class_name="CabinetWClass",
        process_name="explorer.exe",
        is_maximized=True,
        width=1920,
        height=1080,
    )
    predicate = window_verify_from_debug(
        {
            "windows_before": [before.to_dict()],
            "windows_after": [after.to_dict()],
            "signals_before": {"foreground": _chrome_foreground()},
            "signals_after": {"foreground": _explorer_foreground()},
            "signal_kind": "click",
        }
    )
    assert any(item.get("change") == "maximized" for item in predicate.get("state", []))
    assert "foreground" not in predicate


def test_window_verify_ignores_legacy_foreground_when_disappeared_present() -> None:
    """Old analyses that stored both still pass when only disappeared holds."""
    explorer = _win(1, "常用 - 檔案總管", class_name="CabinetWClass", process_name="explorer.exe")
    chrome = _win(2, "YouTube", class_name="Chrome_WidgetWin_1", process_name="chrome.exe")
    recorded = {
        "disappeared": [
            {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
                "process_name": "explorer.exe",
            }
        ],
        "foreground": {
            "class_name": "Chrome_WidgetWin_1",
            "title": "main.py - QualityAssuranceAgent - Cursor",
            "process_name": "Cursor.exe",
        },
    }
    ok, reason = window_verify_satisfied(
        recorded,
        {},
        live_after={"foreground": _chrome_foreground()},
        live_windows_after=[chrome],
    )
    assert ok is True
    assert reason == ""
    ok, reason = window_verify_satisfied(
        recorded,
        {},
        live_after={"foreground": _chrome_foreground()},
        live_windows_after=[explorer, chrome],
    )
    assert ok is False
    assert "disappeared" in reason


def test_window_verify_keeps_foreground_without_structural_assertions() -> None:
    changed = {
        "foreground": _chrome_foreground(),
    }
    ok, reason = window_verify_satisfied(
        changed,
        {},
        live_after={"foreground": _explorer_foreground()},
    )
    assert ok is False
    assert "foreground" in reason
    ok, _reason = window_verify_satisfied(
        changed,
        {},
        live_after={"foreground": _chrome_foreground()},
    )
    assert ok is True


def test_window_verify_omits_unchanged_light_signals_and_requires_changes() -> None:
    from src.recorder.verify_signals import signal_verify_fields

    point = _explorer_foreground()
    same = {
        "point_window": point,
        "clipboard": "old",
        "pid_names": {},
        "orphan_processes": [],
        "caret": {
            "rect": [10, 10, 12, 26],
            "hwnd": 4,
            "class_name": "Edit",
            "process_name": "notepad.exe",
        },
    }
    assert signal_verify_fields(same, dict(same), kind="click") == {}
    moved_caret = dict(same)
    moved_caret["caret"] = {
        "rect": [80, 10, 82, 26],
        "hwnd": 4,
        "class_name": "Edit",
        "process_name": "notepad.exe",
    }
    assert "caret" not in signal_verify_fields(same, moved_caret, kind="text_input")

    copied = dict(same)
    copied["clipboard"] = "copied text"
    clipboard_fields = signal_verify_fields(same, copied, kind="hotkey")
    assert clipboard_fields == {"clipboard": "copied text"}
    ok, _reason = window_verify_satisfied(
        clipboard_fields,
        {},
        live_after={"clipboard": "copied text"},
    )
    assert ok is True
    ok, reason = window_verify_satisfied(
        clipboard_fields,
        {},
        live_after={"clipboard": "old"},
    )
    assert ok is False
    assert "clipboard" in reason

    started = signal_verify_fields(
        {"orphan_processes": [], "pid_names": {}},
        {"orphan_processes": ["notepad.exe"], "pid_names": {}},
        kind="click",
    )
    assert started["process_started"] == ["notepad.exe"]
    explained = signal_verify_fields(
        {"orphan_processes": [], "pid_names": {}},
        {"orphan_processes": [], "pid_names": {"9": "notepad.exe"}},
        kind="click",
        before_windows=[],
        after_windows=[_win(3, "Untitled - Notepad", pid=9, class_name="Notepad")],
    )
    assert "process_started" not in explained
    ok, reason = window_verify_satisfied(
        started,
        {},
        live_after={"pid_names": {}, "orphan_processes": []},
    )
    assert ok is False
    assert "process_started" in reason

    other_field = dict(same)
    other_field["caret"] = {
        "rect": [10, 80, 12, 96],
        "hwnd": 8,
        "class_name": "Chrome_WidgetWin_1",
        "process_name": "chrome.exe",
    }
    caret_fields = signal_verify_fields(same, other_field, kind="text_input")
    assert caret_fields["caret"] == {
        "class_name": "Chrome_WidgetWin_1",
        "process_name": "chrome.exe",
    }
    assert "caret" not in signal_verify_fields(same, other_field, kind="click")
    ok, _reason = window_verify_satisfied(
        caret_fields,
        {},
        live_after={
            "caret": {
                "rect": [400, 500, 402, 520],
                "hwnd": 99,
                "class_name": "Chrome_WidgetWin_1",
                "process_name": "chrome.exe",
            }
        },
    )
    assert ok is True
    ok, reason = window_verify_satisfied(
        caret_fields,
        {},
        live_after={
            "caret": {"class_name": "Edit", "process_name": "notepad.exe"}
        },
    )
    assert ok is False
    assert "caret" in reason
    incomplete = dict(same)
    incomplete["caret"] = {"rect": [10, 80, 12, 96], "hwnd": 8, "class_name": "Edit"}
    assert "caret" not in signal_verify_fields(same, incomplete, kind="text_input")

    other_point = {
        "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
        "title": "快顯主機",
        "process_name": "explorer.exe",
    }
    click_fields = signal_verify_fields(
        {"point_window": point},
        {"point_window": other_point},
        kind="click",
    )
    assert click_fields["click_window"] == point
    assert "click_window" not in signal_verify_fields(
        {"point_window": point},
        {"point_window": point},
        kind="click",
    )


def test_window_verify_omits_click_window_when_it_matches_disappeared() -> None:
    """Close-button click: press-time window is the one that disappeared."""
    explorer = _win(1, "常用 - 檔案總管", class_name="CabinetWClass", process_name="explorer.exe")
    chrome = _win(2, "Google Chrome", class_name="Chrome_WidgetWin_1", process_name="chrome.exe")
    explorer_id = {
        "class_name": "CabinetWClass",
        "title": "常用 - 檔案總管",
        "process_name": "explorer.exe",
    }
    predicate = window_verify_from_debug(
        {
            "windows_before": [explorer.to_dict(), chrome.to_dict()],
            "windows_after": [chrome.to_dict()],
            "signals_before": {"point_window": explorer_id},
            "signals_after": {
                "point_window": {
                    "class_name": "Chrome_WidgetWin_1",
                    "title": "Google Chrome",
                    "process_name": "chrome.exe",
                }
            },
            "signal_kind": "click",
        }
    )
    assert predicate["disappeared"][0]["title"] == "常用 - 檔案總管"
    assert "click_window" not in predicate


def test_window_verify_keeps_click_window_when_disappeared_is_different() -> None:
    """Flyout dismiss: click landed on explorer, not on the flyout that closed."""
    explorer = _win(1, "常用 - 檔案總管", class_name="CabinetWClass", process_name="explorer.exe")
    flyout = _win(
        9,
        "快顯主機",
        class_name="Microsoft.UI.Content.PopupWindowSiteBridge",
        process_name="explorer.exe",
    )
    explorer_id = {
        "class_name": "CabinetWClass",
        "title": "常用 - 檔案總管",
        "process_name": "explorer.exe",
    }
    predicate = window_verify_from_debug(
        {
            "windows_before": [explorer.to_dict(), flyout.to_dict()],
            "windows_after": [explorer.to_dict()],
            "signals_before": {"point_window": explorer_id},
            "signals_after": {
                "point_window": {
                    "class_name": "Shell_TrayWnd",
                    "title": "",
                    "process_name": "explorer.exe",
                }
            },
            "signal_kind": "click",
        }
    )
    assert predicate["disappeared"][0]["title"] == "快顯主機"
    assert predicate["click_window"] == explorer_id


def test_window_verify_ignores_legacy_click_window_matching_disappeared() -> None:
    """close_windows replay has no press point; matching click_window must not fail."""
    chrome = _win(2, "YouTube", class_name="Chrome_WidgetWin_1", process_name="chrome.exe")
    recorded = {
        "disappeared": [
            {
                "class_name": "CabinetWClass",
                "title": "常用 - 檔案總管",
            }
        ],
        "click_window": {
            "class_name": "CabinetWClass",
            "title": "常用 - 檔案總管",
            "process_name": "explorer.exe",
        },
    }
    ok, reason = window_verify_satisfied(
        recorded,
        {},
        live_after={},
        live_windows_after=[chrome],
    )
    assert ok is True
    assert reason == ""


def test_replay_click_window_uses_press_time_window_not_the_settled_cursor(
    monkeypatch,
) -> None:
    from src.recorder.verify_signals import capture_replay_after_signals

    def _cursor_must_not_run() -> tuple[int, int]:
        raise AssertionError("settled cursor")

    monkeypatch.setattr("src.recorder.verify_signals._cursor_xy", _cursor_must_not_run)
    search = {
        "class_name": "Windows.UI.Core.CoreWindow",
        "title": "搜尋",
        "process_name": "SearchHost.exe",
    }
    sample = capture_replay_after_signals(
        [],
        {"click_window": search},
        search,
    )
    assert sample["point_window"] == search
    ok, _reason = window_verify_satisfied(
        {"click_window": search},
        {},
        live_after=sample,
    )
    assert ok is True
    missed = capture_replay_after_signals([], {"click_window": search}, None)
    ok, reason = window_verify_satisfied(
        {"click_window": search},
        {},
        live_after=missed,
    )
    assert ok is False
    assert "搜尋" in reason


def test_window_verify_uia_signals_only_when_the_step_changed_them(monkeypatch) -> None:
    from src.recorder.verify_signals import (
        capture_step_signals,
        read_uia_for_kind,
        signal_verify_fields,
    )

    def _boom(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("automation object")

    monkeypatch.setattr("src.recorder.verify_signals.read_uia_fields", _boom)
    assert read_uia_for_kind("drag", (1, 1)) == {}
    assert read_uia_for_kind("key_press", (1, 1)) == {}

    calls: list[str] = []

    def _reader(kind: str | None, _cursor: tuple[int, int] | None) -> dict[str, object]:
        calls.append(str(kind))
        if kind == "text_input":
            return {"focused": {"name": "Name", "value": "Ada"}}
        if kind == "scroll":
            return {"scroll": 40.0}
        return {"control_state": "selected"}

    capture_step_signals([], cursor_xy=(1, 1), kind="text_input", uia_reader=_reader)
    capture_step_signals([], cursor_xy=(1, 1), kind="scroll", uia_reader=_reader)
    capture_step_signals([], cursor_xy=(1, 1), kind="click", uia_reader=_reader)
    capture_step_signals([], cursor_xy=(1, 1), kind="drag", uia_reader=_reader)
    assert calls == ["text_input", "scroll", "click"]

    assert "control_state" not in signal_verify_fields(
        {"foreground": _explorer_foreground()},
        {"foreground": _explorer_foreground()},
        kind="click",
    )
    selected = signal_verify_fields(
        {"control_state": "unselected"},
        {"control_state": "selected"},
        kind="click",
    )
    assert selected == {}
    ok, _reason = window_verify_satisfied(
        {"control_state": "selected"},
        {},
        live_after={"control_state": "unselected"},
    )
    assert ok is True

    focused = signal_verify_fields(
        {
            "foreground": _chrome_foreground(),
            "focused": {"name": "Google Chrome", "value": ""},
        },
        {
            "foreground": _chrome_foreground(),
            "focused": {"name": "Google Chrome", "value": "hello"},
        },
        kind="text_input",
    )
    assert focused == {"focused": {"value": "hello"}}
    scrolled = signal_verify_fields({"scroll": 10.0}, {"scroll": 40.0}, kind="scroll")
    assert scrolled == {"scroll": 40.0}
    assert "scroll" not in signal_verify_fields({"scroll": 10.0}, {"scroll": 10.2}, kind="scroll")
    assert "scroll" not in signal_verify_fields({"scroll": 10.0}, {"scroll": 40.0}, kind="click")
    ok, _reason = window_verify_satisfied(scrolled, {}, live_after={"scroll": 43.0})
    assert ok is True
    ok, reason = window_verify_satisfied(scrolled, {}, live_after={"scroll": 50.0})
    assert ok is False
    assert "scroll" in reason


def test_window_verify_focused_value_allows_autocomplete_tail() -> None:
    recorded = {"focused": {"value": "nba l"}}
    ok, _reason = window_verify_satisfied(
        recorded,
        {},
        live_after={"focused": {"name": "網址與搜尋列", "value": "nba live"}},
    )
    assert ok is True
    ok, _reason = window_verify_satisfied(
        {"focused": {"value": "nba live"}},
        {},
        live_after={"focused": {"value": "nba l"}},
    )
    assert ok is True
    ok, reason = window_verify_satisfied(
        recorded,
        {},
        live_after={"focused": {"value": "nbanba live"}},
    )
    assert ok is False
    assert "focused" in reason
    ok, reason = window_verify_satisfied(
        {"focused": {"name": "網址與搜尋列", "value": "nba l"}},
        {},
        live_after={"focused": {"name": "搜尋", "value": "nba l"}},
    )
    assert ok is False
    assert "focused" in reason


def test_window_verify_drops_focused_value_unrelated_to_typed_text() -> None:
    debug = {
        "signal_kind": "text_input",
        "signals_before": {"focused": {"name": "網址與搜尋列", "value": ""}},
        "signals_after": {"focused": {"name": "網址與搜尋列", "value": "nba l"}},
    }
    assert window_verify_from_debug(debug, typed_text="nbanba live") == {}
    assert window_verify_from_debug(debug, typed_text="nba live") == {
        "focused": {"value": "nba l"}
    }
    assert window_verify_from_debug(debug, typed_text="nba l") == {
        "focused": {"value": "nba l"}
    }


def test_filter_disabled_window_verify_drops_unchecked_entries() -> None:
    predicate = {
        "disappeared": [
            {"title": "快顯主機", "class_name": "Popup", "process_name": "explorer.exe"},
            {"title": "檔案總管", "class_name": "CabinetWClass", "process_name": "explorer.exe"},
        ],
        "click_window": {
            "title": "檔案總管",
            "class_name": "CabinetWClass",
            "process_name": "explorer.exe",
        },
        "clipboard": "copied",
    }
    assert normalize_window_verify_disabled(
        predicate, ["click_window", "disappeared:0", "disappeared:0"]
    ) == ["click_window", "disappeared:0"]
    assert window_verify_condition_enabled(["disappeared:0"], "disappeared", 0) is False
    assert window_verify_condition_enabled(["disappeared:0"], "disappeared", 1) is True
    filtered = filter_disabled_window_verify(
        predicate, ["disappeared:0", "click_window"]
    )
    assert filtered == {
        "disappeared": [
            {"title": "檔案總管", "class_name": "CabinetWClass", "process_name": "explorer.exe"}
        ],
        "clipboard": "copied",
    }
    assert (
        filter_disabled_window_verify(
            predicate,
            ["clipboard", "disappeared:0", "disappeared:1", "click_window"],
        )
        == {}
    )
    with pytest.raises(ValueError):
        normalize_window_verify_disabled(predicate, ["disappeared"])
    with pytest.raises(ValueError):
        normalize_window_verify_disabled(predicate, ["disappeared:9"])
    assert normalize_window_verify_disabled(predicate, ["missing", "disappeared:9"], strict=False) == []
