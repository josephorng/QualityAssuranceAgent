"""Tests for caption-button icon relabeling after YOLO+OCR."""

from __future__ import annotations

import pytest

from cua_mcp.caption_buttons import relabel_caption_button_detections
from cua_mcp.select_ui_element import UiDetection
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT
from src.recorder.window_snapshot import WindowInfo


def _detection_from_bbox(
    bbox: tuple[int, int, int, int],
    class_id: int,
    *,
    text: str | None = None,
    icons: list[dict[str, str]] | None = None,
) -> UiDetection:
    x, y, w, h = bbox
    return UiDetection(
        bbox=bbox,
        cx=x + w // 2,
        cy=y + h // 2,
        class_id=class_id,
        class_name="element" if class_id == YOLO_CLASS_ELEMENT else "text",
        text=text,
        icons=icons,
    )


def _caption_window(
  bounds: tuple[int, int, int, int],
) -> WindowInfo:
    left, top, right, bottom = bounds
    return WindowInfo(
        hwnd=1,
        title="Test App",
        pid=100,
        left=left - 200,
        top=top,
        width=right - left + 200,
        height=200,
        is_minimized=False,
        is_maximized=False,
        caption_button_bounds=bounds,
    )


@pytest.mark.parametrize(
    ("source_id", "expected"),
    [
        ("複製工作表", "縮小視窗"),
        ("方框、矩形框線", "最大化視窗"),
        ("水平線、最小化", "最小化視窗"),
        ("關閉、取消、刪除", "關閉視窗"),
    ],
)
def test_relabel_caption_icons_inside_strip(source_id: str, expected: str) -> None:
    bounds = (500, 10, 650, 42)
    win = _caption_window(bounds)
    det = _detection_from_bbox(
        (580, 18, 20, 20),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": source_id, "pua": "\ue000"}],
    )
    out = relabel_caption_button_detections([det], coord_offset=(0, 0), windows=[win])
    assert (out[0].icons or [{}])[0]["chinese_id"] == expected


def test_relabel_skips_outside_caption_strip() -> None:
    bounds = (500, 10, 650, 42)
    win = _caption_window(bounds)
    det = _detection_from_bbox(
        (300, 100, 20, 20),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "方框、矩形框線"}],
    )
    out = relabel_caption_button_detections([det], coord_offset=(0, 0), windows=[win])
    assert (out[0].icons or [{}])[0]["chinese_id"] == "方框、矩形框線"


def test_relabel_uses_coord_offset_for_screen_coords() -> None:
    bounds = (1100, 10, 1250, 42)
    win = _caption_window(bounds)
    # Local center (590, 28) + offset (500, 0) => screen (1090, 28) inside bounds.
    det = _detection_from_bbox(
        (580, 18, 20, 20),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "水平線、最小化"}],
    )
    out = relabel_caption_button_detections(
        [det],
        coord_offset=(500, 0),
        windows=[win],
    )
    assert (out[0].icons or [{}])[0]["chinese_id"] == "最小化視窗"


def test_relabel_close_text_in_caption_strip() -> None:
    bounds = (500, 10, 650, 42)
    win = _caption_window(bounds)
    det = _detection_from_bbox((620, 18, 16, 16), YOLO_CLASS_TEXT, text="關閉")
    out = relabel_caption_button_detections([det], windows=[win])
    assert out[0].text == "關閉視窗"
    assert out[0].icons is None


def test_relabel_does_not_rename_cancel_outside_caption() -> None:
    bounds = (500, 10, 650, 42)
    win = _caption_window(bounds)
    det = _detection_from_bbox((200, 300, 40, 20), YOLO_CLASS_TEXT, text="取消")
    out = relabel_caption_button_detections([det], windows=[win])
    assert out[0].text == "取消"


def test_caption_role_from_instruction_accepts_close_icon_only() -> None:
    from cua_mcp.caption_buttons import caption_role_from_instruction

    goal = (
        "將滑鼠移到「關閉視窗」圖示"
        "（在「詳細資料」文字的上面、在「搜尋 常用」文字的右上方），並點擊滑鼠一下。"
    )
    assert caption_role_from_instruction("「關閉視窗」圖示") == "關閉視窗"
    assert caption_role_from_instruction(goal) == "關閉視窗"
    assert caption_role_from_instruction("「最小化視窗」圖示") == "最小化視窗"
    assert caption_role_from_instruction("「最大化視窗」元素") == "最大化視窗"
    assert caption_role_from_instruction("「縮小視窗」圖示") == "縮小視窗"
    assert caption_role_from_instruction("「關閉視窗」圖示左方5個像素") is None
    assert caption_role_from_instruction("「取消」文字") is None
    assert caption_role_from_instruction("關閉「記事本」視窗") is None


def test_caption_point_from_hit_samples_uses_longest_close_run() -> None:
    from cua_mcp.caption_buttons import caption_point_from_hit_samples

    samples: list[tuple[int, int, int]] = []
    for x in range(3700, 3748, 4):
        samples.append((x, 16, 8))
    for x in range(3748, 3796, 4):
        samples.append((x, 16, 9))
    for x in range(3796, 3844, 4):
        samples.append((x, 16, 20))
    # A shorter close run must not win.
    samples.extend([(100, 16, 20), (104, 16, 20)])
    cx, cy, bbox = caption_point_from_hit_samples(samples, 20)
    assert cy == 16
    assert cx == (3796 + 3840) // 2
    assert bbox[0] == 3796


def test_resolve_caption_button_clicks_confirmed_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp.caption_buttons import resolve_caption_button_mouse_point
    from src.recorder.window_snapshot import ClickWindowInfo

    live = ClickWindowInfo(
        hwnd=7,
        title="常用 - 檔案總管",
        process_name="explorer.exe",
        left=1920,
        top=-8,
        width=1920,
        height=1080,
        is_maximized=True,
        class_name="CabinetWClass",
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot.find_matching_click_window",
        lambda _info: live,
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot.dwm_caption_button_bounds",
        lambda _win: (3700, 0, 3840, 32),
    )

    def _sample(hwnd: int, bounds: tuple[int, int, int, int], **_kwargs):
        assert hwnd == 7
        assert bounds == (3700, 0, 3840, 32)
        samples: list[tuple[int, int, int]] = []
        for x in range(3700, 3748, 4):
            samples.append((x, 16, 8))
        for x in range(3748, 3796, 4):
            samples.append((x, 16, 9))
        for x in range(3796, 3844, 4):
            samples.append((x, 16, 20))
        return samples

    monkeypatch.setattr(
        "src.recorder.window_snapshot.sample_caption_nchittest",
        _sample,
    )
    found = resolve_caption_button_mouse_point(
        "「關閉視窗」圖示",
        {
            "hwnd": 7,
            "title": "常用 - 檔案總管",
            "rect": [-8, -8, 1920, 1080],
            "is_maximized": True,
            "class_name": "CabinetWClass",
        },
    )
    assert found is not None
    x, y, meta = found
    assert (x, y) == ((3796 + 3840) // 2, 16)
    assert meta["selection_method"] == "caption_button"
    assert meta["caption_role"] == "關閉視窗"
    assert meta["target_icons"][0]["chinese_id"] == "關閉視窗"
    assert meta["target_bbox"]["y"] == 0
    assert meta["target_bbox"]["h"] == 32


def test_resolve_caption_button_falls_back_without_close_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp.caption_buttons import resolve_caption_button_mouse_point
    from src.recorder.window_snapshot import ClickWindowInfo

    live = ClickWindowInfo(
        hwnd=7,
        title="Chrome",
        process_name="chrome.exe",
        left=0,
        top=0,
        width=800,
        height=600,
        is_maximized=False,
        class_name="Chrome_WidgetWin_1",
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot.find_matching_click_window",
        lambda _info: live,
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot.dwm_caption_button_bounds",
        lambda _win: (700, 0, 800, 32),
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot.sample_caption_nchittest",
        lambda *_a, **_k: [(720, 16, 1), (760, 16, 1)],
    )
    assert (
        resolve_caption_button_mouse_point(
            "「關閉視窗」圖示",
            {"hwnd": 7, "title": "Chrome", "rect": [0, 0, 800, 600]},
        )
        is None
    )


def test_resolve_caption_button_requires_click_window() -> None:
    from cua_mcp.caption_buttons import resolve_caption_button_mouse_point

    assert resolve_caption_button_mouse_point("「關閉視窗」圖示", None) is None


def test_caption_button_cell_bounds_splits_strip_left_to_right() -> None:
    from cua_mcp.caption_buttons import caption_button_cell_bounds

    # Maximized Explorer strip from a live probe: close glyph sat near x=3815.
    bounds = (3693, -9, 3839, 21)
    assert caption_button_cell_bounds(bounds, "最小化視窗") == (3693, -9, 3741, 21)
    assert caption_button_cell_bounds(bounds, "最大化視窗") == (3741, -9, 3789, 21)
    assert caption_button_cell_bounds(bounds, "縮小視窗") == (3741, -9, 3789, 21)
    close = caption_button_cell_bounds(bounds, "關閉視窗")
    assert close == (3789, -9, 3839, 21)
    assert close is not None
    assert close[0] <= 3815 < close[2]
    assert caption_button_cell_bounds((0, 0, 0, 10), "關閉視窗") is None


def test_caption_button_cell_screen_rect_uses_dwm_without_hit_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp.caption_buttons import caption_button_cell_screen_rect
    from src.recorder.window_snapshot import ClickWindowInfo

    live = ClickWindowInfo(
        hwnd=7,
        title="常用 - 檔案總管",
        process_name="explorer.exe",
        left=1912,
        top=-9,
        width=1936,
        height=1048,
        is_maximized=True,
        class_name="CabinetWClass",
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot.find_matching_click_window",
        lambda _info: live,
    )
    monkeypatch.setattr(
        "src.recorder.window_snapshot.dwm_caption_button_bounds",
        lambda _win: (3693, -9, 3839, 21),
    )
    def _fail_hit_test(*_args, **_kwargs):
        raise AssertionError("hit-test not used for the caption ROI")

    monkeypatch.setattr(
        "src.recorder.window_snapshot.sample_caption_nchittest",
        _fail_hit_test,
    )
    assert caption_button_cell_screen_rect(
        "「關閉視窗」圖示",
        {"hwnd": 7, "title": "常用 - 檔案總管", "rect": [-8, -8, 1936, 1048]},
    ) == (3789, -9, 3839, 21)
