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
