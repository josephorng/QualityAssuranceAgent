"""Relabel misread title-bar icon OCR when detections fall in the caption button strip."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cua_mcp.icon_map import load_icon_map
from cua_mcp.select_ui_element import UiDetection

if TYPE_CHECKING:
    from src.recorder.window_snapshot import WindowInfo

# Misread PUA icon labels → canonical window-chrome names when inside caption buttons.
_CAPTION_ICON_RENAMES: dict[str, str] = {
    "複製工作表": "縮小視窗",
    "方框、矩形框線": "最大化視窗",
    "水平線、最小化": "最小化視窗",
    "關閉、取消、刪除": "關閉視窗",
}

# OCR may read the close glyph as plain text instead of a PUA icon.
_CAPTION_TEXT_RENAMES: dict[str, str] = {
    "關閉": "關閉視窗",
    "取消": "關閉視窗",
    "刪除": "關閉視窗",
}


def _window_snapshot():
    from src.recorder.window_snapshot import (
        click_hits_caption_buttons,
        snapshot_top_level_windows,
    )

    return click_hits_caption_buttons, snapshot_top_level_windows


def _icon_description_for_chinese_id(chinese_id: str) -> str:
    for value in load_icon_map().values():
        if not isinstance(value, dict):
            continue
        if str(value.get("chinese_id", "")).strip() == chinese_id:
            return str(value.get("description", "")).strip()
    return ""


def _caption_icon_record(chinese_id: str, *, pua: str = "") -> dict[str, Any]:
    return {
        "pua": pua,
        "chinese_id": chinese_id,
        "icon_description": _icon_description_for_chinese_id(chinese_id),
    }


def _pua_from_detection(det: UiDetection) -> str:
    for icon in det.icons or []:
        if not isinstance(icon, dict):
            continue
        pua = icon.get("pua")
        if isinstance(pua, str) and pua:
            return pua
    return ""


def _rebuild_detection(
    det: UiDetection,
    *,
    text: str | None = None,
    icons: list[dict[str, Any]] | None = None,
) -> UiDetection:
    x, y, w, h = det.bbox
    return UiDetection(
        bbox=det.bbox,
        cx=x + w // 2,
        cy=y + h // 2,
        class_id=det.class_id,
        class_name=det.class_name,
        text=text if text is not None else det.text,
        icons=icons if icons is not None else det.icons,
    )


def point_hits_any_caption_buttons(
    x: int,
    y: int,
    windows: list[WindowInfo] | None = None,
) -> bool:
    """True when ``(x, y)`` in screen coords lies in any window caption button strip."""
    if os.name != "nt":
        return False
    click_hits_caption_buttons, snapshot_top_level_windows = _window_snapshot()
    pool = windows if windows is not None else snapshot_top_level_windows()
    for win in pool:
        if click_hits_caption_buttons((int(x), int(y)), win):
            return True
    return False


def _detection_screen_center(
    det: UiDetection,
    coord_offset: tuple[int, int],
) -> tuple[int, int]:
    ox, oy = coord_offset
    return int(det.cx) + int(ox), int(det.cy) + int(oy)


def _rename_icon_detection(
    det: UiDetection,
    *,
    new_label: str,
) -> UiDetection:
    return _rebuild_detection(
        det,
        icons=[_caption_icon_record(new_label, pua=_pua_from_detection(det))],
        text=None,
    )


def relabel_caption_button_detections(
    detections: list[UiDetection],
    *,
    coord_offset: tuple[int, int] = (0, 0),
    windows: list[WindowInfo] | None = None,
    log_info: Callable[[str], None] | None = None,
) -> list[UiDetection]:
    """Rewrite misread caption icons/text to window-chrome labels inside the caption strip.

    ``coord_offset`` maps detection centers (screenshot-local) to virtual-desktop
    screen coordinates before testing against DWM caption bounds.
    """
    if not detections or os.name != "nt":
        return detections

    click_hits_caption_buttons, snapshot_top_level_windows = _window_snapshot()
    pool = windows if windows is not None else snapshot_top_level_windows()
    if not pool:
        return detections

    out = list(detections)
    relabeled = 0
    for index, det in enumerate(out):
        sx, sy = _detection_screen_center(det, coord_offset)
        if not any(click_hits_caption_buttons((sx, sy), win) for win in pool):
            continue

        new_label: str | None = None
        for icon in det.icons or []:
            if not isinstance(icon, dict):
                continue
            chinese_id = str(icon.get("chinese_id", "")).strip()
            mapped = _CAPTION_ICON_RENAMES.get(chinese_id)
            if mapped:
                new_label = mapped
                break

        if new_label is None:
            text = (det.text or "").strip()
            new_label = _CAPTION_TEXT_RENAMES.get(text)

        if new_label is None:
            continue

        if det.icons:
            out[index] = _rename_icon_detection(det, new_label=new_label)
        elif (det.text or "").strip() in _CAPTION_TEXT_RENAMES:
            out[index] = _rebuild_detection(
                det,
                text=new_label,
                icons=None,
            )
        else:
            continue
        relabeled += 1

    if relabeled and log_info is not None:
        log_info(f"relabel_caption_button_detections: relabeled={relabeled}")
    return out
