"""Relabel misread title-bar icon OCR, and click system caption buttons directly.

A recorded target of 「關閉視窗」 / 「最小化視窗」 / 「最大化視窗」 / 「縮小視窗」
on a known window can be resolved from the DWM caption strip plus
WM_NCHITTEST, skipping screenshot, YOLO, and OCR.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cua_mcp.icon_map import load_icon_map
from cua_mcp.select_ui_element import UiDetection
from src.common.nearby_side import strip_nearby_context_comments

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

# Instruction labels → WM_NCHITTEST codes. Restore (縮小) is the same button
# as maximize; which glyph is painted depends on the window's zoomed state.
_CAPTION_ROLE_HIT_CODES: dict[str, int] = {
    "最小化視窗": 8,  # HTMINBUTTON
    "最大化視窗": 9,  # HTMAXBUTTON
    "縮小視窗": 9,
    "關閉視窗": 20,  # HTCLOSE
}
# Left-to-right cell inside the DWM strip: minimize, maximize/restore, close.
_CAPTION_ROLE_CELL: dict[str, int] = {
    "最小化視窗": 0,
    "最大化視窗": 1,
    "縮小視窗": 1,
    "關閉視窗": 2,
}
_CAPTION_ROLE_RE = re.compile(
    r"「(最小化視窗|最大化視窗|縮小視窗|關閉視窗)」(?:圖示|元素)"
)
_MOVE_PREFIX = "將滑鼠移到"
_CLICK_ACTION_SUFFIX_RE = re.compile(r"(，(?:並|用).+。)$")
_OFFSET_OR_TRACK_RE = re.compile(r"(?:右方|左方|上方|下方)\d+個像素|的\d+%處")
# Samples more than this many pixels apart are different buttons.
_HIT_RUN_MAX_GAP_PX = 8


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


def caption_role_from_instruction(instruction: str) -> str | None:
    """Return the caption-button label when ``instruction`` targets only that button.

    Pixel offsets and scrollbar percents stay on the vision path. Nearby
    parentheticals and the leading 「將滑鼠移到」 / trailing click sentence are
    ignored so both the short cache target and the full script line match.
    """
    text = strip_nearby_context_comments(instruction or "").strip()
    text = _CLICK_ACTION_SUFFIX_RE.sub("", text).strip().rstrip("，").strip()
    if text.startswith(_MOVE_PREFIX):
        text = text[len(_MOVE_PREFIX) :].strip()
    if not text or _OFFSET_OR_TRACK_RE.search(text):
        return None
    match = _CAPTION_ROLE_RE.fullmatch(text)
    if match is None:
        return None
    return match.group(1)


def caption_point_from_hit_samples(
    samples: list[tuple[int, int, int]],
    hit_code: int,
    *,
    max_gap: int = _HIT_RUN_MAX_GAP_PX,
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    """Center and xywh of the longest same-row run with ``hit_code``.

    The bbox is ``(x, y, w, h)`` around that run. ``h`` is 1 because a sample
    row has no height; callers may replace it with the caption-strip height.
    """
    rows: dict[int, list[int]] = {}
    wanted = int(hit_code)
    for x, y, code in samples:
        if int(code) != wanted:
            continue
        rows.setdefault(int(y), []).append(int(x))
    best: tuple[int, int, int, int] | None = None
    gap = max(1, int(max_gap))
    for y, xs in rows.items():
        xs.sort()
        start = prev = xs[0]
        runs: list[tuple[int, int]] = []
        for x in xs[1:]:
            if x - prev > gap:
                runs.append((start, prev))
                start = x
            prev = x
        runs.append((start, prev))
        for x0, x1 in runs:
            length = x1 - x0
            if best is None or length > best[0]:
                best = (length, y, x0, x1)
    if best is None:
        return None
    _length, y, x0, x1 = best
    width = max(1, x1 - x0)
    return (x0 + x1) // 2, y, (x0, y, width, 1)


def caption_button_cell_bounds(
    bounds: tuple[int, int, int, int],
    role: str,
) -> tuple[int, int, int, int] | None:
    """Screen ``(left, top, right, bottom)`` of one button inside a caption strip.

    Windows orders the system buttons left to right as minimize, maximize or
    restore, then close. The close cell keeps any remainder so it reaches the
    strip's right edge.
    """
    cell = _CAPTION_ROLE_CELL.get(role)
    if cell is None:
        return None
    left, top, right, bottom = (int(v) for v in bounds)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    cell_w = width // 3
    if cell_w <= 0:
        return None
    x0 = left + cell * cell_w
    x1 = right if cell == 2 else left + (cell + 1) * cell_w
    if x1 <= x0:
        return None
    return (x0, top, x1, bottom)


def _window_for_caption_click(click_window: dict[str, Any] | None):
    """Live window for a recorded ``click_window``, or None when it cannot be used."""
    if not click_window:
        return None
    from src.recorder.window_snapshot import (
        ClickWindowInfo,
        WindowInfo,
        find_matching_click_window,
    )

    info = (
        click_window
        if isinstance(click_window, ClickWindowInfo)
        else ClickWindowInfo.from_dict(click_window)
    )
    if info is None or info.is_taskbar or info.is_flyout:
        return None
    live = find_matching_click_window(info)
    if live is None or not live.hwnd or live.is_taskbar or live.is_flyout:
        return None
    if live.width <= 0 or live.height <= 0:
        return None
    if live.left <= -30000 or live.top <= -30000:
        return None
    return WindowInfo(
        hwnd=int(live.hwnd),
        title=live.title,
        pid=None,
        left=int(live.left),
        top=int(live.top),
        width=int(live.width),
        height=int(live.height),
        is_minimized=False,
        is_maximized=bool(live.is_maximized),
    )


def caption_button_cell_screen_rect(
    instruction: str,
    click_window: dict[str, Any] | None,
) -> tuple[int, int, int, int] | None:
    """Screen bounds of the caption button cell, without a hit-test.

    Used as the vision ROI when WM_NCHITTEST does not confirm the button
    (Windows 11 Explorer reports the strip but returns HTCLIENT). Returns None
    when DWM has no strip, so the caller keeps the window ROI.
    """
    role = caption_role_from_instruction(instruction)
    if role is None or not click_window:
        return None
    win = _window_for_caption_click(click_window)
    if win is None:
        return None
    from src.recorder.window_snapshot import dwm_caption_button_bounds

    bounds = dwm_caption_button_bounds(win)
    if bounds is None:
        return None
    return caption_button_cell_bounds(bounds, role)


def resolve_caption_button_mouse_point(
    instruction: str,
    click_window: dict[str, Any] | None,
) -> tuple[int, int, dict[str, Any]] | None:
    """Screen point of a system caption button, or None to keep the vision path.

    Requires a recorded ``click_window`` so the click stays on that window.
    DWM must report a caption strip, and WM_NCHITTEST must confirm the button.
    Custom title bars that fail either check return None.
    """
    role = caption_role_from_instruction(instruction)
    if role is None or not click_window:
        return None
    hit_code = _CAPTION_ROLE_HIT_CODES.get(role)
    if hit_code is None:
        return None

    from src.recorder.window_snapshot import (
        dwm_caption_button_bounds,
        sample_caption_nchittest,
    )

    win = _window_for_caption_click(click_window)
    if win is None:
        return None
    bounds = dwm_caption_button_bounds(win)
    if bounds is None:
        return None
    samples = sample_caption_nchittest(
        int(win.hwnd),
        bounds,
        stop_at_hit=hit_code,
    )
    if not samples:
        return None
    found = caption_point_from_hit_samples(samples, hit_code)
    if found is None:
        return None
    cx, cy, (x0, _y0, width, _height) = found
    _left, top, _right, bottom = (int(v) for v in bounds)
    strip_h = max(1, bottom - top)
    if not (top <= cy < bottom):
        cy = (top + bottom) // 2
    return (
        int(cx),
        int(cy),
        {
            "selected_index": 0,
            "class_name": "element",
            "image_center": {"x": int(cx), "y": int(cy)},
            "resolved_center": {"x": int(cx), "y": int(cy)},
            "relative_offset": {"dx": 0, "dy": 0},
            "track_percent": None,
            "char_target": None,
            "char_occurrence": 0,
            "anchor_instruction": (instruction or "").strip(),
            "nearby_objects": [],
            "screenshot_path": "",
            "screenshot_paths": [],
            "target_kind": "element",
            "target_text": "",
            "target_icons": [_caption_icon_record(role)],
            "target_bbox": {
                "x": int(x0),
                "y": int(top),
                "w": max(1, int(width)),
                "h": int(strip_h),
            },
            "selection_method": "caption_button",
            "caption_role": role,
        },
    )
