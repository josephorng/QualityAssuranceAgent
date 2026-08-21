from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from cua_mcp.icon_map import is_pua_char
from cua_mcp.select_ui_element import UiDetection
from src.recorder.models import RecordedEvent, screenshot_path_for_event_end
from src.recorder.vision_context import (
    _global_to_local,
    _global_to_local_end,
    build_vision_context_at_point,
    extract_nearest_text,
    resolve_event_screenshot_path,
)


def _vision_for_llm(vision: dict[str, Any]) -> dict[str, Any]:
    return {
        "used_vision": vision.get("used_vision"),
        "candidate_text": vision.get("candidate_text"),
        "local_cursor": vision.get("local_cursor"),
        "candidates": vision.get("candidates"),
        "detection_count": vision.get("detection_count"),
    }


def _strip_ocr_caret(text: str) -> str:
    """Remove a trailing ``|`` that OCR often reads from the text caret."""
    cleaned = text.rstrip()
    if cleaned.endswith("|"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _is_ocr_option_text(text: str) -> bool:
    """Skip empty / icon-only OCR strings when listing focus-rect options."""
    if not text:
        return False
    if all(is_pua_char(ch) for ch in text):
        return False
    return True


def _focus_rect_to_local(
    event: RecordedEvent,
    focus_rect: tuple[int, int, int, int],
    *,
    use_end: bool,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = focus_rect
    if use_end:
        tl = _global_to_local_end(event, (left, top))
        br = _global_to_local_end(event, (right, bottom))
    else:
        tl = _global_to_local(event, (left, top))
        br = _global_to_local(event, (right, bottom))
    return (int(tl[0]), int(tl[1]), int(br[0]), int(br[1]))


def ocr_texts_inside_focus_rect(
    detections: list[UiDetection],
    local_rect: tuple[int, int, int, int],
) -> list[str]:
    """Return unique OCR strings whose detection centers lie inside ``local_rect``."""
    return [text for text, _cx, _cy in _ocr_entries_inside_focus_rect(detections, local_rect)]


def _ocr_entries_inside_focus_rect(
    detections: list[UiDetection],
    local_rect: tuple[int, int, int, int],
) -> list[tuple[str, int, int]]:
    """Unique ``(text, cx, cy)`` entries whose centers lie inside ``local_rect``."""
    left, top, right, bottom = local_rect
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    entries: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for det in sorted(detections, key=lambda item: (item.cy, item.cx)):
        raw = (det.text or "").strip()
        if not raw:
            continue
        if not (left <= det.cx <= right and top <= det.cy <= bottom):
            continue
        cleaned = _strip_ocr_caret(raw)
        if not _is_ocr_option_text(cleaned) or cleaned in seen:
            continue
        seen.add(cleaned)
        entries.append((cleaned, int(det.cx), int(det.cy)))
    return entries


def _nearest_ocr_entry(
    entries: list[tuple[str, int, int]],
    local_x: int,
    local_y: int,
) -> str | None:
    """Return the text entry whose center is nearest to the local caret point."""
    if not entries:
        return None
    return min(
        entries,
        key=lambda item: (item[1] - local_x) ** 2 + (item[2] - local_y) ** 2,
    )[0]


async def resolve_text_input_text(
    event: RecordedEvent,
    *,
    run_dir: Path,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resolve typing text, preferring recorded keystrokes; OCR kept as alternate."""
    recorded_text = event.text or ""
    anchor = event.cursor_xy or event.anchor_click_xy
    if anchor is None:
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "ocr_text": None,
            "ocr_options": [],
            "source": "recorded",
            "meaningful": None,
            "reason": "vision unavailable: no cursor or anchor coordinates",
            "vision": None,
        }

    # Prefer the dedicated typing end frame (captured at focus on the typing monitor).
    # Shared next-action screenshots can be on another monitor and break OCR anchoring.
    typing_end = screenshot_path_for_event_end(run_dir, event.index)
    if typing_end.is_file():
        local = _global_to_local(event, anchor)
        debug_name = "_end"
        ocr_reason = "after-screenshot OCR"
        ocr_image_path = str(typing_end)
        use_end = True
    else:
        use_after = resolve_event_screenshot_path(event, run_dir, debug_name="_end") is not None
        if use_after:
            local = _global_to_local_end(event, anchor)
            debug_name = "_end"
            ocr_reason = "after-screenshot OCR"
            ocr_image_path = None
            use_end = True
        else:
            local = _global_to_local(event, anchor)
            debug_name = None
            ocr_reason = "vision-first OCR"
            ocr_image_path = None
            use_end = False

    vision = build_vision_context_at_point(
        event,
        local_x=local[0],
        local_y=local[1],
        run_dir=run_dir,
        persist_debug=True,
        reference_xy=anchor,
        debug_name=debug_name,
        image_path=ocr_image_path,
    )
    bgr = vision.pop("bgr", None)
    all_detections = vision.pop("all_detections", [])
    ocr_text: str | None = None
    ocr_options: list[str] = []
    if bgr is not None and all_detections:
        if event.focus_rect is not None:
            # Restrict OCR choices to the caret/UIA focus rect; never import
            # nearest-screen hits (e.g. taskbar 「搜尋」) from outside it.
            local_rect = _focus_rect_to_local(event, event.focus_rect, use_end=use_end)
            entries = _ocr_entries_inside_focus_rect(all_detections, local_rect)
            ocr_options = [text for text, _cx, _cy in entries]
            ocr_text = _nearest_ocr_entry(entries, local[0], local[1])
        else:
            raw_ocr = extract_nearest_text(bgr, all_detections, local[0], local[1])
            if raw_ocr:
                cleaned = _strip_ocr_caret(raw_ocr)
                ocr_text = cleaned or None
            if ocr_text:
                ocr_options = [ocr_text]

    vision_for_llm = _vision_for_llm(vision)
    used_after = debug_name == "_end"
    if recorded_text:
        if log_info is not None:
            shot_label = "after" if used_after else "before"
            log_info(
                f"resolve_text_input_text event={event.index} "
                f"shot={shot_label} recorded={recorded_text!r} ocr={ocr_text!r} "
                f"ocr_options={len(ocr_options)} resolved={recorded_text!r} source=recorded"
            )
        return {
            "text": recorded_text,
            "recorded_text": recorded_text,
            "ocr_text": ocr_text,
            "ocr_options": ocr_options,
            "source": "recorded",
            "meaningful": None,
            "reason": (
                f"prefer recorded text; {ocr_reason} available as alternate"
                if ocr_text or ocr_options
                else "prefer recorded text; vision produced no OCR alternate"
            ),
            "vision": vision_for_llm,
        }

    if ocr_text:
        if log_info is not None:
            shot_label = "after" if used_after else "before"
            log_info(
                f"resolve_text_input_text event={event.index} "
                f"shot={shot_label} recorded={recorded_text!r} resolved={ocr_text!r}"
            )
        return {
            "text": ocr_text,
            "recorded_text": recorded_text,
            "ocr_text": ocr_text,
            "ocr_options": ocr_options or [ocr_text],
            "source": "ocr",
            "meaningful": None,
            "reason": f"recorded text empty; used {ocr_reason}",
            "vision": vision_for_llm,
        }

    return {
        "text": recorded_text,
        "recorded_text": recorded_text,
        "ocr_text": None,
        "ocr_options": ocr_options,
        "source": "recorded",
        "meaningful": None,
        "reason": "vision produced no text; kept recorded text",
        "vision": vision_for_llm,
    }


def event_with_resolved_text(event: RecordedEvent, resolved: dict[str, Any]) -> RecordedEvent:
    """Return a copy of ``event`` with ``text`` replaced by the resolved value."""
    return replace(event, text=resolved["text"])
