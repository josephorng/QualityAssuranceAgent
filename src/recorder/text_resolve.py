from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from cua_mcp.icon_map import is_pua_char
from cua_mcp.select_ui_element import UiDetection
from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT
from src.recorder.models import RecordedEvent, screenshot_path_for_event_end
from src.recorder.vision_context import (
    _global_to_local,
    _global_to_local_end,
    _point_inside_bbox,
    _point_to_bbox_distance_sq,
    build_vision_context_at_point,
    extract_nearest_text,
    resolve_event_screenshot_path,
)

# System caret is typically 1–3px on one axis; real edit fields are wider/taller.
_CARET_THIN_MAX_PX = 8
# Absorb small YOLO vs caret misalignment when expanding an Input clip.
_INPUT_CLIP_PAD_PX = 8


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


def _normalize_ltrb(
    rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return (left, top, right, bottom)


def _is_caret_thin_rect(local_rect: tuple[int, int, int, int]) -> bool:
    """True when ``local_rect`` looks like a caret strip, not an edit-field bounds."""
    left, top, right, bottom = _normalize_ltrb(local_rect)
    return min(right - left, bottom - top) <= _CARET_THIN_MAX_PX


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
    return _normalize_ltrb((int(tl[0]), int(tl[1]), int(br[0]), int(br[1])))


def _bbox_to_ltrb(
    bbox: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (x - pad, y - pad, x + w + pad, y + h + pad)


def _union_ltrb(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    al, at, ar, ab = _normalize_ltrb(a)
    bl, bt, br, bb = _normalize_ltrb(b)
    return (min(al, bl), min(at, bt), max(ar, br), max(ab, bb))


def _select_input_for_caret(
    detections: list[UiDetection],
    local_x: int,
    local_y: int,
) -> UiDetection | None:
    """Prefer the Input containing the caret; else the nearest Input bbox."""
    inputs = [det for det in detections if det.class_id == YOLO_CLASS_INPUT]
    if not inputs:
        return None
    containing = [det for det in inputs if _point_inside_bbox(local_x, local_y, det.bbox)]
    if containing:
        return min(containing, key=lambda det: int(det.bbox[2]) * int(det.bbox[3]))
    return min(
        inputs,
        key=lambda det: _point_to_bbox_distance_sq(local_x, local_y, det.bbox),
    )


def _ocr_clip_rect_for_focus(
    *,
    local_focus_rect: tuple[int, int, int, int],
    detections: list[UiDetection],
    local_x: int,
    local_y: int,
) -> tuple[int, int, int, int] | None:
    """Return the OCR clip rect for a focus rect.

    Wide UIA field bounds are used as-is. Caret-thin rects are not a usable
    clip; expand via the nearest/containing YOLO Input (unioned with the caret)
    so typed text left of the caret is included without importing taskbar OCR.
    """
    if not _is_caret_thin_rect(local_focus_rect):
        return local_focus_rect

    input_det = _select_input_for_caret(detections, local_x, local_y)
    if input_det is None:
        return None

    input_clip = _bbox_to_ltrb(input_det.bbox, pad=_INPUT_CLIP_PAD_PX)
    return _union_ltrb(input_clip, local_focus_rect)


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
    left, top, right, bottom = _normalize_ltrb(local_rect)

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
    *,
    band_top: int | None = None,
    band_bottom: int | None = None,
    prefer_left_of_caret: bool = False,
) -> str | None:
    """Return the text entry whose center is nearest to the local caret point.

    When ``band_top``/``band_bottom`` are set, prefer entries in that vertical
    band (with small slack) so a filename field does not lose to a type
    dropdown on the next row. When ``prefer_left_of_caret`` is set, prefer
    text at or left of the caret (typical typed content in LTR/CJK fields).
    """
    if not entries:
        return None
    pool = entries
    if band_top is not None and band_bottom is not None:
        top, bottom = band_top, band_bottom
        if bottom < top:
            top, bottom = bottom, top
        slack = max(4, (bottom - top) // 2)
        in_band = [
            item
            for item in entries
            if (top - slack) <= item[2] <= (bottom + slack)
        ]
        if in_band:
            pool = in_band
    if prefer_left_of_caret:
        left = [item for item in pool if item[1] <= local_x]
        if left:
            pool = left
    return min(
        pool,
        key=lambda item: (item[1] - local_x) ** 2 + (item[2] - local_y) ** 2,
    )[0]


def _resolve_ocr_from_detections(
    *,
    bgr: Any,
    all_detections: list[UiDetection],
    local_x: int,
    local_y: int,
    focus_rect_local: tuple[int, int, int, int] | None,
) -> tuple[str | None, list[str]]:
    """Return ``(ocr_text, ocr_options)`` from vision detections."""
    if focus_rect_local is not None:
        clip = _ocr_clip_rect_for_focus(
            local_focus_rect=focus_rect_local,
            detections=all_detections,
            local_x=local_x,
            local_y=local_y,
        )
        if clip is not None:
            caret_thin = _is_caret_thin_rect(focus_rect_local)
            entries = _ocr_entries_inside_focus_rect(all_detections, clip)
            ocr_options = [text for text, _cx, _cy in entries]
            if caret_thin:
                left, top, right, bottom = focus_rect_local
                ocr_text = _nearest_ocr_entry(
                    entries,
                    local_x,
                    local_y,
                    band_top=top,
                    band_bottom=bottom,
                    prefer_left_of_caret=True,
                )
            else:
                ocr_text = _nearest_ocr_entry(entries, local_x, local_y)
            return ocr_text, ocr_options
        # Caret-thin with no Input gate: fall back to nearest-text helper.

    raw_ocr = extract_nearest_text(bgr, all_detections, local_x, local_y)
    ocr_text: str | None = None
    if raw_ocr:
        cleaned = _strip_ocr_caret(raw_ocr)
        ocr_text = cleaned or None
    ocr_options = [ocr_text] if ocr_text else []
    return ocr_text, ocr_options


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
        focus_local = (
            _focus_rect_to_local(event, event.focus_rect, use_end=use_end)
            if event.focus_rect is not None
            else None
        )
        ocr_text, ocr_options = _resolve_ocr_from_detections(
            bgr=bgr,
            all_detections=all_detections,
            local_x=local[0],
            local_y=local[1],
            focus_rect_local=focus_local,
        )

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
