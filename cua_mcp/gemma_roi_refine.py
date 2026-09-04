"""Gemma ROI fallback: narrow a region, re-run YOLO/OCR on the crop, index-pick."""

from __future__ import annotations

from typing import Any

import numpy as np

from cua_mcp.llm_json import parse_json_object
from cua_mcp.select_ui_element import (
    UiDetection,
    _format_ui_candidates_text,
    _parse_index_from_llm,
    _sort_detections_reading_order,
)
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END
from src.common.io_utils import imread_bgr, imwrite_bgr
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager, ts_name
from src.common.settings import load_settings
from src.eye.capture import active_monitor_offset

MAX_ROI_ROUNDS = 2
ROI_PAD_FRAC = 0.08
_MIN_CROP_SIDE_PX = 32

_VISUAL_MOUSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "text": {"type": "string"},
    },
    "required": ["index", "text"],
    "additionalProperties": False,
}

_ROI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "image_index": {"type": "integer"},
        "nx1": {"type": "integer"},
        "ny1": {"type": "integer"},
        "nx2": {"type": "integer"},
        "ny2": {"type": "integer"},
    },
    "required": ["found", "image_index", "nx1", "ny1", "nx2", "ny2"],
    "additionalProperties": False,
}


def _log_info(text: str) -> None:
    try:
        get_run_state_manager().log_info(text)
    except RuntimeError:
        pass


def clamp_norm(value: float) -> int:
    """Clamp a normalized coordinate into ``0..1000`` as int."""
    return int(max(0, min(1000, round(value))))


def normalized_roi_to_pixel_box(
    nx1: int,
    ny1: int,
    nx2: int,
    ny2: int,
    width: int,
    height: int,
    *,
    pad_frac: float = ROI_PAD_FRAC,
) -> tuple[int, int, int, int]:
    """Convert a normalized ``0..1000`` ROI to a padded pixel ``(x1,y1,x2,y2)`` box."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    a = clamp_norm(nx1)
    b = clamp_norm(nx2)
    c = clamp_norm(ny1)
    d = clamp_norm(ny2)
    left_n, right_n = (a, b) if a <= b else (b, a)
    top_n, bottom_n = (c, d) if c <= d else (d, c)
    if left_n == right_n:
        right_n = min(1000, left_n + 1)
    if top_n == bottom_n:
        bottom_n = min(1000, top_n + 1)

    x1 = int(left_n / 1000 * width)
    x2 = int(right_n / 1000 * width)
    y1 = int(top_n / 1000 * height)
    y2 = int(bottom_n / 1000 * height)

    if pad_frac > 0:
        pad_x = max(1, int(width * pad_frac))
        pad_y = max(1, int(height * pad_frac))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

    if x2 - x1 < _MIN_CROP_SIDE_PX:
        mid = (x1 + x2) // 2
        x1 = max(0, mid - _MIN_CROP_SIDE_PX // 2)
        x2 = min(width, x1 + _MIN_CROP_SIDE_PX)
        x1 = max(0, x2 - _MIN_CROP_SIDE_PX)
    if y2 - y1 < _MIN_CROP_SIDE_PX:
        mid = (y1 + y2) // 2
        y1 = max(0, mid - _MIN_CROP_SIDE_PX // 2)
        y2 = min(height, y1 + _MIN_CROP_SIDE_PX)
        y1 = max(0, y2 - _MIN_CROP_SIDE_PX)

    return x1, y1, x2, y2


def crop_bgr_with_origin(
    bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, int, int]:
    """Crop ``bgr`` to ``(x1,y1,x2,y2)``; return ``(crop, origin_x, origin_y)``."""
    x1, y1, x2, y2 = box
    h, w = bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return bgr[y1:y2, x1:x2].copy(), x1, y1


def metadata_from_detection(
    chosen: UiDetection,
    *,
    selected_index: int,
    selected_text: str,
    selection_method: str,
    image_paths: list[str],
    instruction: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard mouse-target metadata dict for a chosen detection."""
    x, y, w, h = chosen.bbox
    meta: dict[str, Any] = {
        "selected_index": selected_index,
        "selected_text": selected_text,
        "selection_method": selection_method,
        "screenshot_path": image_paths[0] if image_paths else "",
        "screenshot_paths": list(image_paths),
        "target_kind": chosen.class_name,
        "target_text": chosen.text or "",
        "target_icons": list(chosen.icons or []),
        "target_bbox": {"x": x, "y": y, "w": w, "h": h},
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "resolved_center": {"x": chosen.cx, "y": chosen.cy},
        "anchor_instruction": instruction,
    }
    if extra:
        meta.update(extra)
    return meta


async def pick_candidate_with_gemma(
    instruction: str,
    candidates: list[UiDetection],
    image_paths: list[str],
) -> tuple[int, str, UiDetection]:
    """Ask Gemma to pick one YOLO/OCR candidate index from ``candidates``."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    target = (instruction or "").strip()
    if not target:
        raise ValueError("instruction must be non-empty")

    prompt = get_prompt("visual_mouse_selection").format(
        instruction=target,
        candidates_text=_format_ui_candidates_text(candidates, include_geometry=True),
    )
    response = await get_llm_client().chat_messages(
        load_settings().brain_lm,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": list(image_paths),
            }
        ],
        tools=[],
        response_format=_VISUAL_MOUSE_SCHEMA,
        think=True,
    )
    selected_index, selected_text = _parse_index_from_llm(
        response.content,
        len(candidates),
    )
    return selected_index, selected_text, candidates[selected_index]


def _parse_roi_reply(raw: str, image_count: int) -> dict[str, Any] | None:
    data = parse_json_object(
        raw,
        empty_error="ROI reply empty",
        decode_error_prefix="invalid ROI JSON",
    )
    if not isinstance(data, dict):
        return None
    if not bool(data.get("found")):
        return None
    try:
        image_index = int(data["image_index"])
        nx1 = int(data["nx1"])
        ny1 = int(data["ny1"])
        nx2 = int(data["nx2"])
        ny2 = int(data["ny2"])
    except (KeyError, TypeError, ValueError):
        return None
    if image_index < 0 or image_index >= image_count:
        return None
    return {
        "image_index": image_index,
        "nx1": nx1,
        "ny1": ny1,
        "nx2": nx2,
        "ny2": ny2,
    }


async def ask_gemma_target_roi(
    instruction: str,
    image_paths: list[str],
    *,
    round_hint: str = "",
) -> dict[str, Any] | None:
    """Ask Gemma for a normalized ROI covering ``instruction``; ``None`` if not found."""
    if not image_paths:
        return None
    prompt = get_prompt("visual_target_roi").format(
        instruction=(instruction or "").strip(),
        image_count=len(image_paths),
        round_hint=round_hint or "(none)",
    )
    response = await get_llm_client().chat_messages(
        load_settings().brain_lm,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": list(image_paths),
            }
        ],
        tools=[],
        response_format=_ROI_SCHEMA,
        think=True,
    )
    try:
        return _parse_roi_reply(response.content, len(image_paths))
    except ValueError as exc:
        _log_info(f"gemma_roi: parse failed: {exc}")
        return None


def _offset_detection_xy(det: UiDetection, ox: int, oy: int) -> UiDetection:
    x, y, w, h = det.bbox
    return UiDetection(
        bbox=(x + ox, y + oy, w, h),
        cx=det.cx + ox,
        cy=det.cy + oy,
        class_id=det.class_id,
        class_name=det.class_name,
        text=det.text,
        icons=det.icons,
    )


async def resolve_target_via_gemma_roi(
    instruction: str,
    *,
    image_paths: list[str],
    monitor_indices: list[int],
    captured: list[tuple[int, np.ndarray]] | None = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
    max_rounds: int = MAX_ROI_ROUNDS,
) -> tuple[int, int, dict[str, Any]] | None:
    """
    Stage B fallback: Gemma ROI → YOLO/OCR on crop → index pick.

    Returns ``(global_x, global_y, metadata)`` or ``None`` when the target cannot
    be resolved after ``max_rounds``. Never uses Gemma's bbox center as the click.
    """
    target = (instruction or "").strip()
    if not target:
        raise ValueError("instruction must be non-empty")
    if not image_paths:
        return None

    # Lazy import avoids circular import with select_mouse_target.
    from cua_mcp.select_mouse_target import _detect_mouse_targets_from_bgr

    bgr_by_monitor: dict[int, np.ndarray] = {}
    if captured:
        for monitor_index, bgr in captured:
            bgr_by_monitor[int(monitor_index)] = bgr
    else:
        for path, monitor_index in zip(image_paths, monitor_indices):
            bgr = imread_bgr(path)
            if bgr is not None:
                bgr_by_monitor[int(monitor_index)] = bgr

    if not bgr_by_monitor:
        return None

    # Working view starts as full frames; later rounds may zoom into a crop.
    view_paths: list[str] = []
    view_bgrs: list[np.ndarray] = []
    view_origins: list[tuple[int, int, int]] = []  # (monitor_index, ox, oy) monitor-local
    for path, monitor_index in zip(image_paths, monitor_indices):
        bgr = bgr_by_monitor.get(int(monitor_index))
        if bgr is None:
            bgr = imread_bgr(path)
        if bgr is None:
            continue
        view_paths.append(path)
        view_bgrs.append(bgr)
        view_origins.append((int(monitor_index), 0, 0))
    if not view_bgrs:
        return None

    roi_history: list[dict[str, Any]] = []
    for round_idx in range(max(1, int(max_rounds))):
        round_hint = (
            "First pass: return a loose box covering the target on the full screenshot."
            if round_idx == 0
            else (
                "Previous crop still had no usable YOLO/OCR match. "
                "Return a tighter box around only the named target inside this crop."
            )
        )
        roi = await ask_gemma_target_roi(
            target,
            view_paths,
            round_hint=round_hint,
        )
        if roi is None:
            _log_info(f"gemma_roi: round={round_idx} found=false or parse miss")
            return None

        image_index = int(roi["image_index"])
        if image_index < 0 or image_index >= len(view_bgrs):
            _log_info(f"gemma_roi: round={round_idx} bad image_index={image_index}")
            return None

        view_bgr = view_bgrs[image_index]
        monitor_index, base_ox, base_oy = view_origins[image_index]
        vh, vw = view_bgr.shape[:2]
        pixel_box = normalized_roi_to_pixel_box(
            int(roi["nx1"]),
            int(roi["ny1"]),
            int(roi["nx2"]),
            int(roi["ny2"]),
            vw,
            vh,
        )
        crop, cox, coy = crop_bgr_with_origin(view_bgr, pixel_box)
        monitor_ox = base_ox + cox
        monitor_oy = base_oy + coy
        left, top = active_monitor_offset(monitor_index)

        roi_history.append(
            {
                "round": round_idx,
                "image_index": image_index,
                "nx1": roi["nx1"],
                "ny1": roi["ny1"],
                "nx2": roi["nx2"],
                "ny2": roi["ny2"],
                "pixel_box": list(pixel_box),
                "monitor_index": monitor_index,
                "monitor_origin": [monitor_ox, monitor_oy],
            }
        )
        _log_info(
            "gemma_roi: "
            f"round={round_idx} monitor={monitor_index} "
            f"norm=[{roi['nx1']},{roi['ny1']},{roi['nx2']},{roi['ny2']}] "
            f"pixel={list(pixel_box)} crop={crop.shape[1]}x{crop.shape[0]}"
        )

        try:
            local_dets = _detect_mouse_targets_from_bgr(
                crop,
                yolo_conf_threshold=yolo_conf_threshold,
                coord_offset=(0, 0),
            )
        except RuntimeError as exc:
            _log_info(f"gemma_roi: YOLO on crop failed: {exc}")
            local_dets = []

        # Map crop-local → monitor-local → virtual-desktop.
        global_dets = [
            _offset_detection_xy(
                _offset_detection_xy(det, monitor_ox, monitor_oy),
                left,
                top,
            )
            for det in local_dets
        ]
        global_dets = _sort_detections_reading_order(global_dets)
        _log_info(f"gemma_roi: round={round_idx} crop_candidates={len(global_dets)}")

        if global_dets:
            try:
                paths = get_run_state_manager().require_paths()
                crop_path = paths.yolo_ocr_dir / f"{ts_name()}_roi_r{round_idx}.png"
                imwrite_bgr(crop_path, crop)
                pick_images = [str(crop_path.resolve())]
            except RuntimeError:
                pick_images = [view_paths[image_index]]

            try:
                selected_index, selected_text, chosen = await pick_candidate_with_gemma(
                    target,
                    global_dets,
                    pick_images,
                )
            except Exception as exc:
                _log_info(
                    f"gemma_roi: index pick failed round={round_idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                # Fall through to a tighter ROI round when possible.
                selected_index = -1
                selected_text = ""
                chosen = None  # type: ignore[assignment]

            if chosen is not None and selected_index >= 0:
                meta = metadata_from_detection(
                    chosen,
                    selected_index=selected_index,
                    selected_text=selected_text,
                    selection_method="gemma_roi_yolo",
                    image_paths=image_paths,
                    instruction=target,
                    extra={
                        "roi_rounds": roi_history,
                        "roi_round": round_idx,
                    },
                )
                return chosen.cx, chosen.cy, meta

        # Prepare next round on this crop only (tighter zoom).
        if round_idx + 1 >= max_rounds:
            break
        try:
            paths = get_run_state_manager().require_paths()
            next_path = paths.yolo_ocr_dir / f"{ts_name()}_roi_view_r{round_idx + 1}.png"
            imwrite_bgr(next_path, crop)
            view_paths = [str(next_path.resolve())]
        except RuntimeError:
            # No run dir: keep in-memory crop via a temp-less path — cannot attach
            # bytes to chat_messages, so stop rather than reusing a stale full frame.
            _log_info("gemma_roi: cannot persist crop for next round; stopping")
            break
        view_bgrs = [crop]
        view_origins = [(monitor_index, monitor_ox, monitor_oy)]

    _log_info("gemma_roi: exhausted rounds without a detection pick")
    return None
