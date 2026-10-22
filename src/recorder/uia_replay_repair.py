"""Repair a recorded pointer step from a live UIA element during cache replay.

The recording walk can miss the control (empty ``FindAll`` on Word's start page).
Replay already knows the window and the OCR box, so a settled live walk can name
that same box and replace the OCR instruction.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.common.io_utils import read_json, write_json
from src.recorder.models import RecordedEvent, event_json_path

RectXywh = tuple[int, int, int, int]
LabeledBox = tuple[RectXywh, str]


@dataclass(frozen=True)
class UiaReplayRepairRequest:
    """One recorded OCR target whose UIA snapshot had no usable text."""

    recording_dir: Path
    event_index: int
    side: str
    recorded_bbox: RectXywh
    recorded_text: str
    recorded_window_xywh: RectXywh


@dataclass(frozen=True)
class UiaReplayRepairHit:
    label: str
    bbox: RectXywh
    source: str
    cx: int
    cy: int


_repair_slots: ContextVar[list[UiaReplayRepairRequest | None] | None] = ContextVar(
    "uia_replay_repair_slots",
    default=None,
)


@contextmanager
def bind_uia_replay_repairs(
    slots: list[UiaReplayRepairRequest | None],
) -> Iterator[None]:
    """Queue one slot per upcoming ``find_mouse_point`` call. ``None`` skips that call."""
    token = _repair_slots.set(list(slots))
    try:
        yield
    finally:
        _repair_slots.reset(token)


def take_uia_replay_repair() -> UiaReplayRepairRequest | None:
    """Pop the next replay-repair slot. ``None`` when this call should not repair."""
    slots = _repair_slots.get()
    if not slots:
        return None
    return slots.pop(0)


def snapshot_has_usable_text(snapshot: dict[str, Any] | None) -> bool:
    """True when a recorded snapshot already has a landmark label."""
    if not isinstance(snapshot, dict):
        return False
    texts = snapshot.get("texts")
    if not isinstance(texts, list):
        return False
    from src.recorder.uia_input_boxes import is_usable_uia_landmark_label

    for item in texts:
        if not isinstance(item, dict):
            continue
        if is_usable_uia_landmark_label(str(item.get("text") or "")):
            return True
    return False


def map_recorded_box_to_screen(
    recorded_bbox: RectXywh,
    recorded_window_xywh: RectXywh,
    live_window_xywh: RectXywh,
) -> tuple[RectXywh, tuple[int, int]]:
    """Window-relative recorded box placed on the live window. Returns xywh and center."""
    rx, ry, rw, rh = (int(v) for v in recorded_bbox)
    wx, wy, _ww, _wh = (int(v) for v in recorded_window_xywh)
    lx, ly, _lw, _lh = (int(v) for v in live_window_xywh)
    mapped = (lx + (rx - wx), ly + (ry - wy), rw, rh)
    center = (mapped[0] + rw // 2, mapped[1] + rh // 2)
    return mapped, center


def choose_uia_replay_repair(
    *,
    recorded_bbox: RectXywh,
    recorded_text: str,
    recorded_window_xywh: RectXywh,
    live_window_xywh: RectXywh,
    uia_boxes: list[LabeledBox],
    ocr_boxes: list[LabeledBox],
    point_probe_fn: Any = None,
) -> UiaReplayRepairHit | None:
    """Accept one UIA label when live OCR still shows the recorded target.

    The smallest UIA text rect that contains the mapped point wins. Two different
    names covering that point are left unchanged. An empty ``FindAll`` falls
    through to ``point_probe_fn(x, y) -> (label, xywh) | None``.
    """
    text = str(recorded_text or "").strip()
    if not text:
        return None
    mapped, center = map_recorded_box_to_screen(
        recorded_bbox, recorded_window_xywh, live_window_xywh
    )
    if not _ocr_confirms(mapped, text, ocr_boxes):
        return None
    from src.recorder.uia_input_boxes import is_usable_uia_landmark_label

    containing = [
        (box, label)
        for box, label in uia_boxes
        if is_usable_uia_landmark_label(label)
        and _point_in_box(center[0], center[1], box)
    ]
    if containing:
        names = {label for _box, label in containing}
        if len(names) != 1:
            return None
        box, label = min(containing, key=lambda item: int(item[0][2]) * int(item[0][3]))
        return _hit(label, box, "findall")

    if point_probe_fn is None:
        return None
    probed = point_probe_fn(int(center[0]), int(center[1]))
    if not isinstance(probed, tuple) or len(probed) != 2:
        return None
    label = str(probed[0] or "").strip()
    raw_box = probed[1]
    if not is_usable_uia_landmark_label(label):
        return None
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        box = tuple(int(v) for v in raw_box)
    except (TypeError, ValueError):
        return None
    if not _point_in_box(center[0], center[1], box):
        return None
    return _hit(label, box, "element_from_point")


def replay_repair_slots(
    recording_dir: Path,
    event_index: int,
    calls: list[dict[str, Any]],
) -> list[UiaReplayRepairRequest | None]:
    """Slots aligned with ``find_mouse_point`` calls inside this step's tool list."""
    names = [
        str(call.get("name") or "")
        for call in calls
        if isinstance(call, dict)
    ]
    slots: list[UiaReplayRepairRequest | None] = []
    for name in names:
        if name == "drag":
            slots.append(_load_request(recording_dir, event_index, "start"))
            slots.append(_load_request(recording_dir, event_index, "end"))
        elif name == "move_mouse":
            slots.append(_load_request(recording_dir, event_index, "click"))
    if not any(slot is not None for slot in slots):
        return []
    return slots


def attempt_uia_replay_repair(
    request: UiaReplayRepairRequest,
    click_window: dict[str, Any] | None,
    uia_boxes: list[LabeledBox],
    ocr_boxes: list[LabeledBox],
) -> tuple[int, int, dict[str, Any]] | None:
    """Match, persist the UIA instruction, and return the click point."""
    from src.recorder.uia_input_boxes import uia_text_element_at_point
    from src.recorder.window_snapshot import find_matching_click_window

    if not isinstance(click_window, dict):
        return None
    live = find_matching_click_window(click_window)
    if live is None:
        return None
    hit = choose_uia_replay_repair(
        recorded_bbox=request.recorded_bbox,
        recorded_text=request.recorded_text,
        recorded_window_xywh=request.recorded_window_xywh,
        live_window_xywh=live.screen_rect(),
        uia_boxes=uia_boxes,
        ocr_boxes=ocr_boxes,
        point_probe_fn=uia_text_element_at_point,
    )
    if hit is None:
        _log(
            "uia replay repair miss "
            f"event={request.event_index} side={request.side} "
            f"recorded={request.recorded_text!r}"
        )
        return None
    others = [
        (box, label)
        for box, label in uia_boxes
        if label != hit.label or box != hit.bbox
    ]
    try:
        write_uia_replay_repair(
            request,
            label=hit.label,
            source=hit.source,
            bbox=hit.bbox,
            others=others,
        )
    except Exception as exc:
        _log(
            "uia replay repair persist failed "
            f"event={request.event_index} {type(exc).__name__}: {exc}"
        )
    _log(
        "uia replay repair hit "
        f"event={request.event_index} side={request.side} "
        f"label={hit.label!r} source={hit.source} "
        f"center=[{hit.cx},{hit.cy}]"
    )
    return hit.cx, hit.cy, {
        "selected_index": 0,
        "class_name": "text",
        "image_center": {"x": hit.cx, "y": hit.cy},
        "resolved_center": {"x": hit.cx, "y": hit.cy},
        "relative_offset": {"dx": 0, "dy": 0},
        "track_percent": None,
        "char_target": None,
        "char_occurrence": None,
        "target_kind": "text",
        "target_text": hit.label,
        "target_icons": [],
        "target_bbox": {
            "x": hit.bbox[0],
            "y": hit.bbox[1],
            "w": hit.bbox[2],
            "h": hit.bbox[3],
        },
        "peer_bboxes": [],
        "landmarks": [],
        "selection_method": "uia_replay_repair",
        "uia_replay_repair": {"label": hit.label, "source": hit.source},
    }


def write_uia_replay_repair(
    request: UiaReplayRepairRequest,
    *,
    label: str,
    source: str,
    bbox: RectXywh,
    others: list[LabeledBox],
) -> str | None:
    """Rewrite the analysis instruction from UIA labels and refresh the tool cache."""
    run_dir = Path(request.recording_dir)
    event_path = event_json_path(run_dir, request.event_index)
    raw_event = read_json(event_path, None)
    if not isinstance(raw_event, dict):
        return None
    event = RecordedEvent.from_dict(raw_event)
    analysis_path = run_dir / "analysis" / f"event_{int(request.event_index):03d}.json"
    analysis = read_json(analysis_path, None)
    if not isinstance(analysis, dict):
        analysis = {"event_index": int(request.event_index)}
    instruction = _instruction_from_hit(event, request, label, bbox, others, analysis)
    if not instruction:
        return None
    analysis["instruction"] = instruction
    block = analysis.get("uia_replay_repair")
    if not isinstance(block, dict):
        block = {}
    block[request.side] = {"label": label, "source": source}
    block["label"] = label
    block["source"] = source
    analysis["uia_replay_repair"] = block

    from src.recorder.compile_tool_calls import (
        rebuild_recording_instruction_tool_cache,
        set_analysis_tool_calls,
    )

    set_analysis_tool_calls(
        analysis,
        event,
        instruction,
        run_dir=run_dir,
        vision=_vision_for_side(request.side, label, bbox, others),
    )
    write_json(analysis_path, analysis)
    rebuild_recording_instruction_tool_cache(run_dir)
    from src.common.run_control import mark_recording_replay_dirty
    from src.common.script_helper import invalidate_recording_replay_cache

    invalidate_recording_replay_cache(run_dir)
    mark_recording_replay_dirty()
    return instruction


def _instruction_from_hit(
    event: RecordedEvent,
    request: UiaReplayRepairRequest,
    label: str,
    bbox: RectXywh,
    others: list[LabeledBox],
    analysis: dict[str, Any],
) -> str | None:
    from src.recorder.analyze import rebuild_pointer_instruction

    primary = _candidate(label, bbox)
    extra = [
        _candidate(text, box)
        for box, text in others
        if str(text or "").strip() and (text != label or box != bbox)
    ]
    repaired = _vision([primary, *extra], (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
    if event.kind == "drag":
        other_text = _other_side_text(request)
        if not other_text:
            other_text = _fallback_other_anchor(
                str(analysis.get("instruction") or ""), request
            )
        if not other_text:
            return _replace_anchor(str(analysis.get("instruction") or ""), request, label)
        other = _vision([_candidate(other_text, (0, 0, 8, 8))], (4, 4))
        if request.side == "end":
            rebuilt = rebuild_pointer_instruction(
                event, other, repaired, promote_primary=False
            )
        else:
            rebuilt = rebuild_pointer_instruction(
                event, repaired, other, promote_primary=False
            )
        if rebuilt:
            return rebuilt
        return _replace_anchor(str(analysis.get("instruction") or ""), request, label)

    rebuilt = rebuild_pointer_instruction(event, repaired, promote_primary=False)
    if rebuilt:
        return rebuilt
    return _replace_anchor(str(analysis.get("instruction") or ""), request, label)


def _vision_for_side(
    side: str,
    label: str,
    bbox: RectXywh,
    others: list[LabeledBox],
) -> dict[str, Any]:
    primary = _candidate(label, bbox)
    extra = [_candidate(text, box) for box, text in others if str(text or "").strip()]
    vision = _vision(
        [primary, *extra],
        (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2),
    )
    if side == "end":
        return {"candidates": [_candidate(label, bbox)], "used_vision": True}
    return vision


def _load_request(
    recording_dir: Path,
    event_index: int,
    side: str,
) -> UiaReplayRepairRequest | None:
    run_dir = Path(recording_dir)
    raw_event = read_json(event_json_path(run_dir, event_index), None)
    if not isinstance(raw_event, dict):
        return None
    try:
        event = RecordedEvent.from_dict(raw_event)
    except (KeyError, TypeError, ValueError):
        return None
    analysis = read_json(run_dir / "analysis" / f"event_{int(event_index):03d}.json", None)
    instruction = ""
    if isinstance(analysis, dict):
        instruction = str(analysis.get("instruction") or "")
    from cua_mcp.caption_buttons import caption_role_from_instruction

    if instruction and caption_role_from_instruction(instruction):
        return None
    snapshot = event.end_uia_snapshot if side == "end" else event.uia_snapshot
    if snapshot_has_usable_text(snapshot if isinstance(snapshot, dict) else None):
        return None
    window = event.end_click_window if side == "end" else event.click_window
    recorded_window = _window_xywh(window if isinstance(window, dict) else None)
    if recorded_window is None:
        return None
    from src.recorder.vision_context import load_yolo_ocr_payload

    suffix = "_end" if side == "end" else ""
    primary = _recorded_primary(load_yolo_ocr_payload(run_dir, event_index, suffix=suffix))
    if primary is None:
        return None
    bbox, text = primary
    return UiaReplayRepairRequest(
        recording_dir=run_dir,
        event_index=int(event_index),
        side=side,
        recorded_bbox=bbox,
        recorded_text=text,
        recorded_window_xywh=recorded_window,
    )


def _recorded_primary(payload: dict[str, Any] | None) -> tuple[RectXywh, str] | None:
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        text = str(cand.get("text") or "").strip()
        bbox = cand.get("bbox")
        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        except (TypeError, ValueError):
            continue
        if box[2] <= 0 or box[3] <= 0:
            continue
        return box, text
    return None


def _window_xywh(raw: dict[str, Any] | None) -> RectXywh | None:
    if not isinstance(raw, dict):
        return None
    rect = raw.get("rect")
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        box = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    except (TypeError, ValueError):
        return None
    if box[2] <= 0 or box[3] <= 0:
        return None
    return box


def _other_side_text(request: UiaReplayRepairRequest) -> str:
    from src.recorder.vision_context import load_yolo_ocr_payload

    suffix = "" if request.side == "end" else "_end"
    primary = _recorded_primary(
        load_yolo_ocr_payload(Path(request.recording_dir), request.event_index, suffix=suffix)
    )
    if primary is None:
        return ""
    return primary[1]


def _fallback_other_anchor(instruction: str, request: UiaReplayRepairRequest) -> str:
    """The other drag anchor, taken from the current instruction when YOLO is missing."""
    import re

    names = re.findall(r"「([^」]+)」", instruction or "")
    recorded = request.recorded_text
    for name in names:
        if name != recorded:
            return name
    return ""


def _replace_anchor(
    instruction: str,
    request: UiaReplayRepairRequest,
    label: str,
) -> str | None:
    from src.common.nearby_side import strip_nearby_context_comments

    stripped = strip_nearby_context_comments(instruction or "").strip()
    needle = f"「{request.recorded_text}」"
    if needle not in stripped:
        return None
    return stripped.replace(needle, f"「{label}」", 1)


def _candidate(text: str, bbox: RectXywh) -> dict[str, Any]:
    x, y, w, h = (int(v) for v in bbox)
    return {
        "bbox": [x, y, w, h],
        "center": [x + w // 2, y + h // 2],
        "class_name": "text",
        "text": text,
    }


def _vision(candidates: list[dict[str, Any]], cursor: tuple[int, int]) -> dict[str, Any]:
    return {
        "used_vision": True,
        "candidates": candidates,
        "local_cursor": [int(cursor[0]), int(cursor[1])],
    }


def _hit(label: str, box: RectXywh, source: str) -> UiaReplayRepairHit:
    x, y, w, h = (int(v) for v in box)
    return UiaReplayRepairHit(
        label=label,
        bbox=(x, y, w, h),
        source=source,
        cx=x + w // 2,
        cy=y + h // 2,
    )


def _ocr_confirms(mapped: RectXywh, recorded_text: str, ocr_boxes: list[LabeledBox]) -> bool:
    from cua_mcp.geometry import boxes_overlap
    from cua_mcp.select_mouse_target import (
        _MOUSE_FILTER_SIMILARITY_THRESHOLD,
        _label_similarity,
    )

    for box, text in ocr_boxes:
        if not str(text or "").strip():
            continue
        if not boxes_overlap(mapped, box):
            continue
        if _label_similarity(recorded_text, text) >= _MOUSE_FILTER_SIMILARITY_THRESHOLD:
            return True
    return False


def _point_in_box(x: int, y: int, box: RectXywh) -> bool:
    from cua_mcp.geometry import point_in_rect_xywh

    return point_in_rect_xywh(x, y, box)


def _log(message: str) -> None:
    try:
        from cua_mcp.select_mouse_target import _log_info

        _log_info(message)
    except Exception:
        pass
