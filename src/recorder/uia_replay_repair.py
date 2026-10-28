"""Repair a recorded （YOLO+OCR）pointer step from a live UIA element during cache replay.

Only a side whose stored ``vision_path`` is ``yolo_ocr`` is repaired. The live
FindAll runs beside YOLO+OCR. OCR must still show the recorded target, and one
UIA label must cover that point, before the instruction is rewritten.
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
# (bbox, label) or (bbox, label, uia_control_type)
LabeledBox = tuple[RectXywh, str] | tuple[RectXywh, str, str]


def _labeled_parts(item: LabeledBox) -> tuple[RectXywh, str, str]:
    box = item[0]
    label = str(item[1] or "").strip()
    control_type = str(item[2] or "").strip() if len(item) >= 3 else ""
    return box, label, control_type


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
    control_type: str = ""


@dataclass(frozen=True)
class YoloOcrMoveFallback:
    """One OCR pointer resolve kept after the step instruction is repaired to UIA."""

    instruction: str
    nearby_objects: tuple[str, ...] = ()
    landmark_geometry: dict[str, Any] | None = None


_repair_slots: ContextVar[list[UiaReplayRepairRequest | None] | None] = ContextVar(
    "uia_replay_repair_slots",
    default=None,
)
_yolo_ocr_fallback_slots: ContextVar[list[YoloOcrMoveFallback | None] | None] = ContextVar(
    "yolo_ocr_fallback_slots",
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


@contextmanager
def bind_yolo_ocr_fallbacks(
    slots: list[YoloOcrMoveFallback | None],
) -> Iterator[None]:
    """Queue one OCR fallback per upcoming ``find_mouse_point`` call."""
    token = _yolo_ocr_fallback_slots.set(list(slots))
    try:
        yield
    finally:
        _yolo_ocr_fallback_slots.reset(token)


def take_yolo_ocr_fallback() -> YoloOcrMoveFallback | None:
    """Pop the OCR arguments for this pointer resolve. ``None`` when unset."""
    slots = _yolo_ocr_fallback_slots.get()
    if not slots:
        return None
    return slots.pop(0)


def _yolo_ocr_mode(text: str) -> bool:
    from cua_mcp.instruction_offset import split_vision_mode

    return split_vision_mode(text.strip())[0] == "yolo_ocr"


def yolo_ocr_fallback_payload(value: Any) -> dict[str, Any] | None:
    """Return a stored OCR fallback when it still has a YOLO+OCR sentence and calls."""
    if not isinstance(value, dict):
        return None
    instruction = value.get("instruction")
    calls = value.get("tool_calls")
    if not isinstance(instruction, str) or not _yolo_ocr_mode(instruction):
        return None
    if not isinstance(calls, list) or not calls:
        return None
    return value


def _phrase_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )


def _move_fallback_from_args(
    args: dict[str, Any],
    *,
    instruction_key: str,
    nearby_key: str,
) -> YoloOcrMoveFallback | None:
    instruction = str(args.get(instruction_key) or "").strip()
    if not instruction:
        return None
    geometry = args.get("landmark_geometry")
    return YoloOcrMoveFallback(
        instruction=instruction,
        nearby_objects=_phrase_tuple(args.get(nearby_key)),
        landmark_geometry=geometry if isinstance(geometry, dict) else None,
    )


def remember_yolo_ocr_fallback(
    analysis: dict[str, Any],
    event: RecordedEvent,
    run_dir: Path,
) -> None:
    """Keep the current YOLO+OCR sentence when a later repair replaces it.

    An existing fallback stays. Saving again would store the UIA sentence.
    """
    if yolo_ocr_fallback_payload(analysis.get("yolo_ocr_fallback")) is not None:
        return
    instruction = str(analysis.get("instruction") or "").strip()
    if not _yolo_ocr_mode(instruction):
        return
    from src.recorder.compile_tool_calls import compile_tool_calls

    calls = compile_tool_calls(event, instruction, analysis, run_dir=Path(run_dir))
    if not calls:
        existing = analysis.get("tool_calls")
        calls = existing if isinstance(existing, list) and existing else None
    if not calls:
        return
    analysis["yolo_ocr_fallback"] = {
        "instruction": instruction,
        "tool_calls": calls,
    }


def build_yolo_ocr_fallback_from_vision(
    event: RecordedEvent,
    vision: dict[str, Any] | None,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Compile a durable YOLO+OCR twin from ``vision["yolo_ocr_fallback_vision"]``."""
    if not isinstance(vision, dict):
        return None
    twin = vision.get("yolo_ocr_fallback_vision")
    if not isinstance(twin, dict):
        return None
    twin_candidates = twin.get("candidates")
    if not isinstance(twin_candidates, list) or not twin_candidates:
        return None
    twin_vision = {**twin, "vision_path": "yolo_ocr"}
    twin_vision.pop("vision_path_reason", None)
    destination = vision.get("destination")
    twin_destination: dict[str, Any] | None = None
    if isinstance(destination, dict):
        dest_twin = destination.get("yolo_ocr_fallback_vision")
        if isinstance(dest_twin, dict) and isinstance(dest_twin.get("candidates"), list):
            if dest_twin["candidates"]:
                twin_destination = {**dest_twin, "vision_path": "yolo_ocr"}
                twin_destination.pop("vision_path_reason", None)
    from src.recorder.analyze import rebuild_pointer_instruction
    from src.recorder.compile_tool_calls import compile_tool_calls

    instruction = rebuild_pointer_instruction(
        event,
        twin_vision,
        twin_destination,
        include_nearby=True,
    )
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    if not _yolo_ocr_mode(instruction):
        from cua_mcp.instruction_offset import vision_mode_prefix

        instruction = vision_mode_prefix("yolo_ocr") + instruction.lstrip()
    compile_vision = twin_vision
    if twin_destination is not None:
        compile_vision = {**twin_vision, "destination": twin_destination}
    calls = compile_tool_calls(
        event,
        instruction.strip(),
        vision=compile_vision,
        run_dir=Path(run_dir) if run_dir is not None else None,
    )
    if not calls:
        return None
    return {
        "instruction": instruction.strip(),
        "tool_calls": calls,
    }


def backfill_yolo_ocr_fallbacks(run_dir: Path) -> int:
    """Store a report.json YOLO+OCR line onto a repaired analysis that lost it.

    Report lines follow instruction-event order. Replay reads the analysis field.
    """
    root = Path(run_dir)
    report = read_json(root / "report.json", {})
    lines = report.get("instructions") if isinstance(report, dict) else None
    if not isinstance(lines, list):
        return 0
    from src.common.script_helper import _recording_event_json_paths
    from src.recorder.compile_tool_calls import compile_tool_calls

    updated = 0
    step = 0
    for event_path in _recording_event_json_paths(root):
        raw_event = read_json(event_path, None)
        if not isinstance(raw_event, dict) or not isinstance(raw_event.get("index"), int):
            continue
        event_index = int(raw_event["index"])
        analysis_path = root / "analysis" / f"event_{event_index:03d}.json"
        analysis = read_json(analysis_path, None)
        if not isinstance(analysis, dict):
            continue
        instruction = analysis.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        report_line = lines[step] if step < len(lines) else None
        step += 1
        if yolo_ocr_fallback_payload(analysis.get("yolo_ocr_fallback")) is not None:
            continue
        if not isinstance(analysis.get("uia_replay_repair"), dict):
            continue
        if not isinstance(report_line, str) or not _yolo_ocr_mode(report_line):
            continue
        try:
            event = RecordedEvent.from_dict(raw_event)
        except (KeyError, TypeError, ValueError):
            continue
        calls = compile_tool_calls(event, report_line.strip(), analysis, run_dir=root)
        if not calls:
            continue
        analysis["yolo_ocr_fallback"] = {
            "instruction": report_line.strip(),
            "tool_calls": calls,
        }
        write_json(analysis_path, analysis)
        updated += 1
    return updated


def yolo_ocr_fallback_slots(
    recording_dir: Path,
    event_index: int,
    calls: list[dict[str, Any]],
) -> list[YoloOcrMoveFallback | None]:
    """Slots aligned with ``find_mouse_point`` calls for this step's OCR fallback."""
    analysis = read_json(
        Path(recording_dir) / "analysis" / f"event_{int(event_index):03d}.json",
        None,
    )
    payload = yolo_ocr_fallback_payload(
        analysis.get("yolo_ocr_fallback") if isinstance(analysis, dict) else None
    )
    if payload is None:
        return []
    stored = [item for item in payload["tool_calls"] if isinstance(item, dict)]
    cursor = 0

    def take_named(name: str) -> dict[str, Any] | None:
        nonlocal cursor
        while cursor < len(stored):
            item = stored[cursor]
            cursor += 1
            if str(item.get("name") or "") == name:
                return item
        return None

    slots: list[YoloOcrMoveFallback | None] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        if name == "move_mouse":
            found = take_named("move_mouse")
            args = found.get("arguments") if isinstance(found, dict) else None
            slots.append(
                _move_fallback_from_args(
                    args, instruction_key="instruction", nearby_key="nearby_objects"
                )
                if isinstance(args, dict)
                else None
            )
        elif name == "drag":
            found = take_named("drag")
            args = found.get("arguments") if isinstance(found, dict) else None
            if not isinstance(args, dict):
                slots.extend([None, None])
                continue
            slots.append(
                _move_fallback_from_args(
                    args,
                    instruction_key="start_instruction",
                    nearby_key="start_nearby_objects",
                )
            )
            slots.append(
                _move_fallback_from_args(
                    args,
                    instruction_key="destination_instruction",
                    nearby_key="destination_nearby_objects",
                )
            )
    if not any(slot is not None for slot in slots):
        return []
    return slots


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

    containing: list[tuple[RectXywh, str, str]] = []
    for item in uia_boxes:
        box, label, control_type = _labeled_parts(item)
        if not is_usable_uia_landmark_label(label):
            continue
        if not _point_in_box(center[0], center[1], box):
            continue
        containing.append((box, label, control_type))
    if containing:
        names = {label for _box, label, _control in containing}
        if len(names) != 1:
            return None
        box, label, control_type = min(
            containing, key=lambda item: int(item[0][2]) * int(item[0][3])
        )
        return _hit(label, box, "findall", control_type=control_type)

    if point_probe_fn is None:
        return None
    probed = point_probe_fn(int(center[0]), int(center[1]))
    if not isinstance(probed, tuple) or len(probed) not in (2, 3):
        return None
    label = str(probed[0] or "").strip()
    raw_box = probed[1]
    control_type = str(probed[2] or "").strip() if len(probed) >= 3 else ""
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
    return _hit(label, box, "element_from_point", control_type=control_type)


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
        item
        for item in uia_boxes
        if _labeled_parts(item)[1] != hit.label or _labeled_parts(item)[0] != hit.bbox
    ]
    try:
        write_uia_replay_repair(
            request,
            label=hit.label,
            source=hit.source,
            bbox=hit.bbox,
            others=others,
            control_type=hit.control_type,
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
    control_type: str = "",
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
    instruction = _instruction_from_hit(
        event,
        request,
        label,
        bbox,
        others,
        analysis,
        control_type=control_type,
    )
    if not instruction:
        return None
    remember_yolo_ocr_fallback(analysis, event, run_dir)
    from src.recorder.vision_context import mark_recorded_side_repaired_to_uia

    mark_recorded_side_repaired_to_uia(run_dir, request.event_index, request.side)
    analysis["instruction"] = instruction
    block = analysis.get("uia_replay_repair")
    if not isinstance(block, dict):
        block = {}
    side_block: dict[str, Any] = {"label": label, "source": source}
    if control_type:
        side_block["control_type"] = control_type
    block[request.side] = side_block
    block["label"] = label
    block["source"] = source
    if control_type:
        block["control_type"] = control_type
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
        vision=_vision_for_side(
            request.side, label, bbox, others, control_type=control_type
        ),
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
    *,
    control_type: str = "",
) -> str | None:
    from src.recorder.analyze import rebuild_pointer_instruction

    primary = _candidate(label, bbox, control_type=control_type)
    extra = []
    for item in others:
        box, text, other_type = _labeled_parts(item)
        if not text or (text == label and box == bbox):
            continue
        extra.append(_candidate(text, box, control_type=other_type))
    repaired = _vision([primary, *extra], (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
    from src.recorder.vision_context import VISION_PATH_REPAIR_REASON

    repaired["vision_path"] = "uia"
    repaired["vision_path_reason"] = VISION_PATH_REPAIR_REASON
    if event.kind == "drag":
        other_text = _other_side_text(request)
        if not other_text:
            other_text = _fallback_other_anchor(
                str(analysis.get("instruction") or ""), request
            )
        if not other_text:
            return _replace_anchor(str(analysis.get("instruction") or ""), request, label)
        other = _vision([_candidate(other_text, (0, 0, 8, 8))], (4, 4))
        other_side = "start" if request.side == "end" else "end"
        _copy_stored_vision_mode(request, other, other_side)
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
    *,
    control_type: str = "",
) -> dict[str, Any]:
    primary = _candidate(label, bbox, control_type=control_type)
    extra = [
        _candidate(text, box, control_type=other_type)
        for box, text, other_type in (_labeled_parts(item) for item in others)
        if text
    ]
    vision = _vision(
        [primary, *extra],
        (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2),
    )
    if side == "end":
        return {
            "candidates": [_candidate(label, bbox, control_type=control_type)],
            "used_vision": True,
        }
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
    payload = load_yolo_ocr_payload(run_dir, event_index, suffix=suffix)
    if not isinstance(payload, dict) or payload.get("vision_path") != "yolo_ocr":
        return None
    primary = _recorded_primary(payload)
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


def _first_icon_chinese_id(cand: dict[str, Any]) -> str:
    icons = cand.get("icons")
    if not isinstance(icons, list):
        return ""
    for icon in icons:
        if not isinstance(icon, dict):
            continue
        chinese_id = str(icon.get("chinese_id") or "").strip()
        if chinese_id:
            return chinese_id
    return ""


def _class_repair_label(class_name: str) -> str:
    from cua_mcp.select_ui_element import _ANCHOR_SUFFIX_BY_CLASS

    if class_name not in ("input", "scrollbar"):
        return ""
    return str(_ANCHOR_SUFFIX_BY_CLASS.get(class_name) or "").strip()


def _parse_candidate_bbox(cand: dict[str, Any]) -> RectXywh | None:
    bbox = cand.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except (TypeError, ValueError):
        return None
    if box[2] <= 0 or box[3] <= 0:
        return None
    return box


def _recorded_primary(payload: dict[str, Any] | None) -> tuple[RectXywh, str] | None:
    if not isinstance(payload, dict):
        return None
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    from cua_mcp.icon_map import text_is_pua_only

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        box = _parse_candidate_bbox(cand)
        if box is None:
            continue
        class_label = _class_repair_label(str(cand.get("class_name") or "").strip())
        if class_label:
            return box, class_label
        text = str(cand.get("text") or "").strip()
        if not text:
            continue
        if text_is_pua_only(text):
            chinese_id = _first_icon_chinese_id(cand)
            if chinese_id:
                return box, chinese_id
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


_CLASS_REPAIR_LABELS = frozenset({"輸入欄", "滾動條"})


def _replace_anchor(
    instruction: str,
    request: UiaReplayRepairRequest,
    label: str,
) -> str | None:
    from src.common.nearby_side import strip_nearby_context_comments

    stripped = strip_nearby_context_comments(instruction or "").strip()
    recorded = str(request.recorded_text or "").strip()
    replacement = f"「{label}」"
    needle = f"「{recorded}」"
    if needle in stripped:
        return stripped.replace(needle, replacement, 1)
    if recorded in _CLASS_REPAIR_LABELS and recorded in stripped:
        return stripped.replace(recorded, f"{replacement}文字", 1)
    return None


def _candidate(
    text: str,
    bbox: RectXywh,
    *,
    control_type: str = "",
) -> dict[str, Any]:
    x, y, w, h = (int(v) for v in bbox)
    payload: dict[str, Any] = {
        "bbox": [x, y, w, h],
        "center": [x + w // 2, y + h // 2],
        "class_name": "text",
        "text": text,
    }
    token = str(control_type or "").strip()
    if token:
        from src.recorder.uia_input_boxes import UIA_CONTROL_TYPE_SUFFIX

        if token in UIA_CONTROL_TYPE_SUFFIX:
            payload["uia_control_type"] = token
    return payload


def _copy_stored_vision_mode(
    request: UiaReplayRepairRequest,
    vision: dict[str, Any],
    side: str,
) -> None:
    """Keep the other drag side on the path already stored in its yolo_ocr file."""
    from src.recorder.vision_context import (
        _copy_recorded_vision_mode,
        load_yolo_ocr_payload,
    )

    suffix = "_end" if side == "end" else ""
    _copy_recorded_vision_mode(
        load_yolo_ocr_payload(Path(request.recording_dir), request.event_index, suffix=suffix),
        vision,
    )


def _vision(candidates: list[dict[str, Any]], cursor: tuple[int, int]) -> dict[str, Any]:
    return {
        "used_vision": True,
        "candidates": candidates,
        "local_cursor": [int(cursor[0]), int(cursor[1])],
    }


def _hit(
    label: str,
    box: RectXywh,
    source: str,
    *,
    control_type: str = "",
) -> UiaReplayRepairHit:
    x, y, w, h = (int(v) for v in box)
    token = str(control_type or "").strip()
    if token:
        from src.recorder.uia_input_boxes import UIA_CONTROL_TYPE_SUFFIX

        if token not in UIA_CONTROL_TYPE_SUFFIX:
            token = ""
    return UiaReplayRepairHit(
        label=label,
        bbox=(x, y, w, h),
        source=source,
        cx=x + w // 2,
        cy=y + h // 2,
        control_type=token,
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
