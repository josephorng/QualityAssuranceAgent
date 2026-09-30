from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.io_utils import write_json
from src.common.runtime_context import SCRIPT_PATH_ENV, SMART_GOAL_ENV, SMART_MODE_ENV

_REPORT_VERSION = 1
_RUNTIME_COMMANDS_NAME = "runtime_commands.txt"
_SMART_EVENTS_NAME = "smart_events.jsonl"
_SMART_STATE_NAME = "smart_state.json"
_QUEUE_SCRIPT_LOG_MARKER = "Queue starting coordinator for "
_RUN_FOLDER_TS_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_(\d{8})_(\d{6})_\d+$")
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"
_ROLE_TOOL = "tool"


def should_write_session_report(
    *,
    script_finished: bool,
    user_continues_runtime: bool,
) -> bool:
    """Return whether the hub should write ``report.json`` after a worker finishes."""
    if script_finished and user_continues_runtime:
        return False
    return True


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    return round(max(0.0, delta), 3)


def _parse_step_filename(path: Path) -> tuple[int, int] | None:
    stem = path.stem
    if "_" not in stem:
        return None
    left, right = stem.split("_", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _load_runtime_goals(run_root: Path) -> list[str]:
    path = run_root / _RUNTIME_COMMANDS_NAME
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_step_files(run_root: Path) -> list[tuple[int, int, dict[str, Any]]]:
    steps_dir = run_root / "steps"
    if not steps_dir.is_dir():
        return []
    loaded: list[tuple[int, int, dict[str, Any]]] = []
    for path in sorted(steps_dir.glob("*.json")):
        key = _parse_step_filename(path)
        if key is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            loaded.append((key[0], key[1], payload))
    loaded.sort(key=lambda item: (item[0], item[1]))
    return loaded


def _resolve_goal(
    transcript_counter: int,
    script_step_index: int,
    step_timing: dict[str, Any],
    runtime_goals: list[str],
) -> str:
    goal = step_timing.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    if script_step_index == 0 and 0 <= transcript_counter < len(runtime_goals):
        return runtime_goals[transcript_counter]
    return ""


def _tool_payload_from_message(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


_MOVE_MOUSE_TIMING_ACTIONS = frozenset(
    {"move_mouse", "move_mouse_visual", "check_object_exists"}
)
_DRAG_TIMING_ACTION = "drag"
_TOOL_TIMING_ACTIONS = _MOVE_MOUSE_TIMING_ACTIONS | {_DRAG_TIMING_ACTION}

_MOVE_MOUSE_PHASE_LABELS = {
    "capture": "Screenshot capture",
    "yolo": "YOLO detect",
    "line_refine": "Input line refine",
    "ocr": "OCR",
    "parse_instruction": "Parse instruction (overlapped)",
    "select": "Target select",
    "select_unique": "Unique target (skip LLM)",
    "llm_pick": "LLM target pick",
    "visual_one_pass": "Visual one-pass select",
    "gemma_roi": "Gemma ROI fallback",
    "hand_move": "Cursor move",
}


def _structured_timing(raw: Any) -> dict[str, Any] | None:
    """Accept a tool ``timing`` object that has phases or a total."""
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("phases"), list) or isinstance(raw.get("total_s"), (int, float)):
        return raw
    return None


def _move_mouse_timing_from_tool_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pull structured ``timing`` from a move_mouse-family tool result payload."""
    action = payload.get("action")
    args = payload.get("args")
    timing: Any = None
    if isinstance(args, dict) and isinstance(args.get("timing"), dict):
        timing = args.get("timing")
    elif isinstance(payload.get("timing"), dict):
        timing = payload.get("timing")
    timing = _structured_timing(timing)
    if timing is None:
        return None
    if isinstance(action, str) and action not in _MOVE_MOUSE_TIMING_ACTIONS:
        # Still accept when timing is nested under args from a move-family tool merge.
        if not (isinstance(args, dict) and "timing" in args):
            return None
    return timing


def _details_from_move_mouse_timing(timing: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert move_mouse ``timing.phases`` into time_profile detail rows."""
    phases = timing.get("phases")
    details: list[dict[str, Any]] = []
    fallback_roi = _normalize_roi_xywh(timing.get("ocr_roi"))
    fallback_rois = _normalize_roi_list(timing.get("ocr_rois"))
    if isinstance(phases, list):
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            name = phase.get("name")
            seconds = phase.get("seconds")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(seconds, (int, float)):
                continue
            label = _MOVE_MOUSE_PHASE_LABELS.get(name, name)
            if phase.get("overlapped") is True and "(overlapped)" not in label:
                label = f"{label} (overlapped)"
            detail: dict[str, Any] = {
                "kind": f"move_mouse_{name}",
                "label": label,
                "duration_seconds": round(float(seconds), 3),
            }
            if name in {"yolo", "ocr"}:
                _attach_roi_fields(
                    detail,
                    phase,
                    fallback_roi=fallback_roi,
                    fallback_rois=fallback_rois,
                )
            details.append(detail)
    if details:
        return details
    for key, label in (
        ("capture_s", "Screenshot capture"),
        ("yolo_s", "YOLO detect"),
        ("line_s", "Input line refine"),
        ("ocr_s", "OCR"),
        ("parse_s", "Parse instruction (overlapped)"),
        ("select_s", "Target select"),
        ("hand_move_s", "Cursor move"),
    ):
        raw = timing.get(key)
        if isinstance(raw, (int, float)):
            detail = {
                "kind": f"move_mouse_{key.removesuffix('_s')}",
                "label": label,
                "duration_seconds": round(float(raw), 3),
            }
            phase_name = key.removesuffix("_s")
            if phase_name in {"yolo", "ocr"}:
                _attach_roi_fields(
                    detail,
                    timing,
                    fallback_roi=fallback_roi,
                    fallback_rois=fallback_rois,
                )
            details.append(detail)
    return details


def _normalize_roi_xywh(raw: Any) -> list[int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [int(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _normalize_roi_list(raw: Any) -> list[list[int]] | None:
    if not isinstance(raw, list):
        return None
    rois = [
        roi
        for item in raw
        for roi in (_normalize_roi_xywh(item),)
        if roi is not None
    ]
    return rois or None


def _attach_roi_fields(
    detail: dict[str, Any],
    source: dict[str, Any],
    *,
    fallback_roi: list[int] | None = None,
    fallback_rois: list[list[int]] | None = None,
) -> None:
    """Copy image-local OCR/YOLO ROI onto a time_profile detail row."""
    roi = _normalize_roi_xywh(source.get("ocr_roi")) or fallback_roi
    rois = _normalize_roi_list(source.get("ocr_rois")) or fallback_rois
    if roi is not None:
        detail["roi"] = roi
    if rois is not None:
        detail["rois"] = rois
        if "roi" not in detail and len(rois) == 1:
            detail["roi"] = rois[0]
    pad = source.get("ocr_roi_pad")
    if isinstance(pad, (int, float)):
        detail["roi_pad"] = int(pad)


def _ocr_roi_from_click_window_payload(payload: dict[str, Any]) -> list[int] | None:
    """Best-effort image-local ROI from ``click_window`` + screenshot for older runs."""
    args = payload.get("args")
    if not isinstance(args, dict):
        return None
    click_window = args.get("click_window")
    if not isinstance(click_window, dict):
        return None
    path_raw = args.get("screenshot_path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        paths = args.get("screenshot_paths")
        if isinstance(paths, list):
            path_raw = next(
                (p for p in paths if isinstance(p, str) and p.strip()),
                None,
            )
    if not isinstance(path_raw, str) or not path_raw.strip():
        return None
    path = Path(path_raw)
    if not path.is_file():
        return None
    try:
        from src.common.io_utils import imread_bgr
        from src.recorder.window_snapshot import resolve_ocr_roi_local

        bgr = imread_bgr(path)
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return None
        img_h, img_w = bgr.shape[:2]
        roi = resolve_ocr_roi_local(click_window, image_w=img_w, image_h=img_h)
    except Exception:
        return None
    return _normalize_roi_xywh(roi)


def _find_tool_payload_for_action(
    messages: list[dict[str, Any]],
    start_index: int,
    action: str,
) -> dict[str, Any] | None:
    for message in messages[start_index + 1 :]:
        if not isinstance(message, dict) or message.get("role") != _ROLE_TOOL:
            continue
        payload = _tool_payload_from_message(message.get("content"))
        if payload.get("action") == action:
            return payload
    return None


def _prefix_phase_details(
    details: list[dict[str, Any]],
    *,
    kind_prefix: str,
    label_prefix: str,
) -> list[dict[str, Any]]:
    """Relabel move-mouse phase rows as one side of a drag."""
    prefixed: list[dict[str, Any]] = []
    for detail in details:
        item = dict(detail)
        kind = item.get("kind")
        phase = kind.removeprefix("move_mouse_") if isinstance(kind, str) else "phase"
        item["kind"] = f"{kind_prefix}_{phase}"
        label = item.get("label")
        if isinstance(label, str) and label.strip():
            item["label"] = f"{label_prefix} · {label}"
        prefixed.append(item)
    return prefixed


def _target_timing(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Pull ``timing`` from ``args.<key>`` or a top-level target object."""
    args = payload.get("args")
    target = args.get(key) if isinstance(args, dict) else None
    if not isinstance(target, dict):
        target = payload.get(key)
    if not isinstance(target, dict):
        return None
    return _structured_timing(target.get("timing"))


def _details_from_drag_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], float | None]:
    """Start and destination resolve phases, in that order."""
    details: list[dict[str, Any]] = []
    totals: list[float] = []
    for key, kind_prefix, label_prefix in (
        ("start_target", "drag_start", "起點"),
        ("destination_target", "drag_destination", "終點"),
    ):
        timing = _target_timing(payload, key)
        if timing is None:
            continue
        details.extend(
            _prefix_phase_details(
                _details_from_move_mouse_timing(timing),
                kind_prefix=kind_prefix,
                label_prefix=label_prefix,
            )
        )
        total = timing.get("total_s")
        if isinstance(total, (int, float)):
            totals.append(float(total))
    internal = round(sum(totals), 3) if totals else None
    return details, internal


def _apply_click_window_roi_fallback(
    details: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    roi_kinds: set[str],
) -> None:
    needs_roi = any(
        isinstance(d, dict)
        and d.get("kind") in roi_kinds
        and "roi" not in d
        and "rois" not in d
        for d in details
    )
    if not needs_roi:
        return
    fallback_roi = _ocr_roi_from_click_window_payload(payload)
    if fallback_roi is None:
        return
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if detail.get("kind") not in roi_kinds:
            continue
        if "roi" in detail or "rois" in detail:
            continue
        detail["roi"] = fallback_roi


def _attach_move_mouse_timing_details(
    entry: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    message_index: int,
    next_message: dict[str, Any] | None,
) -> None:
    """Attach nested move_mouse or drag phase timings onto a tool_execution row."""
    if entry.get("kind") != "tool_execution":
        return
    actions = entry.get("actions")
    action_names = (
        [str(a) for a in actions if isinstance(a, str)]
        if isinstance(actions, list)
        else []
    )
    target_action = next(
        (name for name in action_names if name in _TOOL_TIMING_ACTIONS),
        None,
    )
    if target_action is None:
        raw_action = entry.get("action")
        if isinstance(raw_action, str) and raw_action in _TOOL_TIMING_ACTIONS:
            target_action = raw_action
    if target_action is None:
        return

    payload: dict[str, Any] | None = None
    if isinstance(next_message, dict) and next_message.get("role") == _ROLE_TOOL:
        next_payload = _tool_payload_from_message(next_message.get("content"))
        if next_payload.get("action") == target_action:
            payload = next_payload
    if payload is None:
        payload = _find_tool_payload_for_action(messages, message_index, target_action)
    if payload is None:
        return
    if target_action == _DRAG_TIMING_ACTION:
        details, internal = _details_from_drag_payload(payload)
        if not details:
            return
        entry["details"] = details
        if internal is not None:
            entry["tool_internal_seconds"] = internal
        return
    timing = _move_mouse_timing_from_tool_payload(payload)
    if timing is None:
        return
    details = _details_from_move_mouse_timing(timing)
    if details:
        _apply_click_window_roi_fallback(
            details,
            payload,
            roi_kinds={"move_mouse_yolo", "move_mouse_ocr"},
        )
        entry["details"] = details
        total = timing.get("total_s")
        if isinstance(total, (int, float)):
            entry["tool_internal_seconds"] = round(float(total), 3)


def _message_has_screenshots(message: dict[str, Any]) -> bool:
    images = message.get("images")
    return isinstance(images, list) and bool(images)


def _describe_tool_interval(
    message: dict[str, Any],
    next_message: dict[str, Any] | None,
) -> dict[str, Any]:
    """Label the interval after a tool result based on what happens next."""
    payload = _tool_payload_from_message(message.get("content"))
    entry: dict[str, Any] = {}
    action = payload.get("action")
    if isinstance(action, str) and action:
        entry["action"] = action
    if "ok" in payload:
        entry["ok"] = bool(payload.get("ok"))

    if isinstance(next_message, dict):
        if next_message.get("role") == _ROLE_USER and _message_has_screenshots(next_message):
            entry.update(
                {
                    "kind": "screenshot_capture",
                    "label": "Screenshot capture and prompt prep for next decision",
                }
            )
            return entry
        if next_message.get("role") == _ROLE_TOOL:
            next_payload = _tool_payload_from_message(next_message.get("content"))
            next_action = next_payload.get("action")
            actions: list[str] = []
            if isinstance(next_action, str) and next_action.strip():
                actions = [next_action.strip()]
            entry.update(
                {
                    "kind": "tool_execution",
                    "label": (
                        f"Hand tool execution: {', '.join(actions)}"
                        if actions
                        else "Hand tool execution"
                    ),
                }
            )
            if actions:
                entry["actions"] = actions
                entry["action"] = actions[0]
            if "ok" in next_payload:
                entry["ok"] = bool(next_payload.get("ok"))
            return entry

    entry.update(
        {
            "kind": "step_wrap_up",
            "label": "Step wrap-up after final tool (settle / verify prep)",
        }
    )
    return entry


def _extract_assistant_tool_names(message: dict[str, Any]) -> list[str]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        func = call.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _describe_time_profile_entry(
    message: dict[str, Any],
    *,
    for_verify: bool = False,
    next_message: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Map a stamped transcript message to a human-readable phase description.

    Each profile row covers the interval starting at this message's timestamp until
    the next message (or step end):
    - user: screenshots were sent; duration is LLM inference.
    - assistant with tool_calls: LLM chose tools; duration is hand execution.
    - assistant without tool_calls: LLM declared step done; duration is wrap-up.
    - tool: interval after a tool result—labeled by what follows (next decide
      screenshot, execution of the next batched tool, or step wrap-up).
    - deferred settle markers: wait before tools, then hand execution after the wait.

    When ``for_verify`` is True, user-role intervals are labeled ``verify_llm_inference``.
    """
    waited_raw = message.get("deferred_settle_wait_seconds")
    if isinstance(waited_raw, (int, float)) and not isinstance(waited_raw, bool):
        waited = float(waited_raw)
        return {
            "kind": "deferred_settle",
            "label": f"Deferred settle waited {waited:.3f}s before next tool",
        }
    if message.get("deferred_settle_done") is True:
        if isinstance(next_message, dict) and next_message.get("role") == _ROLE_TOOL:
            payload = _tool_payload_from_message(next_message.get("content"))
            action = payload.get("action")
            actions: list[str] = []
            if isinstance(action, str) and action.strip():
                actions = [action.strip()]
            entry: dict[str, Any] = {
                "kind": "tool_execution",
                "label": (
                    f"Hand tool execution: {', '.join(actions)}"
                    if actions
                    else "Hand tool execution"
                ),
            }
            if actions:
                entry["actions"] = actions
                entry["action"] = actions[0]
            if "ok" in payload:
                entry["ok"] = bool(payload.get("ok"))
            return entry
        return {
            "kind": "step_prep",
            "label": "After deferred settle before next phase",
        }

    role = message.get("role")
    if role == _ROLE_USER:
        if for_verify:
            return {
                "kind": "verify_llm_inference",
                "label": "Verify LLM response generation after baseline/live screenshots were sent",
            }
        return {
            "kind": "llm_inference",
            "label": "LLM response generation after prompt and screenshots were sent",
        }
    if role == _ROLE_ASSISTANT:
        tool_names = _extract_assistant_tool_names(message)
        if tool_names:
            # Prefer the deferred-settle marker for the wait interval when present.
            if isinstance(next_message, dict) and isinstance(
                next_message.get("deferred_settle_wait_seconds"), (int, float)
            ):
                return {
                    "kind": "step_prep",
                    "label": "Decide complete; deferred settle starts next",
                }
            joined = ", ".join(tool_names)
            return {
                "kind": "tool_execution",
                "label": f"Hand tool execution: {joined}",
                "actions": tool_names,
            }
        if for_verify:
            return {
                "kind": "step_completion",
                "label": "Verify wrap-up after final verify LLM response",
            }
        return {
            "kind": "step_completion",
            "label": "Step completion after final LLM response",
        }
    if role == _ROLE_TOOL:
        return _describe_tool_interval(message, next_message)
    role_label = str(role) if role else "unknown"
    return {"kind": role_label, "label": f"Unhandled message role: {role_label}"}


def _first_message_timestamp(messages: list[dict[str, Any]]) -> datetime | None:
    for message in messages:
        if not isinstance(message, dict):
            continue
        started = _parse_iso(message.get("timestamp_utc"))
        if started is not None:
            return started
    return None


def _build_time_profile(
    messages: list[dict[str, Any]],
    finished_at_utc: str | None,
    *,
    for_verify: bool = False,
) -> list[dict[str, Any]]:
    if not messages:
        return []

    end_boundary = _parse_iso(finished_at_utc)
    profile: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        started = _parse_iso(message.get("timestamp_utc"))
        if started is None:
            continue

        next_message: dict[str, Any] | None = None
        if index + 1 < len(messages):
            candidate = messages[index + 1]
            if isinstance(candidate, dict):
                next_message = candidate
            next_started = _parse_iso(candidate.get("timestamp_utc") if isinstance(candidate, dict) else None)
        else:
            next_started = end_boundary

        duration = _duration_seconds(started, next_started)
        entry: dict[str, Any] = {
            "started_at_utc": started.isoformat(),
            **_describe_time_profile_entry(
                message,
                for_verify=for_verify,
                next_message=next_message,
            ),
        }
        if duration is not None:
            entry["duration_seconds"] = duration
        _attach_move_mouse_timing_details(
            entry,
            messages=messages,
            message_index=index,
            next_message=next_message,
        )

        profile.append(entry)

    return profile


def _sum_profile_durations(profile: list[dict[str, Any]], kinds: set[str]) -> float:
    total = 0.0
    for entry in profile:
        if entry.get("kind") not in kinds:
            continue
        duration = entry.get("duration_seconds")
        if isinstance(duration, (int, float)):
            total += float(duration)
    return round(total, 3)


_KNOWN_TIMING_KINDS = frozenset(
    {
        "llm_inference",
        "verify_llm_inference",
        "tool_execution",
        "screenshot_capture",
        "deferred_settle",
    }
)
# After the last tool, the interval until the next stamped message is often settle /
# verify prep (hand after-action sleep or script settle before verify).
_WAITING_PROFILE_KINDS = frozenset({"step_wrap_up"})
_DEFERRED_SETTLE_PROFILE_KINDS = frozenset({"deferred_settle"})


def _coerce_positive_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    if seconds <= 0:
        return None
    return seconds


def _build_timing_summary(
    timing: dict[str, Any],
    time_profile: list[dict[str, Any]],
    *,
    settle_after_seconds: float | None = None,
) -> dict[str, Any]:
    execution_llm = _sum_profile_durations(time_profile, {"llm_inference"})
    verify_llm = _sum_profile_durations(time_profile, {"verify_llm_inference"})
    tool_execution = _sum_profile_durations(time_profile, {"tool_execution"})
    screenshot = _sum_profile_durations(time_profile, {"screenshot_capture"})
    deferred_settle = _sum_profile_durations(time_profile, _DEFERRED_SETTLE_PROFILE_KINDS)
    if deferred_settle <= 0:
        timed = _coerce_positive_seconds(timing.get("deferred_settle_waited_seconds"))
        if timed is not None:
            deferred_settle = timed
    accounted = execution_llm + verify_llm + tool_execution + screenshot
    other_kinds = {
        str(entry.get("kind"))
        for entry in time_profile
        if entry.get("kind") not in _KNOWN_TIMING_KINDS and entry.get("kind") is not None
    }
    other_from_profile = _sum_profile_durations(time_profile, other_kinds)

    total_raw = timing.get("duration_seconds")
    total: float | None = float(total_raw) if isinstance(total_raw, (int, float)) else None
    if total is None:
        profile_total = accounted + deferred_settle + other_from_profile
        total = round(profile_total, 3) if profile_total > 0 else None

    # Wall-clock remainder holds settle sleeps and other unprofiled gaps.
    raw_other = other_from_profile
    if total is not None:
        raw_other = round(max(0.0, total - accounted), 3)

    settle = _coerce_positive_seconds(settle_after_seconds)
    if settle is None:
        settle = _coerce_positive_seconds(timing.get("settle_after_seconds"))
    wrap_up = _sum_profile_durations(time_profile, _WAITING_PROFILE_KINDS)
    # Explicit deferred-settle waits (executed before tools) count as waiting first.
    # Outbound settle_after on this step is scheduled for a later step when deferred,
    # so only peel settle_after / wrap-up from the leftover when there was no inbound
    # deferred wait (blocking settle-before-verify / wrap-up case).
    remainder_after_deferred = max(0.0, raw_other - deferred_settle)
    if deferred_settle > 0:
        extra_waiting = min(float(wrap_up or 0.0), remainder_after_deferred)
    elif settle is not None:
        extra_waiting = min(float(settle), remainder_after_deferred)
    else:
        extra_waiting = min(float(wrap_up or 0.0), remainder_after_deferred)
    waiting = round(deferred_settle + extra_waiting, 3)
    other = round(max(0.0, raw_other - waiting), 3)

    summary: dict[str, Any] = {
        "execution_llm_seconds": execution_llm,
        "verify_llm_seconds": verify_llm,
        "tool_execution_seconds": tool_execution,
        "screenshot_seconds": screenshot,
        "waiting_seconds": waiting,
        "other_seconds": other,
    }
    if deferred_settle > 0:
        summary["deferred_settle_seconds"] = round(deferred_settle, 3)
    if total is not None:
        summary["total_seconds"] = round(total, 3)
    return summary


def _build_step_records(
    step_files: list[tuple[int, int, dict[str, Any]]],
    runtime_goals: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for transcript_counter, script_step_index, payload in step_files:
        step_timing_raw = payload.get("step_timing")
        step_timing = dict(step_timing_raw) if isinstance(step_timing_raw, dict) else {}
        messages_raw = payload.get("messages")
        messages = [msg for msg in messages_raw if isinstance(msg, dict)] if isinstance(messages_raw, list) else []
        verification_raw = payload.get("verification")
        verification = (
            [msg for msg in verification_raw if isinstance(msg, dict)]
            if isinstance(verification_raw, list)
            else []
        )

        timing = {
            key: step_timing[key]
            for key in (
                "started_at_utc",
                "finished_at_utc",
                "duration_seconds",
                "status",
                "settle_after_seconds",
                "deferred_settle_waited_seconds",
            )
            if key in step_timing
        }

        finished_at = step_timing.get("finished_at_utc")
        verify_started = _first_message_timestamp(verification)
        actor_end = verify_started.isoformat() if verify_started is not None else finished_at
        actor_profile = _build_time_profile(messages, actor_end)
        verify_profile = _build_time_profile(
            verification,
            finished_at if isinstance(finished_at, str) else None,
            for_verify=True,
        )
        time_profile = actor_profile + verify_profile
        settle_after = _coerce_positive_seconds(step_timing.get("settle_after_seconds"))

        record: dict[str, Any] = {
            "transcript_counter": transcript_counter,
            "script_step_index": script_step_index,
            "goal": _resolve_goal(transcript_counter, script_step_index, step_timing, runtime_goals),
            "timing": timing,
            "time_profile": time_profile,
            "timing_summary": _build_timing_summary(
                timing,
                time_profile,
                settle_after_seconds=settle_after,
            ),
        }
        expected_outcome = step_timing.get("expected_outcome")
        if isinstance(expected_outcome, str) and expected_outcome.strip():
            record["expected_outcome"] = expected_outcome.strip()
        elif expected_outcome is None and "expected_outcome" in step_timing:
            record["expected_outcome"] = None
        verify = step_timing.get("verify")
        if isinstance(verify, dict):
            record["verify"] = verify
        elif verify is None and "verify" in step_timing:
            record["verify"] = None
        records.append(record)
    return records


def _parse_csv_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _parse_csv_args(value: str | None) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _find_step_for_timestamp(
    tool_time: datetime,
    step_records: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    for record in step_records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        started = _parse_iso(timing.get("started_at_utc"))
        finished = _parse_iso(timing.get("finished_at_utc"))
        if started is None:
            continue
        if finished is None:
            if tool_time >= started:
                return record.get("transcript_counter"), record.get("script_step_index")
            continue
        if started <= tool_time <= finished:
            return record.get("transcript_counter"), record.get("script_step_index")

    if not step_records:
        return None, None

    nearest = min(
        step_records,
        key=lambda record: abs(
            (
                _parse_iso((record.get("timing") or {}).get("started_at_utc"))
                or datetime.min.replace(tzinfo=timezone.utc)
            )
            - tool_time
        ).total_seconds(),
    )
    return nearest.get("transcript_counter"), nearest.get("script_step_index")


def _load_tool_results(
    run_root: Path,
    step_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hand_csv = run_root / "hand.csv"
    if not hand_csv.is_file() or hand_csv.stat().st_size == 0:
        return []

    results: list[dict[str, Any]] = []
    with hand_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp_raw = row.get("timestamp")
            tool_time = _parse_iso(timestamp_raw if isinstance(timestamp_raw, str) else None)
            if tool_time is None:
                continue
            transcript_counter, script_step_index = _find_step_for_timestamp(tool_time, step_records)
            entry: dict[str, Any] = {
                "timestamp_utc": tool_time.isoformat(),
                "action": row.get("action") or "",
                "args": _parse_csv_args(row.get("args")),
                "ok": _parse_csv_bool(row.get("ok")),
                "message": row.get("message") or "",
                "screenshot_name": row.get("screenshot_name") or "",
            }
            if transcript_counter is not None:
                entry["transcript_counter"] = transcript_counter
            if script_step_index is not None:
                entry["script_step_index"] = script_step_index
            results.append(entry)

    results.sort(key=lambda item: item["timestamp_utc"])
    return results


def _build_summary(
    step_records: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_steps = 0
    total_duration = 0.0
    has_duration = False
    category_totals = {
        "execution_llm_seconds": 0.0,
        "verify_llm_seconds": 0.0,
        "tool_execution_seconds": 0.0,
        "screenshot_seconds": 0.0,
        "waiting_seconds": 0.0,
        "other_seconds": 0.0,
    }
    has_category = False
    for record in step_records:
        timing = record.get("timing")
        if isinstance(timing, dict):
            if timing.get("status") == "failed":
                failed_steps += 1
            duration = timing.get("duration_seconds")
            if isinstance(duration, (int, float)):
                total_duration += float(duration)
                has_duration = True
        timing_summary = record.get("timing_summary")
        if isinstance(timing_summary, dict):
            for key in category_totals:
                value = timing_summary.get(key)
                if isinstance(value, (int, float)):
                    category_totals[key] += float(value)
                    has_category = True

    failed_tools = sum(1 for item in tool_results if not item.get("ok", False))
    summary: dict[str, Any] = {
        "step_count": len(step_records),
        "tool_call_count": len(tool_results),
        "failed_step_count": failed_steps,
        "failed_tool_count": failed_tools,
    }
    if has_duration:
        summary["total_duration_seconds"] = round(total_duration, 3)
        if step_records:
            summary["avg_step_seconds"] = round(total_duration / len(step_records), 3)
    if has_category:
        for key, value in category_totals.items():
            summary[key] = round(value, 3)
    return summary


def _resolve_script_metadata(run_root: Path) -> dict[str, str]:
    if os.environ.get(SMART_MODE_ENV, "").strip().lower() in ("1", "true", "yes"):
        goal = os.environ.get(SMART_GOAL_ENV, "").strip()
        meta: dict[str, str] = {"run_mode": "smart"}
        script_path_raw = os.environ.get(SCRIPT_PATH_ENV, "").strip()
        if script_path_raw:
            path = Path(script_path_raw)
            from src.common.script_helper import script_display_name

            meta["script_path"] = str(path)
            meta["script_name"] = script_display_name(path)
        else:
            meta["script_name"] = "智能模式"
        if goal:
            meta["smart_goal"] = goal
        return meta

    smart_state_path = run_root / _SMART_STATE_NAME
    if smart_state_path.is_file():
        try:
            payload = json.loads(smart_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            meta = {"run_mode": "smart", "script_name": "智能模式"}
            goal = payload.get("goal")
            if isinstance(goal, str) and goal.strip():
                meta["smart_goal"] = goal.strip()
            return meta

    script_path_raw = os.environ.get(SCRIPT_PATH_ENV, "").strip()
    if script_path_raw:
        path = Path(script_path_raw)
        from src.common.script_helper import script_display_name

        return {"script_path": str(path), "script_name": script_display_name(path)}

    log_path = run_root / "run.log"
    if log_path.is_file():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if _QUEUE_SCRIPT_LOG_MARKER in line:
                    name = line.split(_QUEUE_SCRIPT_LOG_MARKER, 1)[1].strip()
                    if name:
                        return {"script_name": name}
        except OSError:
            pass

    if (run_root / _RUNTIME_COMMANDS_NAME).is_file():
        return {"script_name": _RUNTIME_COMMANDS_NAME}

    return {}


def _load_smart_events(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / _SMART_EVENTS_NAME
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    except OSError:
        return []
    return events


def _load_smart_state(run_root: Path) -> dict[str, Any] | None:
    path = run_root / _SMART_STATE_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_smart_cycles(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group Plan/Act/Verify events by cycle for the report UI."""
    by_cycle: dict[int, dict[str, Any]] = {}
    for event in events:
        cycle = event.get("cycle")
        if not isinstance(cycle, int):
            continue
        bucket = by_cycle.setdefault(
            cycle,
            {"cycle": cycle, "plan": None, "act": None, "verify": None, "events": []},
        )
        phase = event.get("phase")
        if phase in ("plan", "act", "verify"):
            bucket[phase] = event
        bucket["events"].append(event)
    return [by_cycle[key] for key in sorted(by_cycle)]


def _resolve_started_at_utc(run_root: Path, step_records: list[dict[str, Any]]) -> str | None:
    earliest: datetime | None = None
    for record in step_records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        started = _parse_iso(timing.get("started_at_utc"))
        if started is None:
            continue
        if earliest is None or started < earliest:
            earliest = started
    if earliest is not None:
        return earliest.isoformat()

    match = _RUN_FOLDER_TS_RE.match(run_root.name)
    if match is None:
        return None
    date_part, time_part = match.groups()
    try:
        return (
            datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
            .replace(tzinfo=timezone.utc)
            .isoformat()
        )
    except ValueError:
        return None


def build_session_report(run_root: Path, *, session_end_reason: str) -> dict[str, Any]:
    """Aggregate step timing and tool results from a run folder into a report dict."""
    step_files = _load_step_files(run_root)
    runtime_goals = _load_runtime_goals(run_root)
    step_records = _build_step_records(step_files, runtime_goals)
    tool_results = _load_tool_results(run_root, step_records)
    script_meta = _resolve_script_metadata(run_root)
    started_at_utc = _resolve_started_at_utc(run_root, step_records)
    smart_events = _load_smart_events(run_root)
    smart_state = _load_smart_state(run_root)
    smart_cycles = _build_smart_cycles(smart_events)

    report: dict[str, Any] = {
        "version": _REPORT_VERSION,
        "run_id": run_root.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_end_reason": session_end_reason,
        "summary": _build_summary(step_records, tool_results),
        "steps": step_records,
        "tool_results": tool_results,
    }
    if script_meta:
        report.update(script_meta)
    if started_at_utc is not None:
        report["started_at_utc"] = started_at_utc
    if smart_events:
        report["smart_events"] = smart_events
        report["smart_cycles"] = smart_cycles
    if smart_state is not None:
        report["smart_state"] = smart_state
        summary = report.get("summary")
        if isinstance(summary, dict):
            summary["smart_cycle_count"] = len(smart_cycles)
            terminal = smart_state.get("terminal_reason")
            if isinstance(terminal, str) and terminal:
                summary["smart_terminal_reason"] = terminal
    return report


def write_session_report(run_root: Path, *, session_end_reason: str) -> Path:
    """Write ``report.json`` under ``run_root`` and rebuild ``session_steps.html``; return report path."""
    report_path = run_root / "report.json"
    write_json(report_path, build_session_report(run_root, session_end_reason=session_end_reason))

    from src.common.session_html import write_session_html_from_run

    write_session_html_from_run(run_root)
    return report_path
