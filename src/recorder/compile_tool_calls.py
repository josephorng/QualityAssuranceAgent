"""Deterministic RecordedEvent + instruction → MCP tool_calls for replay cache."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.common.instruction_tool_cache import normalize_instruction
from src.common.io_utils import read_json, write_json
from src.common.nearby_side import (
    extract_nearby_hints_by_location,
    extract_nearby_hints_from_instruction,
    merge_nearby_hints,
    nearby_hints_to_phrases,
    strip_nearby_context_comments,
)
from cua_mcp.instruction_offset import split_vision_mode, vision_mode_prefix
from src.recorder.analyze import typed_text_from_instruction
from src.recorder.landmark_geometry import build_landmark_geometry
from src.recorder.models import RecordedEvent, event_json_path
from src.recorder.to_cache import validate_tool_calls
from src.recorder.window_snapshot import should_ignore_window_change

RECORDING_TOOL_CACHE_FILENAME = "instruction_tool_cache.json"
_CACHE_VERSION = 1

_CLICK_POINTER_KINDS = frozenset(
    {"click", "double_click", "triple_click", "right_click", "middle_click", "hold"}
)
_WINDOW_INSTRUCTION_RE = re.compile(
    r"^(最小化|最大化|關閉)「([^」]+)」視窗"
)
_WAIT_INSTRUCTION_RE = re.compile(r"^等待\s*([0-9]+(?:\.[0-9]+)?)\s*秒")
_DRAG_SPLIT_RE = re.compile(r"^從(.+?)拖到(.+)$")
_CLICK_ACTION_SUFFIX_RE = re.compile(r"(，(?:並|用).+。)$")
_MOVE_PREFIX = "將滑鼠移到"
_SCROLL_HOVER_RE = re.compile(r"^在(.+?)附近(?:向上|向下)捲動")


def recording_tool_cache_path(run_dir: Path) -> Path:
    return Path(run_dir) / RECORDING_TOOL_CACHE_FILENAME


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": dict(arguments)}


def _validated(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not calls:
        return None
    if validate_tool_calls(calls) is not None:
        return None
    return calls


def _key_token(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    # ASCII control chars from Ctrl+letter capture → letter
    if len(cleaned) == 1 and not cleaned.isprintable():
        code = ord(cleaned)
        if 1 <= code <= 26:
            return chr(ord("a") + code - 1)
    lower = cleaned.lower()
    aliases = {
        "return": "enter",
        "escape": "esc",
        "control": "ctrl",
        "cmd": "win",
        "command": "win",
        "page_up": "pageup",
        "page_down": "pagedown",
    }
    return aliases.get(lower, lower)


def _modifier_list(modifiers: list[str] | None) -> list[str] | None:
    if not modifiers:
        return None
    out: list[str] = []
    for item in modifiers:
        token = _key_token(str(item))
        if token:
            out.append(token)
    return out or None


def _compile_text_input(event: RecordedEvent, instruction: str) -> list[dict[str, Any]] | None:
    text = typed_text_from_instruction(instruction)
    if text is None and isinstance(event.text, str):
        text = event.text.strip() or None
    if not text:
        return None
    return [_call("type_text", text=text, instruction=instruction)]


def _compile_key_press(event: RecordedEvent, instruction: str) -> list[dict[str, Any]] | None:
    key = _key_token(event.key)
    if not key:
        return None
    return [_call("press_key", key=key, instruction=instruction)]


def _compile_hotkey(event: RecordedEvent, instruction: str) -> list[dict[str, Any]] | None:
    raw_keys = event.keys or []
    keys: list[str] = []
    for item in raw_keys:
        token = _key_token(str(item))
        if not token:
            return None
        keys.append(token)
    if not keys:
        return None
    return [_call("hotkey", keys=keys, instruction=instruction)]


def _compile_wait(event: RecordedEvent, instruction: str) -> list[dict[str, Any]] | None:
    seconds: float | None = None
    if event.duration_seconds is not None:
        seconds = float(event.duration_seconds)
    else:
        match = _WAIT_INSTRUCTION_RE.match(instruction.strip())
        if match:
            seconds = float(match.group(1))
    if seconds is None or seconds <= 0:
        return None
    return [_call("wait", seconds=seconds, instruction=instruction)]


def _title_matches_process_lookup(entry_title: str, target_title: str) -> bool:
    left = (entry_title or "").strip()
    right = (target_title or "").strip()
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _process_name_from_window_rows(
    rows: Any, title: str
) -> str | None:
    if not isinstance(rows, list):
        return None
    for item in rows:
        if not isinstance(item, dict):
            continue
        if not _title_matches_process_lookup(str(item.get("title") or ""), title):
            continue
        process = str(item.get("process_name") or "").strip()
        if process:
            return process
    return None


def _resolve_process_name_for_window_change(
    event: RecordedEvent,
    title: str,
    change: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> str | None:
    """Prefer window_change.process_name; else click_window / verify / snapshot."""
    direct = str(change.get("process_name") or "").strip()
    if direct:
        return direct

    click_window = event.click_window if isinstance(event.click_window, dict) else None
    if click_window is not None:
        click_title = str(click_window.get("title") or "").strip()
        click_process = str(click_window.get("process_name") or "").strip()
        if click_process and (
            not click_title or _title_matches_process_lookup(click_title, title)
        ):
            return click_process

    if isinstance(analysis, dict):
        verify = analysis.get("window_verify")
        if isinstance(verify, dict):
            for key in ("disappeared", "state", "appeared"):
                found = _process_name_from_window_rows(verify.get(key), title)
                if found:
                    return found

    debug = (
        event.window_snapshot_debug
        if isinstance(event.window_snapshot_debug, dict)
        else None
    )
    if debug is not None:
        for key in ("windows_before", "windows_after"):
            found = _process_name_from_window_rows(debug.get(key), title)
            if found:
                return found
    return None


def _compile_window_change(
    event: RecordedEvent,
    instruction: str,
    analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    change = event.window_change if isinstance(event.window_change, dict) else {}
    action = str(change.get("action") or "").strip()
    title = str(change.get("title") or "").strip()
    if not action or not title:
        match = _WINDOW_INSTRUCTION_RE.match(instruction.strip())
        if not match:
            return None
        verb, title = match.group(1), match.group(2)
        action = {"最小化": "minimize", "最大化": "maximize", "關閉": "close"}[verb]
    # Flyout dismissals are never close_windows / chrome tools.
    if should_ignore_window_change({"title": title, "action": action}):
        return None
    tool_by_action = {
        "minimize": "minimize_windows",
        "maximize": "maximize_windows",
        "close": "close_windows",
    }
    tool = tool_by_action.get(action)
    if tool is None:
        return None
    args: dict[str, Any] = {
        "window_title_contains": title,
        "instruction": instruction,
    }
    process_name = _resolve_process_name_for_window_change(
        event, title, change, analysis
    )
    if process_name:
        args["process_name"] = process_name
    return [_call(tool, **args)]


def _effective_left_click_count(kind: str, click_count: int | None) -> int | None:
    if click_count is not None and int(click_count) > 0:
        return int(click_count)
    if kind == "click":
        return 1
    if kind == "double_click":
        return 2
    if kind == "triple_click":
        return 3
    return None


def _short_move_target(instruction: str) -> str:
    """Hub-script line → move_mouse primary target (no prefix/suffix/nearby)."""
    mode, body = split_vision_mode(instruction)
    text = strip_nearby_context_comments(body).strip()
    text = _CLICK_ACTION_SUFFIX_RE.sub("", text).strip().rstrip("，").strip()
    if text.startswith(_MOVE_PREFIX):
        text = text[len(_MOVE_PREFIX) :].strip()
    prefix = vision_mode_prefix(mode)
    return f"{prefix}{text}" if prefix else text


def _nearby_object_phrases(instruction: str) -> list[str] | None:
    phrases = nearby_hints_to_phrases(extract_nearby_hints_from_instruction(instruction))
    return phrases or None


def _move_mouse_call(
    instruction: str,
    *,
    target: str | None = None,
    click_window: dict[str, Any] | None = None,
    vision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build move_mouse with a short target + optional nearby_objects / click_window."""
    resolved = (target if target is not None else _short_move_target(instruction)).strip()
    if not resolved:
        resolved = instruction.strip()
    args: dict[str, Any] = {"instruction": resolved}
    nearby = _nearby_object_phrases(instruction)
    if nearby:
        args["nearby_objects"] = nearby
        # Constellation for non-text targets (e.g. empty 輸入欄) and for
        # empty-space clicks that currently encode raw pixel offsets.
        # On-object 「…」文字 clicks still omit geometry (must match OCR).
        offset_click = _is_relative_offset_position(resolved)
        if (not _move_target_is_text(resolved)) or offset_click:
            hints = extract_nearby_hints_from_instruction(instruction)
            if offset_click:
                from cua_mcp.instruction_offset import (
                    _parse_relative_pixel_offset_regex,
                )
                from src.common.nearby_side import NearbyHint

                ref_anchor, _dx, _dy = _parse_relative_pixel_offset_regex(resolved)
                ref = (ref_anchor or "").strip()
                if ref:
                    hints = [NearbyHint(label=ref, side=None), *hints]
            geometry = build_landmark_geometry(
                vision,
                hints,
                target_label=resolved,
            )
            if geometry is not None:
                args["landmark_geometry"] = geometry
    if click_window is not None:
        args["click_window"] = click_window
    return _call("move_mouse", **args)


def _move_target_is_text(target: str) -> bool:
    """True when the move_mouse primary is an on-object named label.

    UIA control suffixes (``清單項目``, ``按鈕``, …) match by name the same way
    ``文字`` does. A bare ``按鈕`` with no quoted name stays a class anchor.
    """
    text = target or ""
    for suffix in ("圖示", "輸入欄", "滾動條", "未知", "元素"):
        if suffix in text:
            return False
    if "按鈕" in text and "「" not in text:
        return False
    for suffix in (
        "清單項目",
        "功能表項目",
        "樹狀項目",
        "資料項目",
        "超連結",
        "按鈕",
        "文字",
    ):
        if suffix in text:
            return True
    return False


def _is_relative_offset_position(target: str) -> bool:
    """True for empty-space clicks like ``「字型」文字右方12個像素、下方49個像素的位置``."""
    text = target or ""
    return "個像素" in text and "的位置" in text


def _compile_click_family(
    event: RecordedEvent,
    instruction: str,
    *,
    vision: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if event.kind not in _CLICK_POINTER_KINDS:
        return None
    if not _short_move_target(instruction) and not instruction.strip():
        return None
    calls: list[dict[str, Any]] = [
        _move_mouse_call(
            instruction,
            click_window=event.click_window,
            vision=vision,
        )
    ]
    modifiers = _modifier_list(event.modifiers)
    if event.kind == "hold":
        seconds = (
            float(event.duration_seconds)
            if event.duration_seconds is not None
            else 1.0
        )
        seconds = max(seconds, 0.1)
        hold_args: dict[str, Any] = {
            "seconds": seconds,
            "button": event.button or "left",
            "instruction": instruction,
        }
        if modifiers:
            hold_args["modifiers"] = modifiers
        calls.append(_call("hold_mouse", **hold_args))
        return calls

    if event.kind == "right_click":
        calls.append(_call("right_click", instruction=instruction))
        return calls
    if event.kind == "middle_click":
        calls.append(_call("middle_click", instruction=instruction))
        return calls

    count = _effective_left_click_count(event.kind, event.click_count)
    if count is None:
        return None
    if count == 2:
        args: dict[str, Any] = {"instruction": instruction}
        if modifiers:
            args["modifiers"] = modifiers
        calls.append(_call("double_click", **args))
        return calls
    if count == 3:
        args = {"instruction": instruction}
        if modifiers:
            args["modifiers"] = modifiers
        calls.append(_call("triple_click", **args))
        return calls
    click_args: dict[str, Any] = {
        "button": "left",
        "clicks": count,
        "instruction": instruction,
    }
    if modifiers:
        click_args["modifiers"] = modifiers
    calls.append(_call("click", **click_args))
    return calls


def _compile_drag(event: RecordedEvent, instruction: str) -> list[dict[str, Any]] | None:
    if event.kind != "drag":
        return None
    mode, body = split_vision_mode(instruction.strip())
    cleaned = strip_nearby_context_comments(body).strip()
    match = _DRAG_SPLIT_RE.match(cleaned)
    if not match:
        return None
    start = match.group(1).strip()
    destination = match.group(2).strip()
    if not start or not destination:
        return None
    prefix = vision_mode_prefix(mode)
    if prefix:
        start = prefix + start
        destination = prefix + destination
    buckets = extract_nearby_hints_by_location(instruction)
    start_nearby = nearby_hints_to_phrases(buckets.get("起點") or [])
    # Trailing （在…） comments land in 附近; use them for start when 起點 is empty
    # so scrollbar track drags still disambiguate the start resolve.
    if not start_nearby:
        start_nearby = nearby_hints_to_phrases(buckets.get("附近") or [])
    dest_nearby = nearby_hints_to_phrases(
        merge_nearby_hints(buckets.get("終點"), buckets.get("附近"))
    )
    args: dict[str, Any] = {
        "start_instruction": start,
        "destination_instruction": destination,
    }
    if start_nearby:
        args["start_nearby_objects"] = start_nearby
    if dest_nearby:
        args["destination_nearby_objects"] = dest_nearby
    if isinstance(event.click_window, dict):
        args["start_click_window"] = event.click_window
    if isinstance(event.end_click_window, dict):
        args["destination_click_window"] = event.end_click_window
    return [_call("drag", **args)]


def _compile_scroll(
    event: RecordedEvent,
    instruction: str,
    *,
    vision: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if event.kind != "scroll":
        return None
    delta = event.scroll_delta
    if delta is None or delta == 0:
        return None
    # Recorded positive delta → 向上捲動 → scroll tool negative clicks.
    clicks = -int(delta)
    calls: list[dict[str, Any]] = []
    mode, body = split_vision_mode(instruction.strip())
    hover = _SCROLL_HOVER_RE.match(body)
    if hover:
        target = hover.group(1).strip()
        prefix = vision_mode_prefix(mode)
        if prefix:
            target = prefix + target
        calls.append(
            _move_mouse_call(
                instruction,
                target=target,
                click_window=event.click_window,
                vision=vision,
            )
        )
    calls.append(_call("scroll", clicks=clicks, instruction=instruction))
    return calls


def _vision_for_compile(
    analysis: dict[str, Any] | None,
    *,
    run_dir: Path | None,
    event: RecordedEvent,
    vision: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Prefer an explicit vision payload, else analysis candidates, else yolo_ocr."""
    if isinstance(vision, dict) and isinstance(vision.get("candidates"), list):
        return vision
    if isinstance(analysis, dict):
        analysis_vision = analysis.get("vision")
        if (
            isinstance(analysis_vision, dict)
            and isinstance(analysis_vision.get("candidates"), list)
            and analysis_vision.get("candidates")
        ):
            return analysis_vision
    if run_dir is None:
        return None
    from src.recorder.vision_context import vision_from_yolo_ocr

    loaded = vision_from_yolo_ocr(Path(run_dir), int(event.index), suffix="")
    if isinstance(loaded.get("candidates"), list) and loaded["candidates"]:
        return loaded
    return None


def compile_tool_calls(
    event: RecordedEvent,
    instruction: str,
    analysis: dict[str, Any] | None = None,
    *,
    vision: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Compile MCP tool calls for one recording step, or None when unsupported."""
    goal = instruction.strip()
    if not goal:
        return None

    resolved_vision = _vision_for_compile(
        analysis,
        run_dir=run_dir,
        event=event,
        vision=vision,
    )

    # Confident window chrome actions take priority over the underlying pointer kind.
    # Shell flyouts (快顯主機, …) are ignored here: dismissals stay in window_verify
    # as disappeared, not as close_windows.
    if isinstance(event.window_change, dict):
        confidence = event.window_change.get("confidence")
        action = str(event.window_change.get("action") or "")
        if (
            confidence in {"high", "medium"}
            and action in {"minimize", "maximize", "close"}
            and not should_ignore_window_change(event.window_change)
        ):
            if action != "close" or event.window_change.get("from_title_bar_close"):
                compiled = _compile_window_change(event, goal, analysis)
                if compiled is not None:
                    return _validated(compiled)

    kind = event.kind
    if kind == "text_input":
        compiled = _compile_text_input(event, goal)
    elif kind == "key_press":
        compiled = _compile_key_press(event, goal)
    elif kind == "hotkey":
        compiled = _compile_hotkey(event, goal)
    elif kind == "wait":
        compiled = _compile_wait(event, goal)
    elif kind == "drag":
        compiled = _compile_drag(event, goal)
    elif kind == "scroll":
        compiled = _compile_scroll(event, goal, vision=resolved_vision)
    elif kind in _CLICK_POINTER_KINDS:
        # Instruction-shaped window actions (manual edits / added events).
        if _WINDOW_INSTRUCTION_RE.match(goal):
            compiled = _compile_window_change(event, goal, analysis)
        else:
            compiled = _compile_click_family(
                event, goal, vision=resolved_vision
            )
    elif _WINDOW_INSTRUCTION_RE.match(goal):
        compiled = _compile_window_change(event, goal, analysis)
    elif _WAIT_INSTRUCTION_RE.match(goal):
        compiled = _compile_wait(event, goal)
    else:
        compiled = None

    return _validated(compiled)


def set_analysis_tool_calls(
    analysis: dict[str, Any],
    event: RecordedEvent,
    instruction: str | None = None,
    *,
    run_dir: Path | None = None,
    vision: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Set or clear ``tool_calls`` on an analysis dict from the current instruction."""
    goal = instruction if isinstance(instruction, str) else analysis.get("instruction")
    if not isinstance(goal, str) or not goal.strip():
        analysis.pop("tool_calls", None)
        return None
    calls = compile_tool_calls(
        event,
        goal,
        analysis,
        vision=vision,
        run_dir=run_dir,
    )
    if calls is None:
        analysis.pop("tool_calls", None)
        return None
    analysis["tool_calls"] = calls
    return calls


def _recording_cache_entry(
    instruction: str,
    tool_calls: list[dict[str, Any]],
    *,
    source_run_id: str,
) -> tuple[str, dict[str, Any]] | None:
    if validate_tool_calls(tool_calls) is not None:
        return None
    key = normalize_instruction(instruction)
    return key, {
        "instruction": key,
        "tool_calls": tool_calls,
        "updated_at_utc": "",
        "source_run_id": source_run_id,
    }


def rebuild_recording_instruction_tool_cache(run_dir: Path) -> Path:
    """Rewrite ``instruction_tool_cache.json`` from analysis files that have tool_calls."""
    target = recording_tool_cache_path(run_dir)
    entries: dict[str, Any] = {}
    source_run_id = Path(run_dir).name
    analysis_dir = Path(run_dir) / "analysis"
    if analysis_dir.is_dir():
        for analysis_path in sorted(analysis_dir.glob("event_*.json")):
            analysis = read_json(analysis_path, None)
            if not isinstance(analysis, dict):
                continue
            instruction = analysis.get("instruction")
            tool_calls = analysis.get("tool_calls")
            if not isinstance(instruction, str) or not instruction.strip():
                continue
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            entry = _recording_cache_entry(
                instruction, tool_calls, source_run_id=source_run_id
            )
            if entry is not None:
                key, payload = entry
                entries[key] = payload
    write_json(target, {"version": _CACHE_VERSION, "entries": entries})
    return target


def ensure_recording_tool_cache(run_dir: Path) -> None:
    """Recompile analysis tool_calls when possible and rebuild the recording cache.

    Deterministic compiles overwrite stale full hub-script move_mouse args.
    When compile returns None, existing on-disk tool_calls (e.g. LLM-mirrored)
    are left unchanged.

    Builds the instruction cache from the same analysis pass (no second full
    re-read of every event_*.json).
    """
    root = Path(run_dir)
    analysis_dir = root / "analysis"
    if not analysis_dir.is_dir():
        return
    from src.recorder.uia_replay_repair import backfill_yolo_ocr_fallbacks

    backfill_yolo_ocr_fallbacks(root)
    source_run_id = root.name
    entries: dict[str, Any] = {}
    for analysis_path in sorted(analysis_dir.glob("event_*.json")):
        analysis = read_json(analysis_path, None)
        if not isinstance(analysis, dict):
            continue
        instruction = analysis.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        tool_calls_for_cache = analysis.get("tool_calls")
        event_index = analysis.get("event_index")
        if not isinstance(event_index, int):
            match = re.search(r"event_(\d+)\.json$", analysis_path.name)
            if match:
                event_index = int(match.group(1))
        if isinstance(event_index, int):
            event_path = event_json_path(root, event_index)
            event_payload = read_json(event_path, None)
            if isinstance(event_payload, dict):
                event = RecordedEvent.from_dict(event_payload)
                previous = analysis.get("tool_calls")
                compiled = set_analysis_tool_calls(
                    analysis,
                    event,
                    instruction,
                    run_dir=root,
                )
                if compiled is not None:
                    if previous != compiled:
                        write_json(analysis_path, analysis)
                    tool_calls_for_cache = compiled
                else:
                    # set_analysis_tool_calls pops tool_calls from the in-memory
                    # dict when compile fails; keep prior LLM-mirrored calls.
                    tool_calls_for_cache = previous
        if not isinstance(tool_calls_for_cache, list) or not tool_calls_for_cache:
            continue
        entry = _recording_cache_entry(
            instruction, tool_calls_for_cache, source_run_id=source_run_id
        )
        if entry is not None:
            key, payload = entry
            entries[key] = payload
    write_json(
        recording_tool_cache_path(root),
        {"version": _CACHE_VERSION, "entries": entries},
    )


def mirror_tool_calls_into_recording(
    run_dir: Path,
    instruction: str,
    tool_calls: list[dict[str, Any]],
    *,
    source_run_id: str,
) -> None:
    """Persist successful LLM tool_calls into matching analysis + recording cache."""
    from src.common.instruction_tool_cache import upsert_tool_calls

    if validate_tool_calls(tool_calls) is not None:
        return
    key = normalize_instruction(instruction)
    if not key:
        return
    analysis_dir = Path(run_dir) / "analysis"
    if analysis_dir.is_dir():
        for analysis_path in sorted(analysis_dir.glob("event_*.json")):
            analysis = read_json(analysis_path, None)
            if not isinstance(analysis, dict):
                continue
            existing = analysis.get("instruction")
            if not isinstance(existing, str):
                continue
            if normalize_instruction(existing) != key:
                continue
            analysis["tool_calls"] = tool_calls
            write_json(analysis_path, analysis)
            break
    upsert_tool_calls(
        instruction,
        tool_calls,
        source_run_id=source_run_id,
        path=recording_tool_cache_path(run_dir),
    )

