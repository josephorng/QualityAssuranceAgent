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
from src.recorder.analyze import typed_text_from_instruction
from src.recorder.models import RecordedEvent, event_json_path
from src.recorder.to_cache import validate_tool_calls

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


def _compile_window_change(
    event: RecordedEvent, instruction: str
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
    tool_by_action = {
        "minimize": "minimize_windows",
        "maximize": "maximize_windows",
        "close": "close_windows",
    }
    tool = tool_by_action.get(action)
    if tool is None:
        return None
    return [
        _call(
            tool,
            window_title_contains=title,
            instruction=instruction,
        )
    ]


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
    text = strip_nearby_context_comments(instruction).strip()
    text = _CLICK_ACTION_SUFFIX_RE.sub("", text).strip().rstrip("，").strip()
    if text.startswith(_MOVE_PREFIX):
        text = text[len(_MOVE_PREFIX) :].strip()
    return text


def _nearby_object_phrases(instruction: str) -> list[str] | None:
    phrases = nearby_hints_to_phrases(extract_nearby_hints_from_instruction(instruction))
    return phrases or None


def _move_mouse_call(instruction: str, *, target: str | None = None) -> dict[str, Any]:
    """Build move_mouse with a short target + optional nearby_objects."""
    resolved = (target if target is not None else _short_move_target(instruction)).strip()
    if not resolved:
        resolved = instruction.strip()
    args: dict[str, Any] = {"instruction": resolved}
    nearby = _nearby_object_phrases(instruction)
    if nearby:
        args["nearby_objects"] = nearby
    return _call("move_mouse", **args)


def _compile_click_family(
    event: RecordedEvent, instruction: str
) -> list[dict[str, Any]] | None:
    if event.kind not in _CLICK_POINTER_KINDS:
        return None
    if not _short_move_target(instruction) and not instruction.strip():
        return None
    calls: list[dict[str, Any]] = [_move_mouse_call(instruction)]
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
    cleaned = strip_nearby_context_comments(instruction).strip()
    match = _DRAG_SPLIT_RE.match(cleaned)
    if not match:
        return None
    start = match.group(1).strip()
    destination = match.group(2).strip()
    if not start or not destination:
        return None
    buckets = extract_nearby_hints_by_location(instruction)
    start_nearby = nearby_hints_to_phrases(buckets.get("起點") or [])
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
    return [_call("drag", **args)]


def _compile_scroll(event: RecordedEvent, instruction: str) -> list[dict[str, Any]] | None:
    if event.kind != "scroll":
        return None
    delta = event.scroll_delta
    if delta is None or delta == 0:
        return None
    # Recorded positive delta → 向上捲動 → scroll tool negative clicks.
    clicks = -int(delta)
    calls: list[dict[str, Any]] = []
    hover = _SCROLL_HOVER_RE.match(instruction.strip())
    if hover:
        calls.append(_move_mouse_call(instruction, target=hover.group(1).strip()))
    calls.append(_call("scroll", clicks=clicks, instruction=instruction))
    return calls


def compile_tool_calls(
    event: RecordedEvent,
    instruction: str,
    analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Compile MCP tool calls for one recording step, or None when unsupported."""
    del analysis  # Reserved for future landmark-aware compilation.
    goal = instruction.strip()
    if not goal:
        return None

    # Confident window chrome actions take priority over the underlying pointer kind.
    if isinstance(event.window_change, dict):
        confidence = event.window_change.get("confidence")
        action = str(event.window_change.get("action") or "")
        if confidence in {"high", "medium"} and action in {
            "minimize",
            "maximize",
            "close",
        }:
            if action != "close" or event.window_change.get("from_title_bar_close"):
                compiled = _compile_window_change(event, goal)
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
        compiled = _compile_scroll(event, goal)
    elif kind in _CLICK_POINTER_KINDS:
        # Instruction-shaped window actions (manual edits / added events).
        if _WINDOW_INSTRUCTION_RE.match(goal):
            compiled = _compile_window_change(event, goal)
        else:
            compiled = _compile_click_family(event, goal)
    elif _WINDOW_INSTRUCTION_RE.match(goal):
        compiled = _compile_window_change(event, goal)
    elif _WAIT_INSTRUCTION_RE.match(goal):
        compiled = _compile_wait(event, goal)
    else:
        compiled = None

    return _validated(compiled)


def set_analysis_tool_calls(
    analysis: dict[str, Any],
    event: RecordedEvent,
    instruction: str | None = None,
) -> list[dict[str, Any]] | None:
    """Set or clear ``tool_calls`` on an analysis dict from the current instruction."""
    goal = instruction if isinstance(instruction, str) else analysis.get("instruction")
    if not isinstance(goal, str) or not goal.strip():
        analysis.pop("tool_calls", None)
        return None
    calls = compile_tool_calls(event, goal, analysis)
    if calls is None:
        analysis.pop("tool_calls", None)
        return None
    analysis["tool_calls"] = calls
    return calls


def rebuild_recording_instruction_tool_cache(run_dir: Path) -> Path:
    """Rewrite ``instruction_tool_cache.json`` from analysis files that have tool_calls."""
    target = recording_tool_cache_path(run_dir)
    entries: dict[str, Any] = {}
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
            if validate_tool_calls(tool_calls) is not None:
                continue
            key = normalize_instruction(instruction)
            entries[key] = {
                "instruction": key,
                "tool_calls": tool_calls,
                "updated_at_utc": "",
                "source_run_id": Path(run_dir).name,
            }
    write_json(target, {"version": _CACHE_VERSION, "entries": entries})
    return target


def ensure_recording_tool_cache(run_dir: Path) -> None:
    """Recompile analysis tool_calls when possible and rebuild the recording cache.

    Deterministic compiles overwrite stale full hub-script move_mouse args.
    When compile returns None, existing on-disk tool_calls (e.g. LLM-mirrored)
    are left unchanged.
    """
    root = Path(run_dir)
    analysis_dir = root / "analysis"
    if not analysis_dir.is_dir():
        return
    for analysis_path in sorted(analysis_dir.glob("event_*.json")):
        analysis = read_json(analysis_path, None)
        if not isinstance(analysis, dict):
            continue
        instruction = analysis.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        event_index = analysis.get("event_index")
        if not isinstance(event_index, int):
            match = re.search(r"event_(\d+)\.json$", analysis_path.name)
            if not match:
                continue
            event_index = int(match.group(1))
        event_path = event_json_path(root, event_index)
        event_payload = read_json(event_path, None)
        if not isinstance(event_payload, dict):
            continue
        event = RecordedEvent.from_dict(event_payload)
        previous = analysis.get("tool_calls")
        compiled = set_analysis_tool_calls(analysis, event, instruction)
        if compiled is None:
            continue
        if previous != compiled:
            write_json(analysis_path, analysis)
    rebuild_recording_instruction_tool_cache(root)


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

