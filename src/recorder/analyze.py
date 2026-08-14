from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cua_mcp.selection_engine import request_json_with_retry
from src.common.prompting import get_prompt
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent
from src.recorder.to_cache import event_summary_for_llm
from src.recorder.vision_context import (
    append_drag_nearby_context_comments,
    append_nearby_context_comment,
    build_vision_context,
    candidate_anchor_name,
    candidate_offset_for_instruction,
    format_drag_candidate_anchor,
    format_drag_destination_offset_hints,
    format_field_context_hint,
    format_input_context_hint,
    format_scrollbar_context_hint,
    primary_candidate_char_target,
    primary_candidate_offset,
    _visible_text,
)
from cua_mcp.char_target import format_char_target_anchor
from src.recorder.window_snapshot import format_window_change_hint, instruction_for_window_change

_INSTRUCTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
    },
    "required": ["instruction"],
}

_EXPECTED_OUTCOME_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expected_outcome": {"type": ["string", "null"]},
    },
    "required": ["expected_outcome"],
}

_DRAG_ANCHOR_RE = re.compile(r"拖到「([^」]+)」")
_DRAG_SOURCE_RE = re.compile(r"^從「[^」]+」(?:文字|圖示|檔案|資料夾|按鈕|元素)*(?=拖到)")
_DRAG_DESTINATION_SUFFIX_RE = re.compile(r"(文字|圖示|檔案|資料夾|按鈕|元素)*")
_CLICK_TARGET_SUFFIX_RE = re.compile(r"(文字|圖示|檔案|資料夾|按鈕|元素|未知|輸入欄)*")
_DRAG_OFFSET_PHRASE_RE = re.compile(
    r"(?:(?:左方|右方|上方|下方)\d+個像素)(?:、(?:(?:左方|右方|上方|下方)\d+個像素))*"
)
_CLICK_POINTER_KINDS = frozenset(
    {"click", "double_click", "triple_click", "right_click", "middle_click", "hold"}
)
_CLICK_MOVE_PREFIX = "將滑鼠移到"
_POINTER_CLICK_ACTION_SUFFIX_BY_KIND = {
    "click": "，並點擊滑鼠一下。",
    "double_click": "，並連按2下。",
    "triple_click": "，並連按3下。",
    "right_click": "，用右鍵點選。",
    "middle_click": "，並按中鍵一下。",
}
_POINTER_CLICK_MODIFIER_ACTION_BY_KIND = {
    "click": "點擊",
    "double_click": "連按2下",
    "triple_click": "連按3下",
    "right_click": "右鍵點選",
    "middle_click": "中鍵點擊",
}
_GENERIC_CLICK_ANCHORS = frozenset(
    {"文字", "元素", "未知", "輸入欄", "按鈕", "滾動條"}
)
_KEY_DISPLAY_NAMES = {
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "esc": "Esc",
    "escape": "Esc",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "home": "Home",
    "end": "End",
    "page_up": "PageUp",
    "page_down": "PageDown",
    "insert": "Insert",
    "space": "Space",
    "ctrl": "Ctrl",
    "control": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "win": "Win",
    "cmd": "Win",
    "command": "Win",
    "windows": "Win",
}
_MODIFIER_DISPLAY_ORDER = ("Ctrl", "Alt", "Shift", "Win")


def _parse_instruction_reply(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    instruction = data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction missing or empty")
    return {"instruction": instruction.strip()}


def enrich_drag_instruction_source(
    instruction: str,
    vision: dict[str, Any],
) -> str:
    """Replace the drag source with the nearest OCR/YOLO candidate."""
    if "拖到" not in instruction:
        return instruction

    candidates = vision.get("candidates") or []
    if not candidates:
        return instruction

    anchor = format_drag_candidate_anchor(candidates[0])
    if not anchor:
        return instruction

    match = _DRAG_SOURCE_RE.match(instruction)
    if not match:
        return instruction

    return f"從{anchor}{instruction[match.end():]}"


def enrich_drag_instruction_destination(
    instruction: str,
    destination: dict[str, Any],
) -> str:
    """Replace the drag destination anchor with the nearest OCR/YOLO candidate."""
    if "拖到" not in instruction:
        return instruction

    candidates = destination.get("candidates") or []
    if not candidates:
        return instruction

    anchor = format_drag_candidate_anchor(candidates[0])
    if not anchor:
        return instruction

    match = _DRAG_ANCHOR_RE.search(instruction)
    if not match:
        return instruction

    anchor_end = match.end()
    suffix_match = _DRAG_DESTINATION_SUFFIX_RE.match(instruction[anchor_end:])
    insert_at = anchor_end + (suffix_match.end() if suffix_match else 0)
    remainder = instruction[insert_at:]
    remainder = _DRAG_OFFSET_PHRASE_RE.sub("", remainder, count=1)
    if remainder.startswith("的位置"):
        remainder = remainder[len("的位置") :]
    return instruction[: match.start()] + f"拖到{anchor}" + remainder


def enrich_drag_instruction_offset(
    instruction: str,
    destination: dict[str, Any],
) -> str:
    """Normalize drag instructions to the exact OCR-derived relative pixel offset."""
    if "拖到" not in instruction:
        return instruction

    match = _DRAG_ANCHOR_RE.search(instruction)
    if not match:
        return instruction

    offset_phrase = candidate_offset_for_instruction(destination, match.group(1))
    if not offset_phrase:
        return instruction

    anchor_end = match.end()
    suffix_match = _DRAG_DESTINATION_SUFFIX_RE.match(instruction[anchor_end:])
    insert_at = anchor_end + (suffix_match.end() if suffix_match else 0)
    remainder = instruction[insert_at:]
    remainder = _DRAG_OFFSET_PHRASE_RE.sub("", remainder, count=1)
    if remainder.startswith("的位置"):
        remainder = remainder[len("的位置") :]
    return instruction[:insert_at] + offset_phrase + "的位置" + remainder


def enrich_drag_instruction(
    instruction: str,
    *,
    vision: dict[str, Any],
    destination: dict[str, Any],
) -> str:
    """Normalize drag source, destination, and relative pixel offset from vision."""
    instruction = enrich_drag_instruction_source(instruction, vision)
    instruction = enrich_drag_instruction_destination(instruction, destination)
    return enrich_drag_instruction_offset(instruction, destination)


def enrich_click_instruction_offset(
    instruction: str,
    vision: dict[str, Any],
) -> str:
    """Replace vague 附近 with OCR-derived relative pixel offset when click is off-target."""
    candidates = vision.get("candidates") or []
    if not candidates:
        return instruction

    anchor_name = candidate_anchor_name(candidates[0])
    if not anchor_name:
        return instruction

    offset_phrase = candidate_offset_for_instruction(vision, anchor_name)
    if not offset_phrase:
        return instruction

    pattern = re.compile(rf"「{re.escape(anchor_name)}」")
    best: tuple[re.Match[str], int] | None = None
    for match in pattern.finditer(instruction):
        after = instruction[match.end() :]
        suffix_match = _CLICK_TARGET_SUFFIX_RE.match(after)
        insert_at = match.end() + (suffix_match.end() if suffix_match else 0)
        remainder = instruction[insert_at:]
        if remainder.startswith("附近") and not remainder.startswith("附近有"):
            best = (match, insert_at)
            break
        if best is None:
            best = (match, insert_at)

    if best is None:
        return instruction

    _, insert_at = best
    remainder = instruction[insert_at:]
    if remainder.startswith("附近") and not remainder.startswith("附近有"):
        remainder = remainder[len("附近") :]
    remainder = _DRAG_OFFSET_PHRASE_RE.sub("", remainder, count=1)
    if remainder.startswith("的位置"):
        remainder = remainder[len("的位置") :]
    return instruction[:insert_at] + offset_phrase + "的位置" + remainder


_TEXT_INPUT_INSTRUCTION_PREFIX = "輸入「"
_TEXT_INPUT_INSTRUCTION_SUFFIX = "」"


def instruction_for_text_input(text: str) -> str | None:
    """Build a hub-script line for a typing-only recorded event."""
    cleaned = text.strip()
    if not cleaned:
        return None
    return f"{_TEXT_INPUT_INSTRUCTION_PREFIX}{cleaned}{_TEXT_INPUT_INSTRUCTION_SUFFIX}"


def typed_text_from_instruction(instruction: str) -> str | None:
    """Extract the typed payload from ``輸入「...」``, or None if the shape does not match."""
    text = instruction.strip()
    if not text.startswith(_TEXT_INPUT_INSTRUCTION_PREFIX):
        return None
    if not text.endswith(_TEXT_INPUT_INSTRUCTION_SUFFIX):
        return None
    inner = text[len(_TEXT_INPUT_INSTRUCTION_PREFIX) : -len(_TEXT_INPUT_INSTRUCTION_SUFFIX)]
    return inner if inner else None


def _visible_input_field_text(vision: dict[str, Any]) -> str | None:
    """Return OCR text shown inside the nearest input, if available."""
    hint = format_input_context_hint(vision)
    prefix = "輸入欄內可見文字: 「"
    if hint and hint.startswith(prefix) and hint.endswith("」"):
        return hint[len(prefix) : -1] or None
    return None


def _visible_scrollbar_content_label(vision: dict[str, Any]) -> str | None:
    """Return scrollable content beside the nearest scrollbar, when available."""
    hint = format_scrollbar_context_hint(vision)
    prefix = "滾動條旁可見內容: 「"
    if hint and hint.startswith(prefix) and hint.endswith("」"):
        return hint[len(prefix) : -1] or None
    return None


def _click_target_anchor(vision: dict[str, Any]) -> str | None:
    """Return a named click target from the nearest candidate, or None if too generic."""
    candidates = vision.get("candidates") or []
    if not candidates:
        return None

    primary = candidates[0]
    if primary.get("class_name") == "input":
        visible = _visible_input_field_text(vision)
        if visible:
            return f"「{visible}」文字所在的輸入欄"
        return "輸入欄"

    if primary.get("class_name") == "scrollbar":
        visible = _visible_scrollbar_content_label(vision)
        if visible:
            return f"「{visible}」文字區域的滾動條"
        return "滾動條"

    anchor = format_drag_candidate_anchor(primary)
    if anchor is None or anchor in _GENERIC_CLICK_ANCHORS:
        return None
    return anchor


def instruction_for_click(
    event: RecordedEvent,
    vision: dict[str, Any],
) -> str | None:
    """Build a hub-script click line from the nearest OCR/YOLO candidate."""
    if event.kind not in _CLICK_POINTER_KINDS:
        return None

    char_target = primary_candidate_char_target(vision)
    if char_target is not None:
        candidates = vision.get("candidates") or []
        visible = _visible_text(candidates[0].get("text")) if candidates else ""
        if visible:
            clicked_char, occurrence = char_target
            anchor = format_char_target_anchor(visible, clicked_char, occurrence=occurrence)
            return f"{_CLICK_MOVE_PREFIX}{anchor}"

    anchor = _click_target_anchor(vision)
    if not anchor:
        return None

    offset_phrase = primary_candidate_offset(vision)
    if offset_phrase:
        return f"{_CLICK_MOVE_PREFIX}{anchor}{offset_phrase}的位置"
    return f"{_CLICK_MOVE_PREFIX}{anchor}"


def _key_display_name(token: str) -> str | None:
    cleaned = token.strip()
    if not cleaned:
        return None
    # Older recordings stored Windows Ctrl+letter as ASCII control chars
    # (Ctrl+A → \\u0001). Map those back so display is Ctrl+A, not Ctrl+.
    if len(cleaned) == 1 and not cleaned.isprintable():
        code = ord(cleaned)
        if 1 <= code <= 26:
            return chr(ord("A") + code - 1)
    lower = cleaned.lower()
    if lower in _KEY_DISPLAY_NAMES:
        return _KEY_DISPLAY_NAMES[lower]
    if len(lower) >= 2 and lower[0] == "f" and lower[1:].isdigit():
        return f"F{lower[1:]}"
    if lower.startswith("vk_"):
        return None
    if len(cleaned) == 1 and cleaned.isprintable():
        return cleaned.upper() if cleaned.isalpha() else cleaned
    if cleaned.isalnum():
        return cleaned.capitalize()
    return None


def _hotkey_display_combo(keys: list[str]) -> str | None:
    if not keys:
        return None
    displays: list[str] = []
    for token in keys:
        name = _key_display_name(token)
        if name is None:
            return None
        displays.append(name)

    modifiers = [name for name in _MODIFIER_DISPLAY_ORDER if name in displays]
    others = [name for name in displays if name not in _MODIFIER_DISPLAY_ORDER]
    ordered = modifiers + others
    if not ordered:
        return None
    return "+".join(ordered)


def instruction_for_key(event: RecordedEvent) -> str | None:
    """Build a hub-script key/hotkey line from recorded key tokens."""
    if event.kind == "key_press":
        name = _key_display_name(event.key or "")
        if name is None:
            return None
        return f"按下 {name} 鍵"

    if event.kind == "hotkey":
        keys = event.keys or []
        combo = _hotkey_display_combo([str(k) for k in keys])
        if combo is None:
            return None
        return f"按下 {combo}"

    return None


def instruction_for_scroll(
    event: RecordedEvent,
    vision: dict[str, Any],
) -> str | None:
    """Build a hub-script scroll line from scroll_delta and nearest candidate."""
    if event.kind != "scroll":
        return None
    delta = event.scroll_delta
    if delta is None or delta == 0:
        return None

    direction = "向上" if delta > 0 else "向下"
    anchor = _click_target_anchor(vision)
    if anchor:
        return f"在{anchor}附近{direction}捲動"
    return f"{direction}捲動"


def instruction_for_drag(
    vision: dict[str, Any],
    destination: dict[str, Any],
) -> str | None:
    """Build a hub-script drag line from nearest OCR/YOLO candidates."""
    source_candidates = vision.get("candidates") or []
    dest_candidates = destination.get("candidates") or []
    if not source_candidates or not dest_candidates:
        return None

    source_anchor = format_drag_candidate_anchor(source_candidates[0])
    dest_anchor = format_drag_candidate_anchor(dest_candidates[0])
    if not source_anchor or not dest_anchor:
        return None

    dest_name = candidate_anchor_name(dest_candidates[0])
    offset_phrase = (
        candidate_offset_for_instruction(destination, dest_name)
        if dest_name
        else None
    )
    if offset_phrase:
        return f"從{source_anchor}拖到{dest_anchor}{offset_phrase}的位置"
    return f"從{source_anchor}拖到{dest_anchor}"


def _format_hold_duration_label(duration_seconds: float | None) -> str:
    seconds = 1.0 if duration_seconds is None else max(float(duration_seconds), 0.1)
    rounded = round(seconds, 1)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.1f}"


def _hold_action_phrase(button: str | None, duration_seconds: float | None) -> str:
    duration_label = _format_hold_duration_label(duration_seconds)
    if button == "right":
        return f"用右鍵按住約{duration_label}秒"
    return f"按住約{duration_label}秒"


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


def _left_multi_click_action_phrase(count: int) -> str:
    if count <= 1:
        return "點擊滑鼠一下"
    return f"連按{count}下"


def _left_multi_click_modifier_action(count: int) -> str:
    if count <= 1:
        return "點擊"
    return f"連按{count}下"


def _pointer_click_action_suffix(
    kind: str,
    modifiers: list[str] | None,
    *,
    button: str | None = None,
    duration_seconds: float | None = None,
    click_count: int | None = None,
) -> str | None:
    if kind == "hold":
        action = _hold_action_phrase(button, duration_seconds)
        if modifiers:
            combo = _hotkey_display_combo([str(m) for m in modifiers])
            if combo:
                return f"，並{combo}+{action}。"
        return f"，並{action}。"
    left_count = _effective_left_click_count(kind, click_count)
    if left_count is not None:
        if modifiers:
            action = _left_multi_click_modifier_action(left_count)
            combo = _hotkey_display_combo([str(m) for m in modifiers])
            if combo:
                return f"，並{combo}+{action}。"
        if left_count <= 1:
            return _POINTER_CLICK_ACTION_SUFFIX_BY_KIND["click"]
        return f"，並{_left_multi_click_action_phrase(left_count)}。"
    if modifiers:
        action = _POINTER_CLICK_MODIFIER_ACTION_BY_KIND.get(kind)
        combo = _hotkey_display_combo([str(m) for m in modifiers])
        if action and combo:
            return f"，並{combo}+{action}。"
    return _POINTER_CLICK_ACTION_SUFFIX_BY_KIND.get(kind)


def _finalize_instruction(
    instruction: str,
    event: RecordedEvent,
    vision: dict[str, Any] | None,
    destination: dict[str, Any] | None = None,
) -> str:
    if event.kind not in POINTER_EVENT_KINDS or vision is None:
        return instruction
    if event.kind == "drag":
        return append_drag_nearby_context_comments(
            instruction,
            vision,
            destination if isinstance(destination, dict) else {},
        )
    instruction = append_nearby_context_comment(instruction, vision)
    suffix = _pointer_click_action_suffix(
        event.kind,
        event.modifiers,
        button=event.button,
        duration_seconds=event.duration_seconds,
        click_count=event.click_count,
    )
    if suffix:
        return instruction + suffix
    return instruction


def rebuild_pointer_instruction(
    event: RecordedEvent,
    vision: dict[str, Any],
    destination: dict[str, Any] | None = None,
    *,
    include_nearby: bool = True,
) -> str | None:
    """Rebuild a hub-script pointer instruction from ranked vision candidates.

    When ``include_nearby`` is False, omits nearby-context parentheticals so
    callers can apply user-selected landmarks afterward. Click/hold kinds still
    receive their action suffix (e.g. 「，並點擊滑鼠一下。」).
    """
    dest = destination if isinstance(destination, dict) else {}

    if event.kind == "drag":
        base = instruction_for_drag(vision, dest)
        if base is None:
            return None
        if include_nearby:
            return append_drag_nearby_context_comments(base, vision, dest)
        return base

    if event.kind in _CLICK_POINTER_KINDS:
        base = instruction_for_click(event, vision)
        if base is None:
            return None
        if include_nearby:
            base = append_nearby_context_comment(base, vision)
        suffix = _pointer_click_action_suffix(
            event.kind,
            event.modifiers,
            button=event.button,
            duration_seconds=event.duration_seconds,
            click_count=event.click_count,
        )
        return base + suffix if suffix else base

    if event.kind == "scroll":
        base = instruction_for_scroll(event, vision)
        if base is None:
            return None
        if include_nearby:
            return append_nearby_context_comment(base, vision)
        return base

    return None


def _instruction_result(
    instruction: str,
    event: RecordedEvent,
    vision: dict[str, Any] | None,
    destination: dict[str, Any] | None = None,
) -> dict[str, str]:
    return {
        "instruction": _finalize_instruction(
            instruction,
            event,
            vision,
            destination,
        )
    }


async def analyze_event_to_cache(
    event: RecordedEvent,
    *,
    run_dir: Path,
    vision: dict[str, Any] | None = None,
    log_info: Any = None,
) -> dict[str, Any] | None:
    """Return a hub-script instruction for one recorded event, or None on failure."""
    if event.kind == "text_input":
        instruction = instruction_for_text_input(event.text or "")
        if instruction is not None:
            return {"instruction": instruction}

    if event.kind in {"key_press", "hotkey"}:
        key_instruction = instruction_for_key(event)
        if key_instruction is not None:
            return {"instruction": key_instruction}

    if event.window_change:
        deterministic = instruction_for_window_change(event.window_change)
        if deterministic is not None:
            return {"instruction": deterministic}

    if vision is None:
        vision = await build_vision_context(event, run_dir=run_dir, log_info=log_info)

    destination = vision.get("destination") if isinstance(vision.get("destination"), dict) else {}
    if event.kind == "drag":
        drag_instruction = instruction_for_drag(vision, destination)
        if drag_instruction is not None:
            return _instruction_result(drag_instruction, event, vision, destination)

    if event.kind in _CLICK_POINTER_KINDS:
        click_instruction = instruction_for_click(event, vision)
        if click_instruction is not None:
            return _instruction_result(click_instruction, event, vision, destination)

    if event.kind == "scroll":
        scroll_instruction = instruction_for_scroll(event, vision)
        if scroll_instruction is not None:
            return _instruction_result(scroll_instruction, event, vision, destination)

    local = vision.get("local_cursor")
    if isinstance(local, (list, tuple)) and len(local) == 2:
        cursor_x, cursor_y = local[0], local[1]
    else:
        cursor_x, cursor_y = "", ""

    dest_local = destination.get("local_cursor")
    if isinstance(dest_local, (list, tuple)) and len(dest_local) == 2:
        destination_x, destination_y = dest_local[0], dest_local[1]
    else:
        destination_x, destination_y = "", ""

    field_context = format_field_context_hint(
        vision,
        typed_text=event.text if event.kind == "text_input" else None,
    )
    destination_field_context = destination.get("field_context") or "(none)"
    destination_candidate_text = destination.get("candidate_text") or "(none)"
    if event.kind == "drag":
        destination_offset_hints = (
            destination.get("destination_offset_hints")
            or format_drag_destination_offset_hints(destination)
        )
    elif event.kind in _CLICK_POINTER_KINDS:
        destination_offset_hints = format_drag_destination_offset_hints(vision)
    else:
        destination_offset_hints = "(not applicable)"

    prompt = get_prompt("recording_action_to_cache").format(
        event_json=event_summary_for_llm(event),
        cursor_x=cursor_x,
        cursor_y=cursor_y,
        candidate_text=vision.get("candidate_text") or "(none)",
        field_context=field_context,
        destination_x=destination_x,
        destination_y=destination_y,
        destination_candidate_text=destination_candidate_text,
        destination_field_context=destination_field_context,
        destination_offset_hints=destination_offset_hints,
        window_change_hint=format_window_change_hint(event.window_change),
    )

    user_content = prompt
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    shot = event.screenshot_path
    if shot and Path(shot).is_file() and event.kind in {
        "click",
        "double_click",
        "triple_click",
        "right_click",
        "middle_click",
        "hold",
        "scroll",
        "drag",
    }:
        images = [shot]
        if event.kind == "drag":
            end_shot = event.end_screenshot_path
            if end_shot and Path(end_shot).is_file() and end_shot != shot:
                images.append(end_shot)
        messages[0]["images"] = images

    try:
        result = await request_json_with_retry(
            messages=messages,
            response_schema=_INSTRUCTION_RESPONSE_SCHEMA,
            parse_reply=_parse_instruction_reply,
            retry_instruction=get_prompt("recording_action_to_cache_retry"),
            log_info=log_info,
            append_image_sizes=True,
        )
        if event.kind == "drag":
            instruction = enrich_drag_instruction(
                result["instruction"],
                vision=vision,
                destination=destination,
            )
            return _instruction_result(instruction, event, vision, destination)
        if event.kind in _CLICK_POINTER_KINDS:
            instruction = enrich_click_instruction_offset(
                result["instruction"],
                vision,
            )
            return _instruction_result(instruction, event, vision, destination)
        return _instruction_result(result["instruction"], event, vision, destination)
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"analyze_event_to_cache failed event={event.index}: {exc}")
        return None


def _existing_screenshot_path(raw: str | None) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip())
    return str(path) if path.is_file() else None


def before_screenshot_for_outcome(event: RecordedEvent) -> str | None:
    """Return the pre-action screenshot path used as the before frame for verification."""
    return _existing_screenshot_path(event.screenshot_path)


def after_screenshot_for_outcome(
    event: RecordedEvent,
    next_event: RecordedEvent | None,
    *,
    final_after_screenshot: str | None = None,
) -> str | None:
    """Return the after frame for verification.

    Prefer the next event's before-shot, then a drag end shot, then the
    session-level final screenshot taken before the hub window is restored.
    """
    if next_event is not None:
        next_before = _existing_screenshot_path(next_event.screenshot_path)
        if next_before is not None:
            return next_before
    if event.kind == "drag":
        end_shot = _existing_screenshot_path(event.end_screenshot_path)
        if end_shot is not None:
            return end_shot
    return _existing_screenshot_path(final_after_screenshot)


def _parse_expected_outcome_reply(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected_outcome reply must be an object")
    if "expected_outcome" not in data:
        raise ValueError("expected_outcome missing")
    value = data.get("expected_outcome")
    if value is None:
        return {"expected_outcome": None}
    if not isinstance(value, str):
        raise ValueError("expected_outcome must be string or null")
    cleaned = value.strip()
    return {"expected_outcome": cleaned or None}


async def infer_expected_outcome(
    *,
    instruction: str,
    before_screenshot: str,
    after_screenshot: str,
    log_info: Any = None,
) -> str | None:
    """Ask the LLM for a checkable success criterion from before/after screenshots."""
    instruction = instruction.strip()
    if not instruction:
        return None
    before_path = Path(before_screenshot)
    after_path = Path(after_screenshot)
    if not before_path.is_file() or not after_path.is_file():
        return None
    if before_path.resolve() == after_path.resolve():
        return None

    prompt = get_prompt("recording_expected_outcome").format(instruction=instruction)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "images": [str(before_path), str(after_path)],
        }
    ]
    try:
        result = await request_json_with_retry(
            messages=messages,
            response_schema=_EXPECTED_OUTCOME_RESPONSE_SCHEMA,
            parse_reply=_parse_expected_outcome_reply,
            retry_instruction=get_prompt("recording_expected_outcome_retry"),
            log_info=log_info,
            append_image_sizes=True,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        if log_info is not None:
            log_info(f"infer_expected_outcome failed: {exc}")
        return None
    outcome = result.get("expected_outcome")
    return outcome if isinstance(outcome, str) and outcome.strip() else None
