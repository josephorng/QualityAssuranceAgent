from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cua_mcp import hand_tools
# from cua_mcp.tools import mcp_server
from cua_mcp.read_screen_text.ocr_image import (
    format_coordinate_text_from_regions,
    get_coordinates_from_path,
)
from cua_mcp.storage import store_image, store_text, _current_run_paths
from src.common.run_state import get_run_state_manager, ts_name
from src.common.ollama_client import OllamaClient
from src.common.settings import load_settings
from src.eye.capture import capture_active_monitor_to_file
from src.eye import active_monitor_offset

settings = load_settings()
logger = get_run_state_manager()
_ollama = OllamaClient(settings.ollama_host)


# Ollama JSON mode: model names OCR text; we map back to region centers.
_TARGET_TEXT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
}

# Second round when the same OCR text matches multiple regions.
_XY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
    },
    "required": ["x", "y"],
}


def _get_active_capture_offset() -> tuple[int, int]:
    try:
        return active_monitor_offset()
    except Exception as exc:
        logger.log_info(f"Failed active_monitor_offset err={type(exc).__name__}: {exc}")
    return 0, 0


def _to_global_coordinate(local_x: int, local_y: int) -> tuple[int, int]:
    left, top = _get_active_capture_offset()
    return local_x + left, local_y + top


def _llm_text_to_json_object_string(raw: str) -> str:
    """Extract the first JSON object from a possibly markdown-fenced string."""
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_target_text_from_llm_content(raw: str) -> str:
    """Parse `{\"text\": str}` from model text; raises ValueError if missing or invalid."""
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            'Ollama target picker returned empty or non-JSON content; expected {"text": string}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Ollama target picker returned invalid JSON ({exc}); preview={preview!r}"
        ) from exc
    if not isinstance(out, dict):
        raise ValueError(f"Ollama target JSON must be an object, got {type(out).__name__}")
    if "text" not in out:
        raise ValueError(
            f'Ollama target JSON must include "text"; got keys={list(out.keys())!r}; preview={preview!r}'
        )
    text = out["text"]
    if not isinstance(text, str):
        raise ValueError(f'Ollama "text" must be a string, got {type(text).__name__}')
    return text.strip()


def _normalize_match_key(s: str) -> str:
    return " ".join(s.split()).casefold()


def _extract_rows(
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for _box, (cx, cy), preds in regions:
        line = _predictions_to_str(preds)
        if line:
            rows.append((cx, cy, line))
    return rows


def _match_tiers_to_rows(
    target: str,
    rows: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """
    Return every OCR row that matches ``target`` using the first applicable tier
    (exact, casefold, substring target-in-line, substring line-in-target).
    Raises ValueError if none match.
    """
    if not target.strip():
        raise ValueError("Ollama returned empty target text")
    if not rows:
        raise ValueError("OCR regions contain no non-empty text")

    target_key = _normalize_match_key(target)

    exact = [(cx, cy, t) for cx, cy, t in rows if target.strip() == t]
    if exact:
        return exact

    fold = [(cx, cy, t) for cx, cy, t in rows if target_key == _normalize_match_key(t)]
    if fold:
        return fold

    in_line = [
        (cx, cy, t)
        for cx, cy, t in rows
        if target_key and target_key in _normalize_match_key(t)
    ]
    if in_line:
        return in_line

    line_in_target = [
        (cx, cy, t)
        for cx, cy, t in rows
        if _normalize_match_key(t) and _normalize_match_key(t) in target_key
    ]
    if line_in_target:
        return line_in_target

    preview = ", ".join(repr(t) for _x, _y, t in rows[:12])
    if len(rows) > 12:
        preview += ", ..."
    raise ValueError(
        f"could not map target text {target!r} to any OCR line; candidates={preview}"
    )


def _parse_xy_from_llm_content(raw: str) -> tuple[int, int]:
    """Parse ``{\"x\": int, \"y\": int}`` from model text; raises ValueError if invalid."""
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError(
            'expected {"x": int, "y": int}'
        )
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "x" not in out or "y" not in out:
        raise ValueError(f'must include "x" and "y"; preview={preview!r}')
    return int(out["x"]), int(out["y"])


def _disambiguate_duplicate_centers(
    instruction: str,
    chosen_text: str,
    matches: list[tuple[int, int, str]],
    image_path: str,
) -> tuple[int, int]:
    """Second LLM round: pick one of several identical (or tier-equivalent) OCR locations."""
    allowed = {(cx, cy) for cx, cy, _ in matches}
    options_lines = "\n".join(f"[{cx},{cy}] {t}" for cx, cy, t in matches)
    base = (
        "The matching OCR text appears at MORE THAN ONE location on the image.\n"
        "Pick ONE center (x, y) that best matches the Instruction.\n"
        "(x, y) MUST be exactly one of the candidate centers listed below — same "
        "coordinate space as CoordinatesText (image pixels).\n"
        "Output NOTHING except valid JSON matching the server's schema.\n"
        "Do not summarize, explain, or add prose.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Matched text from the first step:\n{chosen_text}\n\n"
        f"Candidate centers (choose exactly one):\n{options_lines}\n"
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": base, "images": [image_path]},
    ]
    try:
        reply = _ollama.chat_messages_sync(
            settings.brain_lm,
            messages=messages,
            use_tools=False,
            response_format=_XY_JSON_SCHEMA,
        )
        x, y = _parse_xy_from_llm_content(reply.content)
    except ValueError as exc:
        logger.log_info(f"_disambiguate_duplicate_centers: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"x": <integer>, "y": <integer>} equal to one candidate '
            "[cx,cy] above. No text before or after the JSON.\n"
        )
        reply = _ollama.chat_messages_sync(
            settings.brain_lm,
            messages=messages,
            use_tools=False,
            response_format="json",
        )
        x, y = _parse_xy_from_llm_content(reply.content)
    if (x, y) not in allowed:
        raise ValueError(
            f"disambiguation returned ({x},{y}) not in allowed {sorted(allowed)}"
        )
    return x, y


def _select_coordinate(
    instruction: str,
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
    screenshot_path: str | Path,
) -> tuple[int, int]:
    path = Path(screenshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"screenshot not found: {path}")
    image_path = str(path.resolve())
    rows = _extract_rows(regions)
    if not rows:
        raise ValueError("OCR regions contain no non-empty text")

    coordinate_text = format_coordinate_text_from_regions(regions)
    base_instructions = (
        "Choose ONE line from CoordinatesText that best matches Instruction.\n"
        "CoordinatesText lines look like: [center_x,center_y] <OCR text for that region>.\n"
        "Reply with the OCR text only (the part after the bracket), copied verbatim from "
        "CoordinatesText when possible so it can be matched.\n"
        "Output NOTHING except valid JSON matching the server's schema.\n"
        "Do not summarize, classify, bullet-list, markdown, translate, explain, add keys, or add prose.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"CoordinatesText:\n{coordinate_text}\n"
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": base_instructions, "images": [image_path]},
    ]
    try:
        reply = _ollama.chat_messages_sync(
            settings.brain_lm,
            messages=messages,
            use_tools=False,
            response_format=_TARGET_TEXT_JSON_SCHEMA,
        )
        chosen = _parse_target_text_from_llm_content(reply.content)
        matches = _match_tiers_to_rows(chosen, rows)
    except ValueError as exc:
        logger.log_info(
            f"_select_coordinate: first attempt failed ({exc}); retrying with format=json."
        )
        messages[0]["content"] += (
            '\nReply with ONLY: {"text": "<string>"} where "text" is the OCR line text from CoordinatesText '
            "(after [cx,cy] ), as verbatim as possible. No text before or after the JSON.\n"
        )
        reply = _ollama.chat_messages_sync(
            settings.brain_lm,
            messages=messages,
            use_tools=False,
            response_format="json",
        )
        chosen = _parse_target_text_from_llm_content(reply.content)
        matches = _match_tiers_to_rows(chosen, rows)

    if len(matches) == 1:
        cx, cy, _ = matches[0]
        return cx, cy

    return _disambiguate_duplicate_centers(instruction, chosen, matches, image_path)


def _predictions_to_str(preds: list[str]) -> str:
    return "".join(preds).strip()


def _clicked_text_at_image_point(
    img_x: int,
    img_y: int,
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
) -> str:
    """OCR text for the region whose box contains (img_x, img_y), else nearest line by center."""
    if not regions:
        return ""
    ix, iy = int(img_x), int(img_y)
    for _box, (cx, cy), preds in regions:
        t = _predictions_to_str(preds)
        if cx == ix and cy == iy:
            return t
    raise ValueError(f"No text found at image point ({img_x}, {img_y})")


def _with_clicked_text(result: dict[str, Any], clicked_text: str) -> dict[str, Any]:
    merged = dict(result)
    merged["clicked_text"] = clicked_text
    return merged


def _resolve_point(instruction: str) -> tuple[int, int, str]:
    paths = logger.require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ocr_dir / name
    capture_active_monitor_to_file(out)

    (off_x, off_y), regions = get_coordinates_from_path(str(out))
    local_x, local_y = _select_coordinate(
        instruction=instruction,
        regions=regions,
        screenshot_path=out,
    )
    img_x, img_y = local_x + off_x, local_y + off_y
    clicked = _clicked_text_at_image_point(img_x, img_y, regions)
    gx, gy = _to_global_coordinate(img_x, img_y)
    return gx, gy, clicked


def _click(instruction: str, button: str = "left") -> dict[str, Any]:
    x, y, clicked = _resolve_point(instruction)
    return _with_clicked_text(hand_tools.click(x=x, y=y, button=button), clicked)


def _click_and_type(
    text: str,
    target_instruction: str,
) -> dict[str, Any]:
    x, y, clicked = _resolve_point(target_instruction)
    return _with_clicked_text(hand_tools.type_text(text=text, coordinate=[x, y]), clicked)


def _press_key(key: str, instruction: str = "") -> dict[str, Any]:
    return hand_tools.hotkey(keys=key)


def _hotkey(keys: list[str] | str, instruction: str = "") -> dict[str, Any]:
    return hand_tools.hotkey(keys=keys)


def _move(instruction: str, duration: float = 0.0) -> dict[str, Any]:
    x, y, clicked = _resolve_point(instruction)
    return _with_clicked_text(hand_tools.move(x=x, y=y, duration=duration), clicked)


def _wait(seconds: float, instruction: str = "") -> dict[str, Any]:
    return hand_tools.wait(seconds=seconds)


def _key(key: str, instruction: str = "") -> dict[str, Any]:
    return hand_tools.key_press(key)


def _mouse_move(instruction: str, duration: float = 0.0) -> dict[str, Any]:
    return _move(instruction=instruction, duration=duration)


def _click_at_instruction(instruction: str, **click_kw: Any) -> dict[str, Any]:
    x, y, clicked = _resolve_point(instruction)
    return _with_clicked_text(hand_tools.click(x=x, y=y, **click_kw), clicked)


def _left_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="left", clicks=1)


def _right_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="right", clicks=1)


def _middle_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="middle", clicks=1)


def _double_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="left", clicks=2, interval=0.1)


def _triple_click(instruction: str) -> dict[str, Any]:
    return _click_at_instruction(instruction, button="left", clicks=3, interval=0.1)


def _left_click_drag(
    instruction_start: str,
    instruction_end: str,
    duration: float = 0.5,
) -> dict[str, Any]:
    x1, y1, t_start = _resolve_point(instruction_start)
    x2, y2, t_end = _resolve_point(instruction_end)
    out = dict(hand_tools.drag(x1, y1, x2, y2, duration=duration, button="left"))
    out["clicked_text_start"] = t_start
    out["clicked_text_end"] = t_end
    return out


def _screenshot(path: str = "", instruction: str = "") -> dict[str, Any]:
    p = path.strip() if path else ""
    storage_dir, _ = _current_run_paths()
    if not p:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        p = str(storage_dir / f"screenshot_{stamp}.png")
    else:
        candidate = Path(p)
        if not candidate.is_absolute():
            p = str(storage_dir / candidate.name)
    return hand_tools.screenshot_to_file(p)


def _cursor_position(instruction: str = "") -> dict[str, Any]:
    return hand_tools.cursor_position()


def _left_mouse_down(instruction: str) -> dict[str, Any]:
    x, y, clicked = _resolve_point(instruction)
    return _with_clicked_text(hand_tools.mouse_down(x, y, button="left"), clicked)


def _left_mouse_up(instruction: str) -> dict[str, Any]:
    x, y, clicked = _resolve_point(instruction)
    return _with_clicked_text(hand_tools.mouse_up(x, y, button="left"), clicked)


def _scroll(instruction: str, clicks: int) -> dict[str, Any]:
    x, y, clicked = _resolve_point(instruction)
    return _with_clicked_text(hand_tools.scroll_at(clicks, x, y), clicked)


def _hold_key(key: str, seconds: float, instruction: str = "") -> dict[str, Any]:
    return hand_tools.hold_key_down(key, seconds)


def _zoom(instruction: str, scroll_clicks: int) -> dict[str, Any]:
    x, y, clicked = _resolve_point(instruction)
    return _with_clicked_text(hand_tools.zoom_scroll(scroll_clicks, x, y), clicked)


def _maximize_window(window_title_contains: str, instruction: str = "") -> dict[str, Any]:
    return hand_tools.maximize_window(
        window_title_contains=window_title_contains,
        instruction=instruction,
    )


def _store_text(
    text: str,
    instruction: str = "",
    title: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return store_text(text=text, title=title, tags=tags)


def _store_image(
    image_path: str,
    instruction: str = "",
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return store_image(image_path=image_path, summary=summary, alias=alias, tags=tags)
