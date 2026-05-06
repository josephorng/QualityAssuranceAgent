from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cua_mcp.read_screen_text.ocr_image import (
    format_coordinate_text_from_regions,
    get_coordinates_from_path,
)
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager, ts_name
from src.common.settings import load_settings
from src.eye import active_monitor_offset
from src.eye.capture import capture_active_monitor_to_file

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
    """Parse `{"text": str}` from model text; raises ValueError if missing or invalid."""
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
    """Parse ``{"x": int, "y": int}`` from model text; raises ValueError if invalid."""
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError('expected {"x": int, "y": int}')
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "x" not in out or "y" not in out:
        raise ValueError(f'must include "x" and "y"; preview={preview!r}')
    return int(out["x"]), int(out["y"])


async def _disambiguate_duplicate_centers(
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
        "(x, y) MUST be exactly one of the candidate centers listed below - same "
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
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_XY_JSON_SCHEMA,
        )
        x, y = _parse_xy_from_llm_content(reply.content)
    except ValueError as exc:
        logger.log_info(f"_disambiguate_duplicate_centers: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"x": <integer>, "y": <integer>} equal to one candidate '
            "[cx,cy] above. No text before or after the JSON.\n"
        )
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
        )
        x, y = _parse_xy_from_llm_content(reply.content)
    if (x, y) not in allowed:
        raise ValueError(
            f"disambiguation returned ({x},{y}) not in allowed {sorted(allowed)}"
        )
    return x, y


async def _select_coordinate(
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
    base_instructions = get_prompt("coordinate_selection").format(
        instruction=instruction,
        coordinate_text=coordinate_text,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": base_instructions, "images": [image_path]},
    ]
    try:
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
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
        reply = await _ollama.chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
        )
        chosen = _parse_target_text_from_llm_content(reply.content)
        matches = _match_tiers_to_rows(chosen, rows)

    if len(matches) == 1:
        cx, cy, _ = matches[0]
        return cx, cy

    return await _disambiguate_duplicate_centers(instruction, chosen, matches, image_path)


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


async def _resolve_point(instruction: str) -> tuple[int, int, str]:
    paths = logger.require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ocr_dir / name
    capture_active_monitor_to_file(out)

    (off_x, off_y), regions = get_coordinates_from_path(str(out))
    local_x, local_y = await _select_coordinate(
        instruction=instruction,
        regions=regions,
        screenshot_path=out,
    )
    img_x, img_y = local_x + off_x, local_y + off_y
    clicked = _clicked_text_at_image_point(img_x, img_y, regions)
    gx, gy = _to_global_coordinate(img_x, img_y)
    return gx, gy, clicked
