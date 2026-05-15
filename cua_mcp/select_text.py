from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
try:
    from opencc import OpenCC
except Exception:  # pragma: no cover - optional dependency
    OpenCC = None  # type: ignore[assignment]

from cua_mcp.read_screen_text.ocr_image import (
    format_coordinate_text_from_regions,
    get_coordinates_from_path,
)
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, get_run_state_manager, ts_name
from src.common.settings import load_settings
from src.eye import active_monitor_offset
from src.eye.capture import capture_active_monitor_to_file

settings = load_settings()


def _run_manager() -> RunStateManager:
    """Always resolve the current singleton (never cache): ``reset_run_state_manager`` replaces it."""
    return get_run_state_manager()


_t2s_converter = OpenCC("t2s") if OpenCC else None

# Ollama JSON mode: model names OCR text; we map back to region centers.
_TARGET_TEXT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
}

# Second round when the same OCR text matches multiple regions.
_DISAMBIGUATE_XY_TEXT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "text": {"type": "string"},
    },
    "required": ["x", "y", "text"],
}


def _get_active_capture_offset() -> tuple[int, int]:
    try:
        return active_monitor_offset()
    except Exception as exc:
        _run_manager().log_info(f"Failed active_monitor_offset err={type(exc).__name__}: {exc}")
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


def _to_simplified_chinese(s: str) -> str:
    """
    Convert Traditional Chinese to Simplified Chinese when OpenCC is available.
    Falls back to the original string if the converter is unavailable.
    """
    if not s or _t2s_converter is None:
        return s
    try:
        return _t2s_converter.convert(s)
    except Exception:
        return s


def _extract_matchable_text_candidates(instruction: str) -> list[str]:
    """
    Extract likely UI text snippets from an instruction that can be compared
    against OCR row texts.
    """
    text = (instruction or "").strip()
    if not text:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        v = " ".join(value.split()).strip()
        if len(v) < 2:
            return
        key = v.casefold()
        if key in seen:
            return
        seen.add(key)
        candidates.append(v)

    # Prefer explicitly quoted UI labels.
    for quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', text):
        add(quoted[0] or quoted[1])

    # Also consider text after common action verbs.
    verb_pattern = (
        r"(?i)\b(?:click|tap|press|select|choose|open|launch|type|enter)\b\s+"
        r"(?:on\s+|the\s+)?(.+?)(?:[.,;]|$)"
    )
    for m in re.finditer(verb_pattern, text):
        add(m.group(1))

    # Fallback to full instruction if nothing else was extracted.
    if not candidates:
        add(text)

    return candidates


def _pick_best_similarity_row(
    candidates: list[str],
    rows: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """
    Return every OCR row whose best similarity score (against any candidate) is
    greater than zero, sorted by descending score.
    """
    if not candidates or not rows:
        return []

    scored: list[tuple[float, int, int, str]] = []
    for cx, cy, row_text in rows:
        row_key = _normalize_match_key(_to_simplified_chinese(row_text))
        if not row_key:
            continue
        best_for_row = 0.0
        for candidate in candidates:
            cand_key = _normalize_match_key(_to_simplified_chinese(candidate))
            if not cand_key:
                continue
            score = SequenceMatcher(None, cand_key, row_key).ratio()
            if score > best_for_row:
                best_for_row = score
        scored.append((best_for_row, cx, cy, row_text))

    positive = [(s, cx, cy, t) for s, cx, cy, t in scored if s > 0]
    positive.sort(key=lambda item: item[0], reverse=True)
    return [(cx, cy, t) for s, cx, cy, t in positive]


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


def _parse_xy_text_from_llm_content(raw: str) -> tuple[int, int, str]:
    """Parse ``{"x": int, "y": int, "text": str}`` from model text; raises ValueError if invalid."""
    json_text = _llm_text_to_json_object_string(raw)
    preview = (raw or "")[:240]
    if not json_text:
        raise ValueError('expected {"x": int, "y": int, "text": string}')
    try:
        out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc}); preview={preview!r}") from exc
    if not isinstance(out, dict) or "x" not in out or "y" not in out or "text" not in out:
        raise ValueError(f'must include "x", "y", and "text"; preview={preview!r}')
    x, y, t = out["x"], out["y"], out["text"]
    if not isinstance(t, str):
        raise ValueError(f'"text" must be a string, got {type(t).__name__}')
    return int(x), int(y), t.strip()


async def _disambiguate_duplicate_centers(
    instruction: str,
    chosen_text: str,
    matches: list[tuple[int, int, str]],
    image_path: str,
) -> tuple[int, int, str]:
    """Second LLM round: pick one of several identical (or tier-equivalent) OCR locations."""
    allowed = {(cx, cy) for cx, cy, _ in matches}
    allowed_str = {s: (cx, cy) for cx, cy, s in matches}
    options_lines = "\n".join(f"[{cx},{cy}] {t}" for cx, cy, t in matches)
    base = (
        "The matching OCR text appears at MORE THAN ONE location on the image.\n"
        "Pick ONE center (x, y) that best matches the Instruction.\n"
        "(x, y) MUST be exactly one of the candidate centers listed below - same "
        "coordinate space as CoordinatesText (image pixels).\n"
        '"text" MUST be the OCR line for that same choice: copy it from the line after '
        "[cx,cy] for your chosen center (as verbatim as possible; OCR may have typos).\n"
        "Keep in mind that the OCR text might have typos and errors, so you need to be careful to match the text correctly.\n"
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
        reply = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_DISAMBIGUATE_XY_TEXT_JSON_SCHEMA,
        )
        x, y, llm_text = _parse_xy_text_from_llm_content(reply.content)
    except ValueError as exc:
        _run_manager().log_info(f"_disambiguate_duplicate_centers: retry ({exc})")
        messages[0]["content"] += (
            '\nReply with ONLY: {"x": <integer>, "y": <integer>, "text": "<string>"} '
            "where x,y equals one candidate [cx,cy] above and \"text\" is that line's "
            "OCR text. No text before or after the JSON.\n"
        )
        reply = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
        )
        x, y, llm_text = _parse_xy_text_from_llm_content(reply.content)
    if (x, y) not in allowed and llm_text not in allowed_str.keys():
        raise ValueError(
            f"disambiguation returned ({x},{y},{llm_text}) not in allowed {matches}"
        )
    if (x, y) not in allowed:
        x, y = allowed_str[llm_text]
    elif llm_text not in allowed_str.keys():
        llm_text = [m[2] for m in matches if m[2] == llm_text][0]
    return x, y, llm_text


async def _select_coordinate(
    target: str,
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

    # Fast path: OCR rows with positive text similarity to instruction-derived candidates.
    extracted_texts = _extract_matchable_text_candidates(target)
    similarity_matches = _pick_best_similarity_row(extracted_texts, rows)
    if len(similarity_matches) == 1:
        cx, cy, matched_text = similarity_matches[0]
        _run_manager().log_info(
            f"_select_coordinate: similarity pre-match picked ({cx},{cy}) from {matched_text!r}"
        )
        return cx, cy
    if len(similarity_matches) > 1:
        image_path = str(path.resolve())
        x, y, dis_text = await _disambiguate_duplicate_centers(
            instruction, target, similarity_matches, image_path
        )
        _run_manager().log_info(
            f"_select_coordinate: similarity pre-match disambiguated to ({x},{y}) {dis_text!r}"
        )
        return x, y

    coordinate_text = format_coordinate_text_from_regions(regions)
    base_instructions = get_prompt("coordinate_selection").format(
        instruction=instruction,
        target=target,
        coordinate_text=coordinate_text,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": base_instructions, "images": [image_path]},
    ]
    try:
        reply = await get_llm_client().chat_messages(
            settings.brain_lm,
            messages=messages,
            tools=[],
            response_format=_TARGET_TEXT_JSON_SCHEMA,
        )
        chosen = _parse_target_text_from_llm_content(reply.content)
        matches = _match_tiers_to_rows(chosen, rows)
    except ValueError as exc:
        _run_manager().log_info(
            f"_select_coordinate: first attempt failed ({exc}); retrying with format=json."
        )
        messages[0]["content"] += (
            '\nReply with ONLY: {"text": "<string>"} where "text" is the OCR line text from CoordinatesText '
            "(after [cx,cy] ), as verbatim as possible. No text before or after the JSON.\n"
        )
        reply = await get_llm_client().chat_messages(
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

    cx, cy, _ = await _disambiguate_duplicate_centers(
        instruction, chosen, matches, image_path
    )
    return cx, cy


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


async def _resolve_point(target: str, instruction: str) -> tuple[int, int, str]:
    paths = _run_manager().require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ocr_dir / name
    capture_active_monitor_to_file(out)

    (off_x, off_y), regions = get_coordinates_from_path(str(out))
    local_x, local_y = await _select_coordinate(
        target=target,
        instruction=instruction,
        regions=regions,
        screenshot_path=out,
    )
    img_x, img_y = local_x + off_x, local_y + off_y
    clicked = _clicked_text_at_image_point(img_x, img_y, regions)
    gx, gy = _to_global_coordinate(img_x, img_y)
    return gx, gy, clicked
