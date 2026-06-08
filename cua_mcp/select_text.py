from __future__ import annotations

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
    get_coordinates_from_image_path,
)
from cua_mcp.icon_map import describe_text_icons, is_pua_char, map_pua_in_text
from cua_mcp.llm_json import parse_json_object
from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, get_run_state_manager, ts_name
from src.eye import active_monitor_offset
from src.eye.capture import capture_active_monitor_to_file


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


def _parse_target_text_from_llm_content(raw: str) -> str:
    """Parse `{"text": str}` from model text; raises ValueError if missing or invalid."""
    out = parse_json_object(
        raw,
        empty_error='Ollama target picker returned empty or non-JSON content; expected {"text": string}',
        decode_error_prefix="Ollama target picker returned invalid JSON",
    )
    preview = (raw or "")[:240]
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


def _sanitize_target_text(target_text: str) -> list[str]:
    """
    Extract likely UI text snippets from an instruction that can be compared
    against OCR row texts.
    """
    text = (target_text or "").strip()
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


def _regions_with_mapped_pua(
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
) -> list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]]:
    """Return a copy of ``regions`` with PUA glyphs replaced by ``chinese_id`` from icon_map."""
    mapped: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]] = []
    for box, center, preds in regions:
        line = _predictions_to_str(preds)
        if not line:
            mapped.append((box, center, preds))
            continue
        mapped.append((box, center, [map_pua_in_text(line)]))
    return mapped


def _build_rows_text(
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
    out = parse_json_object(
        raw,
        empty_error='expected {"x": int, "y": int, "text": string}',
        decode_error_prefix="invalid JSON",
    )
    preview = (raw or "")[:240]
    if not isinstance(out, dict) or "x" not in out or "y" not in out or "text" not in out:
        raise ValueError(f'must include "x", "y", and "text"; preview={preview!r}')
    x, y, t = out["x"], out["y"], out["text"]
    if not isinstance(t, str):
        raise ValueError(f'"text" must be a string, got {type(t).__name__}')
    return int(x), int(y), t.strip()


def _strip_pua_from_text(text: str) -> str:
    """Remove Private Use Area codepoints and normalize whitespace."""
    without_pua = "".join(ch for ch in text if not is_pua_char(ch))
    return " ".join(without_pua.split()).strip()


def _matches_without_pua(
    matches: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Drop PUA from each match line; omit rows that are empty after stripping."""
    filtered: list[tuple[int, int, str]] = []
    for cx, cy, line in matches:
        cleaned = _strip_pua_from_text(line)
        if cleaned:
            filtered.append((cx, cy, cleaned))
    return filtered


async def _disambiguate_duplicate_centers(
    instruction: str,
    chosen_text: str,
    matches: list[tuple[int, int, str]],
    image_path: str,
) -> tuple[int, int, str]:
    """Second LLM round: pick one of several identical (or tier-equivalent) OCR locations."""
    matches = _matches_without_pua(matches)
    if not matches:
        raise ValueError(
            "no text candidates remain after removing PUA icon glyphs from duplicate matches"
        )
    if len(matches) == 1:
        return matches[0]

    options_lines = "\n".join(f"[{cx},{cy}] {t}" for cx, cy, t in matches)
    prompt_content = get_prompt("coordinate_disambiguation").format(
        instruction=instruction,
        chosen_text=chosen_text,
        options_lines=options_lines,
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": prompt_content, "images": [image_path]},
    ]
    x, y, llm_text = await request_json_with_retry(
        messages=messages,
        response_schema=_DISAMBIGUATE_XY_TEXT_JSON_SCHEMA,
        parse_reply=_parse_xy_text_from_llm_content,
        retry_instruction='Reply with ONLY: {"x": <integer>, "y": <integer>, "text": "<string>"} where x,y equals one candidate [cx,cy] above and "text" is that line\'s OCR text. No text before or after the JSON.',
        log_info=lambda m: _run_manager().log_info(f"_disambiguate_duplicate_centers: {m}"),
    )
    
    for cx, cy, t in matches:
        if (x, y) == (cx, cy):
            return x, y, t
        if llm_text == t:
            return cx, cy, t
    raise ValueError(
        f"disambiguation returned ({x},{y},{llm_text!r}) not in allowed {matches}"
    )


async def _select_text_coordinate_via_llm(
    instruction: str,
    target_text: str,
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
    coordinate_and_texts: list[tuple[int, int, str]],
    image_path: str,
) -> tuple[int, int]:
    """Ask the model to pick OCR text from CoordinatesText, then resolve to (cx, cy)."""
    coordinate_text = format_coordinate_text_from_regions(regions)
    base_instructions = get_prompt("coordinate_selection").format(
        instruction=instruction,
        target=target_text,
        coordinate_text=coordinate_text,
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": base_instructions, "images": [image_path]},
    ]
    chosen = await request_json_with_retry(
        messages=messages,
        response_schema=_TARGET_TEXT_JSON_SCHEMA,
        parse_reply=_parse_target_text_from_llm_content,
        retry_instruction='Reply with ONLY: {"text": "<string>"} where "text" is the OCR line text from CoordinatesText (after [cx,cy] ), as verbatim as possible. No text before or after the JSON.',
        log_info=lambda m: _run_manager().log_info(f"_select_coordinate_via_llm: {m}"),
    )
    matches = _match_tiers_to_rows(chosen, coordinate_and_texts)

    if len(matches) == 1:
        cx, cy, _ = matches[0]
        return cx, cy

    cx, cy, _ = await _disambiguate_duplicate_centers(
        instruction, chosen, matches, image_path
    )
    return cx, cy


async def _select_coordinate(
    target_text: str,
    instruction: str,
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
    screenshot_path: str | Path,
) -> tuple[int, int]:
    path = Path(screenshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"screenshot not found: {path}")
    image_path = str(path.resolve())
    regions = _regions_with_mapped_pua(regions)
    coordinate_and_texts = _build_rows_text(regions)
    if not coordinate_and_texts:
        raise ValueError("OCR regions contain no non-empty text")

    # Fast path: OCR rows with positive text similarity to instruction-derived candidates.
    sanitized_target_texts = _sanitize_target_text(target_text)
    similarity_matches = _pick_best_similarity_row(sanitized_target_texts, coordinate_and_texts)
    if len(similarity_matches) == 1:
        cx, cy, matched_text = similarity_matches[0]
        _run_manager().log_info(
            f"_select_coordinate: similarity pre-match picked ({cx},{cy}) from {matched_text!r}"
        )
        return cx, cy
    if len(similarity_matches) > 1:
        image_path = str(path.resolve())
        x, y, dis_text = await _disambiguate_duplicate_centers(
            instruction, target_text, similarity_matches, image_path
        )
        _run_manager().log_info(
            f"_select_coordinate: similarity pre-match disambiguated to ({x},{y}) {dis_text!r}"
        )
        return x, y

    return await _select_text_coordinate_via_llm(
        instruction, target_text, regions, coordinate_and_texts, image_path
    )


def _predictions_to_str(preds: list[str]) -> str:
    return "".join(preds).strip()


def _get_clicked_text_at_image_point(
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
    merged["target_kind"] = "text"
    merged["target_text"] = clicked_text
    merged["target_icons"] = describe_text_icons(clicked_text)
    return merged


async def resolve_text_point(
    target_text: str,
    instruction: str,
    *,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[int, int, str]:
    paths = _run_manager().require_paths()
    name = f"{ts_name()}.png"
    out = paths.yolo_ocr_dir / name
    capture_active_monitor_to_file(out)

    regions = get_coordinates_from_image_path(
        str(out), yolo_conf_threshold=yolo_conf_threshold
    )
    local_x, local_y = await _select_coordinate(
        target_text=target_text,
        instruction=instruction,
        regions=regions,
        screenshot_path=out,
    )
    clicked_text = _get_clicked_text_at_image_point(local_x, local_y, regions)
    gx, gy = _to_global_coordinate(local_x, local_y)
    return gx, gy, clicked_text
