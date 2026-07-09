"""
Shared UI detection types and LLM picker/filter helpers for mouse target selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from cua_mcp.geometry import sort_by_reading_order
from cua_mcp.icon_map import is_pua_char, text_has_pua
from cua_mcp.llm_json import parse_json_object
from cua_mcp.selection_engine import request_json_with_retry
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, get_run_state_manager
from src.common.settings import load_settings


def _run_manager() -> RunStateManager:
    """Always resolve the current singleton (never cache): ``reset_run_state_manager`` replaces it."""
    return get_run_state_manager()


_INDEX_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"index": {"type": "integer"}},
    "required": ["index"],
}
_TEXT_FILTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep_indices": {
            "type": "array",
            "items": {"type": "integer"},
        }
    },
    "required": ["keep_indices"],
}


@dataclass(frozen=True)
class UiDetection:
    """One clickable candidate: bbox and center in screenshot pixels plus optional OCR metadata."""

    bbox: tuple[int, int, int, int]  # x, y, w, h in image pixels
    cx: int
    cy: int
    class_id: int
    class_name: str
    text: str | None = None
    icons: list[dict[str, Any]] | None = None


def _text_is_pua_only(text: str) -> bool:
    """True when ``text`` has PUA codepoints and no other visible characters."""
    if not text:
        return False
    if not text_has_pua(text):
        return False
    non_pua = "".join(ch for ch in text if not is_pua_char(ch)).strip()
    return not non_pua


def _parse_index_from_llm(raw: str, num_candidates: int) -> int:
    """Parse the picker LLM reply; returns the chosen candidate index (0-based)."""
    out = parse_json_object(
        raw,
        empty_error='Ollama UI picker returned empty content; expected {"index": <int>}',
        decode_error_prefix="invalid JSON",
    )
    preview = (raw or "")[:240]
    if not isinstance(out, dict) or "index" not in out:
        raise ValueError(f'must include "index"; preview={preview!r}')
    try:
        idx = int(out["index"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'"index" must be an integer; got {out.get("index")!r}') from exc
    if idx < 0 or idx >= num_candidates:
        raise ValueError(
            f'"index" out of range: {idx} (valid 0..{num_candidates - 1}); preview={preview!r}'
        )
    return idx


def _parse_keep_indices_from_llm(raw: str, max_len: int) -> list[int]:
    """Parse text-filter LLM reply; return deduplicated 0-based indices in ``[0, max_len)``."""
    out = parse_json_object(
        raw,
        empty_error='Ollama text filter returned empty content; expected {"keep_indices": [int, ...]}',
        decode_error_prefix="invalid JSON",
    )
    preview = (raw or "")[:240]
    if "keep_indices" not in out:
        raise ValueError(f'must include "keep_indices"; preview={preview!r}')
    value = out["keep_indices"]
    if not isinstance(value, list):
        raise ValueError(f'"keep_indices" must be a list; preview={preview!r}')
    keep: list[int] = []
    seen: set[int] = set()
    for item in value:
        idx = int(item)
        if idx < 0 or idx >= max_len or idx in seen:
            continue
        seen.add(idx)
        keep.append(idx)
    return keep


_CANDIDATE_CLASS_LABELS: dict[str, str] = {
    "text": "文字(Text)",
    "element": "元素(Element)",
    "input": "輸入欄(Input)",
    "scrollbar": "滾動條(Scrollbar)",
    "unknown": "未知(Unknown)",
}


def _candidate_class_label(class_name: str) -> str:
    return _CANDIDATE_CLASS_LABELS.get(class_name, class_name)


def _sort_detections_reading_order(detections: list[UiDetection]) -> list[UiDetection]:
    """Top-to-bottom, then left-to-right (same spirit as OCR reading order)."""
    return sort_by_reading_order(
        detections,
        center_fn=lambda d: (d.cx, d.cy),
        row_height_fn=lambda d: d.bbox[3],
        x_fn=lambda d: d.cx,
    )


def _format_ui_candidates_text(detections: list[UiDetection]) -> str:
    """Format candidate rows for LLM filter/picker prompts."""
    lines: list[str] = []
    for i, d in enumerate(detections):
        text_part = (
            f" text={d.text!r}"
            if d.text and not _text_is_pua_only(d.text)
            else ""
        )
        chinese_ids = ",".join(
            ii.get("chinese_id", "") for ii in (d.icons or []) if ii.get("chinese_id")
        )
        icon_part = f" icons={chinese_ids}" if chinese_ids else ""
        _bx, _by, bw, bh = d.bbox
        lines.append(
            f"[index {i}] class={_candidate_class_label(d.class_name)} center=[{d.cx},{d.cy}] w={bw} h={bh}"
            f"{text_part}{icon_part}"
        )
    return "\n".join(lines)


async def _select_center_with_ollama(
    instruction: str,
    detections: list[UiDetection],
    image_paths: list[str],
) -> int:
    """
    Ask Ollama for the best candidate index (0-based into ``detections``).

    Falls back to :func:`request_json_with_retry` if the first reply cannot be parsed.
    """
    if not detections:
        raise ValueError("no candidates to pick from")
    candidates_text = _format_ui_candidates_text(detections)
    screenshot_sizes: list[str] = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
            screenshot_sizes.append(f"{img_w}x{img_h}")
    screenshot_size_text = ", ".join(screenshot_sizes) if screenshot_sizes else "unknown"
    prompt = get_prompt("ui_element_selection").format(
        instruction=instruction,
        candidates_text=candidates_text,
        screenshot_sizes=screenshot_size_text,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "images": image_paths,
        },
    ]
    n = len(detections)
    reply1 = await get_llm_client().chat_messages(
        load_settings().brain_lm,
        messages=messages,
        tools=[],
        response_format=_INDEX_JSON_SCHEMA,
        think=True,
    )
    try:
        pool_idx = _parse_index_from_llm(reply1.content, n)
    except ValueError:
        pool_idx = None

    if pool_idx is not None:
        return pool_idx

    return await request_json_with_retry(
        messages=messages,
        response_schema=_INDEX_JSON_SCHEMA,
        parse_reply=lambda raw: _parse_index_from_llm(raw, n),
        retry_instruction=get_prompt("ui_element_selection_retry"),
        log_info=lambda m: _run_manager().log_info(f"_select_center_with_ollama: {m}"),
    )
