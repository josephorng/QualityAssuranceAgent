"""
Shared UI detection types and LLM picker/filter helpers for mouse target selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cua_mcp.geometry import sort_by_reading_order
from cua_mcp.icon_map import is_pua_char, text_has_pua
from cua_mcp.llm_json import parse_json_object
from cua_mcp.selection_engine import request_json_with_retry
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, get_run_state_manager
from src.common.settings import load_settings

_NEIGHBOR_OFFSET_THRESHOLD_PX = 8
_ANCHOR_SUFFIX_BY_CLASS: dict[str, str] = {
    "text": "文字",
    "element": "元素",
    "unknown": "未知",
    "input": "輸入欄",
    "button": "按鈕",
    "scrollbar": "滾動條",
}


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
_MOUSE_FILTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anchor_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "nearby_indices": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["anchor_indices", "nearby_indices"],
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
    return _sanitize_index_list(value, max_len)


def _sanitize_index_list(value: list[Any], max_len: int) -> list[int]:
    """Deduplicate and clamp indices to ``[0, max_len)``, preserving order."""
    keep: list[int] = []
    seen: set[int] = set()
    for item in value:
        idx = int(item)
        if idx < 0 or idx >= max_len or idx in seen:
            continue
        seen.add(idx)
        keep.append(idx)
    return keep


def _parse_anchor_nearby_indices_from_llm(
    raw: str, max_len: int
) -> tuple[list[int], list[int]]:
    """Parse mouse-filter reply into ``(anchor_indices, nearby_indices)``.

    Indices that appear in both lists are kept only in ``anchor_indices``.
    """
    out = parse_json_object(
        raw,
        empty_error=(
            "Ollama mouse filter returned empty content; expected "
            '{"anchor_indices": [int, ...], "nearby_indices": [int, ...]}'
        ),
        decode_error_prefix="invalid JSON",
    )
    preview = (raw or "")[:240]
    if "anchor_indices" not in out or "nearby_indices" not in out:
        raise ValueError(
            f'must include "anchor_indices" and "nearby_indices"; preview={preview!r}'
        )
    anchor_raw = out["anchor_indices"]
    nearby_raw = out["nearby_indices"]
    if not isinstance(anchor_raw, list) or not isinstance(nearby_raw, list):
        raise ValueError(
            f'"anchor_indices" and "nearby_indices" must be lists; preview={preview!r}'
        )
    anchor_indices = _sanitize_index_list(anchor_raw, max_len)
    nearby_indices = _sanitize_index_list(nearby_raw, max_len)
    anchor_set = set(anchor_indices)
    nearby_indices = [i for i in nearby_indices if i not in anchor_set]
    return anchor_indices, nearby_indices


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


def _format_ui_candidates_text(
    detections: list[UiDetection],
    *,
    include_geometry: bool = True,
) -> str:
    """Format candidate rows for LLM filter/picker prompts.

    When ``include_geometry`` is False, omit ``center`` / ``w`` / ``h`` (useful for
    text-only filters that only need class, OCR text, and icons).
    """
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
        if include_geometry:
            _bx, _by, bw, bh = d.bbox
            geometry_part = f" center=[{d.cx},{d.cy}] w={bw} h={bh}"
        else:
            geometry_part = ""
        lines.append(
            f"[index {i}] class={_candidate_class_label(d.class_name)}"
            f"{geometry_part}{text_part}{icon_part}"
        )
    return "\n".join(lines)


def _visible_anchor_text(text: str | None) -> str:
    """OCR text with PUA codepoints stripped; empty when nothing visible remains."""
    if not text:
        return ""
    return "".join(ch for ch in text if not is_pua_char(ch)).strip()


def _detection_anchor_label(d: UiDetection) -> str:
    """Hub-style label such as 「文件」文字 or 「下載」圖示."""
    for icon in d.icons or []:
        if not isinstance(icon, dict):
            continue
        label = str(icon.get("chinese_id") or icon.get("id") or "").strip()
        if label:
            return f"「{label}」圖示"

    visible = _visible_anchor_text(d.text)
    if visible:
        suffix = _ANCHOR_SUFFIX_BY_CLASS.get(d.class_name)
        if suffix:
            if d.class_name == "input":
                return f"「{visible}」文字所在的輸入欄"
            return f"「{visible}」{suffix}"
        return f"「{visible}」"

    return _ANCHOR_SUFFIX_BY_CLASS.get(d.class_name, d.class_name or "元素")


def _offset_direction_parts(dx: int, dy: int) -> list[str]:
    """Traditional Chinese direction+distance parts; dominant axis first."""
    parts: list[str] = []
    include_x = abs(dx) >= _NEIGHBOR_OFFSET_THRESHOLD_PX
    include_y = abs(dy) >= _NEIGHBOR_OFFSET_THRESHOLD_PX
    if not include_x and not include_y:
        # Nearly overlapping: still emit the larger axis so distance is visible.
        if abs(dx) >= abs(dy) and dx != 0:
            include_x = True
        elif dy != 0:
            include_y = True
        else:
            return parts

    def _x_part() -> str:
        return f"右方{dx}個像素" if dx > 0 else f"左方{abs(dx)}個像素"

    def _y_part() -> str:
        return f"下方{dy}個像素" if dy > 0 else f"上方{abs(dy)}個像素"

    if include_x and include_y:
        if abs(dx) >= abs(dy):
            parts.extend([_x_part(), _y_part()])
        else:
            parts.extend([_y_part(), _x_part()])
    elif include_x:
        parts.append(_x_part())
    else:
        parts.append(_y_part())
    return parts


def _relative_neighbor_phrase(self: UiDetection, neighbor: UiDetection) -> str:
    """Describe ``neighbor`` relative to ``self``, e.g. 上方32個像素有「下載」文字."""
    dx = neighbor.cx - self.cx
    dy = neighbor.cy - self.cy
    label = _detection_anchor_label(neighbor)
    parts = _offset_direction_parts(dx, dy)
    if not parts:
        return f"附近有{label}"
    return f"{'、'.join(parts)}有{label}"


def _two_nearest_indices(detections: list[UiDetection], index: int) -> list[int]:
    """Return up to two other candidate indices closest to ``detections[index]`` by center."""
    if index < 0 or index >= len(detections) or len(detections) < 2:
        return []
    origin = detections[index]
    ranked: list[tuple[int, int]] = []
    for j, other in enumerate(detections):
        if j == index:
            continue
        dx = other.cx - origin.cx
        dy = other.cy - origin.cy
        ranked.append((dx * dx + dy * dy, j))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [j for _, j in ranked[:2]]


def _format_ui_candidates_relational(
    anchors: list[UiDetection],
    *,
    neighbors: list[UiDetection] | None = None,
) -> str:
    """Format picker candidates as labels plus two-nearest-neighbor spatial phrases.

    Selectable rows are only ``anchors`` (0-based indices). External ``neighbors``
    (e.g. nearby landmarks) are cited in phrases only, and each such neighbor is
    assigned exclusively to its closest anchor so distant duplicates cannot claim it.
    """

    def _identity_key(d: UiDetection) -> tuple[Any, ...]:
        icon_ids = tuple(
            sorted(
                str(ii.get("chinese_id", ""))
                for ii in (d.icons or [])
                if ii.get("chinese_id")
            )
        )
        return (d.bbox, d.cx, d.cy, d.class_id, d.text or "", icon_ids)

    def _dist2(a: UiDetection, b: UiDetection) -> int:
        dx = a.cx - b.cx
        dy = a.cy - b.cy
        return dx * dx + dy * dy

    exclusive_neighbors: list[UiDetection] = []
    if neighbors:
        seen = {_identity_key(d) for d in anchors}
        for neighbor in neighbors:
            key = _identity_key(neighbor)
            if key in seen:
                continue
            seen.add(key)
            exclusive_neighbors.append(neighbor)

    assigned: list[list[UiDetection]] = [[] for _ in anchors]
    for neighbor in exclusive_neighbors:
        best_i = min(
            range(len(anchors)),
            key=lambda i: (_dist2(anchors[i], neighbor), i),
        )
        assigned[best_i].append(neighbor)

    lines: list[str] = []
    for i, anchor in enumerate(anchors):
        label = _detection_anchor_label(anchor)
        eligible = [anchors[j] for j in range(len(anchors)) if j != i]
        eligible.extend(assigned[i])
        if not eligible:
            lines.append(f"[index {i}] {label}")
            continue
        ranked = sorted(eligible, key=lambda d: (_dist2(anchor, d), d.cx, d.cy))
        clauses = [_relative_neighbor_phrase(anchor, d) for d in ranked[:2]]
        lines.append(f"[index {i}] {label}（{'、'.join(clauses)}）")
    return "\n".join(lines)


async def _select_center_with_ollama(
    anchor_instruction: str,
    anchor_candidates: list[UiDetection],
    image_paths: list[str],
    *,
    neighbor_candidates: list[UiDetection] | None = None,
    nearby_labels: list[str] | None = None,
) -> int:
    """
    Ask Ollama for the best candidate index (0-based into ``detections``).

    ``neighbor_context`` may supply nearby landmarks for relational phrases.
    ``nearby_labels`` are shown as disambiguation hints (not selectable targets).

    Falls back to :func:`request_json_with_retry` if the first reply cannot be parsed.
    """
    if not anchor_candidates:
        raise ValueError("no candidates to pick from")
    candidates_text = _format_ui_candidates_relational(
        anchor_candidates,
        neighbors=neighbor_candidates,
    )
    nearby_text = (
        ", ".join(label.strip() for label in (nearby_labels or []) if label.strip())
        or "(none)"
    )
    prompt = get_prompt("ui_element_selection").format(
        instruction=anchor_instruction,
        nearby_text=nearby_text,
        candidates_text=candidates_text,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "images": image_paths,
        },
    ]
    n = len(anchor_candidates)
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
