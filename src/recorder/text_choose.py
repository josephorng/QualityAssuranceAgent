"""Choose recorded vs OCR typed text (hard rules first; LLM disagreement-only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from cua_mcp.selection_engine import request_json_with_retry
from src.common.prompting import get_prompt
from src.recorder.coalesce import text_contains_cjk
from src.recorder.text_resolve import _strip_ocr_caret

_MASK_CHARS = frozenset("*●•∗＊‧∙·⬤✖✕")

_TEXT_CHOICE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chosen_index": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["chosen_index", "reason"],
}


def normalize_text_for_compare(text: str) -> str:
    """Normalize typed/OCR text for agreement checks."""
    return _strip_ocr_caret(str(text or "")).strip().casefold()


def texts_equivalent(left: str, right: str) -> bool:
    """True when two strings match after caret/case/whitespace normalization."""
    a = normalize_text_for_compare(left)
    b = normalize_text_for_compare(right)
    return bool(a) and a == b


def is_masked_ocr_text(text: str) -> bool:
    """True when OCR looks like a password/mask field (**** / ••••)."""
    cleaned = _strip_ocr_caret(str(text or "")).strip()
    if not cleaned:
        return False
    maskish = sum(1 for ch in cleaned if ch in _MASK_CHARS or ch.isspace())
    has_mask = any(ch in _MASK_CHARS for ch in cleaned)
    return has_mask and maskish >= max(1, int(round(len(cleaned) * 0.8)))


def normalize_uia_text(text: Any) -> str:
    """Normalize a UIA focused-value string; empty when missing/whitespace-only."""
    return _strip_ocr_caret(str(text or "")).strip()


def ime_chinese_flag(text_resolution: dict[str, Any]) -> bool | None:
    """Return recorded IME 中/英 flag, or None when missing (legacy / unknown)."""
    raw = text_resolution.get("ime")
    if not isinstance(raw, dict):
        return None
    chinese = raw.get("chinese")
    return chinese if isinstance(chinese, bool) else None


def collect_ocr_candidates(text_resolution: dict[str, Any]) -> list[str]:
    """Unique OCR strings from ``ocr_text`` / ``ocr_options`` (order preserved)."""
    options: list[str] = []
    seen: set[str] = set()

    def _add(raw: Any) -> None:
        text = _strip_ocr_caret(str(raw or "")).strip()
        if not text or text in seen:
            return
        seen.add(text)
        options.append(text)

    _add(text_resolution.get("ocr_text"))
    raw_options = text_resolution.get("ocr_options")
    if isinstance(raw_options, list):
        for item in raw_options:
            _add(item)
    return options


def build_choice_candidates(text_resolution: dict[str, Any]) -> list[str]:
    """Candidate list for review/LLM: recorded, then UIA, then non-masked OCR."""
    recorded = _strip_ocr_caret(str(text_resolution.get("recorded_text") or "")).strip()
    uia = normalize_uia_text(text_resolution.get("uia_text"))
    candidates: list[str] = []
    seen: set[str] = set()
    if recorded:
        candidates.append(recorded)
        seen.add(recorded)
    if uia and uia not in seen:
        candidates.append(uia)
        seen.add(uia)
    for ocr in collect_ocr_candidates(text_resolution):
        if is_masked_ocr_text(ocr):
            continue
        if ocr in seen:
            continue
        seen.add(ocr)
        candidates.append(ocr)
    return candidates


def needs_llm_text_choice(text_resolution: dict[str, Any]) -> bool:
    """True only when hard rules leave a real multi-candidate ambiguity.

    UIA, IME-gated Chinese/English, and legacy CJK-OCR paths skip the LLM.
    """
    source = str(text_resolution.get("source") or "").strip().lower()
    if source in {"user", "llm", "uia"}:
        return False

    if normalize_uia_text(text_resolution.get("uia_text")):
        return False

    ime_chinese = ime_chinese_flag(text_resolution)
    if ime_chinese is not None:
        # Recorded IME mode decides Chinese vs English without LLM.
        return False

    ocr_candidates = collect_ocr_candidates(text_resolution)
    if any(text_contains_cjk(text) for text in ocr_candidates):
        return False

    recorded = _strip_ocr_caret(str(text_resolution.get("recorded_text") or "")).strip()
    if recorded:
        # English typing hard rule always prefers recorded.
        return False

    if not ocr_candidates:
        return False

    if all(is_masked_ocr_text(text) for text in ocr_candidates):
        return False

    usable = build_choice_candidates(text_resolution)
    return len(usable) >= 2


def apply_text_choice_hard_rules(text_resolution: dict[str, Any]) -> dict[str, Any]:
    """Prefer UIA, then IME-gated Chinese OCR / English recorded; skip LLM."""
    source = str(text_resolution.get("source") or "").strip().lower()
    if source in {"user", "llm"}:
        return text_resolution

    recorded = _strip_ocr_caret(str(text_resolution.get("recorded_text") or "")).strip()
    uia = normalize_uia_text(text_resolution.get("uia_text"))
    ocr_candidates = collect_ocr_candidates(text_resolution)
    primary_ocr = ocr_candidates[0] if ocr_candidates else ""
    ime_chinese = ime_chinese_flag(text_resolution)
    cjk_ocr = next((text for text in ocr_candidates if text_contains_cjk(text)), None)

    updated = dict(text_resolution)
    if uia:
        updated["uia_text"] = uia
    elif "uia_text" not in updated:
        updated["uia_text"] = None

    # 1. Non-empty UIA focused value wins.
    if uia:
        updated["resolved_text"] = uia
        updated["source"] = "uia"
        updated["reason"] = "prefer UIA focused value"
        return updated

    # 2. Masked OCR → recorded (passwords).
    if ocr_candidates and all(is_masked_ocr_text(text) for text in ocr_candidates):
        if recorded:
            updated["resolved_text"] = recorded
            updated["source"] = "recorded"
            updated["reason"] = "masked OCR; prefer recorded"
            return updated

    # 3. English IME for the burst → recorded (ignore nearby CJK UI chrome).
    if ime_chinese is False and recorded:
        updated["resolved_text"] = recorded
        updated["source"] = "recorded"
        updated["reason"] = "English IME; prefer recorded"
        return updated

    # 4. Chinese IME → focus-local CJK OCR when present; else recorded (ASCII under 中).
    if ime_chinese is True:
        if cjk_ocr:
            updated["resolved_text"] = cjk_ocr
            updated["source"] = "ocr"
            updated["reason"] = "Chinese IME; prefer OCR"
            return updated
        if recorded:
            updated["resolved_text"] = recorded
            updated["source"] = "recorded"
            updated["reason"] = "Chinese IME; no CJK OCR, prefer recorded"
            return updated

    # 5. Legacy (no ime): Chinese typing inferred from CJK in OCR → OCR.
    if ime_chinese is None and cjk_ocr:
        updated["resolved_text"] = cjk_ocr
        updated["source"] = "ocr"
        updated["reason"] = "Chinese typing; prefer OCR"
        return updated

    # 6. Recorded agrees with an OCR alternate → recorded.
    if recorded and any(texts_equivalent(recorded, text) for text in ocr_candidates):
        updated["resolved_text"] = recorded
        updated["source"] = "recorded"
        updated["reason"] = "recorded matches OCR alternate; prefer recorded"
        return updated

    # 7. English typing → recorded when present.
    if recorded:
        updated["resolved_text"] = recorded
        updated["source"] = "recorded"
        updated["reason"] = (
            "English typing; prefer recorded"
            if ocr_candidates
            else "prefer recorded text; vision produced no OCR alternate"
        )
        return updated

    # 8. Fallbacks.
    if primary_ocr:
        updated["resolved_text"] = primary_ocr
        updated["source"] = "ocr"
        updated["reason"] = "recorded text empty; used OCR"
        return updated

    updated["resolved_text"] = recorded
    updated["source"] = "recorded"
    updated["reason"] = "vision produced no text; kept recorded text"
    return updated


def _parse_text_choice_reply(raw: str, *, candidate_count: int) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    index = data.get("chosen_index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("chosen_index missing or not an int")
    if index < 0 or index >= candidate_count:
        raise ValueError(f"chosen_index out of range: {index}")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "llm choice"
    return {"chosen_index": index, "reason": reason.strip()}


def _candidates_block(candidates: list[str]) -> str:
    lines = [f"[{i}] {text}" for i, text in enumerate(candidates)]
    return "\n".join(lines) if lines else "(none)"


async def choose_text_input_with_llm(
    text_resolution: dict[str, Any],
    *,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Pick among recorded/OCR candidates via text-only LLM; soft-fail to input."""
    gated = apply_text_choice_hard_rules(text_resolution)
    if not needs_llm_text_choice(gated):
        return gated

    candidates = build_choice_candidates(gated)
    if len(candidates) < 2:
        return gated

    recorded = _strip_ocr_caret(str(gated.get("recorded_text") or "")).strip()
    prompt = get_prompt("recording_text_input_choose").format(
        recorded_text=recorded or "(empty)",
        candidates_block=_candidates_block(candidates),
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    try:
        result = await request_json_with_retry(
            messages=messages,
            response_schema=_TEXT_CHOICE_RESPONSE_SCHEMA,
            parse_reply=lambda raw: _parse_text_choice_reply(
                raw, candidate_count=len(candidates)
            ),
            retry_instruction=get_prompt("recording_text_input_choose_retry"),
            log_info=log_info,
            append_image_sizes=False,
        )
    except Exception as exc:
        if log_info is not None:
            log_info(
                f"choose_text_input_with_llm failed: {type(exc).__name__}: {exc}"
            )
        return gated

    chosen = candidates[int(result["chosen_index"])]
    updated = dict(gated)
    updated["resolved_text"] = chosen
    updated["source"] = "llm"
    updated["reason"] = f"llm: {result['reason']}"
    if log_info is not None:
        log_info(
            f"choose_text_input_with_llm chose index={result['chosen_index']} "
            f"text={chosen!r} reason={result['reason']!r}"
        )
    return updated
