"""Decode Microsoft New Phonetic (注音) keystrokes for zh-TW IME text resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# Traditional Chinese (Taiwan) — Microsoft Bopomofo / New Phonetic.
ZHUYIN_LANG_ID = 0x0404

# Standard Daqian / Microsoft Bopomofo key → 注音 symbol (documentation / debug).
BOPOMOFO_KEY_MAP: dict[str, str] = {
    ",": "ㄝ",
    "-": "ㄦ",
    ".": "ㄡ",
    "/": "ㄥ",
    "0": "ㄢ",
    "1": "ㄅ",
    "2": "ㄉ",
    "3": "ˇ",
    "4": "ˋ",
    "5": "ㄓ",
    "6": "ˊ",
    "7": "˙",
    "8": "ㄚ",
    "9": "ㄞ",
    ";": "ㄤ",
    "a": "ㄇ",
    "b": "ㄖ",
    "c": "ㄏ",
    "d": "ㄎ",
    "e": "ㄍ",
    "f": "ㄑ",
    "g": "ㄕ",
    "h": "ㄘ",
    "i": "ㄛ",
    "j": "ㄨ",
    "k": "ㄜ",
    "l": "ㄠ",
    "m": "ㄩ",
    "n": "ㄙ",
    "o": "ㄟ",
    "p": "ㄣ",
    "q": "ㄆ",
    "r": "ㄐ",
    "s": "ㄋ",
    "t": "ㄔ",
    "u": "ㄧ",
    "v": "ㄒ",
    "w": "ㄊ",
    "x": "ㄌ",
    "y": "ㄗ",
    "z": "ㄈ",
}

_TONE_KEYS = frozenset({"3", "4", "6", "7"})
_PHONEME_KEYS = frozenset(BOPOMOFO_KEY_MAP) - _TONE_KEYS
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_RUN_RE = re.compile(r"^[\x20-\x7e]+")


@dataclass(frozen=True)
class SyllableMatch:
    """One parsed Zhuyin syllable and its lexicon candidates."""

    key: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class DecodeResult:
    """Result of decoding recorded Zhuyin keystrokes (optional OCR reconcile)."""

    ok: bool
    literal_prefix: str = ""
    syllables: tuple[SyllableMatch, ...] = ()
    decoded_text: str = ""
    reconciled_text: str = ""
    ocr_compatible: bool = False
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def is_zhuyin_lang(lang_id: Any) -> bool:
    """True when ``lang_id`` is Traditional Chinese Taiwan (0x0404)."""
    try:
        return (int(lang_id) & 0xFFFF) == ZHUYIN_LANG_ID
    except (TypeError, ValueError):
        return False


def ime_is_zhuyin(ime: Any) -> bool:
    """True when IME payload is Chinese mode under zh-TW Zhuyin."""
    if not isinstance(ime, dict):
        return False
    if ime.get("chinese") is not True:
        return False
    return is_zhuyin_lang(ime.get("lang_id"))


@lru_cache(maxsize=1)
def _load_lexicon() -> dict[str, list[str]]:
    path = Path(__file__).resolve().parent / "data" / "bopomofo_lexicon.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, chars in raw.items():
        if not isinstance(key, str) or not isinstance(chars, list):
            continue
        cleaned = [str(ch) for ch in chars if isinstance(ch, str) and len(ch) == 1]
        if cleaned:
            out[key] = cleaned
    return out


def _normalize_syllable_key(token: str) -> str | None:
    """Normalize a keystroke token to a lexicon syllable key, or None."""
    raw = str(token or "").strip().lower()
    if not raw:
        return None
    # Space is first-tone commit in recording; cin first-tone keys omit tone.
    # Tone digits 3/4/6/7 stay in the key when present.
    if not all(ch in _PHONEME_KEYS or ch in _TONE_KEYS for ch in raw):
        return None
    if not any(ch in _PHONEME_KEYS for ch in raw):
        return None
    # At most one trailing tone key.
    body = raw
    tone = ""
    if body and body[-1] in _TONE_KEYS:
        tone = body[-1]
        body = body[:-1]
    if not body or any(ch in _TONE_KEYS for ch in body):
        return None
    if not all(ch in _PHONEME_KEYS for ch in body):
        return None
    return body + tone


def _lookup_candidates(key: str) -> tuple[str, ...] | None:
    lex = _load_lexicon()
    chars = lex.get(key)
    if not chars:
        # First-tone recordings sometimes still include a dangling digit-less key
        # that matches only with an explicit tone-less cin entry.
        return None
    return tuple(chars)


def _shared_ascii_prefix(recorded: str, ocr_hint: str | None) -> str:
    """Longest ASCII/digit prefix of ``recorded`` also present at start of OCR."""
    if not ocr_hint:
        return ""
    rec = str(recorded or "")
    ocr = str(ocr_hint or "")
    n = 0
    limit = min(len(rec), len(ocr))
    while n < limit:
        rc, oc = rec[n], ocr[n]
        if ord(rc) > 127 or ord(oc) > 127:
            break
        if rc != oc:
            break
        n += 1
    return rec[:n]


def _split_literal_and_syllable_token(token: str) -> tuple[str, str]:
    """Split a whitespace token into leading literal ASCII and syllable key body.

    Leading digits before the first Latin letter are treated as literal (numpad
    digits look identical to top-row Zhuyin keys in ``recorded_text`` alone).
    """
    raw = str(token or "")
    if not raw:
        return "", ""
    # Prefer letter-bearing syllable start.
    for i, ch in enumerate(raw):
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            return raw[:i], raw[i:]
    # Digit/punctuation-only token: try whole token as syllable (e.g. ``20``).
    return "", raw


def _parse_syllables(
    recorded: str,
    *,
    ocr_hint: str | None,
) -> tuple[str, list[SyllableMatch]] | None:
    """Parse ``recorded`` into literal prefix + syllable list, or None on failure."""
    text = str(recorded or "")
    if not text.strip():
        return None

    shared = _shared_ascii_prefix(text, ocr_hint)
    remainder = text[len(shared) :]
    literal_parts: list[str] = [shared] if shared else []
    syllables: list[SyllableMatch] = []

    # Whitespace separates syllables when space is used as 1st-tone commit.
    tokens = remainder.split(" ")
    # Trailing space yields an empty final token; drop empties but keep structure.
    for idx, token in enumerate(tokens):
        if token == "":
            continue
        lit, body = _split_literal_and_syllable_token(token)
        if lit:
            # Only allow an extra literal on the first non-empty token after shared.
            if syllables:
                return None
            literal_parts.append(lit)
        key = _normalize_syllable_key(body)
        if key is None:
            return None
        candidates = _lookup_candidates(key)
        if candidates is None:
            return None
        syllables.append(SyllableMatch(key=key, candidates=candidates))

    if not syllables:
        return None
    return "".join(literal_parts), syllables


def _default_decoded(literal: str, syllables: list[SyllableMatch]) -> str:
    chars = [syl.candidates[0] for syl in syllables]
    return literal + "".join(chars)


def _ocr_cjk_suffix(ocr_hint: str | None, literal: str) -> str | None:
    """CJK characters in OCR after the literal prefix, or None if unusable."""
    if not ocr_hint:
        return None
    ocr = str(ocr_hint)
    if literal and ocr.startswith(literal):
        tail = ocr[len(literal) :]
    else:
        # Fall back: strip a leading ASCII run, keep CJK from the rest.
        m = _ASCII_RUN_RE.match(ocr)
        tail = ocr[m.end() :] if m else ocr
    cjk = "".join(_CJK_RE.findall(tail))
    return cjk if cjk else None


def reconcile_with_ocr(
    literal: str,
    syllables: list[SyllableMatch],
    ocr_hint: str | None,
) -> tuple[str, bool]:
    """Return ``(text, ocr_compatible)`` after phonetic OCR reconcile.

    Same-sound OCR chars (今/金 for ``rup``) are kept; wrong-sound chars
    (合 for ``rup``) are replaced with the top lexicon candidate.
    """
    default = _default_decoded(literal, syllables)
    cjk = _ocr_cjk_suffix(ocr_hint, literal)
    if cjk is None:
        return default, False
    if len(cjk) != len(syllables):
        # Length mismatch — still try a prefix align when OCR is longer/shorter
        # by taking min length; treat as incompatible overall.
        n = min(len(cjk), len(syllables))
        if n == 0:
            return default, False
        out_chars: list[str] = []
        compatible = len(cjk) == len(syllables)
        for i in range(n):
            ch = cjk[i]
            cands = syllables[i].candidates
            if ch in cands:
                out_chars.append(ch)
            else:
                out_chars.append(cands[0])
                compatible = False
        if len(syllables) > n:
            compatible = False
            out_chars.extend(syl.candidates[0] for syl in syllables[n:])
        elif len(cjk) > n:
            compatible = False
        return literal + "".join(out_chars), compatible

    out_chars = []
    compatible = True
    for ch, syl in zip(cjk, syllables, strict=True):
        if ch in syl.candidates:
            out_chars.append(ch)
        else:
            out_chars.append(syl.candidates[0])
            compatible = False
    return literal + "".join(out_chars), compatible


def decode_zhuyin_recorded(
    recorded: str,
    *,
    ocr_hint: str | None = None,
) -> DecodeResult:
    """Decode Microsoft Zhuyin keystrokes; optionally reconcile with OCR.

    Returns ``ok=False`` when the string cannot be parsed as Zhuyin syllables
    (caller should keep existing OCR/recorded hard rules).
    """
    parsed = _parse_syllables(recorded, ocr_hint=ocr_hint)
    if parsed is None:
        return DecodeResult(ok=False, reason="zhuyin parse failed")
    literal, syllables = parsed
    decoded = _default_decoded(literal, syllables)
    reconciled, compatible = reconcile_with_ocr(literal, syllables, ocr_hint)
    return DecodeResult(
        ok=True,
        literal_prefix=literal,
        syllables=tuple(syllables),
        decoded_text=decoded,
        reconciled_text=reconciled,
        ocr_compatible=compatible,
        reason="zhuyin decode ok",
        extra={
            "syllable_keys": [s.key for s in syllables],
        },
    )


def apply_zhuyin_to_text_resolution(
    text_resolution: dict[str, Any],
    *,
    ocr_text: str | None = None,
) -> dict[str, Any]:
    """Attach ``zhuyin_text`` when IME is zh-TW Chinese and decode succeeds.

    Does not choose ``resolved_text``; ``apply_text_choice_hard_rules`` does that.
    """
    updated = dict(text_resolution)
    ime = updated.get("ime")
    if not ime_is_zhuyin(ime):
        return updated
    recorded = str(updated.get("recorded_text") or "")
    hint = ocr_text
    if hint is None:
        hint = updated.get("ocr_text")
        if isinstance(hint, str):
            pass
        else:
            hint = None
    result = decode_zhuyin_recorded(recorded, ocr_hint=hint if hint else None)
    if not result.ok:
        return updated
    updated["zhuyin_text"] = result.reconciled_text
    updated["zhuyin_decoded"] = result.decoded_text
    updated["zhuyin_ocr_compatible"] = result.ocr_compatible
    updated["zhuyin_syllables"] = list(result.extra.get("syllable_keys") or [])
    return updated
