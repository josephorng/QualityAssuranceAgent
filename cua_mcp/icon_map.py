from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_ICON_MAP_PATH = Path(__file__).resolve().parent / "read_screen_text" / "icon_map.json"


@lru_cache(maxsize=1)
def load_icon_map() -> dict[str, Any]:
    if not _ICON_MAP_PATH.is_file():
        raise FileNotFoundError(f"icon map not found: {_ICON_MAP_PATH}")
    raw = json.loads(_ICON_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("icon map must be a JSON object")
    return raw


def is_pua_char(char: str) -> bool:
    """True when ``char`` is in a Unicode Private Use Area (BMP or supplementary planes)."""
    if not char:
        return False
    cp = ord(char)
    if 0xE000 <= cp <= 0xF8FF:
        return True
    if 0xF0000 <= cp <= 0xFFFFD:
        return True
    if 0x100000 <= cp <= 0x10FFFD:
        return True
    return False


def text_has_pua(text: str) -> bool:
    return any(is_pua_char(ch) for ch in text or "")


@lru_cache(maxsize=1)
def _unknown_icon_map_entry() -> dict[str, str]:
    for key, value in load_icon_map().items():
        if not isinstance(value, dict):
            continue
        if value.get("id") == "unknown_icon":
            return {
                "chinese_id": str(value.get("chinese_id", "未知圖示")),
                "description": str(
                    value.get("description", "Unknown or unclear icon type.")
                ),
            }
    return {
        "chinese_id": "未知圖示",
        "description": "Unknown or unclear icon type.",
    }


def unknown_icon_record(*, pua: str = "") -> dict[str, Any]:
    meta = _unknown_icon_map_entry()
    return {
        "pua": pua,
        "chinese_id": meta["chinese_id"],
        "icon_description": meta["description"],
    }


def is_unknown_icon_record(record: dict[str, Any]) -> bool:
    """True when ``record`` came from an unmapped PUA codepoint."""
    pua = record.get("pua")
    if isinstance(pua, str) and pua and is_pua_char(pua):
        return lookup_pua_icon(pua) is None
    return str(record.get("chinese_id", "")) == _unknown_icon_map_entry()["chinese_id"]


def lookup_pua_icon(char: str) -> dict[str, Any] | None:
    if not char:
        return None
    value = load_icon_map().get(char)
    return value if isinstance(value, dict) else None


def map_pua_in_text(text: str) -> str:
    """Replace each PUA codepoint with its ``chinese_id``; unmapped PUA uses ``unknown_icon``."""
    if not text:
        return ""
    unknown = _unknown_icon_map_entry()["chinese_id"]
    parts: list[str] = []
    for ch in text:
        if not is_pua_char(ch):
            parts.append(ch)
            continue
        mapped = lookup_pua_icon(ch)
        if mapped is None:
            parts.append(unknown)
        else:
            parts.append(str(mapped.get("chinese_id", unknown)))
    return "".join(parts).strip()


def describe_text_icons(text: str) -> list[dict[str, Any]]:
    """Map PUA codepoints in ``text`` to icon metadata; unmapped PUA uses ``unknown_icon``."""
    icons: list[dict[str, Any]] = []
    for ch in text or "":
        if not is_pua_char(ch):
            continue
        mapped = lookup_pua_icon(ch)
        if mapped is None:
            icons.append(unknown_icon_record(pua=ch))
            continue
        icons.append(
            {
                "pua": ch,
                "chinese_id": str(mapped.get("chinese_id", "")),
                "icon_description": str(mapped.get("description", "")),
            }
        )
    return icons
