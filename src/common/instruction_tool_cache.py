from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json, write_json
from src.common.settings import load_settings

_CACHE_VERSION = 1
_CACHE_FILENAME = "instruction_tool_cache.json"
_ROLE_ASSISTANT = "assistant"


def cache_file_path() -> Path:
    return Path(load_settings().runs_dir) / _CACHE_FILENAME


def normalize_instruction(text: str) -> str:
    return text.strip()


def _default_cache() -> dict[str, Any]:
    return {"version": _CACHE_VERSION, "entries": {}}


def load_cache(path: Path | None = None) -> dict[str, Any]:
    target = path or cache_file_path()
    raw = read_json(target, _default_cache())
    if not isinstance(raw, dict):
        return _default_cache()
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        raw["entries"] = {}
    raw["version"] = _CACHE_VERSION
    return raw


def save_cache(data: dict[str, Any], path: Path | None = None) -> None:
    target = path or cache_file_path()
    write_json(target, data)


def extract_tool_calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect ordered tool calls from assistant messages in a step transcript."""
    calls: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != _ROLE_ASSISTANT:
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            func = call.get("function")
            if not isinstance(func, dict):
                continue
            name = func.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            arguments = func.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append({"name": name.strip(), "arguments": dict(arguments)})
    return calls


def lookup_tool_calls(instruction: str, path: Path | None = None) -> list[dict[str, Any]] | None:
    key = normalize_instruction(instruction)
    if not key:
        return None
    cache = load_cache(path)
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(key)
    if not isinstance(entry, dict):
        return None
    tool_calls = entry.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            return None
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(arguments, dict):
            arguments = {}
        normalized.append({"name": name.strip(), "arguments": dict(arguments)})
    return normalized


def upsert_tool_calls(
    instruction: str,
    tool_calls: list[dict[str, Any]],
    *,
    source_run_id: str,
    path: Path | None = None,
) -> None:
    key = normalize_instruction(instruction)
    if not key or not tool_calls:
        return
    cache = load_cache(path)
    entries = cache.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        cache["entries"] = entries
    entries[key] = {
        "instruction": key,
        "tool_calls": tool_calls,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": source_run_id,
    }
    save_cache(cache, path)
