"""Helpers for normalizing odd tokens that models embed in tool-call text."""

from __future__ import annotations

from typing import Any

# Gemma / vLLM tool-call syntax wraps string args with this sentinel.
LLM_QUOTE_WRAPPER = '<|"|>'


def strip_llm_quote_wrappers(value: Any) -> Any:
    """Recursively remove ``<|"|>`` wrappers from strings in tool arguments.

    Models sometimes emit ``<|"|>輸入欄<|"|>`` or list items like
    ``<|"|>在「帳號」文字的右邊<|"|>``. Those wrappers break directed nearby-side
    parsing and must be stripped before enrich / execute.
    """
    if isinstance(value, str):
        return value.replace(LLM_QUOTE_WRAPPER, "")
    if isinstance(value, list):
        return [strip_llm_quote_wrappers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_llm_quote_wrappers(item) for item in value)
    if isinstance(value, dict):
        return {key: strip_llm_quote_wrappers(item) for key, item in value.items()}
    return value
