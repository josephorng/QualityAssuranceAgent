from __future__ import annotations

import json
import re
from typing import Any


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_markdown_fences(text: str) -> str:
    """Remove a leading/trailing ``` or ```json fence if present."""
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    first_nl = text.find("\n")
    text = text[first_nl + 1 :] if first_nl != -1 else text[3:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


def _remove_trailing_commas(text: str) -> str:
    """Drop commas that appear immediately before ``}`` or ``]``."""
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def repair_json_object_text(raw: str) -> str:
    """Best-effort syntactic repair for common LLM JSON mistakes.

    Handles markdown fences, trailing commas, wrong closers (``]`` vs ``}``),
    unclosed strings, and missing closing brackets.
    """
    text = _strip_markdown_fences(raw or "")
    start = text.find("{")
    if start == -1:
        return text.strip()
    text = text[start:]

    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text)
        return text[:end]
    except json.JSONDecodeError:
        pass

    text = _remove_trailing_commas(text)
    text = _repair_brackets_and_close(text)
    return _remove_trailing_commas(text)


def _repair_brackets_and_close(text: str) -> str:
    """Walk ``text`` and fix mismatched / missing brackets outside of strings."""
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch == "{":
            stack.append("}")
            out.append(ch)
        elif ch == "[":
            stack.append("]")
            out.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                out.append(ch)
            elif stack:
                # Wrong closer (e.g. ``]`` where ``}`` was expected).
                out.append(stack.pop())
            # else: drop stray closer
        else:
            out.append(ch)

    if in_string:
        out.append('"')
    # Drop a dangling trailing comma before we append missing closers.
    while out and out[-1] in " \t\r\n":
        out.pop()
    if out and out[-1] == ",":
        out.pop()
    while stack:
        out.append(stack.pop())
    return "".join(out)


def extract_json_object_string(raw: str) -> str:
    """Extract the last JSON object from possibly fenced/free-form model output."""
    text = _strip_markdown_fences(raw or "")
    decoder = json.JSONDecoder()
    last: str | None = None
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            _, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        last = text[start:end]
        idx = end
    if last is not None:
        return last

    # Try repairing from each ``{`` when raw_decode cannot consume a span.
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            break
        repaired = repair_json_object_text(text[start:])
        try:
            _, end = decoder.raw_decode(repaired)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        last = repaired[:end]
        idx = start + 1
    if last is not None:
        return last

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    if start != -1:
        return repair_json_object_text(text[start:])
    return text


def parse_json_object(raw: str, *, empty_error: str, decode_error_prefix: str) -> dict[str, Any]:
    preview = (raw or "")[:240]
    candidates: list[str] = []
    for candidate in (
        extract_json_object_string(raw),
        repair_json_object_text(raw),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if not candidates or not any(candidates):
        raise ValueError(empty_error)

    last_exc: json.JSONDecodeError | None = None
    for json_text in candidates:
        try:
            out = json.loads(json_text)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
        if not isinstance(out, dict):
            raise ValueError(
                f"JSON must be an object, got {type(out).__name__}; preview={preview!r}"
            )
        return out

    detail = f" ({last_exc})" if last_exc is not None else ""
    raise ValueError(f"{decode_error_prefix}{detail}; preview={preview!r}")
