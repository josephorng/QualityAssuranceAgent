from __future__ import annotations

from cua_mcp.llm_json import extract_json_object_string, parse_json_object


def test_extract_json_object_string_takes_last_of_two_objects() -> None:
    raw = (
        '```json\n{"accomplished": false, "branch": "retry", "target_step": null}\n```\n'
        "Wait, let me re-evaluate.\n"
        '```json\n{"accomplished": false, "branch": "goto", "target_step": 1}\n```'
    )

    text = extract_json_object_string(raw)
    payload = parse_json_object(
        raw,
        empty_error="empty",
        decode_error_prefix="decode",
    )

    assert '"branch": "goto"' in text
    assert payload["branch"] == "goto"
    assert payload["target_step"] == 1


def test_extract_json_object_string_single_fenced_object() -> None:
    raw = '```json\n{"status":"failed","reason":"target not on screen"}\n```'
    payload = parse_json_object(raw, empty_error="empty", decode_error_prefix="decode")
    assert payload["status"] == "failed"


def test_extract_json_object_string_nested_braces_in_string() -> None:
    raw = '{"reason":"use {retry} then {goto}","branch":"advance"}'
    payload = parse_json_object(raw, empty_error="empty", decode_error_prefix="decode")
    assert payload["branch"] == "advance"
    assert "{goto}" in payload["reason"]


def test_extract_json_object_string_falls_back_when_no_object_decodes() -> None:
    raw = '{"status":"completed","reason":"say "click" now"}'
    text = extract_json_object_string(raw)
    assert text.startswith("{")
    assert text.endswith("}")
