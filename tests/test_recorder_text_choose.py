from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.recorder.text_choose import (
    apply_text_choice_hard_rules,
    build_choice_candidates,
    choose_text_input_with_llm,
    is_masked_ocr_text,
    needs_llm_text_choice,
    texts_equivalent,
)


def test_texts_equivalent_ignores_case_and_trailing_caret() -> None:
    assert texts_equivalent("Chrome", "chrome")
    assert texts_equivalent("office|", "office")
    assert not texts_equivalent("winwinmaster7", "winmaster7")


def test_is_masked_ocr_text() -> None:
    assert is_masked_ocr_text("********")
    assert is_masked_ocr_text("••••")
    assert not is_masked_ocr_text("password")
    assert not is_masked_ocr_text("")


def test_needs_llm_text_choice_skips_hard_rule_paths() -> None:
    assert not needs_llm_text_choice(
        {
            "recorded_text": "chrome",
            "ocr_text": "Chrome",
            "ocr_options": ["Chrome"],
            "source": "recorded",
        }
    )
    assert not needs_llm_text_choice(
        {
            "recorded_text": "!QAZ2wsx",
            "ocr_text": "********",
            "ocr_options": ["********"],
            "source": "recorded",
        }
    )
    assert not needs_llm_text_choice(
        {
            "recorded_text": "",
            "ocr_text": "你好",
            "ocr_options": ["你好"],
            "source": "ocr",
        }
    )
    assert not needs_llm_text_choice(
        {
            "recorded_text": "a",
            "ocr_text": "b",
            "ocr_options": ["b"],
            "source": "user",
        }
    )
    assert not needs_llm_text_choice(
        {
            "recorded_text": "offioffice",
            "uia_text": "office",
            "ocr_text": "office",
            "source": "recorded",
        }
    )


def test_needs_llm_text_choice_skips_chinese_and_english_disagreement() -> None:
    # Chinese OCR → hard rule picks OCR; no LLM.
    assert not needs_llm_text_choice(
        {
            "recorded_text": "2832u04cj842k7g6c.4",
            "ocr_text": "打電話的時候",
            "ocr_options": ["打電話的時候"],
            "source": "recorded",
        }
    )
    # English recorded → hard rule picks recorded; no LLM.
    assert not needs_llm_text_choice(
        {
            "recorded_text": "winwinmaster7",
            "ocr_text": "winmaster7",
            "ocr_options": ["winmaster7"],
            "source": "recorded",
        }
    )


def test_apply_hard_rules_prefers_uia_focused_value() -> None:
    updated = apply_text_choice_hard_rules(
        {
            "recorded_text": "offioffice",
            "uia_text": "office",
            "ocr_text": "office",
            "ocr_options": ["office"],
            "resolved_text": "offioffice",
            "source": "recorded",
            "reason": "prefer recorded",
        }
    )
    assert updated["resolved_text"] == "office"
    assert updated["source"] == "uia"
    assert "UIA" in updated["reason"]


def test_apply_hard_rules_prefers_short_uia_over_longer_cjk_ocr() -> None:
    """Strict UIA-first: non-empty UIA wins even when CJK OCR is a longer extension."""
    updated = apply_text_choice_hard_rules(
        {
            "recorded_text": "jt",
            "uia_text": "今天",
            "ocr_text": "今天是|作麼日子",
            "ocr_options": ["今天是|作麼日子"],
            "resolved_text": "今天是|作麼日子",
            "source": "ocr",
            "reason": "Chinese typing; prefer OCR",
        }
    )
    assert updated["resolved_text"] == "今天"
    assert updated["source"] == "uia"
    assert "UIA" in updated["reason"]


def test_apply_hard_rules_skips_empty_uia() -> None:
    updated = apply_text_choice_hard_rules(
        {
            "recorded_text": "offioffice",
            "uia_text": "   ",
            "ocr_text": "office",
            "ocr_options": ["office"],
            "resolved_text": "offioffice",
            "source": "recorded",
            "reason": "prefer recorded",
        }
    )
    assert updated["resolved_text"] == "offioffice"
    assert updated["source"] == "recorded"
    assert "English" in updated["reason"]


def test_apply_hard_rules_prefers_ocr_for_chinese() -> None:
    updated = apply_text_choice_hard_rules(
        {
            "recorded_text": "nihao",
            "uia_text": None,
            "ocr_text": "你好",
            "ocr_options": ["你好"],
            "resolved_text": "nihao",
            "source": "recorded",
            "reason": "prefer recorded",
        }
    )
    assert updated["resolved_text"] == "你好"
    assert updated["source"] == "ocr"
    assert "Chinese" in updated["reason"]


def test_apply_hard_rules_prefers_recorded_for_english_disagreement() -> None:
    updated = apply_text_choice_hard_rules(
        {
            "recorded_text": "winwinmaster7",
            "uia_text": None,
            "ocr_text": "winmaster7",
            "ocr_options": ["winmaster7"],
            "resolved_text": "winwinmaster7",
            "source": "recorded",
            "reason": "prefer recorded",
        }
    )
    assert updated["resolved_text"] == "winwinmaster7"
    assert updated["source"] == "recorded"
    assert "English" in updated["reason"]


def test_apply_hard_rules_prefers_recorded_for_masked_ocr() -> None:
    updated = apply_text_choice_hard_rules(
        {
            "recorded_text": "!QAZ2wsx",
            "ocr_text": "********",
            "ocr_options": ["********"],
            "resolved_text": "!QAZ2wsx",
            "source": "recorded",
            "reason": "prefer recorded",
        }
    )
    assert updated["resolved_text"] == "!QAZ2wsx"
    assert updated["source"] == "recorded"
    assert "masked" in updated["reason"]


def test_build_choice_candidates_includes_uia_after_recorded() -> None:
    candidates = build_choice_candidates(
        {
            "recorded_text": "offioffice",
            "uia_text": "office",
            "ocr_text": "office",
            "ocr_options": ["office"],
        }
    )
    assert candidates == ["offioffice", "office"]


def test_build_choice_candidates_drops_masked_ocr() -> None:
    candidates = build_choice_candidates(
        {
            "recorded_text": "secret",
            "ocr_text": "******",
            "ocr_options": ["******", "secret"],
        }
    )
    assert candidates == ["secret"]


@pytest.mark.asyncio
async def test_choose_text_input_with_llm_skips_when_hard_rules_decide() -> None:
    resolution = {
        "recorded_text": "winwinmaster7",
        "ocr_text": "winmaster7",
        "ocr_options": ["winmaster7"],
        "resolved_text": "winwinmaster7",
        "source": "recorded",
        "reason": "prefer recorded",
    }
    with patch(
        "src.recorder.text_choose.request_json_with_retry",
        new=AsyncMock(),
    ) as mock_llm:
        updated = await choose_text_input_with_llm(resolution)

    mock_llm.assert_not_awaited()
    assert updated["resolved_text"] == "winwinmaster7"
    assert updated["source"] == "recorded"


@pytest.mark.asyncio
async def test_choose_skips_llm_when_texts_agree() -> None:
    resolution = {
        "recorded_text": "chrome",
        "ocr_text": "Chrome|",
        "ocr_options": ["Chrome|"],
        "resolved_text": "chrome",
        "source": "recorded",
        "reason": "prefer recorded",
    }
    with patch(
        "src.recorder.text_choose.request_json_with_retry",
        new=AsyncMock(),
    ) as mock_llm:
        updated = await choose_text_input_with_llm(resolution)

    mock_llm.assert_not_awaited()
    assert updated["resolved_text"] == "chrome"
    assert updated["source"] == "recorded"


@pytest.mark.asyncio
async def test_choose_skips_llm_for_chinese_ocr() -> None:
    resolution = {
        "recorded_text": "nihao",
        "ocr_text": "你好",
        "ocr_options": ["你好"],
        "resolved_text": "nihao",
        "source": "recorded",
        "reason": "prefer recorded",
    }
    with patch(
        "src.recorder.text_choose.request_json_with_retry",
        new=AsyncMock(),
    ) as mock_llm:
        updated = await choose_text_input_with_llm(resolution)

    mock_llm.assert_not_awaited()
    assert updated["resolved_text"] == "你好"
    assert updated["source"] == "ocr"
