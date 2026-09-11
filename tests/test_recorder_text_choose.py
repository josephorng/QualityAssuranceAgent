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


def test_needs_llm_text_choice_skips_agreement_and_masks() -> None:
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


def test_needs_llm_text_choice_when_recorded_and_ocr_disagree() -> None:
    assert needs_llm_text_choice(
        {
            "recorded_text": "2832u04cj842k7g6c.4",
            "ocr_text": "打電話的時候",
            "ocr_options": ["打電話的時候"],
            "source": "recorded",
        }
    )
    assert needs_llm_text_choice(
        {
            "recorded_text": "winwinmaster7",
            "ocr_text": "winmaster7",
            "ocr_options": ["winmaster7"],
            "source": "recorded",
        }
    )


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
async def test_choose_text_input_with_llm_picks_candidate() -> None:
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
        new=AsyncMock(return_value={"chosen_index": 1, "reason": "doubled chars"}),
    ) as mock_llm:
        updated = await choose_text_input_with_llm(resolution)

    mock_llm.assert_awaited_once()
    assert updated["resolved_text"] == "winmaster7"
    assert updated["source"] == "llm"
    assert "doubled chars" in updated["reason"]
    kwargs = mock_llm.await_args.kwargs
    assert kwargs["append_image_sizes"] is False
    assert "images" not in kwargs["messages"][0]


@pytest.mark.asyncio
async def test_choose_text_input_with_llm_soft_fails_to_existing() -> None:
    resolution = {
        "recorded_text": "abc",
        "ocr_text": "abd",
        "ocr_options": ["abd"],
        "resolved_text": "abc",
        "source": "recorded",
        "reason": "prefer recorded",
    }
    with patch(
        "src.recorder.text_choose.request_json_with_retry",
        new=AsyncMock(side_effect=TimeoutError("boom")),
    ):
        updated = await choose_text_input_with_llm(resolution)

    assert updated["resolved_text"] == "abc"
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
