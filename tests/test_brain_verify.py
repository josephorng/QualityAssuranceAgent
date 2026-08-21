from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain.module import BrainModule
from src.common.models import ScriptStepVerifyResult
from src.common.prompting import get_prompt


def _brain_for_process_step() -> BrainModule:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain.manager.set_step_log_context = MagicMock()
    brain.manager.clear_step_log_context = MagicMock()
    brain.script_lines = [
        "click search",
        "click calculator",
        "click 5",
    ]
    brain.script_expected_outcomes = [
        "search panel open",
        "calculator window open",
        "5 entered",
    ]
    brain._script_step_index = 1
    brain._step_transcript_counter = 3
    brain._update_step_metadata = MagicMock()
    return brain


def test_format_numbered_script_includes_expected_outcomes() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.script_lines = ["open search", "click calc"]
    brain.script_expected_outcomes = ["panel open", None]

    text = brain._format_numbered_script()

    assert "1. open search  | expected: panel open" in text
    assert "2. click calc  | expected: (none)" in text


def test_recording_expected_outcome_prompt_requires_enter_visual_result() -> None:
    text = get_prompt("recording_expected_outcome")

    assert "WindowChangeHint" in text
    assert "Do not return null for Enter/Esc/Tab/hotkeys" in text
    assert "{window_change_hint}" in text


def test_brain_verify_script_step_prompt_has_goto_policy() -> None:
    text = get_prompt("brain_verify_script_step")

    assert "| expected:" in text
    assert "do not retry" in text
    assert "Jump to the latest such prior line" in text
    assert "prior step's expected outcome is no longer true" in text
    assert "{actor_succeeded}" in text
    assert "clearly_unmet" in text
    assert "prefer accomplished true and branch advance" in text
    assert "Search/Start flyout" in text


def test_coerce_verify_result_advances_ambiguous_retry_after_actor_success() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()

    coerced = brain._coerce_verify_result_for_actor_success(
        ScriptStepVerifyResult(
            accomplished=False,
            branch="retry",
            target_step=None,
            clearly_unmet=False,
            reason="search bar visible but unsure about panel",
        ),
        actor_succeeded=True,
    )

    assert coerced.accomplished is True
    assert coerced.branch == "advance"
    assert coerced.clearly_unmet is False
    assert "not clearly unmet" in coerced.reason


def test_coerce_verify_result_keeps_clearly_unmet_retry_after_actor_success() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()

    original = ScriptStepVerifyResult(
        accomplished=False,
        branch="retry",
        target_step=None,
        clearly_unmet=True,
        reason="search panel still closed",
    )
    coerced = brain._coerce_verify_result_for_actor_success(
        original,
        actor_succeeded=True,
    )

    assert coerced is original
    assert coerced.branch == "retry"
    assert coerced.clearly_unmet is True


def test_coerce_verify_result_does_not_coerce_when_actor_failed() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()

    original = ScriptStepVerifyResult(
        accomplished=False,
        branch="retry",
        target_step=None,
        clearly_unmet=False,
        reason="tools failed",
    )
    coerced = brain._coerce_verify_result_for_actor_success(
        original,
        actor_succeeded=False,
    )

    assert coerced is original


@pytest.mark.asyncio
async def test_process_step_coerces_ambiguous_retry_when_actor_ok() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=True)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="retry",
            target_step=None,
            clearly_unmet=False,
            reason="ambiguous search UI",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    brain._verify_script_step.assert_awaited_once()
    assert brain._verify_script_step.await_args.kwargs["actor_succeeded"] is True
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["accomplished"] is True
    assert metadata["verify"]["clearly_unmet"] is False


@pytest.mark.asyncio
async def test_process_step_keeps_clearly_unmet_retry_when_actor_ok() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=True)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="retry",
            target_step=None,
            clearly_unmet=True,
            reason="calculator still closed",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "verify_failed"
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert metadata["verify"]["accomplished"] is False


def test_parse_verify_result_uses_last_json_object() -> None:
    raw = (
        '```json\n{"accomplished": false, "branch": "retry", "target_step": null, '
        '"reason": "retry first"}\n```\n\n'
        "Wait, let me re-evaluate based on the strict logic.\n"
        '```json\n{"accomplished": false, "branch": "goto", "target_step": 1, '
        '"reason": "search panel is not open"}\n```'
    )
    payload = BrainModule._parse_json_object_from_model_content(raw)
    result = ScriptStepVerifyResult.model_validate(payload)
    assert result.branch == "goto"
    assert result.target_step == 1
    assert result.accomplished is False


def test_brain_decide_action_2_does_not_invent_preparatory_methods() -> None:
    text = get_prompt("brain_decide_action_2")

    assert "try new method" not in text
    assert "Retry with a new method" not in text
    assert "Retry only against targets named in CurrentTaskGoal" in text
    assert "Do not add clicks, typing, or moves to anything the goal does not name." in text
    assert "If the named target is not on screen, return status failed JSON" in text


@pytest.mark.asyncio
async def test_process_step_verifies_after_actor_failure_and_applies_goto() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="goto",
            target_step=1,
            reason="search panel closed",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert result.run_complete is False
    assert brain._script_step_index == 0
    assert brain._step_transcript_counter == 4
    brain._verify_script_step.assert_awaited_once()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"]["branch"] == "goto"
    assert metadata["verify"]["target_step"] == 1


@pytest.mark.asyncio
async def test_process_step_aborts_when_actor_fails_and_verify_unavailable() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(return_value=None)

    result = await brain.process_step()

    assert result.step_finished is False
    assert brain._script_step_index == 1
    brain._verify_script_step.assert_awaited_once()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"] is None


@pytest.mark.asyncio
async def test_process_step_soft_fails_advance_when_actor_ok_and_verify_unavailable() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=True)
    brain._verify_script_step = AsyncMock(return_value=None)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    brain._verify_script_step.assert_awaited_once()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["accomplished"] is True
    assert "unparseable" in metadata["verify"]["reason"]


def test_parse_verify_result_repairs_wrong_closing_bracket() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    raw = (
        "```json\n"
        '{"accomplished": false, "branch": "retry", "target_step": 1, '
        '"clearly_unmet": false, "reason": "The search menu is not yet open."]\n'
        "```"
    )
    result = brain._parse_verify_result_from_content(raw)
    assert result is not None
    assert result.branch == "retry"
    assert result.target_step == 1
    assert result.clearly_unmet is False
    assert result.accomplished is False


def test_recover_verify_result_payload_scrapes_fields() -> None:
    raw = (
        'almost json {"accomplished": true, "branch": "advance", '
        '"clearly_unmet": false, "target_step": null, "reason": "panel open"'
    )
    payload = BrainModule._recover_verify_result_payload(raw)
    assert payload is not None
    assert payload["accomplished"] is True
    assert payload["branch"] == "advance"
    assert payload["clearly_unmet"] is False
    result = ScriptStepVerifyResult.model_validate(payload)
    assert result.branch == "advance"


@pytest.mark.asyncio
async def test_process_step_skips_vision_verify_when_expected_empty_and_actor_ok() -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.loop = AsyncMock(return_value=True)
    brain._verify_script_step = AsyncMock()

    result = await brain.process_step()

    assert result.step_finished is True
    assert result.run_complete is False
    assert brain._script_step_index == 2
    brain._verify_script_step.assert_not_awaited()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["expected_outcome"] is None
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["accomplished"] is True


@pytest.mark.asyncio
async def test_process_step_verifies_when_expected_empty_and_actor_failed() -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="retry",
            target_step=None,
            reason="search panel closed",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    brain._verify_script_step.assert_awaited_once()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"]["branch"] == "retry"


@pytest.mark.asyncio
async def test_process_step_verifies_when_expected_present_and_actor_ok() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=True)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=True,
            branch="advance",
            target_step=None,
            reason="calculator window open",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    brain._verify_script_step.assert_awaited_once()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["expected_outcome"] == "calculator window open"
