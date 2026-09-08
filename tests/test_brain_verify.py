from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain.module import BrainModule
from src.common.models import ScriptStepVerifyResult
from src.common.prompting import get_prompt


def _brain_for_process_step(*, max_step_attempts: int = 0) -> BrainModule:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain.manager.set_step_log_context = MagicMock()
    brain.manager.clear_step_log_context = MagicMock()
    brain.manager.require_paths = MagicMock(
        return_value=MagicMock(root=MagicMock(is_dir=lambda: False))
    )
    brain.settings = MagicMock(
        script_max_step_attempts=max_step_attempts,
        script_smart_recovery_max_cycles=3,
        script_smart_recovery_max_per_step=2,
    )
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
    brain.script_baseline_after_paths = [None, None, None]
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


def test_brain_verify_baseline_match_prompt_is_image_only() -> None:
    text = get_prompt("brain_verify_baseline_match")

    assert "Compare two screenshots only" in text
    assert '"match"' in text
    assert "NumberedScript" not in text
    assert "CurrentStepGoal" not in text
    assert "benign drift" in text
    assert "Cancel" in text or "關閉" in text


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
    assert "advance, retry, skip, goto, abort" in text
    assert "Use abort to stop the whole scripted run" in text
    assert "{baseline_attached}" in text
    assert "{baseline_precheck}" in text
    assert "BaselinePrecheck" in text
    assert "RecordedAfterBaselineAttached" in text
    assert "recorded after-frame" in text
    assert "advance, retry, skip, goto, abort, smart" in text
    assert "Use smart when live UI shows an unexpected blocker" in text
    assert "do not re-litigate" in text


def test_should_skip_vision_verify_requires_no_baseline() -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain._script_step_index = 0
    assert brain._should_skip_vision_verify(True) is True
    assert brain._should_skip_vision_verify(False) is False

    brain._current_baseline_after_path = MagicMock(return_value="C:/x.jpeg")
    assert brain._should_skip_vision_verify(True) is False


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


def test_coerce_verify_result_coerces_abort_after_actor_success() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()

    coerced = brain._coerce_verify_result_for_actor_success(
        ScriptStepVerifyResult(
            accomplished=False,
            branch="abort",
            target_step=None,
            clearly_unmet=False,
            reason="model gave up after success",
        ),
        actor_succeeded=True,
    )

    assert coerced.accomplished is True
    assert coerced.branch == "advance"
    assert "Original abort" in coerced.reason


def test_coerce_verify_result_coerces_smart_after_actor_success() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()

    coerced = brain._coerce_verify_result_for_actor_success(
        ScriptStepVerifyResult(
            accomplished=False,
            branch="smart",
            target_step=None,
            clearly_unmet=False,
            reason="ambiguous smart",
        ),
        actor_succeeded=True,
    )

    assert coerced.accomplished is True
    assert coerced.branch == "advance"
    assert "Original smart" in coerced.reason


def test_coerce_verify_result_keeps_clearly_unmet_abort_after_actor_success() -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()

    original = ScriptStepVerifyResult(
        accomplished=False,
        branch="abort",
        target_step=None,
        clearly_unmet=True,
        reason="outcome clearly unmet",
    )
    coerced = brain._coerce_verify_result_for_actor_success(
        original,
        actor_succeeded=True,
    )

    assert coerced is original
    assert coerced.branch == "abort"


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
async def test_process_step_stops_run_when_verify_aborts() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="abort",
            target_step=None,
            clearly_unmet=True,
            reason="click never succeeded; no recovery",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is False
    assert result.run_complete is False
    assert brain._script_step_index == 1
    assert "aborted" in (result.reason or "").lower()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"]["branch"] == "abort"


@pytest.mark.asyncio
async def test_process_step_smart_recovery_success_holds_step(monkeypatch) -> None:
    from src.runtime.script_smart_recovery import ScriptSmartRecoveryResult

    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="smart",
            target_step=None,
            clearly_unmet=True,
            reason="unexpected system popup",
        )
    )
    brain._script_step_smart_recovery_count = MagicMock(return_value=0)

    async def _fake_recovery(**kwargs):
        return ScriptSmartRecoveryResult(
            ok=True,
            reason="dismissed popup",
            events=[{"phase": "act", "summary": "clicked OK"}],
        )

    monkeypatch.setattr(
        "src.runtime.script_smart_recovery.run_script_smart_recovery",
        _fake_recovery,
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert result.run_complete is False
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "smart_recovered"
    assert metadata["verify"]["branch"] == "smart"
    assert metadata["smart_recovery"]["ok"] is True


@pytest.mark.asyncio
async def test_process_step_smart_recovery_failure_stops_run(monkeypatch) -> None:
    from src.runtime.script_smart_recovery import ScriptSmartRecoveryResult

    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="smart",
            target_step=None,
            clearly_unmet=True,
            reason="unexpected system popup",
        )
    )
    brain._script_step_smart_recovery_count = MagicMock(return_value=0)

    async def _fake_recovery(**kwargs):
        return ScriptSmartRecoveryResult(ok=False, reason="could not dismiss", events=[])

    monkeypatch.setattr(
        "src.runtime.script_smart_recovery.run_script_smart_recovery",
        _fake_recovery,
    )

    result = await brain.process_step()

    assert result.step_finished is False
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["smart_recovery"]["ok"] is False


@pytest.mark.asyncio
async def test_process_step_smart_recovery_cap_aborts_without_runner(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="smart",
            target_step=None,
            clearly_unmet=True,
            reason="popup again",
        )
    )
    brain._script_step_smart_recovery_count = MagicMock(return_value=2)
    called = {"n": 0}

    async def _fake_recovery(**kwargs):
        called["n"] += 1
        raise AssertionError("recovery should not run when cap reached")

    monkeypatch.setattr(
        "src.runtime.script_smart_recovery.run_script_smart_recovery",
        _fake_recovery,
    )

    result = await brain.process_step()

    assert result.step_finished is False
    assert called["n"] == 0
    assert "cap reached" in (result.reason or "").lower()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"]["branch"] == "smart"


def test_build_recovery_goal_mentions_script_step() -> None:
    from src.runtime.script_smart_recovery import build_recovery_goal

    text = build_recovery_goal(script_goal="click search", verify_reason="UAC dialog")
    assert "click search" in text
    assert "UAC dialog" in text
    assert "Do not advance the recorded script" in text


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


def test_recover_verify_result_payload_scrapes_abort() -> None:
    raw = (
        '{"accomplished": false, "branch": "abort", "target_step": null, '
        '"clearly_unmet": true, "reason": "unrecoverable"}'
    )
    payload = BrainModule._recover_verify_result_payload(raw)
    assert payload is not None
    assert payload["branch"] == "abort"
    result = ScriptStepVerifyResult.model_validate(payload)
    assert result.branch == "abort"


@pytest.mark.asyncio
async def test_process_step_skips_vision_verify_when_expected_empty_and_actor_ok() -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
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
async def test_process_step_verifies_when_baseline_present_even_if_outcome_empty() -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, "C:/fake/baseline.jpeg", None]
    brain._current_baseline_after_path = MagicMock(return_value="C:/fake/baseline.jpeg")
    brain.loop = AsyncMock(return_value=True)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=True,
            branch="advance",
            target_step=None,
            reason="live matches recorded after",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is True
    brain._verify_script_step.assert_awaited_once()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["baseline_after_path"] == "C:/fake/baseline.jpeg"


@pytest.mark.asyncio
async def test_verify_script_step_baseline_match_advances_without_recovery(tmp_path) -> None:
    live = tmp_path / "live.png"
    baseline = tmp_path / "after.jpeg"
    live.write_bytes(b"live")
    baseline.write_bytes(b"after")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain.settings = MagicMock(brain_lm="test-model")
    brain.script_lines = ["click search"]
    brain.script_expected_outcomes = [None]
    brain.script_baseline_after_paths = [str(baseline)]
    brain._script_step_index = 0
    brain._eye = MagicMock()
    brain._eye.capture_separated_images = AsyncMock(return_value=[str(live)])
    brain._append_step_messages = MagicMock()
    calls: list[dict[str, object]] = []

    async def _chat(model, *, messages, tools, response_format=None):
        calls.append({"messages": messages, "response_format": response_format})
        return MagicMock(
            content='{"match": true, "reason": "same main window and panels"}',
            model_dump=lambda: {"role": "assistant", "content": "ok"},
        )

    brain.ollama = MagicMock()
    brain.ollama.chat_messages = _chat

    result = await brain._verify_script_step(0, 0, actor_succeeded=True)

    assert result is not None
    assert result.accomplished is True
    assert result.branch == "advance"
    assert result.clearly_unmet is False
    assert len(calls) == 1
    user_msg = calls[0]["messages"][0]
    assert user_msg["images"] == [str(live), str(baseline)]
    assert "Compare two screenshots only" in user_msg["content"]
    assert "NumberedScript" not in user_msg["content"]
    brain._append_step_messages.assert_called_once()


@pytest.mark.asyncio
async def test_verify_script_step_baseline_mismatch_runs_recovery(tmp_path) -> None:
    live = tmp_path / "live.png"
    baseline = tmp_path / "after.jpeg"
    live.write_bytes(b"live")
    baseline.write_bytes(b"after")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain.settings = MagicMock(brain_lm="test-model")
    brain.script_lines = ["click search", "click calc"]
    brain.script_expected_outcomes = [None, None]
    brain.script_baseline_after_paths = [str(baseline), None]
    brain._script_step_index = 0
    brain._eye = MagicMock()
    brain._eye.capture_separated_images = AsyncMock(return_value=[str(live)])
    brain._append_step_messages = MagicMock()
    calls: list[dict[str, object]] = []

    async def _chat(model, *, messages, tools, response_format=None):
        calls.append({"messages": messages, "response_format": response_format, "tools": tools})
        if len(calls) == 1:
            return MagicMock(
                content='{"match": false, "reason": "dialog missing in live"}',
                model_dump=lambda: {"role": "assistant", "content": "match-round"},
            )
        return MagicMock(
            content=(
                '{"accomplished": false, "branch": "goto", "target_step": 1, '
                '"clearly_unmet": true, "reason": "need prior open dialog step"}'
            ),
            model_dump=lambda: {"role": "assistant", "content": "recovery-round"},
        )

    brain.ollama = MagicMock()
    brain.ollama.chat_messages = _chat

    result = await brain._verify_script_step(0, 0, actor_succeeded=True)

    assert result is not None
    assert result.branch == "goto"
    assert result.target_step == 1
    assert result.clearly_unmet is True
    assert len(calls) == 2
    assert "Compare two screenshots only" in calls[0]["messages"][0]["content"]
    recovery_msg = calls[1]["messages"][0]
    assert recovery_msg["images"] == [str(live), str(baseline)]
    assert "NumberedScript" in recovery_msg["content"]
    assert "BaselinePrecheck" in recovery_msg["content"]
    assert "mismatch: dialog missing in live" in recovery_msg["content"]
    assert "do not re-litigate" in recovery_msg["content"]


@pytest.mark.asyncio
async def test_verify_script_step_without_baseline_skips_match_round(tmp_path) -> None:
    live = tmp_path / "live.png"
    live.write_bytes(b"live")

    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain.settings = MagicMock(brain_lm="test-model")
    brain.script_lines = ["click search"]
    brain.script_expected_outcomes = ["search panel open"]
    brain.script_baseline_after_paths = [None]
    brain._script_step_index = 0
    brain._eye = MagicMock()
    brain._eye.capture_separated_images = AsyncMock(return_value=[str(live)])
    brain._append_step_messages = MagicMock()
    calls: list[dict[str, object]] = []

    async def _chat(model, *, messages, tools, response_format=None):
        calls.append({"messages": messages})
        return MagicMock(
            content=(
                '{"accomplished": true, "branch": "advance", "target_step": null, '
                '"clearly_unmet": false, "reason": "panel open"}'
            ),
            model_dump=lambda: {"role": "assistant", "content": "ok"},
        )

    brain.ollama = MagicMock()
    brain.ollama.chat_messages = _chat

    result = await brain._verify_script_step(0, 0, actor_succeeded=True)

    assert result is not None
    assert result.branch == "advance"
    assert len(calls) == 1
    user_msg = calls[0]["messages"][0]
    assert user_msg["images"] == [str(live)]
    assert "BaselinePrecheck" in user_msg["content"]
    assert "(none)" in user_msg["content"]
    assert "Compare two screenshots only" not in user_msg["content"]
    assert "NumberedScript" in user_msg["content"]


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
async def test_process_step_stops_when_script_retry_limit_reached(tmp_path) -> None:
    brain = _brain_for_process_step(max_step_attempts=3)
    steps_dir = tmp_path / "steps"
    steps_dir.mkdir()
    (steps_dir / "0_1.json").write_text("{}", encoding="utf-8")
    (steps_dir / "1_1.json").write_text("{}", encoding="utf-8")
    brain.manager.require_paths.return_value.root = tmp_path
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="retry",
            target_step=None,
            clearly_unmet=True,
            reason="target not found",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is False
    assert brain._script_step_index == 1
    assert "3/3 attempt(s)" in result.reason
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["attempt_number"] == 3
    assert metadata["max_attempts"] == 3


@pytest.mark.asyncio
async def test_process_step_allows_retry_below_script_limit(tmp_path) -> None:
    brain = _brain_for_process_step(max_step_attempts=3)
    steps_dir = tmp_path / "steps"
    steps_dir.mkdir()
    (steps_dir / "0_1.json").write_text("{}", encoding="utf-8")
    brain.manager.require_paths.return_value.root = tmp_path
    brain.loop = AsyncMock(return_value=False)
    brain._verify_script_step = AsyncMock(
        return_value=ScriptStepVerifyResult(
            accomplished=False,
            branch="retry",
            target_step=None,
            clearly_unmet=True,
            reason="target not found",
        )
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
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
