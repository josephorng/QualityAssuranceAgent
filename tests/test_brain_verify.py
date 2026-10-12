from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain.module import (
    BrainModule,
    _APPEARED_POLL_MIN_SECONDS,
    _WINDOW_VERIFY_POLL_INTERVAL_SECONDS,
)
from src.common.models import ScriptStepVerifyResult
from src.common.prompting import get_prompt
from src.common.runtime_context import SKIP_VERIFICATION_ENV
from src.recorder.window_snapshot import WindowInfo


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
    brain.script_settle_after_seconds = [None, None, None]
    brain.script_window_verifies = [{}, {}, {}]
    brain._script_step_index = 1
    brain._script_event_indices = []
    brain._step_transcript_counter = 3
    brain._pending_settle_deadline_perf = None
    brain._update_step_metadata = MagicMock()
    return brain


def test_recording_expected_outcome_prompt_requires_enter_visual_result() -> None:
    text = get_prompt("recording_expected_outcome")

    assert "WindowChangeHint" in text
    assert "Do not return null for Enter/Esc/Tab/hotkeys" in text
    assert "{window_change_hint}" in text


def test_brain_decide_action_2_does_not_invent_preparatory_methods() -> None:
    text = get_prompt("brain_decide_action_2")

    assert "try new method" not in text
    assert "Retry with a new method" not in text
    assert "Retry only against targets named in CurrentTaskGoal" in text
    assert "Do not add clicks, typing, or moves to anything the goal does not name." in text
    assert "If the named target is not on screen, return status failed JSON" in text


def test_build_recovery_goal_mentions_script_step() -> None:
    from src.runtime.script_smart_recovery import build_recovery_goal

    text = build_recovery_goal(script_goal="click search", verify_reason="UAC dialog")
    assert "click search" in text
    assert "UAC dialog" in text
    assert "Do not advance the recorded script" in text


@pytest.mark.asyncio
async def test_process_step_saves_record_when_actor_raises() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(side_effect=ConnectionError("All connection attempts failed"))

    result = await brain.process_step()

    assert result.step_finished is False
    assert result.step_index == 1
    assert "All connection attempts failed" in result.reason
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["goal"] == "click calculator"
    assert metadata["started_at_utc"]
    assert metadata["finished_at_utc"]
    brain.manager.clear_step_log_context.assert_called_once()


@pytest.mark.asyncio
async def test_process_step_saves_record_when_cancelled() -> None:
    import asyncio

    brain = _brain_for_process_step()
    brain.loop = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await brain.process_step()

    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["goal"] == "click calculator"
    assert metadata["started_at_utc"]
    assert metadata["finished_at_utc"]


@pytest.mark.asyncio
async def test_process_step_skip_verification_advances_without_checks(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_baseline_after_paths = [None, "after.jpeg", None]
    brain.script_settle_after_seconds = [None, 2.5, None]
    brain.script_window_verifies = [
        {},
        {
            "disappeared": [
                {
                    "class_name": "Popup",
                    "title": "快顯",
                    "process_name": "explorer.exe",
                }
            ]
        },
        {},
    ]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setenv(SKIP_VERIFICATION_ENV, "1")

    def _must_not_snapshot() -> list[WindowInfo]:
        raise AssertionError("window snapshot should not run")

    monkeypatch.setattr("src.brain.module.snapshot_top_level_windows", _must_not_snapshot)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    sleep_mock.assert_not_awaited()
    assert brain._pending_settle_deadline_perf is not None
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["reason"] == "Verification skipped for this queue run."


@pytest.mark.asyncio
async def test_process_step_skip_verification_fails_actor_without_verifier(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)
    monkeypatch.setenv(SKIP_VERIFICATION_ENV, "1")

    result = await brain.process_step()

    assert result.step_finished is False
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"] is None


@pytest.mark.asyncio
async def test_process_step_auto_advances_when_actor_ok_no_window_verify() -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.loop = AsyncMock(return_value=True)

    result = await brain.process_step()

    assert result.step_finished is True
    assert result.run_complete is False
    assert brain._script_step_index == 2
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["expected_outcome"] is None
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["accomplished"] is True
    assert "no window verify" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_auto_advances_when_baseline_present() -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, "C:/fake/baseline.jpeg", None]
    brain._current_baseline_after_path = MagicMock(return_value="C:/fake/baseline.jpeg")
    brain.loop = AsyncMock(return_value=True)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["baseline_after_path"] == "C:/fake/baseline.jpeg"
    assert "no window verify" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_auto_advances_when_expected_outcome_present() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=True)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["expected_outcome"] == "calculator window open"
    assert metadata["verify"]["branch"] == "advance"
    assert "no window verify" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_actor_failure_fails_without_vision_recovery() -> None:
    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=False)

    result = await brain.process_step()

    assert result.step_finished is False
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"] is None


def _flyout_window() -> WindowInfo:
    return WindowInfo(
        hwnd=9,
        title="快顯主機",
        pid=1,
        left=446,
        top=139,
        width=213,
        height=217,
        is_minimized=False,
        is_maximized=False,
        class_name="Microsoft.UI.Content.PopupWindowSiteBridge",
        process_name="explorer.exe",
    )


def _word_window() -> WindowInfo:
    return WindowInfo(
        hwnd=42,
        title="Word",
        pid=2,
        left=0,
        top=0,
        width=800,
        height=600,
        is_minimized=False,
        is_maximized=True,
        class_name="OpusApp",
        process_name="WINWORD.EXE",
    )


def _appeared_word_verify() -> dict:
    return {
        "appeared": [
            {
                "class_name": "OpusApp",
                "title": "Word",
                "process_name": "WINWORD.EXE",
            }
        ]
    }


@pytest.mark.asyncio
async def test_process_step_empty_window_verify_still_auto_advances(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_window_verifies = [{}, {}, {}]
    brain.loop = AsyncMock(return_value=True)
    snapshot = MagicMock(return_value=[_flyout_window()])
    monkeypatch.setattr("src.brain.module.snapshot_top_level_windows", snapshot)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    snapshot.assert_not_called()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["clearly_unmet"] is False


@pytest.mark.asyncio
async def test_process_step_window_verify_miss_retries_clearly_unmet(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    flyout_gone = {
        "disappeared": [
            {
                "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
                "title": "快顯主機",
                "process_name": "explorer.exe",
            }
        ]
    }
    brain.script_window_verifies = [{}, flyout_gone, {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )

    def _signals_must_not_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("signal snapshot is not used for a window-list predicate")

    monkeypatch.setattr(
        "src.brain.module.capture_replay_after_signals",
        _signals_must_not_run,
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "verify_failed"
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert metadata["verify"]["accomplished"] is False


@pytest.mark.asyncio
async def test_process_step_foreground_miss_retries_clearly_unmet(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    chrome = {
        "class_name": "Chrome_WidgetWin_1",
        "title": "Google Chrome",
        "process_name": "chrome.exe",
    }
    brain.script_window_verifies = [{}, {"foreground": chrome}, {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    monkeypatch.setattr(
        "src.brain.module.capture_replay_after_signals",
        lambda _recorded, _press_point=None: {
            "foreground": {
                "class_name": "CabinetWClass",
                "title": "檔案總管",
                "process_name": "explorer.exe",
            }
        },
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert "foreground" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_clipboard_miss_retries_clearly_unmet(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_window_verifies = [{}, {"clipboard": "copied text"}, {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    monkeypatch.setattr(
        "src.brain.module.capture_replay_after_signals",
        lambda _recorded, _press_point=None: {"clipboard": "something else"},
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert "clipboard" in metadata["verify"]["reason"]


def _appeared_poll_deadline_seconds() -> float:
    """Appeared floor (5s) uses the 2.0x continuous-poll deadline (10s)."""
    return float(_APPEARED_POLL_MIN_SECONDS) * 2.0


def _expected_poll_sleeps(deadline: float) -> list[float]:
    """Sleep schedule for continuous polling up to ``deadline``."""
    interval = float(_WINDOW_VERIFY_POLL_INTERVAL_SECONDS)
    sleeps: list[float] = []
    elapsed = 0.0
    while elapsed < deadline:
        delay = min(interval, deadline - elapsed)
        if delay <= 0:
            break
        sleeps.append(delay)
        elapsed += delay
    return sleeps


@pytest.mark.asyncio
async def test_process_step_window_verify_polls_until_appeared(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 2.0, None]
    brain.script_window_verifies = [{}, _appeared_word_verify(), {}]
    brain.loop = AsyncMock(return_value=True)

    snapshots = [
        [],  # before
        [],  # first poll miss
        [_word_window()],  # second poll pass (early exit)
    ]

    def _next_snapshot() -> list[WindowInfo]:
        assert snapshots
        return snapshots.pop(0)

    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        _next_snapshot,
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    # Early exit after one 0.1s interval — not the full 5s/10s budget.
    assert sleep_mock.await_count == 1
    assert sleep_mock.await_args_list[0].args[0] == pytest.approx(
        _WINDOW_VERIFY_POLL_INTERVAL_SECONDS
    )
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"


@pytest.mark.asyncio
async def test_process_step_window_verify_fails_only_at_2x_settle(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 2.0, None]
    brain.script_window_verifies = [{}, _appeared_word_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    expected_sleeps = _expected_poll_sleeps(_appeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert [call.args[0] for call in sleep_mock.await_args_list] == pytest.approx(
        expected_sleeps
    )
    assert sum(expected_sleeps) == pytest.approx(_appeared_poll_deadline_seconds())
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "verify_failed"
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert "appeared" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_appeared_without_settle_uses_poll_floor(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, None, None]
    brain.script_window_verifies = [{}, _appeared_word_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    # Floor is 5.0s → continuous polls until 10s deadline.
    expected_sleeps = _expected_poll_sleeps(_appeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert [call.args[0] for call in sleep_mock.await_args_list] == pytest.approx(
        expected_sleeps
    )
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"
    assert "appeared" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_appeared_tiny_settle_uses_poll_floor(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 0.109, None]
    brain.script_window_verifies = [{}, _appeared_word_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    # Tiny positive settle is floored to 5.0s → deadline 10s.
    expected_sleeps = _expected_poll_sleeps(_appeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert [call.args[0] for call in sleep_mock.await_args_list] == pytest.approx(
        expected_sleeps
    )
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"
    assert "appeared" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_defers_settle_when_no_window_verify(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, "C:/fake/baseline.jpeg", None]
    brain.script_settle_after_seconds = [None, 2.5, None]
    brain.loop = AsyncMock(return_value=True)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    sleep_mock.assert_not_awaited()
    assert brain._pending_settle_deadline_perf is not None
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["settle_after_seconds"] == 2.5


@pytest.mark.asyncio
async def test_process_step_defers_settle_with_expected_outcome_no_baseline(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = ["panel open", "calc open", "5 entered"]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 3.0, None]
    brain.loop = AsyncMock(return_value=True)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    sleep_mock.assert_not_awaited()
    assert brain._pending_settle_deadline_perf is not None
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["settle_after_seconds"] == 3.0


@pytest.mark.asyncio
async def test_await_pending_settle_before_tool_waits_remainder(monkeypatch) -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain._pending_settle_deadline_perf = 100.0
    brain._step_deferred_settle_waited_seconds = 0.0
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    monkeypatch.setattr("src.brain.module.perf_counter", lambda: 97.5)
    messages: list[dict] = []

    waited = await brain._await_pending_settle_before_tool(messages)

    assert waited == 2.5
    sleep_mock.assert_awaited_once_with(2.5)
    assert brain._pending_settle_deadline_perf is None
    assert brain._step_deferred_settle_waited_seconds == 2.5
    assert len(messages) == 2
    assert messages[0]["deferred_settle_wait_seconds"] == 2.5
    assert messages[0]["role"] == "system"
    assert messages[1]["deferred_settle_done"] is True
    assert "waited 2.500s" in messages[1]["content"]


@pytest.mark.asyncio
async def test_await_pending_settle_before_tool_skips_when_already_satisfied(
    monkeypatch,
) -> None:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain._pending_settle_deadline_perf = 100.0
    brain._step_deferred_settle_waited_seconds = 0.0
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    monkeypatch.setattr("src.brain.module.perf_counter", lambda: 101.25)
    messages: list[dict] = [{"role": "user", "content": "x"}]

    waited = await brain._await_pending_settle_before_tool(messages)

    assert waited == 0.0
    sleep_mock.assert_not_awaited()
    assert brain._pending_settle_deadline_perf is None
    assert messages == [{"role": "user", "content": "x"}]
    assert brain._step_deferred_settle_waited_seconds == 0.0


def test_messages_for_llm_drops_deferred_settle_markers() -> None:
    messages = [
        {"role": "user", "content": "goal"},
        {"role": "system", "deferred_settle_wait_seconds": 1.5, "content": "waiting"},
        {"role": "system", "deferred_settle_done": True, "content": "done"},
        {"role": "assistant", "content": "ok"},
    ]
    assert BrainModule._messages_for_llm(messages) == [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_process_step_skips_settle_when_settle_none(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, None, None]
    brain.loop = AsyncMock(return_value=True)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    sleep_mock.assert_not_awaited()
    assert brain._pending_settle_deadline_perf is None


@pytest.mark.asyncio
async def test_process_step_stops_when_script_retry_limit_reached(
    tmp_path, monkeypatch
) -> None:
    brain = _brain_for_process_step(max_step_attempts=3)
    steps_dir = tmp_path / "steps"
    steps_dir.mkdir()
    (steps_dir / "0_1.json").write_text("{}", encoding="utf-8")
    (steps_dir / "1_1.json").write_text("{}", encoding="utf-8")
    brain.manager.require_paths.return_value.root = tmp_path
    flyout_gone = {
        "disappeared": [
            {
                "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
                "title": "快顯主機",
                "process_name": "explorer.exe",
            }
        ]
    }
    brain.script_window_verifies = [{}, flyout_gone, {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )

    result = await brain.process_step()

    assert result.step_finished is False
    assert brain._script_step_index == 1
    assert "3/3 attempt(s)" in result.reason
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["attempt_number"] == 3
    assert metadata["max_attempts"] == 3
    assert metadata["verify"]["branch"] == "retry"


@pytest.mark.asyncio
async def test_process_step_allows_retry_below_script_limit(
    tmp_path, monkeypatch
) -> None:
    brain = _brain_for_process_step(max_step_attempts=3)
    steps_dir = tmp_path / "steps"
    steps_dir.mkdir()
    (steps_dir / "0_1.json").write_text("{}", encoding="utf-8")
    brain.manager.require_paths.return_value.root = tmp_path
    flyout_gone = {
        "disappeared": [
            {
                "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
                "title": "快顯主機",
                "process_name": "explorer.exe",
            }
        ]
    }
    brain.script_window_verifies = [{}, flyout_gone, {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"


def test_parse_json_object_uses_last_json_object() -> None:
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
