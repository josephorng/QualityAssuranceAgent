from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain.module import (
    BrainModule,
    _APPEARED_POLL_MIN_SECONDS,
    _DISAPPEARED_CLOSE_RETRY_INTERVAL_SECONDS,
    _DISAPPEARED_CLOSE_RETRY_MIN_SECONDS,
    _DISAPPEARED_POLL_MIN_SECONDS,
    _SIGNAL_POLL_MIN_SECONDS,
    _WINDOW_VERIFY_POLL_INTERVAL_SECONDS,
    _close_retry_due,
)
from src.common.models import ScriptStepVerifyResult
from src.common.prompting import get_prompt
from src.common.run_state import step_artifact_name
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
    brain._pending_after_capture = False
    brain._hand = MagicMock()
    brain._hand.capture_after_for_last_action = AsyncMock(return_value=None)
    brain._hand.assign_after_for_last_action = AsyncMock(return_value=None)
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
async def test_process_step_reloads_after_started_at(monkeypatch) -> None:
    """Dirty recording reload is inside the step timer, after notify/start stamp."""
    from src.common import run_control

    brain = _brain_for_process_step()
    brain.loop = AsyncMock(return_value=True)
    order: list[str] = []

    def track_notify(*_args, **_kwargs) -> None:
        order.append("notify")

    def track_reload() -> None:
        order.append("reload")

    def track_publish(*_args, **_kwargs) -> None:
        order.append("publish")

    monkeypatch.setattr(run_control, "notify_active_step", track_notify)
    monkeypatch.setattr(run_control, "publish_script_steps", track_publish)
    # process_step imports these names on the brain module.
    monkeypatch.setattr("src.brain.module.notify_active_step", track_notify)
    monkeypatch.setattr("src.brain.module.publish_script_steps", track_publish)
    brain.reload_pending_recording_steps = track_reload

    result = await brain.process_step()

    assert result.step_finished is True
    assert order == ["notify", "reload", "publish"]
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["started_at_utc"]


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
    sleep_mock.assert_awaited_once_with(2.5)
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


@pytest.mark.asyncio
async def test_process_step_move_mouse_failure_holds_for_overlay_retry() -> None:
    brain = _brain_for_process_step()

    async def _failed_move() -> bool:
        brain._hold_step_retry_reason = "Recorded click_window is not on screen."
        return False

    brain.loop = _failed_move

    result = await brain.process_step()

    assert result.step_finished is True
    assert result.run_complete is False
    assert brain._script_step_index == 1
    assert "click_window" in result.reason
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "failed"
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert brain._hold_step_retry_reason is None


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


def _disappeared_flyout_verify() -> dict:
    return {
        "disappeared": [
            {
                "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
                "title": "快顯主機",
                "process_name": "explorer.exe",
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
    assert "no window verify required" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_window_verify_miss_retries_clearly_unmet(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_window_verifies = [{}, _disappeared_flyout_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    def _signals_must_not_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("signal snapshot is not used for a window-list predicate")

    monkeypatch.setattr(
        "src.brain.module.capture_replay_after_signals",
        _signals_must_not_run,
    )

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    expected_sleeps = _expected_poll_sleeps(_disappeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
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
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    expected_sleeps = _expected_poll_sleeps(_signal_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
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
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    expected_sleeps = _expected_poll_sleeps(_signal_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert "clipboard" in metadata["verify"]["reason"]


def _appeared_poll_deadline_seconds() -> float:
    """Appeared floor (5s) uses the 2.0x continuous-poll deadline (10s)."""
    return float(_APPEARED_POLL_MIN_SECONDS) * 2.0


def _disappeared_poll_deadline_seconds() -> float:
    """Disappeared floor (1s) is under the 2s multi threshold → 1.0x deadline."""
    return float(_DISAPPEARED_POLL_MIN_SECONDS)


def _signal_poll_deadline_seconds() -> float:
    """Signal floor (1s) is under the 2s multi threshold → 1.0x deadline."""
    return float(_SIGNAL_POLL_MIN_SECONDS)


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
async def test_process_step_disappeared_without_settle_uses_poll_floor(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, None, None]
    brain.script_window_verifies = [{}, _disappeared_flyout_verify(), {}]
    brain.loop = AsyncMock(return_value=True)

    snapshots = [
        [_flyout_window()],  # before
        [_flyout_window()],  # first poll miss (still present)
        [],  # second poll pass (gone)
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
    # Early exit after one 0.1s interval — not the full 1s budget.
    assert sleep_mock.await_count == 1
    assert sleep_mock.await_args_list[0].args[0] == pytest.approx(
        _WINDOW_VERIFY_POLL_INTERVAL_SECONDS
    )
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"


@pytest.mark.asyncio
async def test_process_step_disappeared_without_settle_fails_at_poll_deadline(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, None, None]
    brain.script_window_verifies = [{}, _disappeared_flyout_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    # Floor is 1.0s → continuous polls until 1.0s deadline (1.0x under multi min).
    expected_sleeps = _expected_poll_sleeps(_disappeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert [call.args[0] for call in sleep_mock.await_args_list] == pytest.approx(
        expected_sleeps
    )
    assert sum(expected_sleeps) == pytest.approx(_disappeared_poll_deadline_seconds())
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "verify_failed"
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert "disappeared" in metadata["verify"]["reason"]


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


def _search_foreground() -> dict[str, str]:
    return {
        "class_name": "Windows.UI.Core.CoreWindow",
        "title": "搜尋",
        "process_name": "SearchHost.exe",
    }


@pytest.mark.asyncio
async def test_process_step_foreground_tiny_settle_uses_poll_floor(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 0.016, None]
    brain.script_window_verifies = [{}, {"foreground": _search_foreground()}, {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    monkeypatch.setattr(
        "src.brain.module.capture_replay_after_signals",
        lambda _recorded, _press_point=None: {
            "foreground": {
                "class_name": "Progman",
                "title": "Program Manager",
                "process_name": "explorer.exe",
            }
        },
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    # Tiny positive settle is floored to 1.0s → deadline 1.0s.
    expected_sleeps = _expected_poll_sleeps(_signal_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert [call.args[0] for call in sleep_mock.await_args_list] == pytest.approx(
        expected_sleeps
    )
    assert sum(expected_sleeps) == pytest.approx(_signal_poll_deadline_seconds())
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"
    assert "foreground" in metadata["verify"]["reason"]


@pytest.mark.asyncio
async def test_process_step_window_verify_polls_until_foreground(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 0.016, None]
    search = _search_foreground()
    brain.script_window_verifies = [{}, {"foreground": search}, {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    signal_samples = [
        {
            "foreground": {
                "class_name": "Progman",
                "title": "Program Manager",
                "process_name": "explorer.exe",
            }
        },
        {"foreground": search},
    ]

    def _next_signals(_recorded: object, _press_point: object = None) -> dict:
        assert signal_samples
        return signal_samples.pop(0)

    monkeypatch.setattr(
        "src.brain.module.capture_replay_after_signals",
        _next_signals,
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    assert sleep_mock.await_count == 1
    assert sleep_mock.await_args_list[0].args[0] == pytest.approx(
        _WINDOW_VERIFY_POLL_INTERVAL_SECONDS
    )
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"


@pytest.mark.asyncio
async def test_process_step_sleeps_settle_when_no_window_verify(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, "C:/fake/baseline.jpeg", None]
    brain.script_settle_after_seconds = [None, 2.5, None]
    brain.loop = AsyncMock(return_value=True)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    sleep_mock.assert_awaited_once_with(2.5)
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["settle_after_seconds"] == 2.5


@pytest.mark.asyncio
async def test_process_step_sleeps_settle_with_expected_outcome_no_baseline(
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
    sleep_mock.assert_awaited_once_with(3.0)
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["settle_after_seconds"] == 3.0


@pytest.mark.asyncio
async def test_seal_pending_after_with_before_assigns_shared_path() -> None:
    brain = _brain_for_process_step()
    brain._pending_after_capture = True

    await brain._seal_pending_after_with_before("C:/runs/eye/next_before.png")

    brain._hand.assign_after_for_last_action.assert_awaited_once_with(
        "C:/runs/eye/next_before.png"
    )
    assert brain._pending_after_capture is False
    brain._hand.capture_after_for_last_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_step_defers_after_to_next_before(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 2.5, None]

    async def _loop_marks_pending() -> bool:
        brain._mark_pending_after_capture()
        return True

    brain.loop = AsyncMock(side_effect=_loop_marks_pending)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    sleep_mock.assert_awaited_once_with(2.5)
    assert brain._pending_after_capture is True
    brain._hand.capture_after_for_last_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_step_defers_after_even_without_settle(monkeypatch) -> None:
    """Advancing steps leave after pending for the next before; no extra capture."""
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, None, None]

    async def _loop_marks_pending() -> bool:
        brain._mark_pending_after_capture()
        return True

    brain.loop = AsyncMock(side_effect=_loop_marks_pending)

    result = await brain.process_step()

    assert result.step_finished is True
    assert result.run_complete is False
    brain._hand.capture_after_for_last_action.assert_not_awaited()
    assert brain._pending_after_capture is True


@pytest.mark.asyncio
async def test_process_step_final_step_captures_dedicated_after(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain._script_step_index = 2  # last script line
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, None, None]

    async def _loop_marks_pending() -> bool:
        brain._mark_pending_after_capture()
        return True

    brain.loop = AsyncMock(side_effect=_loop_marks_pending)

    result = await brain.process_step()

    assert result.step_finished is True
    assert result.run_complete is True
    brain._hand.capture_after_for_last_action.assert_awaited_once()
    assert brain._pending_after_capture is False


@pytest.mark.asyncio
async def test_process_step_window_verify_success_defers_after_to_next_before(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_expected_outcomes = [None, None, None]
    brain.script_baseline_after_paths = [None, None, None]
    brain.script_settle_after_seconds = [None, 2.0, None]
    brain.script_window_verifies = [{}, _appeared_word_verify(), {}]

    async def _loop_marks_pending() -> bool:
        brain._mark_pending_after_capture()
        return True

    brain.loop = AsyncMock(side_effect=_loop_marks_pending)
    snapshots = [
        [],  # before
        [_word_window()],  # first poll pass
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
    brain._hand.capture_after_for_last_action.assert_not_awaited()
    assert brain._pending_after_capture is True
    # Window verify owns settle; do not also sleep settle_after on this step.
    # Appeared floor is 5s with 2.0x deadline → poll sleeps until match (first try).
    sleep_mock.assert_not_awaited()
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "advance"
    assert metadata["verify"]["accomplished"] is True
    assert "window verify passed" in metadata["verify"]["reason"]


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


@pytest.mark.asyncio
async def test_process_step_stops_when_script_retry_limit_reached(
    tmp_path, monkeypatch
) -> None:
    brain = _brain_for_process_step(max_step_attempts=3)
    steps_dir = tmp_path / "steps"
    steps_dir.mkdir()
    (steps_dir / step_artifact_name(0, 1)).write_text("{}", encoding="utf-8")
    (steps_dir / step_artifact_name(1, 1)).write_text("{}", encoding="utf-8")
    brain.manager.require_paths.return_value.root = tmp_path
    brain.script_window_verifies = [{}, _disappeared_flyout_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is False
    assert brain._script_step_index == 1
    assert "3/3 attempt(s)" in result.reason
    expected_sleeps = _expected_poll_sleeps(_disappeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
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
    (steps_dir / step_artifact_name(0, 1)).write_text("{}", encoding="utf-8")
    brain.manager.require_paths.return_value.root = tmp_path
    brain.script_window_verifies = [{}, _disappeared_flyout_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_flyout_window()],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 1
    expected_sleeps = _expected_poll_sleeps(_disappeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["verify"]["branch"] == "retry"


def _word_disappeared_verify() -> dict:
    return {
        "disappeared": [
            {
                "class_name": "OpusApp",
                "title": "Word",
                "process_name": "WINWORD.EXE",
            }
        ]
    }


def _word_splash_window() -> WindowInfo:
    return WindowInfo(
        hwnd=7,
        title="正在開啟 - Word",
        pid=2,
        left=700,
        top=300,
        width=200,
        height=160,
        is_minimized=False,
        is_maximized=False,
        class_name="OpusApp",
        process_name="WINWORD.EXE",
    )


def _loop_marks_close_windows(brain: BrainModule) -> None:
    async def _loop() -> bool:
        brain._step_close_windows_ok = True
        return True

    brain.loop = _loop  # type: ignore[method-assign]


def _expected_close_attempts(deadline: float) -> int:
    """Close attempts while a disappeared window stays up until ``deadline``."""
    interval = float(_WINDOW_VERIFY_POLL_INTERVAL_SECONDS)
    close_interval = float(_DISAPPEARED_CLOSE_RETRY_INTERVAL_SECONDS)
    elapsed = 0.0
    last: float | None = None
    count = 0
    while elapsed < deadline:
        if _close_retry_due(elapsed, last, close_interval):
            count += 1
            last = elapsed
        delay = min(interval, deadline - elapsed)
        if delay <= 0:
            break
        elapsed += delay
    return count


def _record_close_hwnds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    closed: list[int] = []

    def _close(hwnd: int) -> dict[str, object]:
        closed.append(int(hwnd))
        return {"hwnd": int(hwnd), "matched_title": "Word", "status": "closed"}

    monkeypatch.setattr("src.brain.module.close_hwnd", _close)
    return closed


@pytest.mark.asyncio
async def test_disappeared_close_retry_closes_window_and_advances(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_window_verifies = [{}, _word_disappeared_verify(), {}]
    _loop_marks_close_windows(brain)
    snapshots = [
        [_word_window()],
        [_word_window()],
        [],
    ]

    def _next_snapshot() -> list[WindowInfo]:
        assert snapshots
        return snapshots.pop(0)

    monkeypatch.setattr("src.brain.module.snapshot_top_level_windows", _next_snapshot)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)
    transcript: list[dict] = []

    def _capture(messages: list[dict], *_args: object, **_kwargs: object) -> None:
        transcript.extend(messages)

    brain._append_step_messages = _capture

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    assert closed == [42]
    assert sleep_mock.await_count == 1
    assert sleep_mock.await_args_list[0].args[0] == pytest.approx(
        _WINDOW_VERIFY_POLL_INTERVAL_SECONDS
    )
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"
    assert metadata["verify"]["branch"] == "advance"
    assert transcript[0]["disappeared_close_retry"] is True
    assert transcript[0]["tool_calls"][0]["function"]["arguments"]["hwnd"] == 42


@pytest.mark.asyncio
async def test_disappeared_close_retry_stops_at_extended_deadline(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_window_verifies = [{}, _word_disappeared_verify(), {}]
    _loop_marks_close_windows(brain)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_word_window()],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)

    result = await brain.process_step()

    deadline = float(_DISAPPEARED_CLOSE_RETRY_MIN_SECONDS)
    assert result.step_finished is True
    assert brain._script_step_index == 1
    expected_sleeps = _expected_poll_sleeps(deadline)
    assert sleep_mock.await_count == len(expected_sleeps)
    assert sum(expected_sleeps) == pytest.approx(deadline)
    assert len(closed) == _expected_close_attempts(deadline)
    assert set(closed) == {42}
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "verify_failed"
    assert metadata["verify"]["branch"] == "retry"
    assert metadata["verify"]["clearly_unmet"] is True
    assert metadata["verify"]["reason"] == "window verify missed disappeared: Word"


@pytest.mark.asyncio
async def test_disappeared_close_retry_keeps_longer_settle_deadline(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_settle_after_seconds = [None, 6.0, None]
    brain.script_window_verifies = [{}, _word_disappeared_verify(), {}]
    _loop_marks_close_windows(brain)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_word_window()],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)

    await brain.process_step()

    deadline = 12.0
    expected_sleeps = _expected_poll_sleeps(deadline)
    assert sleep_mock.await_count == len(expected_sleeps)
    assert sum(expected_sleeps) == pytest.approx(deadline)
    assert len(closed) == _expected_close_attempts(deadline)


@pytest.mark.asyncio
async def test_disappeared_close_retry_skips_splash_title(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_window_verifies = [{}, _word_disappeared_verify(), {}]
    _loop_marks_close_windows(brain)
    snapshots = [
        [_word_splash_window(), _word_window()],
        [_word_splash_window(), _word_window()],
        [_word_splash_window()],
    ]

    def _next_snapshot() -> list[WindowInfo]:
        assert snapshots
        return snapshots.pop(0)

    monkeypatch.setattr("src.brain.module.snapshot_top_level_windows", _next_snapshot)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    assert closed == [42]
    assert sleep_mock.await_count == 1


@pytest.mark.asyncio
async def test_disappeared_close_retry_skips_when_window_already_gone(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_window_verifies = [{}, _word_disappeared_verify(), {}]
    _loop_marks_close_windows(brain)
    snapshots = [[_word_window()], []]

    def _next_snapshot() -> list[WindowInfo]:
        assert snapshots
        return snapshots.pop(0)

    monkeypatch.setattr("src.brain.module.snapshot_top_level_windows", _next_snapshot)
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    assert closed == []
    assert sleep_mock.await_count == 0


@pytest.mark.asyncio
async def test_disappeared_close_retry_ignores_close_error_and_continues(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_window_verifies = [{}, _word_disappeared_verify(), {}]
    _loop_marks_close_windows(brain)
    snapshots = [[_word_window()], [_word_window()], []]

    def _next_snapshot() -> list[WindowInfo]:
        assert snapshots
        return snapshots.pop(0)

    monkeypatch.setattr("src.brain.module.snapshot_top_level_windows", _next_snapshot)
    monkeypatch.setattr("src.brain.module.asyncio.sleep", AsyncMock())

    def _close(_hwnd: int) -> dict[str, object]:
        raise ValueError("no window with hwnd 42")

    monkeypatch.setattr("src.brain.module.close_hwnd", _close)
    transcript: list[dict] = []

    def _capture(messages: list[dict], *_args: object, **_kwargs: object) -> None:
        transcript.extend(messages)

    brain._append_step_messages = _capture

    result = await brain.process_step()

    assert result.step_finished is True
    assert brain._script_step_index == 2
    payload = json.loads(transcript[1]["content"])
    assert payload["targets"][0]["status"] == "error"
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "completed"


@pytest.mark.asyncio
async def test_foreground_miss_does_not_retry_close(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_window_verifies = [
        {},
        {
            "foreground": {
                "class_name": "Chrome_WidgetWin_1",
                "title": "Google Chrome",
                "process_name": "chrome.exe",
            }
        },
        {},
    ]
    _loop_marks_close_windows(brain)
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
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)

    await brain.process_step()

    expected_sleeps = _expected_poll_sleeps(_signal_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert closed == []


@pytest.mark.asyncio
async def test_appeared_miss_does_not_retry_close(monkeypatch) -> None:
    brain = _brain_for_process_step()
    brain.script_settle_after_seconds = [None, 2.0, None]
    brain.script_window_verifies = [{}, _appeared_word_verify(), {}]
    _loop_marks_close_windows(brain)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)

    await brain.process_step()

    expected_sleeps = _expected_poll_sleeps(_appeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert closed == []


@pytest.mark.asyncio
async def test_disappeared_miss_without_close_windows_only_polls(
    monkeypatch,
) -> None:
    brain = _brain_for_process_step()
    brain.script_window_verifies = [{}, _word_disappeared_verify(), {}]
    brain.loop = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.brain.module.snapshot_top_level_windows",
        lambda: [_word_window()],
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)
    closed = _record_close_hwnds(monkeypatch)

    result = await brain.process_step()

    expected_sleeps = _expected_poll_sleeps(_disappeared_poll_deadline_seconds())
    assert sleep_mock.await_count == len(expected_sleeps)
    assert closed == []
    metadata = brain._update_step_metadata.call_args.args[2]
    assert metadata["status"] == "verify_failed"
    assert metadata["verify"]["clearly_unmet"] is True
    assert result.step_finished is True


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
