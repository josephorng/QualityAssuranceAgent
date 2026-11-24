"""UIA scroll SetScrollPercent / SetValue repair helpers and brain hooks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.brain.module import (
    BrainModule,
    DeferredWindowVerify,
    _SIGNAL_POLL_MIN_SECONDS,
    _WINDOW_VERIFY_POLL_INTERVAL_SECONDS,
    _reason_is_scroll_miss,
)
from src.common.models import ScriptStepVerifyResult
from src.recorder.verify_signals import (
    _range_value_absolute_from_percent,
    recorded_scroll_axis_targets,
)


def _brain() -> BrainModule:
    brain = BrainModule.__new__(BrainModule)
    brain.manager = MagicMock()
    brain.manager.log_info = MagicMock()
    brain.manager.log_error = MagicMock()
    brain.settings = MagicMock(script_max_step_attempts=3)
    brain._script_step_index = 7
    brain._step_transcript_counter = 1
    brain._replay_press_point_window = None
    brain._deferred_window_verify = None
    brain._step_close_windows_ok = False
    brain._update_step_metadata = MagicMock()
    brain._append_step_messages = MagicMock()
    brain._hand = MagicMock()
    brain._hand.capture_after_for_last_action = AsyncMock(return_value=None)
    brain._hand.assign_after_for_last_action = AsyncMock(return_value=None)
    brain.script_lines = ["scroll step"]
    brain.script_expected_outcomes = [None]
    brain.script_baseline_after_paths = [None]
    brain.script_window_verifies = [{"scroll": {"vertical": 0.0}}]
    brain.script_event_indices = []
    brain._script_event_indices = []
    brain._pending_after_capture = False
    brain._hold_step_retry_reason = None
    return brain


def _scroll_miss() -> ScriptStepVerifyResult:
    return ScriptStepVerifyResult(
        accomplished=False,
        branch="retry",
        target_step=None,
        clearly_unmet=True,
        reason="window verify missed scroll",
    )


def test_recorded_scroll_axis_targets_float_and_dict() -> None:
    assert recorded_scroll_axis_targets(19.5) == {"vertical": 19.5}
    assert recorded_scroll_axis_targets({"vertical": 0.0}) == {"vertical": 0.0}
    assert recorded_scroll_axis_targets(
        {"vertical": 10.0, "horizontal": 25.0}
    ) == {"vertical": 10.0, "horizontal": 25.0}
    assert recorded_scroll_axis_targets({"horizontal": 40}) == {"horizontal": 40.0}
    assert recorded_scroll_axis_targets(None) == {}
    assert recorded_scroll_axis_targets({}) == {}
    assert recorded_scroll_axis_targets({"vertical": "x"}) == {}


def test_range_value_absolute_from_percent_edges() -> None:
    assert _range_value_absolute_from_percent(0.0, 0.0, 100.0) == pytest.approx(0.0)
    assert _range_value_absolute_from_percent(100.0, 0.0, 100.0) == pytest.approx(
        100.0
    )
    assert _range_value_absolute_from_percent(50.0, 10.0, 30.0) == pytest.approx(20.0)
    assert _range_value_absolute_from_percent(50.0, 5.0, 5.0) is None


def test_reason_is_scroll_miss_gating() -> None:
    assert _reason_is_scroll_miss("window verify missed scroll") is True
    assert _reason_is_scroll_miss("window verify missed scroll: extra") is True
    assert _reason_is_scroll_miss("window verify missed foreground") is False
    assert _reason_is_scroll_miss("window verify missed disappeared: Word") is False
    assert _reason_is_scroll_miss(None) is False
    assert _reason_is_scroll_miss("") is False


@pytest.mark.asyncio
async def test_poll_window_verify_scroll_set_once_then_advances(
    monkeypatch,
) -> None:
    brain = _brain()
    observations = [
        (_scroll_miss(), []),
        (None, []),
    ]

    async def _observe(*_args: object, **_kwargs: object):
        assert observations
        return observations.pop(0)

    brain._window_verify_observation = _observe  # type: ignore[method-assign]
    set_mock = AsyncMock(return_value=True)
    brain._retry_set_recorded_scroll = set_mock  # type: ignore[method-assign]
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain._poll_window_verify(
        [],
        {"scroll": {"vertical": 0.0}},
        settle_after=float(_SIGNAL_POLL_MIN_SECONDS),
        script_step_index=7,
    )

    assert result is None
    assert set_mock.await_count == 1
    set_mock.assert_awaited_once_with({"scroll": {"vertical": 0.0}})
    assert sleep_mock.await_count == 1
    assert sleep_mock.await_args_list[0].args[0] == pytest.approx(
        _WINDOW_VERIFY_POLL_INTERVAL_SECONDS
    )


@pytest.mark.asyncio
async def test_poll_window_verify_scroll_set_at_most_once(
    monkeypatch,
) -> None:
    brain = _brain()
    miss = _scroll_miss()

    async def _observe(*_args: object, **_kwargs: object):
        return miss, []

    brain._window_verify_observation = _observe  # type: ignore[method-assign]
    set_mock = AsyncMock(return_value=False)
    brain._retry_set_recorded_scroll = set_mock  # type: ignore[method-assign]
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain._poll_window_verify(
        [],
        {"scroll": {"vertical": 0.0}},
        settle_after=float(_SIGNAL_POLL_MIN_SECONDS),
        script_step_index=7,
    )

    assert result is not None
    assert result.reason == "window verify missed scroll"
    assert set_mock.await_count == 1


@pytest.mark.asyncio
async def test_poll_window_verify_skips_set_for_non_scroll_miss(
    monkeypatch,
) -> None:
    brain = _brain()
    miss = ScriptStepVerifyResult(
        accomplished=False,
        branch="retry",
        target_step=None,
        clearly_unmet=True,
        reason="window verify missed foreground",
    )

    async def _observe(*_args: object, **_kwargs: object):
        return miss, []

    brain._window_verify_observation = _observe  # type: ignore[method-assign]
    set_mock = AsyncMock(return_value=True)
    brain._retry_set_recorded_scroll = set_mock  # type: ignore[method-assign]
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.brain.module.asyncio.sleep", sleep_mock)

    result = await brain._poll_window_verify(
        [],
        {"foreground": {"title": "x", "class_name": "y", "process_name": "z"}},
        settle_after=float(_SIGNAL_POLL_MIN_SECONDS),
        script_step_index=0,
    )

    assert result is not None
    assert "foreground" in result.reason
    assert set_mock.await_count == 0


@pytest.mark.asyncio
async def test_deferred_scroll_set_advances_without_full_retry() -> None:
    brain = _brain()
    recorded = {"scroll": {"vertical": 0.0}}
    brain._deferred_window_verify = DeferredWindowVerify(
        script_step_index=0,
        windows_before=[],
        recorded=recorded,
        replay_press_point_window=None,
        step_close_windows_ok=False,
    )
    observations = [
        (_scroll_miss(), []),
        (None, []),
    ]

    async def _observe(*_args: object, **_kwargs: object):
        assert observations
        return observations.pop(0)

    brain._window_verify_observation = _observe  # type: ignore[method-assign]
    set_mock = AsyncMock(return_value=True)
    brain._retry_set_recorded_scroll = set_mock  # type: ignore[method-assign]
    brain._apply_verify_branch = MagicMock(return_value=False)  # type: ignore[method-assign]
    brain._auto_advance_verify_result = MagicMock(  # type: ignore[method-assign]
        return_value=ScriptStepVerifyResult(
            accomplished=True,
            branch="advance",
            target_step=None,
            clearly_unmet=False,
            reason="Actor completed the step with all tools ok; window verify passed.",
        )
    )
    brain._current_goal = MagicMock(return_value="scroll")  # type: ignore[method-assign]
    brain._current_expected_outcome = MagicMock(return_value=None)  # type: ignore[method-assign]
    brain._current_baseline_after_path = MagicMock(return_value=None)  # type: ignore[method-assign]
    brain._script_step_attempt_number = MagicMock(return_value=1)  # type: ignore[method-assign]
    brain._clear_deferred_window_verify = MagicMock()  # type: ignore[method-assign]

    result = await brain._try_deferred_window_verify_after_countdown(
        transcript_counter=1,
        script_step_index=0,
        started_iso="2026-01-01T00:00:00+00:00",
        started_at=0.0,
    )

    assert result is not None
    assert result.step_finished is True
    assert set_mock.await_count == 1
    set_mock.assert_awaited_once_with(recorded)
    brain._apply_verify_branch.assert_called_once()


@pytest.mark.asyncio
async def test_deferred_scroll_set_still_miss_falls_through() -> None:
    brain = _brain()
    recorded = {"scroll": {"vertical": 0.0}}
    brain._deferred_window_verify = DeferredWindowVerify(
        script_step_index=0,
        windows_before=[],
        recorded=recorded,
        replay_press_point_window=None,
        step_close_windows_ok=False,
    )

    async def _observe(*_args: object, **_kwargs: object):
        return _scroll_miss(), []

    brain._window_verify_observation = _observe  # type: ignore[method-assign]
    set_mock = AsyncMock(return_value=False)
    brain._retry_set_recorded_scroll = set_mock  # type: ignore[method-assign]

    result = await brain._try_deferred_window_verify_after_countdown(
        transcript_counter=1,
        script_step_index=0,
        started_iso="2026-01-01T00:00:00+00:00",
        started_at=0.0,
    )

    assert result is None
    assert set_mock.await_count == 1


@pytest.mark.asyncio
async def test_retry_set_recorded_scroll_passes_axes(monkeypatch) -> None:
    brain = _brain()
    calls: list[dict] = []

    def _fake_set(**kwargs: object) -> bool:
        calls.append(dict(kwargs))
        return True

    monkeypatch.setattr("src.brain.module.set_scroll_percents", _fake_set)

    ok = await brain._retry_set_recorded_scroll(
        {"scroll": {"vertical": 0.0, "horizontal": 12.5}}
    )

    assert ok is True
    assert calls == [{"vertical": 0.0, "horizontal": 12.5}]
