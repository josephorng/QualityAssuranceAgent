from __future__ import annotations

from src.hand.module import (
    _AFTER_ACTION_SETTLE_S,
    _AFTER_MOVE_SETTLE_S,
    _after_action_settle_s,
    _focus_rect_after_csv,
    _sample_debug_focus_rect_after,
    _sample_debug_foreground_rect_after,
)


def test_after_action_settle_s_is_shorter_for_move_mouse() -> None:
    assert _after_action_settle_s("move_mouse") == _AFTER_MOVE_SETTLE_S
    assert _after_action_settle_s("move_mouse_visual") == _AFTER_MOVE_SETTLE_S
    assert _AFTER_MOVE_SETTLE_S < _AFTER_ACTION_SETTLE_S


def test_after_action_settle_s_keeps_default_for_other_tools() -> None:
    assert _after_action_settle_s("click") == _AFTER_ACTION_SETTLE_S
    assert _after_action_settle_s("type_text") == _AFTER_ACTION_SETTLE_S


def test_sample_debug_focus_rect_after_formats_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.recorder.focus_point.sample_debug_focus_rect",
        lambda **_kwargs: (10, 20, 30, 40),
    )
    assert _sample_debug_focus_rect_after() == [10, 20, 30, 40]
    assert _focus_rect_after_csv([10, 20, 30, 40]) == "[10,20,30,40]"
    assert _focus_rect_after_csv(None) == ""


def test_sample_debug_foreground_rect_after_formats_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.recorder.verify_signals.sample_debug_foreground_rect",
        lambda: [1, 2, 300, 200],
    )
    assert _sample_debug_foreground_rect_after() == [1, 2, 300, 200]
