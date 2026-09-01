from __future__ import annotations

from src.hand.module import (
    _AFTER_ACTION_SETTLE_S,
    _AFTER_MOVE_SETTLE_S,
    _after_action_settle_s,
)


def test_after_action_settle_s_is_shorter_for_move_mouse() -> None:
    assert _after_action_settle_s("move_mouse") == _AFTER_MOVE_SETTLE_S
    assert _after_action_settle_s("move_mouse_visual") == _AFTER_MOVE_SETTLE_S
    assert _AFTER_MOVE_SETTLE_S < _AFTER_ACTION_SETTLE_S


def test_after_action_settle_s_keeps_default_for_other_tools() -> None:
    assert _after_action_settle_s("click") == _AFTER_ACTION_SETTLE_S
    assert _after_action_settle_s("type_text") == _AFTER_ACTION_SETTLE_S
