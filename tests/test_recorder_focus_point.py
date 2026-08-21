"""Tests for typing focus/caret screen-point resolution."""

from __future__ import annotations

from src.recorder.focus_point import resolve_typing_screen_xy


def test_resolve_typing_prefers_rect_center_when_last_click_outside_focus_rect() -> None:
    caret = (100, 200, 120, 220)  # center 110, 210; click far away
    point = resolve_typing_screen_xy(
        last_click_xy=(2416, 240),
        mouse_xy=(1, 1),
        caret_rect_fn=lambda: caret,
        uia_rect_fn=lambda: None,
    )
    assert point == (110, 210)


def test_resolve_typing_rejects_huge_uia_rect_for_last_click() -> None:
    huge = (0, 0, 1920, 1080)
    point = resolve_typing_screen_xy(
        last_click_xy=(400, 300),
        mouse_xy=(1, 1),
        caret_rect_fn=lambda: None,
        uia_rect_fn=lambda: huge,
    )
    assert point == (400, 300)


def test_resolve_typing_prefers_last_click_inside_caret_rect() -> None:
    caret = (100, 200, 300, 400)
    point = resolve_typing_screen_xy(
        last_click_xy=(150, 250),
        mouse_xy=(1, 1),
        caret_rect_fn=lambda: caret,
        uia_rect_fn=lambda: None,
    )
    assert point == (150, 250)


def test_resolve_typing_falls_back_to_uia_when_caret_missing() -> None:
    uia = (0, 0, 100, 50)  # center 50, 25
    point = resolve_typing_screen_xy(
        last_click_xy=None,
        mouse_xy=(9, 9),
        caret_rect_fn=lambda: None,
        uia_rect_fn=lambda: uia,
    )
    assert point == (50, 25)


def test_resolve_typing_prefers_last_click_inside_uia_rect() -> None:
    uia = (0, 0, 200, 100)
    point = resolve_typing_screen_xy(
        last_click_xy=(40, 30),
        mouse_xy=(9, 9),
        caret_rect_fn=lambda: (0, 0, 0, 0),
        uia_rect_fn=lambda: uia,
    )
    assert point == (40, 30)


def test_resolve_typing_last_resort_last_click_then_mouse() -> None:
    assert resolve_typing_screen_xy(
        last_click_xy=(400, 300),
        mouse_xy=(100, 100),
        caret_rect_fn=lambda: None,
        uia_rect_fn=lambda: None,
    ) == (400, 300)
    assert resolve_typing_screen_xy(
        last_click_xy=None,
        mouse_xy=(100, 100),
        caret_rect_fn=lambda: None,
        uia_rect_fn=lambda: None,
    ) == (100, 100)
    assert resolve_typing_screen_xy(
        last_click_xy=None,
        mouse_xy=None,
        caret_rect_fn=lambda: None,
        uia_rect_fn=lambda: None,
    ) is None


def test_resolve_typing_ignores_empty_caret_and_uses_uia() -> None:
    point = resolve_typing_screen_xy(
        last_click_xy=None,
        mouse_xy=(1, 1),
        caret_rect_fn=lambda: (10, 10, 10, 10),
        uia_rect_fn=lambda: (10, 20, 30, 40),
    )
    assert point == (20, 30)
