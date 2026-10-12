from __future__ import annotations

import pytest

from cua_mcp import hand_tools


@pytest.fixture(autouse=True)
def _clear_clipboard_backup() -> None:
    hand_tools._clipboard_backup = None
    hand_tools._clipboard_backup_set = False
    yield
    hand_tools._clipboard_backup = None
    hand_tools._clipboard_backup_set = False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NBA live", "NBA live"),
        ('<|"|>NBA live<|"|>', "NBA live"),
        ('<|"|>"quoted"<|"|>', '"quoted"'),
        ("  keep spaces  ", "  keep spaces  "),
    ],
)
def test_normalize_typed_text(raw: str, expected: str) -> None:
    assert hand_tools._normalize_typed_text(raw) == expected


def test_type_text_strips_model_quote_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = {"value": ""}

    def fake_paste() -> str:
        return clipboard["value"]

    def fake_copy(text: str) -> None:
        clipboard["value"] = text

    pasted: list[str] = []

    def fake_hotkey(*keys: str) -> None:
        if keys == ("ctrl", "v"):
            pasted.append(clipboard["value"])

    monkeypatch.setattr(
        hand_tools,
        "pyperclip",
        type("P", (), {"paste": staticmethod(fake_paste), "copy": staticmethod(fake_copy)})(),
    )
    monkeypatch.setattr(hand_tools.pyautogui, "hotkey", fake_hotkey)
    monkeypatch.setattr(hand_tools, "sleep", lambda _s: None)

    result = hand_tools.type_text('<|"|>NBA live<|"|>')

    assert pasted == ["NBA live"]
    assert result["text"] == "NBA live"


def test_type_text_leaves_typed_text_on_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = {"value": "previously copied"}

    def fake_paste() -> str:
        return clipboard["value"]

    def fake_copy(text: str) -> None:
        clipboard["value"] = text

    hotkeys: list[tuple[str, str]] = []

    def fake_hotkey(*keys: str) -> None:
        hotkeys.append(tuple(keys))

    monkeypatch.setattr(
        hand_tools,
        "pyperclip",
        type("P", (), {"paste": staticmethod(fake_paste), "copy": staticmethod(fake_copy)})(),
    )
    monkeypatch.setattr(hand_tools.pyautogui, "hotkey", fake_hotkey)
    monkeypatch.setattr(hand_tools, "sleep", lambda _s: None)

    result = hand_tools.type_text("typed value")

    assert hotkeys == [("ctrl", "v")]
    assert clipboard["value"] == "typed value"
    assert result["effective_mode"] == "paste"
    assert "clipboard_restored" not in result


def test_backup_and_restore_clipboard_around_type_text(monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = {"value": "previously copied"}

    def fake_paste() -> str:
        return clipboard["value"]

    def fake_copy(text: str) -> None:
        clipboard["value"] = text

    monkeypatch.setattr(
        hand_tools,
        "pyperclip",
        type("P", (), {"paste": staticmethod(fake_paste), "copy": staticmethod(fake_copy)})(),
    )
    monkeypatch.setattr(hand_tools.pyautogui, "hotkey", lambda *keys: None)
    monkeypatch.setattr(hand_tools, "sleep", lambda _s: None)

    backup = hand_tools.backup_clipboard()
    assert backup["backed_up"] is True

    hand_tools.type_text("typed value")
    assert clipboard["value"] == "typed value"

    restore = hand_tools.restore_clipboard()
    assert restore["restored"] is True
    assert clipboard["value"] == "previously copied"
    assert hand_tools._clipboard_backup_set is False


def test_type_text_still_pastes_when_clipboard_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = {"value": "untouched"}

    def boom_paste() -> str:
        raise RuntimeError("clipboard locked")

    def fake_copy(text: str) -> None:
        clipboard["value"] = text

    monkeypatch.setattr(
        hand_tools,
        "pyperclip",
        type("P", (), {"paste": staticmethod(boom_paste), "copy": staticmethod(fake_copy)})(),
    )
    monkeypatch.setattr(hand_tools.pyautogui, "hotkey", lambda *keys: None)
    monkeypatch.setattr(hand_tools, "sleep", lambda _s: None)

    result = hand_tools.type_text("typed value")

    assert clipboard["value"] == "typed value"
    assert result["effective_mode"] == "paste"


def test_restore_is_noop_when_backup_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = {"value": "untouched"}

    def boom_paste() -> str:
        raise RuntimeError("clipboard locked")

    def fake_copy(text: str) -> None:
        clipboard["value"] = text

    monkeypatch.setattr(
        hand_tools,
        "pyperclip",
        type("P", (), {"paste": staticmethod(boom_paste), "copy": staticmethod(fake_copy)})(),
    )

    backup = hand_tools.backup_clipboard()
    assert backup["backed_up"] is False

    clipboard["value"] = "typed value"
    restore = hand_tools.restore_clipboard()
    assert restore["restored"] is False
    assert clipboard["value"] == "typed value"
    assert hand_tools._clipboard_backup_set is False
