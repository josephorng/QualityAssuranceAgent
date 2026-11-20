from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.recorder.ime_state import (
    ImeInputState,
    _IME_SENDMESSAGE_TIMEOUT_MS,
    _SMTO_ABORTIFHUNG,
    _ime_send_message_timeout,
    read_ime_input_state,
)


def test_ime_input_state_labels() -> None:
    zh = ImeInputState(lang_id=0x0804, chinese_mode=True, conversion=1, opened=True)
    assert zh.lang_label == "0x0804"
    assert zh.mode_label == "zh"
    en = ImeInputState(lang_id=0x0409, chinese_mode=False, conversion=0, opened=False)
    assert en.mode_label == "en"
    unknown = ImeInputState(
        lang_id=None, chinese_mode=None, conversion=None, opened=None
    )
    assert unknown.mode_label == "?"
    assert unknown.lang_label == "?"


def test_read_ime_input_state_does_not_raise() -> None:
    state = read_ime_input_state()
    assert isinstance(state, ImeInputState)


def test_ime_send_message_timeout_returns_none_on_failure() -> None:
    user32 = MagicMock()
    user32.SendMessageTimeoutW.return_value = 0
    assert _ime_send_message_timeout(user32, 1, 0x0283, 1) is None
    args = user32.SendMessageTimeoutW.call_args[0]
    assert args[4] == _SMTO_ABORTIFHUNG
    assert args[5] == _IME_SENDMESSAGE_TIMEOUT_MS


def test_read_ime_input_state_timeout_yields_unknown_chinese_mode() -> None:
    """Hung IME queries must not raise; chinese_mode stays unknown."""
    user32 = MagicMock()
    imm32 = MagicMock()
    user32.GetForegroundWindow.return_value = 123
    user32.GetWindowThreadProcessId.return_value = 1
    user32.GetKeyboardLayout.return_value = 0x0404
    imm32.ImmGetDefaultIMEWnd.return_value = 456
    user32.SendMessageTimeoutW.return_value = 0

    fake_windll = MagicMock()
    fake_windll.user32 = user32
    fake_windll.imm32 = imm32

    with patch("os.name", "nt"), patch("ctypes.windll", fake_windll, create=True):
        state = read_ime_input_state()

    assert state.lang_id == 0x0404
    assert state.chinese_mode is None
    assert state.conversion is None
    assert state.opened is None
