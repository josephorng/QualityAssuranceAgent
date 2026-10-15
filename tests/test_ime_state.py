from __future__ import annotations

from src.recorder.ime_state import ImeInputState, read_ime_input_state


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
