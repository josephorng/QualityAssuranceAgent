"""Windows IME / input-language sampling for the desktop recorder."""

from __future__ import annotations

import os
from dataclasses import dataclass


# Primary language IDs (LANGID low byte).
_LANG_CHINESE = 0x04
_LANG_JAPANESE = 0x11
_LANG_KOREAN = 0x12

_WM_IME_CONTROL = 0x0283
_IMC_GETCONVERSIONMODE = 0x0001
_IMC_GETOPENSTATUS = 0x0005
_IME_CMODE_NATIVE = 0x0001


@dataclass(frozen=True)
class ImeInputState:
    """Snapshot of the foreground thread's input language + IME conversion mode."""

    lang_id: int | None
    """Low word of the keyboard layout HKL (e.g. 0x0409, 0x0804)."""

    chinese_mode: bool | None
    """True when a CJK IME is open in native/中 mode; False for 英/alphanumeric
    or a non-CJK layout; None when the OS could not be queried.
    """

    conversion: int | None
    """Raw Imm conversion-mode bits, when available."""

    opened: bool | None
    """Whether the IME reports open status."""

    @property
    def lang_label(self) -> str:
        if self.lang_id is None:
            return "?"
        return f"0x{self.lang_id:04x}"

    @property
    def mode_label(self) -> str:
        if self.chinese_mode is True:
            return "zh"
        if self.chinese_mode is False:
            return "en"
        return "?"


_UNKNOWN = ImeInputState(
    lang_id=None, chinese_mode=None, conversion=None, opened=None
)


def _is_cjk_lang(lang_id: int) -> bool:
    return (lang_id & 0xFF) in (_LANG_CHINESE, _LANG_JAPANESE, _LANG_KOREAN)


def read_ime_input_state() -> ImeInputState:
    """Read the foreground window thread's layout + 中/英 conversion mode.

    Uses ``GetKeyboardLayout`` for the input locale and
    ``ImmGetDefaultIMEWnd`` + ``WM_IME_CONTROL`` for conversion mode (the same
    signal the language bar's 中/英 indicator uses). Non-Windows hosts return
    an unknown snapshot.
    """
    if os.name != "nt":
        return _UNKNOWN
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        imm32 = ctypes.windll.imm32

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return _UNKNOWN

        tid = user32.GetWindowThreadProcessId(hwnd, None)
        hkl = int(user32.GetKeyboardLayout(tid)) & 0xFFFFFFFF
        lang_id = hkl & 0xFFFF

        if not _is_cjk_lang(lang_id):
            return ImeInputState(
                lang_id=lang_id,
                chinese_mode=False,
                conversion=None,
                opened=None,
            )

        hime = imm32.ImmGetDefaultIMEWnd(hwnd)
        if not hime:
            return ImeInputState(
                lang_id=lang_id,
                chinese_mode=None,
                conversion=None,
                opened=None,
            )

        # SendMessage returns LRESULT; conversion/open are small integers.
        conv = int(
            user32.SendMessageW(hime, _WM_IME_CONTROL, _IMC_GETCONVERSIONMODE, 0)
        )
        opened_raw = int(
            user32.SendMessageW(hime, _WM_IME_CONTROL, _IMC_GETOPENSTATUS, 0)
        )
        opened = bool(opened_raw)
        chinese_mode = bool(opened and (conv & _IME_CMODE_NATIVE))
        return ImeInputState(
            lang_id=lang_id,
            chinese_mode=chinese_mode,
            conversion=conv,
            opened=opened,
        )
    except Exception:
        return _UNKNOWN
