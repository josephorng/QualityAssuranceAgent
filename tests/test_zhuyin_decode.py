"""Unit tests for Microsoft Zhuyin (注音) keystroke decode and OCR reconcile."""

from __future__ import annotations

from src.recorder.zhuyin_decode import (
    decode_zhuyin_recorded,
    ime_is_zhuyin,
    is_zhuyin_lang,
)


def test_is_zhuyin_lang_taiwan() -> None:
    assert is_zhuyin_lang(0x0404)
    assert is_zhuyin_lang(1028)
    assert not is_zhuyin_lang(0x0804)
    assert not is_zhuyin_lang(None)


def test_ime_is_zhuyin_requires_chinese_and_taiwan() -> None:
    assert ime_is_zhuyin({"chinese": True, "lang_id": 1028})
    assert not ime_is_zhuyin({"chinese": False, "lang_id": 1028})
    assert not ime_is_zhuyin({"chinese": True, "lang_id": 0x0804})
    assert not ime_is_zhuyin(None)


def test_decode_rup_wu0_to_jintian() -> None:
    result = decode_zhuyin_recorded("rup wu0 ")
    assert result.ok
    assert result.decoded_text == "今天"
    assert result.reconciled_text == "今天"
    assert result.literal_prefix == ""
    assert [s.key for s in result.syllables] == ["rup", "wu0"]


def test_decode_with_digit_prefix_no_ocr() -> None:
    result = decode_zhuyin_recorded("1234567rup wu0 ")
    assert result.ok
    assert result.literal_prefix == "1234567"
    assert result.decoded_text == "1234567今天"
    assert result.reconciled_text == "1234567今天"
    assert result.ocr_compatible is False


def test_decode_rejects_wrong_sound_ocr_hetian() -> None:
    result = decode_zhuyin_recorded("1234567rup wu0 ", ocr_hint="1234567合天")
    assert result.ok
    assert result.literal_prefix == "1234567"
    assert result.reconciled_text == "1234567今天"
    assert result.ocr_compatible is False


def test_decode_keeps_homophone_ocr_jintian() -> None:
    result = decode_zhuyin_recorded("1234567rup wu0 ", ocr_hint="1234567今天")
    assert result.ok
    assert result.reconciled_text == "1234567今天"
    assert result.ocr_compatible is True


def test_decode_keeps_homophone_ocr_jintian_jin() -> None:
    """金 and 今 share ㄐㄧㄣ; OCR 金 must be preserved."""
    result = decode_zhuyin_recorded("1234567rup wu0 ", ocr_hint="1234567金天")
    assert result.ok
    assert result.reconciled_text == "1234567金天"
    assert result.ocr_compatible is True


def test_decode_tone_key_su3() -> None:
    result = decode_zhuyin_recorded("su3")
    assert result.ok
    assert result.decoded_text[0] == "你"
    assert "你" in result.syllables[0].candidates


def test_decode_failed_parse_leaves_ok_false() -> None:
    result = decode_zhuyin_recorded("hello world")
    assert not result.ok
    assert result.decoded_text == ""
