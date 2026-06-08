from __future__ import annotations

from cua_mcp.select_ui_element import _ocr_regions_to_candidates


def test_ocr_regions_pua_only_without_text_anchor() -> None:
    pua = "\ue000"
    regions = [
        ((0, 0, 10, 10), (5, 5), [pua]),
        ((20, 0, 10, 10), (25, 5), ["Submit"]),
        ((40, 0, 10, 10), (45, 5), []),
    ]
    text, icons = _ocr_regions_to_candidates(regions, need_text_anchor=False)
    assert text == []
    assert len(icons) == 1
    assert icons[0].class_name == "ocr_icon"
    assert icons[0].text == pua


def test_ocr_regions_includes_text_when_need_text_anchor() -> None:
    regions = [
        ((0, 0, 10, 10), (5, 5), ["OK"]),
        ((20, 0, 10, 10), (25, 5), ["\uf000"]),
    ]
    text, icons = _ocr_regions_to_candidates(regions, need_text_anchor=True)
    assert len(text) == 1
    assert text[0].class_name == "text"
    assert text[0].text == "OK"
    assert len(icons) == 1
    assert icons[0].class_name == "ocr_icon"
