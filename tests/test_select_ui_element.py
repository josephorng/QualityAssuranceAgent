from __future__ import annotations

from cua_mcp.select_ui_element import (
    UiDetection,
    _filter_ui_detections_by_icon_name,
    _ocr_regions_to_candidates,
)


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
    pua = "\ue000"
    regions = [
        ((0, 0, 10, 10), (5, 5), ["OK"]),
        ((20, 0, 10, 10), (25, 5), [pua]),
    ]
    text, icons = _ocr_regions_to_candidates(regions, need_text_anchor=True)
    assert len(text) == 1
    assert text[0].class_name == "text"
    assert text[0].text == "OK"
    assert len(icons) == 1
    assert icons[0].class_name == "ocr_icon"


def test_ocr_regions_skips_unknown_pua_icons() -> None:
    regions = [
        ((0, 0, 10, 10), (5, 5), ["\uf000"]),
        ((20, 0, 10, 10), (25, 5), ["\ue000"]),
    ]
    text, icons = _ocr_regions_to_candidates(regions, need_text_anchor=False)
    assert text == []
    assert len(icons) == 1
    assert icons[0].text == "\ue000"


def _icon_detection(chinese_id: str, *, cx: int = 0) -> UiDetection:
    return UiDetection(
        bbox=(cx, 0, 10, 10),
        cx=cx,
        cy=5,
        class_id=2,
        class_name="ocr_icon",
        text="",
        icons=[{"chinese_id": chinese_id}],
    )


def test_filter_detections_by_icon_similarity_keeps_matches() -> None:
    detections = [
        _icon_detection("資料夾", cx=10),
        _icon_detection("關閉", cx=20),
        _icon_detection("搜尋", cx=30),
    ]
    filtered = _filter_ui_detections_by_icon_name(detections, "資料夾")
    assert len(filtered) == 1
    assert filtered[0].cx == 10


def test_filter_detections_by_icon_similarity_keeps_all_when_no_match() -> None:
    detections = [
        _icon_detection("關閉", cx=20),
        _icon_detection("搜尋", cx=30),
    ]
    filtered = _filter_ui_detections_by_icon_name(detections, "資料夾")
    assert filtered == detections
