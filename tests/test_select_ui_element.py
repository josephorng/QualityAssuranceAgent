from __future__ import annotations

from cua_mcp.select_ui_element import (
    UiDetection,
    _filter_ui_detections_by_icon_name,
    _ocr_regions_to_candidates,
)
from cua_mcp.yolo_onnx import (
    PICKER_CLASS_OCR_ICON,
    UI_DETECTION_CLASS_IDS,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
)


def test_ui_detection_class_ids() -> None:
    assert UI_DETECTION_CLASS_IDS == frozenset({
        YOLO_CLASS_ELEMENT,
        YOLO_CLASS_INPUT,
        YOLO_CLASS_SCROLLBAR,
    })
    assert YOLO_CLASS_TEXT not in UI_DETECTION_CLASS_IDS


def test_ocr_regions_pua_and_text_candidates() -> None:
    pua = "\ue002"
    regions = [
        ((0, 0, 10, 10), (5, 5), [pua]),
        ((20, 0, 10, 10), (25, 5), ["Submit"]),
        ((40, 0, 10, 10), (45, 5), []),
    ]
    text, icons = _ocr_regions_to_candidates(regions)
    assert len(text) == 1
    assert text[0].class_name == "text"
    assert text[0].text == "Submit"
    assert len(icons) == 1
    assert icons[0].class_name == "ocr_icon"
    assert icons[0].text == pua


def test_ocr_regions_includes_text_and_pua_icons() -> None:
    pua = "\ue002"
    regions = [
        ((0, 0, 10, 10), (5, 5), ["OK"]),
        ((20, 0, 10, 10), (25, 5), [pua]),
    ]
    text, icons = _ocr_regions_to_candidates(regions)
    assert len(text) == 1
    assert text[0].class_name == "text"
    assert text[0].text == "OK"
    assert len(icons) == 1
    assert icons[0].class_name == "ocr_icon"


def test_ocr_regions_skips_unknown_pua_icons() -> None:
    regions = [
        ((0, 0, 10, 10), (5, 5), ["\uf000"]),
        ((20, 0, 10, 10), (25, 5), ["\ue002"]),
    ]
    text, icons = _ocr_regions_to_candidates(regions)
    assert text == []
    assert len(icons) == 1
    assert icons[0].text == "\ue002"


def _icon_detection(chinese_id: str, *, cx: int = 0) -> UiDetection:
    return UiDetection(
        bbox=(cx, 0, 10, 10),
        cx=cx,
        cy=5,
        class_id=PICKER_CLASS_OCR_ICON,
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
