from __future__ import annotations

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text, _parse_keep_indices_from_llm
from cua_mcp.yolo_onnx import (
    MOUSE_TARGET_CLASS_IDS,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
)


def test_mouse_target_class_ids() -> None:
    assert MOUSE_TARGET_CLASS_IDS == frozenset({
        YOLO_CLASS_TEXT,
        YOLO_CLASS_ELEMENT,
        YOLO_CLASS_INPUT,
        YOLO_CLASS_SCROLLBAR,
    })


def test_detection_from_bbox_text() -> None:
    det = _detection_from_bbox((10, 20, 100, 30), YOLO_CLASS_TEXT, text="Submit")
    assert det.class_name == "text"
    assert det.text == "Submit"
    assert det.cx == 60
    assert det.cy == 35


def test_detection_from_bbox_input() -> None:
    det = _detection_from_bbox((0, 0, 50, 20), YOLO_CLASS_INPUT)
    assert det.class_name == "input"
    assert det.text is None
    assert det.cx == 25
    assert det.cy == 10


def test_format_mouse_candidates_includes_class_and_text() -> None:
    detections = [
        _detection_from_bbox((0, 0, 80, 20), YOLO_CLASS_TEXT, text="OK"),
        _detection_from_bbox((100, 0, 12, 200), YOLO_CLASS_SCROLLBAR),
        _detection_from_bbox((200, 0, 50, 20), YOLO_CLASS_INPUT),
    ]
    text = _format_ui_candidates_text(detections)
    assert "[index 0] class=text" in text
    assert "text='OK'" in text
    assert "[index 1] class=滾動條" in text
    assert "[index 2] class=輸入欄" in text
    assert "center=[106,100]" in text


def test_format_mouse_candidates_omits_pua_only_text() -> None:
    pua = "\ue002"
    detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_ELEMENT, text=pua),
        _detection_from_bbox((30, 0, 80, 20), YOLO_CLASS_TEXT, text=f"OK{pua}"),
    ]
    text = _format_ui_candidates_text(detections)
    assert "text=" not in text.split("\n")[0]
    assert "icons=向下箭頭" in text.split("\n")[0]
    assert "text='OK" in text.split("\n")[1]


def test_parse_keep_indices_from_llm() -> None:
    raw = '{"keep_indices": [0, 2, 2, 99]}'
    keep = _parse_keep_indices_from_llm(raw, max_len=3)
    assert keep == [0, 2]


def test_offset_detection_preserves_class() -> None:
    from cua_mcp.select_mouse_target import _offset_detection

    det = _detection_from_bbox((5, 5, 10, 10), YOLO_CLASS_ELEMENT, text="icon")
    shifted = _offset_detection(det, 100, 200)
    assert shifted.bbox == (105, 205, 10, 10)
    assert shifted.cx == 110
    assert shifted.cy == 210
    assert shifted.class_name == "element"
    assert shifted.text == "icon"


def test_resolve_ocr_class_id_element_plain_text_becomes_unknown() -> None:
    from cua_mcp.select_mouse_target import _resolve_ocr_class_id
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "搜") == PICKER_CLASS_UNKNOWN
    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "g") == PICKER_CLASS_UNKNOWN


def test_resolve_ocr_class_id_keeps_element_for_known_pua_or_empty() -> None:
    from cua_mcp.select_mouse_target import _resolve_ocr_class_id

    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "") == YOLO_CLASS_ELEMENT
    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "\ue002") == YOLO_CLASS_ELEMENT
    assert _resolve_ocr_class_id(YOLO_CLASS_TEXT, "OK") == YOLO_CLASS_TEXT


def test_resolve_ocr_class_id_unknown_pua_only_becomes_unknown() -> None:
    from cua_mcp.select_mouse_target import _resolve_ocr_class_id
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    # Unmapped PUA, or the mapped ``unknown_icon`` entry (U+E01A).
    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "\uf000") == PICKER_CLASS_UNKNOWN
    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "\ue01a") == PICKER_CLASS_UNKNOWN
    assert _resolve_ocr_class_id(YOLO_CLASS_TEXT, "\ue01a") == PICKER_CLASS_UNKNOWN


def test_detection_from_bbox_unknown() -> None:
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    det = _detection_from_bbox((0, 0, 20, 20), PICKER_CLASS_UNKNOWN, text="搜")
    assert det.class_name == "unknown"
    assert det.text == "搜"


def test_format_mouse_candidates_includes_unknown() -> None:
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    detections = [
        _detection_from_bbox((0, 0, 20, 20), PICKER_CLASS_UNKNOWN, text="搜"),
    ]
    text = _format_ui_candidates_text(detections)
    assert "class=未知(Unknown)" in text
    assert "text='搜'" in text


def test_should_keep_known_pua() -> None:
    from cua_mcp.select_mouse_target import _known_icons_for_text, _resolve_ocr_class_id

    pua = "\ue002"
    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, pua) == YOLO_CLASS_ELEMENT
    icons = _known_icons_for_text(pua)
    assert icons
    assert not any("未知" in str(ii.get("chinese_id", "")) for ii in icons)


def test_should_keep_text_with_unmapped_pua_when_label_present() -> None:
    from cua_mcp.select_mouse_target import _known_icons_for_text, _resolve_ocr_class_id

    text = f"OK\uf000"
    assert _resolve_ocr_class_id(YOLO_CLASS_TEXT, text) == YOLO_CLASS_TEXT
    assert _known_icons_for_text(text) is None
