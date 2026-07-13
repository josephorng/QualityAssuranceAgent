from __future__ import annotations

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.select_ui_element import (
    UiDetection,
    _format_ui_candidates_relational,
    _format_ui_candidates_text,
    _parse_keep_indices_from_llm,
    _two_nearest_indices,
)
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
    assert "[index 0] class=文字(Text)" in text
    assert "text='OK'" in text
    assert "[index 1] class=滾動條(Scrollbar)" in text
    assert "[index 2] class=輸入欄(Input)" in text
    assert "center=[106,100]" in text
    assert "w=12" in text
    assert "h=200" in text


def test_format_mouse_candidates_omits_geometry() -> None:
    detections = [
        _detection_from_bbox((0, 0, 80, 20), YOLO_CLASS_TEXT, text="OK"),
        _detection_from_bbox((100, 0, 12, 200), YOLO_CLASS_SCROLLBAR),
    ]
    text = _format_ui_candidates_text(detections, include_geometry=False)
    assert "[index 0] class=文字(Text) text='OK'" in text
    assert "[index 1] class=滾動條(Scrollbar)" in text
    assert "center=" not in text
    assert " w=" not in text
    assert " h=" not in text


def test_format_mouse_candidates_omits_pua_only_text() -> None:
    pua = "\ue002"
    detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_ELEMENT, text=pua),
        _detection_from_bbox((30, 0, 80, 20), YOLO_CLASS_TEXT, text=f"OK{pua}"),
    ]
    text = _format_ui_candidates_text(detections)
    assert "text=" not in text.split("\n")[0]
    assert "icons=" in text.split("\n")[0]
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


def test_dedupe_overlapping_same_icon_keeps_one() -> None:
    from cua_mcp.select_mouse_target import _dedupe_overlapping_detections

    star = [{"chinese_id": "星號、我的最愛"}]
    # Near-identical boxes like the log (center=[539,528], w=16, h=16 vs h=15).
    a = UiDetection(
        bbox=(531, 520, 16, 16),
        cx=539,
        cy=528,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        icons=star,
    )
    b = UiDetection(
        bbox=(531, 521, 16, 15),
        cx=539,
        cy=528,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        icons=star,
    )
    kept = _dedupe_overlapping_detections([a, b])
    assert len(kept) == 1
    assert kept[0].bbox == (531, 520, 16, 16)


def test_dedupe_overlapping_same_text_keeps_one() -> None:
    from cua_mcp.select_mouse_target import _dedupe_overlapping_detections

    a = _detection_from_bbox((489, 245, 29, 15), YOLO_CLASS_TEXT, text="圖片")
    b = _detection_from_bbox((490, 245, 29, 15), YOLO_CLASS_TEXT, text="圖片")
    kept = _dedupe_overlapping_detections([a, b])
    assert len(kept) == 1
    assert kept[0].text == "圖片"


def test_dedupe_keeps_distinct_labels_even_if_overlapping() -> None:
    from cua_mcp.select_mouse_target import _dedupe_overlapping_detections

    a = _detection_from_bbox((100, 100, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "星號、我的最愛"}])
    b = _detection_from_bbox((101, 100, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "時鐘、鬧鐘"}])
    kept = _dedupe_overlapping_detections([a, b])
    assert len(kept) == 2


def test_dedupe_keeps_non_overlapping_same_label() -> None:
    from cua_mcp.select_mouse_target import _dedupe_overlapping_detections

    pin = [{"chinese_id": "圖釘"}]
    a = _detection_from_bbox((100, 100, 12, 12), YOLO_CLASS_ELEMENT, icons=pin)
    b = _detection_from_bbox((300, 100, 12, 12), YOLO_CLASS_ELEMENT, icons=pin)
    kept = _dedupe_overlapping_detections([a, b])
    assert len(kept) == 2


def test_iou_xywh_near_identical_boxes() -> None:
    from cua_mcp.geometry import iou_xywh

    a = (531, 520, 16, 16)
    b = (531, 521, 16, 15)
    assert iou_xywh(a, b) > 0.5


def test_expand_keep_indices_adds_same_text_and_icon_labels() -> None:
    from cua_mcp.select_mouse_target import _expand_keep_indices_with_similar

    detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "圖片"}]),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((100, 0, 30, 15), YOLO_CLASS_TEXT, text="圖片"),
        _detection_from_bbox((150, 0, 30, 15), YOLO_CLASS_TEXT, text="下載"),
        _detection_from_bbox((200, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "圖片"}]),
        _detection_from_bbox((250, 0, 30, 15), YOLO_CLASS_TEXT, text="圖片"),
        _detection_from_bbox((300, 0, 30, 15), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((350, 0, 40, 15), YOLO_CLASS_TEXT, text="文件\\Repos\\Git"),
    ]
    # LLM kept one 文件 and one 圖片 text row (under-recall).
    expanded = _expand_keep_indices_with_similar(detections, [1, 2])
    assert expanded == [0, 1, 2, 4, 5, 6]


def test_expand_keep_indices_ignores_blank_detections() -> None:
    from cua_mcp.select_mouse_target import _expand_keep_indices_with_similar

    detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_ELEMENT),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((100, 0, 20, 20), YOLO_CLASS_ELEMENT),
    ]
    expanded = _expand_keep_indices_with_similar(detections, [0, 1])
    assert expanded == [0, 1]


def test_fallback_filter_by_similarity_picks_anchor_and_nearby() -> None:
    from cua_mcp.select_mouse_target import _fallback_filter_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="下載"),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox(
            (100, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Chrome"}]
        ),
        _detection_from_bbox((150, 0, 30, 15), YOLO_CLASS_TEXT, text="圖片"),
    ]
    kept = _fallback_filter_by_similarity(
        detections,
        "「文件」文字",
        ["「Chrome」圖示", "「圖片」文字"],
    )
    assert [d.text or (d.icons or [{}])[0].get("chinese_id") for d in kept] == [
        "文件",
        "Chrome",
        "圖片",
    ]


def test_fallback_filter_by_similarity_dedupes_same_match() -> None:
    from cua_mcp.select_mouse_target import _fallback_filter_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="下載"),
    ]
    kept = _fallback_filter_by_similarity(
        detections,
        "「文件」文字",
        ["「文件」文字"],
    )
    assert len(kept) == 1
    assert kept[0].text == "文件"


def test_two_nearest_indices_by_center_distance() -> None:
    detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="遠"),  # center 10,10
        _detection_from_bbox((56, 280, 28, 14), YOLO_CLASS_TEXT, text="下載"),  # ~70,287
        _detection_from_bbox((54, 302, 30, 14), YOLO_CLASS_TEXT, text="文件"),  # ~69,309
        _detection_from_bbox((55, 334, 28, 14), YOLO_CLASS_TEXT, text="圖片"),  # ~69,341
        _detection_from_bbox((500, 500, 20, 20), YOLO_CLASS_TEXT, text="遠2"),
    ]
    assert _two_nearest_indices(detections, 2) == [1, 3]


def test_two_nearest_indices_single_and_empty() -> None:
    alone = [_detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="A")]
    assert _two_nearest_indices(alone, 0) == []
    pair = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="A"),
        _detection_from_bbox((40, 0, 20, 20), YOLO_CLASS_TEXT, text="B"),
    ]
    assert _two_nearest_indices(pair, 0) == [1]


def test_format_ui_candidates_relational_uses_neighbor_phrases() -> None:
    detections = [
        _detection_from_bbox((56, 280, 28, 14), YOLO_CLASS_TEXT, text="下載"),
        _detection_from_bbox((54, 302, 30, 14), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((55, 334, 28, 14), YOLO_CLASS_TEXT, text="圖片"),
    ]
    text = _format_ui_candidates_relational(detections)
    lines = text.split("\n")
    assert lines[1].startswith("[index 1] 「文件」文字（")
    assert "上方" in lines[1] and "「下載」文字" in lines[1]
    assert "下方" in lines[1] and "「圖片」文字" in lines[1]
    assert "center=" not in text
    assert " w=" not in text
    assert " h=" not in text


def test_format_ui_candidates_relational_single_candidate() -> None:
    detections = [_detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="文件")]
    text = _format_ui_candidates_relational(detections)
    assert text == "[index 0] 「文件」文字"


def test_format_ui_candidates_relational_icon_label() -> None:
    detections = [
        _detection_from_bbox((0, 0, 16, 16), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "下載"}]),
        _detection_from_bbox((40, 0, 30, 14), YOLO_CLASS_TEXT, text="文件"),
    ]
    text = _format_ui_candidates_relational(detections)
    assert "「下載」圖示" in text
    assert "「文件」文字" in text
    assert "右方" in text
    assert "左方" in text
