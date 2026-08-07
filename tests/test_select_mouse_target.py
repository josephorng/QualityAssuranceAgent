from __future__ import annotations

from pathlib import Path

import pytest

from cua_mcp.select_mouse_target import (
    _detection_from_bbox,
    _detections_similar_to,
    _local_bbox_on_monitor,
    _merge_nearby_labels,
    _monitor_index_from_image_path,
    _normalize_nearby_labels,
    _prefilter_anchors_by_nearby,
    _write_indexed_bbox_overlay_images,
)
from cua_mcp.select_ui_element import (
    UiDetection,
    _assign_exclusive_neighbors_to_anchors,
    _format_ui_candidates_relational,
    _format_ui_candidates_text,
    _format_ui_candidates_with_functions,
    _parse_anchor_nearby_indices_from_llm,
    _parse_function_descriptions_from_llm,
    _parse_index_from_llm,
    _parse_keep_indices_from_llm,
    _two_nearest_indices,
)
from cua_mcp.yolo_onnx import (
    MOUSE_TARGET_CLASS_IDS,
    PICKER_CLASS_UNKNOWN,
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


def test_normalize_nearby_labels_strips_and_dedupes() -> None:
    assert _normalize_nearby_labels(None) == []
    assert _normalize_nearby_labels([]) == []
    assert _normalize_nearby_labels(
        [" 「Edge」圖示 ", "", "「Copilot」圖示", "「Edge」圖示", 12]  # type: ignore[list-item]
    ) == ["「Edge」圖示", "「Copilot」圖示"]


def test_merge_nearby_labels_prefers_earlier_sources() -> None:
    assert _merge_nearby_labels(
        ["「Edge」圖示", "「Copilot」圖示"],
        ["「Copilot」圖示", "「Chrome」圖示"],
        None,
    ) == ["「Edge」圖示", "「Copilot」圖示", "「Chrome」圖示"]


@pytest.mark.asyncio
async def test_resolve_mouse_point_merges_nearby_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from cua_mcp.select_mouse_target import resolve_mouse_point

    captured_nearby: dict[str, list[str]] = {}

    async def fake_parse(instruction: str):
        assert "資料夾" in instruction
        return "「資料夾」圖示", 0, 0, ["「Chrome」圖示"]

    def fake_filter(detections, anchor, nearby):
        captured_nearby["labels"] = list(nearby)
        return [detections[0]], []

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.parse_mouse_target_instruction",
        fake_parse,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.selected_eye_monitor_indices",
        lambda: [1],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.capture_monitor_to_file",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.cv2.imread",
        lambda *_args, **_kwargs: np.zeros((10, 10, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._collect_monitor_detections",
        lambda *_args, **_kwargs: [
            _detection_from_bbox(
                (0, 0, 20, 20),
                YOLO_CLASS_ELEMENT,
                icons=[{"chinese_id": "資料夾"}],
            )
        ],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._filter_mouse_candidates",
        fake_filter,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._run_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type("P", (), {"yolo_ocr_dir": __import__("pathlib").Path(".")})()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )

    gx, gy, meta = await resolve_mouse_point(
        "「資料夾」圖示",
        nearby_objects=["「Edge」圖示", "「Copilot」圖示"],
    )

    assert captured_nearby["labels"] == [
        "「Edge」圖示",
        "「Copilot」圖示",
        "「Chrome」圖示",
    ]
    assert meta["nearby_objects"] == captured_nearby["labels"]
    assert (gx, gy) == (10, 10)

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


def test_format_mouse_candidates_omits_all_text_for_elements() -> None:
    """Element rows never expose OCR text, including mixed PUA+noise."""
    detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_ELEMENT, text="\ue014)"),
        _detection_from_bbox((30, 0, 20, 20), YOLO_CLASS_ELEMENT, text="\ue012e"),
        _detection_from_bbox((60, 0, 80, 20), YOLO_CLASS_TEXT, text="OK"),
    ]
    text = _format_ui_candidates_text(detections, include_geometry=False)
    lines = text.split("\n")
    assert lines[0].startswith("[index 0] class=元素(Element)")
    assert "text=" not in lines[0]
    assert lines[1].startswith("[index 1] class=元素(Element)")
    assert "text=" not in lines[1]
    assert "text='OK'" in lines[2]


def test_parse_keep_indices_from_llm() -> None:
    raw = '{"keep_indices": [0, 2, 2, 99]}'
    keep = _parse_keep_indices_from_llm(raw, max_len=3)
    assert keep == [0, 2]


def test_parse_index_from_llm_returns_index_and_text() -> None:
    raw = (
        '{"index": 1, "text": "「文件」文字 center=(100,200)'
        '（左方27個像素有「目」未知、下方32個像素有「圖片」文字）"}'
    )
    idx, text = _parse_index_from_llm(raw, num_candidates=4)
    assert idx == 1
    assert text.startswith("「文件」文字 center=(100,200)")


def test_parse_index_from_llm_requires_text() -> None:
    with pytest.raises(ValueError, match="index.*text"):
        _parse_index_from_llm('{"index": 0}', num_candidates=2)


def test_parse_index_from_llm_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        _parse_index_from_llm('{"index": 0, "text": "  "}', num_candidates=2)


def test_parse_function_descriptions_from_llm() -> None:
    raw = (
        '{"items": ['
        '{"index": 1, "function": "工作列搜尋"}, '
        '{"index": 0, "function": "Outlook 郵件搜尋"}'
        "]}"
    )
    funcs = _parse_function_descriptions_from_llm(raw, num_candidates=2)
    assert funcs == ["Outlook 郵件搜尋", "工作列搜尋"]


def test_parse_function_descriptions_requires_all_indices() -> None:
    with pytest.raises(ValueError, match="missing function"):
        _parse_function_descriptions_from_llm(
            '{"items": [{"index": 0, "function": "only one"}]}',
            num_candidates=2,
        )


def test_parse_function_descriptions_rejects_duplicate_index() -> None:
    with pytest.raises(ValueError, match="duplicate index"):
        _parse_function_descriptions_from_llm(
            '{"items": ['
            '{"index": 0, "function": "a"}, '
            '{"index": 0, "function": "b"}'
            "]}",
            num_candidates=2,
        )


def test_detections_similar_to_groups_same_label() -> None:
    outlook = _detection_from_bbox((2240, 20, 30, 16), YOLO_CLASS_TEXT, text="搜尋")
    taskbar = _detection_from_bbox((2550, 1040, 30, 16), YOLO_CLASS_TEXT, text="搜尋")
    left_bar = _detection_from_bbox((630, 1040, 30, 16), YOLO_CLASS_TEXT, text="搜尋")
    other = _detection_from_bbox((100, 100, 30, 16), YOLO_CLASS_TEXT, text="關閉")
    detections = [outlook, taskbar, left_bar, other]

    peers = _detections_similar_to(taskbar, detections)
    assert len(peers) == 3
    # Reading order: top row first, then left-to-right on the bottom row.
    assert peers == [outlook, left_bar, taskbar]
    assert other not in peers


def test_detections_similar_to_reading_order_not_chosen_first() -> None:
    """Similar peers are indexed in reading order even when chosen is lower on screen."""
    first = _detection_from_bbox((100, 100, 80, 16), YOLO_CLASS_TEXT, text="104企業大師")
    second = _detection_from_bbox((100, 160, 80, 16), YOLO_CLASS_TEXT, text="104企業大師")
    third = _detection_from_bbox((100, 220, 80, 16), YOLO_CLASS_TEXT, text="104企業大師")
    detections = [first, second, third]

    peers = _detections_similar_to(third, detections)
    assert peers == [first, second, third]
    assert peers.index(third) == 2


def test_detections_similar_to_unique_label_is_singleton() -> None:
    only = _detection_from_bbox((100, 100, 30, 16), YOLO_CLASS_TEXT, text="唯一")
    other = _detection_from_bbox((200, 200, 30, 16), YOLO_CLASS_TEXT, text="關閉")
    peers = _detections_similar_to(only, [only, other])
    assert peers == [only]


def test_monitor_index_from_image_path() -> None:
    assert _monitor_index_from_image_path(r"C:\tmp\stamp_mon2.png") == 2
    assert _monitor_index_from_image_path("runs/yolo_ocr/2026_mon1.png") == 1
    assert _monitor_index_from_image_path("shot.png") is None


def test_local_bbox_on_monitor_clips_and_rejects_offscreen() -> None:
    assert _local_bbox_on_monitor(
        (100, 50, 40, 20),
        left=100,
        top=50,
        img_w=200,
        img_h=100,
    ) == (0, 0, 40, 20)
    assert (
        _local_bbox_on_monitor(
            (0, 0, 10, 10),
            left=100,
            top=50,
            img_w=200,
            img_h=100,
        )
        is None
    )


def test_write_indexed_bbox_overlay_images(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    src = tmp_path / "cap_mon2.png"
    blank = np.zeros((120, 200, 3), dtype=np.uint8)
    import cv2

    assert cv2.imwrite(str(src), blank)

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._monitor_geometry",
        lambda _idx: (1000, 0, 200, 120),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._log_info",
        lambda *_a, **_k: None,
    )

    # Virtual-desktop bbox on monitor 2 (left=1000): local (20,30)
    det = _detection_from_bbox((1020, 30, 40, 20), YOLO_CLASS_TEXT, text="搜尋")
    out_paths = _write_indexed_bbox_overlay_images(
        [det],
        [str(src)],
        [2],
        tmp_path,
        stamp="t1",
    )
    assert len(out_paths) == 1
    annotated = Path(out_paths[0])
    assert annotated.name == "t1_indexed_mon2.png"
    assert annotated.is_file()
    img = cv2.imread(str(annotated))
    assert img is not None
    # Yellow pixel should appear near the labeled box region.
    assert img[30:50, 20:60].max() > 0


def test_format_ui_candidates_with_functions_appends_role() -> None:
    detections = [
        _detection_from_bbox((2240, 20, 30, 16), YOLO_CLASS_TEXT, text="搜尋"),
        _detection_from_bbox((2550, 1040, 30, 16), YOLO_CLASS_TEXT, text="搜尋"),
    ]
    text = _format_ui_candidates_with_functions(
        detections,
        ["Outlook 郵件搜尋", "Windows 工作列搜尋"],
    )
    lines = text.split("\n")
    assert lines[0].endswith("功能：Outlook 郵件搜尋")
    assert lines[1].endswith("功能：Windows 工作列搜尋")


def test_parse_anchor_nearby_indices_from_llm() -> None:
    raw = '{"anchor_indices": [1, 1, 99], "nearby_indices": [0, 2, 1]}'
    anchor, nearby = _parse_anchor_nearby_indices_from_llm(raw, max_len=3)
    assert anchor == [1]
    # 1 is also in nearby_raw but must be dropped because it is an anchor match.
    assert nearby == [0, 2]


def test_parse_anchor_nearby_indices_requires_both_keys() -> None:
    try:
        _parse_anchor_nearby_indices_from_llm(
            '{"keep_indices": [0]}', max_len=3
        )
    except ValueError as exc:
        assert "anchor_indices" in str(exc)
    else:
        raise AssertionError("expected ValueError")


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


def test_normalize_similarity_label_strips_hub_wrappers() -> None:
    from cua_mcp.select_mouse_target import _normalize_similarity_label

    assert _normalize_similarity_label("「擷取」文字") == "擷取"
    assert _normalize_similarity_label("『檔案』圖示") == "檔案"
    assert _normalize_similarity_label("【Edge】圖示") == "Edge"
    assert _normalize_similarity_label("〔排序〕文字") == "排序"
    assert _normalize_similarity_label('[Submit]按鈕') == "Submit"
    assert _normalize_similarity_label('"Chrome"圖示') == "Chrome"
    assert _normalize_similarity_label("  「Edge」圖示  ") == "Edge"
    assert _normalize_similarity_label("擷取") == "擷取"
    assert _normalize_similarity_label("輸入欄") == "輸入欄"
    assert _normalize_similarity_label("") == ""
    assert _normalize_similarity_label("「」文字") == "「」文字"
    assert _normalize_similarity_label('""文字') == '""文字'


def test_label_similarity_hub_query_matches_ocr_near_miss() -> None:
    from cua_mcp.select_mouse_target import _label_similarity

    # OCR misread 擷取 → 握取; hub wrapper must not dilute the ratio below threshold.
    assert _label_similarity("「擷取」文字", "握取") == 0.5
    assert _label_similarity("『擷取』文字", "握取") == 0.5
    assert _label_similarity("【擷取】", "擷取") == 1.0
    assert _label_similarity('"擷取"', "擷取") == 1.0
    assert _label_similarity("[擷取]文字", "擷取") == 1.0
    assert _label_similarity("「擷取」文字", "文字文件") == 0.0
    assert _label_similarity("「Edge」圖示", "Edge") == 1.0


def test_prefilter_keeps_ocr_near_miss_over_shared_suffix() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="握取"),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="文字文件"),
        _detection_from_bbox((100, 0, 30, 15), YOLO_CLASS_TEXT, text="新增文字文件.txt"),
    ]
    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections, "「擷取」文字", []
    )
    assert [detections[i].text for i in anchor_indices] == ["握取"]
    assert nearby_indices == []


def test_prefilter_matches_ocr_typo_panel_settings() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 51, 12), YOLO_CLASS_TEXT, text="面板設定"),
        _detection_from_bbox((50, 0, 75, 17), YOLO_CLASS_TEXT, text="管理面板"),
        _detection_from_bbox(
            (100, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "螢幕"}]
        ),
    ]
    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections,
        "「面板詢定」文字",
        ["「W」未知", "「螢幕」圖示"],
    )
    assert [detections[i].text for i in anchor_indices] == ["面板設定"]
    assert [detections[i].class_name for i in nearby_indices] == ["element"]


def test_prefilter_keeps_top_anchor_score_excludes_partial_panel_match() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 51, 12), YOLO_CLASS_TEXT, text="面板設定"),
        _detection_from_bbox((50, 0, 75, 17), YOLO_CLASS_TEXT, text="管理面板"),
        _detection_from_bbox((100, 0, 53, 12), YOLO_CLASS_TEXT, text="資訊面板"),
    ]
    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections,
        "「面板設定」文字",
        [],
    )
    assert [detections[i].text for i in anchor_indices] == ["面板設定"]
    assert nearby_indices == []


def test_prefilter_keeps_tied_top_anchor_scores() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="Submit"),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="Submit"),
        _detection_from_bbox((100, 0, 30, 15), YOLO_CLASS_TEXT, text="Cancel"),
    ]
    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections,
        "「Submit」文字",
        [],
    )
    assert [detections[i].text for i in anchor_indices] == ["Submit", "Submit"]
    assert nearby_indices == []


def test_prefilter_nearby_keeps_only_top_similarity_score() -> None:
    """「新竹公司」must not also pull in lower-scoring 「新竹總部」as a landmark."""
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 9, 11), YOLO_CLASS_TEXT, text="", icons=[{"chinese_id": "展開節點"}]),
        _detection_from_bbox((50, 0, 80, 14), YOLO_CLASS_TEXT, text="|龅速的網域 (3)"),
        _detection_from_bbox((50, 30, 72, 13), YOLO_CLASS_TEXT, text="新竹公司 (17)"),
        _detection_from_bbox((50, 90, 66, 13), YOLO_CLASS_TEXT, text="新竹總部 (6)"),
        _detection_from_bbox((200, 0, 60, 12), YOLO_CLASS_TEXT, text="龜速的網域"),
    ]
    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections,
        "「展開節點」圖示",
        ["「龜速的網域 (3)」文字", "「新竹公司 (17)」文字"],
    )
    assert [detections[i].icons[0]["chinese_id"] for i in anchor_indices] == ["展開節點"]
    assert [detections[i].text for i in nearby_indices] == [
        "|龅速的網域 (3)",
        "新竹公司 (17)",
    ]


def test_prefilter_nearby_keeps_tied_top_scores() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "資料夾"}]),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="圖片"),
        _detection_from_bbox((100, 0, 30, 15), YOLO_CLASS_TEXT, text="圖片"),
        _detection_from_bbox((150, 0, 30, 15), YOLO_CLASS_TEXT, text="圖檔"),
    ]
    _, nearby_indices = _prefilter_detections_by_similarity(
        detections,
        "「資料夾」圖示",
        ["「圖片」文字"],
    )
    assert [detections[i].text for i in nearby_indices] == ["圖片", "圖片"]


def test_filter_mouse_candidates_splits_anchor_and_nearby_by_similarity() -> None:
    from cua_mcp.select_mouse_target import _filter_mouse_candidates

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="下載"),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox(
            (100, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Chrome"}]
        ),
        _detection_from_bbox((150, 0, 30, 15), YOLO_CLASS_TEXT, text="圖片"),
    ]
    anchor_matches, nearby_matches = _filter_mouse_candidates(
        detections,
        "「文件」文字",
        ["「Chrome」圖示", "「圖片」文字"],
    )
    assert [d.text for d in anchor_matches] == ["文件"]
    assert [d.text or (d.icons or [{}])[0].get("chinese_id") for d in nearby_matches] == [
        "Chrome",
        "圖片",
    ]


def test_filter_mouse_candidates_dedupes_anchor_from_nearby() -> None:
    from cua_mcp.select_mouse_target import _filter_mouse_candidates

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="下載"),
    ]
    anchor_matches, nearby_matches = _filter_mouse_candidates(
        detections,
        "「文件」文字",
        ["「文件」文字"],
    )
    assert len(anchor_matches) == 1
    assert anchor_matches[0].text == "文件"
    assert nearby_matches == []


def test_prefilter_detections_by_similarity_keeps_anchor_and_nearby() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="簡"),
        _detection_from_bbox(
            (50, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "資料夾"}]
        ),
        _detection_from_bbox(
            (100, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Edge"}]
        ),
        _detection_from_bbox(
            (150, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Copilot"}]
        ),
        _detection_from_bbox(
            (200, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Chrome"}]
        ),
        _detection_from_bbox((250, 0, 30, 15), YOLO_CLASS_TEXT, text="下載"),
    ]
    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections,
        "資料夾",
        ["「Edge」圖示", "「Copilot」圖示"],
    )
    anchor_labels = [
        detections[i].text or (detections[i].icons or [{}])[0].get("chinese_id")
        for i in anchor_indices
    ]
    nearby_labels = [
        detections[i].text or (detections[i].icons or [{}])[0].get("chinese_id")
        for i in nearby_indices
    ]
    assert anchor_labels == ["資料夾"]
    assert nearby_labels == ["Edge", "Copilot"]


def test_prefilter_detections_by_similarity_empty_when_no_match() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 30, 15), YOLO_CLASS_TEXT, text="簡"),
        _detection_from_bbox(
            (50, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Chrome"}]
        ),
    ]
    assert _prefilter_detections_by_similarity(
        detections,
        "資料夾",
        ["「Edge」圖示"],
    ) == ([], [])


def test_filter_mouse_candidates_empty_similarity_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp.select_mouse_target import _filter_mouse_candidates

    class _FakeRunManager:
        def log_info(self, message: str) -> None:
            return None

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._run_manager",
        lambda: _FakeRunManager(),
    )

    detections = [
        _detection_from_bbox(
            (0, 0, 448, 12),
            YOLO_CLASS_TEXT,
            text="我們名樣的豐富和美善,為要讓我們可以高學神、服事神,不是為了要服事我們自...",
        ),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="搜尋聊天和訊息"),
    ]
    anchor_matches, nearby_matches = _filter_mouse_candidates(
        detections,
        "「我自己」文字",
        [],
    )
    assert anchor_matches == []
    assert nearby_matches == []


def test_prefilter_detections_by_similarity_keeps_input_and_scrollbar() -> None:
    from cua_mcp.select_mouse_target import _prefilter_detections_by_similarity

    detections = [
        _detection_from_bbox((0, 0, 80, 24), YOLO_CLASS_INPUT),
        _detection_from_bbox((100, 0, 12, 100), YOLO_CLASS_SCROLLBAR),
        _detection_from_bbox((50, 0, 30, 15), YOLO_CLASS_TEXT, text="排序"),
        _detection_from_bbox(
            (200, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "向下V箭頭"}]
        ),
        _detection_from_bbox(
            (300, 0, 20, 20), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "Chrome"}]
        ),
    ]
    anchor_indices, nearby_indices = _prefilter_detections_by_similarity(
        detections,
        "輸入欄",
        ["「排序」文字", "「向下V箭頭」圖示"],
    )
    anchor_labels = [detections[i].class_name for i in anchor_indices]
    nearby_labels = [detections[i].class_name for i in nearby_indices]
    assert anchor_labels == ["input"]
    assert nearby_labels == ["text", "element"]
    assert detections[nearby_indices[0]].text == "排序"
    assert (detections[nearby_indices[1]].icons or [{}])[0].get("chinese_id") == "向下V箭頭"

    anchor_scroll, nearby_scroll = _prefilter_detections_by_similarity(
        detections,
        "滾動條",
        [],
    )
    assert [detections[i].class_name for i in anchor_scroll] == ["scrollbar"]
    assert nearby_scroll == []


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


def test_assign_exclusive_neighbors_to_closest_anchor() -> None:
    anchors = [
        _detection_from_bbox((1000, 160, 60, 20), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((54, 302, 30, 14), YOLO_CLASS_TEXT, text="文件"),
    ]
    nearby = [
        _detection_from_bbox((20, 300, 16, 16), PICKER_CLASS_UNKNOWN, text="目"),
        _detection_from_bbox((55, 334, 28, 14), YOLO_CLASS_TEXT, text="圖片"),
    ]
    assigned = _assign_exclusive_neighbors_to_anchors(anchors, nearby)
    assert assigned[0] == []
    assert {d.text for d in assigned[1]} == {"目", "圖片"}


def test_prefilter_anchors_by_nearby_keeps_full_coverage() -> None:
    """Reproduce 1_1.log: only the 「文件」near 「目」/「圖片」covers both landmarks."""
    wrong_far = _detection_from_bbox((1057, 182, 31, 15), YOLO_CLASS_TEXT, text="文件")
    correct = _detection_from_bbox((54, 302, 30, 14), YOLO_CLASS_TEXT, text="文件")
    other = _detection_from_bbox((1467, 380, 30, 14), YOLO_CLASS_TEXT, text="文件")
    partial = _detection_from_bbox((474, 634, 30, 14), YOLO_CLASS_TEXT, text="文件")
    anchors = [wrong_far, correct, other, partial]
    nearby_matches = [
        _detection_from_bbox((20, 300, 16, 16), PICKER_CLASS_UNKNOWN, text="目"),
        _detection_from_bbox((55, 334, 28, 14), YOLO_CLASS_TEXT, text="圖片"),
        # Second 「圖片」closer to partial; should not steal full coverage from correct.
        _detection_from_bbox((500, 620, 28, 14), YOLO_CLASS_TEXT, text="圖片"),
    ]
    kept = _prefilter_anchors_by_nearby(
        anchors, nearby_matches, ["目未知", "圖片文字"]
    )
    assert kept == [correct]


def test_prefilter_anchors_by_nearby_falls_back_when_unmatched() -> None:
    anchors = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((100, 0, 20, 20), YOLO_CLASS_TEXT, text="文件"),
    ]
    nearby_matches = [
        _detection_from_bbox((200, 200, 20, 20), YOLO_CLASS_TEXT, text="無關"),
    ]
    kept = _prefilter_anchors_by_nearby(
        anchors, nearby_matches, ["目未知", "圖片文字"]
    )
    assert kept == anchors


def test_prefilter_anchors_by_nearby_partial_when_no_full_cover() -> None:
    anchors = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((200, 0, 20, 20), YOLO_CLASS_TEXT, text="文件"),
    ]
    nearby_matches = [
        # Only one landmark present; closest to first anchor.
        _detection_from_bbox((10, 30, 20, 20), YOLO_CLASS_TEXT, text="圖片"),
    ]
    kept = _prefilter_anchors_by_nearby(
        anchors, nearby_matches, ["目未知", "圖片文字"]
    )
    assert kept == [anchors[0]]


def test_prefilter_anchors_by_nearby_noop_without_nearby() -> None:
    anchors = [
        _detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="文件"),
    ]
    assert _prefilter_anchors_by_nearby(anchors, [], ["圖片文字"]) == anchors
    assert _prefilter_anchors_by_nearby(anchors, anchors, []) == anchors


def test_prefilter_anchors_by_nearby_respects_side() -> None:
    """Only the anchor whose bbox places the landmark in the matching cell survives."""
    from src.common.nearby_side import NearbyHint, Side

    # Landmark center at (100, 50). Left-of-landmark anchor has bbox to the left.
    left_anchor = _detection_from_bbox((40, 40, 20, 20), YOLO_CLASS_ELEMENT, text="框")
    right_anchor = _detection_from_bbox((120, 40, 20, 20), YOLO_CLASS_ELEMENT, text="框")
    landmark = _detection_from_bbox((90, 40, 20, 20), YOLO_CLASS_TEXT, text="標籤")
    # left_anchor edges x1=40,x2=60 → landmark cx=100 is RIGHT of bbox → script side LEFT
    # right_anchor edges x1=120,x2=140 → landmark cx=100 is LEFT of bbox → script side RIGHT
    kept = _prefilter_anchors_by_nearby(
        [left_anchor, right_anchor],
        [landmark],
        [NearbyHint(label="「標籤」文字", side=Side.LEFT)],
    )
    assert kept == [left_anchor]


def test_prefilter_anchors_by_nearby_respects_inside_side() -> None:
    from src.common.nearby_side import NearbyHint, Side

    inside_anchor = _detection_from_bbox((50, 40, 20, 20), YOLO_CLASS_TEXT, text="搜尋")
    outside_anchor = _detection_from_bbox((250, 40, 20, 20), YOLO_CLASS_TEXT, text="搜尋")
    input_landmark = _detection_from_bbox(
        (10, 10, 120, 80), YOLO_CLASS_INPUT, text=""
    )
    kept = _prefilter_anchors_by_nearby(
        [outside_anchor, inside_anchor],
        [input_landmark],
        [NearbyHint(label="輸入欄", side=Side.INSIDE)],
    )
    assert kept == [inside_anchor]


def test_prefilter_anchors_by_nearby_relaxes_side_when_none_match() -> None:
    from src.common.nearby_side import NearbyHint, Side

    anchor = _detection_from_bbox((120, 40, 20, 20), YOLO_CLASS_ELEMENT, text="框")
    landmark = _detection_from_bbox((90, 40, 20, 20), YOLO_CLASS_TEXT, text="標籤")
    # Required LEFT but geometry is RIGHT → relax to label-only.
    kept = _prefilter_anchors_by_nearby(
        [anchor],
        [landmark],
        [NearbyHint(label="「標籤」文字", side=Side.LEFT)],
    )
    assert kept == [anchor]


@pytest.mark.asyncio
async def test_resolve_mouse_point_nearby_prefilter_skips_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When nearby uniquely identifies one anchor, skip the picker LLM."""
    import numpy as np

    from cua_mcp.select_mouse_target import resolve_mouse_point

    correct = _detection_from_bbox((54, 302, 30, 14), YOLO_CLASS_TEXT, text="文件")
    wrong = _detection_from_bbox((1057, 182, 31, 15), YOLO_CLASS_TEXT, text="文件")
    landmark_mu = _detection_from_bbox(
        (20, 300, 16, 16), PICKER_CLASS_UNKNOWN, text="目"
    )
    landmark_img = _detection_from_bbox((55, 334, 28, 14), YOLO_CLASS_TEXT, text="圖片")

    async def fake_parse(instruction: str):
        return "文件", 0, 0, []

    def fake_filter(detections, anchor, nearby):
        return [wrong, correct], [landmark_mu, landmark_img]

    async def fail_ollama(*_args, **_kwargs):
        raise AssertionError("picker LLM should be skipped after nearby prefilter")

    async def fail_describe(*_args, **_kwargs):
        raise AssertionError("similar_function_describe should not run on move_mouse")

    async def fail_repick(*_args, **_kwargs):
        raise AssertionError("similar_function_describe should not run on move_mouse")

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.parse_mouse_target_instruction",
        fake_parse,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.selected_eye_monitor_indices",
        lambda: [1],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.capture_monitor_to_file",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.cv2.imread",
        lambda *_args, **_kwargs: np.zeros((10, 10, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._collect_monitor_detections",
        lambda *_args, **_kwargs: [wrong, correct, landmark_mu, landmark_img],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._filter_mouse_candidates",
        fake_filter,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._select_center_with_ollama",
        fail_ollama,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._describe_ui_candidate_functions",
        fail_describe,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._select_center_with_functions",
        fail_repick,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._run_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type("P", (), {"yolo_ocr_dir": __import__("pathlib").Path(".")})()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )

    gx, gy, meta = await resolve_mouse_point(
        "文件文字",
        nearby_objects=["目未知", "圖片文字"],
    )
    assert (gx, gy) == (correct.cx, correct.cy)
    assert meta["selected_index"] == 0
    assert meta["target_text"] == "文件"
    assert "disambiguation" not in meta


@pytest.mark.asyncio
async def test_resolve_mouse_point_does_not_run_function_describe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """similar_function_describe belongs on move_mouse_visual, not move_mouse."""
    import numpy as np

    from cua_mcp.select_mouse_target import resolve_mouse_point

    outlook = _detection_from_bbox((2240, 20, 30, 16), YOLO_CLASS_TEXT, text="搜尋")
    taskbar = _detection_from_bbox((2550, 1040, 30, 16), YOLO_CLASS_TEXT, text="搜尋")
    other = _detection_from_bbox((100, 100, 30, 16), YOLO_CLASS_TEXT, text="關閉")

    async def fake_parse(instruction: str):
        return "搜尋欄位", 0, 0, []

    def fake_filter(detections, anchor, nearby):
        return [outlook, taskbar], []

    async def fake_ollama(anchor, candidates, image_paths, **_kwargs):
        assert candidates == [outlook, taskbar]
        return 1, "picked-taskbar"

    async def fail_describe(*_args, **_kwargs):
        raise AssertionError("describe should not run on move_mouse")

    async def fail_repick(*_args, **_kwargs):
        raise AssertionError("re-pick should not run on move_mouse")

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.parse_mouse_target_instruction",
        fake_parse,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.selected_eye_monitor_indices",
        lambda: [1],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.capture_monitor_to_file",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.cv2.imread",
        lambda *_args, **_kwargs: np.zeros((10, 10, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._collect_monitor_detections",
        lambda *_args, **_kwargs: [outlook, taskbar, other],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._filter_mouse_candidates",
        fake_filter,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._select_center_with_ollama",
        fake_ollama,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._describe_ui_candidate_functions",
        fail_describe,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._select_center_with_functions",
        fail_repick,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._run_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type(
                        "P", (), {"yolo_ocr_dir": __import__("pathlib").Path(".")}
                    )()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )

    gx, gy, meta = await resolve_mouse_point("搜尋欄位")
    assert (gx, gy) == (taskbar.cx, taskbar.cy)
    assert meta["selected_index"] == 1
    assert meta["selected_text"] == "picked-taskbar"
    assert "disambiguation" not in meta


@pytest.mark.asyncio
async def test_resolve_mouse_point_skips_describe_when_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unique label → no describe/re-pick LLM calls."""
    import numpy as np

    from cua_mcp.select_mouse_target import resolve_mouse_point

    only = _detection_from_bbox((100, 100, 30, 16), YOLO_CLASS_TEXT, text="唯一按鈕")
    other = _detection_from_bbox((200, 200, 30, 16), YOLO_CLASS_TEXT, text="關閉")

    async def fake_parse(instruction: str):
        return "唯一按鈕", 0, 0, []

    def fake_filter(detections, anchor, nearby):
        return [only], []

    async def fail_describe(*_args, **_kwargs):
        raise AssertionError("describe should not run for unique labels")

    async def fail_repick(*_args, **_kwargs):
        raise AssertionError("re-pick should not run for unique labels")

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.parse_mouse_target_instruction",
        fake_parse,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.selected_eye_monitor_indices",
        lambda: [1],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.capture_monitor_to_file",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.cv2.imread",
        lambda *_args, **_kwargs: np.zeros((10, 10, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._collect_monitor_detections",
        lambda *_args, **_kwargs: [only, other],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._filter_mouse_candidates",
        fake_filter,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._describe_ui_candidate_functions",
        fail_describe,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._select_center_with_functions",
        fail_repick,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._run_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type("P", (), {"yolo_ocr_dir": __import__("pathlib").Path(".")})()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )

    gx, gy, meta = await resolve_mouse_point("唯一按鈕")
    assert (gx, gy) == (only.cx, only.cy)
    assert "disambiguation" not in meta


def test_format_ui_candidates_relational_uses_neighbor_phrases() -> None:
    detections = [
        _detection_from_bbox((56, 280, 28, 14), YOLO_CLASS_TEXT, text="下載"),
        _detection_from_bbox((54, 302, 30, 14), YOLO_CLASS_TEXT, text="文件"),
        _detection_from_bbox((55, 334, 28, 14), YOLO_CLASS_TEXT, text="圖片"),
    ]
    text = _format_ui_candidates_relational(detections)
    lines = text.split("\n")
    assert lines[1].startswith("[index 1] 「文件」文字 center=(69,309)（")
    assert "上方" in lines[1] and "「下載」文字" in lines[1]
    assert "下方" in lines[1] and "「圖片」文字" in lines[1]
    assert " w=" not in text
    assert " h=" not in text


def test_format_ui_candidates_relational_neighbor_context_not_selectable() -> None:
    """Pick rows are anchors only; nearby landmarks appear only in neighbor clauses."""
    anchors = [
        _detection_from_bbox((54, 302, 30, 14), YOLO_CLASS_TEXT, text="自訂Office 範本"),
        _detection_from_bbox((200, 400, 30, 14), YOLO_CLASS_TEXT, text="自訂Office 範本"),
    ]
    nearby = [
        _detection_from_bbox((56, 280, 28, 14), YOLO_CLASS_TEXT, text="WindowsPowerShell"),
        _detection_from_bbox(
            (20, 300, 16, 16), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "資料夾"}]
        ),
    ]
    text = _format_ui_candidates_relational(anchors, neighbors=nearby)
    lines = text.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("[index 0] 「自訂Office 範本」文字 center=(69,309)（")
    assert "WindowsPowerShell" in lines[0]
    assert "資料夾" in lines[0]
    assert "WindowsPowerShell" not in lines[0].split("（")[0]
    assert "WindowsPowerShell" not in lines[1]
    assert "資料夾" not in lines[1]
    assert "[index 2]" not in text


def test_format_ui_candidates_relational_neighbor_exclusive_to_closest_anchor() -> None:
    """Each landmark neighbor is cited only by the closest matching anchor."""
    anchors = [
        # Far 「文字文件」 (type column for another file)
        _detection_from_bbox((1000, 160, 60, 20), YOLO_CLASS_TEXT, text="文字文件"),
        # Near 「文字文件」 next to WinRAR / Excel
        _detection_from_bbox((100, 200, 60, 20), YOLO_CLASS_TEXT, text="文字文件"),
    ]
    nearby = [
        _detection_from_bbox((80, 160, 80, 20), YOLO_CLASS_TEXT, text="WinRAR麼縮檔"),
        _detection_from_bbox((150, 240, 100, 20), YOLO_CLASS_TEXT, text="Microsoft Excel工作表"),
    ]
    text = _format_ui_candidates_relational(anchors, neighbors=nearby)
    lines = text.split("\n")
    assert len(lines) == 2
    assert "WinRAR麼縮檔" not in lines[0]
    assert "Microsoft Excel工作表" not in lines[0]
    assert "WinRAR麼縮檔" in lines[1]
    assert "Microsoft Excel工作表" in lines[1]


def test_format_ui_candidates_relational_single_candidate() -> None:
    detections = [_detection_from_bbox((0, 0, 20, 20), YOLO_CLASS_TEXT, text="文件")]
    text = _format_ui_candidates_relational(detections)
    assert text == "[index 0] 「文件」文字 center=(10,10)"


def test_format_ui_candidates_relational_icon_label() -> None:
    detections = [
        _detection_from_bbox((0, 0, 16, 16), YOLO_CLASS_ELEMENT, icons=[{"chinese_id": "下載"}]),
        _detection_from_bbox((40, 0, 30, 14), YOLO_CLASS_TEXT, text="文件"),
    ]
    text = _format_ui_candidates_relational(detections)
    assert "「下載」圖示 center=(8,8)" in text
    assert "「文件」文字 center=(55,7)" in text
    assert "右方" in text
    assert "左方" in text


def test_collect_monitor_detections_preserves_order_and_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from cua_mcp.select_mouse_target import _collect_monitor_detections
    from cua_mcp.select_ui_element import UiDetection

    calls: list[int] = []

    def fake_build(bgr, *, yolo_conf_threshold: float = 0.05):
        monitor_tag = int(bgr[0, 0, 0])
        calls.append(monitor_tag)
        return [
            UiDetection(
                bbox=(1, 2, 3, 4),
                cx=2,
                cy=4,
                class_id=YOLO_CLASS_TEXT,
                class_name="text",
                text=f"m{monitor_tag}",
                icons=None,
            )
        ]

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._build_candidates_from_bgr",
        fake_build,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.active_monitor_offset",
        lambda idx: (100 * idx, 10 * idx),
    )

    img1 = np.full((4, 4, 3), 1, dtype=np.uint8)
    img2 = np.full((4, 4, 3), 2, dtype=np.uint8)
    detections = _collect_monitor_detections(
        [(1, img1), (2, img2)],
        yolo_conf_threshold=0.05,
    )

    assert [d.text for d in detections] == ["m1", "m2"]
    assert detections[0].bbox == (101, 12, 3, 4)
    assert detections[1].bbox == (201, 22, 3, 4)
    assert set(calls) == {1, 2}