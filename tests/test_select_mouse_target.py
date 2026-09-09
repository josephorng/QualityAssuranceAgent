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
        return "「資料夾」圖示", 0, 0, ["「Chrome」圖示"], None, 0, None

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
        "cua_mcp.select_mouse_target.imread_bgr",
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

    from src.common.io_utils import imread_bgr, imwrite_bgr

    src = tmp_path / "cap_mon2.png"
    blank = np.zeros((120, 200, 3), dtype=np.uint8)
    assert imwrite_bgr(src, blank)

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
    img = imread_bgr(annotated)
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


def test_resolve_ocr_class_id_empty_element_becomes_unknown() -> None:
    from cua_mcp.select_mouse_target import _resolve_ocr_class_id
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "") == PICKER_CLASS_UNKNOWN


def test_resolve_ocr_class_id_keeps_element_for_known_pua() -> None:
    from cua_mcp.select_mouse_target import _resolve_ocr_class_id

    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "\ue002") == YOLO_CLASS_ELEMENT
    assert _resolve_ocr_class_id(YOLO_CLASS_TEXT, "OK") == YOLO_CLASS_TEXT


def test_resolve_ocr_class_id_unknown_pua_only_becomes_unknown() -> None:
    from cua_mcp.select_mouse_target import _resolve_ocr_class_id
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    # Unmapped PUA, or the mapped ``unknown_icon`` entry (U+E01A).
    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "\uf000") == PICKER_CLASS_UNKNOWN
    assert _resolve_ocr_class_id(YOLO_CLASS_ELEMENT, "\ue01a") == PICKER_CLASS_UNKNOWN
    assert _resolve_ocr_class_id(YOLO_CLASS_TEXT, "\ue01a") == PICKER_CLASS_UNKNOWN


def test_expand_bbox_avoiding_text_clamps_side_near_text() -> None:
    from cua_mcp.geometry import boxes_overlap
    from cua_mcp.select_mouse_target import _expand_bbox_avoiding_text

    # Icon left of text with a 1px gap; +2 expand would overlap without clamp.
    icon = (100, 100, 10, 10)  # 100..110
    text = (111, 100, 40, 10)  # 111..151
    out = _expand_bbox_avoiding_text(icon, [text], img_w=200, img_h=200, extra_margin=2)
    assert out == (98, 98, 13, 14)
    assert not boxes_overlap(out, text)


def test_expand_bbox_side_margins_left_right() -> None:
    from cua_mcp.select_mouse_target import _expand_bbox_avoiding_text

    icon = (100, 100, 10, 10)
    out = _expand_bbox_avoiding_text(
        icon, [], img_w=200, img_h=200, side_margins=(4, 0, 4, 0)
    )
    assert out == (96, 100, 18, 10)


def test_expand_bbox_avoiding_text_keeps_original_when_text_already_overlaps() -> None:
    from cua_mcp.select_mouse_target import _expand_bbox_avoiding_text

    icon = (100, 100, 13, 12)  # 100..113
    text = (110, 100, 50, 12)  # overlaps original
    out = _expand_bbox_avoiding_text(icon, [text], img_w=200, img_h=200, extra_margin=2)
    x, y, w, h = out
    assert x == 98
    assert y == 98
    assert x + w >= 113  # never shrinks below original right edge
    assert h >= 12


def test_is_empty_unknown_detection_requires_blank_ocr() -> None:
    from cua_mcp.select_mouse_target import _is_empty_unknown_detection

    blank = _detection_from_bbox((0, 0, 10, 10), PICKER_CLASS_UNKNOWN)
    with_text = _detection_from_bbox((0, 0, 10, 10), PICKER_CLASS_UNKNOWN, text="搜")
    with_icons = _detection_from_bbox(
        (0, 0, 10, 10),
        PICKER_CLASS_UNKNOWN,
        icons=[{"chinese_id": "星號、我的最愛"}],
    )
    element = _detection_from_bbox((0, 0, 10, 10), YOLO_CLASS_ELEMENT)

    assert _is_empty_unknown_detection(blank) is True
    assert _is_empty_unknown_detection(with_text) is False
    assert _is_empty_unknown_detection(with_icons) is False
    assert _is_empty_unknown_detection(element) is False


def test_single_known_icon_pua_rejects_multi_and_unknown() -> None:
    from cua_mcp.select_mouse_target import _single_known_icon_pua

    assert _single_known_icon_pua("\ue002") == "\ue002"
    assert _single_known_icon_pua("") is None
    assert _single_known_icon_pua("搜") is None
    assert _single_known_icon_pua("\ue01a") is None  # unknown_icon
    assert _single_known_icon_pua("\ue002\ue075") is None


def test_retry_empty_unknown_icon_ocr_upgrades_known_pua(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp.select_mouse_target import (
        _UNKNOWN_ICON_RETRY_SIDE_MARGINS,
        _retry_empty_unknown_icon_ocr,
    )

    unknown = _detection_from_bbox((100, 100, 12, 12), PICKER_CLASS_UNKNOWN)
    # Close enough that +2 right expand would overlap without clamp.
    text = _detection_from_bbox(
        (113, 100, 40, 12), YOLO_CLASS_TEXT, text="類別名稱"
    )
    captured: dict[str, object] = {}

    def fake_ocr(bgr, boxes, *, mode="text", margin=2, **_kwargs):
        captured["boxes"] = list(boxes)
        captured["mode"] = mode
        captured["margin"] = margin
        return [["\ue002"] for _ in boxes]

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._ocr_boxes_on_bgr",
        fake_ocr,
    )

    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    out = _retry_empty_unknown_icon_ocr(bgr, [unknown, text])

    assert captured["mode"] == "icon"
    assert captured["margin"] == 0
    assert len(captured["boxes"]) == len(_UNKNOWN_ICON_RETRY_SIDE_MARGINS)
    # First variant: +2 all; right side clamped before text at x=113.
    assert captured["boxes"][0] == (98, 98, 15, 16)
    assert len(out) == 2
    assert out[0].class_id == YOLO_CLASS_ELEMENT
    assert out[0].text == "\ue002"
    assert out[0].bbox == (100, 100, 12, 12)  # stored bbox unchanged
    assert out[0].icons
    assert out[1].class_id == YOLO_CLASS_TEXT


def test_retry_empty_unknown_uses_later_variant_when_first_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp.select_mouse_target import (
        _UNKNOWN_ICON_RETRY_SIDE_MARGINS,
        _retry_empty_unknown_icon_ocr,
    )

    unknown = _detection_from_bbox((100, 100, 12, 12), PICKER_CLASS_UNKNOWN)
    # Variant 0 (+2 all) empty; variant 1 (+4 L+R) yields known PUA.
    responses = [[], ["\ue075"], [], [], []]

    def fake_ocr(_bgr, boxes, *, mode="text", margin=2, **_kwargs):
        assert len(boxes) == len(_UNKNOWN_ICON_RETRY_SIDE_MARGINS)
        return responses

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._ocr_boxes_on_bgr",
        fake_ocr,
    )

    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    out = _retry_empty_unknown_icon_ocr(bgr, [unknown])

    assert len(out) == 1
    assert out[0].class_id == YOLO_CLASS_ELEMENT
    assert out[0].text == "\ue075"
    assert out[0].bbox == (100, 100, 12, 12)


def test_retry_empty_unknown_rejects_multi_pua_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp.select_mouse_target import _retry_empty_unknown_icon_ocr

    unknown = _detection_from_bbox((10, 10, 12, 12), PICKER_CLASS_UNKNOWN)

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._ocr_boxes_on_bgr",
        lambda *_a, **_k: [["\ue002\ue075"] for _ in range(5)],
    )

    bgr = np.zeros((80, 80, 3), dtype=np.uint8)
    out = _retry_empty_unknown_icon_ocr(bgr, [unknown])

    assert len(out) == 1
    assert out[0].class_id == PICKER_CLASS_UNKNOWN
    assert out[0].text is None


def test_retry_empty_unknown_skips_unknown_with_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp.select_mouse_target import _retry_empty_unknown_icon_ocr

    called = {"n": 0}

    def fake_ocr(*_args, **_kwargs):
        called["n"] += 1
        return []

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._ocr_boxes_on_bgr",
        fake_ocr,
    )

    unknown_with_text = _detection_from_bbox(
        (0, 0, 10, 10), PICKER_CLASS_UNKNOWN, text="搜"
    )
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    out = _retry_empty_unknown_icon_ocr(bgr, [unknown_with_text])

    assert called["n"] == 0
    assert out[0].text == "搜"
    assert out[0].class_id == PICKER_CLASS_UNKNOWN


def test_retry_empty_unknown_leaves_plain_text_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp.select_mouse_target import _retry_empty_unknown_icon_ocr

    unknown = _detection_from_bbox((10, 10, 12, 12), PICKER_CLASS_UNKNOWN)

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._ocr_boxes_on_bgr",
        lambda *_a, **_k: [["搜"] for _ in range(5)],
    )

    bgr = np.zeros((80, 80, 3), dtype=np.uint8)
    out = _retry_empty_unknown_icon_ocr(bgr, [unknown])

    assert len(out) == 1
    assert out[0].class_id == PICKER_CLASS_UNKNOWN
    assert out[0].text is None


def test_ocr_boxes_on_bgr_respects_custom_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from cua_mcp.read_screen_text.ocr_image import _ocr_boxes_on_bgr

    bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    captured: dict[str, object] = {}

    class _FakePredictor:
        def predict_images(self, *_args, **_kwargs):
            return []

    def fake_batched(crops, *_args, **_kwargs):
        captured["crop_shapes"] = [c.shape[:2] for c in crops]
        return [[] for _ in crops]

    monkeypatch.setattr(
        "cua_mcp.read_screen_text.ocr_image._get_ocr_predictor",
        lambda *_a, **_k: _FakePredictor(),
    )
    monkeypatch.setattr(
        "cua_mcp.read_screen_text.ocr_image._ocr_crops_batched",
        fake_batched,
    )

    _ocr_boxes_on_bgr(bgr, [(10, 10, 8, 8)], margin=0)
    assert captured["crop_shapes"] == [(8, 8)]

    _ocr_boxes_on_bgr(bgr, [(10, 10, 8, 8)], margin=2)
    assert captured["crop_shapes"] == [(12, 12)]


def test_split_multi_icon_element_two_pua_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from cua_mcp.read_screen_text.constrained_decode import CharSpan
    from cua_mcp.select_mouse_target import _split_multi_icon_element_detection

    pua_a, pua_b = "\ue000", "\ue001"
    text = pua_a + pua_b
    bbox = (100, 50, 60, 20)
    spans = [
        CharSpan(char=pua_a, t_start=0, t_end=0, x_start=0.0, x_end=48.0),
        CharSpan(char=pua_b, t_start=1, t_end=1, x_start=48.0, x_end=96.0),
    ]
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.ocr_box_with_spans",
        lambda *_args, **_kwargs: (text, spans),
    )
    bgr = np.zeros((200, 400, 3), dtype=np.uint8)
    split = _split_multi_icon_element_detection(bgr, bbox, text)
    assert split is not None
    assert len(split) == 2
    assert split[0].text == pua_a
    assert split[1].text == pua_b
    assert split[0].icons and len(split[0].icons) == 1
    assert split[1].icons and len(split[1].icons) == 1
    assert split[0].icons[0]["chinese_id"] == "資料夾"
    assert split[1].icons[0]["chinese_id"] == "檔案"
    assert split[0].bbox[0] < split[1].bbox[0]


def test_split_multi_icon_element_span_ocr_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from cua_mcp.select_mouse_target import _split_multi_icon_element_detection

    text = "\ue000\ue001"
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.ocr_box_with_spans",
        lambda *_args, **_kwargs: ("", []),
    )
    bgr = np.zeros((40, 80, 3), dtype=np.uint8)
    assert _split_multi_icon_element_detection(bgr, (0, 0, 40, 20), text) is None


def test_split_multi_icon_element_single_icon_and_non_pua_skip() -> None:
    import numpy as np
    from cua_mcp.select_mouse_target import _split_multi_icon_element_detection

    bgr = np.zeros((40, 80, 3), dtype=np.uint8)
    assert _split_multi_icon_element_detection(bgr, (0, 0, 20, 20), "\ue000") is None
    assert _split_multi_icon_element_detection(bgr, (0, 0, 40, 20), "OK\ue000") is None
    assert _split_multi_icon_element_detection(bgr, (0, 0, 40, 20), "搜尋") is None


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


def test_dedupe_keeps_text_over_overlapping_unknown() -> None:
    from cua_mcp.select_mouse_target import _dedupe_overlapping_detections

    text = _detection_from_bbox((100, 100, 40, 16), YOLO_CLASS_TEXT, text="儲存")
    unknown = _detection_from_bbox((101, 100, 40, 16), PICKER_CLASS_UNKNOWN)
    kept = _dedupe_overlapping_detections([unknown, text])
    assert len(kept) == 1
    assert kept[0].class_id == YOLO_CLASS_TEXT
    assert kept[0].text == "儲存"


def test_dedupe_keeps_icons_over_overlapping_unknown() -> None:
    from cua_mcp.select_mouse_target import _dedupe_overlapping_detections

    element = _detection_from_bbox(
        (100, 100, 20, 20),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "下載"}],
    )
    unknown = _detection_from_bbox((100, 100, 18, 18), PICKER_CLASS_UNKNOWN)
    kept = _dedupe_overlapping_detections([unknown, element])
    assert len(kept) == 1
    assert kept[0].class_id == YOLO_CLASS_ELEMENT
    assert kept[0].icons == [{"chinese_id": "下載"}]


def test_dedupe_keeps_overlapping_element_icon_and_text_box() -> None:
    from cua_mcp.select_mouse_target import _dedupe_overlapping_detections

    element = _detection_from_bbox(
        (100, 100, 24, 24),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "下載"}],
    )
    text = _detection_from_bbox((101, 102, 40, 16), YOLO_CLASS_TEXT, text="下載")
    kept = _dedupe_overlapping_detections([element, text])
    assert len(kept) == 2
    classes = {d.class_id for d in kept}
    assert classes == {YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT}

def test_fit_vertical_scrollbar_extends_to_up_down_v_arrows() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    # Track-only YOLO box; arrow buttons sit just outside the ends.
    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 418, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (1684, 555, 11, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    # Combo-box chevrons in the same column must not win.
    combo = _detection_from_bbox(
        (1679, 396, 16, 15),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up, down, combo])
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[0] == 1680
    assert fitted.bbox[2] == 20
    assert fitted.bbox[1] == 418
    assert fitted.bbox[1] + fitted.bbox[3] == 565
    assert fitted.cy == 418 + (565 - 418) // 2


def test_fit_vertical_scrollbar_shrinks_to_triangle_ends() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    scrollbar = _detection_from_bbox((100, 50, 16, 400), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (102, 100, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上三角"}],
    )
    down = _detection_from_bbox(
        (102, 300, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下三角"}],
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up, down])
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox == (100, 100, 16, 212)


def test_fit_horizontal_scrollbar_extends_to_left_right_arrows() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    scrollbar = _detection_from_bbox((220, 500, 100, 16), YOLO_CLASS_SCROLLBAR)
    left = _detection_from_bbox(
        (200, 502, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向左V箭頭"}],
    )
    right = _detection_from_bbox(
        (330, 502, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向右三角"}],
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, left, right])
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[1] == 500
    assert fitted.bbox[3] == 16
    assert fitted.bbox[0] == 200
    assert fitted.bbox[0] + fitted.bbox[2] == 342
    by_bbox = {d.bbox: d for d in out}
    assert by_bbox[left.bbox].icons[0]["chinese_id"] == "向左滾動箭頭"
    assert by_bbox[right.bbox].icons[0]["chinese_id"] == "向右滾動箭頭"


def test_fit_scrollbar_skips_when_end_arrows_missing() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 418, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up])
    assert out[0].bbox == scrollbar.bbox


def test_fit_scrollbar_skips_when_fitted_bbox_overlaps_text() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    # Short YOLO track; far arrows would stretch through a label (event_005-style).
    scrollbar = _detection_from_bbox((177, 400, 29, 80), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (181, 46, 21, 23),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (185, 864, 14, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    text = _detection_from_bbox(
        (158, 420, 71, 11),
        YOLO_CLASS_TEXT,
        text="取消自動校時",
    )
    logs: list[str] = []
    out = fit_scrollbar_bboxes_to_arrow_controls(
        [scrollbar, up, down, text],
        log_info=logs.append,
    )
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox == scrollbar.bbox
    by_bbox = {d.bbox: d for d in out}
    assert by_bbox[up.bbox].icons[0]["chinese_id"] == "向上V箭頭"
    assert by_bbox[down.bbox].icons[0]["chinese_id"] == "向下V箭頭"
    assert any("skipped_overlap=1" in line for line in logs)


def test_fit_scrollbar_skips_when_fitted_bbox_overlaps_input() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    scrollbar = _detection_from_bbox((177, 400, 29, 80), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (181, 46, 21, 23),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上三角"}],
    )
    down = _detection_from_bbox(
        (185, 864, 14, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下三角"}],
    )
    field = _detection_from_bbox((160, 500, 80, 24), YOLO_CLASS_INPUT)
    logs: list[str] = []
    out = fit_scrollbar_bboxes_to_arrow_controls(
        [scrollbar, up, down, field],
        log_info=logs.append,
    )
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox == scrollbar.bbox
    assert any("skipped_overlap=1" in line for line in logs)


def test_fit_scrollbar_allows_overlap_with_other_scrollbar() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    # event_003-style: vertical fit clips a horizontal bar at the corner.
    vertical = _detection_from_bbox((181, 132, 26, 738), YOLO_CLASS_SCROLLBAR)
    horizontal = _detection_from_bbox((205, 853, 1527, 22), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (185, 49, 15, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上三角"}],
    )
    down = _detection_from_bbox(
        (185, 863, 14, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下三角"}],
    )
    out = fit_scrollbar_bboxes_to_arrow_controls(
        [vertical, horizontal, up, down]
    )
    fitted = next(d for d in out if d.bbox[0] == 181)
    assert fitted.bbox == (181, 49, 26, 827)
    assert next(d for d in out if d.bbox == horizontal.bbox).bbox == horizontal.bbox
    by_bbox = {d.bbox: d for d in out}
    assert by_bbox[up.bbox].icons[0]["chinese_id"] == "向上滾動箭頭"
    assert by_bbox[down.bbox].icons[0]["chinese_id"] == "向下滾動箭頭"


def test_fit_scrollbar_still_extends_when_text_does_not_overlap() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 418, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (1684, 555, 11, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    # Label sits clear of the track column.
    text = _detection_from_bbox(
        (1500, 480, 80, 14),
        YOLO_CLASS_TEXT,
        text="側邊文字",
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up, down, text])
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[1] == 418
    assert fitted.bbox[1] + fitted.bbox[3] == 565
    by_bbox = {d.bbox: d for d in out}
    assert by_bbox[up.bbox].icons[0]["chinese_id"] == "向上滾動箭頭"
    assert by_bbox[down.bbox].icons[0]["chinese_id"] == "向下滾動箭頭"


def test_fit_vertical_scrollbar_accepts_unknown_icon_as_end() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 418, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    # Bottom end is an unrecognized icon (common when icon_map misses a glyph).
    unknown_down = _detection_from_bbox(
        (1684, 555, 11, 10),
        PICKER_CLASS_UNKNOWN,
        text="\uf000",
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up, unknown_down])
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[1] == 418
    assert fitted.bbox[1] + fitted.bbox[3] == 565
    relabeled = next(
        d
        for d in out
        if d.bbox == unknown_down.bbox
    )
    assert relabeled.class_name == "element"
    assert relabeled.class_id == YOLO_CLASS_ELEMENT
    assert (relabeled.icons or [])[0]["chinese_id"] == "向下滾動箭頭"
    assert (relabeled.icons or [])[0]["pua"] == "\uf000"
    up_out = next(d for d in out if d.bbox == up.bbox)
    assert (up_out.icons or [])[0]["chinese_id"] == "向上滾動箭頭"


def test_fit_vertical_scrollbar_reclassifies_unknown_top_and_bottom() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    unknown_up = _detection_from_bbox(
        (1683, 418, 12, 13),
        PICKER_CLASS_UNKNOWN,
        text="\uf001",
    )
    unknown_down = _detection_from_bbox(
        (1684, 555, 11, 10),
        PICKER_CLASS_UNKNOWN,
        text="\uf002",
    )
    out = fit_scrollbar_bboxes_to_arrow_controls(
        [scrollbar, unknown_up, unknown_down]
    )
    by_bbox = {d.bbox: d for d in out}
    assert by_bbox[unknown_up.bbox].icons[0]["chinese_id"] == "向上滾動箭頭"
    assert by_bbox[unknown_down.bbox].icons[0]["chinese_id"] == "向下滾動箭頭"
    assert by_bbox[unknown_up.bbox].class_name == "element"
    assert by_bbox[unknown_down.bbox].class_name == "element"


def test_fit_vertical_scrollbar_extends_far_without_gap_limit() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    # Arrows sit well beyond the old 48px gap; still extend to them.
    scrollbar = _detection_from_bbox((1680, 500, 20, 80), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 300, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (1684, 700, 11, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up, down])
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[1] == 300
    assert fitted.bbox[1] + fitted.bbox[3] == 710


def test_fit_vertical_scrollbar_prefers_unknown_over_wrong_direction() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 418, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    # Closer to the bottom end than the unknown, but wrong direction for top
    # priority: bottom should pick unknown before this down-arrow-as-fallback.
    wrong_near_bottom = _detection_from_bbox(
        (1684, 530, 11, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    unknown_down = _detection_from_bbox(
        (1684, 555, 11, 10),
        PICKER_CLASS_UNKNOWN,
        text="\uf000",
    )
    out = fit_scrollbar_bboxes_to_arrow_controls(
        [scrollbar, up, wrong_near_bottom, unknown_down]
    )
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[1] == 418
    assert fitted.bbox[1] + fitted.bbox[3] == 565
    relabeled = next(d for d in out if d.bbox == unknown_down.bbox)
    assert (relabeled.icons or [])[0]["chinese_id"] == "向下滾動箭頭"


def test_fit_vertical_scrollbar_picks_closest_to_center_on_each_side() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    # Center at y=540. Farther preferred arrows should lose to nearer ones
    # on the same side of the center.
    scrollbar = _detection_from_bbox((1680, 500, 20, 80), YOLO_CLASS_SCROLLBAR)
    near_up = _detection_from_bbox(
        (1683, 480, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    far_up = _detection_from_bbox(
        (1683, 300, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    near_down = _detection_from_bbox(
        (1684, 600, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    far_down = _detection_from_bbox(
        (1684, 800, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    # Below-center up-arrow must not be used for the top end.
    up_on_bottom_side = _detection_from_bbox(
        (1683, 560, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    out = fit_scrollbar_bboxes_to_arrow_controls(
        [scrollbar, far_up, near_up, near_down, far_down, up_on_bottom_side]
    )
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[1] == 480
    assert fitted.bbox[1] + fitted.bbox[3] == 612


def test_create_vertical_scrollbar_from_v_arrow_pair() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    up = _detection_from_bbox(
        (1683, 100, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (1684, 400, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    out = create_scrollbars_from_arrow_pairs([up, down])
    scrollbars = [d for d in out if d.class_name == "scrollbar"]
    assert len(scrollbars) == 1
    assert scrollbars[0].bbox == (1683, 100, 13, 312)
    by_bbox = {d.bbox: d for d in out if d.class_name != "scrollbar"}
    assert by_bbox[up.bbox].icons[0]["chinese_id"] == "向上滾動箭頭"
    assert by_bbox[down.bbox].icons[0]["chinese_id"] == "向下滾動箭頭"


def test_create_horizontal_scrollbar_from_triangle_pair() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    left = _detection_from_bbox(
        (200, 502, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向左三角"}],
    )
    right = _detection_from_bbox(
        (500, 504, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向右三角"}],
    )
    out = create_scrollbars_from_arrow_pairs([left, right])
    scrollbars = [d for d in out if d.class_name == "scrollbar"]
    assert len(scrollbars) == 1
    assert scrollbars[0].bbox == (200, 502, 312, 14)
    by_bbox = {d.bbox: d for d in out if d.class_name != "scrollbar"}
    assert by_bbox[left.bbox].icons[0]["chinese_id"] == "向左滾動箭頭"
    assert by_bbox[right.bbox].icons[0]["chinese_id"] == "向右滾動箭頭"


def test_create_scrollbar_skips_when_yolo_scrollbar_overlaps() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    scrollbar = _detection_from_bbox((1680, 100, 20, 320), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 100, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (1684, 400, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    out = create_scrollbars_from_arrow_pairs([scrollbar, up, down])
    assert sum(1 for d in out if d.class_name == "scrollbar") == 1
    assert out[0].bbox == scrollbar.bbox
    assert (out[1].icons or [{}])[0].get("chinese_id") == "向上V箭頭"
    assert (out[2].icons or [{}])[0].get("chinese_id") == "向下V箭頭"


def test_create_scrollbar_skips_when_union_overlaps_text() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    up = _detection_from_bbox(
        (100, 50, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上三角"}],
    )
    down = _detection_from_bbox(
        (100, 300, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下三角"}],
    )
    text = _detection_from_bbox(
        (98, 150, 40, 20),
        YOLO_CLASS_TEXT,
        text="內容",
    )
    out = create_scrollbars_from_arrow_pairs([up, down, text])
    assert not any(d.class_name == "scrollbar" for d in out)


def test_create_scrollbar_skips_when_union_overlaps_input() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    up = _detection_from_bbox(
        (100, 50, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上三角"}],
    )
    down = _detection_from_bbox(
        (100, 300, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下三角"}],
    )
    field = _detection_from_bbox((98, 150, 40, 20), YOLO_CLASS_INPUT)
    out = create_scrollbars_from_arrow_pairs([up, down, field])
    assert not any(d.class_name == "scrollbar" for d in out)


def test_create_scrollbar_ignores_text_overlapping_end_arrow() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    up = _detection_from_bbox(
        (904, 450, 14, 16),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (904, 574, 12, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    # OCR misread on the up-arrow glyph — any overlap is ignored.
    ghost = _detection_from_bbox(
        (903, 454, 12, 9),
        YOLO_CLASS_TEXT,
        text="機",
    )
    out = create_scrollbars_from_arrow_pairs([up, down, ghost])
    scrollbars = [d for d in out if d.class_name == "scrollbar"]
    assert len(scrollbars) == 1
    assert scrollbars[0].bbox == (904, 450, 14, 134)
    by_bbox = {d.bbox: d for d in out}
    assert by_bbox[up.bbox].icons[0]["chinese_id"] == "向上滾動箭頭"
    assert by_bbox[down.bbox].icons[0]["chinese_id"] == "向下滾動箭頭"


def test_create_scrollbar_ignores_partial_text_overlap_on_end_arrow() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    up = _detection_from_bbox(
        (100, 50, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (100, 300, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    # Only partial overlap with the up arrow — still ignored.
    text = _detection_from_bbox(
        (106, 50, 12, 12),
        YOLO_CLASS_TEXT,
        text="半",
    )
    out = create_scrollbars_from_arrow_pairs([up, down, text])
    assert sum(1 for d in out if d.class_name == "scrollbar") == 1


def test_fit_scrollbar_ignores_text_overlapping_end_arrow() -> None:
    from cua_mcp.scrollbar_arrows import fit_scrollbar_bboxes_to_arrow_controls

    scrollbar = _detection_from_bbox((904, 480, 14, 80), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (904, 450, 14, 16),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (904, 574, 12, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    ghost = _detection_from_bbox(
        (903, 454, 12, 9),
        YOLO_CLASS_TEXT,
        text="機",
    )
    out = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up, down, ghost])
    fitted = next(d for d in out if d.class_name == "scrollbar")
    assert fitted.bbox[1] == 450
    assert fitted.bbox[1] + fitted.bbox[3] == 584


def test_create_scrollbar_skips_misaligned_pair() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    up = _detection_from_bbox(
        (100, 50, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    # Far to the right — not the same column.
    down = _detection_from_bbox(
        (300, 400, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    out = create_scrollbars_from_arrow_pairs([up, down])
    assert not any(d.class_name == "scrollbar" for d in out)


def test_create_scrollbar_skips_cross_family_pair() -> None:
    from cua_mcp.scrollbar_arrows import create_scrollbars_from_arrow_pairs

    up = _detection_from_bbox(
        (100, 50, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (100, 400, 12, 12),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下三角"}],
    )
    out = create_scrollbars_from_arrow_pairs([up, down])
    assert not any(d.class_name == "scrollbar" for d in out)


def test_create_scrollbar_does_not_steal_fitted_yolo_arrows() -> None:
    from cua_mcp.scrollbar_arrows import (
        create_scrollbars_from_arrow_pairs,
        fit_scrollbar_bboxes_to_arrow_controls,
    )

    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 418, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上V箭頭"}],
    )
    down = _detection_from_bbox(
        (1684, 555, 11, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    fitted = fit_scrollbar_bboxes_to_arrow_controls([scrollbar, up, down])
    out = create_scrollbars_from_arrow_pairs(fitted)
    assert sum(1 for d in out if d.class_name == "scrollbar") == 1
    fitted_sb = next(d for d in out if d.class_name == "scrollbar")
    assert fitted_sb.bbox[1] == 418
    assert fitted_sb.bbox[1] + fitted_sb.bbox[3] == 565


def test_drop_scrollbars_without_arrow_ends_removes_bare_yolo_bar() -> None:
    from cua_mcp.scrollbar_arrows import drop_scrollbars_without_arrow_ends

    bare = _detection_from_bbox((603, 852, 476, 20), YOLO_CLASS_SCROLLBAR)
    text = _detection_from_bbox((618, 855, 60, 12), YOLO_CLASS_TEXT, text="M1TTKT47A")
    out = drop_scrollbars_without_arrow_ends([bare, text])
    assert not any(d.class_name == "scrollbar" for d in out)
    assert out[0].text == "M1TTKT47A"


def test_drop_scrollbars_without_arrow_ends_keeps_v_arrow_pair() -> None:
    from cua_mcp.scrollbar_arrows import drop_scrollbars_without_arrow_ends

    scrollbar = _detection_from_bbox((1680, 418, 20, 147), YOLO_CLASS_SCROLLBAR)
    up = _detection_from_bbox(
        (1683, 418, 12, 13),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向上滾動箭頭"}],
    )
    down = _detection_from_bbox(
        (1684, 555, 11, 10),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下滾動箭頭"}],
    )
    out = drop_scrollbars_without_arrow_ends([scrollbar, up, down])
    assert sum(1 for d in out if d.class_name == "scrollbar") == 1


def test_drop_scrollbars_without_arrow_ends_rejects_unknown_only_ends() -> None:
    from cua_mcp.scrollbar_arrows import drop_scrollbars_without_arrow_ends
    from cua_mcp.yolo_onnx import PICKER_CLASS_UNKNOWN

    scrollbar = _detection_from_bbox((1680, 440, 20, 100), YOLO_CLASS_SCROLLBAR)
    unknown_up = _detection_from_bbox(
        (1683, 418, 12, 13),
        PICKER_CLASS_UNKNOWN,
        icons=[{"chinese_id": "未知圖示"}],
    )
    unknown_down = _detection_from_bbox(
        (1684, 555, 11, 10),
        PICKER_CLASS_UNKNOWN,
        icons=[{"chinese_id": "未知圖示"}],
    )
    out = drop_scrollbars_without_arrow_ends(
        [scrollbar, unknown_up, unknown_down]
    )
    assert not any(d.class_name == "scrollbar" for d in out)


def test_merge_overlapping_scrollbars_unions_high_iou_pair() -> None:
    from cua_mcp.scrollbar_arrows import merge_overlapping_scrollbars

    a = _detection_from_bbox((100, 50, 20, 200), YOLO_CLASS_SCROLLBAR)
    b = _detection_from_bbox((102, 80, 18, 200), YOLO_CLASS_SCROLLBAR)
    text = _detection_from_bbox((0, 0, 40, 20), YOLO_CLASS_TEXT, text="x")
    out = merge_overlapping_scrollbars([text, a, b])
    scrollbars = [d for d in out if d.class_name == "scrollbar"]
    assert len(scrollbars) == 1
    assert scrollbars[0].bbox == (100, 50, 20, 230)
    assert out[0].text == "x"


def test_merge_overlapping_scrollbars_skips_low_iou() -> None:
    from cua_mcp.scrollbar_arrows import merge_overlapping_scrollbars

    a = _detection_from_bbox((100, 0, 20, 100), YOLO_CLASS_SCROLLBAR)
    b = _detection_from_bbox((100, 200, 20, 100), YOLO_CLASS_SCROLLBAR)
    out = merge_overlapping_scrollbars([a, b])
    assert sum(1 for d in out if d.class_name == "scrollbar") == 2


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


def test_normalize_similarity_label_ocr_canonicalization() -> None:
    from cua_mcp.select_mouse_target import _normalize_similarity_label

    assert _normalize_similarity_label("Foo|") == "Foo"
    assert _normalize_similarity_label("|Foo|") == "Foo"
    assert _normalize_similarity_label(" |Foo| ") == "Foo"
    assert _normalize_similarity_label("a|b") == "a|b"
    assert _normalize_similarity_label("「More...」文字") == "More…"
    assert _normalize_similarity_label("More...") == "More…"
    assert _normalize_similarity_label("Save⋯") == "Save…"
    assert _normalize_similarity_label("Save‥") == "Save…"
    assert _normalize_similarity_label("【Ｅｄｇｅ】圖示") == "Edge"
    assert _normalize_similarity_label("A–B") == "A-B"
    assert _normalize_similarity_label("A—B") == "A-B"
    assert _normalize_similarity_label("A－B") == "A-B"
    assert _normalize_similarity_label("A─B") == "A-B"
    assert _normalize_similarity_label("完成。") == "完成."
    assert _normalize_similarity_label("甲、乙") == "甲,乙"
    assert _normalize_similarity_label("IDE") == "lDE"
    assert _normalize_similarity_label("file1") == "filel"
    assert _normalize_similarity_label("O0") == "00"
    assert _normalize_similarity_label("\u00a0Edge\u00a0") == "Edge"
    assert _normalize_similarity_label("　Edge　") == "Edge"
    assert _normalize_similarity_label("hello\u00a0world") == "hello world"


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


def test_label_similarity_ocr_canonicalization_folds() -> None:
    from cua_mcp.select_mouse_target import _label_similarity

    assert _label_similarity("Foo|", "Foo") == 1.0
    assert _label_similarity("Save...", "Save…") == 1.0
    assert _label_similarity("Save⋯", "Save…") == 1.0
    assert _label_similarity("Save‥", "Save…") == 1.0
    assert _label_similarity("IDE", "lDE") == 1.0
    assert _label_similarity("file1", "filel") == 1.0
    assert _label_similarity("O0", "00") == 1.0
    assert _label_similarity("A–B", "A-B") == 1.0
    assert _label_similarity("A—B", "A-B") == 1.0
    assert _label_similarity("A－B", "A-B") == 1.0
    assert _label_similarity("完成。", "完成.") == 1.0
    assert _label_similarity("【Ｅｄｇｅ】圖示", "Edge") == 1.0
    assert _label_similarity("hello\u00a0world", "hello world") == 1.0
    assert _label_similarity("「More...」文字", "More…") == 1.0


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


def test_admit_unknown_icon_peers_step8_style_picks_end_chevron() -> None:
    """OCR-missed End chevron is admitted; nearby sides drop the Start seed."""
    from cua_mcp.select_mouse_target import (
        _admit_unknown_icon_peers,
        _merge_anchor_detections,
        _prefilter_anchors_by_nearby,
    )

    # Geometry from 政策制定 step 8 / event_014.
    start_arrow = _detection_from_bbox(
        (892, 430, 13, 16),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    end_arrow_unknown = _detection_from_bbox(
        (1048, 430, 14, 15),
        PICKER_CLASS_UNKNOWN,
    )
    large_unknown = _detection_from_bbox(
        (300, 400, 80, 80),
        PICKER_CLASS_UNKNOWN,
    )
    text_unknown = _detection_from_bbox(
        (1050, 500, 14, 14),
        PICKER_CLASS_UNKNOWN,
        text="搜",
    )
    landmark_start_time = _detection_from_bbox(
        (812, 431, 28, 11), YOLO_CLASS_TEXT, text="07:00"
    )
    landmark_end_label = _detection_from_bbox(
        (926, 431, 30, 14), YOLO_CLASS_TEXT, text="終止"
    )
    landmark_error = _detection_from_bbox(
        (770, 463, 199, 16),
        YOLO_CLASS_TEXT,
        text="求起始時間必須小於終止時間!",
    )
    detections = [
        start_arrow,
        end_arrow_unknown,
        large_unknown,
        text_unknown,
        landmark_start_time,
        landmark_end_label,
        landmark_error,
    ]
    nearby_matches = [landmark_start_time, landmark_end_label, landmark_error]
    nearby_phrases = [
        "在「終止」文字的右邊",
        "在「07:00」文字的右邊",
        "在「求起始時間必須小於終止時間!」文字的右上方",
    ]

    admitted = _admit_unknown_icon_peers(
        detections,
        [start_arrow],
        nearby_matches,
        nearby_phrases,
        anchor="「向下V箭頭」圖示",
    )
    assert admitted == [end_arrow_unknown]

    merged = _merge_anchor_detections([start_arrow], admitted)
    kept = _prefilter_anchors_by_nearby(merged, nearby_matches, nearby_phrases)
    assert kept == [end_arrow_unknown]


def test_admit_unknown_icon_peers_requires_icon_target_and_directed_sides() -> None:
    from cua_mcp.select_mouse_target import _admit_unknown_icon_peers

    labeled = _detection_from_bbox(
        (100, 100, 16, 16),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    unknown = _detection_from_bbox((200, 100, 16, 16), PICKER_CLASS_UNKNOWN)
    landmark = _detection_from_bbox((150, 100, 30, 14), YOLO_CLASS_TEXT, text="終止")
    detections = [labeled, unknown, landmark]

    # No directed side → no admit.
    assert (
        _admit_unknown_icon_peers(
            detections,
            [labeled],
            [landmark],
            ["「終止」文字"],
            anchor="「向下V箭頭」圖示",
        )
        == []
    )
    # Text target (no 圖示 / no icon labels on seed) → no admit.
    text_seed = _detection_from_bbox((100, 100, 30, 14), YOLO_CLASS_TEXT, text="終止")
    assert (
        _admit_unknown_icon_peers(
            [text_seed, unknown, landmark],
            [text_seed],
            [landmark],
            ["在「終止」文字的右邊"],
            anchor="「終止」文字",
        )
        == []
    )
    # Wrong side for the unknown → no admit.
    assert (
        _admit_unknown_icon_peers(
            detections,
            [labeled],
            [landmark],
            ["在「終止」文字的左邊"],
            anchor="「向下V箭頭」圖示",
        )
        == []
    )


def test_admit_unknown_icon_peers_skips_size_incompatible() -> None:
    from cua_mcp.select_mouse_target import _admit_unknown_icon_peers

    labeled = _detection_from_bbox(
        (100, 100, 16, 16),
        YOLO_CLASS_ELEMENT,
        icons=[{"chinese_id": "向下V箭頭"}],
    )
    huge_unknown = _detection_from_bbox((200, 100, 60, 60), PICKER_CLASS_UNKNOWN)
    landmark = _detection_from_bbox((150, 100, 30, 14), YOLO_CLASS_TEXT, text="終止")
    assert (
        _admit_unknown_icon_peers(
            [labeled, huge_unknown, landmark],
            [labeled],
            [landmark],
            ["在「終止」文字的右邊"],
            anchor="「向下V箭頭」圖示",
        )
        == []
    )


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
        return "文件", 0, 0, [], None, 0, None

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
        "cua_mcp.select_mouse_target.imread_bgr",
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
        return "搜尋欄位", 0, 0, [], None, 0, None

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
        "cua_mcp.select_mouse_target.imread_bgr",
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
        return "唯一按鈕", 0, 0, [], None, 0, None

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
        "cua_mcp.select_mouse_target.imread_bgr",
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
        "cua_mcp.select_mouse_target._detect_mouse_targets_from_bgr",
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


@pytest.mark.asyncio
async def test_resolve_mouse_point_char_target_uses_span_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp.read_screen_text.constrained_decode import CharSpan
    from cua_mcp.select_mouse_target import resolve_mouse_point

    det = _detection_from_bbox((10, 20, 40, 10), YOLO_CLASS_TEXT, text="搜尋")

    async def fake_parse(instruction: str):
        assert "搜" in instruction
        return "「搜尋」文字", 0, 0, [], "搜", 0, None

    def fake_filter(detections, anchor, nearby):
        return [detections[0]], []

    def fake_ocr_with_spans(_bgr, _bbox, **_kwargs):
        return "搜尋", [
            CharSpan(char="搜", t_start=0, t_end=0, x_start=0.0, x_end=20.0),
            CharSpan(char="尋", t_start=1, t_end=1, x_start=20.0, x_end=40.0),
        ]

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
        "cua_mcp.select_mouse_target.imread_bgr",
        lambda *_args, **_kwargs: np.zeros((100, 100, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._collect_monitor_detections",
        lambda *_args, **_kwargs: [det],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._filter_mouse_candidates",
        fake_filter,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.ocr_box_with_spans",
        fake_ocr_with_spans,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.active_monitor_offset",
        lambda _idx: (0, 0),
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

    gx, gy, meta = await resolve_mouse_point("將滑鼠移到「搜尋」的「搜」字上")

    assert meta["char_target"] == "搜"
    assert meta["resolved_char_center"] == {"x": gx, "y": gy}
    assert (gx, gy) != (det.cx, det.cy)


@pytest.mark.asyncio
async def test_resolve_mouse_point_scrollbar_track_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp.scrollbar_arrows import point_from_scrollbar_percent
    from cua_mcp.select_mouse_target import resolve_mouse_point

    det = _detection_from_bbox((100, 0, 20, 100), YOLO_CLASS_SCROLLBAR)

    async def fake_parse(instruction: str):
        assert "60%" in instruction
        return "滾動條", 0, 0, [], None, 0, 60

    def fake_filter(detections, anchor, nearby):
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
        "cua_mcp.select_mouse_target.imread_bgr",
        lambda *_args, **_kwargs: np.zeros((200, 200, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._collect_monitor_detections",
        lambda *_args, **_kwargs: [det],
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
                    lambda: type("P", (), {"yolo_ocr_dir": Path(".")})()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )

    gx, gy, meta = await resolve_mouse_point("將滑鼠移到滾動條的60%處")
    expected = point_from_scrollbar_percent(det.bbox, 60)
    assert (gx, gy) == expected
    assert meta["track_percent"] == 60
    assert meta["relative_offset"] == {"dx": 0, "dy": 0}