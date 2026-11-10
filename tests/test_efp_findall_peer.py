"""EFP injected as a FindAll-shaped peer for recording primary naming."""

from __future__ import annotations

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT, YOLO_CLASS_TEXT
from src.recorder.vision_context import (
    _efp_peer_detection_from_payload,
    select_click_primary_candidate,
)


def test_efp_peer_beats_bare_input_as_primary() -> None:
    input_det = _detection_from_bbox(
        (0, 0, 400, 900),
        YOLO_CLASS_INPUT,
        text=None,
        source="uia",
    )
    efp = _efp_peer_detection_from_payload(
        {
            "name": "下載 (已釘選)",
            "value": "下載 (已釘選)",
            "bbox": [100, 200, 80, 40],  # screen == local when offset 0
            "control_type": "list_item",
        },
        monitor_offset=(0, 0),
        local_x=120,
        local_y=210,
        existing=[input_det],
    )
    assert efp is not None
    assert efp.class_name == "text"
    # Shell status tail `` (已釘選)`` is stripped for stable landmark text.
    assert efp.text == "下載"
    assert efp.source == "uia"

    # Promote path uses candidate dicts via select_click_primary_candidate.
    candidates = [
        {
            "bbox": list(input_det.bbox),
            "center": [input_det.cx, input_det.cy],
            "class_id": input_det.class_id,
            "class_name": input_det.class_name,
            "text": input_det.text,
            "icons": None,
            "source": "uia",
        },
        {
            "bbox": list(efp.bbox),
            "center": [efp.cx, efp.cy],
            "class_id": efp.class_id,
            "class_name": efp.class_name,
            "text": efp.text,
            "icons": None,
            "source": "uia",
            "uia_control_type": "list_item",
        },
    ]
    primary = select_click_primary_candidate(candidates, 120, 210)
    assert primary is not None
    assert primary.get("text") == "下載"
    assert primary.get("class_name") == "text"


def test_efp_peer_skips_without_bbox() -> None:
    assert (
        _efp_peer_detection_from_payload(
            {"name": "下載 (已釘選)", "value": "下載 (已釘選)"},
            monitor_offset=(0, 0),
            local_x=10,
            local_y=10,
            existing=[],
        )
        is None
    )


def test_efp_peer_dedupes_matching_findall_label() -> None:
    existing = [
        _detection_from_bbox(
            (100, 200, 80, 40),
            YOLO_CLASS_TEXT,
            text="下載",  # same canonical label as EFP after shell-status strip
            source="uia",
            uia_control_type="list_item",
        )
    ]
    assert (
        _efp_peer_detection_from_payload(
            {
                "name": "下載 (已釘選)",
                "value": "下載 (已釘選)",
                "bbox": [100, 200, 80, 40],
                "control_type": "list_item",
            },
            monitor_offset=(0, 0),
            local_x=120,
            local_y=210,
            existing=existing,
        )
        is None
    )


def test_coextensive_text_input_prefers_named_input_hub() -> None:
    """Page-sized Text Name + equal Input → 「…」文字所在的輸入欄 primary."""
    from src.recorder.analyze import instruction_for_click
    from src.recorder.models import RecordedEvent

    bbox = [0, 0, 1903, 791]
    candidates = [
        {
            "bbox": bbox,
            "center": [951, 395],
            "class_id": YOLO_CLASS_TEXT,
            "class_name": "text",
            "text": "文件1",
            "icons": None,
            "source": "uia",
        },
        {
            "bbox": list(bbox),
            "center": [951, 395],
            "class_id": YOLO_CLASS_INPUT,
            "class_name": "input",
            "text": None,
            "icons": None,
            "source": "uia",
        },
    ]
    primary = select_click_primary_candidate(candidates, 960, 400)
    assert primary is not None
    assert primary.get("class_name") == "input"
    assert primary.get("text") == "文件1"

    vision = {"candidates": candidates, "local_cursor": [960, 400]}
    event = RecordedEvent(
        index=3,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(960, 400),
        button="left",
        screenshot_path="",
    )
    assert (
        instruction_for_click(event, vision)
        == "將滑鼠移到「文件1」文字所在的輸入欄"
    )


def test_small_text_leaf_inside_input_still_text_hub() -> None:
    """Explorer-style leaf Text inside a large Input stays 「…」文字."""
    candidates = [
        {
            "bbox": [0, 0, 400, 900],
            "center": [200, 450],
            "class_id": YOLO_CLASS_INPUT,
            "class_name": "input",
            "text": None,
            "icons": None,
            "source": "uia",
        },
        {
            "bbox": [100, 200, 80, 40],
            "center": [140, 220],
            "class_id": YOLO_CLASS_TEXT,
            "class_name": "text",
            "text": "下載",
            "icons": None,
            "source": "uia",
            "uia_control_type": "list_item",
        },
    ]
    primary = select_click_primary_candidate(candidates, 120, 210)
    assert primary is not None
    assert primary.get("class_name") == "text"
    assert primary.get("text") == "下載"


def test_uia_first_miss_reaches_yolo_path_gate() -> None:
    """Documented fallthrough: UIA-first miss leaves chosen None so YOLO runs.

    ``_find_mouse_point_attempt`` only skips YOLO when ``uia_first_chosen`` is
    set; a Name miss keeps ``None`` and enters the ``if uia_first_chosen is
    None:`` YOLO/OCR block (see select_mouse_target.py).
    """
    from cua_mcp.select_mouse_target import _try_uia_first_unique_select

    dets = [
        _detection_from_bbox(
            (0, 0, 400, 900),
            YOLO_CLASS_INPUT,
            text=None,
            source="uia",
        )
    ]
    chosen, reason = _try_uia_first_unique_select(
        dets,
        anchor="下載 (已釘選)",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        filter_by_control_type=False,
        recorded_anchor=True,
    )
    assert chosen is None
    assert reason == "no_anchor"
