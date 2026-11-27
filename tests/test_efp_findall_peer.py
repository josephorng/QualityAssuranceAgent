"""EFP injected as a FindAll-shaped peer for recording primary naming."""

from __future__ import annotations

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT, YOLO_CLASS_SCROLLBAR, YOLO_CLASS_TEXT
from src.recorder.vision_context import (
    _efp_peer_detection_from_payload,
    select_click_primary_candidate,
)


def test_recorded_event_focused_element_roundtrip() -> None:
    from src.recorder.models import RecordedEvent

    focused = {
        "name": "文件 (已釘選)",
        "value": "文件 (已釘選)",
        "bbox": [53, 294, 34, 32],
        "control_type": "tree_item",
        "automation_id": "Documents",
    }
    event = RecordedEvent(
        index=15,
        timestamp_utc="2026-11-18T00:00:00+00:00",
        kind="click",
        cursor_xy=(81, 309),
        focused_element=focused,
        focus_rect=(53, 294, 87, 326),
    )
    raw = event.to_dict()
    assert raw["focused_element"] == focused
    assert raw["focus_rect"] == [53, 294, 87, 326]
    restored = RecordedEvent.from_dict(raw)
    assert restored.focused_element == focused
    assert restored.focus_rect == (53, 294, 87, 326)


def test_event015_shape_focused_repairs_documents_over_propertreehost() -> None:
    """Regression: event_015 geometry must name 「文件」 not ProperTreeHost."""
    host = _detection_from_bbox(
        (2, 128, 333, 876),
        YOLO_CLASS_INPUT,
        text="控制主機|ProperTreeHost",
        source="uia",
    )
    timing: dict = {}
    peer = _efp_peer_detection_from_payload(
        {
            "name": "文件 (已釘選)",
            "value": "文件 (已釘選)",
            "bbox": [53, 166, 34, 32],
        },
        monitor_offset=(0, 0),
        local_x=81,
        local_y=309,
        existing=[host],
        focused_element={
            "name": "文件 (已釘選)",
            "value": "文件 (已釘選)",
            "bbox": [53, 294, 34, 32],
            "control_type": "tree_item",
        },
        timing_out=timing,
    )
    assert peer is not None
    assert timing.get("efp_peer_bbox_source") == "focused_element"
    candidates = [
        {
            "bbox": list(host.bbox),
            "center": [host.cx, host.cy],
            "class_id": host.class_id,
            "class_name": host.class_name,
            "text": host.text,
            "icons": None,
            "source": "uia",
        },
        {
            "bbox": list(peer.bbox),
            "center": [peer.cx, peer.cy],
            "class_id": peer.class_id,
            "class_name": peer.class_name,
            "text": peer.text,
            "icons": None,
            "source": "uia_efp",
            "uia_control_type": peer.uia_control_type,
        },
    ]
    primary = select_click_primary_candidate(candidates, 81, 309)
    assert primary is not None
    assert primary.get("text") == "文件"
    assert "ProperTreeHost" not in str(primary.get("text") or "")


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
    assert efp.source == "uia_efp"

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
            "source": "uia_efp",
            "uia_control_type": "list_item",
        },
    ]
    primary = select_click_primary_candidate(candidates, 120, 210)
    assert primary is not None
    assert primary.get("text") == "下載"
    assert primary.get("class_name") == "text"


def test_efp_peer_click_local_without_bbox() -> None:
    timing: dict = {}
    efp = _efp_peer_detection_from_payload(
        {"name": "下載 (已釘選)", "value": "下載 (已釘選)"},
        monitor_offset=(0, 0),
        local_x=10,
        local_y=10,
        existing=[],
        timing_out=timing,
    )
    assert efp is not None
    assert efp.class_name == "text"
    assert efp.text == "下載"
    assert timing.get("efp_peer_bbox_source") == "click_local"
    assert efp.bbox[0] <= 10 <= efp.bbox[0] + efp.bbox[2]
    assert efp.bbox[1] <= 10 <= efp.bbox[1] + efp.bbox[3]


def test_efp_peer_repairs_with_focused_element_when_bbox_misses_click() -> None:
    """Explorer nav: stale EFP bbox + post-settle focused bounds → named text peer."""
    host = _detection_from_bbox(
        (2, 128, 333, 876),
        YOLO_CLASS_INPUT,
        text="控制主機|ProperTreeHost",
        source="uia",
    )
    timing: dict = {}
    efp = _efp_peer_detection_from_payload(
        {
            "name": "文件 (已釘選)",
            "value": "文件 (已釘選)",
            "bbox": [53, 166, 34, 32],  # misses click y=309
            "control_type": None,
        },
        monitor_offset=(0, 0),
        local_x=81,
        local_y=309,
        existing=[host],
        focused_element={
            "name": "文件 (已釘選)",
            "value": "文件 (已釘選)",
            "bbox": [53, 294, 34, 32],
            "control_type": "tree_item",
            "automation_id": "",
        },
        timing_out=timing,
    )
    assert efp is not None
    assert efp.text == "文件"
    assert efp.class_name == "text"
    assert efp.bbox == (53, 294, 34, 32)
    assert efp.uia_control_type == "tree_item"
    assert timing.get("efp_peer_bbox_source") == "focused_element"

    candidates = [
        {
            "bbox": list(host.bbox),
            "center": [host.cx, host.cy],
            "class_id": host.class_id,
            "class_name": host.class_name,
            "text": host.text,
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
            "source": "uia_efp",
            "uia_control_type": "tree_item",
        },
    ]
    primary = select_click_primary_candidate(candidates, 81, 309)
    assert primary is not None
    assert primary.get("text") == "文件"
    assert primary.get("class_name") == "text"


def test_efp_peer_click_local_when_focused_disagrees() -> None:
    timing: dict = {}
    efp = _efp_peer_detection_from_payload(
        {
            "name": "文件 (已釘選)",
            "value": "文件 (已釘選)",
            "bbox": [53, 166, 34, 32],
        },
        monitor_offset=(0, 0),
        local_x=81,
        local_y=309,
        existing=[],
        focused_element={
            "name": "桌面",
            "value": "桌面",
            "bbox": [53, 294, 34, 32],
        },
        timing_out=timing,
    )
    assert efp is not None
    assert efp.text == "文件"
    assert timing.get("efp_peer_bbox_source") == "click_local"


def test_efp_peer_skips_weak_hub() -> None:
    assert (
        _efp_peer_detection_from_payload(
            {
                "name": "CoreInput",
                "value": "",
                "bbox": [100, 100, 50, 50],
            },
            monitor_offset=(0, 0),
            local_x=110,
            local_y=110,
            existing=[],
        )
        is None
    )


def test_efp_input_bad_bbox_without_focused_does_not_synthesize() -> None:
    assert (
        _efp_peer_detection_from_payload(
            {
                "name": "網址與搜尋列",
                "value": "",
                "bbox": [10, 10, 20, 20],  # misses click
                "control_type": "input",
            },
            monitor_offset=(0, 0),
            local_x=300,
            local_y=83,
            existing=[],
        )
        is None
    )


def test_efp_position_thumb_injects_scrollbar_peer() -> None:
    """WinForms thumb Name=Position (unmapped control_type) → scrollbar, not text."""
    efp = _efp_peer_detection_from_payload(
        {
            "name": "Position",
            "value": "",
            "bbox": [2289, 162, 23, 251],
        },
        monitor_offset=(0, 0),
        local_x=2296,
        local_y=258,
        existing=[],
    )
    assert efp is not None
    assert efp.class_name == "scrollbar"
    assert efp.class_id == YOLO_CLASS_SCROLLBAR
    assert efp.text is None
    assert efp.bbox == (2289, 162, 23, 251)
    assert efp.source == "uia_efp"


def test_efp_structural_thumb_injects_scrollbar_peer() -> None:
    """Explorer thumb + VerticalScrollBar parent → scrollbar, not 「位置」 text."""
    efp = _efp_peer_detection_from_payload(
        {
            "name": "位置",
            "value": "",
            "control_type": "thumb",
            "bbox": [1423, 145, 17, 695],
            "parents": [{"name": "垂直", "automation_id": "VerticalScrollBar"}],
        },
        monitor_offset=(0, 0),
        local_x=1430,
        local_y=284,
        existing=[],
    )
    assert efp is not None
    assert efp.class_name == "scrollbar"
    assert efp.text is None
    assert efp.source == "uia_efp"
    assert efp.bbox == (1423, 145, 17, 695)


def test_efp_position_skipped_when_findall_scrollbar_covers_press() -> None:
    existing = [
        _detection_from_bbox(
            (2289, 137, 23, 468),
            YOLO_CLASS_SCROLLBAR,
            text=None,
            source="uia",
        )
    ]
    assert (
        _efp_peer_detection_from_payload(
            {
                "name": "Position",
                "value": "",
                "bbox": [2289, 162, 23, 251],
            },
            monitor_offset=(0, 0),
            local_x=2296,
            local_y=258,
            existing=existing,
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


def test_efp_input_peer_injects_named_input_when_no_findall_edit() -> None:
    efp = _efp_peer_detection_from_payload(
        {
            "name": "網址與搜尋列",
            "value": "",
            "bbox": [209, 67, 1431, 33],
            "control_type": "input",
        },
        monitor_offset=(0, 0),
        local_x=300,
        local_y=83,
        existing=[],
    )
    assert efp is not None
    assert efp.class_name == "input"
    assert efp.text == "網址與搜尋列"
    assert efp.uia_control_type is None


def test_efp_input_names_overlapping_findall_edit_in_place() -> None:
    """Chrome omnibox: coextensive FindAll Edit gets EFP Name; no second peer."""
    from src.recorder.analyze import instruction_for_click
    from src.recorder.models import RecordedEvent
    from src.recorder.vision_context import (
        _promote_click_primary_detections,
        format_drag_candidate_anchor,
        _detection_to_dict,
    )

    edit = _detection_from_bbox(
        (209, 67, 1431, 33),
        YOLO_CLASS_INPUT,
        text=None,
        source="uia",
    )
    page = _detection_from_bbox(
        (0, 0, 1920, 1080),
        YOLO_CLASS_INPUT,
        text=None,
        source="uia",
    )
    existing = [page, edit]
    peer = _efp_peer_detection_from_payload(
        {
            "name": "網址與搜尋列",
            "value": "",
            "bbox": [209, 67, 1431, 33],
            "control_type": "input",
        },
        monitor_offset=(0, 0),
        local_x=300,
        local_y=83,
        existing=existing,
    )
    assert peer is None
    assert existing[1].class_name == "input"
    assert existing[1].text == "網址與搜尋列"
    # Tight Edit named; page-sized host left unlabeled.
    assert existing[0].text is None

    promoted = _promote_click_primary_detections(existing, 300, 83)
    assert promoted[0].class_name == "input"
    assert promoted[0].text == "網址與搜尋列"
    assert (
        format_drag_candidate_anchor(_detection_to_dict(promoted[0]))
        == "「網址與搜尋列」文字所在的輸入欄"
    )

    vision = {
        "candidates": [_detection_to_dict(d) for d in promoted],
        "local_cursor": [300, 83],
    }
    event = RecordedEvent(
        index=6,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(300, 83),
        button="left",
        screenshot_path="",
    )
    assert (
        instruction_for_click(event, vision)
        == "將滑鼠移到「網址與搜尋列」文字所在的輸入欄"
    )


def test_promote_detections_same_bbox_text_input_keeps_named_input() -> None:
    """Detection-path promote must not leave Text on top when boxes match."""
    from src.recorder.vision_context import (
        _promote_click_primary_detections,
        format_drag_candidate_anchor,
        _detection_to_dict,
    )

    bbox = (209, 67, 1431, 33)
    text = _detection_from_bbox(
        bbox,
        YOLO_CLASS_TEXT,
        text="網址與搜尋列",
        source="uia",
        uia_control_type="input",
    )
    inp = _detection_from_bbox(
        bbox,
        YOLO_CLASS_INPUT,
        text=None,
        source="uia",
    )
    # Nearest order: multi-char text first (content priority).
    out = _promote_click_primary_detections([text, inp], 300, 83)
    assert out[0].class_name == "input"
    assert out[0].text == "網址與搜尋列"
    assert (
        format_drag_candidate_anchor(_detection_to_dict(out[0]))
        == "「網址與搜尋列」文字所在的輸入欄"
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


def test_field_sized_efp_name_inside_edit_prefers_named_input_hub() -> None:
    """WinForms TextBox: EFP Name slightly smaller than FindAll Edit → named input.

    IoU is below the coextensive gate (~0.59) but the Name leaf is mostly
    inside a field-sized Edit, so the hub must be ``「…」文字所在的輸入欄``.
    """
    from src.recorder.analyze import instruction_for_click
    from src.recorder.models import RecordedEvent

    # Matches recorded password field geometry: Name 150x13, Edit 158x21.
    name_bbox = [903, 441, 150, 13]
    edit_bbox = [897, 437, 158, 21]
    candidates = [
        {
            "bbox": name_bbox,
            "center": [978, 447],
            "class_id": YOLO_CLASS_TEXT,
            "class_name": "text",
            "text": "密碼|Textbox|txt_Password",
            "icons": None,
            "source": "uia",
            "uia_control_type": "input",
        },
        {
            "bbox": edit_bbox,
            "center": [976, 447],
            "class_id": YOLO_CLASS_INPUT,
            "class_name": "input",
            "text": None,
            "icons": None,
            "source": "uia",
        },
        {
            "bbox": [839, 440, 29, 16],
            "center": [853, 448],
            "class_id": YOLO_CLASS_TEXT,
            "class_name": "text",
            "text": "密碼",
            "icons": None,
            "source": "uia",
            "uia_control_type": "text",
        },
    ]
    primary = select_click_primary_candidate(candidates, 924, 447)
    assert primary is not None
    assert primary.get("class_name") == "input"
    assert primary.get("text") == "密碼|Textbox|txt_Password"

    vision = {"candidates": candidates, "local_cursor": [924, 447]}
    event = RecordedEvent(
        index=9,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(924, 447),
        button="left",
        screenshot_path="",
    )
    assert (
        instruction_for_click(event, vision)
        == "將滑鼠移到「密碼|Textbox|txt_Password」文字所在的輸入欄"
    )


def test_legacy_efp_name_without_control_type_still_names_input() -> None:
    """Pre-token recordings: empty EFP control_type still promotes onto Edit."""
    candidates = [
        {
            "bbox": [903, 441, 150, 13],
            "center": [978, 447],
            "class_id": YOLO_CLASS_TEXT,
            "class_name": "text",
            "text": "帳號|Textbox|txt_ID",
            "icons": None,
            "source": "uia",
        },
        {
            "bbox": [897, 437, 158, 21],
            "center": [976, 447],
            "class_id": YOLO_CLASS_INPUT,
            "class_name": "input",
            "text": None,
            "icons": None,
            "source": "uia",
        },
    ]
    primary = select_click_primary_candidate(candidates, 924, 447)
    assert primary is not None
    assert primary.get("class_name") == "input"
    assert primary.get("text") == "帳號|Textbox|txt_ID"


def test_findall_text_control_inside_input_does_not_promote() -> None:
    """FindAll Text (uia_control_type=text) inside an Edit stays text hub."""
    candidates = [
        {
            "bbox": [897, 437, 158, 21],
            "center": [976, 447],
            "class_id": YOLO_CLASS_INPUT,
            "class_name": "input",
            "text": None,
            "icons": None,
            "source": "uia",
        },
        {
            "bbox": [910, 440, 40, 14],
            "center": [930, 447],
            "class_id": YOLO_CLASS_TEXT,
            "class_name": "text",
            "text": "密碼",
            "icons": None,
            "source": "uia",
            "uia_control_type": "text",
        },
    ]
    primary = select_click_primary_candidate(candidates, 930, 447)
    assert primary is not None
    assert primary.get("class_name") == "text"
    assert primary.get("text") == "密碼"


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
    # Without hub_source=uia_efp / recorded EFP identity, UIA-first is skipped.
    assert reason == "no_efp_identity"


def test_efp_data_item_before_gate_opens_uia_path_over_covering_scrollbar(
    monkeypatch,
) -> None:
    """EFP G8 alone opens UIA path; FindAll scrollbar is not the hub pool."""
    import numpy as np

    from src.recorder.vision_context import (
        _detect_recording_mouse_targets,
        recording_uia_skip_reason,
        select_click_primary_candidate,
        _detection_to_dict,
    )

    bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Chrome only: large scrollbar covering the cell (no DataItem FindAll texts).
    prefetched = (
        [],
        [(0, 300, 1920, 700)],
        [],
    )

    def fake_detect(*_a, **_k):
        return []

    monkeypatch.setattr(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        fake_detect,
    )
    efp = {
        "name": "G8",
        "value": "",
        "control_type": "data_item",
        "bbox": [466, 477, 72, 22],
    }
    detections, timing = _detect_recording_mouse_targets(
        bgr,
        local_x=500,
        local_y=488,
        uia_prefetched=prefetched,
        element_from_point=efp,
        efp_monitor_offset=(0, 0),
    )
    assert timing.get("vision_path") == "uia"
    assert timing.get("recording_uia_sufficient") is True
    assert timing.get("efp_peer_injected") is True
    assert any(
        d.text == "G8"
        and d.source == "uia_efp"
        and getattr(d, "uia_control_type", None) == "data_item"
        for d in detections
    )
    assert all(d.source != "uia" for d in detections)
    assert recording_uia_skip_reason(detections, local_x=500, local_y=488) is None
    primary = select_click_primary_candidate(
        [_detection_to_dict(d) for d in detections],
        500,
        488,
    )
    assert primary is not None
    assert primary.get("text") == "G8"
    assert primary.get("source") == "uia_efp"
    assert primary.get("uia_control_type") == "data_item"


def test_efp_peer_injects_when_findall_covers_click_disagrees() -> None:
    """Disagreeing FindAll still keeps EFP in the pool; Save wins primary."""
    save = _detection_from_bbox(
        (715, 614, 99, 28),
        YOLO_CLASS_TEXT,
        text="儲存(S)",
        source="uia",
        uia_control_type="button",
    )
    timing: dict = {}
    peer = _efp_peer_detection_from_payload(
        {
            "name": "K15",
            "value": "",
            "bbox": [754, 631, 72, 22],
            "control_type": "data_item",
        },
        monitor_offset=(0, 0),
        local_x=798,
        local_y=632,
        existing=[save],
        timing_out=timing,
    )
    assert peer is not None
    assert peer.text == "K15"
    assert peer.source == "uia_efp"
    assert peer.uia_control_type == "data_item"
    assert timing.get("efp_findall_label_disagree") is True
    primary = select_click_primary_candidate(
        [
            {
                "bbox": list(save.bbox),
                "center": [764, 628],
                "class_id": YOLO_CLASS_TEXT,
                "class_name": "text",
                "text": "儲存(S)",
                "source": "uia",
                "uia_control_type": "button",
            },
            {
                "bbox": list(peer.bbox),
                "center": [peer.cx, peer.cy],
                "class_id": peer.class_id,
                "class_name": peer.class_name,
                "text": peer.text,
                "source": "uia_efp",
                "uia_control_type": "data_item",
            },
        ],
        798,
        632,
    )
    assert primary is not None
    assert primary.get("text") == "儲存(S)"


def test_efp_event009_shape_efp_hub_ignores_findall_save(
    monkeypatch,
) -> None:
    """EFP-only hub: K15 is primary; FindAll 儲存(S) is not in the UIA pool."""
    import numpy as np

    from src.recorder.vision_context import (
        _detect_recording_mouse_targets,
        select_click_primary_candidate,
        _detection_to_dict,
        _promote_click_primary_detections,
        _nearest_candidates,
    )

    bgr = np.zeros((667, 941, 3), dtype=np.uint8)
    prefetched = (
        [],
        [],
        [
            ((715, 614, 99, 28), "儲存(S)", "button"),
            ((826, 614, 88, 28), "取消", "button"),
        ],
    )
    yolo_save = _detection_from_bbox(
        (715, 614, 99, 28),
        YOLO_CLASS_TEXT,
        text="儲存(S)",
        source="ocr",
    )

    def fake_detect(*_a, **_k):
        return [yolo_save]

    monkeypatch.setattr(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        fake_detect,
    )
    efp = {
        "name": "K15",
        "value": "",
        "control_type": "data_item",
        "bbox": [754, 631, 72, 22],
    }
    detections, timing = _detect_recording_mouse_targets(
        bgr,
        local_x=798,
        local_y=632,
        uia_prefetched=prefetched,
        element_from_point=efp,
        efp_monitor_offset=(0, 0),
    )
    assert timing.get("vision_path") == "uia"
    assert timing.get("efp_peer_injected") is True
    assert any(
        d.text == "K15" and d.source == "uia_efp" for d in detections
    )
    assert all(d.source != "uia" for d in detections)
    nearest = _nearest_candidates(detections, 798, 632)
    nearest = _promote_click_primary_detections(nearest, 798, 632)
    primary = select_click_primary_candidate(
        [_detection_to_dict(d) for d in nearest],
        798,
        632,
    )
    assert primary is not None
    assert primary.get("text") == "K15"
    assert primary.get("source") == "uia_efp"


def test_efp_data_item_beats_generic_list_row_hub(monkeypatch) -> None:
    """Grid cell EFP opens UIA path; FindAll row hubs are not in the pool."""
    import numpy as np

    from src.recorder.vision_context import (
        _detect_recording_mouse_targets,
        select_click_primary_candidate,
        _detection_to_dict,
        recording_uia_skip_reason,
    )

    bgr = np.zeros((920, 1918, 3), dtype=np.uint8)
    prefetched = (
        [],
        [],
        [
            ((406, 155, 1347, 24), "清單 row 3", "tree_item"),
            ((406, 132, 1347, 24), "清單 row 2", "tree_item"),
            ((406, 178, 1347, 24), "清單 row 4", "tree_item"),
        ],
    )

    def fake_detect(*_a, **_k):
        return []

    monkeypatch.setattr(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        fake_detect,
    )
    efp = {
        "name": "電腦名稱",
        "value": "lannie-Ubuntu26",
        "control_type": "data_item",
        "bbox": [446, 156, 130, 22],
    }
    detections, timing = _detect_recording_mouse_targets(
        bgr,
        local_x=512,
        local_y=168,
        uia_prefetched=prefetched,
        element_from_point=efp,
        efp_monitor_offset=(0, 0),
    )
    assert timing.get("vision_path") == "uia"
    assert timing.get("efp_peer_injected") is True
    assert all(d.source != "uia" for d in detections)
    assert recording_uia_skip_reason(detections, local_x=512, local_y=168) is None
    primary = select_click_primary_candidate(
        [_detection_to_dict(d) for d in detections],
        512,
        168,
    )
    assert primary is not None
    assert primary.get("text") == "電腦名稱|lannie-Ubuntu26"
    assert primary.get("source") == "uia_efp"
    assert primary.get("uia_control_type") == "data_item"


def test_merge_sparse_uia_detections_with_yolo_peers() -> None:
    from cua_mcp.select_mouse_target import UiDetection
    from src.recorder.vision_context import merge_sparse_uia_detections_with_yolo_peers

    def _det(
        text: str,
        *,
        source: str,
        xy: tuple[int, int] = (10, 10),
    ) -> UiDetection:
        cx, cy = xy
        return UiDetection(
            bbox=(cx - 5, cy - 5, 10, 10),
            cx=cx,
            cy=cy,
            class_id=YOLO_CLASS_TEXT,
            class_name="text",
            text=text,
            source=source,
        )

    uia = [
        _det("N19", source="uia_efp", xy=(100, 100)),
        _det("全選", source="uia", xy=(20, 20)),
    ]
    yolo = [
        _det("N19", source="ocr", xy=(100, 100)),  # duplicate label — skip
        _det("19", source="ocr", xy=(20, 100)),
        _det("18", source="ocr", xy=(20, 80)),
        _det("20", source="ocr", xy=(20, 120)),
    ]
    merged = merge_sparse_uia_detections_with_yolo_peers(
        uia, yolo, min_peer_labels=8
    )
    assert [d.text for d in merged[:2]] == ["N19", "全選"]
    assert {d.text for d in merged[2:]} == {"19", "18", "20"}

    rich_uia = [_det(f"L{i}", source="uia", xy=(i * 10, 0)) for i in range(8)]
    assert merge_sparse_uia_detections_with_yolo_peers(
        rich_uia, yolo, min_peer_labels=8
    ) == rich_uia
