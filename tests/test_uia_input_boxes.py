"""Unit tests for UIA input/scrollbar/text refine geometry and prefer-UIA policy."""

from __future__ import annotations

from src.recorder.uia_input_boxes import (
    UIA_CONTROL_TYPE_SUFFIX,
    _LABELED_TEXT_CONTROL_TYPES,
    _UIA_BUTTON_CONTROL_TYPE_ID,
    _UIA_CHECKBOX_CONTROL_TYPE_ID,
    _UIA_COMBOBOX_CONTROL_TYPE_ID,
    _UIA_CREATE_AND_CONDITION,
    _UIA_CUSTOM_CONTROL_TYPE_ID,
    _UIA_DOCUMENT_CONTROL_TYPE_ID,
    _UIA_EDIT_CONTROL_TYPE_ID,
    _UIA_FIND_ALL_BUILD_CACHE,
    _UIA_FIND_FIRST,
    _UIA_HEADER_CONTROL_TYPE_ID,
    _UIA_HEADERITEM_CONTROL_TYPE_ID,
    _UIA_INPUT_CONTROL_TYPE_IDS,
    _UIA_PANE_CONTROL_TYPE_ID,
    _UIA_RADIOBUTTON_CONTROL_TYPE_ID,
    _UIA_SPLITBUTTON_CONTROL_TYPE_ID,
    _UIA_TABITEM_CONTROL_TYPE_ID,
    _UIA_TEXT_CONTROL_TYPE_ID,
    _UIA_TREEITEM_CONTROL_TYPE_ID,
    attach_findall_timing,
    bind_findall_timing,
    canonicalize_uia_shell_status_label,
    compose_uia_input_landmark_label,
    control_type_token_from_id,
    element_from_point_name_value,
    enrich_uia_snapshot_with_efp_scan,
    filter_uia_input_rects,
    filter_uia_scrollbar_rects,
    control_type_id_from_token,
    _unique_data_item_hit_by_value,
    find_uia_data_item_by_name,
    find_uia_elements_by_name,
    is_oversized_uia_rect,
    is_uia_input_container_rect,
    is_usable_uia_landmark_label,
    merge_nested_element_boxes,
    snapshot_efp_label_covered,
    snapshot_uia_screen,
    split_uia_name_value_label,
    summarize_findall_timing,
    uia_label_combine_name_value,
    uia_label_prefer_value,
    uia_snapshot_covers_efp_at_point,
    uia_snapshot_has_rich_labels,
    uia_snapshot_image_local,
    usable_covering_snapshot_text_labels,
)


def test_uia_input_control_type_ids_include_combobox_and_pane() -> None:
    assert _UIA_EDIT_CONTROL_TYPE_ID in _UIA_INPUT_CONTROL_TYPE_IDS
    assert _UIA_DOCUMENT_CONTROL_TYPE_ID in _UIA_INPUT_CONTROL_TYPE_IDS
    assert _UIA_COMBOBOX_CONTROL_TYPE_ID in _UIA_INPUT_CONTROL_TYPE_IDS
    assert _UIA_PANE_CONTROL_TYPE_ID in _UIA_INPUT_CONTROL_TYPE_IDS


def test_control_type_token_from_id_maps_edit_hosts_to_input() -> None:
    assert control_type_token_from_id(_UIA_EDIT_CONTROL_TYPE_ID) == "input"
    assert control_type_token_from_id(_UIA_DOCUMENT_CONTROL_TYPE_ID) == "input"
    assert control_type_token_from_id(_UIA_COMBOBOX_CONTROL_TYPE_ID) == "input"
    assert control_type_token_from_id(_UIA_PANE_CONTROL_TYPE_ID) == "input"
    assert control_type_token_from_id(_UIA_BUTTON_CONTROL_TYPE_ID) == "button"
    assert control_type_token_from_id(_UIA_TEXT_CONTROL_TYPE_ID) == "text"
    assert control_type_token_from_id(None) == ""


def test_control_type_token_from_id_maps_expanded_labeled_types() -> None:
    assert control_type_token_from_id(_UIA_TABITEM_CONTROL_TYPE_ID) == "tab_item"
    assert control_type_token_from_id(_UIA_CHECKBOX_CONTROL_TYPE_ID) == "check_box"
    assert control_type_token_from_id(_UIA_RADIOBUTTON_CONTROL_TYPE_ID) == "radio_button"
    assert control_type_token_from_id(_UIA_SPLITBUTTON_CONTROL_TYPE_ID) == "split_button"
    assert control_type_token_from_id(_UIA_HEADER_CONTROL_TYPE_ID) == "header"
    assert control_type_token_from_id(_UIA_HEADERITEM_CONTROL_TYPE_ID) == "header_item"
    assert control_type_token_from_id(_UIA_CUSTOM_CONTROL_TYPE_ID) == "custom"
    for type_id in (
        _UIA_TABITEM_CONTROL_TYPE_ID,
        _UIA_CHECKBOX_CONTROL_TYPE_ID,
        _UIA_RADIOBUTTON_CONTROL_TYPE_ID,
        _UIA_SPLITBUTTON_CONTROL_TYPE_ID,
        _UIA_HEADER_CONTROL_TYPE_ID,
        _UIA_HEADERITEM_CONTROL_TYPE_ID,
        _UIA_CUSTOM_CONTROL_TYPE_ID,
    ):
        assert type_id in _LABELED_TEXT_CONTROL_TYPES
    assert UIA_CONTROL_TYPE_SUFFIX["tab_item"] == "索引標籤"
    assert UIA_CONTROL_TYPE_SUFFIX["check_box"] == "核取方塊"
    assert UIA_CONTROL_TYPE_SUFFIX["radio_button"] == "選項按鈕"
    assert UIA_CONTROL_TYPE_SUFFIX["split_button"] == "分割按鈕"
    assert UIA_CONTROL_TYPE_SUFFIX["header"] == "標頭"
    assert UIA_CONTROL_TYPE_SUFFIX["header_item"] == "標頭項目"
    assert UIA_CONTROL_TYPE_SUFFIX["custom"] == "自訂"


def test_findall_build_cache_slot_and_timing_helpers() -> None:
    # Full vtable index for IUIAutomationElement::FindAllBuildCache.
    assert _UIA_FIND_ALL_BUILD_CACHE == 8
    # FindFirst / CreateAndCondition slots used by DataItem Name lookup.
    assert _UIA_FIND_FIRST == 5
    assert _UIA_CREATE_AND_CONDITION == 25
    with bind_findall_timing() as samples:
        from src.recorder.uia_input_boxes import _note_findall_timing

        _note_findall_timing(elapsed_s=0.125, count=12)
        _note_findall_timing(elapsed_s=0.05, count=3)
    total_s, calls, hits = summarize_findall_timing(samples)
    assert calls == 2
    assert hits == 15
    assert abs(total_s - 0.175) < 1e-6
    stamped = attach_findall_timing(
        {"texts": [{"bbox": [1, 2, 3, 4], "text": "A"}], "inputs": [], "scrollbars": []},
        samples,
    )
    assert stamped is not None
    assert stamped["findall_s"] == 0.175
    assert stamped["findall_count"] == 15
    wall_only = attach_findall_timing(
        {"texts": [{"bbox": [1, 2, 3, 4], "text": "A"}], "inputs": [1], "scrollbars": []},
        [],
        wall_s=0.42,
    )
    assert wall_only is not None
    assert wall_only["findall_s"] == 0.42
    assert wall_only["findall_count"] == 2


def test_find_uia_data_item_by_name_rejects_empty_inputs() -> None:
    assert find_uia_data_item_by_name(0, "N15") is None
    assert find_uia_data_item_by_name(123, "") is None
    assert find_uia_data_item_by_name(123, "   ") is None
    assert find_uia_data_item_by_name(0, "電腦名稱", value="Red-Hat10") is None


def test_unique_data_item_hit_by_value_picks_single_match() -> None:
    hits = [
        ((446, 110, 130, 22), "電腦名稱", "Lannie_CentOS8", "data_item"),
        ((446, 155, 130, 22), "電腦名稱", "lannie-ubuntu-20", "data_item"),
        ((446, 179, 130, 22), "電腦名稱", "Red-Hat10", "data_item"),
    ]
    picked = _unique_data_item_hit_by_value(hits, "Red-Hat10")
    assert picked == ((446, 179, 130, 22), "電腦名稱", "Red-Hat10")


def test_unique_data_item_hit_by_value_rejects_zero_or_many() -> None:
    hits = [
        ((446, 110, 130, 22), "電腦名稱", "Lannie_CentOS8", "data_item"),
        ((446, 179, 130, 22), "電腦名稱", "Red-Hat10", "data_item"),
        ((500, 200, 130, 22), "電腦名稱", "Red-Hat10", "data_item"),
    ]
    assert _unique_data_item_hit_by_value(hits, "missing") is None
    assert _unique_data_item_hit_by_value(hits, "Red-Hat10") is None
    assert _unique_data_item_hit_by_value(hits, "") is None
    assert _unique_data_item_hit_by_value([], "Red-Hat10") is None


def test_find_uia_data_item_by_name_value_uses_findall_filter(
    monkeypatch,
) -> None:
    """Non-empty value → FindAll Name hits, then unique Value post-filter."""
    import src.recorder.uia_input_boxes as uia_mod

    monkeypatch.setattr(uia_mod, "os", type("O", (), {"name": "nt"})())

    def fake_findall(hwnd, name, *, control_type_id=None, timeout_s=2.5):
        assert hwnd == 42
        assert name == "電腦名稱"
        assert control_type_id == uia_mod._UIA_DATAITEM_CONTROL_TYPE_ID
        return [
            ((446, 110, 130, 22), "電腦名稱", "Lannie_CentOS8", "data_item"),
            ((446, 179, 130, 22), "電腦名稱", "Red-Hat10", "data_item"),
        ]

    monkeypatch.setattr(uia_mod, "find_uia_elements_by_name", fake_findall)
    hit = find_uia_data_item_by_name(42, "電腦名稱", value="Red-Hat10")
    assert hit == ((446, 179, 130, 22), "電腦名稱", "Red-Hat10")
    assert find_uia_data_item_by_name(42, "電腦名稱", value="missing") is None


def test_find_uia_elements_by_name_rejects_empty_inputs() -> None:
    assert find_uia_elements_by_name(0, "龜速的網域 (3)") == []
    assert find_uia_elements_by_name(123, "") == []
    assert find_uia_elements_by_name(123, "   ") == []


def test_control_type_id_from_token_skips_generic_text() -> None:
    assert control_type_id_from_token("text") is None
    assert control_type_id_from_token("") is None
    assert control_type_id_from_token("tree_item") == _UIA_TREEITEM_CONTROL_TYPE_ID
    assert control_type_id_from_token("button") == _UIA_BUTTON_CONTROL_TYPE_ID


def test_snapshot_efp_cover_ignores_pane_inputs_requires_agreeing_text() -> None:
    snap = {
        "window": [0, 0, 400, 300],
        "inputs": [[0, 0, 400, 300]],
        "scrollbars": [],
        "texts": [
            {"bbox": [350, 280, 390, 295], "text": "遠方分頁"},
            {"bbox": [100, 100, 220, 124], "text": "龜速的網域 (3)|1"},
        ],
    }
    # Pane covers (150,110) but only the tree label agrees with EFP.
    assert usable_covering_snapshot_text_labels(snap, (150, 110)) == [
        "龜速的網域 (3)|1"
    ]
    assert snapshot_efp_label_covered(snap, (150, 110), "龜速的網域 (3)|1")
    assert not snapshot_efp_label_covered(snap, (150, 110), "完全不同的標籤")
    # Far tab does not cover cursor; panes alone never cover.
    sparse = {
        "window": [0, 0, 400, 300],
        "inputs": [[0, 0, 400, 300]],
        "texts": [{"bbox": [350, 280, 390, 295], "text": "遠方分頁"}],
    }
    assert usable_covering_snapshot_text_labels(sparse, (150, 110)) == []
    assert not uia_snapshot_covers_efp_at_point(
        sparse,
        (150, 110),
        efp_payload={"name": "龜速的網域 (3)", "value": "1", "bbox": [100, 100, 120, 24]},
    )
    assert not uia_snapshot_has_rich_labels(sparse)


def test_enrich_uia_snapshot_with_efp_scan_merges_unique_leaves(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def _probe(x: int, y: int) -> dict:
        calls.append((int(x), int(y)))
        if int(x) == 150 and int(y) == 110:
            return {
                "element_from_point_s": 0.01,
                "name": "龜速的網域 (3)",
                "value": "1",
                "bbox": [100, 100, 120, 24],
                "control_type": "tree_item",
            }
        if 160 <= int(x) <= 200 and 140 <= int(y) <= 180:
            return {
                "element_from_point_s": 0.01,
                "name": "新竹公司 (17)",
                "value": "1",
                "bbox": [100, 140, 110, 24],
                "control_type": "tree_item",
            }
        return {"element_from_point_s": 0.01}

    monkeypatch.setattr(
        "src.recorder.uia_input_boxes._element_from_point_name_value_impl",
        _probe,
    )
    snap = {
        "window": [0, 0, 400, 300],
        "inputs": [[0, 0, 400, 300]],
        "scrollbars": [],
        "texts": [{"bbox": [350, 280, 390, 295], "text": "遠方分頁"}],
    }
    out = enrich_uia_snapshot_with_efp_scan(
        snap,
        cursor_xy=(150, 110),
        max_probes=20,
        max_seconds=1.0,
        step_px=40,
        roi_half_px=80,
    )
    assert out is not None
    labels = {str(row.get("text") or "") for row in out["texts"]}
    assert "龜速的網域 (3)|1" in labels
    assert "新竹公司 (17)|1" in labels
    assert out.get("efp_scan_added", 0) >= 1
    assert isinstance(out.get("efp_scan_s"), float)
    assert calls[0] == (150, 110)


def test_snapshot_uia_screen_refetch_on_efp_cover_miss(monkeypatch) -> None:
    walks: list[tuple[int, tuple[int, int] | None]] = []

    def _impl(hwnd: int, *, cursor_xy=None, budgets=None):
        walks.append((int(hwnd), cursor_xy))
        if len(walks) == 1:
            return {
                "window": [0, 0, 400, 300],
                "inputs": [[0, 0, 400, 300]],
                "scrollbars": [],
                "texts": [{"bbox": [350, 280, 390, 295], "text": "遠方分頁"}],
            }
        return {
            "window": [0, 0, 400, 300],
            "inputs": [[0, 0, 400, 300]],
            "scrollbars": [],
            "texts": [
                {"bbox": [350, 280, 390, 295], "text": "遠方分頁"},
                {
                    "bbox": [100, 100, 220, 124],
                    "text": "龜速的網域 (3)|1",
                    "control_type": "tree_item",
                },
            ],
        }

    monkeypatch.setattr(
        "src.recorder.uia_input_boxes._snapshot_uia_screen_impl",
        _impl,
    )
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes._element_from_point_name_value_impl",
        lambda x, y: {
            "element_from_point_s": 0.01,
            "name": "龜速的網域 (3)",
            "value": "1",
            "bbox": [100, 100, 120, 24],
            "control_type": "tree_item",
        },
    )
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.enrich_uia_snapshot_with_efp_scan",
        lambda snap, **_kwargs: snap,
    )

    class _OsStub:
        name = "nt"

    monkeypatch.setattr("src.recorder.uia_input_boxes.os", _OsStub())
    snap = snapshot_uia_screen(99, cursor_xy=(150, 110))
    assert snap is not None
    assert snap.get("findall_refetch") == 1
    assert len(walks) == 2
    assert walks[0][1] == (150, 110)
    assert walks[1][1] == (150, 110)
    assert any(
        str(row.get("text") or "") == "龜速的網域 (3)|1"
        for row in snap.get("texts") or []
    )


def test_snapshot_uia_screen_keeps_expanded_labeled_control_types(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.enumerate_edit_document_scrollbar_screen",
        lambda hwnd, **_kwargs: (
            [],
            [],
            [
                ((10, 20, 30, 40), "面板設定|TopTab|Setting", "tab_item"),
                ((50, 60, 70, 80), "自訂葉", "custom"),
                ((1, 2, 3, 4), "", "custom"),  # unnamed Custom dropped by snapshot
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes._window_screen_xywh",
        lambda hwnd: (0, 0, 100, 100),
    )

    class _OsStub:
        name = "nt"

    monkeypatch.setattr("src.recorder.uia_input_boxes.os", _OsStub())
    snap = snapshot_uia_screen(123)
    assert snap is not None
    assert snap["texts"] == [
        {
            "bbox": [10, 20, 30, 40],
            "text": "面板設定|TopTab|Setting",
            "control_type": "tab_item",
        },
        {
            "bbox": [50, 60, 70, 80],
            "text": "自訂葉",
            "control_type": "custom",
        },
    ]
    assert isinstance(snap.get("findall_s"), float)
    assert snap["findall_s"] >= 0.0
    assert isinstance(snap.get("findall_count"), int)


def test_element_from_point_name_value_extracts_raw_strings(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.uia_text_element_at_point",
        lambda x, y: ("展開|1", (10, 20, 30, 40), "button", "展開", "1"),
    )
    payload = element_from_point_name_value(100, 200)
    assert payload["name"] == "展開"
    assert payload["value"] == "1"
    assert payload["bbox"] == [10, 20, 30, 40]
    assert payload["control_type"] == "button"
    assert isinstance(payload["element_from_point_s"], float)
    assert payload["element_from_point_s"] >= 0.0


def test_element_from_point_name_value_keeps_timing_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.uia_text_element_at_point",
        lambda x, y: None,
    )
    payload = element_from_point_name_value(1, 2)
    assert "name" not in payload
    assert "value" not in payload
    assert "bbox" not in payload
    assert "control_type" not in payload
    assert isinstance(payload["element_from_point_s"], float)
    assert payload["element_from_point_s"] >= 0.0


def test_merge_nested_element_boxes_keeps_larger() -> None:
    outer = (612, 874, 21, 18)
    inner = (615, 876, 14, 14)
    assert merge_nested_element_boxes([inner, outer]) == [outer]


def test_merge_nested_element_boxes_keeps_side_by_side() -> None:
    left = (10, 10, 16, 16)
    right = (40, 10, 16, 16)
    assert sorted(merge_nested_element_boxes([left, right])) == sorted([left, right])


def test_is_oversized_uia_rect_near_full_window() -> None:
    window = (0, 0, 1000, 800)
    assert is_oversized_uia_rect((10, 10, 950, 750), window) is True
    assert is_oversized_uia_rect((100, 100, 200, 40), window) is False
    assert is_oversized_uia_rect((0, 0, 0, 10), window) is True


def test_filter_uia_input_rects_drops_oversized_and_out_of_roi() -> None:
    window = (100, 50, 400, 300)  # screen xywh
    screen_boxes = [
        (120, 80, 300, 120),  # good field
        (100, 50, 500, 350),  # near-window oversized
        (600, 80, 700, 120),  # outside ocr_roi after convert
    ]
    boxes = filter_uia_input_rects(
        screen_boxes,
        window_screen_xywh=window,
        monitor_offset=(100, 50),
        image_w=500,
        image_h=400,
        ocr_roi=(0, 0, 250, 200),
        ocr_roi_pad=0,
    )
    assert boxes == [(20, 30, 180, 40)]


def test_filter_uia_input_rects_drops_tall_containers_keeps_wide_thin() -> None:
    window = (0, 0, 1400, 900)
    screen_boxes = [
        (100, 100, 300, 128),  # normal field
        (20, 200, 1368, 221),  # wide thin single-line (keep)
        (20, 50, 1368, 500),  # tall wide container (~36% area, w/h frac)
        (10, 10, 1560, 820),  # huge area container (>=0.4)
    ]
    boxes = filter_uia_input_rects(
        screen_boxes,
        window_screen_xywh=window,
        monitor_offset=(0, 0),
        image_w=1400,
        image_h=900,
        drop_container_inputs=True,
    )
    assert (100, 100, 200, 28) in boxes
    assert (20, 200, 1348, 21) in boxes
    assert all(box[3] < 100 or box[2] < 900 for box in boxes)
    scrollbars = filter_uia_scrollbar_rects(
        [(1380, 20, 1396, 870)],
        window_screen_xywh=window,
        monitor_offset=(0, 0),
        image_w=1400,
        image_h=900,
        drop_container_inputs=True,
    )
    assert scrollbars == [(1380, 20, 16, 850)]


def test_compose_uia_input_landmark_label_keeps_embedded_automation_id() -> None:
    assert (
        compose_uia_input_landmark_label("傳真號碼|Textbox|txt_Fax", "txt_Fax")
        == "傳真號碼|Textbox|txt_Fax"
    )
    assert compose_uia_input_landmark_label("傳真號碼", "txt_Fax") == "傳真號碼|txt_Fax"
    assert compose_uia_input_landmark_label("", "txt_Fax") == "txt_Fax"


def test_uia_snapshot_image_local_keeps_input_labels() -> None:
    inputs, scrollbars, texts = uia_snapshot_image_local(
        {
            "window": [0, 0, 800, 600],
            "inputs": [[100, 200, 300, 230], [10, 10, 700, 500]],
            "input_labels": ["傳真號碼|txt_Fax", ""],
            "scrollbars": [],
            "texts": [],
        },
        monitor_offset=(0, 0),
        image_w=800,
        image_h=600,
    )
    assert scrollbars is None
    assert texts == []
    assert inputs is not None
    # Tall container dropped; labeled fax field kept as labeled row.
    assert len(inputs) == 1
    box, label, _ct = inputs[0]
    assert box == (100, 200, 200, 30)
    assert label == "傳真號碼|txt_Fax"
    assert is_uia_input_container_rect((10, 10, 690, 490), (0, 0, 800, 600)) is True


def test_filter_uia_input_rects_drops_edge_sliver_from_wrong_monitor() -> None:
    """Off-monitor boxes with no true intersection must not become targets."""
    window = (1912, -9, 1936, 1048)  # maximized on secondary, left overlaps primary
    # Real control on secondary; no overlap with primary (0..1920).
    screen_boxes = [
        (2100, 300, 2250, 330),
        (100, 200, 220, 240),  # fully on primary — keep
    ]
    boxes = filter_uia_input_rects(
        screen_boxes,
        window_screen_xywh=window,
        monitor_offset=(0, 0),
        image_w=1920,
        image_h=1080,
    )
    assert boxes == [(100, 200, 120, 40)]


def test_filter_uia_input_rects_drops_fully_off_and_intersects_partial() -> None:
    from src.recorder.uia_input_boxes import filter_uia_labeled_text_rects

    window = (0, 0, 1920, 1080)
    boxes = filter_uia_input_rects(
        [
            (-500, 300, -350, 330),  # fully off left
            (100, -400, 250, -370),  # fully off top
            (2100, 300, 2102, 330),  # tiny fully off right
            (-50, 300, 100, 330),  # partial left → true intersect
        ],
        window_screen_xywh=window,
        monitor_offset=(0, 0),
        image_w=1920,
        image_h=1080,
    )
    assert boxes == [(0, 300, 100, 30)]

    labeled = filter_uia_labeled_text_rects(
        [
            ((-500, 300, -350, 330), "A", "text"),
            ((-500, 300, -350, 330), "B", "text"),
            ((100, 200, 220, 240), "Keep", "text"),
        ],
        window_screen_xywh=window,
        monitor_offset=(0, 0),
        image_w=1920,
        image_h=1080,
    )
    assert [(box, label) for box, label, _ct in labeled] == [
        ((100, 200, 120, 40), "Keep")
    ]


def test_is_clipped_uia_sliver_keeps_uncropped_small_box() -> None:
    from src.recorder.uia_input_boxes import _is_clipped_uia_sliver

    assert (
        _is_clipped_uia_sliver(orig_w=2, orig_h=16, clipped_w=2, clipped_h=16) is False
    )
    assert (
        _is_clipped_uia_sliver(orig_w=100, orig_h=20, clipped_w=1, clipped_h=20) is True
    )
    assert (
        _is_clipped_uia_sliver(orig_w=100, orig_h=20, clipped_w=40, clipped_h=20) is True
    )


def test_hough_path_still_drops_inputs_overlapping_h_scrollbar() -> None:
    """Hough/YOLO merge still drops boxes that overlap horizontal scrollbars."""
    from cua_mcp.input_box_rectangles import (
        _drop_xywh_boxes_overlapping_horizontal_scrollbars,
    )

    inputs = [(10, 70, 180, 25), (10, 10, 80, 30)]
    h_scrollbars = [(0, 80, 200, 15)]
    kept = _drop_xywh_boxes_overlapping_horizontal_scrollbars(inputs, h_scrollbars)
    assert kept == [(10, 10, 80, 30)]


def test_uia_label_prefer_value_over_name() -> None:
    assert uia_label_prefer_value("  val  ", "name") == "val"
    assert uia_label_prefer_value("", "  name  ") == "name"
    assert uia_label_prefer_value("   ", "name") == "name"
    assert uia_label_prefer_value("", "") == ""


def test_uia_label_combine_name_value() -> None:
    assert uia_label_combine_name_value("龜速的網域", "1") == "龜速的網域|1"
    assert uia_label_combine_name_value("  name  ", "  val  ") == "name|val"
    assert uia_label_combine_name_value("same", "same") == "same"
    assert uia_label_combine_name_value("name", "") == "name"
    assert uia_label_combine_name_value("", "val") == "val"
    assert uia_label_combine_name_value("", "") == ""
    assert uia_label_combine_name_value("Excel - 已釘選", "1") == "Excel|1"


def test_split_uia_name_value_label() -> None:
    assert split_uia_name_value_label("龜速的網域 (3)|1") == ("龜速的網域 (3)", "1")
    assert split_uia_name_value_label("龜速的網域 (3)") == ("龜速的網域 (3)", "")
    assert split_uia_name_value_label("電腦資產|LeftMenu|Asset_PC") == (
        "電腦資產",
        "LeftMenu|Asset_PC",
    )
    assert split_uia_name_value_label("") == ("", "")
    assert split_uia_name_value_label(None) == ("", "")
    assert split_uia_name_value_label("Excel - 已釘選|1") == ("Excel", "1")
    combined = uia_label_combine_name_value("龜速的網域", "1")
    assert split_uia_name_value_label(combined) == ("龜速的網域", "1")
    combined_pipe_value = uia_label_combine_name_value(
        "電腦資產", "LeftMenu|Asset_PC"
    )
    assert split_uia_name_value_label(combined_pipe_value) == (
        "電腦資產",
        "LeftMenu|Asset_PC",
    )


def test_resolve_usable_uia_probe_label_combines_name_and_value() -> None:
    from src.recorder.uia_input_boxes import resolve_usable_uia_probe_label

    assert (
        resolve_usable_uia_probe_label(preferred="1", name="自訂勾選", value="1")
        == "自訂勾選|1"
    )
    assert resolve_usable_uia_probe_label(preferred="1", name="1", value="") == "1"
    assert resolve_usable_uia_probe_label(preferred="False", name="", value="False") == (
        "False"
    )
    assert (
        resolve_usable_uia_probe_label(
            preferred="1", name="龜速的網域", value="1"
        )
        == "龜速的網域|1"
    )
    assert resolve_usable_uia_probe_label(preferred="", name="", value="") == ""


def test_canonicalize_uia_shell_status_label_strips_taskbar_tails() -> None:
    assert canonicalize_uia_shell_status_label("Excel - 已釘選") == "Excel"
    assert (
        canonicalize_uia_shell_status_label("Cursor - 2 個執行中視窗") == "Cursor"
    )
    assert (
        canonicalize_uia_shell_status_label(
            "Google Chrome - 1 個執行中視窗 - 已釘選"
        )
        == "Google Chrome"
    )
    assert (
        canonicalize_uia_shell_status_label("日月晶耀 - 晶鑰系統 - 已釘選")
        == "日月晶耀 - 晶鑰系統"
    )
    assert canonicalize_uia_shell_status_label("文件 (已釘選)") == "文件"
    assert (
        canonicalize_uia_shell_status_label("Chrome - 3 running windows")
        == "Chrome"
    )
    assert canonicalize_uia_shell_status_label("Notes - Pinned") == "Notes"
    assert canonicalize_uia_shell_status_label("Excel") == "Excel"
    assert uia_label_prefer_value("Excel - 已釘選", "name") == "Excel"
    assert is_usable_uia_landmark_label("Excel - 已釘選") is True
    assert is_usable_uia_landmark_label("（已釘選）") is False
    from src.recorder.uia_input_boxes import is_weak_efp_hub_label

    assert is_weak_efp_hub_label("CoreInput") is True
    assert is_weak_efp_hub_label("Windows.UI.Core.CoreWindow") is True
    assert is_weak_efp_hub_label("Launch 檔案總管") is False
    assert is_weak_efp_hub_label("下載 (已釘選)") is False


def test_uia_snapshot_image_local_strips_shell_status_from_old_texts() -> None:
    from src.recorder.uia_input_boxes import uia_snapshot_image_local

    _inputs, _scrollbars, texts = uia_snapshot_image_local(
        {
            "window": [0, 0, 200, 100],
            "inputs": [],
            "scrollbars": [],
            "texts": [
                {
                    "bbox": [10, 10, 80, 30],
                    "text": "Google Chrome - 1 個執行中視窗 - 已釘選",
                    "control_type": "button",
                }
            ],
        },
        monitor_offset=(0, 0),
        image_w=200,
        image_h=100,
    )
    assert texts == [((10, 10, 70, 20), "Google Chrome", "button")]


def test_match_scrollbar_thumb_center_picks_aligned_shorter_thumb() -> None:
    from src.recorder.uia_input_boxes import match_scrollbar_thumb_center

    # Vertical bar at x=900; live thumb mid-track.
    bar = (900, 160, 16, 840)
    thumb = (900, 400, 916, 520)
    wrong_column = (200, 400, 216, 520)
    too_tall = (900, 160, 916, 1000)
    center = match_scrollbar_thumb_center(bar, [wrong_column, too_tall, thumb])
    assert center == (908, 460)


def test_match_scrollbar_thumb_center_rejects_edge_clip_only() -> None:
    from src.recorder.uia_input_boxes import match_scrollbar_thumb_center

    bar = (900, 160, 16, 840)
    # Only clips the right edge of the bar column.
    clipped = (910, 400, 940, 520)
    assert match_scrollbar_thumb_center(bar, [clipped]) is None


def test_scrollbar_thumb_unions_same_line_arrow_buttons() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    thumb = (900, 400, 916, 520)
    up = (900, 160, 916, 176)
    down = (898, 980, 918, 996)
    other_column = (200, 160, 216, 176)
    rects, used = compose_scrollbar_screen_rects([], [thumb], [up, down, other_column])
    assert rects == [(898, 160, 918, 996)]
    assert used == [0, 1]


def test_scrollbar_thumb_alone_is_a_scrollbar() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    thumb = (900, 400, 916, 520)
    rects, used = compose_scrollbar_screen_rects([], [thumb], [])
    assert rects == [thumb]
    assert used == []


def test_scrollbar_thumb_ignores_button_that_only_clips_the_edge() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    thumb = (900, 400, 916, 520)
    clipped = (910, 160, 940, 176)
    rects, used = compose_scrollbar_screen_rects([], [thumb], [clipped])
    assert rects == [thumb]
    assert used == []


def test_scrollbar_thumbs_keep_separate_columns() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    left_thumb = (100, 400, 116, 520)
    right_thumb = (400, 400, 416, 520)
    left_up = (100, 160, 116, 176)
    right_down = (400, 800, 416, 816)
    rects, used = compose_scrollbar_screen_rects(
        [],
        [left_thumb, right_thumb],
        [left_up, right_down],
    )
    assert rects == [(100, 160, 116, 520), (400, 400, 416, 816)]
    assert used == [0, 1]


def test_scrollbar_thumb_inside_control_is_not_added_again() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    control = (890, 150, 920, 1000)
    thumb = (900, 400, 916, 520)
    up = (900, 160, 916, 176)
    down = (900, 980, 916, 996)
    rects, used = compose_scrollbar_screen_rects([control], [thumb], [up, down])
    assert rects == [control]
    assert used == [0, 1]


def test_scrollbar_thumb_inside_huge_control_is_kept() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    window_sized = (0, 0, 800, 900)
    thumb = (780, 400, 796, 520)
    rects, used = compose_scrollbar_screen_rects([window_sized], [thumb], [])
    assert rects == [window_sized, thumb]
    assert used == []


def test_winforms_position_thumb_unions_line_up_down_by_name_gates() -> None:
    """Name gates accept Position + Line up/down; compose unions same column."""
    from src.recorder.uia_input_boxes import (
        compose_scrollbar_screen_rects,
        is_uia_scrollbar_line_button_part,
        is_uia_scrollbar_thumb_part,
    )

    assert is_uia_scrollbar_thumb_part(name="Position")
    assert is_uia_scrollbar_thumb_part(automation_id="ScrollbarThumb")
    assert not is_uia_scrollbar_thumb_part(name="", automation_id="")
    assert not is_uia_scrollbar_thumb_part(name="Slider", automation_id="")
    assert is_uia_scrollbar_line_button_part(name="Line up")
    assert is_uia_scrollbar_line_button_part(name="Line down")
    assert is_uia_scrollbar_line_button_part(name="Page down")
    assert is_uia_scrollbar_line_button_part(name="Column left")
    assert is_uia_scrollbar_line_button_part(automation_id="UpButton")
    assert not is_uia_scrollbar_line_button_part(name="Next")

    # Geometry from recording_20261113_062457 thumb + ends (screen xyxy).
    thumb = (2289, 162, 2312, 413)  # Position bbox xywh → xyxy
    line_up = (2289, 137, 2312, 163)
    line_down = (2289, 579, 2312, 605)
    page_down = (2289, 413, 2312, 580)
    other_column = (278, 71, 301, 97)
    rects, used = compose_scrollbar_screen_rects(
        [],
        [thumb],
        [line_up, line_down, page_down, other_column],
    )
    assert rects == [(2289, 137, 2312, 605)]
    assert used == [0, 1, 2]


def test_line_up_without_thumb_does_not_create_scrollbar() -> None:
    from src.recorder.uia_input_boxes import compose_scrollbar_screen_rects

    line_up = (2289, 137, 2312, 163)
    line_down = (2289, 579, 2312, 605)
    rects, used = compose_scrollbar_screen_rects([], [], [line_up, line_down])
    assert rects == []
    assert used == []


def test_labeled_text_control_types_include_interactive_items() -> None:
    from src.recorder import uia_input_boxes as uia

    assert uia._UIA_BUTTON_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_MENUITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_TREEITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    # DataItem stays out of FindAll (Excel cell blow-up); EFP names cells.
    assert uia._UIA_DATAITEM_CONTROL_TYPE_ID not in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia.UIA_CONTROL_TYPE_SUFFIX["data_item"] == "資料項目"
    assert uia._UIA_LISTITEM_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_TEXT_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES
    assert uia._UIA_HYPERLINK_CONTROL_TYPE_ID in uia._LABELED_TEXT_CONTROL_TYPES


def test_efp_payload_is_usable_data_item() -> None:
    from src.recorder.uia_input_boxes import efp_payload_is_usable_data_item

    assert efp_payload_is_usable_data_item(
        {
            "name": "G8",
            "value": "",
            "control_type": "data_item",
            "bbox": [466, 477, 72, 22],
        }
    )
    assert not efp_payload_is_usable_data_item(
        {"name": "G8", "control_type": "button", "bbox": [1, 1, 10, 10]}
    )
    assert not efp_payload_is_usable_data_item(
        {"name": "Position", "control_type": "data_item", "bbox": [1, 1, 10, 10]}
    )
    assert not efp_payload_is_usable_data_item(
        {"name": "CoreInput", "control_type": "data_item", "bbox": [1, 1, 10, 10]}
    )
    assert not efp_payload_is_usable_data_item(
        {"name": "G8", "control_type": "data_item"}
    )


def test_efp_payload_is_usable_labeled_leaf() -> None:
    from src.recorder.uia_input_boxes import efp_payload_is_usable_labeled_leaf

    assert efp_payload_is_usable_labeled_leaf(
        {
            "name": "G8",
            "value": "",
            "control_type": "data_item",
            "bbox": [466, 477, 72, 22],
        }
    )
    assert efp_payload_is_usable_labeled_leaf(
        {
            "name": "[檔案] 索引標籤",
            "value": "",
            "control_type": "button",
            "bbox": [40, 10, 60, 24],
        }
    )
    assert efp_payload_is_usable_labeled_leaf(
        {
            "name": "Home",
            "value": "",
            "control_type": "tab_item",
            "bbox": [100, 10, 48, 22],
        }
    )
    assert not efp_payload_is_usable_labeled_leaf(
        {"name": "Position", "control_type": "button", "bbox": [1, 1, 10, 10]}
    )
    assert not efp_payload_is_usable_labeled_leaf(
        {"name": "CoreInput", "control_type": "button", "bbox": [1, 1, 10, 10]}
    )
    assert not efp_payload_is_usable_labeled_leaf(
        {"name": "檔案", "control_type": "pane", "bbox": [1, 1, 10, 10]}
    )
    assert not efp_payload_is_usable_labeled_leaf(
        {"name": "檔案", "control_type": "button"}
    )


def test_select_uia_field_clip_picks_smallest_containing_caret() -> None:
    from src.recorder.uia_input_boxes import select_uia_field_clip_containing_point

    caret = (150, 100)
    outer = (50, 50, 400, 200)
    inner = (100, 80, 220, 120)
    other = (500, 80, 600, 120)

    def _enumerate(_hwnd: int):
        return [outer, inner, other]

    chosen = select_uia_field_clip_containing_point(
        caret,
        root_hwnd=1,
        enumerate_fn=_enumerate,
        focused_rect_fn=lambda: None,
        window_screen_xywh=(0, 0, 800, 600),
    )
    assert chosen == inner


def test_select_uia_field_clip_rejects_window_sized_document() -> None:
    from src.recorder.uia_input_boxes import select_uia_field_clip_containing_point

    caret = (400, 300)
    sheet = (0, 0, 800, 600)  # whole window Document

    chosen = select_uia_field_clip_containing_point(
        caret,
        root_hwnd=1,
        enumerate_fn=lambda _hwnd: [sheet],
        focused_rect_fn=lambda: None,
        window_screen_xywh=(0, 0, 800, 600),
    )
    assert chosen is None


def test_select_uia_field_clip_includes_focused_edit() -> None:
    from src.recorder.uia_input_boxes import select_uia_field_clip_containing_point

    caret = (120, 90)
    focused = (100, 80, 300, 110)

    chosen = select_uia_field_clip_containing_point(
        caret,
        root_hwnd=1,
        enumerate_fn=lambda _hwnd: [],
        focused_rect_fn=lambda: focused,
        window_screen_xywh=(0, 0, 800, 600),
    )
    assert chosen == focused


def test_sample_uia_ocr_clip_rect_skips_wide_focus() -> None:
    from src.recorder.uia_input_boxes import sample_uia_ocr_clip_rect

    calls: list[tuple[int, int]] = []

    def _select(caret_xy, **_kwargs):
        calls.append(caret_xy)
        return (10, 20, 30, 40)

    assert (
        sample_uia_ocr_clip_rect(
            caret_xy=(50, 50),
            focus_rect=(10, 40, 400, 70),  # wide field, not caret-thin
            select_fn=_select,
        )
        is None
    )
    assert calls == []

    assert sample_uia_ocr_clip_rect(
        caret_xy=(50, 50),
        focus_rect=(50, 40, 51, 60),
        select_fn=_select,
    ) == (10, 20, 30, 40)
    assert calls == [(50, 50)]


def test_walk_visit_score_prefers_near_cursor_and_penalizes_offscreen() -> None:
    from src.recorder.uia_input_boxes import (
        chebyshev_distance_point_to_xyxy,
        walk_visit_score,
    )

    clip = (0, 0, 1000, 1000)
    near = (90, 90, 110, 110)
    far = (800, 800, 820, 820)
    off = (2000, 2000, 2010, 2010)
    cursor = (100, 100)
    assert chebyshev_distance_point_to_xyxy(100, 100, near) == 0
    assert walk_visit_score(near, clip=clip, cursor_xy=cursor) < walk_visit_score(
        far, clip=clip, cursor_xy=cursor
    )
    assert walk_visit_score(off, clip=clip, cursor_xy=cursor) > walk_visit_score(
        far, clip=clip, cursor_xy=cursor
    )
    assert walk_visit_score(None, clip=clip, cursor_xy=cursor) > walk_visit_score(
        off, clip=clip, cursor_xy=cursor
    )


def test_budgeted_priority_walk_stops_on_max_hits_near_cursor_first() -> None:
    from src.recorder.uia_input_boxes import WalkBudgets, run_budgeted_priority_walk

    # Flat siblings under an implicit root: ids 0..4 with bboxes left-to-right.
    boxes = {
        0: (0, 0, 10, 10),
        1: (100, 0, 110, 10),
        2: (200, 0, 210, 10),
        3: (300, 0, 310, 10),
        4: (400, 0, 410, 10),
    }
    children = {n: [] for n in boxes}
    matched: list[int] = []
    released: list[int] = []

    result = run_budgeted_priority_walk(
        list(boxes.keys()),
        node_bbox=lambda n: boxes[n],
        node_matches=lambda _n: True,
        on_match=lambda n: matched.append(n),
        node_children=lambda n: list(children[n]),
        release_node=lambda n: released.append(n),
        clip=(0, 0, 1000, 1000),
        cursor_xy=(205, 5),
        budgets=WalkBudgets(
            max_hits=2,
            max_nodes=100,
            max_seconds=5.0,
            max_children_per_parent=80,
        ),
    )
    assert result.truncated is True
    assert result.stop_reason == "max_hits"
    assert result.hits == 2
    assert matched[0] == 2  # closest to cursor
    assert set(released) == set(boxes.keys())


def test_budgeted_priority_walk_stops_on_max_nodes() -> None:
    from src.recorder.uia_input_boxes import WalkBudgets, run_budgeted_priority_walk

    boxes = {i: (i * 10, 0, i * 10 + 5, 5) for i in range(10)}
    matched: list[int] = []

    result = run_budgeted_priority_walk(
        list(boxes.keys()),
        node_bbox=lambda n: boxes[n],
        node_matches=lambda _n: False,
        on_match=lambda n: matched.append(n),
        node_children=lambda _n: [],
        release_node=lambda _n: None,
        clip=(0, 0, 1000, 1000),
        cursor_xy=None,
        budgets=WalkBudgets(
            max_hits=500,
            max_nodes=3,
            max_seconds=5.0,
            max_children_per_parent=80,
        ),
    )
    assert result.truncated is True
    assert result.stop_reason == "max_nodes"
    assert result.nodes == 3
    assert matched == []


def test_budgeted_priority_walk_sibling_cap_keeps_nearest_children() -> None:
    from src.recorder.uia_input_boxes import WalkBudgets, run_budgeted_priority_walk

    # Root seed is parent 0; it expands to many children. Cap keeps 2 nearest.
    boxes = {
        0: (0, 0, 500, 500),
        1: (10, 10, 20, 20),
        2: (100, 10, 110, 20),
        3: (200, 10, 210, 20),
        4: (300, 10, 310, 20),
    }
    children = {
        0: [1, 2, 3, 4],
        1: [],
        2: [],
        3: [],
        4: [],
    }
    matched: list[int] = []
    visited: list[int] = []

    def _on_match(n: int) -> None:
        matched.append(n)

    def _release(n: int) -> None:
        visited.append(n)

    result = run_budgeted_priority_walk(
        [0],
        node_bbox=lambda n: boxes[n],
        node_matches=lambda n: n != 0,
        on_match=_on_match,
        node_children=lambda n: list(children[n]),
        release_node=_release,
        clip=(0, 0, 1000, 1000),
        cursor_xy=(15, 15),
        budgets=WalkBudgets(
            max_hits=50,
            max_nodes=50,
            max_seconds=5.0,
            max_children_per_parent=2,
        ),
    )
    assert result.truncated is False
    assert result.stop_reason == "complete"
    assert set(matched) == {1, 2}
    # Dropped siblings are released without becoming matches.
    assert 3 not in matched and 4 not in matched
    assert set(visited) >= {0, 1, 2, 3, 4}


def test_budgeted_priority_walk_stops_on_max_seconds() -> None:
    from src.recorder.uia_input_boxes import WalkBudgets, run_budgeted_priority_walk

    boxes = {i: (i, 0, i + 1, 1) for i in range(5)}
    ticks = {"t": 0.0}

    def _clock() -> float:
        # Advance past budget after the first visit check inside the loop.
        ticks["t"] += 1.0
        return float(ticks["t"])

    result = run_budgeted_priority_walk(
        list(boxes.keys()),
        node_bbox=lambda n: boxes[n],
        node_matches=lambda _n: True,
        on_match=lambda _n: None,
        node_children=lambda _n: [],
        release_node=lambda _n: None,
        clip=(0, 0, 100, 100),
        cursor_xy=None,
        budgets=WalkBudgets(
            max_hits=500,
            max_nodes=500,
            max_seconds=0.5,
            max_children_per_parent=80,
        ),
        clock=_clock,
    )
    assert result.truncated is True
    assert result.stop_reason == "max_seconds"


def test_uia_worker_call_runs_and_is_reentrant() -> None:
    from src.recorder.uia_thread import uia_call, uia_worker_is_current_thread

    marker: list[bool] = []

    def _inner() -> str:
        marker.append(uia_worker_is_current_thread())
        return "ok"

    def _outer() -> str:
        assert uia_worker_is_current_thread()
        return uia_call(_inner, timeout_s=2.5) or ""

    assert uia_call(_outer, timeout_s=2.5) == "ok"
    assert marker == [True]
