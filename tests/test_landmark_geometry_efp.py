"""EFP Name/Value verify after landmark_geometry candidate picks."""

from __future__ import annotations

from cua_mcp.select_ui_element import UiDetection
from cua_mcp.select_mouse_target import (
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
    _DATA_ITEM_EFP_STEP_X,
    _DATA_ITEM_EFP_STEP_Y,
    _chebyshev_ring_offsets,
    _distance_synth_to_det_bbox,
    _efp_gate_required_for_hub,
    _efp_name_value_matches,
    _filter_yolo_for_near_synth_pool,
    _findall_pool_near_synth,
    _keep_uia_first_on_subtarget_efp_soft_miss,
    _label_similarity,
    _mouse_select_fields_from_ocr_fallback_instruction,
    _order_geometry_efp_candidates,
    _phrase_match_pool,
    _preferred_yolo_class_for_near_synth,
    _recorded_efp_agrees_with_hub,
    _recorded_efp_present,
    _scan_efp_neighbors_for_name,
    _try_uia_first_unique_select,
    _verify_geometry_candidates_with_efp,
)


def _det(
    cx: int,
    cy: int,
    *,
    text: str = "",
    class_name: str = "input",
    class_id: int | None = None,
) -> UiDetection:
    if class_id is None:
        class_id = YOLO_CLASS_INPUT if class_name == "input" else YOLO_CLASS_TEXT
    return UiDetection(
        bbox=(cx - 5, cy - 5, 10, 10),
        cx=cx,
        cy=cy,
        class_id=class_id,
        class_name=class_name,
        text=text,
        source="uia",
    )


def _simple_landmark_geometry(
    *,
    landmark_label: str = "LM",
    landmark_xy: tuple[float, float] = (100.0, 100.0),
    target_xy: tuple[float, float] = (150.0, 100.0),
    scale: float = 100.0,
) -> dict:
    """One-landmark constellation whose synth lands at ``target_xy``."""
    dx = (target_xy[0] - landmark_xy[0]) / scale
    dy = (target_xy[1] - landmark_xy[1]) / scale
    return {
        "version": 1,
        "scale": scale,
        "landmarks": [
            {"label": landmark_label, "side": None, "xy": list(landmark_xy)},
        ],
        "landmark_to_landmark": [],
        "landmark_to_target": [
            {"from": landmark_label, "dx": dx, "dy": dy},
        ],
        "target_label": "下載",
    }


def test_recorded_efp_present() -> None:
    assert _recorded_efp_present("下載", "") is True
    assert _recorded_efp_present("", "v") is True
    assert _recorded_efp_present("  ", "") is False
    assert _recorded_efp_present(None, None) is False


def test_efp_gate_required_for_hub_soft_on_input_scrollbar() -> None:
    assert not _efp_gate_required_for_hub(
        anchor="輸入欄",
        control_type_token="input",
        recorded_efp_name="密碼|Textbox|txt_Password",
        recorded_efp_value="",
    )
    assert not _efp_gate_required_for_hub(
        anchor="「密碼|Textbox|txt_Password」文字所在的輸入欄",
        control_type_token="input",
        recorded_efp_name="密碼|Textbox|txt_Password",
        recorded_efp_value="",
    )
    assert not _efp_gate_required_for_hub(
        anchor="「網址與搜尋列」文字所在的輸入欄",
        control_type_token="input",
        recorded_efp_name="網址與搜尋列",
        recorded_efp_value="",
    )
    assert not _efp_gate_required_for_hub(
        anchor="滾動條",
        control_type_token="scrollbar",
        recorded_efp_name="ScrollbarThumb",
        recorded_efp_value="",
    )
    assert _efp_gate_required_for_hub(
        anchor="「下載」文字",
        control_type_token="text",
        recorded_efp_name="下載 (已釘選)",
        recorded_efp_value="下載 (已釘選)",
    )
    assert _efp_gate_required_for_hub(
        anchor="「網址與搜尋列」文字",
        control_type_token="text",
        recorded_efp_name="網址與搜尋列",
        recorded_efp_value="",
    )
    assert not _efp_gate_required_for_hub(
        anchor="「下載」文字",
        control_type_token="text",
        recorded_efp_name="",
        recorded_efp_value="",
    )


def test_efp_name_value_matches_rules() -> None:
    assert _efp_name_value_matches("下載 (已釘選)", "下載 (已釘選)", "下載 (已釘選)", "下載 (已釘選)")
    assert _efp_name_value_matches("下載 (已釘選)", "", "下載 (已釘選)", "")
    assert not _efp_name_value_matches("其他", "下載 (已釘選)", "下載 (已釘選)", "下載 (已釘選)")
    assert not _efp_name_value_matches("下載 (已釘選)", "錯", "下載 (已釘選)", "下載 (已釘選)")
    assert _efp_name_value_matches("下載 (已釘選)", "anything", "下載 (已釘選)", "")


def test_order_prefers_geom_then_distance() -> None:
    a = _det(100, 100)
    b = _det(200, 100)
    c = _det(150, 100)
    ordered = _order_geometry_efp_candidates([a, b, c], (140, 100), preferred=b)
    assert ordered[0] is b
    assert ordered[1] is c  # closer to 140 than a
    assert ordered[2] is a


def test_verify_blank_recorded_keeps_geom_det() -> None:
    preferred = _det(10, 10)
    other = _det(50, 50)
    probes: list[tuple[int, int]] = []

    def probe(x: int, y: int):
        probes.append((x, y))
        return ("", None, "", "X", "Y")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [preferred, other],
        synth=(10, 10),
        geom_det=preferred,
        recorded_name="",
        recorded_value="",
        point_probe_fn=probe,
    )
    assert reason == "hit"
    assert det is preferred
    assert point == (10, 10)
    assert probes == []


def test_verify_retries_second_candidate_on_name_mismatch() -> None:
    wrong = _det(10, 10)
    right = _det(40, 10)

    def probe(x: int, y: int):
        if (x, y) == (10, 10):
            return ("label", None, "text", "錯誤", "錯誤")
        return ("label", None, "text", "下載 (已釘選)", "下載 (已釘選)")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [wrong, right],
        synth=(10, 10),
        geom_det=wrong,
        recorded_name="下載 (已釘選)",
        recorded_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "hit"
    assert det is right
    assert point == (40, 10)


def test_verify_exhaust_returns_miss() -> None:
    a = _det(10, 10)
    b = _det(40, 10)

    def probe(x: int, y: int):
        return ("label", None, "text", "其他", "其他")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [a, b],
        synth=(10, 10),
        geom_det=a,
        recorded_name="下載 (已釘選)",
        recorded_value="",
        point_probe_fn=probe,
    )
    assert reason == "miss"
    assert det is None
    assert point is None


def test_verify_miss_logs_live_efp_name_value(monkeypatch) -> None:
    """Miss diagnostics must include live ElementFromPoint Name/Value samples."""
    a = _det(10, 10, text="Word")
    logs: list[str] = []

    def probe(x: int, y: int):
        if (x, y) == (10, 10):
            return ("label", None, "text", "Word", "")
        return ("label", None, "text", "任務列", "")

    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._log_info",
        lambda msg: logs.append(str(msg)),
    )

    det, point, reason = _verify_geometry_candidates_with_efp(
        [a],
        synth=(12, 10),
        geom_det=None,
        recorded_name="Word - 已釘選",
        recorded_value="",
        point_probe_fn=probe,
    )
    assert reason == "miss"
    assert det is None
    assert point is None
    joined = "\n".join(logs)
    assert "landmark_geometry_efp_miss" in joined
    assert "recorded_name='Word - 已釘選'" in joined
    assert "name='Word'" in joined
    assert "name='任務列'" in joined
    assert "[pool] point=[10,10]" in joined
    assert "[synth] point=[12,10]" in joined


def test_verify_synth_only_hit() -> None:
    def probe(x: int, y: int):
        assert (x, y) == (99, 88)
        return ("label", None, "text", "文件 (已釘選)", "文件 (已釘選)")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [],
        synth=(99, 88),
        geom_det=None,
        recorded_name="文件 (已釘選)",
        recorded_value="文件 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "synth_hit"
    assert det is None
    assert point == (99, 88)


def test_anchor_efp_pool_scan_second_wins() -> None:
    """YOLO/OCR shortlist path: no preferred geom_det; nearer-to-centroid then rest."""
    far_wrong = _det(10, 10)
    near_right = _det(40, 10)
    # Centroid of (10,10) and (40,10) is (25, 10) → near_right is closer.
    synth = (25, 10)

    def probe(x: int, y: int):
        if (x, y) == (40, 10):
            return ("label", None, "text", "下載 (已釘選)", "下載 (已釘選)")
        return ("label", None, "text", "錯誤", "錯誤")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [far_wrong, near_right],
        synth=synth,
        geom_det=None,
        recorded_name="下載 (已釘選)",
        recorded_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "hit"
    assert det is near_right
    assert point == (40, 10)


def test_anchor_efp_pool_scan_all_miss() -> None:
    a = _det(10, 10)
    b = _det(40, 10)

    def probe(x: int, y: int):
        return ("label", None, "text", "其他", "")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [a, b],
        synth=(25, 10),
        geom_det=None,
        recorded_name="下載 (已釘選)",
        recorded_value="",
        point_probe_fn=probe,
    )
    assert reason == "miss"
    assert det is None
    assert point is None


def test_anchor_efp_blank_recorded_skips_probes() -> None:
    a = _det(10, 10)
    probes: list[tuple[int, int]] = []

    def probe(x: int, y: int):
        probes.append((x, y))
        return ("", None, "", "X", "Y")

    # Blank recorded → helper short-circuits; call site keeps unique/LLM.
    assert not _recorded_efp_present("", "")
    det, point, reason = _verify_geometry_candidates_with_efp(
        [a],
        synth=(10, 10),
        geom_det=None,
        recorded_name="",
        recorded_value="",
        point_probe_fn=probe,
    )
    assert reason == "synth_hit"
    assert det is None
    assert point == (10, 10)
    assert probes == []


def test_uia_yolo_subtarget_efp_nearer_wrong_farther_wins() -> None:
    """In-pane YOLO boxes: synth = UIA pane center; first-by-distance wrong Name."""
    pane_center = (2000, 500)
    near_wrong = _det(1990, 400, text="圖片")  # closer to pane center
    far_right = _det(1990, 280, text="下載")

    def probe(x: int, y: int):
        if (x, y) == (1990, 280):
            return ("label", None, "text", "下載 (已釘選)", "下載 (已釘選)")
        return ("label", None, "text", "圖片", "圖片")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [near_wrong, far_right],
        synth=pane_center,
        geom_det=None,
        recorded_name="下載 (已釘選)",
        recorded_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "hit"
    assert det is far_right
    assert point == (1990, 280)


def test_uia_yolo_subtarget_efp_all_miss() -> None:
    pane_center = (2000, 500)
    a = _det(1990, 400)
    b = _det(1990, 280)

    def probe(x: int, y: int):
        return ("label", None, "text", "圖片", "圖片")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [a, b],
        synth=pane_center,
        geom_det=None,
        recorded_name="下載 (已釘選)",
        recorded_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "miss"
    assert det is None
    assert point is None


def test_findall_pool_near_synth_prefers_nearest_and_radius() -> None:
    synth = (150, 100)
    near = _det(148, 100, text="")
    mid = _det(200, 100, text="下載")
    far = _det(900, 900, text="下載")  # beyond max(2*100, 200)=200
    yolo_near = UiDetection(
        bbox=(140, 90, 10, 10),
        cx=145,
        cy=95,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="",
        source="yolo",
    )
    pool = _findall_pool_near_synth(
        [far, mid, near, yolo_near],
        synth,
        scale=100.0,
        k=20,
    )
    assert near in pool
    assert mid in pool
    assert far not in pool
    assert yolo_near in pool  # joint UIA+YOLO pool
    assert pool[0] is near or pool[0] is yolo_near


def test_findall_pool_bbox_containment_beats_toolbar_centers() -> None:
    """Large empty pane containing synth ranks above nearer toolbar text centers."""
    synth = (209, 256)
    toolbar = _det(226, 104, text="貼上")
    # Left-nav pane: center far from synth, but bbox contains synth.
    pane = UiDetection(
        bbox=(2, 128, 333, 876),
        cx=168,
        cy=566,
        class_id=YOLO_CLASS_INPUT,
        class_name="input",
        text="",
        source="uia",
    )
    assert _distance_synth_to_det_bbox(synth, pane) == 0.0
    assert _distance_synth_to_det_bbox(synth, toolbar) > 0.0
    pool = _findall_pool_near_synth([toolbar, pane], synth, scale=463.0, k=20)
    assert pool[0] is pane
    assert toolbar in pool


def test_findall_pool_k_budget() -> None:
    synth = (0, 0)
    dets = [_det(i * 10, 0, text=f"t{i}") for i in range(30)]
    pool = _findall_pool_near_synth(dets, synth, scale=None, k=5)
    assert len(pool) == 5
    assert pool[0].cx == 0
    assert pool[-1].cx == 40


def test_uia_first_findall_near_synth_efp_wins_without_name_match() -> None:
    """Empty FindAll pane near synth + EFP recovers leaf missing from Name shortlist."""
    geom = _simple_landmark_geometry()
    landmark = _det(100, 100, text="LM", class_name="text")
    near_empty = _det(148, 100, text="", class_name="input")
    # Name==下載 but far from synth — would be the only Name shortlist hit.
    far_download = _det(500, 500, text="下載", class_name="text")

    def probe(x: int, y: int):
        if (x, y) == (148, 100):
            return ("label", None, "text", "下載 (已釘選)", "下載 (已釘選)")
        return ("label", None, "text", "其他", "其他")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, near_empty, far_download],
        anchor="下載",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="下載 (已釘選)",
        recorded_efp_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is near_empty


def test_uia_first_joint_yolo_leaf_efp_hit() -> None:
    """YOLO box near synth + EFP recovers leaf FindAll Name shortlist would miss."""
    geom = _simple_landmark_geometry()
    landmark = _det(100, 100, text="LM", class_name="text")
    toolbar = _det(226, 104, text="貼上", class_name="text")
    yolo_leaf = UiDetection(
        bbox=(65, 270, 20, 14),
        cx=75,
        cy=277,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="",
        source="yolo",
    )
    # Far Name match only — not near synth.
    far_download = _det(500, 500, text="下載", class_name="text")

    def probe(x: int, y: int):
        if (x, y) == (75, 277):
            return ("label", None, "text", "下載 (已釘選)", "下載 (已釘選)")
        return ("label", None, "text", "其他", "其他")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, toolbar, yolo_leaf, far_download],
        anchor="下載",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="下載 (已釘選)",
        recorded_efp_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is yolo_leaf


def test_filter_yolo_for_near_synth_keeps_uia() -> None:
    uia = _det(10, 10, text="LM", class_name="text")
    yolo_text = UiDetection(
        bbox=(0, 0, 10, 10),
        cx=5,
        cy=5,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="",
        source="yolo",
    )
    yolo_icon = UiDetection(
        bbox=(20, 20, 10, 10),
        cx=25,
        cy=25,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        text="",
        source="yolo",
    )
    filtered = _filter_yolo_for_near_synth_pool(
        [uia, yolo_text, yolo_icon],
        preferred_yolo_class=YOLO_CLASS_TEXT,
    )
    assert uia in filtered
    assert yolo_text in filtered
    assert yolo_icon not in filtered


def test_preferred_yolo_class_from_bare_match_anchor_and_control_type() -> None:
    """Structured replay: match_anchor=圖片 + control_type=text → TEXT."""
    assert (
        _preferred_yolo_class_for_near_synth("圖片", "text") == YOLO_CLASS_TEXT
    )
    assert (
        _preferred_yolo_class_for_near_synth("圖片", "hyperlink")
        == YOLO_CLASS_TEXT
    )
    assert (
        _preferred_yolo_class_for_near_synth("圖片", "icon") == YOLO_CLASS_ELEMENT
    )
    assert _preferred_yolo_class_for_near_synth("圖片", None) is None


def test_uia_first_text_hub_skips_nearer_yolo_icon() -> None:
    """「…」文字 hub: nearer YOLO icon must not beat farther YOLO text leaf."""
    # Synth lands near both; icon is closer to synth than text.
    geom = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(166.0, 350.0),
        scale=100.0,
    )
    landmark = _det(100, 100, text="LM", class_name="text")
    yolo_icon = UiDetection(
        bbox=(159, 343, 14, 14),
        cx=166,
        cy=350,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        text="",
        source="yolo",
    )
    yolo_text = UiDetection(
        bbox=(180, 343, 26, 13),
        cx=193,
        cy=349,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="",
        source="yolo",
    )

    def probe(x: int, y: int):
        if (x, y) in {(166, 350), (193, 349)}:
            return ("label", None, "text", "圖片 (已釘選)", "圖片 (已釘選)")
        return ("label", None, "text", "其他", "其他")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, yolo_icon, yolo_text],
        anchor="「圖片」文字",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="圖片 (已釘選)",
        recorded_efp_value="圖片 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is yolo_text


def test_uia_first_bare_match_anchor_text_control_skips_nearer_icon() -> None:
    """Replay-shaped args: match_anchor=圖片 + control_type=text filters YOLO."""
    geom = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(166.0, 350.0),
        scale=100.0,
    )
    landmark = _det(100, 100, text="LM", class_name="text")
    yolo_icon = UiDetection(
        bbox=(159, 343, 14, 14),
        cx=166,
        cy=350,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        text="",
        source="yolo",
    )
    yolo_text = UiDetection(
        bbox=(180, 343, 26, 13),
        cx=193,
        cy=349,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="",
        source="yolo",
    )

    def probe(x: int, y: int):
        if (x, y) in {(166, 350), (193, 349)}:
            return ("label", None, "text", "圖片 (已釘選)", "圖片 (已釘選)")
        return ("label", None, "text", "其他", "其他")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, yolo_icon, yolo_text],
        anchor="圖片",
        control_type_token="text",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="圖片 (已釘選)",
        recorded_efp_value="圖片 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is yolo_text


def test_uia_first_icon_hub_skips_nearer_yolo_text() -> None:
    """「…」圖示 hub: nearer YOLO text must not beat farther YOLO element."""
    geom = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(193.0, 349.0),
        scale=100.0,
    )
    landmark = _det(100, 100, text="LM", class_name="text")
    yolo_text = UiDetection(
        bbox=(180, 343, 26, 13),
        cx=193,
        cy=349,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="",
        source="yolo",
    )
    yolo_icon = UiDetection(
        bbox=(159, 343, 14, 14),
        cx=166,
        cy=350,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        text="",
        source="yolo",
    )

    def probe(x: int, y: int):
        if (x, y) in {(166, 350), (193, 349)}:
            return ("label", None, "text", "圖片 (已釘選)", "圖片 (已釘選)")
        return ("label", None, "text", "其他", "其他")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, yolo_text, yolo_icon],
        anchor="「圖片」圖示",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="圖片 (已釘選)",
        recorded_efp_value="圖片 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is yolo_icon


def test_uia_first_ambiguous_hub_keeps_nearer_yolo_icon() -> None:
    """No 文字/圖示 wrapper: both YOLO classes stay eligible; nearer icon wins."""
    geom = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(166.0, 350.0),
        scale=100.0,
    )
    landmark = _det(100, 100, text="LM", class_name="text")
    yolo_icon = UiDetection(
        bbox=(159, 343, 14, 14),
        cx=166,
        cy=350,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        text="",
        source="yolo",
    )
    yolo_text = UiDetection(
        bbox=(180, 343, 26, 13),
        cx=193,
        cy=349,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="",
        source="yolo",
    )

    def probe(x: int, y: int):
        if (x, y) in {(166, 350), (193, 349)}:
            return ("label", None, "text", "圖片 (已釘選)", "圖片 (已釘選)")
        return ("label", None, "text", "其他", "其他")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, yolo_icon, yolo_text],
        anchor="圖片",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="圖片 (已釘選)",
        recorded_efp_value="圖片 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is yolo_icon


def test_verify_prefer_synth_first_snaps_to_matching_pool() -> None:
    """Synth matches first, but a pool det that also EFP-matches wins (snap)."""
    findall = _det(40, 10, text="Launch 檔案總管", class_name="text")

    def probe(x: int, y: int):
        # Both synth and FindAll center expose CoreInput.
        return ("label", None, "text", "CoreInput", "")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [findall],
        synth=(99, 88),
        geom_det=None,
        recorded_name="CoreInput",
        recorded_value="",
        point_probe_fn=probe,
        prefer_synth_first=True,
    )
    assert reason == "hit"
    assert det is findall
    assert point == (40, 10)


def test_verify_pool_first_findall_over_synth_coreinput() -> None:
    """Pool-first: FindAll Launch box with CoreInput EFP beats bare synth."""
    geom = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(918.0, 614.0),
        scale=100.0,
    )
    landmark = _det(100, 100, text="LM", class_name="text")
    findall = UiDetection(
        bbox=(861, 559, 114, 111),
        cx=918,
        cy=614,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="Launch 檔案總管",
        source="uia",
    )

    def probe(x: int, y: int):
        # Synth and FindAll center both report CoreInput (EFP channel).
        if (x, y) in {(918, 614), (918, 638), (893, 638)} or (
            abs(x - 918) < 5 and abs(y - 614) < 5
        ):
            return ("label", None, "text", "CoreInput", "")
        if (x, y) == (int(geom["landmarks"][0]["xy"][0]), int(geom["landmarks"][0]["xy"][1])):
            return ("label", None, "text", "其他", "")
        # Default: CoreInput so synth would also match if probed first.
        return ("label", None, "text", "CoreInput", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, findall],
        anchor="Launch 檔案總管",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="CoreInput",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is findall


def test_ocr_fallback_fields_use_twin_anchor() -> None:
    """Soft-miss adopt: OCR twin instruction becomes match_anchor (FindAll/OCR)."""
    fields = _mouse_select_fields_from_ocr_fallback_instruction(
        "（YOLO+OCR）「檔案總管」文字",
        None,
    )
    assert fields["match_anchor"] == "「檔案總管」文字" or "檔案總管" in str(
        fields["match_anchor"]
    )
    # Exact parse: hub unwrap / regex leaves the quoted text form.
    assert "檔案總管" in str(fields["match_anchor"])
    assert fields["recorded_anchor_match"] is False
    # Similarity: UIA hub vs OCR twin stays below threshold (unchanged).
    assert _label_similarity("Launch 檔案總管", "檔案總管") < 0.8


def test_uia_first_findall_near_synth_efp_miss() -> None:
    geom = _simple_landmark_geometry()
    landmark = _det(100, 100, text="LM", class_name="text")
    near_empty = _det(148, 100, text="", class_name="input")
    far_download = _det(500, 500, text="下載", class_name="text")

    def probe(x: int, y: int):
        return ("label", None, "text", "其他", "其他")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, near_empty, far_download],
        anchor="下載",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="下載 (已釘選)",
        recorded_efp_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_miss"
    assert chosen is None


def test_uia_first_input_hub_efp_miss_continues_name_shortlist() -> None:
    """Input hubs soft-miss constellation EFP and continue into Name unique."""
    geom = _simple_landmark_geometry()
    landmark = _det(100, 100, text="LM", class_name="text")
    named_input = _det(
        148, 100, text="密碼|Textbox|txt_Password", class_name="input"
    )

    def probe(x: int, y: int):
        return ("label", None, "text", "其他", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, named_input],
        anchor="密碼|Textbox|txt_Password",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        filter_by_control_type=True,
        control_type_token="input",
        recorded_anchor=True,
        landmark_geometry=geom,
        recorded_efp_name="密碼|Textbox|txt_Password",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason != "landmark_efp_miss"
    assert reason == "hit"
    assert chosen is named_input


def test_uia_first_input_hub_matches_automation_id_tail() -> None:
    """FindAll Name|AutomationId agrees with recorded Name|Textbox|AutomationId."""
    geom = _simple_landmark_geometry()
    landmark = _det(100, 100, text="LM", class_name="text")
    named_input = _det(148, 100, text="傳真號碼|txt_Fax", class_name="input")
    sibling = _det(200, 100, text="連絡電話|txt_Phone", class_name="input")

    def probe(x: int, y: int):
        return ("label", None, "text", "其他", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, named_input, sibling],
        anchor="傳真號碼|Textbox|txt_Fax",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        filter_by_control_type=True,
        control_type_token="input",
        recorded_anchor=True,
        landmark_geometry=geom,
        recorded_efp_name="傳真號碼|Textbox|txt_Fax",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason == "hit"
    assert chosen is named_input


def test_phrase_match_pool_keeps_input_and_scrollbar_by_class_name() -> None:
    unlabeled_input = _det(976, 447, text="", class_name="input")
    sibling_text = _det(853, 448, text="密碼", class_name="text")
    scrollbar = _det(50, 50, text="", class_name="scrollbar", class_id=YOLO_CLASS_SCROLLBAR)

    input_pool = _phrase_match_pool(
        [unlabeled_input, sibling_text, scrollbar],
        "密碼|Textbox|txt_Password",
        control_type_token="input",
    )
    assert input_pool == [unlabeled_input]

    scrollbar_pool = _phrase_match_pool(
        [unlabeled_input, sibling_text, scrollbar],
        "滾動條",
        control_type_token="scrollbar",
    )
    assert scrollbar_pool == [scrollbar]


def test_phrase_match_pool_omnibox_named_input_excludes_same_name_text() -> None:
    """「網址與搜尋列」文字所在的輸入欄 must not unique-hit a Text leaf."""
    omnibox = _det(924, 83, text="網址與搜尋列", class_name="input")
    # Same Name wrongly typed as text (old recording shape).
    text_leaf = _det(924, 83, text="網址與搜尋列", class_name="text")
    pool = _phrase_match_pool(
        [text_leaf, omnibox],
        "網址與搜尋列",
        control_type_token="input",
    )
    assert pool == [omnibox]


def test_uia_first_input_hub_soft_miss_does_not_unique_hit_sibling_text() -> None:
    """Unlabeled Edit + same-Name Text must not commit the Text after soft EFP."""
    geom = _simple_landmark_geometry()
    landmark = _det(100, 100, text="LM", class_name="text")
    sibling_text = _det(853, 448, text="密碼", class_name="text")
    unlabeled_input = _det(976, 447, text="", class_name="input")

    def probe(x: int, y: int):
        return ("label", None, "text", "其他", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, sibling_text, unlabeled_input],
        anchor="密碼|Textbox|txt_Password",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        filter_by_control_type=True,
        control_type_token="input",
        recorded_anchor=True,
        landmark_geometry=geom,
        recorded_efp_name="密碼|Textbox|txt_Password",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason == "no_anchor"
    assert chosen is None


def test_uia_first_geom_unavailable_continues_name_unique() -> None:
    """Missing landmarks → no synth; unique Name still commits."""
    geom = _simple_landmark_geometry(landmark_label="MISSING")
    only = _det(148, 100, text="下載", class_name="text")

    chosen, reason = _try_uia_first_unique_select(
        [only],
        anchor="下載",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="下載 (已釘選)",
        recorded_efp_value="",
        point_probe_fn=lambda x, y: ("", None, "", "X", "Y"),
    )
    assert reason == "hit"
    assert chosen is only


def test_uia_first_blank_efp_keeps_name_unique() -> None:
    geom = _simple_landmark_geometry()
    landmark = _det(100, 100, text="LM", class_name="text")
    only_download = _det(500, 500, text="下載", class_name="text")
    probes: list[tuple[int, int]] = []

    def probe(x: int, y: int):
        probes.append((x, y))
        return ("label", None, "text", "下載 (已釘選)", "下載 (已釘選)")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, only_download],
        anchor="下載",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason == "hit"
    assert chosen is only_download
    assert probes == []


def test_uia_first_empty_pool_synth_hit() -> None:
    """No FindAll centers in radius → one EFP at synth point."""
    landmark = _det(100, 100, text="LM", class_name="text")
    # Synth far from every detection center so radius filter empties the pool.
    # radius = max(2*50, 200) = 200; landmark dist to (800,800) >> 200.
    geom_far = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(800.0, 800.0),
        scale=50.0,
    )

    def probe(x: int, y: int):
        assert (x, y) == (800, 800)
        return ("label", None, "text", "下載 (已釘選)", "下載 (已釘選)")

    chosen, reason = _try_uia_first_unique_select(
        [landmark],
        anchor="下載",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom_far,
        recorded_efp_name="下載 (已釘選)",
        recorded_efp_value="下載 (已釘選)",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_synth_hit"
    assert chosen is not None
    assert (chosen.cx, chosen.cy) == (800, 800)


def test_chebyshev_ring_counts() -> None:
    assert len(_chebyshev_ring_offsets(1)) == 8
    assert len(_chebyshev_ring_offsets(2)) == 16
    assert (0, -1) in _chebyshev_ring_offsets(1)
    assert (0, -2) in _chebyshev_ring_offsets(2)


def test_scan_efp_neighbors_skips_occupied_bbox_then_hits() -> None:
    """Tall synth cell bbox skips in-cell ring points; next leaf is K10."""
    synth = (500, 400)
    # Cell covers ring-1 up (500,380); K10 is first free cardinal on ring 2.
    k11_bbox = (480, 380, 40, 40)  # y=[380,420) → (500,380) occupied
    k10_point = (500, 360)  # (0, -2) * step_y
    live_points: list[tuple[int, int]] = []

    def probe(x: int, y: int):
        live_points.append((x, y))
        if (x, y) == synth:
            return ("K11", k11_bbox, "data_item", "K11", "")
        if (x, y) == k10_point:
            return ("K10", (480, 350, 40, 20), "data_item", "K10", "")
        return ("other", (x - 5, y - 5, 10, 10), "data_item", "other", "")

    hit = _scan_efp_neighbors_for_name(
        synth,
        "K10",
        "",
        point_probe_fn=probe,
        step_x=_DATA_ITEM_EFP_STEP_X,
        step_y=_DATA_ITEM_EFP_STEP_Y,
    )
    assert hit == k10_point
    # Seed + live probes; (500, 380) must not be live-probed.
    assert live_points[0] == synth
    assert (500, 380) not in live_points
    assert k10_point in live_points


def test_uia_first_data_item_neighbor_hit() -> None:
    """Constellation miss at K11 → neighbor scan finds K10 (FindFirst absent)."""
    landmark = _det(100, 100, text="LM", class_name="text")
    synth_xy = (800, 800)
    geom_far = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(float(synth_xy[0]), float(synth_xy[1])),
        scale=50.0,
    )
    k11_bbox = (780, 780, 40, 40)  # y=[780,820) → (800,780) occupied
    k10_point = (800, 760)  # (0, -2) * 20

    def probe(x: int, y: int):
        # Pool/synth verify path + neighbor scan share this probe.
        if (x, y) == synth_xy:
            return ("K11", k11_bbox, "data_item", "K11", "")
        if (x, y) == k10_point:
            return ("K10", (780, 750, 40, 20), "data_item", "K10", "")
        # Other pool centers (landmark) miss Name.
        return ("LM", (x - 5, y - 5, 10, 10), "text", "LM", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark],
        anchor="K10",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        control_type_token="data_item",
        landmark_geometry=geom_far,
        recorded_efp_name="K10",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_neighbor_hit"
    assert chosen is not None
    assert (chosen.cx, chosen.cy) == k10_point


def test_uia_first_data_item_name_findfirst_hit_before_scan() -> None:
    """Pool/synth miss + FindFirst hit commits without needing neighbor scan."""
    landmark = _det(100, 100, text="LM", class_name="text")
    synth_xy = (800, 800)
    geom_far = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(float(synth_xy[0]), float(synth_xy[1])),
        scale=50.0,
    )
    scan_calls = {"n": 0}

    def probe(x: int, y: int):
        # Wrong cell at synth — would force a neighbor walk without FindFirst.
        if (x, y) == synth_xy:
            return ("K15", (780, 780, 40, 40), "data_item", "K15", "")
        scan_calls["n"] += 1
        return ("LM", (x - 5, y - 5, 10, 10), "text", "LM", "")

    def lookup(name: str):
        assert name == "N15"
        return ((2900, 630, 72, 22), "N15", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark],
        anchor="N15",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        control_type_token="data_item",
        landmark_geometry=geom_far,
        recorded_efp_name="N15",
        recorded_efp_value="",
        point_probe_fn=probe,
        data_item_lookup_fn=lookup,
    )
    assert reason == "uia_data_item_name_hit"
    assert chosen is not None
    assert (chosen.cx, chosen.cy) == (2936, 641)
    assert chosen.uia_control_type == "data_item"
    assert scan_calls["n"] == 0


def test_uia_first_data_item_findfirst_miss_then_neighbor_hit() -> None:
    """FindFirst miss still allows scan-EFP neighbor hit."""
    landmark = _det(100, 100, text="LM", class_name="text")
    synth_xy = (800, 800)
    geom_far = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(float(synth_xy[0]), float(synth_xy[1])),
        scale=50.0,
    )
    k11_bbox = (780, 780, 40, 40)
    k10_point = (800, 760)

    def probe(x: int, y: int):
        if (x, y) == synth_xy:
            return ("K11", k11_bbox, "data_item", "K11", "")
        if (x, y) == k10_point:
            return ("K10", (780, 750, 40, 20), "data_item", "K10", "")
        return ("LM", (x - 5, y - 5, 10, 10), "text", "LM", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark],
        anchor="K10",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        control_type_token="data_item",
        landmark_geometry=geom_far,
        recorded_efp_name="K10",
        recorded_efp_value="",
        point_probe_fn=probe,
        data_item_lookup_fn=lambda _name: None,
    )
    assert reason == "landmark_efp_neighbor_hit"
    assert chosen is not None
    assert (chosen.cx, chosen.cy) == k10_point


def test_uia_first_data_item_findfirst_and_neighbor_miss() -> None:
    """FindFirst miss + neighbor miss → landmark_efp_miss."""
    landmark = _det(100, 100, text="LM", class_name="text")
    synth_xy = (800, 800)
    geom_far = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(float(synth_xy[0]), float(synth_xy[1])),
        scale=50.0,
    )

    def probe(x: int, y: int):
        if (x, y) == synth_xy:
            return ("K15", (780, 780, 40, 40), "data_item", "K15", "")
        return ("OTHER", (x - 5, y - 5, 10, 10), "data_item", "OTHER", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark],
        anchor="N15",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        control_type_token="data_item",
        landmark_geometry=geom_far,
        recorded_efp_name="N15",
        recorded_efp_value="",
        point_probe_fn=probe,
        data_item_lookup_fn=lambda _name: None,
    )
    assert reason == "landmark_efp_miss"
    assert chosen is None


def test_uia_first_data_item_geom_unavailable_findfirst_hit() -> None:
    """No synth geometry → FindFirst can still resolve the cell."""
    # Landmark label missing from detections → geom unavailable.
    orphan = _det(50, 50, text="OTHER", class_name="text")

    def lookup(name: str):
        assert name == "N15"
        return ((1000, 600, 70, 20), "N15", "")

    chosen, reason = _try_uia_first_unique_select(
        [orphan],
        anchor="N15",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        control_type_token="data_item",
        landmark_geometry=_simple_landmark_geometry(
            landmark_label="MISSING",
            landmark_xy=(100.0, 100.0),
            target_xy=(1000.0, 600.0),
        ),
        recorded_efp_name="N15",
        recorded_efp_value="",
        point_probe_fn=lambda *_a: ("x", None, "text", "x", ""),
        data_item_lookup_fn=lookup,
    )
    assert reason == "uia_data_item_name_hit"
    assert chosen is not None
    assert (chosen.cx, chosen.cy) == (1035, 610)


def test_recorded_efp_agrees_with_hub_rejects_overlay_poison() -> None:
    assert _recorded_efp_agrees_with_hub(
        "下載", "下載 (已釘選)", "下載 (已釘選)"
    )
    assert not _recorded_efp_agrees_with_hub(
        "WinMaster7 Console", "NVIDIA GeForce Overlay", ""
    )
    # Weak chrome Names still allow constellation FindAll-over-EFP.
    assert _recorded_efp_agrees_with_hub(
        "Launch 檔案總管", "CoreInput", ""
    )


def test_uia_first_skips_constellation_efp_when_hub_disagrees() -> None:
    """Poisoned overlay EFP must not false-hit; fall through to Name shortlist."""
    geom = _simple_landmark_geometry(
        landmark_label="側欄地標",
        landmark_xy=(50.0, 50.0),
        target_xy=(172.0, 67.0),
        scale=100.0,
    )
    landmark = _det(50, 50, text="側欄地標", class_name="text")
    hub = UiDetection(
        bbox=(167, 62, 10, 10),
        cx=172,
        cy=67,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="WinMaster7 Console",
        source="uia",
        uia_control_type="list_item",
    )
    sibling = UiDetection(
        bbox=(282, 51, 10, 10),
        cx=287,
        cy=56,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="SourceAsk",
        source="uia",
        uia_control_type="list_item",
    )
    # Fullscreen overlay — would be dist=0 from synth and match poisoned EFP.
    overlay = UiDetection(
        bbox=(0, 0, 2560, 1600),
        cx=1280,
        cy=800,
        class_id=YOLO_CLASS_INPUT,
        class_name="input",
        text="NVIDIA GeForce Overlay",
        source="uia",
    )

    def probe(x: int, y: int):
        return (
            "label",
            None,
            "text",
            "NVIDIA GeForce Overlay",
            "",
        )

    chosen, reason = _try_uia_first_unique_select(
        [landmark, hub, sibling, overlay],
        anchor="WinMaster7 Console",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        filter_by_control_type=True,
        control_type_token="list_item",
        recorded_anchor=True,
        landmark_geometry=geom,
        recorded_efp_name="NVIDIA GeForce Overlay",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason != "landmark_efp_hit"
    assert reason != "landmark_efp_synth_hit"
    assert chosen is not overlay
    assert chosen is hub


def test_findall_pool_near_synth_drops_oversize_overlay() -> None:
    icon = _det(172, 67, text="WinMaster7 Console", class_name="text")
    sibling = _det(287, 56, text="SourceAsk", class_name="text")
    overlay = UiDetection(
        bbox=(0, 0, 2560, 1600),
        cx=1280,
        cy=800,
        class_id=YOLO_CLASS_INPUT,
        class_name="input",
        text="NVIDIA GeForce Overlay",
        source="uia",
    )
    pool = _findall_pool_near_synth(
        [icon, sibling, overlay],
        (172, 67),
        scale=100.0,
        k=20,
    )
    assert overlay not in pool
    assert icon in pool
    assert sibling in pool


def test_uia_first_oversize_overlay_does_not_win_efp_even_if_name_matches() -> None:
    """When EFP agrees with hub, fullscreen overlay still must not pool-win."""
    geom = _simple_landmark_geometry(
        landmark_label="側欄地標",
        landmark_xy=(50.0, 50.0),
        target_xy=(172.0, 67.0),
        scale=100.0,
    )
    landmark = _det(50, 50, text="側欄地標", class_name="text")
    hub = _det(172, 67, text="下載", class_name="text")
    sibling = _det(287, 56, text="鄰居", class_name="text")
    overlay = UiDetection(
        bbox=(0, 0, 2560, 1600),
        cx=1280,
        cy=800,
        class_id=YOLO_CLASS_INPUT,
        class_name="input",
        text="下載",
        source="uia",
    )

    def probe(x: int, y: int):
        # Every point reports the recorded Name — overlay would false-hit
        # without the oversize filter.
        return ("label", None, "text", "下載", "")

    chosen, reason = _try_uia_first_unique_select(
        [overlay, landmark, hub, sibling],
        anchor="下載",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        landmark_geometry=geom,
        recorded_efp_name="下載",
        recorded_efp_value="",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is hub
    assert chosen is not overlay


def test_verify_snaps_hit_to_efp_leaf_bbox() -> None:
    """Large FindAll row + matching EFP leaf bbox → commit leaf geometry."""
    # Tree-row hub (wide); Value cell is the true EFP leaf under row center.
    row = UiDetection(
        bbox=(648, 1284, 1622, 36),
        cx=1459,
        cy=1302,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="Level2 row 4",
        source="uia",
    )
    leaf_bbox = (872, 1284, 1398, 36)

    def probe(x: int, y: int):
        if (x, y) == (1459, 1302):
            return ("Value", leaf_bbox, "data_item", "Value", "RufusYan")
        return ("other", None, "text", "other", "")

    det, point, reason = _verify_geometry_candidates_with_efp(
        [row],
        synth=(1459, 1302),
        geom_det=None,
        recorded_name="Value",
        recorded_value="RufusYan",
        point_probe_fn=probe,
    )
    assert reason == "hit"
    assert det is not None
    assert det is not row
    assert det.bbox == leaf_bbox
    assert point == (872 + 1398 // 2, 1284 + 36 // 2)
    assert "RufusYan" in str(det.text or "")


def test_uia_first_landmark_efp_hit_snaps_to_leaf_bbox() -> None:
    """landmark_efp_hit hub is the EFP Value cell, not the FindAll tree row."""
    geom = _simple_landmark_geometry(
        landmark_xy=(100.0, 100.0),
        target_xy=(1459.0, 1302.0),
        scale=100.0,
    )
    landmark = _det(100, 100, text="LM", class_name="text")
    row = UiDetection(
        bbox=(648, 1284, 1622, 36),
        cx=1459,
        cy=1302,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="Level2 row 4",
        source="uia",
    )
    leaf_bbox = (872, 1284, 1398, 36)

    def probe(x: int, y: int):
        if (x, y) == (1459, 1302):
            return ("Value", leaf_bbox, "data_item", "Value", "RufusYan")
        return ("other", None, "text", "other", "")

    chosen, reason = _try_uia_first_unique_select(
        [landmark, row],
        anchor="Value|RufusYan",
        nearby_hints=[],
        nearby_labels=[],
        nearby_phrases=[],
        control_type_token="data_item",
        landmark_geometry=geom,
        recorded_efp_name="Value",
        recorded_efp_value="RufusYan",
        point_probe_fn=probe,
    )
    assert reason == "landmark_efp_hit"
    assert chosen is not None
    assert chosen is not row
    assert chosen.bbox == leaf_bbox
    assert (chosen.cx, chosen.cy) == (872 + 1398 // 2, 1284 + 36 // 2)


def test_keep_uia_first_on_subtarget_efp_soft_miss_reasons() -> None:
    assert _keep_uia_first_on_subtarget_efp_soft_miss("landmark_efp_hit") is True
    assert _keep_uia_first_on_subtarget_efp_soft_miss("landmark_efp_synth_hit") is True
    assert _keep_uia_first_on_subtarget_efp_soft_miss("landmark_efp_neighbor_hit") is True
    assert _keep_uia_first_on_subtarget_efp_soft_miss("uia_data_item_name_hit") is True
    assert _keep_uia_first_on_subtarget_efp_soft_miss("hit") is False
    assert _keep_uia_first_on_subtarget_efp_soft_miss("uia_first_unique") is False
    assert _keep_uia_first_on_subtarget_efp_soft_miss("") is False
