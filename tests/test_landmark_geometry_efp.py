"""EFP Name/Value verify after landmark_geometry candidate picks."""

from __future__ import annotations

from cua_mcp.select_ui_element import UiDetection
from cua_mcp.select_mouse_target import (
    _efp_name_value_matches,
    _order_geometry_efp_candidates,
    _recorded_efp_present,
    _verify_geometry_candidates_with_efp,
)


def _det(cx: int, cy: int, *, text: str = "", class_name: str = "input") -> UiDetection:
    return UiDetection(
        bbox=(cx - 5, cy - 5, 10, 10),
        cx=cx,
        cy=cy,
        class_id=0,
        class_name=class_name,
        text=text,
        source="uia",
    )


def test_recorded_efp_present() -> None:
    assert _recorded_efp_present("下載", "") is True
    assert _recorded_efp_present("", "v") is True
    assert _recorded_efp_present("  ", "") is False
    assert _recorded_efp_present(None, None) is False


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
        return ("", (0, 0, 1, 1), "", "X", "Y")

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
            return ("label", (0, 0, 1, 1), "text", "錯誤", "錯誤")
        return ("label", (0, 0, 1, 1), "text", "下載 (已釘選)", "下載 (已釘選)")

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
        return ("label", (0, 0, 1, 1), "text", "其他", "其他")

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


def test_verify_synth_only_hit() -> None:
    def probe(x: int, y: int):
        assert (x, y) == (99, 88)
        return ("label", (0, 0, 1, 1), "text", "文件 (已釘選)", "文件 (已釘選)")

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
            return ("label", (0, 0, 1, 1), "text", "下載 (已釘選)", "下載 (已釘選)")
        return ("label", (0, 0, 1, 1), "text", "錯誤", "錯誤")

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
        return ("label", (0, 0, 1, 1), "text", "其他", "")

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
        return ("", (0, 0, 1, 1), "", "X", "Y")

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
            return ("label", (0, 0, 1, 1), "text", "下載 (已釘選)", "下載 (已釘選)")
        return ("label", (0, 0, 1, 1), "text", "圖片", "圖片")

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
        return ("label", (0, 0, 1, 1), "text", "圖片", "圖片")

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
