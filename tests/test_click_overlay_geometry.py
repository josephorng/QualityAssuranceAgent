from __future__ import annotations

from src.recorder.vision_context import (
    match_nearby_landmark_overlays,
    primary_target_rect_ltrb,
    resolve_click_overlay_geometry,
)


def _vision() -> dict:
    return {
        "local_cursor": [50, 40],
        "candidates": [
            {
                "bbox": [30, 30, 40, 20],
                "center": [50, 40],
                "class_name": "input",
                "text": None,
                "icons": None,
            },
            {
                "bbox": [5, 30, 18, 16],
                "center": [14, 38],
                "class_name": "text",
                "text": "A1",
                "icons": None,
            },
            {
                "bbox": [40, 5, 20, 12],
                "center": [50, 11],
                "class_name": "text",
                "text": "字型",
                "icons": None,
            },
        ],
    }


def test_primary_target_rect_ltrb() -> None:
    assert primary_target_rect_ltrb(_vision()) == (30, 30, 70, 50)


def test_match_nearby_landmark_overlays_uses_instruction_side() -> None:
    overlays = match_nearby_landmark_overlays(
        _vision(),
        "將滑鼠移到輸入欄（在「A1」文字的右邊、在「字型」文字的下面）",
    )
    labels = {item["label"] for item in overlays}
    assert "「A1」文字" in labels
    assert "「字型」文字" in labels
    a1 = next(item for item in overlays if item["label"] == "「A1」文字")
    assert a1["side"] == "right"
    assert a1["rect"] == (5, 30, 23, 46)


def test_resolve_click_overlay_geometry_pairs_target_and_landmarks() -> None:
    target, landmarks = resolve_click_overlay_geometry(
        _vision(),
        "點擊輸入欄（在「A1」文字的右邊）",
    )
    assert target == (30, 30, 70, 50)
    assert landmarks and landmarks[0]["label"] == "「A1」文字"
