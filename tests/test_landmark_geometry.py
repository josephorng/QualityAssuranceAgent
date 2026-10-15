"""Tests for scale-normalized landmark constellation helpers."""

from __future__ import annotations

from src.common.nearby_side import NearbyHint, Side
from src.recorder.landmark_geometry import (
    build_landmark_geometry,
    filter_landmarks_by_constellation,
    parse_landmark_geometry,
    synthesize_target_point,
    target_far_from_synthesized,
    vectors_agree,
)
from src.recorder.compile_tool_calls import compile_tool_calls
from src.recorder.models import RecordedEvent


def _cand(label_text: str, bbox: tuple[int, int, int, int], *, class_name: str = "text"):
    return {
        "bbox": list(bbox),
        "class_name": class_name,
        "text": label_text,
        "icons": [],
    }


def _vision_three_landmarks() -> dict:
    # A(10,10), B(110,10), C(10,110); target at (60, 60)
    return {
        "used_vision": True,
        "local_cursor": [60, 60],
        "candidates": [
            _cand("搜尋", (50, 50, 20, 20)),
            _cand("eee", (0, 0, 20, 20)),
            _cand("晴時多雲", (100, 0, 20, 20)),
            _cand("Ln 1", (0, 100, 20, 20)),
        ],
    }


def test_build_full_graph_pair_count_and_scale() -> None:
    vision = _vision_three_landmarks()
    hints = [
        NearbyHint("「eee」文字", Side.LEFT),
        NearbyHint("「晴時多雲」文字", Side.RIGHT),
        NearbyHint("「Ln 1」文字", Side.BELOW),
    ]
    geom = build_landmark_geometry(vision, hints, target_label="「搜尋」文字")
    assert geom is not None
    assert geom["version"] == 1
    assert len(geom["landmarks"]) == 3
    # Full directed graph: N*(N-1)
    assert len(geom["landmark_to_landmark"]) == 3 * 2
    assert len(geom["landmark_to_target"]) == 3
    # Median pairwise among (10,10)/(110,10)/(10,110): distances 100, 100, ~141.4 → 100
    assert abs(geom["scale"] - 100.0) < 1e-6
    # Target from eee (10,10) → (60,60) = (0.5, 0.5) in units of scale 100
    by_from = {row["from"]: row for row in geom["landmark_to_target"]}
    assert abs(by_from["「eee」文字"]["dx"] - 0.5) < 1e-6
    assert abs(by_from["「eee」文字"]["dy"] - 0.5) < 1e-6


def test_filter_drops_wrong_ocr_twin() -> None:
    vision = _vision_three_landmarks()
    hints = [
        NearbyHint("「eee」文字", Side.LEFT),
        NearbyHint("「晴時多雲」文字", Side.RIGHT),
        NearbyHint("「Ln 1」文字", Side.BELOW),
    ]
    recorded = build_landmark_geometry(vision, hints)
    assert recorded is not None
    # Twin 「eee」 far away breaks vectors to the other two landmarks.
    matched = {
        "「eee」文字": (800.0, 800.0),
        "「晴時多雲」文字": (110.0, 10.0),
        "「Ln 1」文字": (10.0, 110.0),
    }
    kept = filter_landmarks_by_constellation(recorded, matched)
    assert "「eee」文字" not in kept
    assert "「晴時多雲」文字" in kept
    assert "「Ln 1」文字" in kept


def test_synthesize_target_point_from_vectors() -> None:
    vision = _vision_three_landmarks()
    hints = [
        NearbyHint("「eee」文字", Side.LEFT),
        NearbyHint("「晴時多雲」文字", Side.RIGHT),
        NearbyHint("「Ln 1」文字", Side.BELOW),
    ]
    recorded = build_landmark_geometry(vision, hints)
    assert recorded is not None
    kept = {
        "「eee」文字": (10.0, 10.0),
        "「晴時多雲」文字": (110.0, 10.0),
        "「Ln 1」文字": (10.0, 110.0),
    }
    point = synthesize_target_point(recorded, kept)
    assert point is not None
    assert abs(point[0] - 60) <= 1
    assert abs(point[1] - 60) <= 1


def test_vectors_agree_and_target_far_helpers() -> None:
    assert vectors_agree((1.0, 0.0), (0.99, 0.05))
    assert not vectors_agree((1.0, 0.0), (0.0, 1.0))
    assert not target_far_from_synthesized((10, 10), (12, 10), scale=100.0)
    assert target_far_from_synthesized((10, 10), (80, 10), scale=100.0)


def test_parse_landmark_geometry_rejects_bad_payload() -> None:
    assert parse_landmark_geometry(None) is None
    assert parse_landmark_geometry({"version": 99}) is None
    assert parse_landmark_geometry({"version": 1, "landmarks": []}) is None


def test_compile_attaches_landmark_geometry_for_input_not_text() -> None:
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        click_count=1,
    )
    vision = _vision_three_landmarks()
    vision["candidates"][0] = {
        "bbox": [50, 50, 20, 20],
        "class_name": "input",
        "text": "",
        "icons": [],
    }
    input_instruction = (
        "將滑鼠移到輸入欄（在「eee」文字的左邊、在「晴時多雲」文字的右邊、"
        "在「Ln 1」文字的上面），並點擊滑鼠一下。"
    )
    calls = compile_tool_calls(
        event,
        input_instruction,
        vision=vision,
    )
    assert calls is not None
    move = calls[0]
    assert move["name"] == "move_mouse"
    geom = move["arguments"].get("landmark_geometry")
    assert isinstance(geom, dict)
    assert len(geom["landmarks"]) == 3
    assert len(geom["landmark_to_landmark"]) == 6
    assert geom["target_label"] == "輸入欄"

    text_instruction = (
        "將滑鼠移到「搜尋」文字（在「eee」文字的左邊、在「晴時多雲」文字的右邊、"
        "在「Ln 1」文字的上面），並點擊滑鼠一下。"
    )
    text_calls = compile_tool_calls(
        event,
        text_instruction,
        vision=_vision_three_landmarks(),
    )
    assert text_calls is not None
    assert "landmark_geometry" not in text_calls[0]["arguments"]

    offset_instruction = (
        "將滑鼠移到「搜尋」文字右方12個像素、下方49個像素的位置"
        "（在「eee」文字的左邊、在「晴時多雲」文字的右邊、在「Ln 1」文字的上面），"
        "並點擊滑鼠一下。"
    )
    offset_calls = compile_tool_calls(
        event,
        offset_instruction,
        vision=_vision_three_landmarks(),
    )
    assert offset_calls is not None
    offset_geom = offset_calls[0]["arguments"].get("landmark_geometry")
    assert isinstance(offset_geom, dict)
    # Reference text is pinned into the constellation with nearby landmarks.
    labels = {row["label"] for row in offset_geom["landmarks"]}
    assert "「搜尋」文字" in labels
    assert len(offset_geom["landmarks"]) >= 3
