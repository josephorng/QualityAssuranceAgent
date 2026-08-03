"""Tests for 9-section directional nearby landmarks."""

from __future__ import annotations

from src.common.nearby_side import (
    LandmarkCell,
    NearbyHint,
    Side,
    anchor_satisfies_side,
    enrich_nearby_objects_from_goal,
    enrich_tool_arguments_from_goal,
    extract_nearby_hints_from_instruction,
    format_directed_phrase,
    format_nearby_context_comment,
    landmark_cell_from_anchor_bbox,
    merge_nearby_hints,
    parse_nearby_hint_string,
    side_from_anchor_bbox,
)


# Anchor bbox (x, y, w, h) → edges x1=10, y1=20, x2=40, y2=50
_ANCHOR = (10, 20, 30, 30)


def test_landmark_cell_nine_sections() -> None:
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 25, 35) == LandmarkCell.CENTER
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 5, 35) == LandmarkCell.LEFT
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 45, 35) == LandmarkCell.RIGHT
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 25, 10) == LandmarkCell.ABOVE
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 25, 60) == LandmarkCell.BELOW
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 5, 10) == LandmarkCell.UPPER_LEFT
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 45, 10) == LandmarkCell.UPPER_RIGHT
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 5, 60) == LandmarkCell.LOWER_LEFT
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 45, 60) == LandmarkCell.LOWER_RIGHT


def test_landmark_cell_edges_belong_to_center_band() -> None:
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 10, 35) == LandmarkCell.CENTER
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 40, 35) == LandmarkCell.CENTER
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 25, 20) == LandmarkCell.CENTER
    assert landmark_cell_from_anchor_bbox(_ANCHOR, 25, 50) == LandmarkCell.CENTER


def test_side_inversion_table() -> None:
    assert side_from_anchor_bbox(_ANCHOR, 45, 35) == Side.LEFT  # landmark right
    assert side_from_anchor_bbox(_ANCHOR, 5, 35) == Side.RIGHT
    assert side_from_anchor_bbox(_ANCHOR, 25, 10) == Side.BELOW  # landmark above
    assert side_from_anchor_bbox(_ANCHOR, 25, 60) == Side.ABOVE
    assert side_from_anchor_bbox(_ANCHOR, 45, 10) == Side.LOWER_LEFT
    assert side_from_anchor_bbox(_ANCHOR, 5, 10) == Side.LOWER_RIGHT
    assert side_from_anchor_bbox(_ANCHOR, 45, 60) == Side.UPPER_LEFT
    assert side_from_anchor_bbox(_ANCHOR, 5, 60) == Side.UPPER_RIGHT
    assert side_from_anchor_bbox(_ANCHOR, 25, 35) is None


def test_anchor_satisfies_side_round_trip() -> None:
    assert anchor_satisfies_side(_ANCHOR, 45, 35, Side.LEFT)
    assert not anchor_satisfies_side(_ANCHOR, 5, 35, Side.LEFT)


def test_format_and_parse_directed_phrase() -> None:
    phrase = format_directed_phrase("「顯示已授權電腦」文字", Side.LEFT)
    assert phrase == "在「顯示已授權電腦」文字的左邊"
    hint = parse_nearby_hint_string(phrase)
    assert hint == NearbyHint(label="「顯示已授權電腦」文字", side=Side.LEFT)


def test_parse_nearby_hint_string_undirected() -> None:
    assert parse_nearby_hint_string("「Edge」圖示") == NearbyHint(
        label="「Edge」圖示", side=None
    )


def test_format_nearby_context_comment_mixed() -> None:
    comment = format_nearby_context_comment(
        [
            NearbyHint("「A」文字", Side.LEFT),
            NearbyHint("「B」圖示", None),
        ]
    )
    assert comment == "（在「A」文字的左邊、附近有「B」圖示）"


def test_format_nearby_context_comment_drag_locations() -> None:
    assert format_nearby_context_comment(
        [NearbyHint("「OneNote」文字", None)],
        location="起點",
    ) == "（起點附近有「OneNote」文字）"
    assert format_nearby_context_comment(
        [NearbyHint("「Desktop」文字", Side.RIGHT)],
        location="終點",
    ) == "（終點在「Desktop」文字的右邊）"


def test_extract_nearby_hints_from_instruction() -> None:
    hints = extract_nearby_hints_from_instruction(
        "將滑鼠移到「矩形框線」圖示（在「顯示已授權電腦」文字的左邊、附近有「顯示未安裝電腦」文字）"
    )
    assert hints == [
        NearbyHint("「顯示已授權電腦」文字", Side.LEFT),
        NearbyHint("「顯示未安裝電腦」文字", None),
    ]


def test_merge_nearby_hints_prefers_earlier_and_upgrades_side() -> None:
    merged = merge_nearby_hints(
        ["「Edge」圖示"],
        [NearbyHint("「Edge」圖示", Side.LEFT), "「Chrome」圖示"],
    )
    assert merged == [
        NearbyHint("「Edge」圖示", Side.LEFT),
        NearbyHint("「Chrome」圖示", None),
    ]


def test_enrich_nearby_objects_from_goal_restores_stripped_sides() -> None:
    goal = "將滑鼠移到輸入欄（在「joseph」文字的下面、在「確定」文字的上面），並點擊滑鼠一下。"
    enriched = enrich_nearby_objects_from_goal(
        goal,
        ["「joseph」文字", "「確定」文字"],
    )
    assert enriched == [
        "在「joseph」文字的下面",
        "在「確定」文字的上面",
    ]


def test_enrich_nearby_objects_from_goal_injects_when_nearby_omitted() -> None:
    goal = "將滑鼠移到輸入欄（在「joseph」文字的下面、在「確定」文字的上面）"
    assert enrich_nearby_objects_from_goal(goal, None) == [
        "在「joseph」文字的下面",
        "在「確定」文字的上面",
    ]


def test_enrich_nearby_objects_from_goal_keeps_extra_llm_landmarks() -> None:
    goal = "點擊按鈕（在「確定」文字的上面）"
    enriched = enrich_nearby_objects_from_goal(
        goal,
        ["「確定」文字", "「取消」文字"],
    )
    assert enriched == [
        "在「確定」文字的上面",
        "「取消」文字",
    ]


def test_enrich_nearby_objects_from_goal_noop_without_directed_sides() -> None:
    goal = "將滑鼠移到輸入欄並點擊"
    assert enrich_nearby_objects_from_goal(goal, ["「帳號」文字"]) == ["「帳號」文字"]
    assert enrich_nearby_objects_from_goal(goal, None) is None


def test_enrich_tool_arguments_from_goal_move_mouse() -> None:
    goal = "將滑鼠移到輸入欄（在「joseph」文字的下面、在「確定」文字的上面）"
    enriched = enrich_tool_arguments_from_goal(
        "move_mouse",
        {
            "instruction": "輸入欄",
            "nearby_objects": ["「joseph」文字", "「確定」文字"],
        },
        goal,
    )
    assert enriched["instruction"] == "輸入欄"
    assert enriched["nearby_objects"] == [
        "在「joseph」文字的下面",
        "在「確定」文字的上面",
    ]


def test_enrich_tool_arguments_from_goal_drag() -> None:
    goal = "從圖示拖到資料夾（在「Desktop」文字的右邊）"
    enriched = enrich_tool_arguments_from_goal(
        "drag",
        {
            "start_instruction": "圖示",
            "destination_instruction": "資料夾",
            "destination_nearby_objects": ["「Desktop」文字"],
        },
        goal,
    )
    assert enriched["destination_nearby_objects"] == ["在「Desktop」文字的右邊"]
    assert "start_nearby_objects" not in enriched


def test_enrich_tool_arguments_from_goal_drag_upgrades_matching_labels_only() -> None:
    goal = "從圖示拖到資料夾（起點在「Chrome」圖示的左邊、終點在「Desktop」文字的右邊）"
    enriched = enrich_tool_arguments_from_goal(
        "drag",
        {
            "start_nearby_objects": ["「Chrome」圖示"],
            "destination_nearby_objects": ["「Desktop」文字"],
        },
        goal,
    )
    assert enriched["start_nearby_objects"] == ["在「Chrome」圖示的左邊"]
    assert enriched["destination_nearby_objects"] == ["在「Desktop」文字的右邊"]


def test_enrich_tool_arguments_from_goal_ignores_other_tools() -> None:
    args = {"button": "left", "instruction": "輸入欄"}
    assert enrich_tool_arguments_from_goal("click", args, "（在「a」的下面）") == args
