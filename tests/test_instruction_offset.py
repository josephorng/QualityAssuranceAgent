from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cua_mcp.instruction_offset import (
    _parse_relative_pixel_offset_regex,
    parse_mouse_target_instruction,
)
from src.recorder.vision_context import format_drag_relative_offset_phrase


def test_parse_relative_pixel_offset_regex_right_and_up() -> None:
    anchor, dx, dy = _parse_relative_pixel_offset_regex(
        "「振銓」文字右方5個像素、上方28個像素的位置"
    )
    assert anchor == "「振銓」文字"
    assert dx == 5
    assert dy == -28


def test_parse_relative_pixel_offset_regex_down_only() -> None:
    anchor, dx, dy = _parse_relative_pixel_offset_regex(
        "「iniseape」文字下方57個像素的位置"
    )
    assert anchor == "「iniseape」文字"
    assert dx == 0
    assert dy == 57


def test_parse_relative_pixel_offset_regex_no_offset() -> None:
    anchor, dx, dy = _parse_relative_pixel_offset_regex("「Chrome」圖示")
    assert anchor == "「Chrome」圖示"
    assert dx == 0
    assert dy == 0


def test_parse_relative_pixel_offset_regex_round_trip_with_recorder_phrase() -> None:
    phrase = format_drag_relative_offset_phrase(12, 49)
    assert phrase is not None
    anchor, dx, dy = _parse_relative_pixel_offset_regex(f"「Desktop」文字{phrase}的位置")
    assert anchor == "「Desktop」文字"
    assert dx == 12
    assert dy == 49


def test_parse_relative_pixel_offset_regex_strips_trailing_nearby_comment() -> None:
    anchor, dx, dy = _parse_relative_pixel_offset_regex(
        "從「Chrome」圖示（起點附近有「Desktop」文字）拖到「OneNote」文字左方8個像素、下方58個像素的位置"
        "（終點附近有「Recycle Bin」圖示）"
    )
    assert anchor == "從「Chrome」圖示拖到「OneNote」文字"
    assert dx == -8
    assert dy == 58


@pytest.mark.asyncio
async def test_parse_relative_pixel_offset_uses_llm() -> None:
    from src.common.nearby_side import NearbyHint

    with patch(
        "cua_mcp.instruction_offset.request_json_with_retry",
        new=AsyncMock(
            return_value=(
                "「振銓」文字",
                5,
                -28,
                [NearbyHint(label="「圖片」文字", side=None)],
            ),
        ),
    ) as mock_request:
        anchor, dx, dy, nearby = await parse_mouse_target_instruction(
            "「振銓」文字右方5個像素、上方28個像素的位置（附近有「圖片」文字）"
        )

    assert anchor == "「振銓」文字"
    assert dx == 5
    assert dy == -28
    assert nearby == [NearbyHint(label="「圖片」文字", side=None)]
    mock_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_relative_pixel_offset_passes_raw_instruction_to_llm() -> None:
    from src.common.nearby_side import NearbyHint

    raw = "點擊「文件」文字（附近有「圖片」文字、「下載」文字）"
    with patch(
        "cua_mcp.instruction_offset.request_json_with_retry",
        new=AsyncMock(
            return_value=(
                "「文件」文字",
                0,
                0,
                [
                    NearbyHint(label="「圖片」文字", side=None),
                    NearbyHint(label="「下載」文字", side=None),
                ],
            ),
        ),
    ) as mock_request:
        await parse_mouse_target_instruction(raw)

    messages = mock_request.await_args.kwargs["messages"]
    assert raw in messages[0]["content"]
    assert "附近有「圖片」文字" in messages[0]["content"]


@pytest.mark.asyncio
async def test_parse_relative_pixel_offset_empty_instruction_skips_llm() -> None:
    with patch(
        "cua_mcp.instruction_offset.request_json_with_retry",
        new=AsyncMock(),
    ) as mock_request:
        anchor, dx, dy, nearby = await parse_mouse_target_instruction("   ")

    assert anchor == ""
    assert dx == 0
    assert dy == 0
    assert nearby == []
    mock_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_parse_relative_pixel_offset_falls_back_to_regex() -> None:
    from src.common.nearby_side import NearbyHint, Side

    with patch(
        "cua_mcp.instruction_offset.request_json_with_retry",
        new=AsyncMock(side_effect=ValueError("bad llm reply")),
    ):
        anchor, dx, dy, nearby = await parse_mouse_target_instruction(
            "「iniseape」文字下方57個像素的位置（附近有「圖片」文字）"
        )

    assert anchor == "「iniseape」文字"
    assert dx == 0
    assert dy == 57
    assert nearby == [NearbyHint(label="「圖片」文字", side=None)]

    with patch(
        "cua_mcp.instruction_offset.request_json_with_retry",
        new=AsyncMock(side_effect=ValueError("bad llm reply")),
    ):
        _, _, _, directed = await parse_mouse_target_instruction(
            "「矩形框線」圖示（在「顯示已授權電腦」文字的左邊）"
        )
    assert directed == [
        NearbyHint(label="「顯示已授權電腦」文字", side=Side.LEFT),
    ]
