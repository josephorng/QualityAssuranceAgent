from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_check_object_exists_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp import tool_module

    move_calls: list[tuple[int, int]] = []
    click_calls: list[str] = []

    async def fake_find(instruction: str, nearby_objects: list[str] | None = None, **_kwargs):
        assert instruction == "「取代目的地中的檔案」文字"
        assert nearby_objects == ["「取消」文字"]
        return None

    monkeypatch.setattr(tool_module, "find_mouse_point", fake_find)
    monkeypatch.setattr(
        tool_module.hand_tools,
        "move",
        lambda x, y, duration=0.0: move_calls.append((x, y)) or {"ok": True, "x": x, "y": y},
    )
    monkeypatch.setattr(
        tool_module.hand_tools,
        "click",
        lambda button="left", clicks=1, interval=0.0: click_calls.append(button) or {"ok": True},
    )

    result = await tool_module._check_object_exists(
        instruction="「取代目的地中的檔案」文字",
        nearby_objects=["「取消」文字"],
    )

    assert result["exists"] is False
    assert result["instruction"] == "「取代目的地中的檔案」文字"
    assert result["nearby_objects_arg"] == ["「取消」文字"]
    assert move_calls == []
    assert click_calls == []


@pytest.mark.asyncio
async def test_check_object_exists_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp import tool_module

    move_calls: list[tuple[int, int]] = []
    click_calls: list[str] = []

    async def fake_find(instruction: str, nearby_objects: list[str] | None = None, **_kwargs):
        assert instruction == "「取代目的地中的檔案」文字"
        return 120, 340, {
            "selected_index": 0,
            "class_name": "text",
            "image_center": {"x": 120, "y": 340},
            "resolved_center": {"x": 120, "y": 340},
            "relative_offset": {"dx": 0, "dy": 0},
            "anchor_instruction": instruction,
            "nearby_objects": [],
            "screenshot_path": "shot.png",
            "screenshot_paths": ["shot.png"],
            "target_kind": "text",
            "target_text": "取代目的地中的檔案",
            "target_icons": [],
            "target_bbox": {"x": 100, "y": 330, "w": 40, "h": 20},
        }

    monkeypatch.setattr(tool_module, "find_mouse_point", fake_find)
    monkeypatch.setattr(
        tool_module.hand_tools,
        "move",
        lambda x, y, duration=0.0: move_calls.append((x, y)) or {"ok": True, "x": x, "y": y},
    )
    monkeypatch.setattr(
        tool_module.hand_tools,
        "click",
        lambda button="left", clicks=1, interval=0.0: click_calls.append(button) or {"ok": True},
    )

    result = await tool_module._check_object_exists(
        instruction="「取代目的地中的檔案」文字",
    )

    assert result["exists"] is True
    assert result["instruction"] == "「取代目的地中的檔案」文字"
    assert result["x"] == 120
    assert result["y"] == 340
    assert result["target_kind"] == "text"
    assert result["target_text"] == "取代目的地中的檔案"
    assert result["target_bbox"] == {"x": 100, "y": 330, "w": 40, "h": 20}
    assert result["screenshot_path"] == "shot.png"
    assert move_calls == []
    assert click_calls == []


def test_check_object_exists_registered_in_tool_functions() -> None:
    from cua_mcp.tools import TOOL_FUNCTIONS, check_object_exists

    assert check_object_exists in TOOL_FUNCTIONS
    assert check_object_exists.__name__ == "check_object_exists"


@pytest.mark.asyncio
async def test_resolve_mouse_point_raises_when_find_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp import select_mouse_target

    async def fake_find(instruction: str, nearby_objects: list[str] | None = None, **_kwargs):
        return None

    monkeypatch.setattr(select_mouse_target, "find_mouse_point", fake_find)

    with pytest.raises(ValueError, match="No mouse target matched"):
        await select_mouse_target.resolve_mouse_point("「不存在」文字")
