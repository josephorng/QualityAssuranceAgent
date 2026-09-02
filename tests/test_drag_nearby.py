from __future__ import annotations

import pytest

from src.common.runtime_context import SMART_MODE_ENV


@pytest.mark.asyncio
async def test_drag_forwards_start_and_destination_nearby(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp import tool_module

    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    calls: list[tuple[str, list[str] | None]] = []

    async def fake_resolve(instruction: str, nearby_objects: list[str] | None = None):
        calls.append((instruction, None if nearby_objects is None else list(nearby_objects)))
        if instruction.startswith("start"):
            return 10, 20, {
                "target_kind": "text",
                "target_text": "start",
                "target_icons": [],
                "target_bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
            }
        return 30, 40, {
            "target_kind": "text",
            "target_text": "dest",
            "target_icons": [],
            "target_bbox": {"x": 2, "y": 2, "w": 1, "h": 1},
        }

    monkeypatch.setattr(tool_module, "resolve_mouse_point", fake_resolve)
    monkeypatch.setattr(
        tool_module,
        "_drag_at_points",
        lambda x1, y1, x2, y2, duration=0.5, button="left": {
            "ok": True,
            "action": "drag",
            "args": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": duration, "button": button},
        },
    )

    result = await tool_module._drag(
        start_instruction="start-anchor",
        destination_instruction="dest-anchor",
        start_nearby_objects=["「Desktop」文字"],
        destination_nearby_objects=["「新增文字文件txt」文字"],
    )

    assert calls == [
        ("start-anchor", ["「Desktop」文字"]),
        ("dest-anchor", ["「新增文字文件txt」文字"]),
    ]
    assert result["start_nearby_objects_arg"] == ["「Desktop」文字"]
    assert result["destination_nearby_objects_arg"] == ["「新增文字文件txt」文字"]
    assert result["args"]["x1"] == 10
    assert result["args"]["y2"] == 40


@pytest.mark.asyncio
async def test_smart_mode_drag_uses_visual_mouse_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp import tool_module

    monkeypatch.setenv(SMART_MODE_ENV, "1")
    visual_calls: list[str] = []
    mouse_calls: list[str] = []

    async def fake_visual(instruction: str):
        visual_calls.append(instruction)
        if instruction.startswith("start"):
            return 11, 22, {
                "target_kind": "icon",
                "target_text": "start",
                "target_icons": [],
                "target_bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
            }
        return 33, 44, {
            "target_kind": "text",
            "target_text": "dest",
            "target_icons": [],
            "target_bbox": {"x": 2, "y": 2, "w": 1, "h": 1},
        }

    async def fake_resolve(instruction: str, nearby_objects: list[str] | None = None):
        mouse_calls.append(instruction)
        raise AssertionError("resolve_mouse_point should not be used in smart mode")

    monkeypatch.setattr(tool_module, "resolve_visual_mouse_point", fake_visual)
    monkeypatch.setattr(tool_module, "resolve_mouse_point", fake_resolve)
    monkeypatch.setattr(
        tool_module,
        "_drag_at_points",
        lambda x1, y1, x2, y2, duration=0.5, button="left": {
            "ok": True,
            "action": "drag",
            "args": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": duration, "button": button},
        },
    )

    result = await tool_module._drag(
        start_instruction="start-icon",
        destination_instruction="dest-folder",
        start_nearby_objects=["ignored-in-smart"],
        destination_nearby_objects=["also-ignored"],
    )

    assert visual_calls == ["start-icon", "dest-folder"]
    assert mouse_calls == []
    assert result["args"]["x1"] == 11
    assert result["args"]["y2"] == 44
    assert result["target_text"] == "dest"


@pytest.mark.asyncio
async def test_drag_pins_scrollbar_track_percent_to_start_scrollbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp import tool_module
    from cua_mcp.scrollbar_arrows import point_from_scrollbar_percent

    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    scrollbar_bbox = {"x": 1886, "y": 268, "w": 15, "h": 231}
    bbox_tuple = (1886, 268, 15, 231)
    calls: list[tuple[str, list[str] | None]] = []

    async def fake_resolve(instruction: str, nearby_objects: list[str] | None = None):
        calls.append((instruction, None if nearby_objects is None else list(nearby_objects)))
        if "11%" in instruction:
            return 1893, 293, {
                "class_name": "scrollbar",
                "target_kind": "scrollbar",
                "target_text": "",
                "target_icons": [],
                "target_bbox": dict(scrollbar_bbox),
                "track_percent": 11,
            }
        raise AssertionError(
            "destination must not be independently resolved for track-percent drags"
        )

    monkeypatch.setattr(tool_module, "resolve_mouse_point", fake_resolve)
    monkeypatch.setattr(
        tool_module,
        "_drag_at_points",
        lambda x1, y1, x2, y2, duration=0.5, button="left": {
            "ok": True,
            "action": "drag",
            "args": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": duration, "button": button},
        },
    )

    result = await tool_module._drag(
        start_instruction="滾動條的11%處",
        destination_instruction="滾動條的100%處",
        start_nearby_objects=["「2024/10/16 10:13:25」文字的右邊"],
        destination_nearby_objects=["「10.40.7.104 <DHCP>」文字的右邊"],
    )

    expected_x2, expected_y2 = point_from_scrollbar_percent(bbox_tuple, 100)
    assert len(calls) == 1
    assert result["args"]["x1"] == 1893
    assert result["args"]["y1"] == 293
    assert result["args"]["x2"] == expected_x2
    assert result["args"]["y2"] == expected_y2
    assert result["destination_target"]["track_percent"] == 100
    assert result["destination_target"]["target_bbox"] == scrollbar_bbox


@pytest.mark.asyncio
async def test_drag_falls_back_when_start_is_not_scrollbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp import tool_module

    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    calls: list[str] = []

    async def fake_resolve(instruction: str, nearby_objects: list[str] | None = None):
        calls.append(instruction)
        if "11%" in instruction:
            return 10, 20, {
                "class_name": "text",
                "target_kind": "text",
                "target_text": "foo",
                "target_icons": [],
                "target_bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
            }
        return 30, 40, {
            "class_name": "text",
            "target_kind": "text",
            "target_text": "bar",
            "target_icons": [],
            "target_bbox": {"x": 2, "y": 2, "w": 1, "h": 1},
        }

    monkeypatch.setattr(tool_module, "resolve_mouse_point", fake_resolve)
    monkeypatch.setattr(
        tool_module,
        "_drag_at_points",
        lambda x1, y1, x2, y2, duration=0.5, button="left": {
            "ok": True,
            "action": "drag",
            "args": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": duration, "button": button},
        },
    )

    await tool_module._drag(
        start_instruction="滾動條的11%處",
        destination_instruction="滾動條的100%處",
    )

    assert len(calls) == 2
