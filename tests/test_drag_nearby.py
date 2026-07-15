from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_drag_forwards_start_and_destination_nearby(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp import tool_module

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
