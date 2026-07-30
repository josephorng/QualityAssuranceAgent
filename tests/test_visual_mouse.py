from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cua_mcp.screen_context import ScreenContext
from cua_mcp.select_ui_element import UiDetection
from cua_mcp.tools import TOOL_FUNCTIONS, get_mode_tool_functions
from cua_mcp.visual_mouse import resolve_visual_mouse_point
from src.common.runtime_context import RUNTIME_COMMAND_MODE_ENV, SMART_MODE_ENV


@pytest.mark.asyncio
async def test_visual_mouse_selects_candidate_in_one_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        UiDetection(
            bbox=(10, 20, 100, 30),
            cx=60,
            cy=35,
            class_id=0,
            class_name="text",
            text="Search",
        ),
        UiDetection(
            bbox=(200, 40, 300, 40),
            cx=350,
            cy=60,
            class_id=2,
            class_name="input",
            text="Ask Google or type a URL",
        ),
    ]
    context = ScreenContext(
        screenshot_paths=["current.png"],
        ocr_text=(
            "[index 0] class=文字 center=[60,35] text='Search'\n"
            "[index 1] class=輸入欄 center=[350,60] text='Ask Google or type a URL'"
        ),
        candidate_count=2,
        monitor_indices=[1],
        candidates=candidates,
    )

    calls: list[dict[str, Any]] = []

    class _FakeClient:
        async def chat_messages(self, model: str, **kwargs: Any) -> Any:
            calls.append({"model": model, **kwargs})
            return SimpleNamespace(
                content='{"index":1,"text":"browser address input"}'
            )

    async def fake_capture_screen_context(**_kwargs: Any) -> ScreenContext:
        return context

    monkeypatch.setattr(
        "cua_mcp.visual_mouse.capture_screen_context",
        fake_capture_screen_context,
    )
    monkeypatch.setattr("cua_mcp.visual_mouse.get_llm_client", lambda: _FakeClient())
    monkeypatch.setattr(
        "cua_mcp.visual_mouse.load_settings",
        lambda: SimpleNamespace(brain_lm="vision-model"),
    )

    x, y, metadata = await resolve_visual_mouse_point("browser address bar")

    assert (x, y) == (350, 60)
    assert metadata["selected_index"] == 1
    assert metadata["selection_method"] == "visual_one_pass"
    assert metadata["target_text"] == "Ask Google or type a URL"
    assert len(calls) == 1
    assert calls[0]["tools"] == []
    assert calls[0]["messages"][0]["images"] == ["current.png"]
    assert "YOLO/OCRCandidates" in calls[0]["messages"][0]["content"]


def test_visual_mouse_is_available_to_brain() -> None:
    names = {tool.__name__ for tool in TOOL_FUNCTIONS}
    assert "move_mouse_visual" in names


def test_script_mode_hides_standard_move_mouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)

    names = {tool.__name__ for tool in get_mode_tool_functions()}

    assert "move_mouse" not in names
    assert "move_mouse_visual" in names


def test_smart_mode_hides_visual_move_mouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SMART_MODE_ENV, "1")
    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)

    names = {tool.__name__ for tool in get_mode_tool_functions()}

    assert "move_mouse" in names
    assert "move_mouse_visual" not in names
