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
    assert "disambiguation" not in metadata
    assert len(calls) == 1
    assert calls[0]["tools"] == []
    assert calls[0]["messages"][0]["images"] == ["current.png"]
    assert "YOLO/OCRCandidates" in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_visual_mouse_runs_similar_function_describe_for_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the one-pass pick, label-similar peers trigger function-describe re-pick."""
    outlook = UiDetection(
        bbox=(2240, 20, 30, 16),
        cx=2255,
        cy=28,
        class_id=0,
        class_name="text",
        text="搜尋",
    )
    taskbar = UiDetection(
        bbox=(2550, 1040, 30, 16),
        cx=2565,
        cy=1048,
        class_id=0,
        class_name="text",
        text="搜尋",
    )
    other = UiDetection(
        bbox=(100, 100, 30, 16),
        cx=115,
        cy=108,
        class_id=0,
        class_name="text",
        text="關閉",
    )
    candidates = [outlook, taskbar, other]
    context = ScreenContext(
        screenshot_paths=["current.png"],
        ocr_text="candidates",
        candidate_count=3,
        monitor_indices=[1],
        candidates=candidates,
    )

    class _FakeClient:
        async def chat_messages(self, model: str, **kwargs: Any) -> Any:
            return SimpleNamespace(
                content='{"index":1,"text":"picked-taskbar"}'
            )

    async def fake_capture_screen_context(**_kwargs: Any) -> ScreenContext:
        return context

    async def fake_describe(anchor, peers, image_paths):
        assert peers[0] is taskbar
        assert outlook in peers
        return [
            "Windows 工作列搜尋" if d is taskbar else "Outlook 郵件搜尋欄"
            for d in peers
        ]

    async def fake_repick(anchor, peers, functions, image_paths, **_kwargs):
        assert peers[0] is taskbar
        outlook_idx = next(i for i, d in enumerate(peers) if d is outlook)
        return outlook_idx, "picked-outlook-after-describe"

    monkeypatch.setattr(
        "cua_mcp.visual_mouse.capture_screen_context",
        fake_capture_screen_context,
    )
    monkeypatch.setattr("cua_mcp.visual_mouse.get_llm_client", lambda: _FakeClient())
    monkeypatch.setattr(
        "cua_mcp.visual_mouse.load_settings",
        lambda: SimpleNamespace(brain_lm="vision-model"),
    )
    monkeypatch.setattr(
        "cua_mcp.visual_mouse.get_run_state_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type(
                        "P", (), {"yolo_ocr_dir": __import__("pathlib").Path(".")}
                    )()
                ),
            },
        )(),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._write_indexed_bbox_overlay_images",
        lambda *_a, **_k: ["shot.png"],
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._describe_ui_candidate_functions",
        fake_describe,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._select_center_with_functions",
        fake_repick,
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._log_info",
        lambda *_a, **_k: None,
    )

    x, y, metadata = await resolve_visual_mouse_point("搜尋欄位")

    assert (x, y) == (outlook.cx, outlook.cy)
    assert metadata["disambiguation"] == "similar_function_describe"
    assert metadata["similar_count"] == 2
    assert metadata["initial_selected_index"] == 1
    assert metadata["selected_text"] == "picked-outlook-after-describe"
    assert metadata["target_text"] == "搜尋"


def test_visual_mouse_is_available_to_brain() -> None:
    names = {tool.__name__ for tool in TOOL_FUNCTIONS}
    assert "move_mouse_visual" in names


def test_script_mode_hides_visual_move_mouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)

    names = {tool.__name__ for tool in get_mode_tool_functions()}

    assert "move_mouse" in names
    assert "move_mouse_visual" not in names


def test_smart_mode_hides_standard_move_mouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SMART_MODE_ENV, "1")
    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)

    names = {tool.__name__ for tool in get_mode_tool_functions()}

    assert "move_mouse" not in names
    assert "move_mouse_visual" in names
