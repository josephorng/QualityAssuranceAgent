"""Tests for Gemma ROI → YOLO-on-crop fallback and Stage A visual pick."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cua_mcp.gemma_roi_refine import (
    clamp_norm,
    crop_bgr_with_origin,
    normalized_roi_to_pixel_box,
    pick_candidate_with_gemma,
    resolve_target_via_gemma_roi,
)
from cua_mcp.screen_context import ScreenContext
from cua_mcp.select_ui_element import UiDetection
from cua_mcp.visual_mouse import resolve_visual_mouse_point


def test_clamp_norm() -> None:
    assert clamp_norm(-10) == 0
    assert clamp_norm(1000) == 1000
    assert clamp_norm(1500) == 1000
    assert clamp_norm(500.4) == 500


def test_normalized_roi_to_pixel_box_basic() -> None:
    # 0..1000 over 1000x500 → half box without pad would be 250..750 x 125..375;
    # with default pad_frac=0.08 expands by 80x40.
    box = normalized_roi_to_pixel_box(250, 250, 750, 750, 1000, 500, pad_frac=0.0)
    assert box == (250, 125, 750, 375)


def test_normalized_roi_to_pixel_box_swaps_inverted() -> None:
    box = normalized_roi_to_pixel_box(800, 800, 200, 200, 1000, 1000, pad_frac=0.0)
    assert box == (200, 200, 800, 800)


def test_normalized_roi_to_pixel_box_applies_pad() -> None:
    box = normalized_roi_to_pixel_box(500, 500, 500, 500, 1000, 1000, pad_frac=0.1)
    # Degenerate point expands to at least min side, plus pad.
    x1, y1, x2, y2 = box
    assert x1 < x2 and y1 < y2
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 1000 and y2 <= 1000


def test_crop_bgr_with_origin() -> None:
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    bgr[10:30, 40:80] = 255
    crop, ox, oy = crop_bgr_with_origin(bgr, (40, 10, 80, 30))
    assert (ox, oy) == (40, 10)
    assert crop.shape == (20, 40, 3)
    assert int(crop.max()) == 255


@pytest.mark.asyncio
async def test_pick_candidate_with_gemma(monkeypatch: pytest.MonkeyPatch) -> None:
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
            text="URL",
        ),
    ]

    class _FakeClient:
        async def chat_messages(self, model: str, **kwargs: Any) -> Any:
            return SimpleNamespace(content='{"index":1,"text":"address"}')

    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.get_llm_client", lambda: _FakeClient()
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.load_settings",
        lambda: SimpleNamespace(brain_lm="vision-model"),
    )

    idx, text, chosen = await pick_candidate_with_gemma(
        "address bar", candidates, ["shot.png"]
    )
    assert idx == 1
    assert text == "address"
    assert chosen is candidates[1]


@pytest.mark.asyncio
async def test_resolve_target_via_gemma_roi_maps_crop_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = np.zeros((200, 400, 3), dtype=np.uint8)
    # Put a white patch in the ROI region so crop is non-empty.
    full[40:80, 100:180] = 200

    roi_calls = {"n": 0}

    async def fake_roi(instruction: str, image_paths: list[str], **kwargs: Any):
        roi_calls["n"] += 1
        return {
            "image_index": 0,
            "nx1": 250,
            "ny1": 200,
            "nx2": 450,
            "ny2": 400,
        }

    crop_det = UiDetection(
        bbox=(10, 5, 40, 20),
        cx=30,
        cy=15,
        class_id=0,
        class_name="text",
        text="保管人",
    )

    def fake_detect(bgr, **kwargs):
        assert bgr.ndim == 3
        return [crop_det]

    async def fake_pick(instruction, candidates, image_paths):
        assert len(candidates) == 1
        # Crop origin from padded ROI; detect is crop-local then offset.
        chosen = candidates[0]
        return 0, "custodian", chosen

    monkeypatch.setattr("cua_mcp.gemma_roi_refine.ask_gemma_target_roi", fake_roi)
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._detect_mouse_targets_from_bgr",
        fake_detect,
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.pick_candidate_with_gemma",
        fake_pick,
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.active_monitor_offset",
        lambda _i: (1000, 500),
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.get_run_state_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type(
                        "P",
                        (),
                        {"yolo_ocr_dir": __import__("pathlib").Path(".")},
                    )()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.imwrite_bgr",
        lambda *_a, **_k: True,
    )

    x, y, meta = await resolve_target_via_gemma_roi(
        "保管人",
        image_paths=["full.png"],
        monitor_indices=[1],
        captured=[(1, full)],
        max_rounds=1,
    )

    # crop-local (30,15) + monitor-local crop origin + virtual offset (1000,500)
    assert meta["selection_method"] == "gemma_roi_yolo"
    assert meta["target_text"] == "保管人"
    assert roi_calls["n"] == 1
    # Verify global mapping: chosen.cx/cy already include offsets from resolve.
    assert x == meta["resolved_center"]["x"]
    assert y == meta["resolved_center"]["y"]
    assert x >= 1000 and y >= 500


@pytest.mark.asyncio
async def test_resolve_target_via_gemma_roi_returns_none_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = np.zeros((100, 100, 3), dtype=np.uint8)

    async def fake_roi(*_a, **_k):
        return {
            "image_index": 0,
            "nx1": 100,
            "ny1": 100,
            "nx2": 400,
            "ny2": 400,
        }

    monkeypatch.setattr("cua_mcp.gemma_roi_refine.ask_gemma_target_roi", fake_roi)
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._detect_mouse_targets_from_bgr",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.active_monitor_offset",
        lambda _i: (0, 0),
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.get_run_state_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type(
                        "P",
                        (),
                        {"yolo_ocr_dir": __import__("pathlib").Path(".")},
                    )()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.imwrite_bgr",
        lambda *_a, **_k: True,
    )

    result = await resolve_target_via_gemma_roi(
        "missing",
        image_paths=["full.png"],
        monitor_indices=[1],
        captured=[(1, full)],
        max_rounds=2,
    )
    assert result is None


@pytest.mark.asyncio
async def test_find_mouse_point_stage_a_on_similarity_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp import select_mouse_target as smt

    det = UiDetection(
        bbox=(10, 20, 40, 16),
        cx=30,
        cy=28,
        class_id=0,
        class_name="text",
        text="設定",
    )

    async def fake_parse(instruction: str):
        return ("面板詢定", 0, 0, [], None, 0, None)

    async def fake_pick(instruction, candidates, image_paths):
        assert candidates == [det]
        return 0, "fuzzy match", det

    monkeypatch.setattr(smt, "parse_mouse_target_instruction", fake_parse)
    monkeypatch.setattr(smt, "selected_eye_monitor_indices", lambda: [1])
    monkeypatch.setattr(
        smt,
        "_run_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type(
                        "P",
                        (),
                        {"yolo_ocr_dir": __import__("pathlib").Path(".")},
                    )()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )
    monkeypatch.setattr(smt, "grab_monitor_bgr", lambda *_a, **_k: (1, np.zeros((50, 50, 3), dtype=np.uint8)))
    monkeypatch.setattr(smt, "imwrite_bgr", lambda *_a, **_k: True)
    monkeypatch.setattr(
        smt,
        "_collect_monitor_detections",
        lambda *_a, **_k: [det],
    )
    monkeypatch.setattr(
        smt,
        "_filter_mouse_candidates",
        lambda *_a, **_k: ([], []),
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.pick_candidate_with_gemma",
        fake_pick,
    )

    found = await smt.find_mouse_point("「面板詢定」文字")
    assert found is not None
    x, y, meta = found
    assert (x, y) == (30, 28)
    assert meta["selection_method"] == "visual_one_pass_fallback"
    assert meta["target_text"] == "設定"


@pytest.mark.asyncio
async def test_find_mouse_point_stage_b_when_no_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp import select_mouse_target as smt

    async def fake_parse(instruction: str):
        return ("保管人", 2, 3, [], None, 0, None)

    async def fake_roi(*_a, **_k):
        return (
            100,
            200,
            {
                "selection_method": "gemma_roi_yolo",
                "target_text": "保管人",
                "target_kind": "text",
                "target_icons": [],
                "target_bbox": {"x": 90, "y": 190, "w": 20, "h": 20},
                "image_center": {"x": 100, "y": 200},
                "resolved_center": {"x": 100, "y": 200},
                "screenshot_path": "a.png",
                "screenshot_paths": ["a.png"],
                "selected_index": 0,
                "selected_text": "ok",
                "anchor_instruction": "保管人",
            },
        )

    monkeypatch.setattr(smt, "parse_mouse_target_instruction", fake_parse)
    monkeypatch.setattr(smt, "selected_eye_monitor_indices", lambda: [1])
    monkeypatch.setattr(
        smt,
        "_run_manager",
        lambda: type(
            "M",
            (),
            {
                "require_paths": staticmethod(
                    lambda: type(
                        "P",
                        (),
                        {"yolo_ocr_dir": __import__("pathlib").Path(".")},
                    )()
                ),
                "log_info": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )
    monkeypatch.setattr(smt, "grab_monitor_bgr", lambda *_a, **_k: (1, np.zeros((50, 50, 3), dtype=np.uint8)))
    monkeypatch.setattr(smt, "imwrite_bgr", lambda *_a, **_k: True)
    monkeypatch.setattr(
        smt,
        "_collect_monitor_detections",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "cua_mcp.gemma_roi_refine.resolve_target_via_gemma_roi",
        fake_roi,
    )

    found = await smt.find_mouse_point("「保管人」文字")
    assert found is not None
    x, y, meta = found
    assert (x, y) == (102, 203)  # + offset 2,3
    assert meta["selection_method"] == "gemma_roi_yolo"
    assert meta["relative_offset"] == {"dx": 2, "dy": 3}


@pytest.mark.asyncio
async def test_visual_mouse_empty_candidates_uses_roi_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ScreenContext(
        screenshot_paths=["current.png"],
        ocr_text="(no OCR/YOLO candidates)",
        candidate_count=0,
        monitor_indices=[1],
        candidates=[],
    )

    async def fake_capture(**_kwargs: Any) -> ScreenContext:
        return context

    async def fake_roi(*_a, **_k):
        return (
            50,
            60,
            {
                "selection_method": "gemma_roi_yolo",
                "selected_index": 0,
                "selected_text": "found",
                "screenshot_path": "current.png",
                "screenshot_paths": ["current.png"],
                "target_kind": "text",
                "target_text": "OK",
                "target_icons": [],
                "target_bbox": {"x": 40, "y": 50, "w": 20, "h": 20},
                "image_center": {"x": 50, "y": 60},
                "resolved_center": {"x": 50, "y": 60},
                "anchor_instruction": "OK",
            },
        )

    monkeypatch.setattr(
        "cua_mcp.visual_mouse.capture_screen_context",
        fake_capture,
    )
    monkeypatch.setattr(
        "cua_mcp.visual_mouse.resolve_target_via_gemma_roi",
        fake_roi,
    )

    x, y, meta = await resolve_visual_mouse_point("OK 按鈕")
    assert (x, y) == (50, 60)
    assert meta["selection_method"] == "gemma_roi_yolo"


@pytest.mark.asyncio
async def test_visual_mouse_empty_roi_miss_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ScreenContext(
        screenshot_paths=["current.png"],
        ocr_text="(no OCR/YOLO candidates)",
        candidate_count=0,
        monitor_indices=[1],
        candidates=[],
    )

    async def fake_capture(**_kwargs: Any) -> ScreenContext:
        return context

    async def fake_roi(*_a, **_k):
        return None

    monkeypatch.setattr(
        "cua_mcp.visual_mouse.capture_screen_context",
        fake_capture,
    )
    monkeypatch.setattr(
        "cua_mcp.visual_mouse.resolve_target_via_gemma_roi",
        fake_roi,
    )

    with pytest.raises(ValueError, match="No YOLO/OCR candidates"):
        await resolve_visual_mouse_point("missing")
