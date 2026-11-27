"""Click-time UIA snapshots are reused at analysis and skip a later FindAll."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cua_mcp.yolo_onnx import YOLO_CLASS_TEXT
from src.recorder.uia_input_boxes import uia_snapshot_image_local


def test_uia_snapshot_image_local_does_not_query_hwnd(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_hwnd: int):
        raise AssertionError("FindAll must not run for a stored snapshot")

    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.enumerate_edit_document_scrollbar_screen",
        _boom,
    )
    inputs, scrollbars, texts = uia_snapshot_image_local(
        {
            "window": [0, 0, 400, 300],
            "inputs": [],
            "scrollbars": [],
            "texts": [{"bbox": [240, 8, 284, 52], "text": "應用程式及其他"}],
        },
        monitor_offset=(0, 0),
        image_w=400,
        image_h=300,
    )
    assert inputs is None
    assert scrollbars is None
    assert texts
    assert texts[0][1] == "應用程式及其他"
    assert texts[0][0] == (240, 8, 44, 44)
    assert texts[0][2] == ""


def test_uia_snapshot_image_local_keeps_control_type() -> None:
    inputs, scrollbars, texts = uia_snapshot_image_local(
        {
            "window": [0, 0, 400, 300],
            "inputs": [],
            "scrollbars": [],
            "texts": [
                {
                    "bbox": [240, 8, 284, 52],
                    "text": "應用程式及其他",
                    "control_type": "list_item",
                },
                {
                    "bbox": [10, 10, 40, 30],
                    "text": "舊標籤",
                },
            ],
        },
        monitor_offset=(0, 0),
        image_w=400,
        image_h=300,
    )
    assert inputs is None
    assert scrollbars is None
    by_text = {label: (box, control_type) for box, label, control_type in texts}
    assert by_text["應用程式及其他"][1] == "list_item"
    assert by_text["舊標籤"][1] == ""


def test_uia_snapshot_image_local_keeps_input_name_automation_id() -> None:
    inputs, scrollbars, texts = uia_snapshot_image_local(
        {
            "window": [0, 0, 400, 300],
            "inputs": [[50, 80, 200, 110]],
            "input_labels": ["傳真號碼|txt_Fax"],
            "scrollbars": [],
            "texts": [],
        },
        monitor_offset=(0, 0),
        image_w=400,
        image_h=300,
    )
    assert scrollbars is None
    assert texts == []
    assert inputs is not None
    assert len(inputs) == 1
    box, label, _ct = inputs[0]
    assert box == (50, 80, 150, 30)
    assert label == "傳真號碼|txt_Fax"

    from cua_mcp.select_mouse_target import _detections_from_uia_prefetched

    dets = _detections_from_uia_prefetched((inputs, None, []))
    assert len(dets) == 1
    assert dets[0].class_name == "input"
    assert dets[0].text == "傳真號碼|txt_Fax"


def test_detect_uses_prefetched_snapshot_and_keeps_ocr_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FindAll alone is not a hub; usable EFP opens UIA with YOLO twin nearby."""
    from cua_mcp.select_mouse_target import _detection_from_bbox
    from src.recorder.vision_context import (
        VISION_PATH_EFP_MISS_REASON,
        _detect_recording_mouse_targets,
    )

    bgr = np.zeros((80, 120, 3), dtype=np.uint8)
    yolo_calls: list[str] = []
    ocr = _detection_from_bbox(
        (0, 40, 20, 3), YOLO_CLASS_TEXT, text="列00", source="ocr"
    )

    def _fake_yolo(*_a, **_k):
        yolo_calls.append("yolo")
        return [ocr]

    monkeypatch.setattr(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        _fake_yolo,
    )
    labeled = [((10, 10, 30, 20), "應用程式及其他")] + [
        ((0, 40 + i * 4, 20, 3), f"列{i:02d}") for i in range(8)
    ]
    # Without EFP: FindAll snapshot must not open UIA path.
    found_yolo, timing_yolo = _detect_recording_mouse_targets(
        bgr,
        local_x=20,
        local_y=16,
        uia_prefetched=(None, None, labeled),
    )
    assert yolo_calls == ["yolo"]
    assert timing_yolo["vision_path"] == "yolo_ocr"
    assert timing_yolo.get("vision_path_reason") == VISION_PATH_EFP_MISS_REASON
    assert [d.text for d in found_yolo] == ["列00"]

    efp = {
        "name": "應用程式及其他",
        "value": "",
        "control_type": "list_item",
        "bbox": [10, 10, 30, 20],
    }
    found, timing = _detect_recording_mouse_targets(
        bgr,
        local_x=20,
        local_y=16,
        uia_prefetched=(None, None, labeled),
        element_from_point=efp,
        efp_monitor_offset=(0, 0),
    )
    assert timing["vision_path"] == "uia"
    assert timing.get("recording_uia_sufficient") is True
    assert "yolo_ocr_fallback_detections" in timing
    assert found[0].text == "應用程式及其他"
    assert found[0].source == "uia_efp"
    assert any(det.text == "列00" and det.source == "ocr" for det in found)


def test_detect_without_snapshot_returns_only_yolo_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cua_mcp.select_mouse_target import _detect_mouse_targets_from_bgr

    bgr = np.zeros((80, 120, 3), dtype=np.uint8)
    xyxy = np.asarray([[10, 10, 40, 30]], dtype=np.float32)
    scores = np.asarray([0.9], dtype=np.float32)
    class_ids = np.asarray([YOLO_CLASS_TEXT], dtype=np.int32)
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.run_yolo_onnx_end2end",
        lambda *_a, **_k: (xyxy, scores, class_ids),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target.merge_yolo_inputs_with_line_rectangles",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.list_uia_input_and_scrollbar_boxes_image_local",
        lambda *_a, **_k: ([], [], []),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._recognize_ocr_boxes",
        lambda *_a, **_k: ([["舊"]], 0, 1),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._retry_empty_unknown_icon_ocr",
        lambda _bgr, cands: cands,
    )
    found = _detect_mouse_targets_from_bgr(
        bgr,
        refine_with_uia=True,
        live_input_hwnd=99,
        live_input_monitor_offset=(0, 0),
        live_input_window_screen_xywh=(0, 0, 120, 80),
        ocr_class_ids=frozenset({YOLO_CLASS_TEXT}),
    )
    assert any(det.text == "舊" and det.source != "uia" for det in found)


def test_analysis_prefers_click_snapshot_over_live_hwnd(tmp_path: Path) -> None:
    from src.recorder.models import RecordedEvent
    from src.recorder.vision_context import build_vision_context_at_point
    from src.recorder.window_snapshot import ClickWindowInfo

    run_dir = tmp_path / "snap"
    (run_dir / "screenshots").mkdir(parents=True)
    (run_dir / "screenshots" / "event_001.jpeg").write_bytes(b"x")
    event = RecordedEvent(
        index=1,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(20, 20),
        screenshot_path=str(run_dir / "screenshots" / "event_001.jpeg"),
        monitor_offset=(0, 0),
        click_window={
            "hwnd": 132566,
            "title": "Microsoft 365 Copilot",
            "rect": [0, 0, 400, 300],
            "is_maximized": True,
        },
        uia_snapshot={
            "window": [0, 0, 400, 300],
            "inputs": [],
            "scrollbars": [],
            "texts": [{"bbox": [240, 8, 284, 52], "text": "應用程式及其他"}],
        },
    )
    seen: dict = {}

    def fake_detect(_bgr, **kwargs):
        seen.update(kwargs)
        return []

    live = ClickWindowInfo(
        hwnd=132566,
        title="Microsoft 365 Copilot",
        process_name="M365Copilot.exe",
        left=0,
        top=0,
        width=400,
        height=300,
        is_maximized=True,
    )
    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((300, 400, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_detect,
    ), patch(
        "src.recorder.window_snapshot.find_matching_click_window",
        return_value=live,
    ):
        build_vision_context_at_point(
            event,
            local_x=262,
            local_y=30,
            run_dir=run_dir,
            persist_debug=False,
        )
    assert seen.get("live_input_hwnd") is None
    # Snapshot is passed into YOLO fallback for input/scrollbar prefer-replace.
    assert seen.get("uia_prefetched") is not None
    assert seen.get("refine_with_uia") is True


def _copilot_window(hwnd: int = 132566) -> object:
    from src.recorder.window_snapshot import ClickWindowInfo

    return ClickWindowInfo(
        hwnd=hwnd,
        title="Microsoft 365 Copilot",
        process_name="M365Copilot.exe",
        left=0,
        top=0,
        width=120,
        height=80,
        is_maximized=False,
    )




def test_recording_press_leaves_uia_snapshot_empty(tmp_path: Path) -> None:
    """Phase 2: press latch/worker must not attach FindAll snapshots."""
    from src.recorder.capture import RecordingSession, _DeferredCaptureJob

    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._accepting_input = True
    with (
        patch.object(session, "_click_window_payload_at", return_value=_copilot_window()),
        patch.object(session, "_cached_windows_before", return_value=()),
        patch.object(session, "_cached_signals_before", return_value=None),
        patch.object(session, "_seal_open_after_cache"),
        patch.object(session, "_latch_settle_frame", return_value=(None, None)),
        patch.object(session, "_enqueue"),
        patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
        ) as snap,
        patch(
            "src.recorder.uia_input_boxes.element_from_point_name_value",
            return_value={"name": "關閉", "value": "", "control_type": "button"},
        ),
        patch.object(
            session,
            "_pending_screenshot_from_settle_or_capture",
            return_value=None,
        ),
    ):
        session._capture_pending_left_press(tmp_path, 20, 20)
        assert session._left_press_uia[1] is None
        session._worker_pending_left_press(
            _DeferredCaptureJob(
                action="pending_left_press",
                cursor_xy=(20, 20),
                press_seq=1,
            )
        )
    snap.assert_not_called()
    assert session._left_press_uia[1] is None


def test_recording_scroll_leaves_uia_snapshot_empty(tmp_path: Path) -> None:
    from src.recorder.capture import RecordingSession, _DeferredCaptureJob

    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._accepting_input = True
    persisted: list = []

    def _persist(item) -> None:
        persisted.append(item)

    with (
        patch.object(session, "_persist_queued_event", side_effect=_persist),
        patch.object(
            session,
            "_probe_element_from_point_at",
            return_value={"name": "Pane", "value": ""},
        ),
        patch.object(
            session,
            "_pending_screenshot_from_settle_or_capture",
            return_value=("shot.jpeg", 0, (0, 0)),
        ),
        patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
        ) as snap,
    ):
        # Use the same worker path as scroll/middle-click.
        session._worker_mouse_pointer_event(
            _DeferredCaptureJob(
                action="mouse_pointer_event",
                kind="scroll",
                cursor_xy=(20, 20),
                event_index=1,
                button="middle",
                scroll_delta=-1,
                click_window=_copilot_window(),
            )
        )
    snap.assert_not_called()
    assert len(persisted) == 1
    assert persisted[0].uia_snapshot is None
