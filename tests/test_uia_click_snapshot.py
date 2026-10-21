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


def test_detect_uses_prefetched_snapshot_and_skips_findall(
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
        "cua_mcp.select_mouse_target._recognize_ocr_boxes",
        lambda _bgr, boxes, _modes, **_k: ([[""] for _ in boxes], 0, len(boxes)),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._retry_empty_unknown_icon_ocr",
        lambda _bgr, cands: cands,
    )

    def _boom(*_a, **_k):
        raise AssertionError("analysis-time FindAll must not run")

    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.try_uia_refine_inputs_and_scrollbars",
        _boom,
    )
    found = _detect_mouse_targets_from_bgr(
        bgr,
        refine_with_uia=True,
        live_input_hwnd=99,
        live_input_monitor_offset=(0, 0),
        live_input_window_screen_xywh=(0, 0, 120, 80),
        uia_prefetched=(
            None,
            None,
            [((10, 10, 30, 20), "應用程式及其他")],
        ),
        ocr_class_ids=frozenset({YOLO_CLASS_TEXT}),
    )
    assert any(det.text == "應用程式及其他" and det.source == "uia" for det in found)


def test_detect_without_snapshot_still_calls_live_enumerator(
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
        "cua_mcp.select_mouse_target._recognize_ocr_boxes",
        lambda *_a, **_k: ([["舊"]], 0, 1),
    )
    monkeypatch.setattr(
        "cua_mcp.select_mouse_target._retry_empty_unknown_icon_ocr",
        lambda _bgr, cands: cands,
    )
    calls: list[int] = []

    def _fake_refine(*_a, **_k):
        calls.append(1)
        return None, None, [], {"uia_inputs": 0, "uia_texts": 0, "yolo_inputs": 0,
                                 "covered": False, "yolo_covered": False, "used": "hough",
                                 "uia_scrollbars": 0, "yolo_scrollbars": 0,
                                 "scrollbar_covered": False, "scrollbar_used": "yolo"}

    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.try_uia_refine_inputs_and_scrollbars",
        _fake_refine,
    )
    _detect_mouse_targets_from_bgr(
        bgr,
        refine_with_uia=True,
        live_input_hwnd=99,
        live_input_monitor_offset=(0, 0),
        live_input_window_screen_xywh=(0, 0, 120, 80),
        ocr_class_ids=frozenset({YOLO_CLASS_TEXT}),
    )
    assert calls == [1]


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
    prefetched = seen.get("uia_prefetched")
    assert prefetched is not None
    assert prefetched[2][0][1] == "應用程式及其他"
    assert prefetched[2][0][0] == (240, 8, 44, 44)


def test_click_starts_findall_off_the_capture_worker(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from src.recorder.capture import RecordingSession, _DeferredCaptureJob
    from src.recorder.window_snapshot import ClickWindowInfo

    snapshot = {
        "window": [0, 0, 100, 80],
        "inputs": [],
        "scrollbars": [],
        "texts": [{"bbox": [10, 10, 40, 30], "text": "應用程式及其他"}],
    }
    window = ClickWindowInfo(
        hwnd=132566,
        title="Microsoft 365 Copilot",
        process_name="M365Copilot.exe",
        left=0,
        top=0,
        width=100,
        height=80,
        is_maximized=False,
    )
    calls: list[int] = []

    def _fake_shot(_x: int, _y: int, dest: Path) -> tuple[str, int, tuple[int, int]]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")
        return str(dest), 1, (0, 0)

    def _snap(hwnd: int) -> dict:
        calls.append(int(hwnd))
        return snapshot

    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._uia_executor = ThreadPoolExecutor(max_workers=1)
    session._start_window_step = lambda _item: None  # type: ignore[method-assign]
    session._notify_event = lambda _event: None  # type: ignore[method-assign]
    session._start_settle_probe = lambda _event: None  # type: ignore[method-assign]
    session._left_press_seq = 1
    session._left_press_captures[1] = (None, [])
    try:
        with patch(
            "src.recorder.capture._capture_screenshot_at_point",
            side_effect=_fake_shot,
        ), patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            side_effect=_snap,
        ):
            session._begin_click_uia(1, window, side="left")
            session._left_press_uia_futures[1].result(timeout=2)
            session._worker_pending_left_press(
                _DeferredCaptureJob(
                    action="pending_left_press",
                    press_seq=1,
                    cursor_xy=(20, 20),
                )
            )
            assert calls == [132566]
            session._worker_emit_left_gesture(
                _DeferredCaptureJob(
                    action="emit_left_gesture",
                    kind="click",
                    press_seq=1,
                    cursor_xy=(20, 20),
                    event_index=1,
                    timestamp_utc="2026-10-21T00:40:00+00:00",
                    button="left",
                )
            )
            assert calls == [132566]
    finally:
        session._uia_executor.shutdown(wait=True)
    raw = json.loads((tmp_path / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["uia_snapshot"]["texts"][0]["text"] == "應用程式及其他"


def test_emit_notifies_after_findall_finishes(tmp_path: Path) -> None:
    from concurrent.futures import Future

    from src.recorder.capture import RecordingSession, _DeferredCaptureJob

    snapshot = {
        "window": [0, 0, 10, 10],
        "inputs": [],
        "scrollbars": [],
        "texts": [{"bbox": [0, 0, 4, 4], "text": "應用程式及其他"}],
    }
    pending: Future[dict | None] = Future()
    notified: list = []
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._start_window_step = lambda _item: None  # type: ignore[method-assign]
    session._start_settle_probe = lambda _event: None  # type: ignore[method-assign]
    session.set_on_event(notified.append)
    session._left_press_seq = 1
    session._left_press_captures[1] = (("pending.jpeg", 1, (0, 0)), [])
    session._left_press_uia_futures[1] = pending
    with patch(
        "src.recorder.capture._finalize_screenshot",
        return_value=("shot.jpeg", 1, (0, 0)),
    ):
        session._worker_emit_left_gesture(
            _DeferredCaptureJob(
                action="emit_left_gesture",
                kind="click",
                press_seq=1,
                cursor_xy=(20, 20),
                event_index=1,
                timestamp_utc="t",
                button="left",
            )
        )
        assert notified == []
        pending.set_result(snapshot)
    assert notified[0].uia_snapshot["texts"][0]["text"] == "應用程式及其他"
    raw = json.loads((tmp_path / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert raw["uia_snapshot"]["texts"][0]["text"] == "應用程式及其他"


def test_drag_emit_stores_release_snapshot(tmp_path: Path) -> None:
    from src.recorder.capture import RecordingSession, _DeferredCaptureJob
    from src.recorder.window_snapshot import ClickWindowInfo

    press = {
        "window": [0, 0, 40, 40],
        "inputs": [],
        "scrollbars": [],
        "texts": [{"bbox": [0, 0, 10, 10], "text": "按下"}],
    }
    release = {
        "window": [80, 0, 40, 40],
        "inputs": [],
        "scrollbars": [],
        "texts": [{"bbox": [80, 0, 100, 20], "text": "放開"}],
    }
    end_window = ClickWindowInfo(
        hwnd=77,
        title="Drop",
        process_name="app.exe",
        left=80,
        top=0,
        width=40,
        height=40,
        is_maximized=False,
    )
    from concurrent.futures import Future

    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._left_press_seq = 3
    session._left_press_captures[3] = (("pending.jpeg", 1, (0, 0)), [])
    press_future: Future[dict | None] = Future()
    press_future.set_result(press)
    release_future: Future[dict | None] = Future()
    release_future.set_result(release)
    session._left_press_uia_futures[3] = press_future
    session._drag_end_uia_futures[3] = release_future
    queued: list = []
    session._persist_queued_event = queued.append  # type: ignore[method-assign]
    with patch(
        "src.recorder.capture._finalize_screenshot",
        return_value=("shot.jpeg", 1, (0, 0)),
    ), patch(
        "src.recorder.capture._finalize_drag_end_screenshot",
        return_value=("end.jpeg", 1, (0, 0)),
    ), patch(
        "src.recorder.capture._release_monitor_index",
        return_value=None,
    ), patch.object(
        session,
        "_uia_snapshot_for_window",
        side_effect=AssertionError("capture worker must not FindAll"),
    ):
        session._worker_emit_left_gesture(
            _DeferredCaptureJob(
                action="emit_left_gesture",
                kind="drag",
                press_seq=3,
                cursor_xy=(10, 10),
                end_xy=(90, 10),
                event_index=2,
                timestamp_utc="t",
                button="left",
                end_click_window=end_window,
            )
        )
    assert queued[0].uia_snapshot["texts"][0]["text"] == "按下"
    assert queued[0].end_uia_snapshot["texts"][0]["text"] == "放開"
