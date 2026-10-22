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


def test_detect_uses_prefetched_snapshot_and_skips_findall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.recorder.vision_context import _detect_recording_mouse_targets

    bgr = np.zeros((80, 120, 3), dtype=np.uint8)
    yolo_calls: list[str] = []

    def _boom(*_a, **_k):
        yolo_calls.append("yolo")
        raise AssertionError("YOLO must not run when usable UIA covers the click")

    monkeypatch.setattr(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        _boom,
    )
    labeled = [((10, 10, 30, 20), "應用程式及其他")] + [
        ((0, 40 + i * 4, 20, 3), f"列{i:02d}") for i in range(8)
    ]
    found, timing = _detect_recording_mouse_targets(
        bgr,
        local_x=20,
        local_y=16,
        uia_prefetched=(None, None, labeled),
    )
    assert yolo_calls == []
    assert timing["vision_path"] == "uia"
    assert any(det.text == "應用程式及其他" and det.source == "uia" for det in found)


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
    assert seen.get("uia_prefetched") is None
    assert seen.get("refine_with_uia") is True


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


def _wait_preclick(session: object) -> dict:
    import time

    for _ in range(100):
        slot = session._preclick_uia  # type: ignore[attr-defined]
        if slot is not None and isinstance(slot.snapshot, dict):
            return slot.snapshot
        time.sleep(0.02)
    raise AssertionError("pre-click FindAll did not finish")


def test_preclick_cache_hit_skips_second_findall(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from src.recorder.capture import RecordingSession

    calls: list[int] = []

    def _snap(hwnd: int) -> dict:
        calls.append(int(hwnd))
        return {
            "window": [0, 0, 10, 10],
            "inputs": [],
            "scrollbars": [],
            "texts": [{"bbox": [0, 0, 4, 4], "text": "搜尋"}],
        }

    session = RecordingSession(runs_root=tmp_path)
    session._uia_executor = ThreadPoolExecutor(max_workers=1)
    window = _copilot_window()
    try:
        with patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            side_effect=_snap,
        ):
            session._queue_preclick_uia(window.hwnd, frame_id=3)
            _wait_preclick(session)
            session._latch_press_uia(1, window, side="left")
            session._latch_press_uia(2, window, side="left")
    finally:
        session._uia_executor.shutdown(wait=True)
    assert calls == [window.hwnd]
    assert session._left_press_uia_futures[1].result()["texts"][0]["text"] == "搜尋"
    assert session._left_press_uia_futures[2].result()["texts"][0]["text"] == "搜尋"


def test_preclick_hwnd_miss_uses_click_time_findall(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from src.recorder.capture import RecordingSession

    calls: list[int] = []

    def _snap(hwnd: int) -> dict:
        calls.append(int(hwnd))
        return {"window": None, "inputs": [], "scrollbars": [], "texts": []}

    session = RecordingSession(runs_root=tmp_path)
    session._uia_executor = ThreadPoolExecutor(max_workers=1)
    cached = _copilot_window(10)
    other = _copilot_window(20)
    try:
        with patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            side_effect=_snap,
        ):
            session._queue_preclick_uia(cached.hwnd, frame_id=1)
            _wait_preclick(session)
            session._latch_press_uia(1, other, side="right")
            session._right_press_uia_futures[1].result(timeout=2)
    finally:
        session._uia_executor.shutdown(wait=True)
    assert calls == [10, 20]


def test_preclick_late_completion_does_not_fill_the_slot(tmp_path: Path) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from src.recorder.capture import RecordingSession

    started = threading.Event()
    release = threading.Event()

    def _snap(_hwnd: int) -> dict:
        started.set()
        release.wait(2)
        return {"window": None, "inputs": [], "scrollbars": [], "texts": [{"bbox": [0, 0, 1, 1], "text": "晚到"}]}

    session = RecordingSession(runs_root=tmp_path)
    session._uia_executor = ThreadPoolExecutor(max_workers=1)
    try:
        with patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            side_effect=_snap,
        ):
            session._queue_preclick_uia(5, frame_id=1)
            assert started.wait(2)
            session._mark_preclick_uia_stale()
            release.set()
            session._uia_executor.shutdown(wait=True)
            session._uia_executor = None
    finally:
        release.set()
    slot = session._preclick_uia
    assert slot is not None
    assert slot.stale is True
    assert slot.snapshot is None


def test_preclick_queues_on_first_before_shot_only(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from src.recorder.capture import RecordingSession, _RawBeforeShot

    calls: list[int] = []

    def _snap(hwnd: int) -> dict:
        calls.append(int(hwnd))
        return {"window": None, "inputs": [], "scrollbars": [], "texts": []}

    session = RecordingSession(runs_root=tmp_path)
    session._uia_executor = ThreadPoolExecutor(max_workers=1)
    session._foreground_hwnd = lambda: 42  # type: ignore[method-assign]
    frame = _RawBeforeShot(
        frame_id=1,
        rgb=b"\x00\x00\x00",
        size=(1, 1),
        monitor_index=1,
        monitor_offset=(0, 0),
    )
    later = _RawBeforeShot(
        frame_id=2,
        rgb=b"\x00\x00\x00",
        size=(1, 1),
        monitor_index=1,
        monitor_offset=(0, 0),
    )
    try:
        with patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            side_effect=_snap,
        ):
            assert session._install_raw_before_shot(frame) is True
            assert session._install_raw_before_shot(later) is True
            _wait_preclick(session)
    finally:
        session._uia_executor.shutdown(wait=True)
    assert calls == [42]


def test_startup_before_shot_change_marks_preclick_stale(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from src.recorder.capture import RecordingSession, _RawBeforeShot

    def _snap(_hwnd: int) -> dict:
        return {"window": None, "inputs": [], "scrollbars": [], "texts": []}

    session = RecordingSession(runs_root=tmp_path)
    session._uia_executor = ThreadPoolExecutor(max_workers=1)
    session._foreground_hwnd = lambda: 42  # type: ignore[method-assign]
    first = _RawBeforeShot(
        frame_id=1,
        rgb=b"\x00\x00\x00",
        size=(1, 1),
        monitor_index=1,
        monitor_offset=(0, 0),
    )
    changed = _RawBeforeShot(
        frame_id=2,
        rgb=b"\xff\xff\xff",
        size=(1, 1),
        monitor_index=1,
        monitor_offset=(0, 0),
    )
    try:
        with patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            side_effect=_snap,
        ):
            session._install_raw_before_shot(first)
            _wait_preclick(session)
            session._preclick_compare_mono = 0.0
            session._install_raw_before_shot(changed)
    finally:
        session._uia_executor.shutdown(wait=False, cancel_futures=True)
    slot = session._preclick_uia
    assert slot is not None
    assert slot.stale is True
    assert slot.snapshot is None


def test_preclick_queues_on_first_plateau_only(tmp_path: Path) -> None:
    import json
    from concurrent.futures import ThreadPoolExecutor

    from src.recorder.capture import RecordingSession

    calls: list[int] = []

    def _snap(hwnd: int) -> dict:
        calls.append(int(hwnd))
        return {"window": None, "inputs": [], "scrollbars": [], "texts": []}

    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._uia_executor = ThreadPoolExecutor(max_workers=1)
    events = tmp_path / "events"
    events.mkdir()
    (events / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "t",
                "kind": "click",
                "click_window": {"hwnd": 77},
            }
        ),
        encoding="utf-8",
    )
    try:
        with patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            side_effect=_snap,
        ):
            session._persist_observed_settle(1, 0.03)
            session._persist_observed_settle(1, 1.5)
            _wait_preclick(session)
    finally:
        session._uia_executor.shutdown(wait=True)
    assert calls == [77]
    assert session._preclick_startup_compare is False


@pytest.mark.asyncio
async def test_replay_uia_path_skips_yolo_and_miss_drops_uia_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp import select_mouse_target as smt
    from cua_mcp.select_mouse_target import _detection_from_bbox
    from cua_mcp.yolo_onnx import YOLO_CLASS_TEXT
    from src.recorder.uia_input_boxes import MIN_USABLE_UIA_LANDMARKS

    uia = [
        _detection_from_bbox((20, 20, 40, 16), YOLO_CLASS_TEXT, text="記事本", source="uia")
    ] + [
        _detection_from_bbox(
            (10, 40 + i * 12, 40, 10),
            YOLO_CLASS_TEXT,
            text=f"旁{i:02d}",
            source="uia",
        )
        for i in range(MIN_USABLE_UIA_LANDMARKS)
    ]
    yolo_calls: list[str] = []
    ocr = _detection_from_bbox((80, 80, 30, 16), YOLO_CLASS_TEXT, text="搜尋", source="ocr")

    def fake_grab():
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        return [1], ["shot.png"], [(1, frame)], 0.01

    def fake_uia(_captured, _window):
        return list(uia)

    def fake_finish(*_a, **_k):
        yolo_calls.append("yolo")
        return (
            [1],
            ["shot.png"],
            [(1, np.zeros((20, 20, 3), dtype=np.uint8))],
            [ocr],
            {
                "capture_s": 0.01,
                "yolo_s": 0.1,
                "line_s": 0.0,
                "uia_input_s": 0.0,
                "uia_scrollbar_s": 0.0,
                "uia_text_s": 0.0,
                "ocr_s": 0.2,
                "icon_ocr_retry_s": 0.0,
                "total_s": 0.31,
            },
        )

    monkeypatch.setattr(smt, "_grab_selected_monitors", fake_grab)
    monkeypatch.setattr(smt, "_uia_only_detections_for_click_window", fake_uia)
    monkeypatch.setattr(smt, "_finish_captured_mouse_candidates", fake_finish)
    monkeypatch.setattr(
        smt,
        "parse_mouse_target_instruction",
        lambda _i: ("「記事本」文字", 0, 0, [], None, 0, None),
    )
    window = {"hwnd": 7, "left": 0, "top": 0}
    found = await smt.find_mouse_point("將滑鼠移到「記事本」文字", click_window=window)
    assert yolo_calls == []
    assert found is not None
    assert found[2]["timing"]["vision_path"] == "uia"
    assert found[2]["selection_method"] == "uia_first_unique"
    assert (found[0], found[1]) == (uia[0].cx, uia[0].cy)

    def fake_uia_miss(_captured, _window):
        return [
            _detection_from_bbox((10, 10, 40, 16), YOLO_CLASS_TEXT, text="其他", source="uia")
            for _ in range(MIN_USABLE_UIA_LANDMARKS)
        ]

    monkeypatch.setattr(smt, "_uia_only_detections_for_click_window", fake_uia_miss)
    monkeypatch.setattr(
        smt,
        "parse_mouse_target_instruction",
        lambda _i: ("「搜尋」文字", 0, 0, [], None, 0, None),
    )
    found = await smt.find_mouse_point("將滑鼠移到「搜尋」文字", click_window=window)
    assert yolo_calls == ["yolo"]
    assert found is not None
    assert found[2]["timing"]["vision_path"] == "yolo_ocr"
    assert found[2]["target_text"] == "搜尋"
    assert "https://" not in found[2]["target_text"]


@pytest.mark.asyncio
async def test_yolo_mode_instruction_skips_findall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp import select_mouse_target as smt
    from cua_mcp.select_mouse_target import _detection_from_bbox
    from cua_mcp.yolo_onnx import YOLO_CLASS_TEXT

    ocr = _detection_from_bbox((20, 20, 40, 16), YOLO_CLASS_TEXT, text="空自文件", source="ocr")
    uia_calls: list[str] = []

    def fake_uia(*_a, **_k):
        uia_calls.append("findall")
        raise AssertionError("YOLO+OCR instructions must not FindAll")

    def fake_capture(*_a, **_k):
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        return (
            [1],
            ["shot.png"],
            [(1, frame)],
            [ocr],
            {
                "capture_s": 0.01,
                "yolo_s": 0.1,
                "line_s": 0.0,
                "uia_input_s": 0.0,
                "uia_scrollbar_s": 0.0,
                "uia_text_s": 0.0,
                "ocr_s": 0.2,
                "icon_ocr_retry_s": 0.0,
                "total_s": 0.31,
                "vision_path": "yolo_ocr",
            },
        )

    monkeypatch.setattr(smt, "_uia_only_detections_for_click_window", fake_uia)
    monkeypatch.setattr(smt, "_grab_selected_monitors", fake_uia)
    monkeypatch.setattr(smt, "_capture_and_detect_mouse_candidates", fake_capture)
    window = {"hwnd": 7, "left": 0, "top": 0}
    found = await smt.find_mouse_point(
        "（YOLO+OCR）「空自文件」文字",
        click_window=window,
    )
    assert uia_calls == []
    assert found is not None
    assert (found[0], found[1]) == (ocr.cx, ocr.cy)
    assert found[2]["target_text"] == "空自文件"


@pytest.mark.asyncio
async def test_yolo_mode_finishes_deferred_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp import select_mouse_target as smt
    from cua_mcp.select_mouse_target import _detection_from_bbox
    from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT

    partial = _detection_from_bbox((20, 20, 40, 16), YOLO_CLASS_ELEMENT, source="yolo")
    finished = _detection_from_bbox(
        (20, 20, 40, 16), YOLO_CLASS_TEXT, text="空自文件", source="ocr"
    )
    uia_calls: list[str] = []
    finish_calls: list[object] = []

    def fake_uia(*_a, **_k):
        uia_calls.append("findall")
        raise AssertionError("YOLO+OCR instructions must not FindAll")

    def fake_capture(*_a, **_k):
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        return (
            [1],
            ["shot.png"],
            [(1, frame)],
            [partial],
            {
                "capture_s": 0.01,
                "yolo_s": 0.1,
                "line_s": 0.0,
                "uia_input_s": 0.0,
                "uia_scrollbar_s": 0.0,
                "uia_text_s": 0.0,
                "ocr_s": 0.0,
                "icon_ocr_retry_s": 0.0,
                "total_s": 0.11,
                "vision_path": "yolo_ocr",
                "_ocr_pending": ["pending"],
            },
        )

    def fake_finish(pendings, *, timing_base):
        finish_calls.append(pendings)
        timing = dict(timing_base)
        timing["ocr_s"] = 0.2
        return [finished], timing

    monkeypatch.setattr(smt, "_uia_only_detections_for_click_window", fake_uia)
    monkeypatch.setattr(smt, "_grab_selected_monitors", fake_uia)
    monkeypatch.setattr(smt, "_capture_and_detect_mouse_candidates", fake_capture)
    monkeypatch.setattr(smt, "_finish_deferred_ocr_pendings", fake_finish)
    found = await smt.find_mouse_point("（YOLO+OCR）「空自文件」文字")
    assert uia_calls == []
    assert finish_calls == [["pending"]]
    assert found is not None
    assert (found[0], found[1]) == (finished.cx, finished.cy)
    assert found[2]["target_text"] == "空自文件"


@pytest.mark.asyncio
async def test_uia_mode_typed_miss_falls_through_to_yolo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from cua_mcp import select_mouse_target as smt
    from cua_mcp.select_mouse_target import _detection_from_bbox
    from cua_mcp.yolo_onnx import YOLO_CLASS_TEXT
    from src.recorder.uia_input_boxes import MIN_USABLE_UIA_LANDMARKS

    uia = [
        _detection_from_bbox(
            (10, 10 + i * 12, 40, 10),
            YOLO_CLASS_TEXT,
            text=f"旁{i:02d}",
            source="uia",
            uia_control_type="list_item",
        )
        for i in range(MIN_USABLE_UIA_LANDMARKS)
    ]
    ocr = _detection_from_bbox((80, 80, 30, 16), YOLO_CLASS_TEXT, text="記事本", source="ocr")
    yolo_calls: list[str] = []

    def fake_grab():
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        return [1], ["shot.png"], [(1, frame)], 0.01

    def fake_uia(_captured, _window):
        return list(uia)

    def fake_finish(*_a, **_k):
        yolo_calls.append("yolo")
        return (
            [1],
            ["shot.png"],
            [(1, np.zeros((20, 20, 3), dtype=np.uint8))],
            [ocr],
            {
                "capture_s": 0.01,
                "yolo_s": 0.1,
                "line_s": 0.0,
                "uia_input_s": 0.0,
                "uia_scrollbar_s": 0.0,
                "uia_text_s": 0.0,
                "ocr_s": 0.2,
                "icon_ocr_retry_s": 0.0,
                "total_s": 0.31,
            },
        )

    monkeypatch.setattr(smt, "_grab_selected_monitors", fake_grab)
    monkeypatch.setattr(smt, "_uia_only_detections_for_click_window", fake_uia)
    monkeypatch.setattr(smt, "_finish_captured_mouse_candidates", fake_finish)
    window = {"hwnd": 7, "left": 0, "top": 0}
    found = await smt.find_mouse_point("（UIA）「記事本」按鈕", click_window=window)
    assert yolo_calls == ["yolo"]
    assert found is not None
    assert found[2]["target_text"] == "記事本"
