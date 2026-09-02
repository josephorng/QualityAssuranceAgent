"""Tests for mid-recording vision fingerprint cache and prefetch worker."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cua_mcp.select_mouse_target import _detection_from_bbox
from cua_mcp.yolo_onnx import YOLO_CLASS_TEXT
from src.recorder.models import RecordedEvent
from src.recorder.orchestrator import prepare_event_vision
from src.recorder.vision_context import (
    build_vision_context,
    try_rebuild_vision_from_cache,
    vision_source_fingerprint,
)
from src.recorder.vision_prefetch import VisionPrefetchWorker


def _parallel_safe_drag_fakes(start_detections, end_detections):
    start_img = np.zeros((100, 100, 3), dtype=np.uint8)
    end_img = np.full((100, 100, 3), 1, dtype=np.uint8)

    def fake_imread(path):
        name = Path(path).name.lower()
        return end_img if "end" in name else start_img

    def fake_build(bgr, **_kwargs):
        return end_detections if int(bgr[0, 0, 0]) == 1 else start_detections

    return fake_imread, fake_build


def _click_event(tmp_path: Path, *, index: int = 1) -> RecordedEvent:
    shot = tmp_path / "screenshots" / f"event_{index:03d}.jpeg"
    shot.parent.mkdir(parents=True, exist_ok=True)
    shot.write_bytes(b"not-a-real-jpeg")
    return RecordedEvent(
        index=index,
        timestamp_utc="t",
        kind="click",
        cursor_xy=(110, 210),
        monitor_offset=(100, 200),
        screenshot_path=str(shot),
    )


def test_vision_source_fingerprint_stable_and_sensitive(tmp_path: Path) -> None:
    event = _click_event(tmp_path)
    first = vision_source_fingerprint(event)
    assert first == vision_source_fingerprint(event)
    assert first != vision_source_fingerprint(replace(event, kind="double_click"))
    assert first != vision_source_fingerprint(replace(event, text="x"))
    assert first != vision_source_fingerprint(
        replace(event, end_screenshot_path=str(tmp_path / "other.jpeg"))
    )


@pytest.mark.asyncio
async def test_try_rebuild_vision_from_cache_round_trip_skips_triton(tmp_path: Path) -> None:
    (tmp_path / "yolo_ocr").mkdir()
    event = _click_event(tmp_path)
    fake_detections = [
        _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="儲存"),
    ]
    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        return_value=fake_detections,
    ) as detect:
        vision = await build_vision_context(event, run_dir=tmp_path, persist_debug=True)
        assert detect.call_count == 1

    payload = json.loads((tmp_path / "yolo_ocr" / "event_001.json").read_text(encoding="utf-8"))
    assert payload["source_fingerprint"] == vision_source_fingerprint(event)

    with patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=AssertionError("should not call Triton"),
    ):
        rebuilt = try_rebuild_vision_from_cache(event, tmp_path)

    assert rebuilt is not None
    assert rebuilt["used_vision"] is True
    assert rebuilt["candidate_text"] == vision["candidate_text"]
    assert len(rebuilt["candidates"]) == len(vision["candidates"])
    assert rebuilt["field_context"] == vision["field_context"]


@pytest.mark.asyncio
async def test_try_rebuild_vision_from_cache_misses_stale_fingerprint(tmp_path: Path) -> None:
    (tmp_path / "yolo_ocr").mkdir()
    event = _click_event(tmp_path)
    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        return_value=[
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="儲存"),
        ],
    ):
        await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    stale = replace(event, kind="double_click", click_count=2)
    assert try_rebuild_vision_from_cache(stale, tmp_path) is None


@pytest.mark.asyncio
async def test_try_rebuild_drag_vision_from_cache(tmp_path: Path) -> None:
    (tmp_path / "yolo_ocr").mkdir()
    start_shot = tmp_path / "event_start.jpeg"
    end_shot = tmp_path / "event_end.jpeg"
    start_shot.write_bytes(b"x")
    end_shot.write_bytes(b"y")
    event = RecordedEvent(
        index=3,
        timestamp_utc="t",
        kind="drag",
        cursor_xy=(110, 210),
        end_xy=(310, 210),
        monitor_offset=(100, 200),
        end_monitor_offset=(100, 200),
        screenshot_path=str(start_shot),
        end_screenshot_path=str(end_shot),
    )
    start_detections = [
        _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="音量"),
    ]
    end_detections = [
        _detection_from_bbox((200, 8, 40, 10), YOLO_CLASS_TEXT, text="100%"),
    ]
    fake_imread, fake_build = _parallel_safe_drag_fakes(start_detections, end_detections)
    with patch(
        "src.recorder.vision_context.imread_bgr",
        side_effect=fake_imread,
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=fake_build,
    ):
        original = await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    with patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=AssertionError("should not call Triton"),
    ):
        rebuilt = try_rebuild_vision_from_cache(event, tmp_path)

    assert rebuilt is not None
    assert rebuilt["candidate_text"] == original["candidate_text"]
    assert rebuilt["destination"]["candidate_text"] == original["destination"]["candidate_text"]


@pytest.mark.asyncio
async def test_prepare_event_vision_uses_cache_hit(tmp_path: Path) -> None:
    (tmp_path / "yolo_ocr").mkdir()
    event = _click_event(tmp_path)
    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        return_value=[
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="開啟"),
        ],
    ):
        await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    logs: list[str] = []
    with patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=AssertionError("should not call Triton"),
    ), patch(
        "src.recorder.orchestrator.build_vision_context",
        side_effect=AssertionError("should not rebuild vision"),
    ):
        prepared = await prepare_event_vision(
            event,
            run_dir=tmp_path,
            log_info=logs.append,
        )

    assert prepared.vision["used_vision"] is True
    assert any("vision cache hit" in line for line in logs)


@pytest.mark.asyncio
async def test_prepare_event_vision_recomputes_on_stale_cache(tmp_path: Path) -> None:
    (tmp_path / "yolo_ocr").mkdir()
    event = _click_event(tmp_path)
    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        return_value=[
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="開啟"),
        ],
    ):
        await build_vision_context(event, run_dir=tmp_path, persist_debug=True)

    stale = replace(event, kind="double_click", click_count=2)
    detect_calls = {"n": 0}

    def _detect(_bgr):
        detect_calls["n"] += 1
        return [
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="開啟"),
        ]

    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        side_effect=_detect,
    ):
        prepared = await prepare_event_vision(
            stale,
            run_dir=tmp_path,
            log_info=lambda _t: None,
        )

    assert detect_calls["n"] == 1
    assert prepared.vision["used_vision"] is True


def test_vision_prefetch_worker_writes_yolo_ocr(tmp_path: Path) -> None:
    (tmp_path / "yolo_ocr").mkdir()
    event = _click_event(tmp_path)
    worker = VisionPrefetchWorker()
    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        return_value=[
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="預取"),
        ],
    ):
        worker.start(tmp_path)
        worker.enqueue(event)
        worker.drain_and_stop(timeout=30.0)

    out = tmp_path / "yolo_ocr" / "event_001.json"
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source_fingerprint"] == vision_source_fingerprint(event)
    assert payload["candidates"]


def test_vision_prefetch_drain_reports_progress(tmp_path: Path) -> None:
    (tmp_path / "yolo_ocr").mkdir()
    events = [_click_event(tmp_path, index=i) for i in range(1, 4)]
    worker = VisionPrefetchWorker()
    progress: list[tuple[int, int]] = []

    def on_progress(current: int, total: int) -> None:
        progress.append((current, total))

    with patch(
        "src.recorder.vision_context.imread_bgr",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "src.recorder.vision_context._detect_mouse_targets_from_bgr",
        return_value=[
            _detection_from_bbox((8, 8, 10, 10), YOLO_CLASS_TEXT, text="預取"),
        ],
    ):
        worker.start(tmp_path)
        for event in events:
            worker.enqueue(event)
        worker.drain_and_stop(timeout=30.0, on_progress=on_progress)

    assert progress[0] == (0, 3)
    assert progress[-1] == (3, 3)


def test_vision_prefetch_drain_empty_queue_reports_zero_total(tmp_path: Path) -> None:
    worker = VisionPrefetchWorker()
    progress: list[tuple[int, int]] = []

    def on_progress(current: int, total: int) -> None:
        progress.append((current, total))

    worker.start(tmp_path)
    worker.drain_and_stop(timeout=5.0, on_progress=on_progress)
    assert progress == [(0, 0)]


def test_vision_prefetch_runs_events_in_parallel(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import time

    (tmp_path / "yolo_ocr").mkdir()
    events = [_click_event(tmp_path, index=i) for i in range(1, 5)]
    worker = VisionPrefetchWorker(max_workers=4)
    monkeypatch.setenv("RECORDING_VISION_WORKERS", "4")

    active = 0
    max_active = 0
    lock = threading.Lock()

    async def slow_prepare(event, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        with lock:
            active -= 1
        from src.recorder.orchestrator import _PreparedEvent

        return _PreparedEvent(
            event=event,
            event_for_llm=event,
            vision={"used_vision": False, "candidate_text": "", "local_cursor": None},
            text_resolution=None,
        )

    started = time.perf_counter()
    with patch(
        "src.recorder.vision_prefetch.prepare_event_vision",
        new=slow_prepare,
    ):
        worker.start(tmp_path)
        for event in events:
            worker.enqueue(event)
        worker.drain_and_stop(timeout=30.0)
    elapsed = time.perf_counter() - started

    assert max_active >= 2
    assert elapsed < 0.20
