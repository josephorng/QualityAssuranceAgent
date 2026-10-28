"""Recording-time YOLO+OCR twin under analysis.yolo_ocr_fallback."""

from __future__ import annotations

import json
from pathlib import Path

from src.recorder.orchestrator import _write_event_analysis
from src.recorder.models import RecordedEvent
from src.recorder.uia_replay_repair import build_yolo_ocr_fallback_from_vision


def _click_event() -> RecordedEvent:
    return RecordedEvent.from_dict(
        {
            "index": 3,
            "timestamp_utc": "2026-10-21T00:00:00+00:00",
            "kind": "click",
            "button": "left",
            "click_window": {
                "hwnd": 5,
                "title": "App",
                "process_name": "app.exe",
                "rect": [0, 0, 200, 200],
                "class_name": "AppWindow",
            },
        }
    )


def test_build_yolo_ocr_fallback_from_vision_builds_twin() -> None:
    event = _click_event()
    vision = {
        "used_vision": True,
        "vision_path": "uia",
        "candidates": [
            {
                "bbox": [10, 20, 40, 16],
                "center": [30, 28],
                "class_name": "text",
                "text": "檔案",
                "uia_control_type": "button",
            }
        ],
        "candidate_text": "檔案",
        "yolo_ocr_fallback_vision": {
            "used_vision": True,
            "vision_path": "yolo_ocr",
            "candidates": [
                {
                    "bbox": [12, 22, 36, 14],
                    "center": [30, 29],
                    "class_name": "text",
                    "text": "檔案",
                }
            ],
            "candidate_text": "檔案",
            "local_cursor": (30, 29),
        },
    }
    fallback = build_yolo_ocr_fallback_from_vision(event, vision)
    assert fallback is not None
    assert fallback["instruction"].startswith("（YOLO+OCR）")
    assert "檔案" in fallback["instruction"]
    move = next(call for call in fallback["tool_calls"] if call["name"] == "move_mouse")
    assert move["arguments"]["vision_mode"] == "yolo_ocr"


def test_write_event_analysis_persists_and_preserves_fallback(tmp_path: Path) -> None:
    run = tmp_path / "recording"
    (run / "analysis").mkdir(parents=True)
    (run / "events").mkdir()
    analysis_path = run / "analysis" / "event_003.json"
    event = _click_event()
    (run / "events" / "event_003.json").write_text(
        json.dumps(event.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    vision = {
        "used_vision": True,
        "vision_path": "uia",
        "candidates": [
            {
                "bbox": [10, 20, 40, 16],
                "center": [30, 28],
                "class_name": "text",
                "text": "檔案",
                "uia_control_type": "button",
            }
        ],
        "candidate_text": "檔案",
        "yolo_ocr_fallback_vision": {
            "used_vision": True,
            "vision_path": "yolo_ocr",
            "candidates": [
                {
                    "bbox": [12, 22, 36, 14],
                    "center": [30, 29],
                    "class_name": "text",
                    "text": "檔案OCR",
                }
            ],
            "candidate_text": "檔案OCR",
            "local_cursor": (30, 29),
        },
    }
    _write_event_analysis(
        analysis_path,
        event=event,
        result={},
        instruction="（UIA）將滑鼠移到「檔案」按鈕，並點擊滑鼠一下。",
        vision=vision,
        expected_outcome=None,
        use_expected_outcome=False,
        elapsed_since_previous=None,
        settle_after_seconds=None,
        text_resolution=None,
        run_dir=run,
    )
    saved = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert saved["instruction"].startswith("（UIA）")
    assert saved["yolo_ocr_fallback"]["instruction"].startswith("（YOLO+OCR）")
    assert "檔案OCR" in saved["yolo_ocr_fallback"]["instruction"]

    # Re-write without a twin vision; previous OCR fallback must stay.
    _write_event_analysis(
        analysis_path,
        event=event,
        result={},
        instruction="（UIA）將滑鼠移到「檔案」按鈕，並點擊滑鼠一下。",
        vision={
            "used_vision": True,
            "vision_path": "uia",
            "candidates": vision["candidates"],
            "candidate_text": "檔案",
        },
        expected_outcome=None,
        use_expected_outcome=False,
        elapsed_since_previous=None,
        settle_after_seconds=None,
        text_resolution=None,
        run_dir=run,
    )
    saved_again = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert (
        saved_again["yolo_ocr_fallback"]["instruction"]
        == saved["yolo_ocr_fallback"]["instruction"]
    )
