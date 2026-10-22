"""Replay-time UIA repair: recorded OCR box against a live UIA element."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.recorder.uia_replay_repair import (
    choose_uia_replay_repair,
    write_uia_replay_repair,
    UiaReplayRepairRequest,
)
from src.recorder.window_snapshot import ClickWindowInfo

_RECORDED_BBOX = (10, 20, 40, 16)
_RECORDED_WINDOW = (0, 0, 200, 200)
_LIVE_WINDOW = (1000, 500, 200, 200)
# Mapped box is (1010, 520, 40, 16); center is (1030, 528).
_UIA_BOX = (1000, 500, 80, 40)
_OCR_BOX = (1010, 520, 40, 16)


def _choose(**overrides):
    kwargs = dict(
        recorded_bbox=_RECORDED_BBOX,
        recorded_text="空自文件",
        recorded_window_xywh=_RECORDED_WINDOW,
        live_window_xywh=_LIVE_WINDOW,
        uia_boxes=[(_UIA_BOX, "空白文件")],
        ocr_boxes=[(_OCR_BOX, "空白文件")],
        point_probe_fn=None,
    )
    kwargs.update(overrides)
    return choose_uia_replay_repair(**kwargs)


def test_unique_uia_rect_uses_uia_name_and_rect_center() -> None:
    hit = _choose()
    assert hit is not None
    assert hit.label == "空白文件"
    assert hit.source == "findall"
    assert (hit.cx, hit.cy) == (1040, 520)
    assert hit.bbox == _UIA_BOX


def test_two_uia_names_covering_the_point_do_not_repair() -> None:
    hit = _choose(
        uia_boxes=[
            (_UIA_BOX, "空白文件"),
            ((990, 490, 200, 80), "歡迎使用 Word"),
        ]
    )
    assert hit is None


def test_ocr_far_from_recorded_text_does_not_repair() -> None:
    hit = _choose(ocr_boxes=[(_OCR_BOX, "搜尋檔案")])
    assert hit is None


def test_element_from_point_when_findall_is_empty() -> None:
    probe_box = (1020, 510, 30, 20)

    def probe(x: int, y: int):
        assert (x, y) == (1030, 528)
        return "空白文件", probe_box

    hit = _choose(uia_boxes=[], point_probe_fn=probe)
    assert hit is not None
    assert hit.label == "空白文件"
    assert hit.source == "element_from_point"
    assert (hit.cx, hit.cy) == (1035, 520)


def test_write_replaces_ocr_instruction_and_leaves_event_snapshot(tmp_path: Path) -> None:
    run = tmp_path / "recording"
    (run / "events").mkdir(parents=True)
    (run / "analysis").mkdir()
    (run / "yolo_ocr").mkdir()
    event = {
        "index": 6,
        "timestamp_utc": "2026-10-21T00:00:00+00:00",
        "kind": "click",
        "button": "left",
        "click_window": {
            "hwnd": 5,
            "title": "Word",
            "process_name": "WINWORD.EXE",
            "rect": [0, 0, 200, 200],
            "is_maximized": True,
            "class_name": "OpusApp",
        },
        "uia_snapshot": {
            "window": [0, 0, 200, 200],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    }
    event_path = run / "events" / "event_006.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    analysis = {
        "event_index": 6,
        "instruction": (
            "將滑鼠移到「空自文件」文字（在「碟覽」文字的左下方），並點擊滑鼠一下。"
        ),
    }
    (run / "analysis" / "event_006.json").write_text(
        json.dumps(analysis, ensure_ascii=False), encoding="utf-8"
    )
    from src.recorder.models import RecordedEvent
    from src.recorder.vision_context import (
        try_rebuild_vision_from_cache,
        vision_source_fingerprint,
    )

    event_obj = RecordedEvent.from_dict(event)
    yolo_payload = {
        "source_fingerprint": vision_source_fingerprint(event_obj),
        "candidates": [
            {
                "bbox": [10, 20, 40, 16],
                "center": [30, 28],
                "class_name": "text",
                "text": "空自文件",
            }
        ],
        "candidate_text": "空自文件",
        "detection_count": 1,
        "vision_path": "yolo_ocr",
        "vision_path_reason": "沒有 UIA 快照",
    }
    yolo_path = run / "yolo_ocr" / "event_006.json"
    yolo_path.write_text(json.dumps(yolo_payload, ensure_ascii=False), encoding="utf-8")
    request = UiaReplayRepairRequest(
        recording_dir=run,
        event_index=6,
        side="click",
        recorded_bbox=_RECORDED_BBOX,
        recorded_text="空自文件",
        recorded_window_xywh=_RECORDED_WINDOW,
    )
    instruction = write_uia_replay_repair(
        request,
        label="空白文件",
        source="findall",
        bbox=_UIA_BOX,
        others=[],
    )
    assert instruction is not None
    assert instruction.startswith("（UIA）")
    assert "空白文件" in instruction
    assert "碟覽" not in instruction
    saved = json.loads((run / "analysis" / "event_006.json").read_text(encoding="utf-8"))
    assert saved["uia_replay_repair"]["label"] == "空白文件"
    assert saved["uia_replay_repair"]["source"] == "findall"
    assert saved["tool_calls"]
    kept = json.loads(event_path.read_text(encoding="utf-8"))
    assert kept["uia_snapshot"]["texts"] == []
    cache = json.loads((run / "instruction_tool_cache.json").read_text(encoding="utf-8"))
    assert any("空白文件" in key for key in cache["entries"])
    stored = json.loads(yolo_path.read_text(encoding="utf-8"))
    assert stored["vision_path"] == "uia"
    assert stored["vision_path_reason"] == "重播修復改為 UIA"
    from src.recorder.analyze import rebuild_pointer_instruction

    rebuilt = try_rebuild_vision_from_cache(event_obj, run)
    assert rebuilt is not None
    assert rebuilt["vision_path"] == "uia"
    again = rebuild_pointer_instruction(event_obj, rebuilt, include_nearby=False)
    assert again is not None
    assert again.startswith("（UIA）")


def test_drag_repair_prefixes_uia_only_when_both_sides_are_uia(tmp_path: Path) -> None:
    run = tmp_path / "recording"
    (run / "events").mkdir(parents=True)
    (run / "analysis").mkdir()
    (run / "yolo_ocr").mkdir()
    window = {
        "hwnd": 5,
        "title": "Word",
        "process_name": "WINWORD.EXE",
        "rect": [0, 0, 200, 200],
        "is_maximized": True,
        "class_name": "OpusApp",
    }
    event = {
        "index": 4,
        "timestamp_utc": "t",
        "kind": "drag",
        "cursor_xy": [10, 20],
        "end_xy": [80, 20],
        "click_window": window,
        "end_click_window": window,
    }
    (run / "events" / "event_004.json").write_text(
        json.dumps(event), encoding="utf-8"
    )
    (run / "analysis" / "event_004.json").write_text(
        json.dumps(
            {"event_index": 4, "instruction": "從「甲」文字拖到「乙」文字"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    from src.recorder.models import RecordedEvent
    from src.recorder.vision_context import vision_source_fingerprint

    fingerprint = vision_source_fingerprint(RecordedEvent.from_dict(event))

    def payload(text: str) -> dict:
        return {
            "source_fingerprint": fingerprint,
            "candidates": [
                {
                    "bbox": [0, 0, 10, 10],
                    "center": [5, 5],
                    "class_name": "text",
                    "text": text,
                }
            ],
            "candidate_text": text,
            "detection_count": 1,
            "vision_path": "yolo_ocr",
            "vision_path_reason": "沒有 UIA 快照",
        }

    (run / "yolo_ocr" / "event_004.json").write_text(
        json.dumps(payload("甲"), ensure_ascii=False), encoding="utf-8"
    )
    (run / "yolo_ocr" / "event_004_end.json").write_text(
        json.dumps(payload("乙"), ensure_ascii=False), encoding="utf-8"
    )
    request = UiaReplayRepairRequest(
        recording_dir=run,
        event_index=4,
        side="start",
        recorded_bbox=(0, 0, 10, 10),
        recorded_text="甲",
        recorded_window_xywh=(0, 0, 200, 200),
    )
    one_side = write_uia_replay_repair(
        request,
        label="甲項",
        source="findall",
        bbox=(0, 0, 10, 10),
        others=[],
    )
    assert one_side is not None
    assert one_side.startswith("（YOLO+OCR）")
    end_request = UiaReplayRepairRequest(
        recording_dir=run,
        event_index=4,
        side="end",
        recorded_bbox=(0, 0, 10, 10),
        recorded_text="乙",
        recorded_window_xywh=(0, 0, 200, 200),
    )
    both = write_uia_replay_repair(
        end_request,
        label="乙項",
        source="findall",
        bbox=(0, 0, 10, 10),
        others=[],
    )
    assert both is not None
    assert both.startswith("（UIA）")


def test_uia_walk_uses_live_hwnd_when_recorded_hwnd_is_stale(monkeypatch) -> None:
    from cua_mcp import select_mouse_target as smt

    seen: dict[str, int] = {}

    def fake_match(_click_window):
        return ClickWindowInfo(
            hwnd=99,
            title="Word",
            process_name="WINWORD.EXE",
            left=10,
            top=20,
            width=30,
            height=40,
            is_maximized=True,
        )

    def fake_snapshot(hwnd: int):
        seen["hwnd"] = int(hwnd)
        return {
            "window": [10, 20, 30, 40],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        }

    monkeypatch.setattr(
        "src.recorder.window_snapshot.find_matching_click_window", fake_match
    )
    monkeypatch.setattr(
        "src.recorder.uia_input_boxes.snapshot_uia_screen", fake_snapshot
    )
    monkeypatch.setattr(smt, "active_monitor_offset", lambda _index: (0, 0))
    bgr = np.zeros((80, 80, 3), dtype=np.uint8)
    found = smt._uia_only_detections_for_click_window(
        [(1, bgr)],
        {"hwnd": 5, "title": "Word", "process_name": "WINWORD.EXE", "rect": [0, 0, 30, 40]},
    )
    assert seen["hwnd"] == 99
    assert found == []


def test_replay_repair_slots_only_for_yolo_ocr(tmp_path: Path) -> None:
    from src.recorder.uia_replay_repair import replay_repair_slots

    run = tmp_path / "recording"
    (run / "events").mkdir(parents=True)
    (run / "analysis").mkdir()
    (run / "yolo_ocr").mkdir()
    event = {
        "index": 2,
        "timestamp_utc": "t",
        "kind": "click",
        "button": "left",
        "click_window": {
            "hwnd": 5,
            "title": "常用 - 檔案總管",
            "process_name": "explorer.exe",
            "rect": [0, 0, 200, 200],
            "class_name": "CabinetWClass",
        },
        "uia_snapshot": {
            "window": [0, 0, 200, 200],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    }
    (run / "events" / "event_002.json").write_text(json.dumps(event), encoding="utf-8")
    (run / "analysis" / "event_002.json").write_text(
        json.dumps(
            {
                "event_index": 2,
                "instruction": "（YOLO+OCR）將滑鼠移到「文件」文字，並點擊滑鼠一下。",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "vision_path": "yolo_ocr",
        "candidates": [
            {"bbox": [10, 20, 30, 14], "text": "文件", "class_name": "text"}
        ],
    }
    yolo_path = run / "yolo_ocr" / "event_002.json"
    yolo_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    calls = [{"name": "move_mouse", "arguments": {}}]
    slots = replay_repair_slots(run, 2, calls)
    assert len(slots) == 1 and slots[0] is not None
    assert slots[0].recorded_text == "文件"
    payload["vision_path"] = "uia"
    yolo_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert replay_repair_slots(run, 2, calls) == []
