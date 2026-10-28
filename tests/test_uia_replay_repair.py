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


def test_choose_repairs_when_ocr_confirms_icon_chinese_id() -> None:
    hit = _choose(
        recorded_text="向上雙V箭頭",
        ocr_boxes=[(_OCR_BOX, "向上雙V箭頭")],
        uia_boxes=[(_UIA_BOX, "顯示桌面")],
    )
    assert hit is not None
    assert hit.label == "顯示桌面"
    assert hit.source == "findall"


def test_choose_misses_when_icon_ocr_does_not_match() -> None:
    hit = _choose(
        recorded_text="向上雙V箭頭",
        ocr_boxes=[(_OCR_BOX, "搜尋")],
        uia_boxes=[(_UIA_BOX, "顯示桌面")],
    )
    assert hit is None


def test_choose_repairs_when_ocr_confirms_input_class() -> None:
    hit = _choose(
        recorded_text="輸入欄",
        ocr_boxes=[(_OCR_BOX, "輸入欄")],
        uia_boxes=[(_UIA_BOX, "公式列")],
    )
    assert hit is not None
    assert hit.label == "公式列"
    assert hit.source == "findall"


def test_choose_misses_when_input_class_does_not_overlap() -> None:
    hit = _choose(
        recorded_text="輸入欄",
        ocr_boxes=[((1200, 700, 40, 16), "輸入欄")],
        uia_boxes=[(_UIA_BOX, "公式列")],
    )
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


def test_repair_keeps_yolo_ocr_instruction_and_tool_calls(tmp_path: Path) -> None:
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
        "uia_snapshot": {"window": [0, 0, 200, 200], "inputs": [], "scrollbars": [], "texts": []},
    }
    (run / "events" / "event_006.json").write_text(json.dumps(event), encoding="utf-8")
    yolo_instruction = (
        "（YOLO+OCR）將滑鼠移到「文件」文字（在「新增」文字的下面），並點擊滑鼠一下。"
    )
    (run / "analysis" / "event_006.json").write_text(
        json.dumps(
            {"event_index": 6, "instruction": yolo_instruction},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run / "yolo_ocr" / "event_006.json").write_text(
        json.dumps(
            {
                "vision_path": "yolo_ocr",
                "candidates": [
                    {"bbox": [10, 20, 40, 16], "center": [30, 28], "class_name": "text", "text": "文件"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    request = UiaReplayRepairRequest(
        recording_dir=run,
        event_index=6,
        side="click",
        recorded_bbox=_RECORDED_BBOX,
        recorded_text="文件",
        recorded_window_xywh=_RECORDED_WINDOW,
    )
    first = write_uia_replay_repair(
        request,
        label="文件 (已釘選)",
        source="element_from_point",
        bbox=_UIA_BOX,
        others=[],
    )
    assert first is not None and first.startswith("（UIA）")
    saved = json.loads((run / "analysis" / "event_006.json").read_text(encoding="utf-8"))
    fallback = saved["yolo_ocr_fallback"]
    assert fallback["instruction"] == yolo_instruction
    move = next(call for call in fallback["tool_calls"] if call["name"] == "move_mouse")
    assert "文件" in move["arguments"]["instruction"]
    assert "已釘選" not in move["arguments"]["instruction"]
    assert any("新增" in item for item in move["arguments"]["nearby_objects"])

    again = write_uia_replay_repair(
        request,
        label="文件 (已釘選)",
        source="findall",
        bbox=_UIA_BOX,
        others=[],
    )
    assert again is not None and again.startswith("（UIA）")
    saved_again = json.loads((run / "analysis" / "event_006.json").read_text(encoding="utf-8"))
    assert saved_again["yolo_ocr_fallback"]["instruction"] == yolo_instruction


def test_backfill_restores_report_yolo_line(tmp_path: Path) -> None:
    from src.recorder.uia_replay_repair import backfill_yolo_ocr_fallbacks

    run = tmp_path / "recording"
    (run / "events").mkdir(parents=True)
    (run / "analysis").mkdir()
    yolo_instruction = "（YOLO+OCR）將滑鼠移到「文件」文字，並點擊滑鼠一下。"
    event = {
        "index": 2,
        "timestamp_utc": "t",
        "kind": "click",
        "button": "left",
        "click_window": {
            "hwnd": 5,
            "title": "Explorer",
            "process_name": "explorer.exe",
            "rect": [0, 0, 200, 200],
            "class_name": "CabinetWClass",
        },
    }
    (run / "events" / "event_001.json").write_text(
        json.dumps({"index": 1, "timestamp_utc": "t", "kind": "click", "button": "left"}),
        encoding="utf-8",
    )
    (run / "analysis" / "event_001.json").write_text(
        json.dumps({"event_index": 1, "instruction": "（UIA）第一步"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run / "events" / "event_002.json").write_text(json.dumps(event), encoding="utf-8")
    (run / "analysis" / "event_002.json").write_text(
        json.dumps(
            {
                "event_index": 2,
                "instruction": "（UIA）將滑鼠移到「文件 (已釘選)」文字，並點擊滑鼠一下。",
                "uia_replay_repair": {"label": "文件 (已釘選)", "source": "element_from_point"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run / "report.json").write_text(
        json.dumps({"instructions": ["（UIA）第一步", yolo_instruction]}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert backfill_yolo_ocr_fallbacks(run) == 1
    saved = json.loads((run / "analysis" / "event_002.json").read_text(encoding="utf-8"))
    assert saved["yolo_ocr_fallback"]["instruction"] == yolo_instruction
    assert saved["yolo_ocr_fallback"]["tool_calls"][0]["name"] == "move_mouse"
    assert backfill_yolo_ocr_fallbacks(run) == 0


def test_recorded_primary_resolves_pua_to_chinese_id() -> None:
    from src.recorder.uia_replay_repair import _recorded_primary

    box, text = _recorded_primary(
        {
            "candidates": [
                {
                    "bbox": [820, 1035, 31, 40],
                    "text": "\ue059",
                    "class_name": "element",
                    "icons": [
                        {
                            "pua": "\ue059",
                            "chinese_id": "向上雙V箭頭",
                        }
                    ],
                }
            ]
        }
    )
    assert box == (820, 1035, 31, 40)
    assert text == "向上雙V箭頭"


def test_recorded_primary_keeps_unmapped_pua() -> None:
    from src.recorder.uia_replay_repair import _recorded_primary

    box, text = _recorded_primary(
        {
            "candidates": [
                {
                    "bbox": [10, 20, 16, 16],
                    "text": "\ue059",
                    "class_name": "element",
                    "icons": None,
                }
            ]
        }
    )
    assert box == (10, 20, 16, 16)
    assert text == "\ue059"


def test_replay_repair_slots_resolves_icon_pua_primary(tmp_path: Path) -> None:
    from src.recorder.uia_replay_repair import replay_repair_slots

    run = tmp_path / "recording"
    (run / "events").mkdir(parents=True)
    (run / "analysis").mkdir()
    (run / "yolo_ocr").mkdir()
    event = {
        "index": 12,
        "timestamp_utc": "t",
        "kind": "click",
        "button": "left",
        "click_window": {
            "hwnd": 5,
            "title": "",
            "process_name": "explorer.exe",
            "rect": [0, 1032, 1920, 48],
            "class_name": "Shell_TrayWnd",
            "is_taskbar": True,
        },
        "uia_snapshot": {
            "window": [0, 1032, 1920, 48],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    }
    (run / "events" / "event_012.json").write_text(json.dumps(event), encoding="utf-8")
    (run / "analysis" / "event_012.json").write_text(
        json.dumps(
            {
                "event_index": 12,
                "instruction": (
                    "（YOLO+OCR）將滑鼠移到「向上雙V箭頭」圖示，並點擊滑鼠一下。"
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "vision_path": "yolo_ocr",
        "candidates": [
            {
                "bbox": [820, 1035, 31, 40],
                "text": "\ue059",
                "class_name": "element",
                "icons": [{"pua": "\ue059", "chinese_id": "向上雙V箭頭"}],
            }
        ],
    }
    (run / "yolo_ocr" / "event_012.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    slots = replay_repair_slots(run, 12, [{"name": "move_mouse", "arguments": {}}])
    assert len(slots) == 1 and slots[0] is not None
    assert slots[0].recorded_text == "向上雙V箭頭"
    assert slots[0].recorded_bbox == (820, 1035, 31, 40)


def test_write_replaces_icon_instruction_anchor(tmp_path: Path) -> None:
    run = tmp_path / "recording"
    (run / "events").mkdir(parents=True)
    (run / "analysis").mkdir()
    (run / "yolo_ocr").mkdir()
    event = {
        "index": 12,
        "timestamp_utc": "2026-10-21T00:00:00+00:00",
        "kind": "click",
        "button": "left",
        "click_window": {
            "hwnd": 5,
            "title": "",
            "process_name": "explorer.exe",
            "rect": [0, 1032, 1920, 48],
            "is_maximized": False,
            "class_name": "Shell_TrayWnd",
            "is_taskbar": True,
        },
        "uia_snapshot": {
            "window": [0, 1032, 1920, 48],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    }
    event_path = run / "events" / "event_012.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    yolo_instruction = (
        "（YOLO+OCR）將滑鼠移到「向上雙V箭頭」圖示（在「搜尋」文字的右邊），並點擊滑鼠一下。"
    )
    (run / "analysis" / "event_012.json").write_text(
        json.dumps(
            {"event_index": 12, "instruction": yolo_instruction},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    from src.recorder.models import RecordedEvent
    from src.recorder.vision_context import vision_source_fingerprint

    event_obj = RecordedEvent.from_dict(event)
    yolo_payload = {
        "source_fingerprint": vision_source_fingerprint(event_obj),
        "candidates": [
            {
                "bbox": [820, 1035, 31, 40],
                "center": [835, 1055],
                "class_name": "element",
                "text": "\ue059",
                "icons": [{"pua": "\ue059", "chinese_id": "向上雙V箭頭"}],
            }
        ],
        "candidate_text": "向上雙V箭頭",
        "detection_count": 1,
        "vision_path": "yolo_ocr",
        "vision_path_reason": "沒有 UIA 快照",
    }
    yolo_path = run / "yolo_ocr" / "event_012.json"
    yolo_path.write_text(json.dumps(yolo_payload, ensure_ascii=False), encoding="utf-8")
    request = UiaReplayRepairRequest(
        recording_dir=run,
        event_index=12,
        side="click",
        recorded_bbox=(820, 1035, 31, 40),
        recorded_text="向上雙V箭頭",
        recorded_window_xywh=(0, 1032, 1920, 48),
    )
    instruction = write_uia_replay_repair(
        request,
        label="顯示桌面",
        source="findall",
        bbox=(800, 1040, 40, 30),
        others=[],
    )
    assert instruction is not None
    assert instruction.startswith("（UIA）")
    assert "顯示桌面" in instruction
    assert "向上雙V箭頭" not in instruction
    saved = json.loads((run / "analysis" / "event_012.json").read_text(encoding="utf-8"))
    assert saved["uia_replay_repair"]["label"] == "顯示桌面"
    assert saved["yolo_ocr_fallback"]["instruction"] == yolo_instruction


def test_replace_anchor_uses_chinese_id_not_pua() -> None:
    from src.recorder.uia_replay_repair import _replace_anchor

    request = UiaReplayRepairRequest(
        recording_dir=Path("."),
        event_index=12,
        side="click",
        recorded_bbox=(820, 1035, 31, 40),
        recorded_text="向上雙V箭頭",
        recorded_window_xywh=(0, 1032, 1920, 48),
    )
    rewritten = _replace_anchor(
        "（YOLO+OCR）將滑鼠移到「向上雙V箭頭」圖示，並點擊滑鼠一下。",
        request,
        "顯示桌面",
    )
    assert rewritten == "（YOLO+OCR）將滑鼠移到「顯示桌面」圖示，並點擊滑鼠一下。"
    pua_request = UiaReplayRepairRequest(
        recording_dir=Path("."),
        event_index=12,
        side="click",
        recorded_bbox=(820, 1035, 31, 40),
        recorded_text="\ue059",
        recorded_window_xywh=(0, 1032, 1920, 48),
    )
    assert (
        _replace_anchor(
            "（YOLO+OCR）將滑鼠移到「向上雙V箭頭」圖示，並點擊滑鼠一下。",
            pua_request,
            "顯示桌面",
        )
        is None
    )


def test_labeled_ocr_confirm_boxes_includes_icon_ids() -> None:
    from cua_mcp.select_mouse_target import _labeled_ocr_confirm_boxes, _labeled_text_boxes
    from cua_mcp.select_ui_element import UiDetection
    from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT

    text_det = UiDetection(
        bbox=(10, 20, 40, 16),
        cx=30,
        cy=28,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="搜尋",
        source="ocr",
    )
    icon_det = UiDetection(
        bbox=(820, 1035, 31, 40),
        cx=835,
        cy=1055,
        class_id=YOLO_CLASS_ELEMENT,
        class_name="element",
        text="\ue059",
        icons=[{"pua": "\ue059", "chinese_id": "向上雙V箭頭"}],
        source="ocr",
    )
    uia_text = UiDetection(
        bbox=(50, 50, 20, 10),
        cx=60,
        cy=55,
        class_id=YOLO_CLASS_TEXT,
        class_name="text",
        text="Ignore",
        source="uia",
    )
    boxes = _labeled_ocr_confirm_boxes([text_det, icon_det, uia_text])
    assert ((10, 20, 40, 16), "搜尋") in boxes
    assert ((820, 1035, 31, 40), "向上雙V箭頭") in boxes
    assert all(label != "Ignore" for _box, label in boxes)
    assert all(label != "\ue059" for _box, label in boxes)
    text_only = _labeled_text_boxes([text_det, icon_det, uia_text], source="ocr")
    assert text_only == [((10, 20, 40, 16), "搜尋")]


def test_recorded_primary_prefers_input_over_neighbor_ocr() -> None:
    from src.recorder.uia_replay_repair import _recorded_primary

    box, text = _recorded_primary(
        {
            "candidates": [
                {
                    "bbox": [222, 242, 848, 45],
                    "text": None,
                    "class_name": "input",
                },
                {
                    "bbox": [344, 296, 71, 15],
                    "text": "料編輯列",
                    "class_name": "text",
                },
            ]
        }
    )
    assert box == (222, 242, 848, 45)
    assert text == "輸入欄"


def test_recorded_primary_prefers_scrollbar_class() -> None:
    from src.recorder.uia_replay_repair import _recorded_primary

    box, text = _recorded_primary(
        {
            "candidates": [
                {
                    "bbox": [1900, 100, 20, 800],
                    "text": None,
                    "class_name": "scrollbar",
                },
                {
                    "bbox": [10, 20, 40, 16],
                    "text": "內容",
                    "class_name": "text",
                },
            ]
        }
    )
    assert box == (1900, 100, 20, 800)
    assert text == "滾動條"


def test_replace_anchor_bare_input_class() -> None:
    from src.recorder.uia_replay_repair import _replace_anchor

    request = UiaReplayRepairRequest(
        recording_dir=Path("."),
        event_index=5,
        side="click",
        recorded_bbox=(222, 242, 848, 45),
        recorded_text="輸入欄",
        recorded_window_xywh=(0, 0, 1920, 1080),
    )
    rewritten = _replace_anchor(
        "（YOLO+OCR）將滑鼠移到輸入欄（在「字型」文字的下面），並點擊滑鼠一下。",
        request,
        "公式列",
    )
    # Nearby parentheticals are stripped before the class-token rewrite.
    assert rewritten == "（YOLO+OCR）將滑鼠移到「公式列」文字，並點擊滑鼠一下。"


def test_labeled_ocr_confirm_boxes_includes_input_and_scrollbar() -> None:
    from cua_mcp.select_mouse_target import _labeled_ocr_confirm_boxes
    from cua_mcp.select_ui_element import UiDetection
    from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT, YOLO_CLASS_SCROLLBAR

    uia_input = UiDetection(
        bbox=(222, 242, 848, 45),
        cx=646,
        cy=264,
        class_id=YOLO_CLASS_INPUT,
        class_name="input",
        text=None,
        source="uia",
    )
    yolo_scrollbar = UiDetection(
        bbox=(1900, 100, 20, 800),
        cx=1910,
        cy=500,
        class_id=YOLO_CLASS_SCROLLBAR,
        class_name="scrollbar",
        text=None,
        source="yolo",
    )
    boxes = _labeled_ocr_confirm_boxes([uia_input, yolo_scrollbar])
    assert ((222, 242, 848, 45), "輸入欄") in boxes
    assert ((1900, 100, 20, 800), "滾動條") in boxes
