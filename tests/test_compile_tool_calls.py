from __future__ import annotations

import json
from pathlib import Path

from src.common.instruction_tool_cache import lookup_tool_calls
from src.recorder.compile_tool_calls import (
    compile_tool_calls,
    ensure_recording_tool_cache,
    rebuild_recording_instruction_tool_cache,
    recording_tool_cache_path,
    set_analysis_tool_calls,
)
from src.recorder.models import RecordedEvent


def _event(**kwargs) -> RecordedEvent:
    base = {
        "index": 1,
        "timestamp_utc": "t",
        "kind": "click",
    }
    base.update(kwargs)
    return RecordedEvent(**base)


def test_compile_text_input() -> None:
    event = _event(kind="text_input", text="hello")
    instruction = "輸入「hello」"
    calls = compile_tool_calls(event, instruction)
    assert calls == [
        {"name": "type_text", "arguments": {"text": "hello", "instruction": instruction}}
    ]


def test_compile_key_press() -> None:
    event = _event(kind="key_press", key="enter")
    instruction = "按下 Enter 鍵"
    calls = compile_tool_calls(event, instruction)
    assert calls == [
        {"name": "press_key", "arguments": {"key": "enter", "instruction": instruction}}
    ]


def test_compile_hotkey() -> None:
    event = _event(kind="hotkey", keys=["ctrl", "a"])
    instruction = "按下 Ctrl+A"
    calls = compile_tool_calls(event, instruction)
    assert calls == [
        {
            "name": "hotkey",
            "arguments": {"keys": ["ctrl", "a"], "instruction": instruction},
        }
    ]


def test_compile_wait() -> None:
    event = _event(kind="wait", duration_seconds=2.5)
    instruction = "等待 3 秒"
    calls = compile_tool_calls(event, instruction)
    assert calls == [
        {"name": "wait", "arguments": {"seconds": 2.5, "instruction": instruction}}
    ]


def test_compile_click_move_and_click() -> None:
    event = _event(kind="click", click_count=1)
    instruction = "將滑鼠移到「影片」文字，並點擊滑鼠一下。"
    calls = compile_tool_calls(event, instruction)
    assert calls is not None
    assert [c["name"] for c in calls] == ["move_mouse", "click"]
    assert calls[0]["arguments"] == {"instruction": "「影片」文字"}
    assert calls[1]["arguments"]["clicks"] == 1
    assert calls[1]["arguments"]["instruction"] == instruction


def test_compile_click_with_directed_nearby() -> None:
    event = _event(kind="click", click_count=1)
    instruction = (
        "將滑鼠移到「搜尋」文字（在輸入欄的裡面、在「取消圓圈」圖示的右下方、"
        "在「多雲時睛」文字的右邊、在「2026/9/15」文字的左邊），並點擊滑鼠一下。"
    )
    calls = compile_tool_calls(event, instruction)
    assert calls is not None
    assert calls[0] == {
        "name": "move_mouse",
        "arguments": {
            "instruction": "「搜尋」文字",
            "nearby_objects": [
                "在輸入欄的裡面",
                "在「取消圓圈」圖示的右下方",
                "在「多雲時睛」文字的右邊",
                "在「2026/9/15」文字的左邊",
            ],
        },
    }
    assert calls[1]["name"] == "click"


def test_compile_click_keeps_pixel_offset_on_target() -> None:
    event = _event(kind="click", click_count=1)
    instruction = (
        "將滑鼠移到「自訂Office 範本」文字左方14個像素、下方39個像素的位置，並點擊滑鼠一下。"
    )
    calls = compile_tool_calls(event, instruction)
    assert calls is not None
    assert calls[0]["arguments"]["instruction"] == (
        "「自訂Office 範本」文字左方14個像素、下方39個像素的位置"
    )


def test_compile_right_click() -> None:
    event = _event(kind="right_click")
    instruction = "將滑鼠移到「檔案」文字，用右鍵點選。"
    calls = compile_tool_calls(event, instruction)
    assert calls is not None
    assert [c["name"] for c in calls] == ["move_mouse", "right_click"]
    assert calls[0]["arguments"]["instruction"] == "「檔案」文字"


def test_compile_scroll_with_hover() -> None:
    event = _event(kind="scroll", scroll_delta=-3)
    instruction = "在「清單」文字附近向下捲動"
    calls = compile_tool_calls(event, instruction)
    assert calls is not None
    assert [c["name"] for c in calls] == ["move_mouse", "scroll"]
    assert calls[0]["arguments"]["instruction"] == "「清單」文字"
    assert calls[1]["arguments"]["clicks"] == 3


def test_compile_window_maximize() -> None:
    event = _event(
        kind="click",
        window_change={
            "action": "maximize",
            "title": "記事本",
            "confidence": "high",
        },
    )
    instruction = "最大化「記事本」視窗"
    calls = compile_tool_calls(event, instruction)
    assert calls == [
        {
            "name": "maximize_windows",
            "arguments": {
                "window_title_contains": "記事本",
                "instruction": instruction,
            },
        }
    ]


def test_compile_drag() -> None:
    event = _event(kind="drag")
    instruction = "從「A」圖示拖到「B」文字（附近有「C」文字）"
    calls = compile_tool_calls(event, instruction)
    assert calls == [
        {
            "name": "drag",
            "arguments": {
                "start_instruction": "「A」圖示",
                "destination_instruction": "「B」文字",
                "destination_nearby_objects": ["「C」文字"],
            },
        }
    ]


def test_compile_drag_with_start_and_end_landmarks() -> None:
    event = _event(kind="drag")
    instruction = (
        "從「A」圖示（起點在「Desktop」文字的左邊）拖到「B」文字"
        "（終點在「新增」文字的上面）"
    )
    calls = compile_tool_calls(event, instruction)
    assert calls == [
        {
            "name": "drag",
            "arguments": {
                "start_instruction": "「A」圖示",
                "destination_instruction": "「B」文字",
                "start_nearby_objects": ["在「Desktop」文字的左邊"],
                "destination_nearby_objects": ["在「新增」文字的上面"],
            },
        }
    ]


def test_compile_rejects_condition() -> None:
    event = _event(kind="condition")
    assert compile_tool_calls(event, "如果畫面上有「X」") is None


def test_ensure_upgrades_full_hub_script_move_mouse(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir(parents=True)
    event = _event(kind="click", click_count=1)
    (run_dir / "events" / "event_001.json").write_text(
        json.dumps(event.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    instruction = (
        "將滑鼠移到「搜尋」文字（在輸入欄的裡面），並點擊滑鼠一下。"
    )
    stale = [
        {
            "name": "move_mouse",
            "arguments": {"instruction": instruction},
        },
        {
            "name": "click",
            "arguments": {"button": "left", "clicks": 1, "instruction": instruction},
        },
    ]
    (run_dir / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "instruction": instruction,
                "tool_calls": stale,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ensure_recording_tool_cache(run_dir)
    analysis = json.loads(
        (run_dir / "analysis" / "event_001.json").read_text(encoding="utf-8")
    )
    assert analysis["tool_calls"][0]["arguments"] == {
        "instruction": "「搜尋」文字",
        "nearby_objects": ["在輸入欄的裡面"],
    }


def test_rebuild_and_ensure_recording_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir(parents=True)
    event = _event(kind="text_input", text="abc")
    (run_dir / "events" / "event_001.json").write_text(
        json.dumps(event.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    instruction = "輸入「abc」"
    analysis = {"event_index": 1, "instruction": instruction}
    (run_dir / "analysis" / "event_001.json").write_text(
        json.dumps(analysis, ensure_ascii=False),
        encoding="utf-8",
    )

    ensure_recording_tool_cache(run_dir)
    analysis_after = json.loads(
        (run_dir / "analysis" / "event_001.json").read_text(encoding="utf-8")
    )
    assert analysis_after["tool_calls"][0]["name"] == "type_text"
    cache_path = recording_tool_cache_path(run_dir)
    assert cache_path.is_file()
    assert lookup_tool_calls(instruction, path=cache_path) == analysis_after["tool_calls"]

    analysis_after["instruction"] = "輸入「xyz」"
    set_analysis_tool_calls(analysis_after, event, "輸入「xyz」")
    (run_dir / "analysis" / "event_001.json").write_text(
        json.dumps(analysis_after, ensure_ascii=False),
        encoding="utf-8",
    )
    rebuild_recording_instruction_tool_cache(run_dir)
    assert lookup_tool_calls(instruction, path=cache_path) is None
    assert lookup_tool_calls("輸入「xyz」", path=cache_path) is not None
