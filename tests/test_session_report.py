from __future__ import annotations

import csv
import json
from pathlib import Path

from src.common.session_report import (
    build_session_report,
    should_write_session_report,
    write_session_report,
)


def _write_step(
    steps_dir: Path,
    transcript_counter: int,
    script_step_index: int,
    *,
    goal: str,
    messages: list[dict],
    status: str = "completed",
    expected_outcome: str | None = None,
    verify: dict | None = None,
) -> None:
    steps_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "messages": messages,
        "step_timing": {
            "started_at_utc": "2026-06-11T06:00:00+00:00",
            "finished_at_utc": "2026-06-11T06:00:10+00:00",
            "duration_seconds": 10.0,
            "status": status,
            "step_index": script_step_index,
            "goal": goal,
        },
    }
    if expected_outcome is not None:
        payload["step_timing"]["expected_outcome"] = expected_outcome
    if verify is not None:
        payload["step_timing"]["verify"] = verify
    (steps_dir / f"{transcript_counter}_{script_step_index}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_hand_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["timestamp", "action", "args", "ok", "screenshot_name", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_should_write_session_report_skips_script_continue(tmp_path: Path) -> None:
    assert should_write_session_report(script_finished=True, user_continues_runtime=True) is False
    assert should_write_session_report(script_finished=True, user_continues_runtime=False) is True
    assert should_write_session_report(script_finished=False, user_continues_runtime=False) is True


def test_build_session_report_aggregates_steps_tools_and_profile(tmp_path: Path) -> None:
    run_root = tmp_path / "task_demo"
    run_root.mkdir()
    steps_dir = run_root / "steps"

    _write_step(
        steps_dir,
        0,
        0,
        goal="Open File Explorer",
        messages=[
            {
                "role": "user",
                "timestamp_utc": "2026-06-11T06:00:00+00:00",
                "content": "observe",
            },
            {
                "role": "assistant",
                "timestamp_utc": "2026-06-11T06:00:02+00:00",
                "content": "decide",
                "tool_calls": [
                    {
                        "function": {
                            "name": "click",
                            "arguments": {"x": 1, "y": 2},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "timestamp_utc": "2026-06-11T06:00:05+00:00",
                "content": json.dumps({"action": "click", "ok": True, "args": {"x": 1, "y": 2}}),
            },
        ],
        expected_outcome="Explorer window is open",
        verify={
            "accomplished": True,
            "branch": "advance",
            "target_step": None,
            "clearly_unmet": False,
            "reason": "Explorer is visible",
        },
    )
    _write_step(
        steps_dir,
        1,
        0,
        goal="",
        messages=[],
        status="completed",
    )
    step_one_path = steps_dir / "1_0.json"
    step_one_payload = json.loads(step_one_path.read_text(encoding="utf-8"))
    step_one_payload["step_timing"].update(
        {
            "started_at_utc": "2026-06-11T06:00:15+00:00",
            "finished_at_utc": "2026-06-11T06:00:25+00:00",
            "duration_seconds": 10.0,
            "goal": "Type hello",
        }
    )
    step_one_path.write_text(json.dumps(step_one_payload), encoding="utf-8")
    (run_root / "runtime_commands.txt").write_text("Runtime fallback\n", encoding="utf-8")
    _write_hand_csv(
        run_root / "hand.csv",
        [
            {
                "timestamp": "2026-06-11T06:00:06+00:00",
                "action": "click",
                "args": json.dumps({"x": 1, "y": 2}),
                "ok": "True",
                "screenshot_name": "shot.png",
                "message": "",
            },
            {
                "timestamp": "2026-06-11T06:00:20+00:00",
                "action": "type",
                "args": json.dumps({"text": "hello"}),
                "ok": "False",
                "screenshot_name": "",
                "message": "failed",
            },
        ],
    )

    report = build_session_report(run_root, session_end_reason="completed")

    assert report["version"] == 1
    assert report["run_id"] == "task_demo"
    assert report["session_end_reason"] == "completed"
    assert report["summary"]["step_count"] == 2
    assert report["summary"]["tool_call_count"] == 2
    assert report["summary"]["failed_tool_count"] == 1
    assert report["summary"]["total_duration_seconds"] == 20.0

    assert report["steps"][0]["goal"] == "Open File Explorer"
    assert report["steps"][1]["goal"] == "Type hello"
    assert report["steps"][0]["expected_outcome"] == "Explorer window is open"
    assert report["steps"][0]["verify"] == {
        "accomplished": True,
        "branch": "advance",
        "target_step": None,
        "clearly_unmet": False,
        "reason": "Explorer is visible",
    }
    assert report["steps"][1].get("verify") is None or "verify" not in report["steps"][1]

    profile = report["steps"][0]["time_profile"]
    assert profile[0]["kind"] == "llm_inference"
    assert profile[0]["duration_seconds"] == 2.0
    assert profile[1]["kind"] == "tool_execution"
    assert profile[1]["actions"] == ["click"]
    assert profile[2]["kind"] == "step_wrap_up"
    assert profile[2]["action"] == "click"
    assert profile[2]["ok"] is True

    timing_summary = report["steps"][0]["timing_summary"]
    assert timing_summary["execution_llm_seconds"] == 2.0
    assert timing_summary["tool_execution_seconds"] == 3.0
    assert timing_summary["total_seconds"] == 10.0
    assert report["summary"]["avg_step_seconds"] == 10.0
    assert report["summary"]["execution_llm_seconds"] == 2.0

    tool_results = report["tool_results"]
    assert [item["action"] for item in tool_results] == ["click", "type"]
    assert tool_results[0]["transcript_counter"] == 0
    assert tool_results[1]["transcript_counter"] == 1
    assert tool_results[1]["ok"] is False


def test_build_session_report_splits_verify_llm_from_verification(tmp_path: Path) -> None:
    run_root = tmp_path / "task_verify_profile"
    run_root.mkdir()
    steps_dir = run_root / "steps"
    steps_dir.mkdir()
    payload = {
        "messages": [
            {
                "role": "user",
                "timestamp_utc": "2026-06-11T06:00:00+00:00",
                "content": "observe",
            },
            {
                "role": "assistant",
                "timestamp_utc": "2026-06-11T06:00:03+00:00",
                "content": "done",
            },
        ],
        "verification": [
            {
                "role": "user",
                "timestamp_utc": "2026-06-11T06:00:05+00:00",
                "content": "verify",
            },
            {
                "role": "assistant",
                "timestamp_utc": "2026-06-11T06:00:08+00:00",
                "content": '{"accomplished": true}',
            },
        ],
        "step_timing": {
            "started_at_utc": "2026-06-11T06:00:00+00:00",
            "finished_at_utc": "2026-06-11T06:00:10+00:00",
            "duration_seconds": 10.0,
            "status": "completed",
            "step_index": 0,
            "goal": "Open notepad",
            "expected_outcome": "Notepad is open",
            "verify": {
                "accomplished": True,
                "branch": "advance",
                "target_step": None,
                "clearly_unmet": False,
                "reason": "ok",
            },
        },
    }
    (steps_dir / "0_0.json").write_text(json.dumps(payload), encoding="utf-8")

    report = build_session_report(run_root, session_end_reason="completed")
    step = report["steps"][0]
    profile = step["time_profile"]
    kinds = [entry["kind"] for entry in profile]
    assert "llm_inference" in kinds
    assert "verify_llm_inference" in kinds

    actor_llm = next(entry for entry in profile if entry["kind"] == "llm_inference")
    assert actor_llm["duration_seconds"] == 3.0
    # Actor wrap-up ends at verify start (06:00:03 -> 06:00:05)
    actor_done = next(entry for entry in profile if entry["kind"] == "step_completion")
    assert actor_done["duration_seconds"] == 2.0

    verify_llm = next(entry for entry in profile if entry["kind"] == "verify_llm_inference")
    assert verify_llm["duration_seconds"] == 3.0

    summary = step["timing_summary"]
    assert summary["execution_llm_seconds"] == 3.0
    assert summary["verify_llm_seconds"] == 3.0
    assert summary["total_seconds"] == 10.0
    assert report["summary"]["verify_llm_seconds"] == 3.0
    assert report["summary"]["execution_llm_seconds"] == 3.0
    assert report["summary"]["avg_step_seconds"] == 10.0


def test_time_profile_labels_post_tool_screenshot_only_before_decide_user(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "task_decide_screenshot"
    run_root.mkdir()
    steps_dir = run_root / "steps"
    _write_step(
        steps_dir,
        0,
        0,
        goal="Move then click",
        messages=[
            {
                "role": "user",
                "timestamp_utc": "2026-06-11T06:00:00+00:00",
                "content": "decide",
                "images": ["shot1.png"],
            },
            {
                "role": "assistant",
                "timestamp_utc": "2026-06-11T06:00:02+00:00",
                "tool_calls": [
                    {"function": {"name": "move_mouse", "arguments": {"instruction": "target"}}}
                ],
            },
            {
                "role": "tool",
                "timestamp_utc": "2026-06-11T06:00:05+00:00",
                "content": json.dumps({"action": "move_mouse", "ok": True}),
            },
            {
                "role": "user",
                "timestamp_utc": "2026-06-11T06:00:06+00:00",
                "content": "decide next",
                "images": ["shot2.png"],
            },
            {
                "role": "assistant",
                "timestamp_utc": "2026-06-11T06:00:08+00:00",
                "content": '{"status":"completed"}',
            },
        ],
    )

    report = build_session_report(run_root, session_end_reason="completed")
    profile = report["steps"][0]["time_profile"]
    assert profile[2]["kind"] == "screenshot_capture"
    assert profile[2]["action"] == "move_mouse"


def test_time_profile_labels_tool_to_tool_gap_as_next_tool_execution(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "task_cache_replay"
    run_root.mkdir()
    steps_dir = run_root / "steps"
    _write_step(
        steps_dir,
        0,
        0,
        goal="Replay move and click",
        messages=[
            {
                "role": "user",
                "timestamp_utc": "2026-06-11T06:00:00+00:00",
                "content": "Cache replay for task: Replay move and click",
                "images": ["shot1.png"],
                "cache_replay": True,
            },
            {
                "role": "assistant",
                "timestamp_utc": "2026-06-11T06:00:00+00:00",
                "cache_replay": True,
                "tool_calls": [
                    {"function": {"name": "move_mouse", "arguments": {"instruction": "target"}}},
                    {"function": {"name": "click", "arguments": {"button": "left"}}},
                ],
            },
            {
                "role": "tool",
                "timestamp_utc": "2026-06-11T06:00:03+00:00",
                "content": json.dumps({"action": "move_mouse", "ok": True}),
            },
            {
                "role": "tool",
                "timestamp_utc": "2026-06-11T06:00:05+00:00",
                "content": json.dumps({"action": "click", "ok": True}),
            },
        ],
    )

    report = build_session_report(run_root, session_end_reason="completed")
    profile = report["steps"][0]["time_profile"]
    assert profile[1]["kind"] == "tool_execution"
    assert profile[1]["actions"] == ["move_mouse", "click"]
    assert profile[1]["duration_seconds"] == 3.0
    assert profile[2]["kind"] == "tool_execution"
    assert profile[2]["actions"] == ["click"]
    assert profile[2]["action"] == "click"
    assert profile[2]["duration_seconds"] == 2.0
    assert profile[3]["kind"] == "step_wrap_up"
    assert profile[3]["action"] == "click"
    assert report["steps"][0]["timing_summary"]["screenshot_seconds"] == 0.0
    assert report["steps"][0]["timing_summary"]["tool_execution_seconds"] == 5.0


def test_time_profile_includes_move_mouse_internal_timing_details(tmp_path: Path) -> None:
    run_root = tmp_path / "task_move_mouse_timing"
    run_root.mkdir()
    steps_dir = run_root / "steps"
    move_timing = {
        "total_s": 4.5,
        "capture_s": 0.2,
        "yolo_s": 1.0,
        "line_s": 0.1,
        "ocr_s": 2.5,
        "parse_s": 0.3,
        "select_s": 0.4,
        "hand_move_s": 0.01,
        "phases": [
            {"name": "capture", "seconds": 0.2},
            {"name": "yolo", "seconds": 1.0},
            {"name": "line_refine", "seconds": 0.1},
            {"name": "ocr", "seconds": 2.5},
            {"name": "parse_instruction", "seconds": 0.3, "overlapped": True},
            {"name": "llm_pick", "seconds": 0.4},
            {"name": "hand_move", "seconds": 0.01},
        ],
    }
    _write_step(
        steps_dir,
        0,
        0,
        goal="Move to target",
        messages=[
            {
                "role": "user",
                "timestamp_utc": "2026-06-11T06:00:00+00:00",
                "content": "Cache replay",
                "images": ["shot1.png"],
                "cache_replay": True,
            },
            {
                "role": "assistant",
                "timestamp_utc": "2026-06-11T06:00:00+00:00",
                "cache_replay": True,
                "tool_calls": [
                    {"function": {"name": "move_mouse", "arguments": {"instruction": "t"}}},
                    {"function": {"name": "click", "arguments": {"button": "left"}}},
                ],
            },
            {
                "role": "tool",
                "timestamp_utc": "2026-06-11T06:00:05+00:00",
                "content": json.dumps(
                    {
                        "action": "move_mouse",
                        "ok": True,
                        "args": {
                            "instruction": "t",
                            "timing": move_timing,
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "timestamp_utc": "2026-06-11T06:00:06+00:00",
                "content": json.dumps({"action": "click", "ok": True}),
            },
        ],
    )

    report = build_session_report(run_root, session_end_reason="completed")
    profile = report["steps"][0]["time_profile"]
    move_entry = profile[1]
    assert move_entry["kind"] == "tool_execution"
    assert "move_mouse" in move_entry["actions"]
    assert move_entry["tool_internal_seconds"] == 4.5
    details = move_entry["details"]
    assert details[0]["kind"] == "move_mouse_capture"
    assert details[1]["kind"] == "move_mouse_yolo"
    assert details[1]["duration_seconds"] == 1.0
    assert details[3]["kind"] == "move_mouse_ocr"
    assert details[3]["duration_seconds"] == 2.5
    assert any(d["kind"] == "move_mouse_llm_pick" for d in details)
    assert any(d["kind"] == "move_mouse_hand_move" for d in details)


def test_write_session_report_creates_report_json(tmp_path: Path) -> None:
    run_root = tmp_path / "task_write"
    run_root.mkdir()
    steps_dir = run_root / "steps"
    _write_step(steps_dir, 0, 0, goal="do thing", messages=[])

    report_path = write_session_report(run_root, session_end_reason="user_ended")

    assert report_path == run_root / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["session_end_reason"] == "user_ended"
    assert payload["steps"][0]["goal"] == "do thing"
