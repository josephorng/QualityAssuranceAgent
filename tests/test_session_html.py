from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image

from src.common.session_html import (
    _typed_text_candidates,
    recording_html_path,
    runs_index_html_path,
    session_html_path,
    write_recording_html_from_run,
    write_runs_index_html,
    write_session_html_from_run,
)


def _make_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(20, 120, 200)).save(path)
    return path


def _make_jpeg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(20, 120, 200)).save(path, format="JPEG")
    return path


def _write_recording_fixture(
    run_root: Path,
    *,
    with_analysis: bool = True,
    with_html: bool = False,
) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "events").mkdir(exist_ok=True)
    (run_root / "screenshots").mkdir(exist_ok=True)
    shot = _make_jpeg(run_root / "screenshots" / "event_001.jpeg")
    event = {
        "index": 1,
        "timestamp_utc": "2026-07-21T04:00:00+00:00",
        "kind": "click",
        "cursor_xy": [10, 20],
        "screenshot_path": str(shot),
        "text": None,
    }
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(event, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:30+00:00",
                "event_count": 1,
                "events": ["events/event_001.json"],
            }
        ),
        encoding="utf-8",
    )
    if with_analysis:
        (run_root / "analysis").mkdir(exist_ok=True)
        (run_root / "analysis" / "event_001.json").write_text(
            json.dumps({"event_index": 1, "instruction": "點擊「搜尋」按鈕"}, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_root / "report.json").write_text(
            json.dumps(
                {
                    "run_id": run_root.name,
                    "recorded": 1,
                    "processed": 1,
                    "cached": 1,
                    "skipped": 0,
                    "cancelled": False,
                    "errors": [],
                    "instructions": ["點擊「搜尋」按鈕"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if with_html:
        (run_root / "recording_steps.html").write_text("<html></html>", encoding="utf-8")
    return run_root


_HAND_CSV_FIELDS = [
    "timestamp",
    "action",
    "args",
    "ok",
    "screenshot_name",
    "screenshot_before_path",
    "screenshot_after_path",
    "message",
]


def _write_step(
    run_root: Path,
    *,
    transcript_counter: int,
    script_step_index: int,
    goal: str,
    started_at: str,
    finished_at: str,
) -> None:
    steps_dir = run_root / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "messages": [],
        "step_timing": {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "duration_seconds": 10.0,
            "status": "completed",
            "step_index": script_step_index,
            "goal": goal,
        },
    }
    (steps_dir / f"{transcript_counter}_{script_step_index}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_hand_csv(run_root: Path, rows: list[dict], *, header: bool = True) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "hand.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HAND_CSV_FIELDS)
        if header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_write_session_html_renders_all_steps(tmp_path: Path) -> None:
    run_root = tmp_path / "task_html"
    before = _make_png(run_root / "eye" / "before.png")
    after = _make_png(run_root / "eye" / "after.png")

    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:00+00:00",
                "action": "click",
                "args": {"button": "left", "instruction": "搜尋欄"},
                "ok": True,
                "screenshot_name": str(before),
                "screenshot_before_path": str(before),
                "screenshot_after_path": str(after),
                "message": "executed",
            },
            {
                "timestamp": "2026-06-11T06:00:05+00:00",
                "action": "type_text",
                "args": {"text": "hello"},
                "ok": False,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "failed",
            },
        ],
    )

    path = write_session_html_from_run(run_root)

    assert path == session_html_path(run_root)
    html = path.read_text(encoding="utf-8")

    assert html.startswith("<!DOCTYPE html>")
    assert "<h1>task_html</h1>" in html
    assert "動作 1：click" in html
    assert "動作 2：type_text" in html
    assert '<a href="../index.html">← 報告列表</a>' in html
    assert html.count('<details class="instruction-group">') == 1
    assert '<span class="instruction-number">1.</span>' in html
    assert "手部動作" in html
    assert html.count('<details class="args"><summary>參數</summary>') == 2
    assert "<th>button</th><td>left</td>" in html
    assert "<dt>指令</dt><dd>搜尋欄</dd>" in html
    assert "<th>text</th><td>hello</td>" in html
    assert "<details open" not in html
    assert "成功" in html
    assert "失敗" in html
    # first step references screenshots relative to the run folder
    assert 'src="eye/before.png"' in html
    assert 'src="eye/after.png"' in html
    # second step has no screenshots
    assert "無螢幕截圖" in html


def test_write_session_html_groups_hand_operations_by_user_instruction(tmp_path: Path) -> None:
    run_root = tmp_path / "task_grouped"
    _write_step(
        run_root,
        transcript_counter=0,
        script_step_index=0,
        goal="最小化所有視窗。",
        started_at="2026-06-11T06:00:00+00:00",
        finished_at="2026-06-11T06:00:10+00:00",
    )
    _write_step(
        run_root,
        transcript_counter=1,
        script_step_index=0,
        goal="點擊「資料夾」圖示",
        started_at="2026-06-11T06:00:11+00:00",
        finished_at="2026-06-11T06:00:20+00:00",
    )
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:05+00:00",
                "action": "minimize_windows",
                "args": {"instruction": "最小化所有視窗"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            },
            {
                "timestamp": "2026-06-11T06:00:12+00:00",
                "action": "move_mouse",
                "args": {"instruction": "「資料夾」圖示"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            },
            {
                "timestamp": "2026-06-11T06:00:15+00:00",
                "action": "click",
                "args": {"instruction": "「資料夾」圖示"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            },
        ],
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert html.count('<details class="instruction-group">') == 2
    assert '<span class="instruction-number">1.</span>' in html
    assert '<span class="instruction-number">2.</span>' in html
    assert "最小化所有視窗。" in html
    assert "點擊「資料夾」圖示" in html
    assert html.index("最小化所有視窗。") < html.index("點擊「資料夾」圖示")
    assert html.index('instruction-number">1.') < html.index("最小化所有視窗。")
    assert html.index('instruction-number">2.') < html.index("點擊「資料夾」圖示")
    assert html.index("最小化所有視窗。") < html.index("動作 1：minimize_windows")
    assert html.index("點擊「資料夾」圖示") < html.index("動作 2：move_mouse")
    assert "動作 3：click" in html
    assert "<details open" not in html


def test_write_session_html_includes_verification_result(tmp_path: Path) -> None:
    run_root = tmp_path / "task_verify_html"
    _write_step(
        run_root,
        transcript_counter=0,
        script_step_index=0,
        goal="點擊「搜尋」",
        started_at="2026-06-11T06:00:00+00:00",
        finished_at="2026-06-11T06:00:10+00:00",
    )
    step_path = run_root / "steps" / "0_0.json"
    payload = json.loads(step_path.read_text(encoding="utf-8"))
    payload["step_timing"].update(
        {
            "status": "verify_failed",
            "expected_outcome": "搜尋介面已開啟。",
            "verify": {
                "accomplished": False,
                "branch": "retry",
                "target_step": None,
                "clearly_unmet": True,
                "reason": "Search panel is still closed",
            },
        }
    )
    step_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:05+00:00",
                "action": "click",
                "args": {"instruction": "「搜尋」"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert '<span class="badge fail">retry</span>' in html
    assert 'session-verify-title">驗證結果' in html
    assert "<dt>Status</dt><dd>verify_failed</dd>" in html
    assert "預期結果：搜尋介面已開啟。" in html
    assert "session-expected-outcome" not in html
    assert "<dt>Accomplished</dt><dd>false</dd>" in html
    assert "<dt>Branch</dt><dd>retry</dd>" in html
    assert "<dt>Clearly unmet</dt><dd>true</dd>" in html
    assert "<dt>Reason</dt><dd>Search panel is still closed</dd>" in html
    assert "動作 1：click" in html
    assert "<dt>Expected</dt>" not in html


def test_write_session_html_merges_smart_cycle_with_executed_tools(tmp_path: Path) -> None:
    run_root = tmp_path / "smart_20260730_090228_245442"
    timestamp = "2026-07-30T09:02:46+00:00"
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": timestamp,
                "action": "double_click",
                "args": {"instruction": "Telegram 圖示"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )
    (run_root / "report.json").write_text(
        json.dumps(
            {
                "smart_goal": "Send a message",
                "smart_cycles": [
                    {
                        "cycle": 1,
                        "plan": {
                            "instruction": "double_click(Telegram 圖示)",
                            "expected_outcome": "Telegram opens",
                            "rationale": "Open Telegram first",
                        },
                        "act": {"ok": True, "reason": "Actor completed"},
                        "verify": {
                            "branch": "advance",
                            "reason": "Telegram is visible",
                            "updated_state": "Telegram is open",
                        },
                    },
                    {
                        "cycle": 2,
                        "plan": {
                            "status": "completed",
                            "rationale": "The goal is complete",
                        },
                        "act": None,
                        "verify": None,
                    },
                ],
                "steps": [
                    {
                        "transcript_counter": 0,
                        "script_step_index": 0,
                        "goal": "double_click(Telegram 圖示)",
                    }
                ],
                "tool_results": [
                    {
                        "transcript_counter": 0,
                        "script_step_index": 0,
                        "timestamp_utc": timestamp,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert html.count('<details class="instruction-group">') == 2
    assert html.count("Executed tools (1)") == 1
    assert "Open Telegram first" in html
    assert "Telegram opens" in html
    assert "Actor completed" in html
    assert "Telegram is visible" in html
    assert "Telegram is open" in html
    assert "動作 1：double_click" in html
    assert 'href="../index.html#smart"' in html
    first_cycle_end = html.index("</details>", html.index("Open Telegram first"))
    assert html.index("動作 1：double_click") < first_cycle_end
    assert html.index("The goal is complete") > first_cycle_end


def test_write_session_html_shows_instruction_above_status(tmp_path: Path) -> None:
    run_root = tmp_path / "task_instruction"
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:00+00:00",
                "action": "click",
                "args": {"button": "left", "instruction": "點擊「搜尋」欄位"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert "<dt>指令</dt><dd>點擊「搜尋」欄位</dd>" in html
    assert html.index("點擊「搜尋」欄位") < html.index("<dt>狀態</dt>")
    assert "<th>instruction</th>" not in html
    assert "動作 1：click" in html


def test_write_session_html_omits_instruction_when_absent(tmp_path: Path) -> None:
    run_root = tmp_path / "task_no_instruction"
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:00+00:00",
                "action": "press_key",
                "args": {"key": "enter"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert "<dt>指令</dt>" not in html


def test_write_session_html_formats_timestamp(tmp_path: Path) -> None:
    run_root = tmp_path / "task_time"
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-07-17T06:30:32.126232+00:00",
                "action": "click",
                "args": {"button": "left"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    # Raw ISO artifacts should be gone, replaced by a readable form. The exact wall-clock value
    # depends on the local timezone, so assert on the format rather than a fixed date.
    assert "2026-07-17T06:30:32.126232+00:00" not in html
    assert ".126232" not in html
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:\d{2}\)", html)


def test_write_session_html_escapes_markup(tmp_path: Path) -> None:
    run_root = tmp_path / "task_escape"
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:00+00:00",
                "action": "type_text",
                "args": {"text": "<script>alert(1)</script>"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "done",
            }
        ],
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_write_session_html_handles_headerless_csv(tmp_path: Path) -> None:
    run_root = tmp_path / "task_headerless"
    before = _make_png(run_root / "eye" / "b.png")

    # Legacy hand.csv written without a header row.
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:00+00:00",
                "action": "move_mouse",
                "args": {"x": 1, "y": 2},
                "ok": True,
                "screenshot_name": str(before),
                "screenshot_before_path": str(before),
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
        header=False,
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    # The single data row is rendered (not consumed as a header).
    assert "動作 1：move_mouse" in html
    assert "<th>x</th><td>1</td>" in html
    assert "<th>y</th><td>2</td>" in html


def test_write_session_html_is_idempotent(tmp_path: Path) -> None:
    run_root = tmp_path / "task_twice"
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-06-11T06:00:00+00:00",
                "action": "click",
                "args": {"button": "left"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )

    write_session_html_from_run(run_root)
    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert html.count("動作 1：click") == 1


def test_write_session_html_without_hand_csv(tmp_path: Path) -> None:
    run_root = tmp_path / "task_no_actions"
    path = write_session_html_from_run(run_root)

    assert path.is_file()
    html = path.read_text(encoding="utf-8")
    assert "<h1>task_no_actions</h1>" in html
    assert "動作 1" not in html


def test_write_session_html_title_uses_script_name_and_datetime(tmp_path: Path) -> None:
    run_root = tmp_path / "task_20260722_060427_312875"
    run_root.mkdir()
    (run_root / "report.json").write_text(
        json.dumps(
            {
                "script_name": "click_folder.txt",
                "started_at_utc": "2026-07-22T06:04:27.373338+00:00",
            }
        ),
        encoding="utf-8",
    )
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-07-22T06:04:27+00:00",
                "action": "click",
                "args": {"button": "left"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )

    html = write_session_html_from_run(run_root).read_text(encoding="utf-8")

    assert "工作階段步驟紀錄：" not in html
    assert "task_20260722_060427_312875" not in html.split("<h1>")[1].split("</h1>")[0]
    assert "click_folder.txt · " in html
    assert re.search(
        r"<h1>click_folder\.txt · \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:\d{2}\)</h1>",
        html,
    )


def test_write_session_html_rebuilds_runs_index(tmp_path: Path) -> None:
    run_root = tmp_path / "task_20260721_120000_000001"
    _write_hand_csv(
        run_root,
        [
            {
                "timestamp": "2026-07-21T04:00:00+00:00",
                "action": "click",
                "args": {"button": "left"},
                "ok": True,
                "screenshot_name": "",
                "screenshot_before_path": "",
                "screenshot_after_path": "",
                "message": "executed",
            }
        ],
    )

    write_session_html_from_run(run_root)

    index_path = runs_index_html_path(tmp_path)
    assert index_path.is_file()
    index_html = index_path.read_text(encoding="utf-8")
    assert 'href="task_20260721_120000_000001/session_steps.html"' in index_html
    assert "task_20260721_120000_000001" in index_html


def test_write_runs_index_lists_multiple_runs_newest_first(tmp_path: Path) -> None:
    older = tmp_path / "task_20260720_100000_000001"
    newer = tmp_path / "task_20260721_100000_000002"
    for run_root in (older, newer):
        run_root.mkdir()
        (run_root / "session_steps.html").write_text("<html></html>", encoding="utf-8")
        (run_root / "report.json").write_text(
            json.dumps(
                {
                    "script_name": f"{run_root.name}.txt",
                    "started_at_utc": "2026-07-21T10:00:00+00:00"
                    if run_root is newer
                    else "2026-07-20T10:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

    skipped = tmp_path / "task_20260721_110000_000003"
    skipped.mkdir()
    (skipped / "report.json").write_text('{"run_id": "skipped"}', encoding="utf-8")

    index_path = write_runs_index_html(tmp_path)
    html = index_path.read_text(encoding="utf-8")

    assert 'href="task_20260721_100000_000002/session_steps.html"' in html
    assert 'href="task_20260720_100000_000001/session_steps.html"' in html
    assert "task_20260721_100000_000002.txt" in html
    assert "task_20260720_100000_000001.txt" in html
    assert "task_20260721_110000_000003" not in html
    assert html.index("task_20260721_100000_000002.txt") < html.index("task_20260720_100000_000001.txt")


def test_write_runs_index_includes_report_json_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "task_20260721_130000_000010"
    run_root.mkdir()
    (run_root / "session_steps.html").write_text("<html></html>", encoding="utf-8")
    (run_root / "report.json").write_text(
        """
        {
          "version": 1,
          "run_id": "task_20260721_130000_000010",
          "script_name": "drag_file_to_folder.txt",
          "started_at_utc": "2026-07-21T13:00:10+00:00",
          "session_end_reason": "completed",
          "summary": {
            "step_count": 4,
            "tool_call_count": 7,
            "failed_step_count": 1,
            "failed_tool_count": 2,
            "total_duration_seconds": 95.4
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")

    assert "drag_file_to_folder.txt" in html
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:\d{2}\)", html)
    assert "completed" in html
    assert ">4<" in html
    assert ">7<" in html
    assert 'data-label="失敗步驟"' in html
    assert 'data-label="失敗工具"' in html
    assert 'data-sort="1"' in html
    assert 'data-sort="2"' in html
    assert "1m 35s" in html
    assert 'data-type="num"' in html
    assert 'data-sort="95.4"' in html
    assert "點選欄位標題可排序" in html
    assert "sortable" in html
    assert '預設依時間由新到舊' in html
    assert 'isTimeColumn' in html
    assert 'class="delete-run"' in html
    assert 'class="bug-run"' in html
    assert 'class="select-run"' in html
    assert 'class="select-all"' in html
    assert 'class="bulk-bug"' in html
    assert 'class="bulk-delete"' in html
    assert "勾選多筆後可批次回報或刪除" in html
    assert 'data-run-id="task_20260721_130000_000010"' in html
    assert "/api/runs/" in html
    assert 'postRunAction(runId, "bug")' in html
    assert 'postRunAction(runId, "delete")' in html


def test_write_runs_index_uses_run_log_script_name_when_report_missing(tmp_path: Path) -> None:
    run_root = tmp_path / "task_20260721_140000_000011"
    run_root.mkdir()
    (run_root / "session_steps.html").write_text("<html></html>", encoding="utf-8")
    (run_root / "run.log").write_text(
        "[2026-07-21T14:00:00+00:00] [app_main_hub] Queue starting coordinator for demo_script.txt\n",
        encoding="utf-8",
    )

    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")

    assert "demo_script.txt" in html
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d{2}:\d{2}\)", html)


def test_write_runs_index_empty_when_no_reports(tmp_path: Path) -> None:
    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")

    assert "尚無報告" in html
    assert "共 0 筆報告" in html
    assert "執行報告" in html
    assert "錄製紀錄" in html
    assert "尚無錄製" in html


def test_write_recording_html_renders_events_and_instructions(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000001"
    _write_recording_fixture(run_root)

    path = write_recording_html_from_run(run_root)
    html = path.read_text(encoding="utf-8")

    assert path == recording_html_path(run_root)
    assert "點擊「搜尋」按鈕" in html
    assert 'class="copy-instruction"' in html
    assert 'class="delete-instruction"' in html
    assert 'class="collapse-instruction"' in html
    assert 'class="select-step"' in html
    assert 'class="select-all-steps"' in html
    assert 'class="bulk-delete-steps"' in html
    assert "刪除選取" in html
    assert "全選" in html
    assert "initStepSelection" in html
    assert "/events/delete" in html
    assert 'data-instruction="點擊「搜尋」按鈕"' in html
    assert "button.copy-instruction" in html
    assert "button.delete-instruction" in html
    assert "button.collapse-instruction" in html
    assert 'class="copy-all-instructions"' in html
    assert "複製全部指令" in html
    assert 'class="rename-recording"' in html
    assert "重新命名" in html
    assert "instructionCopyText" in html
    assert "# expected_outcome: " in html
    assert 'data-expected-outcome="' not in html
    assert "刪除" in html
    assert "新增步驟" in html
    assert 'class="add-recording-step"' in html
    assert 'class="add-instruction"' in html
    assert 'class="add-wait-instruction"' in html
    assert 'data-after-event-index="1"' in html
    assert 'data-duration-seconds="3"' in html
    assert 'id="add-step-dialog"' in html
    assert 'value="condition"' in html
    assert "條件" in html
    assert 'id="event-1"' in html
    assert 'class="step-instruction-input"' in html
    assert "/api/runs/" in html
    assert "/delete" in html
    assert "點擊" in html
    assert "screenshots/event_001.jpeg" in html
    assert "動作前截圖" in html
    assert "動作後截圖" in html
    assert 'href="../index.html#recordings"' in html
    assert "游標" in html


def test_write_recording_html_uses_next_event_screenshot_as_after(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000040"
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "screenshots").mkdir()
    shot1 = _make_jpeg(run_root / "screenshots" / "event_001.jpeg")
    shot2 = _make_jpeg(run_root / "screenshots" / "event_002.jpeg")
    for index, shot in ((1, shot1), (2, shot2)):
        (run_root / "events" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": f"2026-07-21T04:00:0{index}+00:00",
                    "kind": "click",
                    "cursor_xy": [10, 20],
                    "screenshot_path": str(shot),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:00+00:00",
                "event_count": 2,
                "events": ["events/event_001.json", "events/event_002.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "analysis").mkdir()
    for index, instruction in ((1, "點擊「搜尋」"), (2, "點擊「確定」")):
        (run_root / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps({"event_index": index, "instruction": instruction}, ensure_ascii=False),
            encoding="utf-8",
        )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")

    first_group = html.split('data-event-index="1"', 1)[1].split("</details>", 1)[0]
    second_group = html.split('data-event-index="2"', 1)[1].split("</details>", 1)[0]

    assert first_group.index("screenshots/event_001.jpeg") < first_group.index(
        "screenshots/event_002.jpeg"
    )
    assert 'alt="動作前截圖"' in first_group
    assert 'alt="動作後截圖"' in first_group
    assert 'src="screenshots/event_001.jpeg"' in first_group
    assert 'src="screenshots/event_002.jpeg"' in first_group
    assert 'src="screenshots/event_002.jpeg"' in second_group
    assert "無螢幕截圖" in second_group
    assert 'src="screenshots/event_001.jpeg"' not in second_group


def test_write_recording_html_adds_wait_button_from_elapsed(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000041"
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "analysis").mkdir()
    (run_root / "screenshots").mkdir()
    shot1 = _make_jpeg(run_root / "screenshots" / "event_001.jpeg")
    shot2 = _make_jpeg(run_root / "screenshots" / "event_002.jpeg")
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "click",
                "cursor_xy": [10, 20],
                "screenshot_path": str(shot1),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "events" / "event_002.json").write_text(
        json.dumps(
            {
                "index": 2,
                "timestamp_utc": "2026-07-21T04:00:03.250000+00:00",
                "kind": "click",
                "cursor_xy": [30, 40],
                "screenshot_path": str(shot2),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps({"event_index": 1, "instruction": "點擊「搜尋」"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "analysis" / "event_002.json").write_text(
        json.dumps(
            {
                "event_index": 2,
                "instruction": "點擊「確定」",
                "elapsed_since_previous_seconds": 3.25,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:00+00:00",
                "event_count": 2,
                "events": ["events/event_001.json", "events/event_002.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    first_group = html.split('data-event-index="1"', 1)[1].split("</details>", 1)[0]
    second_group = html.split('data-event-index="2"', 1)[1].split("</details>", 1)[0]

    assert 'class="add-wait-instruction"' in first_group
    assert 'data-after-event-index="1"' in first_group
    assert 'data-duration-seconds="4"' in first_group
    assert "在此步驟後加入等待 4 秒" in first_group
    assert "間隔" in first_group
    assert "4 秒" in first_group
    assert 'class="add-wait-instruction"' in second_group
    assert 'data-after-event-index="2"' in second_group
    assert 'data-duration-seconds="3"' in second_group
    assert "在此步驟後加入等待 3 秒" in second_group
    assert "加入等待" in first_group
    assert "button.add-wait-instruction" in html
    assert "確定在此步驟後加入等待" in html


def test_write_recording_html_chains_typing_after_to_next_before(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000042"
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "screenshots").mkdir()
    before = _make_jpeg(run_root / "screenshots" / "event_001.jpeg")
    shared_after = _make_jpeg(run_root / "screenshots" / "event_002.jpeg")
    _make_jpeg(run_root / "screenshots" / "event_001_end.jpeg")
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:01+00:00",
                "kind": "text_input",
                "text": "office",
                "screenshot_path": str(before),
                "end_screenshot_path": str(shared_after),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "events" / "event_002.json").write_text(
        json.dumps(
            {
                "index": 2,
                "timestamp_utc": "2026-07-21T04:00:02+00:00",
                "kind": "click",
                "cursor_xy": [10, 20],
                "screenshot_path": str(shared_after),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:00+00:00",
                "event_count": 2,
                "events": ["events/event_001.json", "events/event_002.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "analysis").mkdir()
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps({"event_index": 1, "instruction": "輸入「office」"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "analysis" / "event_002.json").write_text(
        json.dumps({"event_index": 2, "instruction": "點擊「確定」"}, ensure_ascii=False),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    first_group = html.split('data-event-index="1"', 1)[1].split("</details>", 1)[0]

    assert 'src="screenshots/event_001.jpeg"' in first_group
    assert 'src="screenshots/event_002.jpeg"' in first_group
    assert 'src="screenshots/event_001_end.jpeg"' not in first_group
    assert first_group.index("screenshots/event_001.jpeg") < first_group.index(
        "screenshots/event_002.jpeg"
    )


def test_write_recording_html_resolves_foreign_absolute_screenshot_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000041"
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "screenshots").mkdir()
    shot1 = _make_jpeg(run_root / "screenshots" / "event_001.jpeg")
    shot2 = _make_jpeg(run_root / "screenshots" / "event_002.jpeg")
    foreign1 = Path(r"C:\OtherMachine\runs\recording_x\screenshots\event_001.jpeg")
    foreign2 = Path(r"C:\OtherMachine\runs\recording_x\screenshots\event_002.jpeg")
    for index, foreign in ((1, foreign1), (2, foreign2)):
        (run_root / "events" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": f"2026-07-21T04:00:0{index}+00:00",
                    "kind": "click",
                    "cursor_xy": [10, 20],
                    "screenshot_path": str(foreign),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:00+00:00",
                "event_count": 2,
                "events": ["events/event_001.json", "events/event_002.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    first_group = html.split('data-event-index="1"', 1)[1].split("</details>", 1)[0]

    assert shot1.exists() and shot2.exists()
    assert 'src="screenshots/event_001.jpeg"' in first_group
    assert 'src="screenshots/event_002.jpeg"' in first_group
    assert "OtherMachine" not in html


def test_write_recording_html_badge_uses_click_count(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000031"
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "click",
                "click_count": 5,
                "cursor_xy": [10, 20],
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:00+00:00",
                "event_count": 1,
                "events": ["events/event_001.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "analysis").mkdir()
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "instruction": "將滑鼠移到「向右滾動箭頭」圖示，並連按5下。",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")

    assert 'data-kind="click"' in html
    assert '<span class="badge neutral">連按5下</span>' in html
    assert "將滑鼠移到「向右滾動箭頭」圖示，並連按5下。" in html


def test_write_recording_html_copy_includes_expected_outcome(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000030"
    _write_recording_fixture(run_root)
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "instruction": "點擊「搜尋」按鈕",
                "expected_outcome": '對話框顯示 "搜尋"',
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")

    assert 'data-instruction="點擊「搜尋」按鈕"' in html
    assert 'data-expected-outcome="對話框顯示 &quot;搜尋&quot;"' in html
    assert 'class="instruction-expected"' in html
    assert "預期結果：對話框顯示" in html
    assert 'class="instruction-summary-text"' in html
    assert "預期結果" in html
    assert 'class="expected-outcome-input"' in html
    assert 'class="apply-expected-outcome"' in html
    assert "對話框顯示" in html
    assert "function instructionCopyText" in html
    assert "# expected_outcome: " in html
    assert "function applyExpectedOutcome" in html
    assert "instruction-expected" in html


def test_write_recording_html_renders_landmark_multiselect(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000020"
    _write_recording_fixture(run_root)
    instruction = (
        "將滑鼠移到「搜尋」文字（在「已選取 2 個項目」文字的左下方），並點擊滑鼠一下。"
    )
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {"event_index": 1, "instruction": instruction},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "yolo_ocr").mkdir(exist_ok=True)
    (run_root / "yolo_ocr" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "candidates": [
                    {
                        "bbox": [40, 40, 20, 20],
                        "center": [50, 50],
                        "class_name": "text",
                        "text": "搜尋",
                    },
                    {
                        "bbox": [10, 90, 80, 14],
                        "center": [50, 97],
                        "class_name": "text",
                        "text": "已選取 2 個項目",
                    },
                    {
                        "bbox": [90, 90, 50, 14],
                        "center": [115, 97],
                        "class_name": "text",
                        "text": "45 個項目",
                    },
                    {
                        "bbox": [200, 40, 20, 20],
                        "center": [210, 50],
                        "class_name": "element",
                        "text": "",
                        "icons": [{"chinese_id": "Chrome"}],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")

    assert "目標與地標" in html
    assert "點擊目標" in html
    assert 'class="landmarks-groups"' in html
    assert 'class="landmarks-side-groups"' in html
    assert 'data-side-group="' in html
    assert 'data-primary-index="0"' in html
    assert 'data-primary-index="1"' in html
    assert 'class="apply-landmarks"' in html
    assert "Ctrl+Enter" in html
    assert "syncLandmarksDirty" in html
    assert 'data-label="「已選取 2 個項目」文字"' in html
    assert 'data-label="「45 個項目」文字"' in html
    assert 'data-label="「Chrome」圖示"' in html
    assert 'data-label="「已選取 2 個項目」文字"' in html
    assert "checked" in html
    assert "套用</button>" in html
    assert "/api/runs/" in html
    assert 'class="rerun-yolo-ocr"' in html
    assert "YOLO/OCR 未偵測到目標" not in html


def test_write_recording_html_renders_char_target_checkbox(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000044"
    _write_recording_fixture(run_root)
    instruction = "將滑鼠移到「搜尋」文字，並點擊滑鼠一下。"
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {"event_index": 1, "instruction": instruction, "use_char_target": False},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "yolo_ocr").mkdir(exist_ok=True)
    (run_root / "yolo_ocr" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "candidates": [
                    {
                        "bbox": [40, 40, 20, 20],
                        "center": [50, 50],
                        "class_name": "text",
                        "text": "搜尋",
                        "clicked_char": "搜",
                        "clicked_char_index": 0,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    assert "點擊字元" in html
    assert 'class="use-char-target"' in html
    assert 'class="apply-char-target"' in html
    assert "指定點擊字元：「搜尋」的「搜」字上" in html
    assert "/char_target" in html
    assert 'class="use-char-target" checked' not in html


def test_write_recording_html_checks_char_target_when_enabled(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000045"
    _write_recording_fixture(run_root)
    instruction = "將滑鼠移到「搜尋」的「搜」字上，並點擊滑鼠一下。"
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {"event_index": 1, "instruction": instruction, "use_char_target": True},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "yolo_ocr").mkdir(exist_ok=True)
    (run_root / "yolo_ocr" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "candidates": [
                    {
                        "bbox": [40, 40, 20, 20],
                        "center": [50, 50],
                        "class_name": "text",
                        "text": "搜尋",
                        "clicked_char": "搜",
                        "clicked_char_index": 0,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    assert 'class="use-char-target" checked' in html


def test_write_recording_html_infers_char_target_from_legacy_instruction(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "recording_20260721_120000_000046"
    _write_recording_fixture(run_root)
    instruction = "將滑鼠移到「搜尋」的「搜」字上，並點擊滑鼠一下。"
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps({"event_index": 1, "instruction": instruction}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "yolo_ocr").mkdir(exist_ok=True)
    (run_root / "yolo_ocr" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "candidates": [
                    {
                        "bbox": [40, 40, 20, 20],
                        "center": [50, 50],
                        "class_name": "text",
                        "text": "搜尋",
                        "clicked_char": "搜",
                        "clicked_char_index": 0,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    assert 'class="use-char-target" checked' in html


def test_write_recording_html_shows_yolo_retry_when_failed(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000021"
    _write_recording_fixture(run_root)
    (run_root / "yolo_ocr").mkdir(exist_ok=True)
    (run_root / "yolo_ocr" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "candidates": [],
                "detection_count": 0,
                "yolo_error": "Triton infer timed out after 20s",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    assert 'class="rerun-yolo-ocr"' in html
    assert "vision-retry failed" in html
    assert "YOLO/OCR 未偵測到目標" in html or "timed out" in html
    assert "/events/" in html and "/yolo_ocr" in html


def test_write_recording_html_escapes_markup(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000002"
    _write_recording_fixture(run_root, with_analysis=False)
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "text_input",
                "text": "<script>alert(1)</script>",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "analysis").mkdir(exist_ok=True)
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {"event_index": 1, "instruction": "輸入「<b>hi</b>」"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in html
    assert 'value="&lt;b&gt;hi&lt;/b&gt;"' in html
    assert "輸入「&lt;b&gt;hi&lt;/b&gt;」" in html
    assert 'class="rerun-yolo-ocr"' not in html
    assert 'data-instruction="輸入「&lt;b&gt;hi&lt;/b&gt;」"' in html
    assert 'class="typed-text-input"' in html
    assert 'class="apply-typed-text"' in html
    assert "/events/" in html
    assert "/text" in html


def test_write_recording_html_renders_typed_text_editor_from_event(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000003"
    _write_recording_fixture(run_root, with_analysis=False)
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "text_input",
                "text": "hello world",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")

    assert 'class="typed-text"' in html
    assert 'value="hello world"' in html
    assert "typed-text-choice" in html
    assert "鍵盤：hello world" in html
    assert "套用</button>" in html
    assert "<dt>文字</dt>" not in html


def test_write_recording_html_shows_ocr_and_recorded_text_choices(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_20260721_120000_000043"
    _write_recording_fixture(run_root, with_analysis=False)
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "text_input",
                "text": "ooffice",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "analysis").mkdir(exist_ok=True)
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "instruction": "輸入「ooffice」",
                "text_resolution": {
                    "recorded_text": "ooffice",
                    "ocr_text": "搜尋",
                    "ocr_options": ["什麼是套利?", "搜尋"],
                    "resolved_text": "ooffice",
                    "source": "recorded",
                    "reason": "prefer recorded text; after-screenshot OCR available as alternate",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")

    assert 'value="ooffice"' in html
    assert 'class="typed-text-choice selected"' in html
    assert 'data-selected-source="recorded"' in html
    assert "OCR：什麼是套利?" in html
    assert "OCR：搜尋" in html
    assert "鍵盤：ooffice" in html
    assert 'data-text="什麼是套利?"' in html
    assert 'data-text="搜尋"' in html
    assert "choice_text" in html
    assert "selectTypedTextChoice" in html
    assert "updateTypedTextChoiceLabel" in html


def test_typed_text_candidates_default_to_recorded_when_available() -> None:
    event = {"kind": "text_input", "text": "ooffice"}
    analysis = {
        "text_resolution": {
            "recorded_text": "ooffice",
            "ocr_text": "office",
            "resolved_text": "ooffice",
            "source": "recorded",
        }
    }
    recorded, ocr_options, active, active_source = _typed_text_candidates(
        event,
        analysis,
        "輸入「ooffice」",
    )
    assert recorded == "ooffice"
    assert ocr_options == ["office"]
    assert active == "ooffice"
    assert active_source == "recorded"


def test_write_runs_index_lists_recordings_in_recordings_tab(tmp_path: Path) -> None:
    task = tmp_path / "task_20260721_100000_000001"
    task.mkdir()
    (task / "session_steps.html").write_text("<html></html>", encoding="utf-8")
    (task / "report.json").write_text(
        json.dumps({"script_name": "demo.txt", "started_at_utc": "2026-07-21T10:00:00+00:00"}),
        encoding="utf-8",
    )

    recording = tmp_path / "recording_20260721_110000_000002"
    _write_recording_fixture(recording)
    write_recording_html_from_run(recording)

    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")

    assert 'data-tab="runs"' in html
    assert 'data-tab="smart"' in html
    assert 'data-tab="recordings"' in html
    assert ">智能模式<" in html
    assert 'href="task_20260721_100000_000001/session_steps.html"' in html
    assert 'href="recording_20260721_110000_000002/recording_steps.html"' in html
    assert "demo.txt" in html
    assert "recording_20260721_110000_000002" in html
    assert 'data-label="已分析"' in html
    assert 'data-label="事件"' in html


def test_write_runs_index_lists_sibling_recordings_folder(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    recordings_root = tmp_path / "recordings"
    runs_root.mkdir()
    recordings_root.mkdir()

    recording = recordings_root / "recording_20260721_120000_000003"
    _write_recording_fixture(recording)
    write_recording_html_from_run(recording, update_index=False)

    html = write_runs_index_html(
        runs_root, recordings_root=recordings_root
    ).read_text(encoding="utf-8")
    assert 'href="../recordings/recording_20260721_120000_000003/recording_steps.html"' in html
    assert "recording_20260721_120000_000003" in html


def test_write_runs_index_lists_renamed_recording_without_prefix(tmp_path: Path) -> None:
    recording = tmp_path / "開啟神網"
    _write_recording_fixture(recording)
    write_recording_html_from_run(recording)

    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")
    assert "開啟神網" in html
    assert 'href="%E9%96%8B%E5%95%9F%E7%A5%9E%E7%B6%B2/recording_steps.html"' in html or (
        "recording_steps.html" in html and "開啟神網" in html
    )
    # Must not appear as a normal execution report.
    runs_panel_start = html.index('id="tab-runs"')
    smart_panel_start = html.index('id="tab-smart"')
    runs_panel = html[runs_panel_start:smart_panel_start]
    assert "開啟神網" not in runs_panel


def test_write_runs_index_lists_smart_runs_in_smart_tab(tmp_path: Path) -> None:
    task = tmp_path / "task_20260721_100000_000001"
    task.mkdir()
    (task / "session_steps.html").write_text("<html></html>", encoding="utf-8")
    (task / "report.json").write_text(
        json.dumps({"script_name": "demo.txt", "started_at_utc": "2026-07-21T10:00:00+00:00"}),
        encoding="utf-8",
    )

    smart = tmp_path / "smart_20260721_120000_000003"
    smart.mkdir()
    (smart / "session_steps.html").write_text("<html></html>", encoding="utf-8")
    (smart / "report.json").write_text(
        json.dumps(
            {
                "run_mode": "smart",
                "script_name": "open_outlook.txt",
                "script_path": str(tmp_path / "open_outlook.txt"),
                "smart_goal": "打開 Outlook 並讀取最新郵件",
                "started_at_utc": "2026-07-21T12:00:00+00:00",
                "session_end_reason": "completed",
                "summary": {
                    "step_count": 2,
                    "tool_call_count": 3,
                    "failed_step_count": 0,
                    "failed_tool_count": 0,
                    "total_duration_seconds": 42.5,
                    "smart_cycle_count": 2,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")

    assert 'data-tab="smart"' in html
    assert 'id="tab-smart"' in html
    assert 'href="smart_20260721_120000_000003/session_steps.html"' in html
    assert "open_outlook.txt" in html
    assert "打開 Outlook 並讀取最新郵件" not in html
    assert "demo.txt" in html
    assert ">目標<" in html

    runs_panel_start = html.index('id="tab-runs"')
    smart_panel_start = html.index('id="tab-smart"')
    recordings_panel_start = html.index('id="tab-recordings"')
    runs_panel = html[runs_panel_start:smart_panel_start]
    smart_panel = html[smart_panel_start:recordings_panel_start]
    assert 'href="task_20260721_100000_000001/session_steps.html"' in runs_panel
    assert 'href="smart_20260721_120000_000003/session_steps.html"' not in runs_panel
    assert 'href="smart_20260721_120000_000003/session_steps.html"' in smart_panel
    assert 'href="task_20260721_100000_000001/session_steps.html"' not in smart_panel
    assert "smart: true" in html


def test_write_runs_index_backfills_missing_recording_html(tmp_path: Path) -> None:
    recording = tmp_path / "recording_20260721_120000_000010"
    _write_recording_fixture(recording, with_html=False)

    assert not (recording / "recording_steps.html").is_file()

    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")

    assert (recording / "recording_steps.html").is_file()
    assert 'href="recording_20260721_120000_000010/recording_steps.html"' in html
    assert "點擊「搜尋」按鈕" in (recording / "recording_steps.html").read_text(encoding="utf-8")


def test_write_runs_index_refreshes_existing_recording_html(tmp_path: Path) -> None:
    recording = tmp_path / "recording_20260721_120000_000012"
    _write_recording_fixture(recording, with_html=False)
    (recording / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "text_input",
                "text": "hello",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (recording / "recording_steps.html").write_text("<html>stale</html>", encoding="utf-8")

    write_runs_index_html(tmp_path)

    html = (recording / "recording_steps.html").read_text(encoding="utf-8")
    assert "stale" not in html
    assert 'class="typed-text-input"' in html
    assert 'value="hello"' in html


def test_write_runs_index_excludes_recording_from_runs_tab(tmp_path: Path) -> None:
    recording = tmp_path / "recording_20260721_130000_000011"
    _write_recording_fixture(recording)
    (recording / "session_steps.html").write_text("<html></html>", encoding="utf-8")
    write_recording_html_from_run(recording)

    html = write_runs_index_html(tmp_path).read_text(encoding="utf-8")

    assert 'href="recording_20260721_130000_000011/recording_steps.html"' in html
    assert 'href="recording_20260721_130000_000011/session_steps.html"' not in html


def test_write_recording_html_follows_session_order_for_display_numbers(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "recording_20260721_140000_000050"
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "analysis").mkdir()
    for index, instruction in ((1, "第一步"), (2, "第二步")):
        (run_root / "events" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": f"2026-07-21T04:00:0{index}+00:00",
                    "kind": "click",
                    "cursor_xy": [10, 20],
                    "screenshot_path": "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_root / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps({"event_index": index, "instruction": instruction}, ensure_ascii=False),
            encoding="utf-8",
        )
    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:00+00:00",
                "event_count": 2,
                "events": ["events/event_002.json", "events/event_001.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    html = write_recording_html_from_run(run_root).read_text(encoding="utf-8")
    pos_two = html.index('data-event-index="2"')
    pos_one = html.index('data-event-index="1"')
    assert pos_two < pos_one

    group_two = html.split('data-event-index="2"', 1)[1].split("</details>", 1)[0]
    group_one = html.split('data-event-index="1"', 1)[1].split("</details>", 1)[0]
    assert '<span class="instruction-number">1.</span>' in group_two
    assert '<span class="instruction-number">2.</span>' in group_one
    assert "第二步" in group_two
    assert "第一步" in group_one
    assert 'class="add-recording-step"' in html
    assert 'id="add-step-dialog"' in html
    assert "自訂指令" in html
