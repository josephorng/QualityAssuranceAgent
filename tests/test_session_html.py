from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image

from src.common.session_html import (
    runs_index_html_path,
    session_html_path,
    write_runs_index_html,
    write_session_html_from_run,
)


def _make_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(20, 120, 200)).save(path)
    return path


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
    assert 'class="delete-run"' in html
    assert 'class="bug-run"' in html
    assert 'data-run-id="task_20260721_130000_000010"' in html
    assert "/api/runs/" in html
    assert "/bug" in html


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
