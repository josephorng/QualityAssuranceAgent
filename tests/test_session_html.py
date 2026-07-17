from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image

from src.common.session_html import (
    session_html_path,
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
    assert "task_html" in html
    assert "步驟 1：click" in html
    assert "步驟 2：type_text" in html
    # args rendered as a collapsed key/value table
    assert '<details class="args"><summary>參數</summary>' in html
    assert "<th>button</th><td>left</td>" in html
    assert "<th>instruction</th><td>搜尋欄</td>" in html
    assert "<th>text</th><td>hello</td>" in html
    # collapsed by default (no open attribute on the args block)
    assert "<details open" not in html
    assert "成功" in html
    assert "失敗" in html
    # first step references screenshots relative to the run folder
    assert 'src="eye/before.png"' in html
    assert 'src="eye/after.png"' in html
    # second step has no screenshots
    assert "無螢幕截圖" in html


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

    assert '<p class="instruction">點擊「搜尋」欄位</p>' in html
    # instruction appears before the 狀態 (status) row
    assert html.index("點擊「搜尋」欄位") < html.index("狀態")


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

    assert 'class="instruction"' not in html


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
    assert "步驟 1：move_mouse" in html
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

    assert html.count("步驟 1：click") == 1


def test_write_session_html_without_hand_csv(tmp_path: Path) -> None:
    run_root = tmp_path / "task_no_actions"
    path = write_session_html_from_run(run_root)

    assert path.is_file()
    html = path.read_text(encoding="utf-8")
    assert "task_no_actions" in html
    assert "步驟 1" not in html
