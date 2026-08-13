from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.common.runs_report_server import (
    RunsReportServer,
    apply_recording_event_landmarks,
    apply_recording_event_text,
    delete_recording_event,
    delete_run_report_folder,
    ensure_runs_report_server,
    resolve_deletable_run_folder,
    stop_runs_report_server,
    zip_run_report_to_bug_share,
)
from src.common.session_html import write_runs_index_html


def _make_report_run(runs_root: Path, name: str) -> Path:
    run_root = runs_root / name
    run_root.mkdir(parents=True)
    (run_root / "session_steps.html").write_text("<html></html>", encoding="utf-8")
    (run_root / "report.json").write_text(
        json.dumps({"script_name": f"{name}.txt", "session_end_reason": "completed"}),
        encoding="utf-8",
    )
    (run_root / "note.txt").write_text("bug payload", encoding="utf-8")
    return run_root


def _make_recording_landmark_run(runs_root: Path, name: str) -> Path:
    run_root = runs_root / name
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "analysis").mkdir()
    (run_root / "yolo_ocr").mkdir()
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "click",
                "cursor_xy": [10, 20],
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    instruction = (
        "將滑鼠移到「搜尋」文字（在「已選取 2 個項目」文字的左下方），並點擊滑鼠一下。"
    )
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "instruction": instruction,
                "wait_instruction": "等待 2 秒",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "report.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "recorded": 1,
                "instructions": ["等待 2 秒", instruction],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
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
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "session.json").write_text(
        json.dumps({"run_id": name, "event_count": 1}),
        encoding="utf-8",
    )
    return run_root


def test_resolve_deletable_run_folder_rejects_traversal(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_report_run(runs_root, "task_ok")

    with pytest.raises(ValueError):
        resolve_deletable_run_folder(runs_root, "../outside")
    with pytest.raises(ValueError):
        resolve_deletable_run_folder(runs_root, "task_ok/../task_ok")
    with pytest.raises(ValueError):
        resolve_deletable_run_folder(runs_root, "missing_task")


def test_delete_run_report_folder_removes_dir_and_rebuilds_index(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    keep = _make_report_run(runs_root, "task_keep")
    gone = _make_report_run(runs_root, "task_gone")
    write_runs_index_html(runs_root)

    deleted = delete_run_report_folder(runs_root, "task_gone")

    assert deleted.name == "task_gone"
    assert not gone.exists()
    assert keep.exists()
    index_html = (runs_root / "index.html").read_text(encoding="utf-8")
    assert "task_gone" not in index_html
    assert "task_keep" in index_html


def test_zip_run_report_to_bug_share_copies_archive(tmp_path: Path) -> None:
    import zipfile

    runs_root = tmp_path / "runs"
    share = tmp_path / "CUA-BUG"
    _make_report_run(runs_root, "task_bug_zip")

    dest = zip_run_report_to_bug_share(runs_root, "task_bug_zip", dest_dir=share)

    assert dest.parent == share
    assert dest.name.startswith("task_bug_zip_")
    assert dest.suffix == ".zip"
    assert dest.is_file()
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    assert any(name.endswith("note.txt") for name in names)
    assert any("task_bug_zip" in name for name in names)
    assert (runs_root / "task_bug_zip").is_dir()


def test_runs_report_server_delete_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_report_run(runs_root, "task_http_delete")
    write_runs_index_html(runs_root)

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        req = urllib.request.Request(
            f"{base}/api/runs/task_http_delete/delete",
            method="POST",
            data=b"",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["deleted"] == "task_http_delete"
        assert not (runs_root / "task_http_delete").exists()

        bad = urllib.request.Request(
            f"{base}/api/runs/not-a-real-run/delete",
            method="POST",
            data=b"",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad, timeout=5)
        assert exc_info.value.code == 400

        traversal = urllib.request.Request(
            f"{base}/api/runs/..%2Fsecret/delete",
            method="POST",
            data=b"",
        )
        with pytest.raises(urllib.error.HTTPError) as trav_info:
            urllib.request.urlopen(traversal, timeout=5)
        assert trav_info.value.code in {400, 404}
    finally:
        server.stop()


def test_runs_report_server_bug_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    share = tmp_path / "CUA-BUG"
    _make_report_run(runs_root, "task_http_bug")
    write_runs_index_html(runs_root)

    monkeypatch.setattr(
        "src.common.runs_report_server.BUG_REPORT_SHARE_DIR",
        share,
    )

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        req = urllib.request.Request(
            f"{base}/api/runs/task_http_bug/bug",
            method="POST",
            data=b"",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["run_id"] == "task_http_bug"
        copied = Path(payload["copied_to"])
        assert copied.is_file()
        assert copied.parent == share
        assert (runs_root / "task_http_bug").is_dir()
    finally:
        server.stop()


def test_ensure_runs_report_server_reuses_same_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    stop_runs_report_server()
    try:
        first = ensure_runs_report_server(runs_root)
        second = ensure_runs_report_server(runs_root)
        assert first is second
        assert first.is_running()
    finally:
        stop_runs_report_server()


def test_apply_recording_event_landmarks_persists_and_rebuilds(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_landmark_edit")

    result = apply_recording_event_landmarks(
        runs_root,
        "recording_landmark_edit",
        1,
        selected=[
            {"label": "「45 個項目」文字", "side": "lower_right"},
            {"label": "「已選取 2 個項目」文字", "side": "lower_left"},
        ],
    )

    expected = (
        "將滑鼠移到「搜尋」文字（在「45 個項目」文字的右下方、"
        "在「已選取 2 個項目」文字的左下方），並點擊滑鼠一下。"
    )
    assert result["instruction"] == expected
    assert result["rebuilt"] is False
    analysis = json.loads(
        (runs_root / "recording_landmark_edit" / "analysis" / "event_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert analysis["instruction"] == expected
    assert analysis["landmarks"]["selected"][0]["label"] == "「45 個項目」文字"
    report = json.loads(
        (runs_root / "recording_landmark_edit" / "report.json").read_text(encoding="utf-8")
    )
    assert report["instructions"] == ["等待 2 秒", expected]
    html = (runs_root / "recording_landmark_edit" / "recording_steps.html").read_text(
        encoding="utf-8"
    )
    assert expected in html


def test_apply_recording_event_primary_target_reorders_and_rebuilds(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_primary_swap")

    result = apply_recording_event_landmarks(
        runs_root,
        "recording_primary_swap",
        1,
        selected=[
            {"label": "「搜尋」文字", "side": "upper_left"},
            {"label": "「已選取 2 個項目」文字", "side": "lower_left"},
        ],
        primary_index=2,
    )

    assert result["rebuilt"] is True
    assert result["instruction"].startswith("將滑鼠移到「45 個項目」文字")
    assert "並點擊滑鼠一下" in result["instruction"]
    assert "「搜尋」文字" in result["instruction"]
    assert "「已選取 2 個項目」文字" in result["instruction"]

    yolo = json.loads(
        (runs_root / "recording_primary_swap" / "yolo_ocr" / "event_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert yolo["candidates"][0]["text"] == "45 個項目"
    assert yolo["candidates"][1]["text"] == "搜尋"
    assert "[index 0] class=" in yolo["candidate_text"]
    assert "45 個項目" in yolo["candidate_text"]

    analysis = json.loads(
        (runs_root / "recording_primary_swap" / "analysis" / "event_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert analysis["instruction"] == result["instruction"]
    assert analysis["landmarks"]["primary_index"] == 0
    selected_labels = {item["label"] for item in analysis["landmarks"]["selected"]}
    assert "「45 個項目」文字" not in selected_labels


def test_runs_report_server_landmarks_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_http_landmarks")

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        body = json.dumps(
            {
                "selected": [
                    {"label": "「45 個項目」文字", "side": "lower_right"},
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_landmarks/events/1/landmarks",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert "「45 個項目」文字的右下方" in payload["instruction"]
        assert "並點擊滑鼠一下" in payload["instruction"]
    finally:
        server.stop()


def _make_recording_two_event_run(runs_root: Path, name: str) -> Path:
    run_root = runs_root / name
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "analysis").mkdir()
    (run_root / "screenshots").mkdir()
    (run_root / "yolo_ocr").mkdir()

    shot1 = run_root / "screenshots" / "event_001.jpeg"
    shot2 = run_root / "screenshots" / "event_002.jpeg"
    shot1.write_bytes(b"\xff\xd8\xff\xd9")
    shot2.write_bytes(b"\xff\xd8\xff\xd9")

    for index, instruction in (
        (1, "點擊「搜尋」按鈕"),
        (2, "點擊「確定」按鈕"),
    ):
        (run_root / "events" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": f"2026-07-21T04:00:0{index}+00:00",
                    "kind": "click",
                    "cursor_xy": [10, 20],
                    "screenshot_path": str(run_root / "screenshots" / f"event_{index:03d}.jpeg"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_root / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {"event_index": index, "instruction": instruction},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_root / "yolo_ocr" / f"event_{index:03d}.json").write_text(
            json.dumps({"event_index": index, "candidates": []}, ensure_ascii=False),
            encoding="utf-8",
        )

    (run_root / "session.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "started_at_utc": "2026-07-21T04:00:00+00:00",
                "stopped_at_utc": "2026-07-21T04:01:00+00:00",
                "event_count": 2,
                "events": ["events/event_001.json", "events/event_002.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "report.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "recorded": 2,
                "processed": 2,
                "cached": 2,
                "instructions": ["點擊「搜尋」按鈕", "點擊「確定」按鈕"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_root


def test_delete_recording_event_removes_files_and_rebuilds(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_delete_event")

    result = delete_recording_event(runs_root, "recording_delete_event", 1)

    assert result == {"event_index": 1, "remaining": 1}
    assert not (run_root / "events" / "event_001.json").exists()
    assert not (run_root / "analysis" / "event_001.json").exists()
    assert not (run_root / "screenshots" / "event_001.jpeg").exists()
    assert not (run_root / "yolo_ocr" / "event_001.json").exists()
    assert (run_root / "events" / "event_002.json").is_file()
    assert (run_root / "analysis" / "event_002.json").is_file()

    session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
    assert session["event_count"] == 1
    assert session["events"] == ["events/event_002.json"]

    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    assert report["recorded"] == 1
    assert report["processed"] == 1
    assert report["cached"] == 1
    assert report["instructions"] == ["點擊「確定」按鈕"]

    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert "點擊「確定」按鈕" in html
    assert "點擊「搜尋」按鈕" not in html
    assert 'class="delete-instruction"' in html


def test_runs_report_server_event_delete_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_http_delete_event")

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_delete_event/events/2/delete",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["event_index"] == 2
        assert payload["remaining"] == 1
        assert not (run_root / "events" / "event_002.json").exists()
        assert (run_root / "events" / "event_001.json").is_file()

        missing = urllib.request.Request(
            f"{base}/api/runs/recording_http_delete_event/events/2/delete",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(missing, timeout=5)
        assert exc_info.value.code == 400
    finally:
        server.stop()


def _make_recording_text_input_run(runs_root: Path, name: str) -> Path:
    run_root = runs_root / name
    run_root.mkdir(parents=True)
    (run_root / "events").mkdir()
    (run_root / "analysis").mkdir()
    (run_root / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-07-21T04:00:00+00:00",
                "kind": "text_input",
                "text": "wrong ocr",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    instruction = "輸入「wrong ocr」"
    (run_root / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "event_index": 1,
                "instruction": instruction,
                "wait_instruction": "等待 2 秒",
                "text_resolution": {
                    "recorded_text": "wrng",
                    "resolved_text": "wrong ocr",
                    "source": "ocr",
                    "reason": "vision-first OCR",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "report.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "recorded": 1,
                "instructions": ["等待 2 秒", instruction],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_root


def test_apply_recording_event_text_persists_and_rebuilds(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_text_input_run(runs_root, "recording_text_edit")

    result = apply_recording_event_text(
        runs_root,
        "recording_text_edit",
        1,
        text="  正確文字  ",
    )

    assert result == {"text": "正確文字", "instruction": "輸入「正確文字」"}
    event = json.loads(
        (runs_root / "recording_text_edit" / "events" / "event_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert event["text"] == "正確文字"
    analysis = json.loads(
        (runs_root / "recording_text_edit" / "analysis" / "event_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert analysis["instruction"] == "輸入「正確文字」"
    assert analysis["text_resolution"]["resolved_text"] == "正確文字"
    assert analysis["text_resolution"]["source"] == "user"
    assert analysis["text_resolution"]["recorded_text"] == "wrng"
    report = json.loads(
        (runs_root / "recording_text_edit" / "report.json").read_text(encoding="utf-8")
    )
    assert report["instructions"] == ["等待 2 秒", "輸入「正確文字」"]
    html = (runs_root / "recording_text_edit" / "recording_steps.html").read_text(
        encoding="utf-8"
    )
    assert 'value="正確文字"' in html
    assert "輸入「正確文字」" in html


def test_apply_recording_event_text_rejects_non_text_input(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_click_no_text")

    with pytest.raises(ValueError, match="typed-text"):
        apply_recording_event_text(
            runs_root,
            "recording_click_no_text",
            1,
            text="nope",
        )


def test_apply_recording_event_text_rejects_empty(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_text_input_run(runs_root, "recording_text_empty")

    with pytest.raises(ValueError, match="empty"):
        apply_recording_event_text(
            runs_root,
            "recording_text_empty",
            1,
            text="   ",
        )


def test_runs_report_server_text_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_text_input_run(runs_root, "recording_http_text")

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        body = json.dumps({"text": "fixed value"}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_text/events/1/text",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["text"] == "fixed value"
        assert payload["instruction"] == "輸入「fixed value」"
    finally:
        server.stop()
