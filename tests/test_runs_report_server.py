from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.common.runs_report_server import (
    RunsReportServer,
    add_recording_event,
    apply_recording_event_char_target,
    apply_recording_event_expected_outcome,
    apply_recording_event_instruction,
    apply_recording_event_landmarks,
    apply_recording_event_text,
    delete_recording_event,
    delete_recording_events,
    delete_run_report_folder,
    ensure_runs_report_server,
    rename_recording_folder,
    rerun_recording_event_yolo_ocr,
    resolve_deletable_run_folder,
    stop_runs_report_server,
    sync_recording_events,
    zip_run_report_to_bug_share,
)
from src.common.script_helper import collect_recording_script_text
from src.common.session_html import write_runs_index_html
from src.recorder.models import RecordedEvent


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
                "expected_outcome": "搜尋結果已顯示",
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
                "expected_outcomes": [None, "搜尋結果已顯示"],
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


def test_report_server_serves_sibling_recording_from_index_href(tmp_path: Path) -> None:
    """Index under runs/ links to ../recordings/...; serve from common parent."""
    runs_root = tmp_path / "runs"
    recordings_root = tmp_path / "recordings"
    runs_root.mkdir()
    _make_recording_landmark_run(recordings_root, "demo_rec")
    write_runs_index_html(runs_root, recordings_root=recordings_root)
    index_html = (runs_root / "index.html").read_text(encoding="utf-8")
    assert 'href="../recordings/demo_rec/recording_steps.html"' in index_html

    stop_runs_report_server()
    try:
        server = ensure_runs_report_server(tmp_path)
        with urllib.request.urlopen(
            f"{server.base_url}/runs/index.html"
        ) as response:
            assert response.status == 200
            body = response.read().decode("utf-8")
            assert "demo_rec" in body
        with urllib.request.urlopen(
            f"{server.base_url}/recordings/demo_rec/recording_steps.html"
        ) as response:
            assert response.status == 200
            page = response.read().decode("utf-8")
            assert "demo_rec" in page
            assert "尚無錄製事件" in page or "instruction-group" in page or "錄製" in page
    finally:
        stop_runs_report_server()


def test_resolve_deletable_run_folder_finds_sibling_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    recordings_root = tmp_path / "recordings"
    runs_root.mkdir()
    recording = recordings_root / "sibling_rec"
    recording.mkdir(parents=True)

    monkeypatch.setattr(
        "src.common.settings.resolve_runs_dir",
        lambda configured=None: runs_root.resolve(),
    )
    monkeypatch.setattr(
        "src.common.settings.resolve_recordings_dir",
        lambda configured=None: recordings_root.resolve(),
    )

    found = resolve_deletable_run_folder(tmp_path, "sibling_rec")
    assert found == recording.resolve()


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


def _add_clicked_char(run_root: Path, *, char: str = "搜", index: int = 0) -> None:
    yolo_path = run_root / "yolo_ocr" / "event_001.json"
    payload = json.loads(yolo_path.read_text(encoding="utf-8"))
    payload["candidates"][0]["clicked_char"] = char
    payload["candidates"][0]["clicked_char_index"] = index
    yolo_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_apply_recording_event_char_target_rebuilds_parent_and_char(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_landmark_run(runs_root, "recording_char_target")
    _add_clicked_char(run_root)

    enabled = apply_recording_event_char_target(
        runs_root,
        "recording_char_target",
        1,
        use_char_target=True,
    )
    expected_char = (
        "將滑鼠移到「搜尋」的「搜」字上（在「已選取 2 個項目」文字的左下方），並點擊滑鼠一下。"
    )
    assert enabled["instruction"] == expected_char
    assert enabled["use_char_target"] is True
    analysis = json.loads(
        (run_root / "analysis" / "event_001.json").read_text(encoding="utf-8")
    )
    assert analysis["instruction"] == expected_char
    assert analysis["use_char_target"] is True

    disabled = apply_recording_event_char_target(
        runs_root,
        "recording_char_target",
        1,
        use_char_target=False,
    )
    expected_parent = (
        "將滑鼠移到「搜尋」文字（在「已選取 2 個項目」文字的左下方），並點擊滑鼠一下。"
    )
    assert disabled["instruction"] == expected_parent
    assert disabled["use_char_target"] is False
    analysis = json.loads(
        (run_root / "analysis" / "event_001.json").read_text(encoding="utf-8")
    )
    assert analysis["use_char_target"] is False
    assert analysis["instruction"] == expected_parent


def test_apply_recording_event_primary_rebuild_skips_char_target_by_default(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_landmark_run(runs_root, "recording_char_rebuild")
    yolo_path = run_root / "yolo_ocr" / "event_001.json"
    payload = json.loads(yolo_path.read_text(encoding="utf-8"))
    payload["candidates"][2]["clicked_char"] = "項"
    payload["candidates"][2]["clicked_char_index"] = 0
    yolo_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = apply_recording_event_landmarks(
        runs_root,
        "recording_char_rebuild",
        1,
        selected=[{"label": "「搜尋」文字", "side": "upper_left"}],
        primary_index=2,
    )
    assert result["rebuilt"] is True
    assert result["instruction"].startswith("將滑鼠移到「45 個項目」文字")
    assert "的「項」字上" not in result["instruction"]


def test_runs_report_server_char_target_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_landmark_run(runs_root, "recording_http_char")
    _add_clicked_char(run_root)

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        body = json.dumps({"use_char_target": True}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_char/events/1/char_target",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["use_char_target"] is True
        assert "「搜尋」的「搜」字上" in payload["instruction"]
    finally:
        server.stop()


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


def test_rerun_recording_event_yolo_ocr_rebuilds_instruction(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_landmark_run(runs_root, "recording_yolo_retry")
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

    def fake_yolo(event, *, run_dir, persist_debug=True):
        del persist_debug
        payload = {
            "used_vision": True,
            "candidate_text": "搜尋",
            "candidates": [
                {
                    "bbox": [40, 40, 32, 16],
                    "center": [56, 48],
                    "class_name": "text",
                    "text": "搜尋",
                }
            ],
            "detection_count": 1,
            "local_cursor": (56, 48),
        }
        (run_dir / "yolo_ocr" / "event_001.json").write_text(
            json.dumps(
                {
                    "event_index": event.index,
                    "candidates": payload["candidates"],
                    "detection_count": 1,
                    "candidate_text": "搜尋",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return payload

    monkeypatch.setattr(
        "src.common.runs_report_server.run_pointer_event_yolo_ocr",
        fake_yolo,
    )

    result = rerun_recording_event_yolo_ocr(runs_root, "recording_yolo_retry", 1)
    assert "搜尋" in result["instruction"]
    assert result["candidate_count"] == 1
    analysis = json.loads((run_root / "analysis" / "event_001.json").read_text(encoding="utf-8"))
    assert analysis["instruction"] == result["instruction"]
    assert analysis["vision"]["candidate_text"] == "搜尋"
    assert "landmarks" not in analysis
    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert "搜尋" in html


def test_rerun_recording_event_yolo_ocr_raises_when_empty(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_yolo_empty")

    monkeypatch.setattr(
        "src.common.runs_report_server.run_pointer_event_yolo_ocr",
        lambda *args, **kwargs: {
            "used_vision": True,
            "candidates": [],
            "detection_count": 0,
            "yolo_error": "Triton infer timed out after 20s",
        },
    )
    with pytest.raises(RuntimeError, match="timed out"):
        rerun_recording_event_yolo_ocr(runs_root, "recording_yolo_empty", 1)


def test_runs_report_server_yolo_ocr_endpoint(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_http_yolo")

    monkeypatch.setattr(
        "src.common.runs_report_server.rerun_recording_event_yolo_ocr",
        lambda *args, **kwargs: {
            "instruction": "將滑鼠移到「搜尋」文字，並點擊滑鼠一下。",
            "detection_count": 4,
            "candidate_count": 4,
        },
    )

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_yolo/events/1/yolo_ocr",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["candidate_count"] == 4
        assert "搜尋" in payload["instruction"]
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


def test_sync_recording_events_writes_survivors_and_purges_absorbed(tmp_path: Path) -> None:
    run_root = tmp_path / "recording_sync_coalesce"
    (run_root / "events").mkdir(parents=True)
    (run_root / "screenshots").mkdir()
    for index in (1, 2, 3):
        shot = run_root / "screenshots" / f"event_{index:03d}.jpeg"
        shot.write_bytes(b"jpeg")
        (run_root / "events" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": f"2026-08-13T08:00:0{index}+00:00",
                    "kind": "click",
                    "cursor_xy": [100, 200],
                    "button": "left",
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
                "started_at_utc": "2026-08-13T08:00:00+00:00",
                "stopped_at_utc": "2026-08-13T08:00:03+00:00",
                "event_count": 3,
                "events": [
                    "events/event_001.json",
                    "events/event_002.json",
                    "events/event_003.json",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    survivors = [
        RecordedEvent(
            index=1,
            timestamp_utc="2026-08-13T08:00:01+00:00",
            kind="triple_click",
            cursor_xy=(100, 200),
            button="left",
            screenshot_path=str(run_root / "screenshots" / "event_001.jpeg"),
        )
    ]
    result = sync_recording_events(run_root, survivors)

    assert result == {"kept": 1, "purged": [2, 3], "remaining": 1}
    assert (run_root / "events" / "event_001.json").is_file()
    assert not (run_root / "events" / "event_002.json").exists()
    assert not (run_root / "events" / "event_003.json").exists()
    assert not (run_root / "screenshots" / "event_002.jpeg").exists()
    assert not (run_root / "screenshots" / "event_003.jpeg").exists()
    kept = json.loads((run_root / "events" / "event_001.json").read_text(encoding="utf-8"))
    assert kept["kind"] == "triple_click"
    session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
    assert session["event_count"] == 1
    assert session["events"] == ["events/event_001.json"]


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


def _make_recording_three_event_run(runs_root: Path, name: str) -> Path:
    run_root = _make_recording_two_event_run(runs_root, name)
    shot3 = run_root / "screenshots" / "event_003.jpeg"
    shot3.write_bytes(b"\xff\xd8\xff\xd9")
    (run_root / "events" / "event_003.json").write_text(
        json.dumps(
            {
                "index": 3,
                "timestamp_utc": "2026-07-21T04:00:03+00:00",
                "kind": "click",
                "cursor_xy": [30, 40],
                "screenshot_path": str(shot3),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "analysis" / "event_003.json").write_text(
        json.dumps(
            {"event_index": 3, "instruction": "點擊「取消」按鈕"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_root / "yolo_ocr" / "event_003.json").write_text(
        json.dumps({"event_index": 3, "candidates": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
    session["event_count"] = 3
    session["events"] = [
        "events/event_001.json",
        "events/event_002.json",
        "events/event_003.json",
    ]
    (run_root / "session.json").write_text(
        json.dumps(session, ensure_ascii=False),
        encoding="utf-8",
    )
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    report["recorded"] = 3
    report["processed"] = 3
    report["cached"] = 3
    report["instructions"] = [
        "點擊「搜尋」按鈕",
        "點擊「確定」按鈕",
        "點擊「取消」按鈕",
    ]
    (run_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_root


def test_delete_recording_events_bulk_removes_files_and_rebuilds(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_three_event_run(runs_root, "recording_bulk_delete")

    result = delete_recording_events(runs_root, "recording_bulk_delete", [1, 3, 1])

    assert result == {
        "event_indices": [1, 3],
        "deleted": 2,
        "remaining": 1,
    }
    assert not (run_root / "events" / "event_001.json").exists()
    assert not (run_root / "events" / "event_003.json").exists()
    assert (run_root / "events" / "event_002.json").is_file()
    assert not (run_root / "screenshots" / "event_001.jpeg").exists()
    assert not (run_root / "screenshots" / "event_003.jpeg").exists()

    session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
    assert session["event_count"] == 1
    assert session["events"] == ["events/event_002.json"]

    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    assert report["recorded"] == 1
    assert report["instructions"] == ["點擊「確定」按鈕"]

    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert "點擊「確定」按鈕" in html
    assert "點擊「搜尋」按鈕" not in html
    assert "點擊「取消」按鈕" not in html
    assert 'class="select-step"' in html
    assert 'class="bulk-delete-steps"' in html


def test_runs_report_server_events_bulk_delete_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_three_event_run(runs_root, "recording_http_bulk_delete")

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_bulk_delete/events/delete",
            method="POST",
            data=json.dumps({"event_indices": [1, 3]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["deleted"] == 2
        assert payload["remaining"] == 1
        assert payload["event_indices"] == [1, 3]
        assert (run_root / "events" / "event_002.json").is_file()
        assert not (run_root / "events" / "event_001.json").exists()

        bad = urllib.request.Request(
            f"{base}/api/runs/recording_http_bulk_delete/events/delete",
            method="POST",
            data=json.dumps({"event_indices": [9]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad, timeout=5)
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

    assert result == {
        "text": "正確文字",
        "instruction": "輸入「正確文字」",
        "source": None,
    }
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


def test_apply_recording_event_text_updates_selected_ocr_choice(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_text_input_run(runs_root, "recording_text_ocr_edit")
    analysis_path = run_root / "analysis" / "event_001.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["text_resolution"]["ocr_text"] = "wrong ocr"
    analysis["text_resolution"]["ocr_options"] = ["wrong ocr", "搜尋"]
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False),
        encoding="utf-8",
    )

    result = apply_recording_event_text(
        runs_root,
        "recording_text_ocr_edit",
        1,
        text="正確文字",
        source="ocr",
        choice_text="wrong ocr",
    )

    assert result["source"] == "ocr"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["text_resolution"]["ocr_text"] == "正確文字"
    assert analysis["text_resolution"]["ocr_options"] == ["正確文字", "搜尋"]
    assert analysis["text_resolution"]["recorded_text"] == "wrng"
    assert analysis["text_resolution"]["resolved_text"] == "正確文字"
    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert "OCR：正確文字" in html
    assert "鍵盤：wrng" in html
    assert 'data-selected-source="ocr"' in html


def test_apply_recording_event_text_updates_selected_recorded_choice(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_text_input_run(runs_root, "recording_text_kb_edit")
    analysis_path = run_root / "analysis" / "event_001.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["text_resolution"]["source"] = "recorded"
    analysis["text_resolution"]["resolved_text"] = "wrng"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False),
        encoding="utf-8",
    )

    result = apply_recording_event_text(
        runs_root,
        "recording_text_kb_edit",
        1,
        text="fixed typed",
        source="recorded",
    )

    assert result["source"] == "recorded"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["text_resolution"]["recorded_text"] == "fixed typed"
    assert analysis["text_resolution"]["resolved_text"] == "fixed typed"
    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert "鍵盤：fixed typed" in html
    assert 'data-selected-source="recorded"' in html


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
        assert payload.get("source") is None
    finally:
        server.stop()


def test_runs_report_server_text_endpoint_with_source(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_text_input_run(runs_root, "recording_http_text_source")
    analysis_path = run_root / "analysis" / "event_001.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["text_resolution"]["ocr_text"] = "wrong ocr"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False),
        encoding="utf-8",
    )

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        body = json.dumps(
            {"text": "fixed value", "source": "ocr", "choice_text": "wrong ocr"},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_text_source/events/1/text",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["source"] == "ocr"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        assert analysis["text_resolution"]["ocr_text"] == "fixed value"
    finally:
        server.stop()


def test_apply_recording_event_expected_outcome_persists_and_rebuilds(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_outcome_edit")

    result = apply_recording_event_expected_outcome(
        runs_root,
        "recording_outcome_edit",
        1,
        expected_outcome="  對話框已開啟  ",
    )

    assert result == {"expected_outcome": "對話框已開啟", "use_expected_outcome": True}
    analysis = json.loads(
        (runs_root / "recording_outcome_edit" / "analysis" / "event_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert analysis["expected_outcome"] == "對話框已開啟"
    report = json.loads(
        (runs_root / "recording_outcome_edit" / "report.json").read_text(encoding="utf-8")
    )
    assert report["expected_outcomes"] == [None, "對話框已開啟"]
    html = (runs_root / "recording_outcome_edit" / "recording_steps.html").read_text(
        encoding="utf-8"
    )
    assert 'class="expected-outcome-input"' in html
    assert "對話框已開啟" in html
    assert 'data-expected-outcome="對話框已開啟"' in html


def test_apply_recording_event_expected_outcome_clears_when_empty(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_outcome_clear")

    result = apply_recording_event_expected_outcome(
        runs_root,
        "recording_outcome_clear",
        1,
        expected_outcome="   ",
    )

    assert result == {"expected_outcome": None, "use_expected_outcome": False}
    analysis = json.loads(
        (runs_root / "recording_outcome_clear" / "analysis" / "event_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert analysis["expected_outcome"] is None
    report = json.loads(
        (runs_root / "recording_outcome_clear" / "report.json").read_text(encoding="utf-8")
    )
    assert report["expected_outcomes"] == [None, None]


def test_runs_report_server_expected_outcome_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "recording_http_outcome")

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        body = json.dumps(
            {"expected_outcome": "畫面已更新"},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_outcome/events/1/expected_outcome",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["expected_outcome"] == "畫面已更新"
    finally:
        server.stop()


def test_add_recording_event_appends_wait_text_and_manual(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_add_append")

    wait_result = add_recording_event(
        runs_root,
        "recording_add_append",
        kind="wait",
        duration_seconds=3,
    )
    assert wait_result["event_index"] == 3
    assert wait_result["remaining"] == 3
    assert wait_result["instruction"] == "等待 3 秒"

    text_result = add_recording_event(
        runs_root,
        "recording_add_append",
        kind="text_input",
        text="  hello  ",
    )
    assert text_result["event_index"] == 4
    assert text_result["instruction"] == "輸入「hello」"

    manual_result = add_recording_event(
        runs_root,
        "recording_add_append",
        kind="manual",
        instruction="確認視窗已開啟",
        expected_outcome="  對話框出現  ",
    )
    assert manual_result["event_index"] == 5
    assert manual_result["instruction"] == "確認視窗已開啟"

    condition_result = add_recording_event(
        runs_root,
        "recording_add_append",
        kind="condition",
        text="「取代目的地中的檔案」文字",
        presence="has",
        then_action="點擊「取代目的地中的檔案」文字",
    )
    assert condition_result["event_index"] == 6
    assert (
        condition_result["instruction"]
        == "如果畫面上有「取代目的地中的檔案」文字，則點擊「取代目的地中的檔案」文字"
    )

    session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
    assert session["event_count"] == 6
    assert session["events"] == [
        "events/event_001.json",
        "events/event_002.json",
        "events/event_003.json",
        "events/event_004.json",
        "events/event_005.json",
        "events/event_006.json",
    ]

    wait_event = json.loads((run_root / "events" / "event_003.json").read_text(encoding="utf-8"))
    assert wait_event["kind"] == "wait"
    assert wait_event["duration_seconds"] == 3.0
    wait_analysis = json.loads(
        (run_root / "analysis" / "event_003.json").read_text(encoding="utf-8")
    )
    assert wait_analysis["instruction"] == "等待 3 秒"

    text_event = json.loads((run_root / "events" / "event_004.json").read_text(encoding="utf-8"))
    assert text_event["kind"] == "text_input"
    assert text_event["text"] == "hello"

    manual_analysis = json.loads(
        (run_root / "analysis" / "event_005.json").read_text(encoding="utf-8")
    )
    assert manual_analysis["expected_outcome"] == "對話框出現"

    condition_event = json.loads(
        (run_root / "events" / "event_006.json").read_text(encoding="utf-8")
    )
    assert condition_event["kind"] == "condition"
    assert condition_event["text"] == "「取代目的地中的檔案」文字"
    assert condition_event["presence"] == "has"
    assert condition_event["then_action"] == "點擊「取代目的地中的檔案」文字"

    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    assert report["instructions"] == [
        "點擊「搜尋」按鈕",
        "點擊「確定」按鈕",
        "等待 3 秒",
        "輸入「hello」",
        "確認視窗已開啟",
        "如果畫面上有「取代目的地中的檔案」文字，則點擊「取代目的地中的檔案」文字",
    ]
    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert "等待 3 秒" in html
    assert "輸入「hello」" in html
    assert "確認視窗已開啟" in html
    assert "如果畫面上有「取代目的地中的檔案」文字，則點擊「取代目的地中的檔案」文字" in html
    assert "條件" in html
    assert 'value="condition"' in html
    assert 'id="add-step-dialog"' in html
    assert (run_root / "screenshots" / "event_003.jpeg").is_file()


def test_add_recording_event_condition_missing_without_then(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_add_condition")

    result = add_recording_event(
        runs_root,
        "recording_add_condition",
        kind="condition",
        text="「載入中」文字",
        presence="missing",
    )
    assert result["instruction"] == "如果畫面上沒有「載入中」文字"

    event = json.loads((run_root / "events" / "event_003.json").read_text(encoding="utf-8"))
    assert event["kind"] == "condition"
    assert event["presence"] == "missing"
    assert "then_action" not in event

    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert "如果畫面上沒有「載入中」文字" in html
    assert "畫面上沒有" in html
    assert 'data-kind="condition"' in html


def test_add_recording_event_inserts_between_without_renumbering(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_add_insert")

    result = add_recording_event(
        runs_root,
        "recording_add_insert",
        kind="wait",
        after_event_index=1,
        duration_seconds=2,
    )
    assert result["event_index"] == 3

    session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
    assert session["events"] == [
        "events/event_001.json",
        "events/event_003.json",
        "events/event_002.json",
    ]
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    assert report["instructions"] == [
        "點擊「搜尋」按鈕",
        "等待 2 秒",
        "點擊「確定」按鈕",
    ]
    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    first = html.split('data-event-index="1"', 1)[1]
    wait_at = first.index('data-event-index="3"')
    second_at = first.index('data-event-index="2"')
    assert wait_at < second_at
    number_after_wait = first[wait_at:].split('instruction-number">', 1)[1][:2]
    assert number_after_wait == "2."
    assert '<span class="instruction-number">1.</span>' in html
    assert 'id="event-3"' in html


def test_add_recording_event_wait_clears_virtual_wait_instruction(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_add_wait_clear")
    analysis_path = run_root / "analysis" / "event_002.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["wait_instruction"] = "等待 11 秒"
    analysis["elapsed_since_previous_seconds"] = 10.25
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")

    result = add_recording_event(
        runs_root,
        "recording_add_wait_clear",
        kind="wait",
        after_event_index=1,
        duration_seconds=11,
    )
    assert result["instruction"] == "等待 11 秒"

    cleared = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert "wait_instruction" not in cleared
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    assert report["instructions"] == [
        "點擊「搜尋」按鈕",
        "等待 11 秒",
        "點擊「確定」按鈕",
    ]
    html = (run_root / "recording_steps.html").read_text(encoding="utf-8")
    assert html.count('data-kind="wait"') == 1
    wait_group = html.split('data-kind="wait"', 1)[1].split("</details>", 1)[0]
    assert "等待 11 秒" in wait_group
    assert 'data-instruction="等待 11 秒"' in wait_group


def test_delete_recording_event_preserves_custom_session_order(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_add_then_delete")
    add_recording_event(
        runs_root,
        "recording_add_then_delete",
        kind="manual",
        after_event_index=1,
        instruction="中間步驟",
    )

    result = delete_recording_event(runs_root, "recording_add_then_delete", 1)
    assert result["remaining"] == 2

    session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
    assert session["events"] == ["events/event_003.json", "events/event_002.json"]
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    assert report["instructions"] == ["中間步驟", "點擊「確定」按鈕"]


def test_add_recording_event_rejects_unknown_kind_and_empty_instruction(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_two_event_run(runs_root, "recording_add_reject")

    with pytest.raises(ValueError, match="unknown kind"):
        add_recording_event(
            runs_root,
            "recording_add_reject",
            kind="drag",
            instruction="nope",
        )
    with pytest.raises(ValueError, match="instruction is empty"):
        add_recording_event(
            runs_root,
            "recording_add_reject",
            kind="click",
            instruction="   ",
        )
    with pytest.raises(ValueError, match="instruction is empty"):
        add_recording_event(
            runs_root,
            "recording_add_reject",
            kind="manual",
        )


def test_apply_recording_event_instruction_persists(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_edit_instruction")

    result = apply_recording_event_instruction(
        runs_root,
        "recording_edit_instruction",
        1,
        instruction="  點擊「新目標」  ",
    )
    assert result == {
        "instruction": "點擊「新目標」",
        "use_char_target": False,
    }
    analysis = json.loads((run_root / "analysis" / "event_001.json").read_text(encoding="utf-8"))
    assert analysis["instruction"] == "點擊「新目標」"
    report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
    assert report["instructions"][0] == "點擊「新目標」"


def test_runs_report_server_add_event_endpoint(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_two_event_run(runs_root, "recording_http_add")

    server = RunsReportServer(runs_root)
    try:
        base = server.start()
        body = json.dumps(
            {
                "kind": "wait",
                "after_event_index": 1,
                "duration_seconds": 4,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/runs/recording_http_add/events/add",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
        assert payload["ok"] is True
        assert payload["event_index"] == 3
        assert payload["instruction"] == "等待 4 秒"
        session = json.loads((run_root / "session.json").read_text(encoding="utf-8"))
        assert session["events"] == [
            "events/event_001.json",
            "events/event_003.json",
            "events/event_002.json",
        ]

        bad = urllib.request.Request(
            f"{base}/api/runs/recording_http_add/events/add",
            method="POST",
            data=json.dumps({"kind": "nope"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad, timeout=5)
        assert exc_info.value.code == 400
    finally:
        server.stop()


def test_apply_instruction_updates_collected_script_text(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_landmark_run(runs_root, "recording_script_apply")
    result = apply_recording_event_instruction(
        runs_root,
        "recording_script_apply",
        1,
        instruction="點擊「新目標」",
    )
    assert result["instruction"] == "點擊「新目標」"
    script = collect_recording_script_text(run_root)
    assert "等待 2 秒" in script
    assert "點擊「新目標」" in script
    assert "# expected_outcome: 搜尋結果已顯示" in script
    assert not (run_root / "script.txt").exists()


def test_rename_recording_folder_moves_and_rejects_illegal(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = _make_recording_landmark_run(runs_root, "recording_to_rename")
    (run_root / "recording_steps.html").write_text("<html></html>", encoding="utf-8")

    result = rename_recording_folder(runs_root, "recording_to_rename", "開啟神網")
    assert result == {"old_id": "recording_to_rename", "new_id": "開啟神網"}
    assert not (runs_root / "recording_to_rename").exists()
    assert (runs_root / "開啟神網" / "session.json").is_file()
    assert (runs_root / "開啟神網" / "recording_steps.html").is_file()

    with pytest.raises(ValueError, match="invalid"):
        rename_recording_folder(runs_root, "開啟神網", "bad/name")
    with pytest.raises(ValueError, match="already exists"):
        _make_recording_landmark_run(runs_root, "other")
        rename_recording_folder(runs_root, "開啟神網", "other")


def test_resolve_deletable_run_folder_accepts_unicode(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _make_recording_landmark_run(runs_root, "錄製 測試")
    target = resolve_deletable_run_folder(runs_root, "錄製 測試")
    assert target.name == "錄製 測試"