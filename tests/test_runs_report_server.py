from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.common.runs_report_server import (
    RunsReportServer,
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
