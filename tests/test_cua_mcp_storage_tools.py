from __future__ import annotations

import os
from pathlib import Path

from cua_mcp.tool_module import _screenshot, _store_image, _store_text
from src.common.io_utils import read_json


def test_store_text_writes_storage_json(tmp_path: Path) -> None:
    os.environ["CUA_RUN_ROOT"] = str(tmp_path / "run_a")
    result = _store_text("important context", title="ctx", tags=["main"])
    entry = result["entry"]

    storage_json = Path(os.environ["CUA_RUN_ROOT"]) / "storage.json"
    rows = read_json(storage_json, default=[])

    assert result["status"] == "stored"
    assert entry["type"] == "text"
    assert entry["text"] == "important context"
    assert isinstance(rows, list)
    assert rows[-1]["text"] == "important context"


def test_store_image_copies_file_and_indexes(tmp_path: Path) -> None:
    os.environ["CUA_RUN_ROOT"] = str(tmp_path / "run_b")
    source_image = tmp_path / "source.png"
    source_image.write_bytes(b"fakepng")

    result = _store_image(str(source_image), summary="screen context", alias="context.png")
    entry = result["entry"]

    storage_root = Path(os.environ["CUA_RUN_ROOT"])
    storage_json = storage_root / "storage.json"
    stored_path = Path(entry["stored_path"])
    rows = read_json(storage_json, default=[])

    assert result["status"] == "stored"
    assert entry["type"] == "image"
    assert stored_path.exists()
    assert stored_path.parent == storage_root / "storage"
    assert stored_path.name == "context.png"
    assert isinstance(rows, list)
    assert rows[-1]["stored_path"] == str(stored_path)


def test_screenshot_relative_path_writes_under_run_storage(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run_screenshot"
    monkeypatch.setenv("CUA_RUN_ROOT", str(run_root))
    recorded: dict[str, str] = {}

    def fake_screenshot_to_file(path: str | None = None):
        assert path is not None
        recorded["path"] = path
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fakepng")
        return {"path": path}

    monkeypatch.setattr(
        "cua_mcp.tool_module.hand_tools.screenshot_to_file",
        fake_screenshot_to_file,
    )

    out = _screenshot(path="evidence.png")
    storage = run_root / "storage"
    assert Path(out["path"]) == storage / "evidence.png"
    assert recorded["path"] == str(storage / "evidence.png")


def test_screenshot_empty_path_uses_timestamp_in_run_storage(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run_screenshot2"
    monkeypatch.setenv("CUA_RUN_ROOT", str(run_root))
    recorded: dict[str, str] = {}

    def fake_screenshot_to_file(path: str | None = None):
        assert path is not None
        recorded["path"] = path
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fakepng")
        return {"path": path}

    monkeypatch.setattr(
        "cua_mcp.tool_module.hand_tools.screenshot_to_file",
        fake_screenshot_to_file,
    )

    out = _screenshot(path="")
    storage = run_root / "storage"
    assert Path(out["path"]).parent == storage
    assert Path(out["path"]).name.startswith("screenshot_")
    assert Path(out["path"]).suffix == ".png"
