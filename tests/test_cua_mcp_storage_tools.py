from __future__ import annotations

import os
from pathlib import Path

from cua_mcp.tools import store_image, store_text
from src.common.io_utils import read_json


def test_store_text_writes_storage_json(tmp_path: Path) -> None:
    os.environ["CUA_RUN_ROOT"] = str(tmp_path / "run_a")
    result = store_text("important context", title="ctx", tags=["main"])
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

    result = store_image(str(source_image), summary="screen context", alias="context.png")
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
