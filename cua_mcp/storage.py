from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyperclip

from src.common.io_utils import read_json, write_json
from src.common.runtime_context import get_runtime_env


def _current_run_paths() -> tuple[Path, Path]:
    run_root, _ = get_runtime_env()
    storage_dir = run_root / "storage"
    storage_json = run_root / "storage.json"
    storage_dir.mkdir(parents=True, exist_ok=True)
    if not storage_json.exists():
        write_json(storage_json, [])
    return storage_dir, storage_json


def _append_storage_entry(entry: dict[str, Any]) -> dict[str, Any]:
    _, storage_json = _current_run_paths()
    rows = read_json(storage_json, default=[])
    if not isinstance(rows, list):
        rows = []
    rows.append(entry)
    write_json(storage_json, rows)
    return entry


def store_text(text: str, title: str = "", tags: list[str] | None = None) -> dict[str, Any]:
    """Store text context to this run's storage.json index."""
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "timestamp": now,
        "type": "text",
        "title": title,
        "text": text,
        "tags": tags or [],
    }
    _append_storage_entry(record)
    return {"status": "stored", "entry": record}


def store_clipboard_text(
    title: str = "",
    tags: list[str] | None = None,
    file_name: str = "",
) -> dict[str, Any]:
    """Read the OS clipboard as text, write it under this run's storage/ folder, and index it."""
    raw = pyperclip.paste()
    text = "" if raw is None else raw if isinstance(raw, str) else str(raw)
    if not text.strip():
        raise ValueError("clipboard is empty or has no usable text")

    storage_dir, _ = _current_run_paths()
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    raw_name = (file_name or f"clipboard_{stamp}").strip()
    base = Path(raw_name).name
    if not base.lower().endswith(".txt"):
        base = f"{base}.txt"
    dst_path = storage_dir / base
    if dst_path.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        dst_path = storage_dir / f"{dst_path.stem}_{suffix}{dst_path.suffix}"

    dst_path.write_text(text, encoding="utf-8")

    record = {
        "timestamp": now.isoformat(),
        "type": "text",
        "source": "clipboard",
        "title": title,
        "text": text,
        "stored_path": str(dst_path),
        "file_name": dst_path.name,
        "tags": tags or [],
    }
    _append_storage_entry(record)
    return {"status": "stored", "entry": record}


def store_image(
    image_path: str,
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Copy an image to this run's storage folder and index it."""
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not src.is_file():
        raise ValueError(f"image_path must be a file: {image_path}")

    storage_dir, _ = _current_run_paths()
    safe_alias = alias.strip()
    dst_name = safe_alias or src.name
    dst_path = storage_dir / dst_name
    if dst_path.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        dst_path = storage_dir / f"{dst_path.stem}_{suffix}{dst_path.suffix}"
    shutil.copy2(src, dst_path)

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "timestamp": now,
        "type": "image",
        "summary": summary,
        "source_path": str(src),
        "stored_path": str(dst_path),
        "file_name": dst_path.name,
        "tags": tags or [],
    }
    _append_storage_entry(record)
    return {"status": "stored", "entry": record}
