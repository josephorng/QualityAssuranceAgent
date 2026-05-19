from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_text(path: Path, text: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def pop_last_nonempty_line(path: Path) -> str | None:
    """Remove the last non-empty line from a text file; return it, or None if empty."""
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        path.write_text("", encoding="utf-8")
        return None
    removed = lines.pop()
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return removed


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    ensure_parent(path)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
