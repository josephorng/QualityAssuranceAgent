from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
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


def imread_bgr(path: str | os.PathLike[str]) -> Any:
    """Read an OpenCV BGR image; Unicode paths work on Windows (unlike ``cv2.imread``)."""
    import cv2
    import numpy as np

    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def open_path_with_default_app(path: Path) -> None:
    """Open a file or folder with the OS default application."""
    resolved = path.resolve()
    if sys.platform == "win32":
        os.startfile(resolved)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(resolved)], check=True)
        return
    subprocess.run(["xdg-open", str(resolved)], check=True)


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    ensure_parent(path)
    # ``run_state.init_run`` pre-creates an empty ``hand.csv``, so a plain ``path.exists()``
    # check would skip the header forever. Write the header whenever the file has no content.
    needs_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
