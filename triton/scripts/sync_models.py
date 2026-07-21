#!/usr/bin/env python3
"""Copy ONNX weights from cua_mcp/ into the Triton model repository."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1] / "model_repository"
# Network clone of the local triton/ folder
REMOTE_REPO = Path(r"\\192.168.0.9\Joseph\yolo+ocr\triton") / "model_repository"

SOURCES = {
    Path("yolo_ui") / "1" / "model.onnx": ROOT / "cua_mcp" / "best.onnx",
    Path("crnn_ocr") / "1" / "model.onnx": ROOT
    / "cua_mcp"
    / "read_screen_text"
    / "ocr_model_finetuned.onnx",
}


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"copied {src} -> {dest}")


def main() -> int:
    missing: list[str] = []
    for rel, src in SOURCES.items():
        if not src.is_file():
            missing.append(str(src))
            continue
        _copy(src, REPO / rel)
        try:
            _copy(src, REMOTE_REPO / rel)
        except OSError as exc:
            print(f"failed to copy to remote {REMOTE_REPO / rel}: {exc}", file=sys.stderr)
            return 1

    if missing:
        print("missing source ONNX files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
