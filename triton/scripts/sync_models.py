#!/usr/bin/env python3
"""Copy ONNX weights from cua_mcp/ into the Triton model repository."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1] / "model_repository"

SOURCES = {
    REPO / "yolo_ui" / "1" / "model.onnx": ROOT / "cua_mcp" / "best.onnx",
    REPO / "crnn_ocr" / "1" / "model.onnx": ROOT
    / "cua_mcp"
    / "read_screen_text"
    / "ocr_model_finetuned.onnx",
}


def main() -> int:
    missing: list[str] = []
    for dest, src in SOURCES.items():
        if not src.is_file():
            missing.append(str(src))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"copied {src} -> {dest}")

    if missing:
        print("missing source ONNX files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
