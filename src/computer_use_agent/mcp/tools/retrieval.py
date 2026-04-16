from __future__ import annotations

from datetime import datetime


def get_running_programs(args: dict) -> dict:
    _ = args
    return {"ok": True, "programs": ["explorer.exe", "cursor.exe", "powershell.exe"]}


def mock_ocr(args: dict) -> dict:
    target = args.get("target", "screen")
    now = datetime.utcnow().isoformat()
    return {"ok": True, "target": target, "text": f"Mock OCR output @ {now}."}
