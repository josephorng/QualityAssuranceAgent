from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.io_utils import update_csv_last_row
from src.common.models import ToolCommand
from src.hand.module import (
    HAND_CSV_FIELDNAMES,
    HandModule,
    _focus_rect_after_csv,
    _sample_debug_focus_rect_after,
    _sample_debug_foreground_rect_after,
)


def test_sample_debug_focus_rect_after_formats_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.recorder.focus_point.sample_debug_focus_rect",
        lambda **_kwargs: (10, 20, 30, 40),
    )
    assert _sample_debug_focus_rect_after() == [10, 20, 30, 40]
    assert _focus_rect_after_csv([10, 20, 30, 40]) == "[10,20,30,40]"
    assert _focus_rect_after_csv(None) == ""


def test_sample_debug_foreground_rect_after_formats_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.recorder.verify_signals.sample_debug_foreground_rect",
        lambda: [1, 2, 300, 200],
    )
    assert _sample_debug_foreground_rect_after() == [1, 2, 300, 200]


def test_update_csv_last_row_patches_after_columns(tmp_path: Path) -> None:
    path = tmp_path / "hand.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HAND_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "t0",
                "action": "move_mouse",
                "args": "{}",
                "ok": "True",
                "screenshot_name": "",
                "screenshot_before_path": "before0.png",
                "screenshot_after_path": "",
                "message": "executed",
                "focus_rect_after": "",
                "foreground_rect_after": "",
            }
        )
        writer.writerow(
            {
                "timestamp": "t1",
                "action": "click",
                "args": "{}",
                "ok": "True",
                "screenshot_name": "",
                "screenshot_before_path": "before1.png",
                "screenshot_after_path": "",
                "message": "executed",
                "focus_rect_after": "",
                "foreground_rect_after": "",
            }
        )

    assert update_csv_last_row(
        path,
        {
            "screenshot_after_path": "after1.png",
            "focus_rect_after": "[1,2,3,4]",
            "foreground_rect_after": "[5,6,7,8]",
        },
    )

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["screenshot_after_path"] == ""
    assert rows[1]["screenshot_after_path"] == "after1.png"
    assert rows[1]["focus_rect_after"] == "[1,2,3,4]"
    assert rows[1]["foreground_rect_after"] == "[5,6,7,8]"


def _hand_for_unit(tmp_path: Path) -> HandModule:
    hand = HandModule.__new__(HandModule)
    hand.settings = MagicMock()
    hand.ollama = MagicMock()
    hand.run_root = tmp_path
    hand.run_id = "test"
    hand.manager = MagicMock()
    hand.manager.require_paths = MagicMock(
        return_value=MagicMock(hand_csv=tmp_path / "hand.csv", eye_dir=tmp_path)
    )
    hand.manager.log_info = MagicMock()
    hand.manager.log_error = MagicMock()
    hand._eye = MagicMock()
    hand._busy = False
    import asyncio

    hand._lock = asyncio.Lock()
    return hand


@pytest.mark.asyncio
async def test_exec_action_skips_after_screenshot(monkeypatch, tmp_path: Path) -> None:
    hand = _hand_for_unit(tmp_path)
    capture = AsyncMock(return_value="before.png")
    hand._capture_report_screenshot = capture
    monkeypatch.setattr(
        "src.hand.module.mcp_server.call_tool",
        AsyncMock(return_value={"ok": True}),
    )

    result = await hand._exec_action(ToolCommand(action="click", args={"button": "left"}))

    assert result.ok is True
    assert result.screenshot_before_path == "before.png"
    assert result.screenshot_after_path is None
    assert result.focus_rect_after is None
    assert result.foreground_rect_after is None
    assert capture.await_count == 1
    assert capture.await_args.args[0] == "before-action"


@pytest.mark.asyncio
async def test_capture_after_for_last_action_patches_hand_csv(
    monkeypatch, tmp_path: Path
) -> None:
    hand = _hand_for_unit(tmp_path)
    hand_csv = tmp_path / "hand.csv"
    with hand_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HAND_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "click",
                "args": "{}",
                "ok": "True",
                "screenshot_name": "",
                "screenshot_before_path": "before.png",
                "screenshot_after_path": "",
                "message": "executed",
                "focus_rect_after": "",
                "foreground_rect_after": "",
            }
        )

    hand._capture_report_screenshot = AsyncMock(return_value=str(tmp_path / "after.png"))
    monkeypatch.setattr(
        "src.hand.module._sample_debug_focus_rect_after",
        lambda: [8, 6, 40, 28],
    )
    monkeypatch.setattr(
        "src.hand.module._sample_debug_foreground_rect_after",
        lambda: [4, 4, 120, 56],
    )

    updated = await hand.capture_after_for_last_action()

    assert updated is not None
    assert updated["screenshot_after_path"] == str(tmp_path / "after.png")
    with hand_csv.open(encoding="utf-8", newline="") as handle:
        row = list(csv.DictReader(handle))[-1]
    assert row["screenshot_after_path"] == str(tmp_path / "after.png")
    assert row["focus_rect_after"] == "[8,6,40,28]"
    assert row["foreground_rect_after"] == "[4,4,120,56]"
