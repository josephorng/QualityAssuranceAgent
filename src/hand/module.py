from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.types import TextContent

from cua_mcp.tools import mcp_server
from src.common.io_utils import append_csv_row
from src.common.models import ExecutionResult, ToolCommand
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings


def _parse_mcp_tool_result(tool_output: Any) -> dict[str, Any] | None:
    """
    FastMCP call_tool(..., convert_result=True) turns dict returns into TextContent JSON,
    not a plain dict. Recover the structured payload for logging and the brain loop.
    """
    if isinstance(tool_output, dict):
        return tool_output
    if not tool_output:
        return None
    blocks = tool_output if isinstance(tool_output, (list, tuple)) else (tool_output,)
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    if not parts:
        return None
    try:
        data = json.loads("".join(parts))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _merged_tool_args(original: dict[str, Any], tool_output: Any) -> dict[str, Any]:
    parsed = _parse_mcp_tool_result(tool_output)
    if parsed is None:
        return original
    return {**original, **parsed}


class HandModule:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.run_root, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.run_id, self.run_root.name)
        self._busy = False
        self._lock = asyncio.Lock()
        self.manager.log_info(f"Hand module initialized run_id={self.run_id}")

    @property
    def busy(self) -> bool:
        return self._busy

    def _resolve_screenshot_path(self, screenshot_name: str | None) -> str | None:
        if not screenshot_name:
            return None
        candidate = Path(screenshot_name)
        return str(self.manager.require_paths().eye_dir / candidate.name)

    async def _exec_action(self, cmd: ToolCommand) -> ExecutionResult:
        action = cmd.action
        args = cmd.args
        screenshot_path = self._resolve_screenshot_path(cmd.screenshot_name)
        try:
            tool_output = await mcp_server.call_tool(action, args)
            return ExecutionResult(
                ok=True,
                action=action,
                args=_merged_tool_args(args, tool_output),
                timestamp=datetime.now(timezone.utc),
                screenshot_name=screenshot_path,
                message=cmd.reason or "executed",
            )
        except Exception as exc:
            return ExecutionResult(
                ok=False,
                action=action,
                args=args,
                timestamp=datetime.now(timezone.utc),
                screenshot_name=screenshot_path,
                message=str(exc),
            )

    async def execute_tool_command(self, cmd: ToolCommand) -> ExecutionResult:
        async with self._lock:
            self._busy = True
            result = await self._exec_action(cmd)
            append_csv_row(
                self.manager.require_paths().hand_csv,
                fieldnames=["timestamp", "action", "args", "ok", "screenshot_name", "message"],
                row={
                    "timestamp": result.timestamp.isoformat(),
                    "action": result.action,
                    "args": result.args,
                    "ok": result.ok,
                    "screenshot_name": result.screenshot_name or "",
                    "message": result.message,
                },
            )
            self._busy = False
            return result

