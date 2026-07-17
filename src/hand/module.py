from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.types import TextContent

from cua_mcp.tools import mcp_server
from src.common.io_utils import append_csv_row
from src.common.models import ExecutionResult, ToolCommand
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings

if TYPE_CHECKING:
    from src.eye.module import EyeModule

_AFTER_ACTION_SETTLE_S = 1.0


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
    def __init__(self, eye: EyeModule | None = None) -> None:
        self.settings = load_settings()
        self.ollama = get_llm_client()
        self.run_root, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.run_id, self.run_root.name)
        self._eye = eye
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
        if candidate.is_file():
            return str(candidate)
        resolved = self.manager.require_paths().eye_dir / candidate.name
        return str(resolved) if resolved.is_file() else None

    async def _capture_report_screenshot(
        self,
        label: str,
        settle_delay: float = 0.0,
    ) -> str | None:
        if self._eye is None:
            return None
        if settle_delay > 0:
            await asyncio.sleep(settle_delay)
        try:
            event = await self._eye.capture_once()
        except Exception as exc:
            self.manager.log_error(f"Failed to capture {label} screenshot for report: {exc}")
            return None
        return event.screenshot_path

    async def _remap_unknown_action(
        self,
        action: str,
        args: dict[str, Any],
        error_message: str,
    ) -> tuple[str, dict[str, Any]] | None:
        try:
            tools = await mcp_server.list_tools()
        except Exception as exc:
            self.manager.log_error(f"Unable to list tools for remap: {exc}")
            return None

        tool_names = sorted({tool.name for tool in tools if getattr(tool, "name", None)})
        if not tool_names:
            return None

        prompt = get_prompt("hand_remap_tool").format(
            action=action,
            failed_args_json=json.dumps(args, ensure_ascii=False),
            error_message=error_message,
            available_tools="\n".join(f"- {name}" for name in tool_names),
        )

        try:
            reply = await self.ollama.chat_messages(
                model=self.settings.brain_lm,
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                response_format="json",
            )
        except Exception as exc:
            self.manager.log_error(f"Ollama remap call failed: {exc}")
            return None

        if not reply or not reply.content:
            return None
        try:
            payload = json.loads(reply.content)
        except json.JSONDecodeError:
            self.manager.log_error(f"Ollama remap returned non-JSON content: {reply.content}")
            return None
        if not isinstance(payload, dict):
            return None

        candidate_action = payload.get("action")
        candidate_args = payload.get("args", args)
        if not isinstance(candidate_action, str):
            return None
        candidate_action = candidate_action.strip()
        if candidate_action not in tool_names:
            self.manager.log_error(f"Ollama remap suggested invalid tool: {candidate_action}")
            return None
        if not isinstance(candidate_args, dict):
            candidate_args = args
        return candidate_action, candidate_args

    async def _exec_action(self, cmd: ToolCommand) -> ExecutionResult:
        action = cmd.action
        args = cmd.args
        screenshot_before_path = self._resolve_screenshot_path(cmd.screenshot_before_path)
        screenshot_after_path = self._resolve_screenshot_path(cmd.screenshot_after_path)
        screenshot_path = screenshot_before_path or self._resolve_screenshot_path(cmd.screenshot_name)

        if screenshot_before_path is None:
            screenshot_before_path = await self._capture_report_screenshot("before-action")

        try:
            tool_output = await mcp_server.call_tool(action, args)
            if screenshot_after_path is None:
                screenshot_after_path = await self._capture_report_screenshot(
                    "after-action", settle_delay=_AFTER_ACTION_SETTLE_S
                )
            return ExecutionResult(
                ok=True,
                action=action,
                args=_merged_tool_args(args, tool_output),
                timestamp=datetime.now(timezone.utc),
                screenshot_name=screenshot_before_path,
                screenshot_before_path=screenshot_before_path,
                screenshot_after_path=screenshot_after_path,
                message=cmd.reason or "executed",
            )
        except Exception as exc:
            error_message = str(exc)
            if "Unknown tool" in error_message:
                remapped = await self._remap_unknown_action(action, args, error_message)
                if remapped is not None:
                    remapped_action, remapped_args = remapped
                    try:
                        tool_output = await mcp_server.call_tool(remapped_action, remapped_args)
                        if screenshot_after_path is None:
                            screenshot_after_path = await self._capture_report_screenshot(
                                "after-action", settle_delay=_AFTER_ACTION_SETTLE_S
                            )
                        return ExecutionResult(
                            ok=True,
                            action=remapped_action,
                            args=_merged_tool_args(remapped_args, tool_output),
                            timestamp=datetime.now(timezone.utc),
                            screenshot_name=screenshot_before_path,
                            screenshot_before_path=screenshot_before_path,
                            screenshot_after_path=screenshot_after_path,
                            message=f"{cmd.reason or 'executed'} (remapped from {action})",
                        )
                    except Exception as remap_exc:
                        error_message = (
                            f"{error_message}; remap retry failed with {remapped_action}: {remap_exc}"
                        )
            if screenshot_after_path is None:
                screenshot_after_path = await self._capture_report_screenshot(
                    "after-action", settle_delay=_AFTER_ACTION_SETTLE_S
                )
            return ExecutionResult(
                ok=False,
                action=action,
                args=args,
                timestamp=datetime.now(timezone.utc),
                screenshot_name=screenshot_before_path,
                screenshot_before_path=screenshot_before_path,
                screenshot_after_path=screenshot_after_path,
                message=error_message,
            )

    async def execute_tool_command(self, cmd: ToolCommand) -> ExecutionResult:
        async with self._lock:
            self._busy = True
            result = await self._exec_action(cmd)
            paths = self.manager.require_paths()
            append_csv_row(
                paths.hand_csv,
                fieldnames=[
                    "timestamp",
                    "action",
                    "args",
                    "ok",
                    "screenshot_name",
                    "screenshot_before_path",
                    "screenshot_after_path",
                    "message",
                ],
                row={
                    "timestamp": result.timestamp.isoformat(),
                    "action": result.action,
                    "args": result.args,
                    "ok": result.ok,
                    "screenshot_name": result.screenshot_name or "",
                    "screenshot_before_path": result.screenshot_before_path or "",
                    "screenshot_after_path": result.screenshot_after_path or "",
                    "message": result.message,
                },
            )
            self._busy = False
            return result

