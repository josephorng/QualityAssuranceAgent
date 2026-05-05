from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from ollama import AsyncClient, ChatResponse, Client, Message
from pydantic import BaseModel

from src.common.run_state import get_run_state_manager


def _tool_functions() -> list[Any]:
    """Deferred import so callers (e.g. cua_mcp.tool_module) avoid circular imports with cua_mcp.tools."""
    from cua_mcp.tools import TOOL_FUNCTIONS

    return TOOL_FUNCTIONS

ResponseFormatParam = Literal["json"] | dict[str, Any] | None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]

class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int = 60) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = AsyncClient(host=self.host)
        self.sync_client = Client(host=self.host)

    async def chat_messages(
        self,
        model: str,
        messages: list[dict[str, Any]],
        use_tools: bool = True,
        response_format: ResponseFormatParam = None,
    ) -> Message:
        """
        Run chat with an explicit message list (no merge with _message_history).
        Used for multi-turn tool loops where the caller owns the full transcript.
        
        Returns:
            Message: The response message.
        """
        get_run_state_manager().log_info(
            f"Ollama chat_messages for model={model} n_messages={len(messages)} "
            f"use_tools={use_tools} response_format_set={response_format is not None}"
        )
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        }
        if response_format is not None:
            chat_kwargs["format"] = response_format
        if use_tools:
            chat_kwargs["tools"] = _tool_functions()
        response: ChatResponse = await self.client.chat(**chat_kwargs)
        get_run_state_manager().log_info(f"Ollama chat_messages response=\n{response}")
        response_message = response.message
        tool_calls = response_message.tool_calls
        if not response_message.content and not tool_calls:
            get_run_state_manager().log_info("Ollama returned empty response and no tools; retrying in 5 seconds.")
            await asyncio.sleep(5)
            return await self.chat_messages(
                model=model,
                messages=messages,
                use_tools=use_tools,
                response_format=response_format,
            )
        return response_message

    def chat_messages_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
        use_tools: bool = True,
        response_format: ResponseFormatParam = None,
    ) -> Message:
        """
        Same contract as chat_messages, using the synchronous Ollama client (for MCP / pyautogui paths).
        """
        get_run_state_manager().log_info(
            f"Ollama chat_messages_sync for model={model} n_messages={len(messages)} "
            f"use_tools={use_tools} response_format_set={response_format is not None}"
        )
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        }
        if response_format is not None:
            chat_kwargs["format"] = response_format
        if use_tools:
            chat_kwargs["tools"] = _tool_functions()
        response: ChatResponse = self.sync_client.chat(**chat_kwargs)
        get_run_state_manager().log_info(f"Ollama chat_messages_sync response=\n{response}")
        response_message = response.message
        tool_calls = response_message.tool_calls
        if not response_message.content and not tool_calls:
            get_run_state_manager().log_info(
                "Ollama returned empty response and no tools; retrying in 5 seconds (sync)."
            )
            time.sleep(5)
            return self.chat_messages_sync(
                model=model,
                messages=messages,
                use_tools=use_tools,
                response_format=response_format,
            )
        return response_message
