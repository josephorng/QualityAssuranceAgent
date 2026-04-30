from __future__ import annotations

import asyncio
from typing import Any

from ollama import AsyncClient
from ollama import ChatResponse, Message

from cua_mcp.tools import TOOL_FUNCTIONS
from src.common.run_state import get_run_state_manager

from pydantic import BaseModel

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]

class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int = 60) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = AsyncClient(host=self.host)

    async def chat_messages(
        self,
        model: str,
        messages: list[dict[str, Any]],
        use_tools: bool = True,
    ) -> Message:
        """
        Run chat with an explicit message list (no merge with _message_history).
        Used for multi-turn tool loops where the caller owns the full transcript.
        
        Returns:
            Message: The response message.
        """
        get_run_state_manager().log_info(
            f"Ollama chat_messages for model={model} n_messages={len(messages)} use_tools={use_tools}"
        )
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        }
        if use_tools:
            chat_kwargs["tools"] = TOOL_FUNCTIONS
        response: ChatResponse = await self.client.chat(**chat_kwargs)
        get_run_state_manager().log_info(f"Ollama chat_messages response=\n{response}")
        response_message = response.message
        content = response_message.content
        tool_calls = response_message.tool_calls
        if not content and not tool_calls:
            get_run_state_manager().log_info("Ollama returned empty response and no tools; retrying in 5 seconds.")
            await asyncio.sleep(5)
            return await self.chat_messages(
                model=model,
                messages=messages,
                use_tools=use_tools,
            )
        return response_message
