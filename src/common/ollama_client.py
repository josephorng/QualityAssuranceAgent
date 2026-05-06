from __future__ import annotations

import asyncio
from typing import Any, Literal

from ollama import AsyncClient, ChatResponse, Message
from pydantic import BaseModel

from src.common.run_state import get_run_state_manager

ResponseFormatParam = Literal["json"] | dict[str, Any] | None


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
        tools: list[Any] | None = None,
        response_format: ResponseFormatParam = None,
    ) -> Message:
        """
        Run chat with an explicit message list (no merge with _message_history).
        Used for multi-turn tool loops where the caller owns the full transcript.

        Args:
            tools: Tool definitions to provide to the model. If None, uses the default tool set from ``cua_mcp.tools.TOOL_FUNCTIONS``. Pass an empty list to disable tools.
        
        Returns:
            Message: The response message.
        """
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        }
        if response_format is not None:
            chat_kwargs["format"] = response_format

        if tools:
            chat_kwargs["tools"] = tools
        
        last_assistant_idx = -1
        for idx in reversed(range(len(messages))):
            if messages[idx].get("role") == "assistant":
                last_assistant_idx = idx
                break
        get_run_state_manager().log_info(
            f"Ollama chat_messages for model={model} n_messages={len(messages)} "
            f"tools_count={len(tools) if tools else 0} "
            f"response_format_set={response_format is not None}"
            f"last_assistant_messages=\n{messages[last_assistant_idx:]}"
        )
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
                tools=tools,
                response_format=response_format,
            )
        return response_message

