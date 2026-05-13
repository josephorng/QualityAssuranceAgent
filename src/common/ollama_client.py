from __future__ import annotations

import asyncio
from typing import Any

from ollama import AsyncClient, ChatResponse, Message

from src.common.llm_client import (
    LLMClient,
    ResponseFormatParam,
    ThinkParam,
    ToolCall,
)
from src.common.run_state import get_run_state_manager

__all__ = ["OllamaClient", "ResponseFormatParam", "ThinkParam", "ToolCall"]


class OllamaClient(LLMClient):
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
        *,
        append_image_sizes: bool = True,
        think: ThinkParam = None,
    ) -> Message:
        """
        Run chat with an explicit message list (no merge with _message_history).
        Used for multi-turn tool loops where the caller owns the full transcript.

        Args:
            tools: Tool definitions to provide to the model. If None, uses the default tool set from ``cua_mcp.tools.TOOL_FUNCTIONS``. Pass an empty list to disable tools.
            think: When set, forwarded to Ollama ``think`` (thinking models).
        
        Returns:
            Message: The response message.
        """
        prepared_messages = (
            self._append_last_message_image_sizes(messages)
            if append_image_sizes
            else messages
        )
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": prepared_messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        }
        if response_format is not None:
            chat_kwargs["format"] = response_format

        if tools:
            chat_kwargs["tools"] = tools

        if think is not None:
            chat_kwargs["think"] = think

        last_assistant_idx = -1
        for idx in reversed(range(len(prepared_messages))):
            if prepared_messages[idx].get("role") == "assistant":
                last_assistant_idx = idx
                break
        get_run_state_manager().log_info(
            f"Ollama chat_messages for model={model} n_messages={len(prepared_messages)} "
            f"tools_count={len(tools) if tools else 0} "
            f"response_format_set={response_format is not None}"
            f"last_assistant_messages=\n{prepared_messages[last_assistant_idx:]}"
        )
        response: ChatResponse = await self.client.chat(**chat_kwargs)
        get_run_state_manager().log_info(f"Ollama chat_messages response=\n{response}")
        response_message = response.message
        tool_calls = response_message.tool_calls
        has_thinking = bool((response_message.thinking or "").strip())
        if not response_message.content and not tool_calls and not has_thinking:
            get_run_state_manager().log_info("Ollama returned empty response and no tools; retrying in 5 seconds.")
            await asyncio.sleep(5)
            return await self.chat_messages(
                model=model,
                messages=messages,
                tools=tools,
                response_format=response_format,
                append_image_sizes=append_image_sizes,
                think=think,
            )
        return response_message
