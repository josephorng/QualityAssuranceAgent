from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from ollama import AsyncClient, ChatResponse, Message
from PIL import Image
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
        *,
        append_image_sizes: bool = True,
    ) -> Message:
        """
        Run chat with an explicit message list (no merge with _message_history).
        Used for multi-turn tool loops where the caller owns the full transcript.

        Args:
            tools: Tool definitions to provide to the model. If None, uses the default tool set from ``cua_mcp.tools.TOOL_FUNCTIONS``. Pass an empty list to disable tools.
        
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
        if not response_message.content and not tool_calls:
            get_run_state_manager().log_info("Ollama returned empty response and no tools; retrying in 5 seconds.")
            await asyncio.sleep(5)
            return await self.chat_messages(
                model=model,
                messages=messages,
                tools=tools,
                response_format=response_format,
                append_image_sizes=append_image_sizes,
            )
        return response_message

    def _append_last_message_image_sizes(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages
        last = messages[-1]
        images = last.get("images")
        if not isinstance(images, list) or not images:
            return messages
        content = last.get("content")
        if not isinstance(content, str):
            return messages
        size_entries: list[str] = []
        for image in images:
            if not isinstance(image, str):
                continue
            image_path = Path(image)
            try:
                with Image.open(image_path) as image_obj:
                    width, height = image_obj.size
                size_entries.append(f"{image_path.name}={width}x{height}")
            except (OSError, ValueError):
                size_entries.append(f"{image_path.name}=unavailable")
        if not size_entries:
            return messages
        message_with_sizes = dict(last)
        message_with_sizes["content"] = (
            f"{content}\n\nImageSizes: {', '.join(size_entries)}"
        )
        return [*messages[:-1], message_with_sizes]

