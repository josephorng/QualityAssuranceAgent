from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ollama import AsyncClient

from cua_mcp.tools import get_ollama_tools
from pydantic import BaseModel
from src.common.run_state import RunStateManager

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]

class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int = 60, log_manager: RunStateManager | None = None) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = AsyncClient(host=self.host)
        self._message_history: list[dict[str, Any]] = []
        self.log_manager = log_manager

    async def _stream_chat(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        use_tools: bool = False,
        store_messages: bool = False,
        interrupt_checker: Callable[[], bool] | None = None,
    ) -> tuple[str, list[ToolCall], bool]:
        chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        
        if self.log_manager:
            self.log_manager.log_debug(f"Ollama generating for model={model} prompt=\n{prompt}\nimage_paths={image_paths} use_tools={use_tools} store_messages={store_messages}")
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_paths:
            message["images"] = image_paths
        request_messages = [*self._message_history, message] if store_messages else [message]
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "stream": True,
            "options": {"num_ctx": 4096},
        }
        if use_tools:
            chat_kwargs["tools"] = get_ollama_tools()
        stream = await self.client.chat(**chat_kwargs)
        async for part in stream:
            if interrupt_checker and interrupt_checker():
                return "INTERRUPTED", True
            chunk_content = part.get("message", {}).get("content", "")
            if chunk_content:
                print(chunk_content, end="", flush=True)
                chunks.append(chunk_content)
            if "tool_calls" in part:
                tool_calls.extend([ToolCall(name=call["name"], arguments=call["arguments"]) for call in part["tool_calls"]])
        if chunks:
            print()
        response_text = "".join(chunks).strip()
        if store_messages:
            self._message_history.append(message)
            self._message_history.append({"role": "assistant", "content": response_text, "tool_calls": tool_calls})
        if self.log_manager:
            self.log_manager.log_debug(f"Ollama generated response_text=\n{response_text}\ntool_calls={tool_calls}")
        return response_text, tool_calls, False

    async def generate(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        use_tools: bool = False,
        store_messages: bool = False,
    ) -> tuple[str, list[ToolCall]]:
        text, tool_calls, _ = await self._stream_chat(
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            use_tools=use_tools,
            store_messages=store_messages,
        )
        return text, tool_calls

    async def generate_json(
        self,
        model: str,
        prompt: str,
        fallback: dict[str, Any],
        image_paths: list[str] | None = None,
        use_tools: bool = False,
        store_messages: bool = False,
    ) -> tuple[dict[str, Any], list[ToolCall]]:
        text, tool_calls = await self.generate(
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            use_tools=use_tools,
            store_messages=store_messages,
        )
        try:
            return json.loads(text), tool_calls
        except json.JSONDecodeError:
            return fallback, tool_calls

    def clear_message_history(self) -> None:
        self._message_history = []


async def generate_brain_decision(
    model_name: str,
    prompt: str,
    ollama_host: str,
    image_paths: list[str] | None = None,
    interrupt_checker: Callable[[], bool] | None = None,
) -> str:
    """Stream a brain decision and allow mid-generation interruption."""
    client = AsyncClient(host=ollama_host)
    try:
        chunks: list[str] = []
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_paths:
            message["images"] = image_paths
        stream = await client.chat(
            model=model_name,
            messages=[message],
            stream=True,
        )
        async for part in stream:
            if interrupt_checker and interrupt_checker():
                print("Interrupt received! Abandoning current thought...")
                return "INTERRUPTED"
            chunk_content = part.get("message", {}).get("content", "")
            if chunk_content:
                print(chunk_content, end="", flush=True)
                chunks.append(chunk_content)
        if chunks:
            print()
        return "".join(chunks).strip()
    except Exception as e:
        print(f"Brain Error: {e}")
        return ""
