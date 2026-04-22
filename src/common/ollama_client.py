from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from ollama import AsyncClient

from cua_mcp.tools import get_ollama_tools
from pydantic import BaseModel
from src.common.run_state import get_run_state_manager

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]

class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int = 60) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = AsyncClient(host=self.host)
        self._message_history: list[dict[str, Any]] = []

    async def _stream_chat(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        store_messages: bool = False,
        interrupt_checker: Callable[[], bool] | None = None,
    ) -> tuple[str, bool]:
        chunks: list[str] = []
        
        get_run_state_manager().log_info(
            f"Ollama generating for model={model} prompt=\n{prompt}\nimage_paths={image_paths} store_messages={store_messages}"
        )
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
        stream = await self.client.chat(**chat_kwargs)
        async for part in stream:
            if interrupt_checker and interrupt_checker():
                return "INTERRUPTED", True
            chunk_content = part.get("message", {}).get("content", "")
            if chunk_content:
                print(chunk_content, end="", flush=True)
                chunks.append(chunk_content)
        if chunks:
            print()
        response_text = "".join(chunks).strip()
        if not response_text:
            get_run_state_manager().log_info("Ollama returned empty response; retrying in 5 seconds.")
            await asyncio.sleep(5)
            return await self._stream_chat(
                model=model,
                prompt=prompt,
                image_paths=image_paths,
                store_messages=store_messages,
                interrupt_checker=interrupt_checker,
            )
        if store_messages:
            self._message_history.append(message)
            self._message_history.append({"role": "assistant", "content": response_text})
        get_run_state_manager().log_info(f"Ollama generated response_text=\n{response_text}")
        return response_text, False

    async def _chat_with_tool_calls(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        store_messages: bool = False,
    ) -> tuple[str, list[ToolCall]]:
        get_run_state_manager().log_info(
            f"Ollama generating with tools for model={model} prompt=\n{prompt}\nimage_paths={image_paths} store_messages={store_messages}"
        )
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_paths:
            message["images"] = image_paths
        request_messages = [*self._message_history, message] if store_messages else [message]
        response = await self.client.chat(
            model=model,
            messages=request_messages,
            stream=False,
            options={"num_ctx": 4096},
            tools=get_ollama_tools(),
        )
        response_message = response.get("message", {})
        response_text = response_message.get("content", "").strip()
        print(f"response_message={response_message}")
        raw_tool_calls = response_message.get("tool_calls", [])
        tool_calls = [ToolCall(name=call["name"], arguments=call["arguments"]) for call in raw_tool_calls]
        print(f"tool_calls={tool_calls}")
        if not response_text and not tool_calls:
            get_run_state_manager().log_info("Ollama returned empty response with tools; retrying in 5 seconds.")
            await asyncio.sleep(5)
            return await self._chat_with_tool_calls(
                model=model,
                prompt=prompt,
                image_paths=image_paths,
                store_messages=store_messages,
            )
        if store_messages:
            self._message_history.append(message)
            self._message_history.append({"role": "assistant", "content": response_text, "tool_calls": raw_tool_calls})
        get_run_state_manager().log_info(f"Ollama generated response_text=\n{response_text}\ntool_calls={tool_calls}")
        return response_text, tool_calls

    async def generate(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        use_tools: bool = False,
        store_messages: bool = False,
    ) -> tuple[str, list[ToolCall]]:
        if use_tools:
            return await self._chat_with_tool_calls(
                model=model,
                prompt=prompt,
                image_paths=image_paths,
                store_messages=store_messages,
            )
        text, _ = await self._stream_chat(
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            store_messages=store_messages,
        )
        return text, []

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
