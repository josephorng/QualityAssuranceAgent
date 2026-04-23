from __future__ import annotations

import asyncio
import json
import re
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

    def _extract_tool_call(self, raw_call: Any) -> ToolCall | None:
        # Ollama tool call payloads may be dict-like or object-like, and
        # function metadata can be top-level or nested under `function`.
        name: str | None = None
        arguments: Any = {}

        if isinstance(raw_call, dict):
            if isinstance(raw_call.get("function"), dict):
                function = raw_call["function"]
                name = function.get("name")
                arguments = function.get("arguments", {})
            else:
                name = raw_call.get("name")
                arguments = raw_call.get("arguments", {})
        else:
            function = getattr(raw_call, "function", None)
            if function is not None:
                name = getattr(function, "name", None)
                arguments = getattr(function, "arguments", {})
            else:
                name = getattr(raw_call, "name", None)
                arguments = getattr(raw_call, "arguments", {})

        if not isinstance(name, str) or not name.strip():
            return None

        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
            arguments = parsed_arguments

        if not isinstance(arguments, dict):
            arguments = {}

        return ToolCall(name=name, arguments=arguments)

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

    async def _chat(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        store_messages: bool = False,
        use_tools: bool = True,
    ) -> tuple[str, list[ToolCall]]:
        get_run_state_manager().log_info(
            f"Ollama generating chat for model={model} prompt=\n{prompt}\n"
            f"image_paths={image_paths} store_messages={store_messages} use_tools={use_tools}"
        )
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_paths:
            message["images"] = image_paths
        request_messages = [*self._message_history, message] if store_messages else [message]
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        }
        if use_tools:
            chat_kwargs["tools"] = get_ollama_tools()
        response = await self.client.chat(
            **chat_kwargs,
        )
        get_run_state_manager().log_info(f"Ollama generated response=\n{response}")
        response_message = response.get("message", {})
        response_text = response_message.get("content", "").strip()
        raw_tool_calls_value = response_message.get("tool_calls", [])
        raw_tool_calls: list[Any] = raw_tool_calls_value if isinstance(raw_tool_calls_value, list) else []
        tool_calls: list[ToolCall] = []
        for raw_call in raw_tool_calls:
            parsed_call = self._extract_tool_call(raw_call)
            if parsed_call is not None:
                tool_calls.append(parsed_call)
        if not response_text and not tool_calls:
            get_run_state_manager().log_info("Ollama returned empty response with tools; retrying in 5 seconds.")
            await asyncio.sleep(5)
            return await self._chat(
                model=model,
                prompt=prompt,
                image_paths=image_paths,
                store_messages=store_messages,
                use_tools=use_tools,
            )
        if store_messages:
            self._message_history.append(message)
            self._message_history.append({"role": "assistant", "content": response_text, "tool_calls": raw_tool_calls})        
        return response_text, tool_calls

    async def generate(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        use_tools: bool = False,
        store_messages: bool = False,
    ) -> tuple[str, list[ToolCall]]:
        return await self._chat(
            model=model,
            prompt=prompt,
            image_paths=image_paths,
            store_messages=store_messages,
            use_tools=use_tools,
        )

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
        normalized_text = text.strip()
        if normalized_text.startswith("```"):
            # Accept common LLM formatting: fenced JSON blocks.
            fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", normalized_text, flags=re.DOTALL)
            if fenced_match:
                normalized_text = fenced_match.group(1).strip()
        try:
            return json.loads(normalized_text), tool_calls
        except json.JSONDecodeError:
            get_run_state_manager().log_info(f"Ollama returned invalid JSON: {text}")
            return fallback, tool_calls

    def clear_message_history(self) -> None:
        self._message_history = []

