from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ollama import AsyncClient


class OllamaClient:
    def __init__(self, host: str, timeout_seconds: int = 60) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = AsyncClient(host=self.host)

    async def _stream_chat(
        self,
        model: str,
        prompt: str,
        image_paths: list[str] | None = None,
        interrupt_checker: Callable[[], bool] | None = None,
    ) -> tuple[str, bool]:
        chunks: list[str] = []
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_paths:
            message["images"] = image_paths
        stream = await self.client.chat(
            model=model,
            messages=[message],
            stream=True,
            options={"num_ctx": 4096},
        )
        async for part in stream:
            if interrupt_checker and interrupt_checker():
                return "INTERRUPTED", True
            chunk_content = part.get("message", {}).get("content", "")
            if chunk_content:
                print(chunk_content, end="", flush=True)
                chunks.append(chunk_content)
        if chunks:
            print()
        return "".join(chunks).strip(), False

    async def generate(self, model: str, prompt: str, image_paths: list[str] | None = None) -> str:
        text, _ = await self._stream_chat(model=model, prompt=prompt, image_paths=image_paths)
        return text

    async def generate_json(
        self,
        model: str,
        prompt: str,
        fallback: dict[str, Any],
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        text = await self.generate(model=model, prompt=prompt, image_paths=image_paths)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return fallback


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
