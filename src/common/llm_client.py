from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Literal, Union

from ollama import Message
from PIL import Image
from pydantic import BaseModel

ResponseFormatParam = Literal["json"] | dict[str, Any] | None
ThinkParam = Union[bool, Literal["low", "medium", "high"], None]


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class LLMClient(abc.ABC):
    """
    Abstract interface for chat-style LLM backends used across the project.

    Implementations accept Ollama-style message dicts (``{"role", "content", "images"}``)
    and return an ``ollama.Message`` so existing callers can stay unchanged when a
    different backend is swapped in.
    """

    @abc.abstractmethod
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
        Run a chat completion with an explicit message list.

        Args:
            model: Backend model identifier.
            messages: Ollama-style message dicts. Image references go in ``images``.
            tools: Tool definitions for the model. ``None`` lets the implementation
                fall back to its default tool set; an empty list disables tools.
            response_format: ``"json"`` or a JSON schema dict, when supported.
            append_image_sizes: Append ``ImageSizes:`` annotation to the last user
                message when images are present (helps coordinate-aware prompts).
            think: Optional ``think`` hint forwarded to thinking-capable backends.

        Returns:
            Message: Assistant message in ``ollama.Message`` shape.
        """
        ...

    @staticmethod
    def _append_last_message_image_sizes(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append ``ImageSizes: name=WxH`` to the trailing message when it has images."""
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
