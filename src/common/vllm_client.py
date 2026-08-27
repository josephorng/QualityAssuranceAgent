from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import httpx
from ollama import Message
from ollama._utils import convert_function_to_tool

from src.common.llm_client import LLMClient, ResponseFormatParam, ThinkParam
from src.common.run_state import get_run_state_manager

# Ollama exposes an OpenAI-compatible API at /v1/chat/completions on port 11434.
OLLAMA_OPENAI_COMPAT_URL = "http://192.168.13.101:11434"

__all__ = [
    "VLLMClient",
    "OLLAMA_OPENAI_COMPAT_URL",
    "_encode_image_data_url",
    "_translate_messages_to_openai",
]


def _normalize_tool_descriptor(tool: Any) -> dict[str, Any]:
    """
    Convert any of the accepted tool shapes (Python callable, ollama ``Tool`` object,
    or pre-built OpenAI-style dict) into the OpenAI ``/v1/chat/completions``
    ``tools[]`` schema: ``{"type": "function", "function": {...}}``.
    """
    if callable(tool):
        tool_obj = convert_function_to_tool(tool)
        return tool_obj.model_dump(exclude_none=True)
    if hasattr(tool, "model_dump"):
        return tool.model_dump(exclude_none=True)
    if isinstance(tool, dict):
        return tool
    raise TypeError(f"Unsupported tool descriptor type: {type(tool).__name__}")


def _coerce_arguments_to_json_string(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {}, ensure_ascii=False)


_CALL_SYNTAX_PATTERN = re.compile(
    r"call:(?P<name>[A-Za-z_][\w]*)\{(?P<args>[^}]*)\}"
)


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_call_syntax_list(value: str) -> list[Any] | None:
    """Parse a call-syntax list value into a Python list, or None if not a list."""
    if not (value.startswith("[") and value.endswith("]")):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return parsed

    inner = value[1:-1].strip()
    if not inner:
        return []

    items: list[str] = []
    for item in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", inner):
        cleaned = _strip_wrapping_quotes(item.strip())
        if cleaned:
            items.append(cleaned)
    return items


def _parse_call_syntax_value(value: str) -> Any:
    """Coerce a raw call-syntax value; lists become real lists for Pydantic tools."""
    parsed_list = _parse_call_syntax_list(value)
    if parsed_list is not None:
        return parsed_list
    return value


def _parse_call_syntax_arguments(args_str: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    if not args_str.strip():
        return arguments
    for part in re.split(r",(?=\w+:)", args_str):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        arguments[key.strip()] = _parse_call_syntax_value(value.strip())
    return arguments


def _parse_call_syntax_tool_calls(content: str) -> list[Message.ToolCall]:
    """
    Parse vLLM text tool invocations like
    ``call:minimize_windows{instruction:...,window_title_contains:all}``.
    """
    tool_calls: list[Message.ToolCall] = []
    for match in _CALL_SYNTAX_PATTERN.finditer(content):
        arguments = _parse_call_syntax_arguments(match.group("args"))
        tool_calls.append(
            Message.ToolCall(
                function=Message.ToolCall.Function(
                    name=match.group("name"),
                    arguments=arguments,
                )
            )
        )
    return tool_calls


def _coerce_arguments_to_dict(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw": arguments}
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    return {}


def _encode_image_data_url(path: str | Path) -> str:
    """Read an image file and return a ``data:<mime>;base64,...`` URL."""
    image_path = Path(path)
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime or not mime.startswith("image/"):
        suffix = image_path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _content_with_images(text: Any, image_paths: list[Any] | None) -> Any:
    """Build OpenAI multimodal content when image paths are present."""
    paths = [str(p) for p in (image_paths or []) if p]
    if not paths:
        return text if text is not None else ""
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": text if isinstance(text, str) else ("" if text is None else str(text)),
        }
    ]
    for path in paths:
        try:
            url = _encode_image_data_url(path)
        except OSError:
            continue
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _keep_latest_message_images(
    messages: list[dict[str, Any]],
    *,
    max_images: int = 2,
) -> list[dict[str, Any]]:
    """
    Return message copies with images retained only on the newest image-bearing message.

    vLLM counts images across the complete prompt and this deployment accepts at
    most two. Older screenshots are stale after subsequent UI actions, so preserve
    their text/tool history while removing their ``images`` fields.
    """
    prepared = [dict(message) for message in messages]
    newest_image_index: int | None = None
    for index in range(len(prepared) - 1, -1, -1):
        images = prepared[index].get("images")
        if isinstance(images, list) and any(images):
            newest_image_index = index
            break

    for index, message in enumerate(prepared):
        if index != newest_image_index:
            message.pop("images", None)
            continue
        images = message.get("images")
        if isinstance(images, list):
            kept = [path for path in images if path][:max(0, max_images)]
            if kept:
                message["images"] = kept
            else:
                message.pop("images", None)
    return prepared


def _translate_messages_to_openai(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Translate Ollama-style messages into OpenAI ``/v1/chat/completions`` ``messages[]``.

    When a message includes ``images: [path, ...]``, those files are embedded as
    OpenAI multimodal ``image_url`` parts (data URIs).

    Tool calls in assistant messages are given stable ids (``call_<i>_<j>``);
    immediately following ``tool`` messages are paired with those ids in order,
    since the existing callers append tool replies right after the assistant
    message that produced them.
    """
    out: list[dict[str, Any]] = []
    pending_tool_call_ids: list[str] = []

    for msg_idx, raw in enumerate(messages):
        msg = dict(raw)
        role = msg.get("role")

        if role == "tool":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": msg.get("content", ""),
            }
            if pending_tool_call_ids:
                tool_msg["tool_call_id"] = pending_tool_call_ids.pop(0)
            else:
                tool_msg["tool_call_id"] = f"call_{msg_idx}_orphan"
            tool_name = msg.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                tool_msg["name"] = tool_name
            out.append(tool_msg)
            continue

        content_value = msg.get("content")
        images = msg.get("images")
        openai_msg: dict[str, Any] = {
            "role": role or "user",
            "content": _content_with_images(
                content_value, images if isinstance(images, list) else None
            ),
        }

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                converted: list[dict[str, Any]] = []
                for call_idx, tc in enumerate(tool_calls):
                    function = (tc or {}).get("function") if isinstance(tc, dict) else None
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name", "")
                    arguments = function.get("arguments", {})
                    call_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        and isinstance(tc.get("id"), str)
                        and tc.get("id")
                        else f"call_{msg_idx}_{call_idx}"
                    )
                    converted.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": _coerce_arguments_to_json_string(arguments),
                            },
                        }
                    )
                    pending_tool_call_ids.append(call_id)
                if converted:
                    openai_msg["tool_calls"] = converted
                    if not openai_msg.get("content"):
                        openai_msg["content"] = None

        out.append(openai_msg)

    return out


def _translate_openai_message_to_ollama(message: dict[str, Any]) -> Message:
    """Convert an OpenAI ``choices[0].message`` dict back into an ``ollama.Message``."""
    role = message.get("role", "assistant") or "assistant"
    content = message.get("content")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        content = "".join(text_parts)
    if content is None:
        content = ""

    thinking = message.get("reasoning_content") or message.get("reasoning") or None

    tool_calls_payload: list[Message.ToolCall] | None = None
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list) and raw_tool_calls:
        converted: list[Message.ToolCall] = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function") or {}
            name = function.get("name", "")
            arguments = _coerce_arguments_to_dict(function.get("arguments", {}))
            converted.append(
                Message.ToolCall(
                    function=Message.ToolCall.Function(name=name, arguments=arguments)
                )
            )
        if converted:
            tool_calls_payload = converted

    if not tool_calls_payload and isinstance(content, str):
        parsed_calls = _parse_call_syntax_tool_calls(content)
        if parsed_calls:
            tool_calls_payload = parsed_calls
            content = _CALL_SYNTAX_PATTERN.sub("", content).strip() or None

    return Message(
        role=role,
        content=content if content else None,
        thinking=thinking,
        tool_calls=tool_calls_payload,
    )


def _translate_response_format(response_format: ResponseFormatParam) -> dict[str, Any] | None:
    """Convert the project-wide response_format shorthand into OpenAI's schema."""
    if response_format is None:
        return None
    if response_format == "json":
        return {"type": "json_object"}
    if isinstance(response_format, dict):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "Response",
                "schema": response_format,
                "strict": False,
            },
        }
    raise ValueError(f"Unsupported response_format: {response_format!r}")


class VLLMClient(LLMClient):
    """
    LLM client for OpenAI-compatible chat completion APIs.

    Configured for **Ollama** at ``OLLAMA_OPENAI_COMPAT_URL`` (OpenAI-compatible
    routes). Use model name ``gemma4:26b`` in callers (e.g. ``brain_lm`` in
    ``runs/agent_settings.json``).

    The wire format is translated to/from ``ollama.Message`` so the rest of the
    codebase can keep using the existing message shape and tool descriptors.
    """

    def __init__(
        self,
        host: str | None = None,
        timeout_seconds: int = 120,
        *,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.host = (host or OLLAMA_OPENAI_COMPAT_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._endpoint = f"{self.host}/v1/chat/completions"

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
        Run a chat completion against the OpenAI-compatible server.

        Incoming ``images`` paths are embedded as multimodal ``image_url`` parts.
        ``append_image_sizes`` and ``think`` are accepted for interface compatibility
        but ignored.
        """
        prepared_messages = _keep_latest_message_images(messages)
        openai_messages = _translate_messages_to_openai(prepared_messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        translated_format = _translate_response_format(response_format)
        if translated_format is not None:
            payload["response_format"] = translated_format

        if tools:
            payload["tools"] = [_normalize_tool_descriptor(tool) for tool in tools]
            # vLLM defaults to tool_choice=auto, which 400s unless the server was
            # started with --enable-auto-tool-choice. "none" still exposes tools to
            # the model; replies use call:name{...} text we parse below.
            payload["tool_choice"] = "none"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Log without embedding base64 image payloads.
        log_messages = [
            {k: ("<images omitted>" if k == "images" else v) for k, v in msg.items()}
            for msg in prepared_messages
        ]
        last_assistant_idx = -1
        for idx in reversed(range(len(log_messages))):
            if log_messages[idx].get("role") == "assistant":
                last_assistant_idx = idx
                break
        get_run_state_manager().log_info(
            f"VLLM chat_messages for model={model} n_messages={len(prepared_messages)} "
            f"tools_count={len(tools) if tools else 0} "
            f"response_format_set={response_format is not None} "
            f"endpoint={self._endpoint} "
            f"last_assistant_messages=\n{log_messages[last_assistant_idx:]}"
            f"headers=\n{headers}"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self._endpoint, headers=headers, json=payload
                )
        except httpx.TimeoutException as exc:
            get_run_state_manager().log_info(
                f"VLLM chat_messages timeout after {self.timeout_seconds}s "
                f"endpoint={self._endpoint}: {exc}"
            )
            raise
        except httpx.HTTPError as exc:
            get_run_state_manager().log_info(
                f"VLLM chat_messages transport error endpoint={self._endpoint}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        if response.is_error:
            get_run_state_manager().log_info(
                f"VLLM chat_messages error status={response.status_code} body={response.text}"
            )
        response.raise_for_status()
        body = response.json()
        get_run_state_manager().log_info(f"VLLM chat_messages response=\n{body}")

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("VLLM response contained no choices")
        message_dict = choices[0].get("message") or {}
        response_message = _translate_openai_message_to_ollama(message_dict)

        has_content = bool((response_message.content or "").strip())
        has_tool_calls = bool(response_message.tool_calls)
        has_thinking = bool((response_message.thinking or "").strip())
        if not has_content and not has_tool_calls and not has_thinking:
            get_run_state_manager().log_info(
                "VLLM returned empty response and no tools; retrying in 5 seconds."
            )
            await asyncio.sleep(5)
            return await self.chat_messages(
                model=model,
                messages=messages,
                tools=tools,
                response_format=response_format,
                think=think,
            )
        return response_message
