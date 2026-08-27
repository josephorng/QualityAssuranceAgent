from __future__ import annotations

from typing import Any, Callable, TypeVar

import httpx

from src.common.llm_factory import get_llm_client
from src.common.settings import load_settings

T = TypeVar("T")

# Transient transport failures: retry once, then let callers soft-fail.
_TRANSPORT_RETRY_ERRORS = (
    httpx.TimeoutException,
    httpx.TransportError,
    TimeoutError,
    OSError,
)


async def request_json_with_retry(
    *,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    parse_reply: Callable[[str], T],
    retry_instruction: str,
    log_info: Callable[[str], None] | None = None,
    think: bool = False,
    append_image_sizes: bool = True,
) -> T:
    """
    Run a structured LLM call and retry once with response_format='json' on parse failures.

    Also retries once on transport timeouts / connection errors before re-raising.
    """

    async def _chat(response_format: Any) -> Any:
        return await get_llm_client().chat_messages(
            load_settings().brain_lm,
            messages=messages,
            tools=[],
            response_format=response_format,
            think=think,
            append_image_sizes=append_image_sizes,
        )

    async def _chat_with_transport_retry(response_format: Any) -> Any:
        try:
            return await _chat(response_format)
        except _TRANSPORT_RETRY_ERRORS as exc:
            if log_info is not None:
                log_info(
                    f"request_json_with_retry: transport retry "
                    f"({type(exc).__name__}: {exc})"
                )
            return await _chat(response_format)

    try:
        reply = await _chat_with_transport_retry(response_schema)
        return parse_reply(reply.content)
    except ValueError as exc:
        if log_info is not None:
            log_info(f"request_json_with_retry: retry ({exc})")
        messages[0]["content"] += f"\n{retry_instruction}\n"
        retry = await _chat_with_transport_retry("json")
        return parse_reply(retry.content)
