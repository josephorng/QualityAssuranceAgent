from __future__ import annotations

from typing import Any, Callable, TypeVar

from src.common.llm_factory import get_llm_client
from src.common.settings import load_settings

T = TypeVar("T")


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
    """
    try:
        reply = await get_llm_client().chat_messages(
            load_settings().brain_lm,
            messages=messages,
            tools=[],
            response_format=response_schema,
            think=think,
            append_image_sizes=append_image_sizes,
        )
        return parse_reply(reply.content)
    except ValueError as exc:
        if log_info is not None:
            log_info(f"request_json_with_retry: retry ({exc})")
        messages[0]["content"] += f"\n{retry_instruction}\n"
        retry = await get_llm_client().chat_messages(
            load_settings().brain_lm,
            messages=messages,
            tools=[],
            response_format="json",
            append_image_sizes=append_image_sizes,
        )
        return parse_reply(retry.content)
