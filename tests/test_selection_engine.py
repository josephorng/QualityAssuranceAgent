from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cua_mcp.selection_engine import request_json_with_retry


@pytest.mark.asyncio
async def test_request_json_with_retry_uses_brain_lm_from_settings() -> None:
    reply = MagicMock()
    reply.content = '{"instruction": "按下 Enter 鍵。"}'
    chat_messages = AsyncMock(return_value=reply)
    client = MagicMock()
    client.chat_messages = chat_messages

    with patch("cua_mcp.selection_engine.get_llm_client", return_value=client), patch(
        "cua_mcp.selection_engine.load_settings",
        return_value=MagicMock(brain_lm="google/gemma-4-26B-A4B-it"),
    ):
        result = await request_json_with_retry(
            messages=[{"role": "user", "content": "test"}],
            response_schema={"type": "object", "properties": {"instruction": {"type": "string"}}},
            parse_reply=lambda raw: __import__("json").loads(raw),
            retry_instruction="retry",
        )

    assert result == {"instruction": "按下 Enter 鍵。"}
    chat_messages.assert_awaited_once()
    assert chat_messages.await_args.args[0] == "google/gemma-4-26B-A4B-it"


@pytest.mark.asyncio
async def test_request_json_with_retry_retries_once_on_timeout() -> None:
    import httpx

    reply = MagicMock()
    reply.content = '{"instruction": "ok"}'
    chat_messages = AsyncMock(
        side_effect=[httpx.ReadTimeout("timed out"), reply],
    )
    client = MagicMock()
    client.chat_messages = chat_messages
    logs: list[str] = []

    with patch("cua_mcp.selection_engine.get_llm_client", return_value=client), patch(
        "cua_mcp.selection_engine.load_settings",
        return_value=MagicMock(brain_lm="google/gemma-4-26B-A4B-it"),
    ):
        result = await request_json_with_retry(
            messages=[{"role": "user", "content": "test"}],
            response_schema={"type": "object"},
            parse_reply=lambda raw: __import__("json").loads(raw),
            retry_instruction="retry",
            log_info=logs.append,
        )

    assert result == {"instruction": "ok"}
    assert chat_messages.await_count == 2
    assert any("transport retry" in line for line in logs)


@pytest.mark.asyncio
async def test_request_json_with_retry_raises_after_transport_retries_exhausted() -> None:
    import httpx

    chat_messages = AsyncMock(side_effect=httpx.ReadTimeout("still timed out"))
    client = MagicMock()
    client.chat_messages = chat_messages

    with patch("cua_mcp.selection_engine.get_llm_client", return_value=client), patch(
        "cua_mcp.selection_engine.load_settings",
        return_value=MagicMock(brain_lm="google/gemma-4-26B-A4B-it"),
    ):
        with pytest.raises(httpx.ReadTimeout):
            await request_json_with_retry(
                messages=[{"role": "user", "content": "test"}],
                response_schema={"type": "object"},
                parse_reply=lambda raw: __import__("json").loads(raw),
                retry_instruction="retry",
            )

    assert chat_messages.await_count == 2
