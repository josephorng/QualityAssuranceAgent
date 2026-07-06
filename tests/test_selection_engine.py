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
