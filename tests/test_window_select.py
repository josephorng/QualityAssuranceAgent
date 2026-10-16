from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cua_mcp import hand_tools


class _Window:
    def __init__(self, title: str, process_name: str | None = None) -> None:
        self.title = title
        self.process_name = process_name
        self.isMinimized = False
        self.closed = False
        self.minimized = False
        self.maximized = False

    def activate(self) -> None:
        return None

    def restore(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def minimize(self) -> None:
        self.minimized = True
        self.isMinimized = True

    def maximize(self) -> None:
        self.maximized = True


def _llm(content: str) -> MagicMock:
    client = MagicMock()
    client.chat_messages = AsyncMock(return_value=SimpleNamespace(content=content))
    return client


def _install_windows(monkeypatch: pytest.MonkeyPatch, windows: list[_Window]) -> None:
    monkeypatch.setattr(hand_tools.gw, "getAllWindows", lambda: windows)
    monkeypatch.setattr(
        hand_tools, "load_settings", lambda: SimpleNamespace(brain_lm="test-model")
    )
    monkeypatch.setattr(
        hand_tools,
        "_process_name_for_window",
        lambda w: getattr(w, "process_name", None),
    )


@pytest.mark.asyncio
async def test_close_windows_uses_unique_title_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _Window("常用 - 檔案總管")
    other = _Window("Git - 檔案總管")
    _install_windows(monkeypatch, [other, target])
    client = _llm('{"indices": [0]}')
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: client)

    result = await hand_tools.close_windows("常用 - 檔案總管")

    assert result["status"] == "success"
    assert result["selection_mode"] == "substring_unique"
    assert target.closed is True
    assert other.closed is False
    client.chat_messages.assert_not_called()


@pytest.mark.asyncio
async def test_close_windows_fails_when_selector_returns_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_explorer = _Window("Git - 檔案總管")
    cursor = _Window("main.py - Cursor")
    _install_windows(monkeypatch, [git_explorer, cursor])
    client = _llm('{"indices": []}')
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: client)

    with pytest.raises(ValueError, match="no window matched '常用 - 檔案總管'"):
        await hand_tools.close_windows(
            "常用 - 檔案總管",
            instruction="關閉「常用 - 檔案總管」視窗",
        )

    assert git_explorer.closed is False
    assert cursor.closed is False
    prompt = client.chat_messages.await_args.kwargs["messages"][0]["content"]
    assert "return an empty list" in prompt
    assert "Do not substitute a different window" in prompt


@pytest.mark.asyncio
async def test_close_windows_keeps_natural_language_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome = _Window("Google Chrome - Gmail")
    notepad = _Window("Untitled - Notepad")
    _install_windows(monkeypatch, [chrome, notepad])
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: _llm('{"indices": [0]}'))

    result = await hand_tools.close_windows("the browser")

    assert result["status"] == "success"
    assert result["selection_mode"] == "ollama_no_substring_match"
    assert result["targets"] == [
        {"matched_title": "Google Chrome - Gmail", "status": "closed"}
    ]
    assert chrome.closed is True
    assert notepad.closed is False


@pytest.mark.asyncio
async def test_close_windows_fails_when_disambiguation_returns_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Window("Report - Notepad")
    second = _Window("Notes - Notepad")
    _install_windows(monkeypatch, [first, second])
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: _llm("```json\n{\"indices\": []}\n```"))

    with pytest.raises(ValueError, match="no window matched 'Notepad'"):
        await hand_tools.minimize_windows("Notepad")

    assert first.minimized is False
    assert second.minimized is False


@pytest.mark.asyncio
async def test_unique_title_ignores_process_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _Window("資料日期_20260821.json - 記事本", "Notepad.exe")
    other = _Window("Git - 檔案總管", "explorer.exe")
    _install_windows(monkeypatch, [other, target])
    client = _llm('{"indices": [0]}')
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: client)

    result = await hand_tools.maximize_windows(
        "資料日期_20260821.json - 記事本",
        process_name="chrome.exe",
    )

    assert result["status"] == "success"
    assert result["selection_mode"] == "substring_unique"
    assert result["process_name"] == "chrome.exe"
    assert target.maximized is True
    client.chat_messages.assert_not_called()


@pytest.mark.asyncio
async def test_process_unique_when_title_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notepad = _Window("Untitled - Notepad", "Notepad.exe")
    chrome = _Window("Google Chrome", "chrome.exe")
    _install_windows(monkeypatch, [chrome, notepad])
    client = _llm('{"indices": [0]}')
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: client)

    result = await hand_tools.close_windows(
        "資料日期_20260821.json - 記事本",
        process_name="Notepad.exe",
    )

    assert result["status"] == "success"
    assert result["selection_mode"] == "process_unique"
    assert notepad.closed is True
    assert chrome.closed is False
    client.chat_messages.assert_not_called()


@pytest.mark.asyncio
async def test_substring_process_unique_narrows_many_titles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _Window("Report - Notepad", "Notepad.exe")
    notes = _Window("Notes - Notepad", "WINWORD.EXE")
    _install_windows(monkeypatch, [report, notes])
    client = _llm('{"indices": [1]}')
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: client)

    result = await hand_tools.minimize_windows(
        "Notepad",
        process_name="notepad.exe",
    )

    assert result["status"] == "success"
    assert result["selection_mode"] == "substring_process_unique"
    assert report.minimized is True
    assert notes.minimized is False
    client.chat_messages.assert_not_called()


@pytest.mark.asyncio
async def test_process_many_empty_llm_pick_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Window("Report - Notepad", "Notepad.exe")
    second = _Window("Notes - Notepad", "Notepad.exe")
    _install_windows(monkeypatch, [first, second])
    client = _llm('{"indices": []}')
    monkeypatch.setattr(hand_tools, "get_llm_client", lambda: client)

    with pytest.raises(ValueError, match="no window matched 'missing title'"):
        await hand_tools.close_windows(
            "missing title",
            process_name="Notepad.exe",
        )

    assert first.closed is False
    assert second.closed is False
    prompt = client.chat_messages.await_args.kwargs["messages"][0]["content"]
    assert "Notepad.exe" in prompt
