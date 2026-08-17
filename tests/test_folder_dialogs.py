from pathlib import Path

from src.common import folder_dialogs


def test_ask_directories_uses_windows_multi_select_result(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "rec_a"
    second = tmp_path / "rec_b"
    first.mkdir()
    second.mkdir()

    def fake_windows(**_kwargs: object) -> list[Path]:
        return [first, second]

    monkeypatch.setattr(folder_dialogs, "_ask_directories_windows", fake_windows)
    assert folder_dialogs.ask_directories(title="選擇錄製資料夾") == [first, second]


def test_ask_directories_fallback_when_windows_dialog_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    chosen = tmp_path / "rec_a"
    chosen.mkdir()
    monkeypatch.setattr(folder_dialogs, "_ask_directories_windows", lambda **_kwargs: None)
    monkeypatch.setattr(
        "tkinter.filedialog.askdirectory",
        lambda **_kwargs: str(chosen),
    )
    assert folder_dialogs.ask_directories() == [chosen]


def test_ask_directories_cancel_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(folder_dialogs, "_ask_directories_windows", lambda **_kwargs: [])
    assert folder_dialogs.ask_directories() == []
