from pathlib import Path

from src.common.io_utils import pop_last_nonempty_line


def test_pop_last_nonempty_line_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "commands.txt"
    path.write_text("", encoding="utf-8")
    assert pop_last_nonempty_line(path) is None
    assert path.read_text(encoding="utf-8") == ""


def test_pop_last_nonempty_line_single_line(tmp_path: Path) -> None:
    path = tmp_path / "commands.txt"
    path.write_text("only line\n", encoding="utf-8")
    assert pop_last_nonempty_line(path) == "only line"
    assert path.read_text(encoding="utf-8") == ""


def test_pop_last_nonempty_line_trailing_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "commands.txt"
    path.write_text("first\nsecond\n\n", encoding="utf-8")
    assert pop_last_nonempty_line(path) == "second"
    assert path.read_text(encoding="utf-8") == "first\n"


def test_pop_last_nonempty_line_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    assert pop_last_nonempty_line(path) is None
