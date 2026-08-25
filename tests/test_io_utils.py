from pathlib import Path

from PIL import Image

from src.common.io_utils import imread_bgr, imwrite_bgr, pop_last_nonempty_line


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


def test_imread_bgr_unicode_path(tmp_path: Path) -> None:
    folder = tmp_path / "中文目录"
    folder.mkdir()
    path = folder / "截图.png"
    Image.new("RGB", (6, 4), (255, 0, 0)).save(path)

    bgr = imread_bgr(path)
    assert bgr is not None
    assert bgr.shape == (4, 6, 3)
    # Pillow RGB red becomes OpenCV BGR.
    assert int(bgr[0, 0, 2]) == 255
    assert int(bgr[0, 0, 0]) == 0


def test_imread_bgr_missing_file(tmp_path: Path) -> None:
    assert imread_bgr(tmp_path / "missing.png") is None


def test_imwrite_bgr_unicode_path(tmp_path: Path) -> None:
    import numpy as np

    folder = tmp_path / "中文目录"
    folder.mkdir()
    path = folder / "輸出.png"
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)

    assert imwrite_bgr(path, image) is True
    assert path.is_file()
    bgr = imread_bgr(path)
    assert bgr is not None
    assert bgr.shape == (4, 6, 3)
    assert int(bgr[0, 0, 2]) == 255
    assert int(bgr[0, 0, 0]) == 0
