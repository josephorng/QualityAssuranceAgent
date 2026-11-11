"""Unit tests for recording frame similarity helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import numpy as np

from src.recorder.frame_similarity import (
    bgr_mean_abs_diff,
    compare_frames,
    frames_similar,
    last_settle_frame_path,
    mean_abs_diff,
)


def _write_gray(path: Path, value: int, size: tuple[int, int] = (320, 180)) -> None:
    Image.new("L", size, color=int(value)).save(path, format="JPEG", quality=95)


def test_mean_abs_diff_identical(tmp_path: Path) -> None:
    a = tmp_path / "a.jpeg"
    b = tmp_path / "b.jpeg"
    _write_gray(a, 128)
    _write_gray(b, 128)
    assert mean_abs_diff(a, b) < 0.01
    assert frames_similar(a, b) is True
    similar, mad = compare_frames(a, b)
    assert similar is True
    assert mad is not None and mad < 0.01


def test_mean_abs_diff_different(tmp_path: Path) -> None:
    a = tmp_path / "a.jpeg"
    b = tmp_path / "b.jpeg"
    _write_gray(a, 0)
    _write_gray(b, 255)
    assert mean_abs_diff(a, b) > 0.5
    assert frames_similar(a, b) is False
    similar, mad = compare_frames(a, b)
    assert similar is False
    assert mad is not None and mad > 0.5


def test_bgr_mean_abs_diff_identical() -> None:
    frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    assert bgr_mean_abs_diff(frame, frame.copy()) < 0.01


def test_bgr_mean_abs_diff_different() -> None:
    a = np.zeros((180, 320, 3), dtype=np.uint8)
    b = np.full((180, 320, 3), 255, dtype=np.uint8)
    assert bgr_mean_abs_diff(a, b) > 0.5


def test_last_settle_frame_path(tmp_path: Path) -> None:
    assert last_settle_frame_path(tmp_path) == (
        tmp_path / "screenshots" / "_last_settle_frame.jpeg"
    )
