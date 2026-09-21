"""Unit tests for recording settle frame similarity helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.recorder.frame_similarity import (
    apply_settle_sample,
    compare_frames,
    frames_similar,
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


def test_apply_settle_sample_first_keeps_staging(tmp_path: Path) -> None:
    staging = tmp_path / "staging.jpeg"
    _write_gray(staging, 40)
    kept, age, observed = apply_settle_sample(
        kept_path=None,
        kept_age_s=None,
        staging_path=staging,
        sample_age_s=0.5,
    )
    assert kept == staging
    assert age == 0.5
    assert observed is None
    assert staging.is_file()


def test_apply_settle_sample_different_promotes_staging(tmp_path: Path) -> None:
    kept = tmp_path / "kept.jpeg"
    staging = tmp_path / "staging.jpeg"
    _write_gray(kept, 0)
    _write_gray(staging, 255)
    new_kept, age, observed = apply_settle_sample(
        kept_path=kept,
        kept_age_s=0.5,
        staging_path=staging,
        sample_age_s=1.0,
    )
    assert new_kept == staging
    assert age == 1.0
    assert observed is None
    assert not kept.is_file()
    assert staging.is_file()


def test_apply_settle_sample_similar_returns_observed(tmp_path: Path) -> None:
    kept = tmp_path / "kept.jpeg"
    staging = tmp_path / "staging.jpeg"
    _write_gray(kept, 100)
    _write_gray(staging, 100)
    new_kept, age, observed = apply_settle_sample(
        kept_path=kept,
        kept_age_s=1.0,
        staging_path=staging,
        sample_age_s=1.5,
    )
    assert new_kept == staging
    assert age == 1.5
    assert observed == 1.0
    assert staging.is_file()
    assert not kept.is_file()
