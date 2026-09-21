"""Unit tests for recording settle frame similarity helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.recorder.frame_similarity import (
    append_settle_probe_debug_record,
    apply_settle_sample,
    archive_settle_probe_sample,
    compare_frames,
    frames_similar,
    mean_abs_diff,
    settle_probe_debug_dir,
    settle_probe_debug_log_path,
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


def test_archive_settle_probe_sample_and_debug_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    source = tmp_path / "src.jpeg"
    _write_gray(source, 80)

    archived = archive_settle_probe_sample(
        run_dir=run_dir,
        event_index=2,
        sample_index=1,
        source=source,
        age_s=1.016,
        tag="sample",
    )
    assert archived is not None
    assert archived.is_file()
    assert archived.parent == settle_probe_debug_dir(run_dir, 2)
    assert "age1p016s" in archived.name

    append_settle_probe_debug_record(
        run_dir,
        2,
        {
            "sample": 1,
            "age_s": 1.016,
            "kind": "first_keep",
            "path": archived.name,
        },
    )
    append_settle_probe_debug_record(
        run_dir,
        2,
        {
            "sample": 2,
            "age_s": 2.1,
            "kind": "compare",
            "mad": 0.03983,
            "threshold": 0.02,
            "similar": False,
        },
    )
    log_path = settle_probe_debug_log_path(run_dir, 2)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"mad": 0.03983' in lines[1]