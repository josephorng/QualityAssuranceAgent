from __future__ import annotations

from pathlib import Path

from src.common.run_state import (
    parse_step_artifact_path,
    parse_step_artifact_stem,
    step_artifact_name,
    step_artifact_stem,
    step_filenames_are_one_based,
)


def test_step_artifact_name_is_one_based_on_disk() -> None:
    assert step_artifact_stem(0, 0) == "1_1"
    assert step_artifact_name(0, 0) == "1_1.json"
    assert step_artifact_name(14, 14, suffix=".log") == "15_15.log"


def test_parse_one_based_stem() -> None:
    assert parse_step_artifact_stem("1_1", one_based=True) == (0, 0)
    assert parse_step_artifact_stem("15_15", one_based=True) == (14, 14)
    assert parse_step_artifact_stem("0_0", one_based=True) is None


def test_parse_legacy_zero_based_stem() -> None:
    assert parse_step_artifact_stem("0_0", one_based=False) == (0, 0)
    assert parse_step_artifact_stem("14_14", one_based=False) == (14, 14)


def test_folder_detection_distinguishes_legacy_and_new(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "0_0.json").write_text("{}", encoding="utf-8")
    (legacy / "1_1.json").write_text("{}", encoding="utf-8")
    assert step_filenames_are_one_based(legacy) is False
    assert parse_step_artifact_path(legacy / "1_1.json") == (1, 1)

    modern = tmp_path / "modern"
    modern.mkdir()
    (modern / "1_1.json").write_text("{}", encoding="utf-8")
    (modern / "2_2.json").write_text("{}", encoding="utf-8")
    assert step_filenames_are_one_based(modern) is True
    assert parse_step_artifact_path(modern / "1_1.json") == (0, 0)
    assert parse_step_artifact_path(modern / "2_2.json") == (1, 1)
