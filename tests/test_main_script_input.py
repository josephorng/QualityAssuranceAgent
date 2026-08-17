import builtins
from pathlib import Path

from src.common import script_helper


def test_parse_executable_lines_from_text_matches_file(tmp_path: Path) -> None:
    script = tmp_path / "sample.txt"
    script.write_text(
        "\n# comment line\nopen chrome\n   \n# another comment\nsearch cats\n",
        encoding="utf-8",
    )
    from_disk = script_helper.parse_script_lines(script)
    from_text = script_helper.parse_executable_lines_from_text(script.read_text(encoding="utf-8"))
    assert from_disk == from_text == ["open chrome", "search cats"]


def test_executable_source_line_numbers_skips_blank_and_comments() -> None:
    raw = "\n# comment line\nopen chrome\n   \n# another comment\nsearch cats\n"
    assert script_helper.executable_source_line_numbers(raw) == [3, 6]


def test_parse_script_lines_skips_blank_and_comments(tmp_path: Path) -> None:
    script = tmp_path / "sample.txt"
    script.write_text(
        "\n# comment line\nopen chrome\n   \n# another comment\nsearch cats\n",
        encoding="utf-8",
    )
    assert script_helper.parse_script_lines(script) == ["open chrome", "search cats"]


def test_parse_script_steps_with_outcomes_reads_expected_outcome_comments() -> None:
    raw = (
        "點擊「確定」\n"
        "# expected_outcome: 對話框已關閉\n"
        "等待 2 秒\n"
        "輸入「hello」\n"
        "# expected_outcome: 輸入欄顯示 hello\n"
    )
    steps, outcomes = script_helper.parse_script_steps_with_outcomes(raw)
    assert steps == ["點擊「確定」", "等待 2 秒", "輸入「hello」"]
    assert outcomes == ["對話框已關閉", None, "輸入欄顯示 hello"]


def test_format_script_lines_with_outcomes_round_trip() -> None:
    lines = script_helper.format_script_lines_with_outcomes(
        ["點擊「確定」", "等待 2 秒"],
        ["對話框已關閉", None],
    )
    assert lines == [
        "點擊「確定」",
        "# expected_outcome: 對話框已關閉",
        "等待 2 秒",
    ]
    steps, outcomes = script_helper.parse_script_steps_with_outcomes("\n".join(lines))
    assert steps == ["點擊「確定」", "等待 2 秒"]
    assert outcomes == ["對話框已關閉", None]


def test_recording_script_helpers(tmp_path: Path) -> None:
    run_dir = tmp_path / "開啟神網"
    run_dir.mkdir()
    (run_dir / "session.json").write_text(
        '{"run_id": "開啟神網", "events": ["events/event_001.json"]}',
        encoding="utf-8",
    )
    events = run_dir / "events"
    analysis = run_dir / "analysis"
    events.mkdir()
    analysis.mkdir()
    (events / "event_001.json").write_text(
        '{"index": 1, "kind": "click"}',
        encoding="utf-8",
    )
    (analysis / "event_001.json").write_text(
        '{"event_index": 1, "wait_instruction": "等待 2 秒",'
        ' "instruction": "點擊「確定」", "expected_outcome": "對話框已關閉"}',
        encoding="utf-8",
    )
    text = script_helper.collect_recording_script_text(run_dir)
    assert text == "等待 2 秒\n點擊「確定」\n# expected_outcome: 對話框已關閉\n"
    assert script_helper.is_recording_dir(run_dir)
    assert script_helper.is_recording_script_path(run_dir)
    assert script_helper.resolve_runnable_script_path(run_dir) == run_dir
    assert script_helper.script_display_name(run_dir) == "開啟神網"
    assert script_helper.load_runnable_script_text(run_dir) == text
    legacy = run_dir / "script.txt"
    legacy.write_text("stale\n", encoding="utf-8")
    assert script_helper.resolve_runnable_script_path(legacy) == run_dir
    assert script_helper.load_runnable_script_text(legacy) == text
    plain = tmp_path / "scripts" / "plain.txt"
    plain.parent.mkdir()
    plain.write_text("step\n", encoding="utf-8")
    assert not script_helper.is_recording_script_path(plain)
    assert script_helper.resolve_runnable_script_path(plain) == plain
    assert script_helper.script_display_name(plain) == "plain.txt"


def test_partition_recording_dirs_keeps_valid_and_skips_duplicates(tmp_path: Path) -> None:
    rec_a = tmp_path / "rec_a"
    rec_b = tmp_path / "rec_b"
    rec_a.mkdir()
    rec_b.mkdir()
    (rec_a / "session.json").write_text("{}", encoding="utf-8")
    (rec_b / "session.json").write_text("{}", encoding="utf-8")
    other = tmp_path / "not_a_recording"
    other.mkdir()
    added, invalid = script_helper.partition_recording_dirs(
        [rec_a, other, rec_a, rec_b],
        existing=[rec_a],
    )
    assert added == [rec_b]
    assert invalid == [other]


def test_resolve_task_and_script_from_cli_task(monkeypatch, tmp_path: Path) -> None:
    task, script_path, lines = script_helper.resolve_task_and_script("typed task", tmp_path)
    assert task == "typed task"
    assert lines == ["typed task"]
    assert script_path.parent == tmp_path / "scripts"
    assert script_path.exists()


def test_resolve_task_and_script_selects_script(monkeypatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    chosen = scripts_dir / "chosen.txt"
    chosen.write_text("# comment\nstep one\n\nstep two\n", encoding="utf-8")
    answers = iter(["2", "1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    task, script_path, lines = script_helper.resolve_task_and_script(None, tmp_path)
    assert task == "step one"
    assert script_path == chosen
    assert lines == ["step one", "step two"]
