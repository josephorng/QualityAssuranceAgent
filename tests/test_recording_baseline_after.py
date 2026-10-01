from __future__ import annotations

import json
import os
from pathlib import Path

from main import prepare_run_session
from src.common.runtime_context import (
    RUNTIME_COMMAND_MODE_ENV,
    SCRIPT_BASELINE_AFTER_ENV,
    SCRIPT_LINES_ENV,
    SCRIPT_OUTCOMES_ENV,
    SCRIPT_PATH_ENV,
    SCRIPT_SETTLE_AFTER_ENV,
    SMART_GOAL_ENV,
    SMART_MODE_ENV,
)
from src.common.script_helper import (
    collect_recording_baseline_after_paths,
    collect_recording_instruction_event_indices,
    collect_recording_instructions,
    collect_recording_settle_after_seconds,
    collect_recording_window_verifies,
)


def _write_recording(tmp_path: Path) -> Path:
    run_dir = tmp_path / "rec_demo"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir()
    (run_dir / "screenshots").mkdir()

    shot0 = run_dir / "screenshots" / "event_000.jpeg"
    shot1 = run_dir / "screenshots" / "event_001.jpeg"
    final_after = run_dir / "screenshots" / "final_after.jpeg"
    shot0.write_bytes(b"before0")
    shot1.write_bytes(b"before1")
    final_after.write_bytes(b"final")

    events = []
    timestamps = (
        "2026-09-07T00:00:00+00:00",
        "2026-09-07T00:00:09+00:00",
    )
    for index, (shot, ts) in enumerate(zip((shot0, shot1), timestamps)):
        rel = f"events/event_{index:03d}.json"
        events.append(rel)
        (run_dir / rel).write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": ts,
                    "kind": "click",
                    "screenshot_path": str(shot),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        analysis = {
            "instruction": f"click step {index}",
            "expected_outcome": None,
            # Verification (and thus baseline) is opt-in per step.
            "use_expected_outcome": True,
        }
        if index == 0:
            # Leftover virtual wait must be ignored by collectors.
            analysis["wait_instruction"] = "等待 1 秒"
        (run_dir / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps(analysis, ensure_ascii=False),
            encoding="utf-8",
        )

    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": "rec_demo",
                "started_at_utc": "2026-09-07T00:00:00+00:00",
                "stopped_at_utc": "2026-09-07T00:00:17+00:00",
                "events": events,
                "final_after_screenshot": str(final_after),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_collect_recording_baseline_after_paths_ignores_virtual_wait(tmp_path: Path) -> None:
    run_dir = _write_recording(tmp_path)
    instructions, outcomes = collect_recording_instructions(run_dir)
    baselines = collect_recording_baseline_after_paths(run_dir)
    settles = collect_recording_settle_after_seconds(run_dir)

    assert instructions == ["click step 0", "click step 1"]
    assert outcomes == [None, None]
    assert len(baselines) == len(instructions)
    assert baselines[0] is not None
    assert baselines[0].endswith("event_001.jpeg")
    assert baselines[1] is not None
    assert baselines[1].endswith("final_after.jpeg")
    assert settles == [9.0, None]


def test_instruction_event_indices_skip_events_without_instruction(tmp_path: Path) -> None:
    run_dir = _write_recording(tmp_path)
    (run_dir / "events" / "event_002.json").write_text(
        json.dumps(
            {
                "index": 2,
                "timestamp_utc": "2026-09-07T00:00:10+00:00",
                "kind": "click",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "events" / "event_003.json").write_text(
        json.dumps(
            {
                "index": 3,
                "timestamp_utc": "2026-09-07T00:00:12+00:00",
                "kind": "click",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "analysis" / "event_003.json").write_text(
        json.dumps(
            {
                "instruction": "click step 3",
                "expected_outcome": None,
                "use_expected_outcome": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "events" / "event_004.json").write_text(
        json.dumps(
            {
                "index": 4,
                "timestamp_utc": "2026-09-07T00:00:13+00:00",
                "kind": "click",
                "screenshot_path": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "analysis" / "event_004.json").write_text(
        json.dumps({"instruction": "   "}, ensure_ascii=False),
        encoding="utf-8",
    )
    session_path = run_dir / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["events"] = [
        "events/event_000.json",
        "events/event_002.json",
        "events/event_001.json",
        "events/event_004.json",
        "events/event_003.json",
    ]
    session_path.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")

    instructions, _outcomes = collect_recording_instructions(run_dir)
    indices = collect_recording_instruction_event_indices(run_dir)
    assert instructions == ["click step 0", "click step 1", "click step 3"]
    assert indices == [0, 1, 3]
    assert len(indices) == len(instructions)


def test_collect_recording_settle_prefers_analysis_settle_after(tmp_path: Path) -> None:
    run_dir = _write_recording(tmp_path)
    first = run_dir / "analysis" / "event_000.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["settle_after_seconds"] = 2.5
    first.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    settles = collect_recording_settle_after_seconds(run_dir)
    assert settles[0] == 2.5
    assert settles[1] is None


def test_collect_recording_settle_keeps_zero_observed(tmp_path: Path) -> None:
    run_dir = _write_recording(tmp_path)
    event0 = run_dir / "events" / "event_000.json"
    payload = json.loads(event0.read_text(encoding="utf-8"))
    payload["observed_settle_seconds"] = 0.0
    event0.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    settles = collect_recording_settle_after_seconds(run_dir)
    assert settles[0] == 0.0
    assert settles[1] is None


def test_collect_recording_settle_keeps_zero_analysis_settle(tmp_path: Path) -> None:
    run_dir = _write_recording(tmp_path)
    first = run_dir / "analysis" / "event_000.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["settle_after_seconds"] = 0.0
    first.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    settles = collect_recording_settle_after_seconds(run_dir)
    assert settles[0] == 0.0
    assert settles[1] is None


def test_collect_recording_settle_prefers_observed_over_timestamp_gap(
    tmp_path: Path,
) -> None:
    run_dir = _write_recording(tmp_path)
    event0 = run_dir / "events" / "event_000.json"
    payload = json.loads(event0.read_text(encoding="utf-8"))
    payload["observed_settle_seconds"] = 1.5
    event0.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    settles = collect_recording_settle_after_seconds(run_dir)
    assert settles[0] == 1.5
    assert settles[1] is None


def test_collect_recording_settle_analysis_wins_over_observed(tmp_path: Path) -> None:
    run_dir = _write_recording(tmp_path)
    event0 = run_dir / "events" / "event_000.json"
    event_payload = json.loads(event0.read_text(encoding="utf-8"))
    event_payload["observed_settle_seconds"] = 1.5
    event0.write_text(json.dumps(event_payload, ensure_ascii=False), encoding="utf-8")
    first = run_dir / "analysis" / "event_000.json"
    analysis_payload = json.loads(first.read_text(encoding="utf-8"))
    analysis_payload["settle_after_seconds"] = 2.25
    first.write_text(json.dumps(analysis_payload, ensure_ascii=False), encoding="utf-8")

    settles = collect_recording_settle_after_seconds(run_dir)
    assert settles[0] == 2.25


def test_collect_recording_baseline_after_paths_skips_disabled_verification(
    tmp_path: Path,
) -> None:
    run_dir = _write_recording(tmp_path)
    first = run_dir / "analysis" / "event_000.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["use_expected_outcome"] = False
    first.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    baselines = collect_recording_baseline_after_paths(run_dir)
    assert baselines[0] is None
    assert baselines[1] is not None
    assert baselines[1].endswith("final_after.jpeg")


def test_collect_recording_baseline_after_paths_survives_renamed_folder(tmp_path: Path) -> None:
    """Absolute screenshot paths from a pre-rename folder still resolve by basename."""
    run_dir = _write_recording(tmp_path)
    old_root = tmp_path / "recording_old_id"
    for event_path in (run_dir / "events").glob("event_*.json"):
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        name = Path(payload["screenshot_path"]).name
        payload["screenshot_path"] = str(old_root / "screenshots" / name)
        event_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    session_path = run_dir / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["final_after_screenshot"] = str(old_root / "screenshots" / "final_after.jpeg")
    session_path.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")

    assert not old_root.exists()
    baselines = collect_recording_baseline_after_paths(run_dir)
    assert baselines[0] is not None and baselines[0].endswith("event_001.jpeg")
    assert Path(baselines[0]).is_file()
    assert baselines[1] is not None and baselines[1].endswith("final_after.jpeg")
    assert Path(baselines[1]).is_file()


def test_prepare_run_session_seeds_baseline_and_settle_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_COMMAND_MODE_ENV, raising=False)
    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    monkeypatch.delenv(SMART_GOAL_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_PATH_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_LINES_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_OUTCOMES_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_BASELINE_AFTER_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_SETTLE_AFTER_ENV, raising=False)

    run_dir = _write_recording(tmp_path)
    instructions, _ = collect_recording_instructions(run_dir)
    prepare_run_session(
        runs_root=tmp_path / "runs",
        task=instructions[0],
        runtime_mode=False,
        selected_script_path=run_dir,
        script_steps=instructions,
        eye_monitor_indices=[1],
        clear_runs_root=False,
    )

    baselines = json.loads(os.environ[SCRIPT_BASELINE_AFTER_ENV])
    settles = json.loads(os.environ[SCRIPT_SETTLE_AFTER_ENV])
    assert len(baselines) == len(instructions)
    assert isinstance(baselines[0], str) and baselines[0].endswith("event_001.jpeg")
    assert isinstance(baselines[1], str) and baselines[1].endswith("final_after.jpeg")
    assert settles == [9.0, None]


def test_prepare_run_session_drops_baselines_when_script_edited(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(SCRIPT_BASELINE_AFTER_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_SETTLE_AFTER_ENV, raising=False)
    run_dir = _write_recording(tmp_path)
    # Edited hub script: fewer lines than collected baselines
    prepare_run_session(
        runs_root=tmp_path / "runs2",
        task="only one",
        runtime_mode=False,
        selected_script_path=run_dir,
        script_steps=["only one edited line"],
        eye_monitor_indices=[1],
        clear_runs_root=False,
    )
    baselines = json.loads(os.environ[SCRIPT_BASELINE_AFTER_ENV])
    settles = json.loads(os.environ[SCRIPT_SETTLE_AFTER_ENV])
    assert baselines == [None]
    assert settles == [None]


def test_collect_recording_window_verifies_skips_unchecked(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec_verify"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir()
    (run_dir / "events" / "event_001.json").write_text(
        json.dumps(
            {
                "index": 1,
                "timestamp_utc": "2026-09-07T00:00:00+00:00",
                "kind": "click",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "analysis" / "event_001.json").write_text(
        json.dumps(
            {
                "instruction": "點擊「搜尋」",
                "window_verify": {
                    "disappeared": [
                        {"title": "快顯主機", "class_name": "Popup", "process_name": "explorer.exe"},
                        {"title": "檔案總管", "class_name": "CabinetWClass", "process_name": "explorer.exe"},
                    ],
                    "clipboard": "copied",
                },
                "window_verify_disabled": ["disappeared:0", "clipboard"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    verifies = collect_recording_window_verifies(run_dir)

    assert verifies == [
        {
            "disappeared": [
                {"title": "檔案總管", "class_name": "CabinetWClass", "process_name": "explorer.exe"}
            ]
        }
    ]


def test_collect_recording_window_verifies_drops_unrelated_focused_value(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec_focus"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir()
    (run_dir / "events" / "event_006.json").write_text(
        json.dumps(
            {
                "index": 6,
                "timestamp_utc": "2026-10-01T06:34:00+00:00",
                "kind": "text_input",
                "text": "nbanba live",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "analysis" / "event_006.json").write_text(
        json.dumps(
            {
                "instruction": "輸入「nbanba live」",
                "window_verify": {"focused": {"value": "nba l"}},
                "tool_calls": [
                    {
                        "name": "type_text",
                        "arguments": {"text": "nbanba live", "instruction": "輸入「nbanba live」"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert collect_recording_window_verifies(run_dir) == [{}]
