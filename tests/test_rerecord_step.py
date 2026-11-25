"""Mid-replay multi-action re-record: splice capture over event N and reanalyze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.common.io_utils import read_json
from src.common.run_control import take_recording_replay_dirty
from src.common.script_helper import (
    invalidate_recording_replay_cache,
    load_recording_replay_steps,
    merge_unstarted_recording_steps,
    recording_event_json_paths,
)
from src.recorder.rerecord_step import (
    RERECORDABLE_KINDS,
    EmptyRerecordError,
    event_kind_is_rerecordable,
    overwrite_pointer_event,
    purge_rerecord_analysis_artifacts,
    reanalyze_recording_event,
    replace_and_reanalyze_pointer_event,
    splice_and_reanalyze_rerecord_capture,
    splice_rerecord_events,
)


def _write_recording(run_dir: Path, *, event_count: int = 2) -> None:
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "screenshots").mkdir(exist_ok=True)
    (run_dir / "yolo_ocr").mkdir(exist_ok=True)
    (run_dir / "signals").mkdir(exist_ok=True)
    events: list[str] = []
    for index in range(1, event_count + 1):
        rel = f"events/event_{index:03d}.json"
        events.append(rel)
        shot = f"screenshots/event_{index:03d}.jpeg"
        (run_dir / shot).write_bytes(b"jpeg")
        (run_dir / rel).write_text(
            json.dumps(
                {
                    "index": index,
                    "timestamp_utc": f"2026-09-07T00:00:0{index}+00:00",
                    "kind": "click",
                    "button": "left",
                    "click_count": 1,
                    "cursor_xy": [10 * index, 20 * index],
                    "screenshot_path": shot,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "analysis" / f"event_{index:03d}.json").write_text(
            json.dumps(
                {
                    "event_index": index,
                    "instruction": f"click {index}",
                    "expected_outcome": f"result {index}",
                    "use_expected_outcome": True,
                    "window_verify": {"appeared": [{"title": "A"}]},
                    "window_verify_disabled": ["appeared:0"],
                    "tool_calls": [{"name": "click", "arguments": {}}],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "yolo_ocr" / f"event_{index:03d}.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (run_dir / "signals" / f"event_{index:03d}_before.json").write_text(
            "{}",
            encoding="utf-8",
        )
    (run_dir / "session.json").write_text(
        json.dumps({"run_id": run_dir.name, "events": events, "event_count": event_count}),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "instructions": [f"click {i}" for i in range(1, event_count + 1)],
                "expected_outcomes": [f"result {i}" for i in range(1, event_count + 1)],
            }
        ),
        encoding="utf-8",
    )


def test_event_kind_is_rerecordable() -> None:
    assert RERECORDABLE_KINDS == {
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "hold",
        "drag",
        "scroll",
        "text_input",
        "key_press",
        "hotkey",
    }
    assert event_kind_is_rerecordable("click")
    assert event_kind_is_rerecordable("drag")
    assert event_kind_is_rerecordable("text_input")
    assert event_kind_is_rerecordable("hotkey")
    assert not event_kind_is_rerecordable("wait")
    assert not event_kind_is_rerecordable(None)


def _write_temp_capture(temp_dir: Path, *, kinds: list[str]) -> None:
    (temp_dir / "events").mkdir(parents=True, exist_ok=True)
    (temp_dir / "screenshots").mkdir(exist_ok=True)
    (temp_dir / "yolo_ocr").mkdir(exist_ok=True)
    events: list[str] = []
    for index, kind in enumerate(kinds, start=1):
        rel = f"events/event_{index:03d}.json"
        events.append(rel)
        shot = f"screenshots/event_{index:03d}.jpeg"
        (temp_dir / shot).write_bytes(f"temp-{index}".encode("utf-8"))
        payload: dict[str, Any] = {
            "index": index,
            "timestamp_utc": f"2026-11-18T00:00:0{index}+00:00",
            "kind": kind,
            "screenshot_path": shot,
        }
        if kind in {"click", "double_click", "right_click", "middle_click", "hold", "drag"}:
            payload["button"] = "left" if kind != "right_click" else "right"
            payload["cursor_xy"] = [100 * index, 200 * index]
            if kind == "drag":
                payload["end_xy"] = [100 * index + 40, 200 * index + 80]
                end_shot = f"screenshots/event_{index:03d}_end.jpeg"
                (temp_dir / end_shot).write_bytes(b"end")
                payload["end_screenshot_path"] = end_shot
            if kind == "click":
                payload["click_count"] = 1
            if kind == "double_click":
                payload["click_count"] = 2
        elif kind == "scroll":
            payload["cursor_xy"] = [50, 50]
            payload["scroll_delta"] = -3
        elif kind == "text_input":
            payload["text"] = f"hello{index}"
            payload["cursor_xy"] = [10, 10]
        elif kind == "key_press":
            payload["key"] = "enter"
        elif kind == "hotkey":
            payload["keys"] = ["ctrl", "s"]
        (temp_dir / rel).write_text(json.dumps(payload), encoding="utf-8")
    (temp_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": temp_dir.name,
                "events": events,
                "event_count": len(events),
            }
        ),
        encoding="utf-8",
    )


def test_splice_replaces_one_event_with_many(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=3)
    temp_dir = tmp_path / "tmp_capture"
    _write_temp_capture(temp_dir, kinds=["drag", "click", "scroll"])

    new_indices = splice_rerecord_events(run_dir, 2, temp_dir)

    assert new_indices == [4, 5, 6]
    session = read_json(run_dir / "session.json", {})
    assert session["events"] == [
        "events/event_001.json",
        "events/event_004.json",
        "events/event_005.json",
        "events/event_006.json",
        "events/event_003.json",
    ]
    assert not (run_dir / "events" / "event_002.json").is_file()
    assert not (run_dir / "analysis" / "event_002.json").is_file()
    drag = read_json(run_dir / "events" / "event_004.json", {})
    assert drag["kind"] == "drag"
    assert drag["index"] == 4
    assert drag["end_xy"] == [140, 280]
    assert (run_dir / "screenshots" / "event_004_end.jpeg").is_file()
    assert (run_dir / "events" / "event_001.json").is_file()
    assert (run_dir / "events" / "event_003.json").is_file()
    ordered = recording_event_json_paths(run_dir)
    assert [p.name for p in ordered] == [
        "event_001.json",
        "event_004.json",
        "event_005.json",
        "event_006.json",
        "event_003.json",
    ]


def test_splice_insert_after_keeps_anchor_and_adds_many(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=3)
    temp_dir = tmp_path / "tmp_capture"
    _write_temp_capture(temp_dir, kinds=["drag", "click"])

    new_indices = splice_rerecord_events(
        run_dir, 2, temp_dir, mode="insert_after"
    )

    assert new_indices == [4, 5]
    session = read_json(run_dir / "session.json", {})
    assert session["events"] == [
        "events/event_001.json",
        "events/event_002.json",
        "events/event_004.json",
        "events/event_005.json",
        "events/event_003.json",
    ]
    assert (run_dir / "events" / "event_002.json").is_file()
    assert (run_dir / "analysis" / "event_002.json").is_file()
    drag = read_json(run_dir / "events" / "event_004.json", {})
    assert drag["kind"] == "drag"
    assert drag["index"] == 4
    ordered = recording_event_json_paths(run_dir)
    assert [p.name for p in ordered] == [
        "event_001.json",
        "event_002.json",
        "event_004.json",
        "event_005.json",
        "event_003.json",
    ]


def test_splice_empty_capture_keeps_original(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=2)
    temp_dir = tmp_path / "tmp_capture"
    _write_temp_capture(temp_dir, kinds=[])

    with pytest.raises(EmptyRerecordError):
        splice_rerecord_events(run_dir, 2, temp_dir)

    assert (run_dir / "events" / "event_002.json").is_file()
    session = read_json(run_dir / "session.json", {})
    assert session["events"] == [
        "events/event_001.json",
        "events/event_002.json",
    ]

    with pytest.raises(EmptyRerecordError):
        splice_rerecord_events(run_dir, 2, temp_dir, mode="insert_after")
    assert session["events"] == [
        "events/event_001.json",
        "events/event_002.json",
    ]


def test_splice_and_reanalyze_preserves_outcome_on_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=2)
    temp_dir = tmp_path / "tmp_capture"
    _write_temp_capture(temp_dir, kinds=["click", "drag"])

    async def fake_reanalyze(
        run_dir,
        event_index,
        *,
        log_info=None,
        preserve_from_analysis=None,
        rebuild_caches=True,
    ):
        analysis_path = Path(run_dir) / "analysis" / f"event_{event_index:03d}.json"
        payload = {
            "event_index": event_index,
            "instruction": f"new {event_index}",
            "use_expected_outcome": False,
        }
        if isinstance(preserve_from_analysis, dict):
            outcome = preserve_from_analysis.get("expected_outcome")
            if isinstance(outcome, str) and outcome.strip():
                payload["expected_outcome"] = outcome.strip()
                payload["use_expected_outcome"] = True
            disabled = preserve_from_analysis.get("window_verify_disabled")
            if isinstance(disabled, list):
                payload["window_verify_disabled"] = list(disabled)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "event_index": event_index,
            "instruction": payload["instruction"],
            "kind": "click" if event_index == 3 else "drag",
        }

    monkeypatch.setattr(
        "src.recorder.rerecord_step.reanalyze_recording_event",
        fake_reanalyze,
    )
    monkeypatch.setattr(
        "src.recorder.compile_tool_calls.rebuild_recording_instruction_tool_cache",
        lambda _run_dir: _run_dir / "instruction_tool_cache.json",
    )
    monkeypatch.setattr(
        "src.common.session_html.write_recording_html_from_run",
        lambda run_dir, update_index=True: run_dir / "recording_steps.html",
    )
    take_recording_replay_dirty()

    result = splice_and_reanalyze_rerecord_capture(run_dir, 2, temp_dir)
    assert result["mode"] == "replace"
    assert result["event_indices"] == [3, 4]
    assert result["instructions"] == ["new 3", "new 4"]
    first = read_json(run_dir / "analysis" / "event_003.json", {})
    last = read_json(run_dir / "analysis" / "event_004.json", {})
    assert "expected_outcome" not in first or first.get("expected_outcome") in (None, "")
    assert last["expected_outcome"] == "result 2"
    assert last["window_verify_disabled"] == ["appeared:0"]
    assert take_recording_replay_dirty() is True


def test_splice_and_reanalyze_insert_after_does_not_preserve_anchor_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=2)
    temp_dir = tmp_path / "tmp_capture"
    _write_temp_capture(temp_dir, kinds=["click", "drag"])

    async def fake_reanalyze(
        run_dir,
        event_index,
        *,
        log_info=None,
        preserve_from_analysis=None,
        rebuild_caches=True,
    ):
        analysis_path = Path(run_dir) / "analysis" / f"event_{event_index:03d}.json"
        payload = {
            "event_index": event_index,
            "instruction": f"inserted {event_index}",
            "use_expected_outcome": False,
        }
        if isinstance(preserve_from_analysis, dict):
            outcome = preserve_from_analysis.get("expected_outcome")
            if isinstance(outcome, str) and outcome.strip():
                payload["expected_outcome"] = outcome.strip()
                payload["use_expected_outcome"] = True
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "event_index": event_index,
            "instruction": payload["instruction"],
            "kind": "click",
        }

    monkeypatch.setattr(
        "src.recorder.rerecord_step.reanalyze_recording_event",
        fake_reanalyze,
    )
    monkeypatch.setattr(
        "src.recorder.compile_tool_calls.rebuild_recording_instruction_tool_cache",
        lambda _run_dir: _run_dir / "instruction_tool_cache.json",
    )
    monkeypatch.setattr(
        "src.common.session_html.write_recording_html_from_run",
        lambda run_dir, update_index=True: run_dir / "recording_steps.html",
    )

    result = splice_and_reanalyze_rerecord_capture(
        run_dir, 2, temp_dir, mode="insert_after"
    )
    assert result["mode"] == "insert_after"
    assert result["replaced_event_index"] is None
    assert result["anchor_event_index"] == 2
    assert result["event_indices"] == [3, 4]
    assert (run_dir / "events" / "event_002.json").is_file()
    last = read_json(run_dir / "analysis" / "event_004.json", {})
    assert "expected_outcome" not in last or last.get("expected_outcome") in (None, "")
    session = read_json(run_dir / "session.json", {})
    assert session["events"] == [
        "events/event_001.json",
        "events/event_002.json",
        "events/event_003.json",
        "events/event_004.json",
    ]


def test_purge_removes_yolo_and_signals_keeps_analysis(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)
    end_shot = run_dir / "screenshots" / "event_002_end.jpeg"
    end_shot.write_bytes(b"end")

    purge_rerecord_analysis_artifacts(run_dir, 2)

    assert (run_dir / "analysis" / "event_002.json").is_file()
    assert (run_dir / "events" / "event_002.json").is_file()
    assert (run_dir / "screenshots" / "event_002.jpeg").is_file()
    assert not (run_dir / "yolo_ocr" / "event_002.json").is_file()
    assert not (run_dir / "signals" / "event_002_before.json").is_file()
    assert not end_shot.is_file()
    assert (run_dir / "yolo_ocr" / "event_001.json").is_file()


def test_overwrite_keeps_event_index_and_session_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)
    monkeypatch.setattr(
        "src.recorder.rerecord_step._WINDOW_AFTER_SETTLE_S",
        0.0,
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.snapshot_top_level_windows",
        lambda: [],
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_step_signals",
        lambda **_kwargs: {"foreground": {"title": "after"}},
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_fast_signals",
        lambda **_kwargs: {},
    )

    shot = run_dir / "screenshots" / "event_002.jpeg"
    shot.write_bytes(b"new-jpeg")
    event = overwrite_pointer_event(
        run_dir,
        2,
        kind="double_click",
        cursor_xy=(100, 200),
        button="left",
        timestamp_utc="2026-10-30T00:00:00+00:00",
        screenshot_path=str(shot),
        monitor_index=1,
        monitor_offset=(0, 0),
        click_window=None,
        windows_before=(),
        signals_before={"foreground": {"title": "before"}},
        click_count=2,
    )

    assert event.index == 2
    assert event.kind == "double_click"
    raw = read_json(run_dir / "events" / "event_002.json", {})
    assert raw["index"] == 2
    assert raw["kind"] == "double_click"
    assert raw["cursor_xy"] == [100, 200]
    assert raw["click_count"] == 2
    session = read_json(run_dir / "session.json", {})
    assert session["events"] == [
        "events/event_001.json",
        "events/event_002.json",
    ]
    assert (run_dir / "events" / "event_001.json").is_file()
    assert not (run_dir / "events" / "event_003.json").is_file()


@pytest.mark.asyncio
async def test_reanalyze_rewrites_instruction_and_preserves_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)

    async def fake_prepare(event, *, run_dir, log_info):
        from src.recorder.orchestrator import _PreparedEvent

        return _PreparedEvent(
            event=event,
            event_for_llm=event,
            vision={
                "used_vision": True,
                "candidate_text": "儲存",
                "candidates": [],
                "local_cursor": [10, 20],
            },
            text_resolution=None,
            vision_seconds=0.01,
            vision_cache_hit=False,
        )

    async def fake_analyze(event, *, run_dir, vision=None, log_info=None):
        return {"instruction": "點擊「儲存」"}

    monkeypatch.setattr(
        "src.recorder.orchestrator.prepare_event_vision",
        fake_prepare,
    )
    monkeypatch.setattr(
        "src.recorder.analyze.analyze_event_to_cache",
        fake_analyze,
    )
    monkeypatch.setattr(
        "src.recorder.compile_tool_calls.compile_tool_calls",
        lambda *_args, **_kwargs: [{"name": "click", "arguments": {"text": "儲存"}}],
    )
    monkeypatch.setattr(
        "src.recorder.compile_tool_calls.rebuild_recording_instruction_tool_cache",
        lambda _run_dir: _run_dir / "instruction_tool_cache.json",
    )
    monkeypatch.setattr(
        "src.common.session_html.write_recording_html_from_run",
        lambda run_dir, update_index=True: run_dir / "recording_steps.html",
    )
    monkeypatch.setattr(
        "src.recorder.orchestrator._bind_run_state_for_analysis",
        lambda *_args, **_kwargs: None,
    )
    take_recording_replay_dirty()

    result = await reanalyze_recording_event(run_dir, 2)
    assert result["event_index"] == 2
    assert result["instruction"] == "點擊「儲存」"

    analysis = read_json(run_dir / "analysis" / "event_002.json", {})
    assert analysis["instruction"] == "點擊「儲存」"
    assert analysis["expected_outcome"] == "result 2"
    assert analysis["use_expected_outcome"] is True
    assert analysis["window_verify_disabled"] == ["appeared:0"]
    assert "settle_after_seconds" not in analysis
    previous = read_json(run_dir / "analysis" / "event_001.json", {})
    assert "settle_after_seconds" not in previous

    report = read_json(run_dir / "report.json", {})
    assert report["instructions"][1] == "點擊「儲存」"
    assert take_recording_replay_dirty() is True


def test_replace_and_reanalyze_keeps_index_for_replay_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir)
    monkeypatch.setattr("src.recorder.rerecord_step._WINDOW_AFTER_SETTLE_S", 0.0)
    monkeypatch.setattr(
        "src.recorder.rerecord_step.snapshot_top_level_windows",
        lambda: [],
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_step_signals",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "src.recorder.rerecord_step.capture_fast_signals",
        lambda **_kwargs: {},
    )

    async def fake_reanalyze(
        run_dir,
        event_index,
        *,
        log_info=None,
        preserve_from_analysis=None,
        rebuild_caches=True,
    ):
        analysis_path = Path(run_dir) / "analysis" / f"event_{event_index:03d}.json"
        analysis_path.write_text(
            json.dumps(
                {
                    "event_index": event_index,
                    "instruction": "點擊「新目標」",
                    "use_expected_outcome": False,
                }
            ),
            encoding="utf-8",
        )
        report = read_json(run_dir / "report.json", {})
        report["instructions"] = ["click 1", "點擊「新目標」"]
        (run_dir / "report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        from src.common.run_control import mark_recording_replay_dirty

        invalidate_recording_replay_cache(run_dir)
        mark_recording_replay_dirty()
        return {
            "event_index": event_index,
            "instruction": "點擊「新目標」",
            "kind": "click",
        }

    monkeypatch.setattr(
        "src.recorder.rerecord_step.reanalyze_recording_event",
        fake_reanalyze,
    )

    shot = run_dir / "screenshots" / "event_002.jpeg"
    shot.write_bytes(b"replaced")
    result = replace_and_reanalyze_pointer_event(
        run_dir,
        2,
        kind="click",
        cursor_xy=(50, 60),
        button="left",
        timestamp_utc="2026-10-30T01:00:00+00:00",
        screenshot_path=str(shot),
        monitor_index=1,
        monitor_offset=(0, 0),
        click_window=None,
        windows_before=(),
        signals_before={},
    )
    assert result["event_index"] == 2
    assert result["instruction"] == "點擊「新目標」"

    invalidate_recording_replay_cache(run_dir)
    fresh = load_recording_replay_steps(run_dir)
    merged = merge_unstarted_recording_steps(
        step_index=1,
        current_lines=["click 1", "click 2"],
        current_outcomes=[None, None],
        current_baselines=[None, None],
        current_verifies=[{}, {}],
        current_event_indices=[1, 2],
        fresh_lines=fresh[0],
        fresh_outcomes=fresh[1],
        fresh_baselines=fresh[2],
        fresh_verifies=fresh[3],
        fresh_event_indices=fresh[4],
    )
    assert merged[0] == ["click 1", "點擊「新目標」"]
    assert merged[4] == [1, 2]


def test_step_rerecord_cancel_discards_temp_without_splice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.recorder.rerecord_step import StepRerecordSession

    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=2)
    cancelled: list[bool] = []
    done: list[object] = []
    uia_wired: list[bool] = []
    escape_wired: list[bool] = []

    class _FakeCapture:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._count = 0

        def set_on_uia_status(self, callback: object) -> None:
            uia_wired.append(callback is not None)

        def set_on_escape_cancel(self, callback: object) -> None:
            escape_wired.append(callback is not None)

        def start(self, **kwargs: object) -> Path:
            return Path(kwargs["existing_run_dir"])  # type: ignore[index]

        def begin_stop(self) -> Path | None:
            return None

        def finalize_stop(self, **kwargs: object) -> Path | None:
            return None

        def event_count(self) -> int:
            return self._count

    monkeypatch.setattr("src.recorder.capture.RecordingSession", _FakeCapture)

    session = StepRerecordSession()
    session.arm(
        run_dir=run_dir,
        event_index=2,
        on_uia_status=lambda _payload: None,
        on_done=lambda result: done.append(result),
        on_cancelled=lambda: cancelled.append(True),
    )
    assert session.is_armed()
    assert uia_wired == [True]
    assert escape_wired == [True]
    temp = run_dir / ".rerecord_tmp"
    assert temp.is_dir()
    session.cancel()
    assert cancelled == [True]
    assert done == []
    assert not session.is_armed()
    assert not temp.exists()
    assert (run_dir / "events" / "event_002.json").is_file()


def test_load_events_falls_back_when_manifest_events_empty(tmp_path: Path) -> None:
    """Seeded ``events: []`` must not hide on-disk event JSON (rerecord temp)."""
    from src.recorder.orchestrator import _load_events

    run_dir = tmp_path / "tmp_capture"
    _write_temp_capture(run_dir, kinds=["text_input"])
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "events": [],
                "event_count": 0,
            }
        ),
        encoding="utf-8",
    )

    events = _load_events(run_dir)
    assert len(events) == 1
    assert events[0].kind == "text_input"
    assert events[0].text == "hello1"


def test_splice_loads_orphan_files_when_manifest_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=2)
    temp_dir = tmp_path / "tmp_capture"
    _write_temp_capture(temp_dir, kinds=["text_input"])
    (temp_dir / "session.json").write_text(
        json.dumps(
            {
                "run_id": temp_dir.name,
                "events": [],
                "event_count": 0,
            }
        ),
        encoding="utf-8",
    )

    new_indices = splice_rerecord_events(run_dir, 2, temp_dir)
    assert new_indices == [3]
    session = read_json(run_dir / "session.json", {})
    assert "events/event_003.json" in session["events"]
    assert (run_dir / "events" / "event_003.json").is_file()


def test_begin_replay_gesture_capture_shows_uia_overlay() -> None:
    import inspect

    from app_main_hub import MainHub

    source = inspect.getsource(MainHub._begin_replay_gesture_capture)
    assert "_show_recording_uia_status_overlay" in source
    assert "on_uia_status=self._on_recording_uia_status" in source
    finish_src = inspect.getsource(MainHub._on_finish_replay_rerecord)
    assert "_destroy_recording_uia_status_overlay" in finish_src


def test_replay_rerecord_done_does_not_auto_resume() -> None:
    """Hub done handler must stay paused (no _on_resume_run call)."""
    import inspect

    from app_main_hub import MainHub

    source = inspect.getsource(MainHub._on_replay_rerecord_done)
    assert "_on_resume_run" not in source
    assert "request_step_jump" in source
    assert 'mode == "insert_after"' in source
    assert "resume_at" in source
    assert "block_start" in source


def test_refresh_open_recording_script_preview_reloads_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a repair, the hub recording preview must re-read disk without reset-to-1."""
    from app_main_hub import MainHub
    from src.common.script_helper import collect_recording_script_text

    run_dir = tmp_path / "rec"
    _write_recording(run_dir, event_count=2)
    (run_dir / "analysis" / "event_002.json").write_text(
        json.dumps(
            {
                "event_index": 2,
                "instruction": "repaired step",
                "use_expected_outcome": False,
            }
        ),
        encoding="utf-8",
    )

    hub = MainHub.__new__(MainHub)
    hub._script_path = run_dir
    hub._active_recording_dir = run_dir
    hub._replay_nav_index = 1
    hub._track_script_step_status = False
    hub._script_step_status_by_line = {}
    hub._script_step_status_baseline = "\n"
    hub._suppress_script_cache_sync = False
    texts: list[str] = ["old text"]
    steps_set: list[int] = []

    class _FakeText:
        def configure(self, **_kwargs: object) -> None:
            return None

        def cget(self, _key: str) -> str:
            return "disabled"

        def delete(self, *_args: object) -> None:
            return None

        def insert(self, _index: object, text: str) -> None:
            texts[0] = text

        def get(self, *_args: object) -> str:
            return texts[0]

    hub._script_text = _FakeText()
    hub._script_step_entry = None
    hub._reset_textbox_undo = lambda *_a, **_k: None  # type: ignore[method-assign]
    hub._apply_script_editor_lock = lambda: None  # type: ignore[method-assign]
    hub._clear_script_step_statuses = lambda: None  # type: ignore[method-assign]
    hub._refresh_script_line_numbers = lambda: None  # type: ignore[method-assign]
    hub._mark_script_clean = lambda: None  # type: ignore[method-assign]
    hub._set_script_step_entry_value = (  # type: ignore[method-assign]
        lambda step: steps_set.append(int(step))
    )
    hub._highlight_script_step = lambda _step: None  # type: ignore[method-assign]
    hub._is_recording_script_open = lambda: True  # type: ignore[method-assign]

    MainHub._refresh_open_recording_script_preview(hub, prefer_step_1based=2)

    assert "repaired step" in texts[0]
    assert texts[0] == collect_recording_script_text(run_dir)
    assert steps_set == [2]


def test_replay_rerecord_done_refreshes_recording_preview() -> None:
    import inspect

    from app_main_hub import MainHub

    source = inspect.getsource(MainHub._on_replay_rerecord_done)
    assert "_refresh_open_recording_script_preview" in source
    delete_src = inspect.getsource(MainHub._finish_replay_step_delete)
    assert "_refresh_open_recording_script_preview" in delete_src
    edit_src = inspect.getsource(MainHub._on_save_replay_step_edit)
    assert "_refresh_open_recording_script_preview" in edit_src


def test_replay_insert_done_jumps_past_inserted_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insert keeps the anchor step and resumes after the whole inserted block."""
    import app_main_hub
    from app_main_hub import MainHub

    hub = MainHub.__new__(MainHub)
    hub._replay_overlay = None
    hub._replay_step_instructions = ["a", "b", "c"]
    hub._replay_event_indices = [1, 2, 3]
    hub._replay_step_errors = {2: "later fail"}
    hub._replay_nav_index = 1
    hub._replay_nav_held = False
    hub._script_path = None
    hub._script_text = None
    hub._active_recording_dir = None
    jumps: list[int] = []
    shown: list[int] = []
    refreshed: list[int | None] = []

    hub._restore_pause_button_after_rerecord = lambda: None  # type: ignore[method-assign]
    hub._destroy_recording_uia_status_overlay = lambda: None  # type: ignore[method-assign]
    hub._show_replay_step = lambda index: shown.append(index)  # type: ignore[method-assign]
    hub._refresh_open_recording_script_preview = (  # type: ignore[method-assign]
        lambda *, prefer_step_1based=None: refreshed.append(prefer_step_1based)
    )
    hub._status = type("S", (), {"configure": lambda *a, **k: None})()
    monkeypatch.setattr(app_main_hub, "request_step_jump", lambda index: jumps.append(index))

    MainHub._on_replay_rerecord_done(
        hub,
        {
            "mode": "insert_after",
            "instructions": ["x", "y"],
            "event_indices": [4, 5],
        },
        1,
    )

    assert hub._replay_step_instructions == ["a", "b", "x", "y", "c"]
    assert hub._replay_event_indices == [1, 2, 4, 5, 3]
    assert hub._replay_step_errors == {4: "later fail"}
    # Inserted at index 2..3; resume on following step "c" (index 4).
    assert hub._replay_nav_index == 4
    assert hub._replay_run_index == 4
    assert jumps == [4]
    assert shown == [4]
    assert refreshed == [5]


def test_replay_replace_done_jumps_past_rerecord_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace splices 1→N and resumes after the new block."""
    import app_main_hub
    from app_main_hub import MainHub

    hub = MainHub.__new__(MainHub)
    hub._replay_overlay = None
    hub._replay_step_instructions = ["a", "b", "c"]
    hub._replay_event_indices = [1, 2, 3]
    hub._replay_step_errors = {1: "fail", 2: "later"}
    hub._replay_nav_index = 1
    hub._replay_nav_held = False
    hub._script_path = None
    hub._script_text = None
    hub._active_recording_dir = None
    jumps: list[int] = []
    shown: list[int] = []

    hub._restore_pause_button_after_rerecord = lambda: None  # type: ignore[method-assign]
    hub._destroy_recording_uia_status_overlay = lambda: None  # type: ignore[method-assign]
    hub._show_replay_step = lambda index: shown.append(index)  # type: ignore[method-assign]
    hub._refresh_open_recording_script_preview = (  # type: ignore[method-assign]
        lambda *, prefer_step_1based=None: None
    )
    hub._status = type("S", (), {"configure": lambda *a, **k: None})()
    monkeypatch.setattr(app_main_hub, "request_step_jump", lambda index: jumps.append(index))

    MainHub._on_replay_rerecord_done(
        hub,
        {
            "mode": "replace",
            "instructions": ["x", "y"],
            "event_indices": [4, 5],
        },
        1,
    )

    assert hub._replay_step_instructions == ["a", "x", "y", "c"]
    assert hub._replay_event_indices == [1, 4, 5, 3]
    assert hub._replay_step_errors == {3: "later"}
    assert hub._replay_nav_index == 3
    assert jumps == [3]
    assert shown == [3]


def test_replay_replace_done_past_end_jumps_to_script_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When repair ends the script, jump to len(steps) so 繼續 finishes."""
    import app_main_hub
    from app_main_hub import MainHub

    hub = MainHub.__new__(MainHub)
    hub._replay_overlay = None
    hub._replay_step_instructions = ["a", "b"]
    hub._replay_event_indices = [1, 2]
    hub._replay_step_errors = {}
    hub._replay_nav_index = 1
    hub._script_path = None
    hub._script_text = None
    hub._active_recording_dir = None
    jumps: list[int] = []
    shown: list[int | None] = []

    hub._restore_pause_button_after_rerecord = lambda: None  # type: ignore[method-assign]
    hub._destroy_recording_uia_status_overlay = lambda: None  # type: ignore[method-assign]
    hub._show_replay_step = lambda index: shown.append(index)  # type: ignore[method-assign]
    hub._refresh_open_recording_script_preview = (  # type: ignore[method-assign]
        lambda *, prefer_step_1based=None: None
    )
    hub._status = type("S", (), {"configure": lambda *a, **k: None})()
    monkeypatch.setattr(app_main_hub, "request_step_jump", lambda index: jumps.append(index))

    MainHub._on_replay_rerecord_done(
        hub,
        {
            "mode": "replace",
            "instructions": ["x", "y"],
            "event_indices": [3, 4],
        },
        1,
    )

    assert hub._replay_step_instructions == ["a", "x", "y"]
    assert hub._replay_nav_index == 2
    assert hub._replay_run_index == 3
    assert jumps == [3]
    assert shown == [2]
