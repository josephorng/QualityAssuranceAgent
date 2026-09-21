"""Tests for recording settle probe selection and consecutive-sample retention."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.recorder.capture import (
    _SETTLE_PROBE_FIRST_S,
    _SETTLE_PROBE_INTERVAL_S,
)
from src.recorder.frame_similarity import apply_settle_sample, last_settle_frame_path
from src.recorder.models import RecordedEvent
from src.recorder.orchestrator import _next_instruction_event_settle


def _write_gray(path: Path, value: int) -> None:
    Image.new("L", (64, 64), color=int(value)).save(path, format="JPEG", quality=95)


def test_settle_probe_timing_defaults_are_one_second() -> None:
    assert _SETTLE_PROBE_FIRST_S == 1.0
    assert _SETTLE_PROBE_INTERVAL_S == 1.0


def test_apply_settle_sample_change_then_stable(tmp_path: Path) -> None:
    s1 = tmp_path / "s1.jpeg"
    s2 = tmp_path / "s2.jpeg"
    s3 = tmp_path / "s3.jpeg"
    _write_gray(s1, 0)
    _write_gray(s2, 255)
    _write_gray(s3, 255)

    kept, age, observed = apply_settle_sample(
        kept_path=None,
        kept_age_s=None,
        staging_path=s1,
        sample_age_s=1.0,
    )
    assert observed is None
    assert kept == s1
    assert age == 1.0

    kept, age, observed = apply_settle_sample(
        kept_path=kept,
        kept_age_s=age,
        staging_path=s2,
        sample_age_s=2.0,
    )
    assert observed is None
    assert kept == s2
    assert age == 2.0
    assert not s1.is_file()

    kept, age, observed = apply_settle_sample(
        kept_path=kept,
        kept_age_s=age,
        staging_path=s3,
        sample_age_s=3.0,
    )
    assert observed == 2.0
    assert kept == s2
    assert not s3.is_file()


def test_next_instruction_event_settle_prefers_observed() -> None:
    events = [
        RecordedEvent(
            index=0,
            timestamp_utc="2026-09-07T00:00:00+00:00",
            kind="click",
            observed_settle_seconds=1.5,
        ),
        RecordedEvent(
            index=1,
            timestamp_utc="2026-09-07T00:00:10+00:00",
            kind="click",
        ),
    ]
    settle = _next_instruction_event_settle(
        events=events,
        event_pos=0,
        event=events[0],
        prepared_list=[object(), object()],
        instruction_results=[{"instruction": "a"}, {"instruction": "b"}],
        stopped_at_utc="2026-09-07T00:00:20+00:00",
    )
    assert settle == 1.5


def test_next_instruction_event_settle_falls_back_to_gap() -> None:
    events = [
        RecordedEvent(
            index=0,
            timestamp_utc="2026-09-07T00:00:00+00:00",
            kind="click",
        ),
        RecordedEvent(
            index=1,
            timestamp_utc="2026-09-07T00:00:10+00:00",
            kind="click",
        ),
    ]
    settle = _next_instruction_event_settle(
        events=events,
        event_pos=0,
        event=events[0],
        prepared_list=[object(), object()],
        instruction_results=[{"instruction": "a"}, {"instruction": "b"}],
        stopped_at_utc="2026-09-07T00:00:20+00:00",
    )
    assert settle == 10.0


def test_pending_screenshot_reuses_last_settle_on_same_monitor(tmp_path: Path) -> None:
    from src.recorder.capture import RecordingSession

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    stable = last_settle_frame_path(run_dir)
    Image.new("RGB", (40, 30), color=(10, 20, 30)).save(stable, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = __import__("threading").Lock()
    session._last_settle_frame = (str(stable), 1, (0, 0))
    session._run_dir = run_dir

    dest = run_dir / "screenshots" / "_pending_capture.jpeg"

    def _fake_monitor_at_point(x: int, y: int):
        return 1, 0, 0, 1920, 1080

    import src.recorder.capture as capture_mod

    original = capture_mod._monitor_at_point
    capture_mod._monitor_at_point = _fake_monitor_at_point  # type: ignore[assignment]
    try:
        info = session._pending_screenshot_from_settle_or_capture(run_dir, 10, 10, dest)
    finally:
        capture_mod._monitor_at_point = original  # type: ignore[assignment]

    assert info[0] == str(dest)
    assert info[1] == 1
    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_pending_screenshot_live_captures_when_monitor_mismatches(
    tmp_path: Path, monkeypatch
) -> None:
    from src.recorder.capture import RecordingSession

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    stable = last_settle_frame_path(run_dir)
    Image.new("RGB", (40, 30), color=(10, 20, 30)).save(stable, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = __import__("threading").Lock()
    session._last_settle_frame = (str(stable), 1, (0, 0))
    session._run_dir = run_dir

    dest = run_dir / "screenshots" / "_pending_capture.jpeg"
    live = run_dir / "screenshots" / "live.jpeg"
    Image.new("RGB", (40, 30), color=(200, 0, 0)).save(live, format="JPEG")

    import src.recorder.capture as capture_mod

    monkeypatch.setattr(
        capture_mod,
        "_monitor_at_point",
        lambda x, y: (2, 1920, 0, 1920, 1080),
    )

    def _fake_capture(x, y, target):
        import shutil

        shutil.copy2(live, target)
        return str(target), 2, (1920, 0)

    monkeypatch.setattr(capture_mod, "_capture_screenshot_at_point", _fake_capture)

    info = session._pending_screenshot_from_settle_or_capture(run_dir, 2000, 10, dest)
    assert info[1] == 2
    assert dest.is_file()
    # Live capture path used (monitor mismatch).
    r, _g, _b = Image.open(dest).convert("RGB").getpixel((0, 0))
    assert r > 150


def test_publish_last_settle_frame_copies_durable_candidate(tmp_path: Path) -> None:
    from src.recorder.capture import RecordingSession

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    source = run_dir / "screenshots" / "_settle_stable_001.jpeg"
    Image.new("RGB", (32, 24), color=(1, 2, 3)).save(source, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = __import__("threading").Lock()
    session._run_dir = run_dir
    session._last_settle_frame = None
    session._publish_last_settle_frame(source, 1, (0, 0))

    dest = last_settle_frame_path(run_dir)
    assert dest.is_file()
    assert session._last_settle_frame == (str(dest), 1, (0, 0))
