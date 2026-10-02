"""Tests for recording settle probe selection and consecutive-sample retention."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.recorder.capture import (
    _SETTLE_PROBE_FIRST_S,
    _SETTLE_PROBE_MAX_WINDOW_S,
    settle_probe_interval_s,
)
from src.recorder.frame_similarity import apply_settle_sample, last_settle_frame_path
from src.recorder.models import RecordedEvent
from src.recorder.orchestrator import _next_instruction_event_settle


def _write_gray(path: Path, value: int) -> None:
    Image.new("L", (64, 64), color=int(value)).save(path, format="JPEG", quality=95)


def test_settle_probe_interval_is_fixed() -> None:
    assert settle_probe_interval_s(0) == 0.0
    assert [settle_probe_interval_s(i) for i in range(1, 10)] == [1.0] * 9
    assert _SETTLE_PROBE_FIRST_S == 0.0
    assert _SETTLE_PROBE_MAX_WINDOW_S == 55.0
    assert settle_probe_interval_s(-3) == 0.0


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
    assert kept == s3
    assert age == 3.0
    assert s3.is_file()
    assert not s2.is_file()


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
        trailing_settle_end_utc="2026-09-07T00:00:20+00:00",
    )
    assert settle == 1.5


def test_next_instruction_event_settle_keeps_zero_observed() -> None:
    events = [
        RecordedEvent(
            index=0,
            timestamp_utc="2026-09-07T00:00:00+00:00",
            kind="click",
            observed_settle_seconds=0.0,
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
        trailing_settle_end_utc="2026-09-07T00:00:20+00:00",
    )
    assert settle == 0.0


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
        trailing_settle_end_utc="2026-09-07T00:00:20+00:00",
    )
    assert settle == 10.0


def test_next_instruction_event_settle_last_ignores_stop_uses_trailing() -> None:
    events = [
        RecordedEvent(
            index=0,
            timestamp_utc="2026-09-07T00:00:00+00:00",
            kind="click",
        ),
    ]
    # Without trailing end, last event has no settle (stop time must not be used).
    assert (
        _next_instruction_event_settle(
            events=events,
            event_pos=0,
            event=events[0],
            prepared_list=[object()],
            instruction_results=[{"instruction": "a"}],
            trailing_settle_end_utc=None,
        )
        is None
    )
    # Trailing restore click timestamp is allowed as settle end for the last action.
    assert (
        _next_instruction_event_settle(
            events=events,
            event_pos=0,
            event=events[0],
            prepared_list=[object()],
            instruction_results=[{"instruction": "a"}],
            trailing_settle_end_utc="2026-09-07T00:00:04+00:00",
        )
        == 4.0
    )


def test_text_input_settle_is_one_second() -> None:
    """Typing steps settle for one second. Keystroke time and later pauses are not waits."""
    events = [
        RecordedEvent(
            index=0,
            timestamp_utc="2026-09-07T00:00:00+00:00",
            kind="text_input",
            text="google",
            observed_settle_seconds=4.5,
        ),
        RecordedEvent(
            index=1,
            timestamp_utc="2026-09-07T00:00:08+00:00",
            kind="click",
        ),
    ]
    settle = _next_instruction_event_settle(
        events=events,
        event_pos=0,
        event=events[0],
        prepared_list=[object(), object()],
        instruction_results=[{"instruction": "type"}, {"instruction": "click"}],
    )
    assert settle == 1.0


def test_text_input_settle_last_step_is_one_second() -> None:
    events = [
        RecordedEvent(
            index=0,
            timestamp_utc="2026-09-07T00:00:00+00:00",
            kind="text_input",
            text="done",
        ),
    ]
    settle = _next_instruction_event_settle(
        events=events,
        event_pos=0,
        event=events[0],
        prepared_list=[object()],
        instruction_results=[{"instruction": "type"}],
        trailing_settle_end_utc="2026-09-07T00:00:06+00:00",
    )
    assert settle == 1.0


def test_settle_probe_ticks_run_off_event_queue(tmp_path: Path, monkeypatch) -> None:
    """Settle comparisons must not wait behind click-persist work on the event queue."""
    from src.recorder.capture import RecordingSession, _SETTLE_PROBE_FIRST_S

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    (run_dir / "events").mkdir(parents=True)

    session = RecordingSession.__new__(RecordingSession)
    session._lock = __import__("threading").Lock()
    session._settle_tick_lock = __import__("threading").Lock()
    session._run_dir = run_dir
    session._events = []
    session._settle_probe = None
    session._settle_probe_timer = None
    session._last_settle_frame = None
    session._event_queue = __import__("queue").Queue()
    session._log = lambda *_a, **_k: None  # type: ignore[method-assign]

    fired: list[int] = []

    def _fake_run(event_index: int) -> None:
        fired.append(event_index)

    monkeypatch.setattr(session, "_run_settle_probe_tick", _fake_run)

    from src.recorder.capture import _SettleProbeState

    session._settle_probe = _SettleProbeState(
        event_index=7,
        started_monotonic=__import__("time").monotonic(),
        cursor_xy=(10, 10),
    )
    session._schedule_settle_probe_tick(0.05, 7)
    assert session._event_queue.empty()
    deadline = __import__("time").monotonic() + 1.0
    while not fired and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    assert fired == [7]
    # Keep default first delay documented for callers.
    assert _SETTLE_PROBE_FIRST_S == settle_probe_interval_s(0) == 0.0


def test_schedule_next_settle_probe_tick_uses_fixed_interval(
    tmp_path: Path, monkeypatch
) -> None:
    import threading
    import time

    from src.recorder.capture import RecordingSession, _SettleProbeState

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._settle_tick_lock = threading.Lock()
    session._run_dir = run_dir
    session._settle_probe = _SettleProbeState(
        event_index=3,
        started_monotonic=time.monotonic(),
        interval_step=0,
    )
    session._settle_probe_timer = None

    delays: list[float] = []

    def _capture_schedule(delay_s: float, event_index: int) -> None:
        delays.append(float(delay_s))
        assert event_index == 3

    monkeypatch.setattr(session, "_schedule_settle_probe_tick", _capture_schedule)

    session._schedule_next_settle_probe_tick(3)
    session._schedule_next_settle_probe_tick(3)
    session._schedule_next_settle_probe_tick(3)
    session._schedule_next_settle_probe_tick(3)
    assert delays == [0.0, 1.0, 1.0, 1.0]
    assert session._settle_probe.interval_step == 4


def test_settle_probe_continues_after_first_similar_plateau(
    tmp_path: Path, monkeypatch
) -> None:
    """Similar frames record observed_settle once but keep the probe running."""
    import json
    import threading
    import time

    from src.recorder.capture import RecordingSession, _SettleProbeState
    from src.recorder.models import event_json_path

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    (run_dir / "events").mkdir(parents=True)
    event_path = event_json_path(run_dir, 2)
    event_path.write_text(
        json.dumps(
            {
                "index": 2,
                "timestamp_utc": "2026-09-29T00:00:00+00:00",
                "kind": "click",
            }
        ),
        encoding="utf-8",
    )

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._settle_tick_lock = threading.Lock()
    session._event_file_lock = threading.Lock()
    session._run_dir = run_dir
    session._events = []
    session._settle_pub_seq = 0
    session._settle_pins = {}
    session._last_settle_frame = None
    session._settle_probe_timer = None
    session._log = lambda *_a, **_k: None  # type: ignore[method-assign]
    session._settle_probe = _SettleProbeState(
        event_index=2,
        started_monotonic=time.monotonic() - 1.0,
        cursor_xy=(10, 10),
    )

    schedules: list[int] = []
    monkeypatch.setattr(
        session,
        "_schedule_next_settle_probe_tick",
        lambda event_index: schedules.append(int(event_index)),
    )

    gray = tmp_path / "frame.jpeg"
    Image.new("L", (32, 24), color=40).save(gray, format="JPEG")

    def _fake_capture(x, y, target):
        import shutil

        shutil.copy2(gray, target)
        return str(target), 1, (0, 0)

    import src.recorder.capture as capture_mod

    monkeypatch.setattr(capture_mod, "_capture_screenshot_at_point", _fake_capture)
    monkeypatch.setattr(
        capture_mod,
        "compare_frames",
        lambda *_a, **_k: (True, 0.0),
    )

    # First sample: keep only.
    session._run_settle_probe_tick(2)
    assert session._settle_probe is not None
    assert session._settle_probe.observed_recorded is False
    assert schedules == [2]

    # Second sample: similar → record observed, keep probing.
    session._run_settle_probe_tick(2)
    assert session._settle_probe is not None
    assert session._settle_probe.observed_recorded is True
    assert schedules == [2, 2]
    raw = json.loads(event_path.read_text(encoding="utf-8"))
    assert "observed_settle_seconds" in raw

    # Third similar sample: do not overwrite observed; still scheduled.
    first_observed = raw["observed_settle_seconds"]
    session._run_settle_probe_tick(2)
    assert session._settle_probe is not None
    assert schedules == [2, 2, 2]
    raw2 = json.loads(event_path.read_text(encoding="utf-8"))
    assert raw2["observed_settle_seconds"] == first_observed


def test_start_settle_probe_does_not_wait_for_inflight_tick(
    tmp_path: Path,
) -> None:
    """Replacing the probe must not block on a tick that still holds the lock."""
    import threading
    import time

    from src.recorder.capture import RecordingSession, _SettleProbeState
    from src.recorder.models import RecordedEvent

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._settle_tick_lock = threading.Lock()
    session._run_dir = run_dir
    session._events = []
    session._settle_pub_seq = 0
    session._settle_pins = {}
    session._last_settle_frame = None
    session._settle_probe_timer = None
    session._log = lambda *_a, **_k: None  # type: ignore[method-assign]
    session._settle_probe = _SettleProbeState(
        event_index=2,
        started_monotonic=time.monotonic(),
        cursor_xy=(1, 1),
    )

    held = threading.Event()
    release = threading.Event()

    def _hold_tick_lock() -> None:
        with session._settle_tick_lock:
            held.set()
            assert release.wait(2.0)

    holder = threading.Thread(target=_hold_tick_lock, daemon=True)
    holder.start()
    assert held.wait(1.0)

    newer = RecordedEvent(
        index=3,
        timestamp_utc="2026-09-29T00:00:00+00:00",
        kind="click",
        cursor_xy=(10, 10),
    )
    started = time.monotonic()
    session._start_settle_probe(newer)
    elapsed = time.monotonic() - started
    release.set()
    holder.join(timeout=1.0)

    assert elapsed < 0.2
    probe = session._settle_probe
    assert probe is not None
    assert probe.event_index == 3


def test_superseded_tick_does_not_publish_over_newer_frame(
    tmp_path: Path, monkeypatch
) -> None:
    """An old tick finishing after replace must not clobber the newer publish."""
    import threading
    import time

    from PIL import Image

    from src.recorder.capture import RecordingSession, _SettleProbeState

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    old_src = run_dir / "screenshots" / "old.jpeg"
    new_src = run_dir / "screenshots" / "new.jpeg"
    Image.new("RGB", (16, 12), color=(200, 0, 0)).save(old_src, format="JPEG")
    Image.new("RGB", (16, 12), color=(0, 0, 200)).save(new_src, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._settle_tick_lock = threading.Lock()
    session._run_dir = run_dir
    session._settle_pub_seq = 0
    session._settle_pins = {}
    session._last_settle_frame = None
    session._settle_probe = _SettleProbeState(
        event_index=4,
        started_monotonic=time.monotonic(),
    )

    assert session._publish_last_settle_frame(
        new_src, 1, (0, 0), event_index=4
    )
    published = session._last_settle_frame
    assert published is not None
    assert Path(published[0]).is_file()

    # Simulate event-3 tick finishing after replace to event 4.
    assert not session._publish_last_settle_frame(
        old_src, 1, (0, 0), event_index=3
    )
    assert session._last_settle_frame == published
    r, _g, b = Image.open(published[0]).convert("RGB").getpixel((0, 0))
    assert b > 150
    assert r < 50


def test_finish_settle_against_final_after_persists_observed(
    tmp_path: Path, monkeypatch
) -> None:
    """Last-step settle matches all-monitor samples to final_after."""
    import json
    import threading
    import time

    from src.recorder.capture import RecordingSession, _SettleProbeState
    from src.recorder.models import RecordedEvent, event_json_path

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    (run_dir / "events").mkdir(parents=True)

    final_after = run_dir / "screenshots" / "final_after.jpeg"
    _write_gray(final_after, 100)

    event = RecordedEvent(
        index=5,
        timestamp_utc="2026-09-21T00:00:00+00:00",
        kind="click",
    )
    event_json_path(run_dir, 5).write_text(
        json.dumps(event.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._settle_tick_lock = threading.Lock()
    session._run_dir = run_dir
    session._events = [event]
    session._settle_probe_timer = None
    session._last_settle_frame = None
    session._log = lambda *_a, **_k: None  # type: ignore[method-assign]
    session._settle_probe = _SettleProbeState(
        event_index=5,
        started_monotonic=time.monotonic() - 1.25,
        cursor_xy=(10, 10),
    )

    def _fake_capture(dest: Path) -> str:
        _write_gray(Path(dest), 100)
        return str(dest)

    monkeypatch.setattr(
        "src.recorder.capture.capture_all_screens_to_file",
        _fake_capture,
    )
    monkeypatch.setattr(session, "_publish_last_settle_frame", lambda *_a, **_k: None)

    session._finish_settle_against_final_after(final_after)

    raw = json.loads(event_json_path(run_dir, 5).read_text(encoding="utf-8"))
    assert raw["observed_settle_seconds"] >= 1.0
    assert session._settle_probe is None


def test_finish_settle_against_final_after_skips_when_already_observed(
    tmp_path: Path, monkeypatch
) -> None:
    import json
    import threading
    import time

    from src.recorder.capture import RecordingSession, _SettleProbeState
    from src.recorder.models import RecordedEvent, event_json_path

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    (run_dir / "events").mkdir(parents=True)
    final_after = run_dir / "screenshots" / "final_after.jpeg"
    _write_gray(final_after, 100)

    event = RecordedEvent(
        index=3,
        timestamp_utc="2026-09-21T00:00:00+00:00",
        kind="click",
        observed_settle_seconds=1.031,
    )
    event_json_path(run_dir, 3).write_text(
        json.dumps(event.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._settle_tick_lock = threading.Lock()
    session._run_dir = run_dir
    session._events = [event]
    session._settle_probe_timer = None
    session._last_settle_frame = None
    session._log = lambda *_a, **_k: None  # type: ignore[method-assign]
    session._settle_probe = _SettleProbeState(
        event_index=3,
        started_monotonic=time.monotonic(),
    )

    captures: list[Path] = []

    def _fake_capture(dest: Path) -> str:
        captures.append(Path(dest))
        return str(dest)

    monkeypatch.setattr(
        "src.recorder.capture.capture_all_screens_to_file",
        _fake_capture,
    )

    session._finish_settle_against_final_after(final_after)
    assert captures == []
    assert session._settle_probe is None
    raw = json.loads(event_json_path(run_dir, 3).read_text(encoding="utf-8"))
    assert raw["observed_settle_seconds"] == 1.031


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


def test_before_shot_keeps_settle_sample_latched_at_click(tmp_path: Path) -> None:
    """The next step's before-shot is the settle sample finished before the click."""
    from src.recorder.capture import RecordingSession

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    latched_settle = run_dir / "screenshots" / "_settle_pub_00001.jpeg"
    Image.new("RGB", (40, 30), color=(200, 0, 0)).save(latched_settle, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = __import__("threading").Lock()
    session._window_refresh_lock = __import__("threading").Lock()
    session._context_epoch = 0
    session._last_settle_signals = None
    session._settle_pins = {}
    session._settle_pub_seq = 1
    session._pre_click_monitors = ((1, 0, 0, 1920, 1080),)
    session._last_settle_frame = (str(latched_settle), 1, (0, 0))
    session._run_dir = run_dir

    settle_latched = session._latch_settle_frame()
    newer = run_dir / "screenshots" / "_settle_pub_00002.jpeg"
    Image.new("RGB", (40, 30), color=(0, 200, 0)).save(newer, format="JPEG")
    session._publish_settle_frame((str(newer), 1, (0, 0)))
    assert latched_settle.is_file()

    dest = run_dir / "screenshots" / "event_007.jpeg"
    info = session._pending_screenshot_from_settle_or_capture(
        run_dir,
        10,
        10,
        dest,
        settle_latched,
    )
    assert info[1] == 1
    r, _g, _b = Image.open(dest).convert("RGB").getpixel((0, 0))
    assert r > 150
    assert session._settle_pins == {}
    assert not latched_settle.is_file()


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


def test_before_shot_capture_publishes_without_settle_probe(tmp_path: Path, monkeypatch) -> None:
    """A click can pin a frame before any settle probe has sampled."""
    import threading

    from src.recorder.capture import RecordingSession

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    shot = run_dir / "screenshots" / "grab.jpeg"
    Image.new("RGB", (16, 12), color=(0, 180, 0)).save(shot, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._screenshot_grab_lock = threading.Lock()
    session._before_shot_stop = threading.Event()
    session._run_dir = run_dir
    session._settle_pub_seq = 0
    session._settle_pins = {}
    session._last_settle_frame = None
    session._settle_probe = None
    session._log = lambda *_a, **_k: None  # type: ignore[method-assign]
    session._last_pointer_cursor_xy = (4, 4)

    def _fake_capture(x, y, target):
        import shutil

        shutil.copy2(shot, target)
        return str(target), 2, (0, 0)

    import src.recorder.capture as capture_mod

    monkeypatch.setattr(capture_mod, "_capture_screenshot_at_point", _fake_capture)

    assert session._capture_and_publish_before_shot()
    published = session._last_settle_frame
    assert published is not None
    assert Path(published[0]).name.startswith("_settle_pub_")
    assert published[1:] == (2, (0, 0))
    _r, g, _b = Image.open(published[0]).convert("RGB").getpixel((0, 0))
    assert g > 150


def test_settle_tick_does_not_replace_before_shot(tmp_path: Path, monkeypatch) -> None:
    """Settle samples must not clobber the frame the next click will pin."""
    import threading
    import time

    from src.recorder.capture import RecordingSession, _SettleProbeState

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    before = run_dir / "screenshots" / "before.jpeg"
    settle = run_dir / "screenshots" / "settle.jpeg"
    Image.new("RGB", (16, 12), color=(200, 0, 0)).save(before, format="JPEG")
    Image.new("RGB", (16, 12), color=(0, 0, 200)).save(settle, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = threading.Lock()
    session._settle_tick_lock = threading.Lock()
    session._screenshot_grab_lock = threading.Lock()
    session._event_file_lock = threading.Lock()
    session._run_dir = run_dir
    session._events = []
    session._settle_pub_seq = 0
    session._settle_pins = {}
    session._last_settle_frame = None
    session._settle_probe_timer = None
    session._log = lambda *_a, **_k: None  # type: ignore[method-assign]
    session._last_pointer_cursor_xy = (1, 1)
    session._settle_probe = _SettleProbeState(
        event_index=4,
        started_monotonic=time.monotonic(),
        cursor_xy=(1, 1),
    )
    assert session._publish_last_settle_frame(before, 1, (0, 0))
    pinned = session._last_settle_frame
    assert pinned is not None

    def _fake_capture(x, y, target):
        import shutil

        shutil.copy2(settle, target)
        return str(target), 1, (0, 0)

    import src.recorder.capture as capture_mod

    monkeypatch.setattr(capture_mod, "_capture_screenshot_at_point", _fake_capture)
    monkeypatch.setattr(session, "_schedule_next_settle_probe_tick", lambda *_a, **_k: None)

    session._run_settle_probe_tick(4)
    assert session._last_settle_frame == pinned
    r, _g, b = Image.open(pinned[0]).convert("RGB").getpixel((0, 0))
    assert r > 150
    assert b < 50
    assert session._settle_probe is not None
    assert session._settle_probe.kept_path


def test_publish_last_settle_frame_copies_durable_candidate(tmp_path: Path) -> None:
    from src.recorder.capture import RecordingSession

    run_dir = tmp_path / "rec"
    (run_dir / "screenshots").mkdir(parents=True)
    source = run_dir / "screenshots" / "_settle_stable_001.jpeg"
    Image.new("RGB", (32, 24), color=(1, 2, 3)).save(source, format="JPEG")

    session = RecordingSession.__new__(RecordingSession)
    session._lock = __import__("threading").Lock()
    session._window_refresh_lock = __import__("threading").Lock()
    session._context_epoch = 0
    session._last_settle_signals = None
    session._settle_pins = {}
    session._settle_pub_seq = 0
    session._run_dir = run_dir
    session._last_settle_frame = None
    session._publish_last_settle_frame(source, 1, (0, 0))

    dest = last_settle_frame_path(run_dir)
    assert dest.is_file()
    published = session._last_settle_frame
    assert published is not None
    assert Path(published[0]).is_file()
    assert Path(published[0]).name.startswith("_settle_pub_")
    assert published[1:] == (1, (0, 0))
