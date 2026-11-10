"""Settle-UIA LED status emissions and idle ElementFromPoint label."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import (
    RecordingSession,
    _DeferredCaptureJob,
    _IDLE_EFP_DEBOUNCE_S,
    _QueuedEvent,
)
from src.recorder.window_snapshot import ClickWindowInfo


def test_collect_and_publish_settle_uia_emits_yellow_then_green() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)

    def _snap(hwnd: int) -> dict:
        return {
            "window": [0, 0, 10, 10],
            "inputs": [],
            "scrollbars": [],
            "texts": [{"bbox": [0, 0, 1, 1], "text": "A"}],
        }

    with patch("src.recorder.uia_input_boxes.snapshot_uia_screen", side_effect=_snap):
        out = session._collect_and_publish_settle_uia(42, after_event_index=3)

    assert out is not None
    assert [row["state"] for row in statuses] == ["yellow", "green"]
    assert statuses[-1]["after_index"] == 3


def test_collect_and_publish_settle_uia_emits_red_on_fail() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)

    with patch("src.recorder.uia_input_boxes.snapshot_uia_screen", return_value=None):
        out = session._collect_and_publish_settle_uia(42, after_event_index=1)

    assert out is None
    assert [row["state"] for row in statuses] == ["yellow", "red"]


def test_collect_and_publish_settle_uia_red_for_bad_hwnd() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    assert session._collect_and_publish_settle_uia(0, after_event_index=2) is None
    assert statuses[-1]["state"] == "red"
    assert statuses[-1]["after_index"] == 2


def test_persist_pointer_pulses_attach_then_red(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    (tmp_path / "events").mkdir(parents=True, exist_ok=True)
    session.set_on_uia_status(statuses.append)
    session._start_settle_probe = lambda _event: None  # type: ignore[method-assign]
    session._start_window_step = lambda _item: None  # type: ignore[method-assign]
    session._notify_event = lambda _event: None  # type: ignore[method-assign]

    item = _QueuedEvent(
        kind="click",
        cursor_xy=(1, 2),
        event_index=1,
        timestamp_utc="t",
        screenshot_path="a.jpeg",
        monitor_index=1,
        monitor_offset=(0, 0),
        uia_snapshot={
            "hwnd": 9,
            "window": [0, 0, 1, 1],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    )
    session._persist_queued_event(item)
    assert any(row.get("pulse") == "attach" for row in statuses)
    assert statuses[-1]["state"] == "red"


def test_persist_pointer_pulses_miss_without_snapshot(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    (tmp_path / "events").mkdir(parents=True, exist_ok=True)
    session.set_on_uia_status(statuses.append)
    session._start_settle_probe = lambda _event: None  # type: ignore[method-assign]
    session._start_window_step = lambda _item: None  # type: ignore[method-assign]
    session._notify_event = lambda _event: None  # type: ignore[method-assign]

    item = _QueuedEvent(
        kind="click",
        cursor_xy=(1, 2),
        event_index=1,
        timestamp_utc="t",
        screenshot_path="a.jpeg",
        monitor_index=1,
        monitor_offset=(0, 0),
        uia_snapshot=None,
    )
    session._persist_queued_event(item)
    assert any(row.get("pulse") == "miss" for row in statuses)


def test_window_step_sealed_emits_red() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._resolve_window_change = lambda _item: (None, None, {"ok": True})  # type: ignore[method-assign]
    session._notify_pending_text_input = lambda _i: None  # type: ignore[method-assign]
    session._release_drag_uia_notify = lambda _item: None  # type: ignore[method-assign]
    session._patch_window_change = lambda *_a, **_k: None  # type: ignore[method-assign]

    item = _QueuedEvent(
        kind="click",
        cursor_xy=(1, 1),
        event_index=5,
        timestamp_utc="t",
        windows_before=(
            ClickWindowInfo(
                hwnd=1,
                title="A",
                process_name="a.exe",
                left=0,
                top=0,
                width=10,
                height=10,
                is_maximized=False,
            ),
        ),
        window_after_sealed=True,
    )
    session._window_step_body(item)
    assert statuses[-1]["state"] == "red"
    assert statuses[-1]["after_index"] == 5


def test_idle_efp_timer_cancel_on_move_drops_stale(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._accepting_input = True
    session.set_on_uia_status(statuses.append)

    calls: list[tuple[int, int]] = []

    def _probe(x: int, y: int):
        calls.append((x, y))
        return ("label", (0, 0, 1, 1), "text", "NameA", "ValA")

    with patch(
        "src.recorder.uia_input_boxes.uia_text_element_at_point",
        side_effect=_probe,
    ):
        session._schedule_idle_efp(10, 20)
        first_gen = session._idle_efp_generation
        # Move again before debounce — bumps generation and cancels timer.
        session._schedule_idle_efp(30, 40)
        assert session._idle_efp_generation == first_gen + 1
        time.sleep(_IDLE_EFP_DEBOUNCE_S + 0.15)
        # Only the second schedule should enqueue; drain worker manually.
        job = None
        try:
            while True:
                item = session._event_queue.get_nowait()
                if isinstance(item, _DeferredCaptureJob) and item.action == "idle_efp":
                    job = item
        except Exception:
            pass
        assert job is not None
        assert job.cursor_xy == (30, 40)
        session._worker_idle_efp(job)

    assert calls == [(30, 40)]
    assert statuses[-1]["efp_name"] == "NameA"
    assert statuses[-1]["efp_value"] == "ValA"
    assert statuses[-1]["efp_pending"] is False


def test_worker_idle_efp_ignores_stale_generation() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._accepting_input = True
    session.set_on_uia_status(statuses.append)
    session._idle_efp_generation = 9

    with patch(
        "src.recorder.uia_input_boxes.uia_text_element_at_point",
        return_value=("x", (0, 0, 1, 1), "text", "Stale", "X"),
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(1, 1),
                meta={"generation": 3},
            )
        )
    assert statuses == []


def test_recording_uia_status_overlay_apply_status() -> None:
    import customtkinter as ctk

    from src.recorder.stop_overlay import RecordingUiaStatusOverlay

    root = ctk.CTk()
    root.withdraw()
    try:
        overlay = RecordingUiaStatusOverlay(root)
        overlay.apply_status(
            {
                "state": "green",
                "pulse": None,
                "efp_name": "關閉",
                "efp_value": "",
                "efp_pending": False,
            }
        )
        assert overlay._state == "green"
        assert overlay._label is not None
        assert "關閉" in str(overlay._label.cget("text"))
        overlay.destroy()
    finally:
        root.destroy()


def test_should_exclude_uia_status_skips_remote_and_vm(monkeypatch) -> None:
    from src.recorder import stop_overlay as so

    monkeypatch.delenv(so._UIA_STATUS_EXCLUDE_FROM_CAPTURE_ENV, raising=False)
    monkeypatch.setattr(so, "_is_remote_display_session", lambda: False)
    monkeypatch.setattr(so, "_is_virtual_machine_guest", lambda: False)
    assert so._should_exclude_uia_status_from_capture() is True

    monkeypatch.setattr(so, "_is_remote_display_session", lambda: True)
    assert so._should_exclude_uia_status_from_capture() is False

    monkeypatch.setattr(so, "_is_remote_display_session", lambda: False)
    monkeypatch.setattr(so, "_is_virtual_machine_guest", lambda: True)
    assert so._should_exclude_uia_status_from_capture() is False

    monkeypatch.setattr(so, "_is_virtual_machine_guest", lambda: False)
    monkeypatch.setenv(so._UIA_STATUS_EXCLUDE_FROM_CAPTURE_ENV, "0")
    assert so._should_exclude_uia_status_from_capture() is False


def test_apply_exclude_from_capture_noop_when_gated(monkeypatch) -> None:
    from src.recorder import stop_overlay as so

    calls: list[tuple[int, int]] = []

    class _User32:
        @staticmethod
        def SetWindowDisplayAffinity(hwnd: int, affinity: int) -> int:
            calls.append((hwnd, affinity))
            return 1

    monkeypatch.setattr(so, "_should_exclude_uia_status_from_capture", lambda: False)
    monkeypatch.setattr(so, "_toplevel_hwnd", lambda _win: 42)
    monkeypatch.setattr(so.ctypes, "windll", type("W", (), {"user32": _User32()})())
    so._apply_exclude_from_capture(object())
    assert calls == []

    monkeypatch.setattr(so, "_should_exclude_uia_status_from_capture", lambda: True)
    so._apply_exclude_from_capture(object())
    assert calls == [(42, so._WDA_EXCLUDEFROMCAPTURE)]


def test_apply_click_through_sets_opaque_layered_alpha(monkeypatch) -> None:
    from src.recorder import stop_overlay as so

    styles: list[int] = []
    alphas: list[tuple[int, int, int, int]] = []

    class _User32:
        @staticmethod
        def GetWindowLongW(_hwnd: int, _index: int) -> int:
            return 0

        @staticmethod
        def SetWindowLongW(_hwnd: int, _index: int, style: int) -> int:
            styles.append(int(style))
            return style

        @staticmethod
        def SetLayeredWindowAttributes(
            hwnd: int, color: int, alpha: int, flags: int
        ) -> int:
            alphas.append((hwnd, color, alpha, flags))
            return 1

    monkeypatch.setattr(so, "_toplevel_hwnd", lambda _win: 7)
    monkeypatch.setattr(so.ctypes, "windll", type("W", (), {"user32": _User32()})())
    so._apply_click_through(object())
    assert styles and (styles[-1] & so._WS_EX_LAYERED)
    assert styles[-1] & so._WS_EX_TRANSPARENT
    assert alphas == [(7, 0, 255, so._LWA_ALPHA)]
