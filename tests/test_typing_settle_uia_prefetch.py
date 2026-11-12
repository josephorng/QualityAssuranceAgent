"""Typing-pause settle UIA prefetch + weak-EFP OCR demote."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import RecordingSession, _DeferredCaptureJob
from src.recorder.vision_context import (
    VISION_PATH_WEAK_EFP_OCR_DISAGREE_REASON,
    _maybe_demote_uia_path_for_weak_efp_ocr_disagree,
    _press_efp_is_weak_hub,
)
from src.recorder.window_snapshot import ClickWindowInfo


def _search_flyout(hwnd: int = 131762) -> ClickWindowInfo:
    return ClickWindowInfo(
        hwnd=hwnd,
        title="搜尋",
        process_name="SearchHost.exe",
        left=500,
        top=140,
        width=350,
        height=800,
        is_maximized=False,
        is_flyout=True,
        class_name="Windows.UI.Core.CoreWindow",
    )


def _snap_with_label(label: str) -> dict:
    return {
        "window": [500, 140, 350, 800],
        "inputs": [],
        "scrollbars": [],
        "texts": [
            {
                "bbox": [550, 260, 859, 310],
                "text": label,
                "control_type": "button",
            }
        ],
    }


def test_invalidate_typing_host_clears_settle_and_leds_red() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._accepting_input = True
    session._last_settle_uia = {
        "hwnd": 131762,
        "class_name": "Windows.UI.Core.CoreWindow",
        "cache_kind": "settle",
        **_snap_with_label("變更螢幕保護裝置"),
    }
    session._last_settle_uia_by_hwnd = {
        131762: dict(session._last_settle_uia),
        20: {
            "hwnd": 20,
            "class_name": "Shell_TrayWnd",
            "window": [0, 1000, 100, 40],
            "inputs": [],
            "scrollbars": [],
            "texts": [],
        },
    }
    session._last_settle_uia_after_index = 2

    with patch.object(session, "_foreground_hwnd", return_value=131762):
        gen = session._invalidate_settle_uia_for_typing_host()

    assert gen == 1
    assert session._typing_uia_generation == 1
    assert session._last_settle_uia is None
    assert 131762 not in session._last_settle_uia_by_hwnd
    assert 20 in session._last_settle_uia_by_hwnd  # shell peer kept
    assert session._last_settle_uia_after_index == 2
    assert statuses[-1]["state"] == "red"


def test_typing_prefetch_publishes_attachable_while_text_pending(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)
    session.set_on_uia_status(statuses.append)
    session._accepting_input = True
    session._run_dir = tmp_path
    session._pending_text_meta = {"index": 3, "cursor_xy": (649, 297)}
    session._last_settle_uia_after_index = 2
    session._events = []  # previous_index → 0; publish will set after_index=0
    # Pretend an event exists so generation matches a prior step.
    from src.recorder.models import RecordedEvent

    session._events = [
        RecordedEvent.from_dict(
            {
                "index": 2,
                "timestamp_utc": "2026-11-12T00:00:00+00:00",
                "kind": "scroll",
            }
        )
    ]
    session._last_settle_uia_after_index = 2
    session._typing_uia_generation = 5
    flyout = _search_flyout(131762)

    def _snap(hwnd: int) -> dict:
        assert hwnd == 131762
        return _snap_with_label("Google Chrome")

    with (
        patch.object(session, "_foreground_hwnd", return_value=131762),
        patch("src.recorder.uia_input_boxes.snapshot_uia_screen", side_effect=_snap),
        patch(
            "src.recorder.capture._window_class_name",
            return_value="Windows.UI.Core.CoreWindow",
        ),
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=flyout,
        ),
    ):
        session._idle_efp_cursor_xy = (649, 297)
        session._prefetch_typing_settle_uia(typing_uia_generation=5)

    assert session._last_settle_uia is not None
    assert session._last_settle_uia["cache_kind"] == "typing"
    assert session._last_settle_uia["texts"][0]["text"] == "Google Chrome"
    assert session._last_settle_uia_after_index == 2
    assert session._pending_text_meta is not None
    attached = session._cached_uia_before(flyout, cursor_xy=(649, 297))
    assert attached is not None
    assert attached["texts"][0]["text"] == "Google Chrome"
    assert "yellow" in [row["state"] for row in statuses]
    assert statuses[-1]["state"] in {"green", "red"}


def test_typing_prefetch_discards_stale_generation(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._accepting_input = True
    session._run_dir = tmp_path
    session._pending_text_meta = {"index": 3}
    session._typing_uia_generation = 9
    session._last_settle_uia = None

    with (
        patch.object(session, "_foreground_hwnd", return_value=131762),
        patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
            return_value=_snap_with_label("Google Chrome"),
        ),
        patch(
            "src.recorder.capture._window_class_name",
            return_value="Windows.UI.Core.CoreWindow",
        ),
    ):
        session._prefetch_typing_settle_uia(typing_uia_generation=8)

    assert session._last_settle_uia is None


def test_schedule_pre_key_settle_enqueues_typing_generation(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._accepting_input = True
    session._run_dir = tmp_path
    session._pending_text_meta = {"index": 3}
    session._last_settle_uia = {
        "hwnd": 131762,
        **_snap_with_label("Recent"),
    }
    queued: list[_DeferredCaptureJob] = []

    def _enqueue(job: _DeferredCaptureJob) -> None:
        queued.append(job)

    with (
        patch.object(session, "_foreground_hwnd", return_value=131762),
        patch.object(session, "_enqueue", side_effect=_enqueue),
        patch("src.recorder.capture._PRE_KEY_SETTLE_S", 0.01),
    ):
        session._schedule_pre_key_settle()
        assert session._last_settle_uia is None
        assert session._typing_uia_generation == 1
        # Fire the timer synchronously.
        assert session._pending_pre_key_timer is not None
        session._pending_pre_key_timer.cancel()
        session._on_pre_key_settle_timer()

    assert len(queued) == 1
    assert queued[0].action == "settle_pre_key"
    assert queued[0].meta == {"typing_uia_generation": 1}


def test_press_efp_coreinput_is_weak_hub() -> None:
    assert _press_efp_is_weak_hub(
        {"name": "CoreInput", "value": "", "bbox": [0, 0, 100, 100]}
    )
    assert not _press_efp_is_weak_hub(
        {"name": "Google Chrome", "value": "", "bbox": [0, 0, 40, 20]}
    )


def test_demote_uia_when_weak_efp_and_ocr_disagree() -> None:
    timing = {"vision_path": "uia", "recording_uia_sufficient": True}
    demoted = _maybe_demote_uia_path_for_weak_efp_ocr_disagree(
        vision_timing=timing,
        element_from_point={"name": "CoreInput", "value": ""},
        uia_primary_text="變更螢幕保護裝置",
        yolo_primary_text="Google Chrome",
    )
    assert demoted is True
    assert timing["vision_path"] == "yolo_ocr"
    assert timing["vision_path_reason"] == VISION_PATH_WEAK_EFP_OCR_DISAGREE_REASON
    assert timing["recording_uia_sufficient"] is False


def test_demote_skips_when_labels_agree() -> None:
    timing = {"vision_path": "uia"}
    assert not _maybe_demote_uia_path_for_weak_efp_ocr_disagree(
        vision_timing=timing,
        element_from_point={"name": "CoreInput", "value": ""},
        uia_primary_text="Google Chrome",
        yolo_primary_text="Google Chrome",
    )
    assert timing["vision_path"] == "uia"
