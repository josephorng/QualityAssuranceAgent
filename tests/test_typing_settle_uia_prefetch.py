"""Weak-EFP OCR demote helpers (typing FindAll prefetch removed in Phase 2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import RecordingSession, _DeferredCaptureJob
from src.recorder.vision_context import (
    VISION_PATH_WEAK_EFP_OCR_DISAGREE_REASON,
    _maybe_demote_uia_path_for_weak_efp_ocr_disagree,
    _press_efp_is_weak_hub,
)


def test_schedule_pre_key_settle_enqueues_without_findall(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._accepting_input = True
    session._run_dir = tmp_path
    session._pending_text_meta = {"index": 3}
    queued: list[_DeferredCaptureJob] = []

    def _enqueue(job: _DeferredCaptureJob) -> None:
        queued.append(job)

    with (
        patch.object(session, "_enqueue", side_effect=_enqueue),
        patch("src.recorder.capture._PRE_KEY_SETTLE_S", 0.01),
    ):
        session._schedule_pre_key_settle()
        assert session._pending_pre_key_timer is not None
        session._pending_pre_key_timer.cancel()
        session._on_pre_key_settle_timer()

    assert len(queued) == 1
    assert queued[0].action == "settle_pre_key"
    assert queued[0].meta is None


def test_settle_pre_key_job_skips_findall_prefetch(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._accepting_input = True
    session._pending_text_meta = {"index": 3}
    with (
        patch.object(session, "_refresh_pending_pre_key") as refresh,
        patch(
            "src.recorder.uia_input_boxes.snapshot_uia_screen",
        ) as snap,
    ):
        session._process_deferred_capture_job(
            _DeferredCaptureJob(action="settle_pre_key")
        )
    refresh.assert_called_once()
    snap.assert_not_called()


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


def test_demote_skips_when_ocr_soft_agrees_with_findall_list_item_name() -> None:
    """Short OCR hub vs longer Search list-item Name must keep UIA."""
    timing = {"vision_path": "uia", "recording_uia_sufficient": True}
    assert not _maybe_demote_uia_path_for_weak_efp_ocr_disagree(
        vision_timing=timing,
        element_from_point={"name": "CoreInput", "value": ""},
        uia_primary_text="Google Chrome, 應用程式, 按右鍵以檢視選項",
        yolo_primary_text="Google Chrome",
    )
    assert timing["vision_path"] == "uia"
    assert timing["recording_uia_sufficient"] is True
