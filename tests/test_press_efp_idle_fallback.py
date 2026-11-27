"""Pre-click idle ElementFromPoint fallback for recorded press name/value."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import (
    RecordingSession,
    _DeferredCaptureJob,
    _IDLE_EFP_PRESS_MAX_DIST_PX,
)
from src.recorder.window_snapshot import ClickWindowInfo


def test_resolve_press_efp_prefers_idle_when_live_differs() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    idle = {
        "name": "[檔案] 索引標籤",
        "value": "",
        "bbox": [1924, 59, 67, 37],
        "control_type": "button",
        "element_from_point_s": 0.01,
    }
    live = {
        "name": "返回",
        "value": "",
        "bbox": [1920, 59, 175, 56],
        "control_type": "list_item",
        "element_from_point_s": 0.07,
    }
    chosen = session._resolve_press_element_from_point(live, idle)
    assert chosen is not None
    assert chosen["name"] == "[檔案] 索引標籤"
    assert chosen["bbox"] == [1924, 59, 67, 37]


def test_resolve_press_efp_prefers_idle_when_findall_empty_live_differs() -> None:
    """Empty FindAll must not block near-press idle fallback."""
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    idle = {"name": "[檔案] 索引標籤", "value": "", "bbox": [1, 1, 10, 10]}
    live = {"name": "返回", "value": "", "bbox": [0, 0, 100, 50]}
    chosen = session._resolve_press_element_from_point(live, idle)
    assert chosen is not None
    assert chosen["name"] == "[檔案] 索引標籤"


def test_resolve_press_efp_keeps_live_when_names_match() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    idle = {"name": "儲存", "value": "", "bbox": [1, 1, 10, 10]}
    live = {
        "name": "儲存",
        "value": "",
        "bbox": [2, 2, 12, 12],
        "element_from_point_s": 0.02,
    }
    chosen = session._resolve_press_element_from_point(live, idle)
    assert chosen is not None
    assert chosen["bbox"] == [2, 2, 12, 12]
    assert chosen["element_from_point_s"] == 0.02


def test_resolve_press_efp_keeps_live_when_idle_empty() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    live = {"name": "返回", "value": ""}
    chosen = session._resolve_press_element_from_point(live, None)
    assert chosen is not None
    assert chosen["name"] == "返回"


def test_resolve_press_efp_idle_only_when_live_empty() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    idle = {"name": "[檔案] 索引標籤", "value": ""}
    chosen = session._resolve_press_element_from_point(
        {"element_from_point_s": 0.01},
        idle,
    )
    assert chosen is not None
    assert chosen["name"] == "[檔案] 索引標籤"


def test_idle_efp_latch_requires_near_press_xy(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    idle = {
        "name": "[檔案] 索引標籤",
        "value": "",
        "bbox": [10, 10, 20, 20],
        "control_type": "button",
    }
    session._idle_efp_payload = idle
    session._idle_efp_cursor_xy = (100, 100)
    with session._lock:
        near = session._idle_efp_payload_near_press_locked(100, 100)
        far = session._idle_efp_payload_near_press_locked(
            100 + _IDLE_EFP_PRESS_MAX_DIST_PX + 1,
            100,
        )
    assert near is not None
    assert near["name"] == "[檔案] 索引標籤"
    assert far is None


def test_idle_efp_latch_when_press_inside_leaf_bbox(tmp_path: Path) -> None:
    """Click can drift on a large button; latch if press stays in idle bbox."""
    session = RecordingSession(runs_root=tmp_path)
    idle = {
        "name": "儲存(S)",
        "value": "",
        "bbox": [2635, 613, 99, 28],
        "control_type": "button",
    }
    session._idle_efp_payload = idle
    # Probe near text center; press ~34px right still on the button.
    session._idle_efp_cursor_xy = (2660, 627)
    with session._lock:
        latched = session._idle_efp_payload_near_press_locked(2718, 631)
        outside = session._idle_efp_payload_near_press_locked(2800, 631)
    assert latched is not None
    assert latched["name"] == "儲存(S)"
    assert outside is None


def test_worker_idle_efp_stores_full_payload(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    session._accepting_input = True
    session._idle_efp_generation = 1
    click_win = ClickWindowInfo(
        hwnd=7,
        title="Word",
        process_name="WINWORD.EXE",
        left=0,
        top=0,
        width=100,
        height=100,
        is_maximized=False,
        class_name="OpusApp",
    )
    payload = {
        "element_from_point_s": 0.01,
        "name": "[檔案] 索引標籤",
        "value": "",
        "bbox": [30, 60, 67, 37],
        "control_type": "button",
    }
    with (
        patch(
            "src.recorder.uia_input_boxes.element_from_point_name_value",
            return_value=payload,
        ),
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=click_win,
        ),
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(40, 70),
                meta={"generation": 1},
            )
        )
    assert session._idle_efp_payload is not None
    assert session._idle_efp_payload["name"] == "[檔案] 索引標籤"
    assert session._idle_efp_payload["bbox"] == [30, 60, 67, 37]
    assert session._idle_efp_cursor_xy == (40, 70)
    log_text = (tmp_path / "record.log").read_text(encoding="utf-8")
    assert "element_from_point idle generation=1 xy=(40,70)" in log_text
    assert "name='[檔案] 索引標籤'" in log_text
    assert "control_type='button'" in log_text


def test_worker_pending_left_press_idle_fallback(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    click_win = ClickWindowInfo(
        hwnd=1,
        title="Word",
        process_name="WINWORD.EXE",
        left=0,
        top=0,
        width=100,
        height=100,
        is_maximized=False,
        class_name="OpusApp",
    )
    idle = {
        "name": "[檔案] 索引標籤",
        "value": "",
        "bbox": [30, 60, 67, 37],
        "control_type": "button",
        "element_from_point_s": 0.01,
    }
    live = {
        "name": "返回",
        "value": "",
        "bbox": [0, 50, 175, 56],
        "control_type": "list_item",
        "element_from_point_s": 0.08,
    }
    session._idle_efp_payload = idle
    session._idle_efp_cursor_xy = (40, 70)
    with (
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=click_win,
        ),
        patch(
            "src.recorder.capture._finalize_screenshot",
            return_value=("shot.jpeg", 1, (0, 0)),
        ),
        patch(
            "src.recorder.uia_input_boxes.element_from_point_name_value",
            return_value=live,
        ) as live_probe,
    ):
        session._capture_pending_left_press(tmp_path, 40, 70)
        assert session._left_press_efp_idle[1] is not None
        assert session._left_press_efp_idle[1]["name"] == "[檔案] 索引標籤"
        session._worker_pending_left_press(
            _DeferredCaptureJob(
                action="pending_left_press",
                cursor_xy=(40, 70),
                press_seq=1,
            )
        )
        live_probe.assert_not_called()
    recorded = session._left_press_efp[1]
    assert recorded is not None
    assert recorded["name"] == "[檔案] 索引標籤"
    assert recorded["bbox"] == [30, 60, 67, 37]


def test_worker_pending_left_press_keeps_live_when_idle_far(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    click_win = ClickWindowInfo(
        hwnd=1,
        title="Word",
        process_name="WINWORD.EXE",
        left=0,
        top=0,
        width=100,
        height=100,
        is_maximized=False,
        class_name="OpusApp",
    )
    idle = {
        "name": "[檔案] 索引標籤",
        "value": "",
        # Far from press (40, 70): outside bbox and >12px from probe cursor.
        "bbox": [200, 200, 20, 20],
    }
    live = {"name": "返回", "value": "", "bbox": [0, 50, 175, 56]}
    session._idle_efp_payload = idle
    session._idle_efp_cursor_xy = (210, 210)
    with (
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=click_win,
        ),
        patch(
            "src.recorder.capture._finalize_screenshot",
            return_value=("shot.jpeg", 1, (0, 0)),
        ),
        patch(
            "src.recorder.uia_input_boxes.element_from_point_name_value",
            return_value=live,
        ),
    ):
        session._capture_pending_left_press(tmp_path, 40, 70)
        assert session._left_press_efp_idle[1] is None
        session._worker_pending_left_press(
            _DeferredCaptureJob(
                action="pending_left_press",
                cursor_xy=(40, 70),
                press_seq=1,
            )
        )
    recorded = session._left_press_efp[1]
    assert recorded is not None
    assert recorded["name"] == "返回"


def test_resolve_press_efp_keeps_live_when_uia_snapshot_ignored() -> None:
    """Phase 2: FindAll disagree gate removed; live usable EFP is kept."""
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    live = {
        "name": "K15",
        "value": "",
        "bbox": [2674, 630, 72, 22],
        "control_type": "data_item",
    }
    snapshot = {
        "window": [1920, -1, 941, 667],
        "texts": [
            {
                "bbox": [2635, 613, 2734, 641],
                "text": "儲存(S)",
                "control_type": "button",
            }
        ],
    }
    chosen = session._resolve_press_element_from_point(
        live,
        None,
        cursor_xy=(2718, 631),
        uia_snapshot=snapshot,
    )
    assert chosen is not None
    assert chosen["name"] == "K15"


def test_worker_pending_right_press_idle_fallback(tmp_path: Path) -> None:
    session = RecordingSession(runs_root=tmp_path)
    session._run_dir = tmp_path
    click_win = ClickWindowInfo(
        hwnd=1,
        title="App",
        process_name="app.exe",
        left=0,
        top=0,
        width=100,
        height=100,
        is_maximized=False,
        class_name="App",
    )
    idle = {"name": "貼上", "value": "", "bbox": [10, 10, 20, 20]}
    live = {"name": "其他", "value": "", "bbox": [0, 0, 80, 80]}
    session._idle_efp_payload = idle
    session._idle_efp_cursor_xy = (15, 15)
    with (
        patch(
            "src.recorder.capture.resolve_click_window",
            return_value=click_win,
        ),
        patch(
            "src.recorder.capture._finalize_screenshot",
            return_value=("shot.jpeg", 1, (0, 0)),
        ),
        patch(
            "src.recorder.uia_input_boxes.element_from_point_name_value",
            return_value=live,
        ) as live_probe,
    ):
        session._capture_pending_right_press(tmp_path, 15, 15)
        assert session._right_press_efp_idle[1] is not None
        session._worker_pending_right_press(
            _DeferredCaptureJob(
                action="pending_right_press",
                cursor_xy=(15, 15),
                press_seq=1,
            )
        )
        live_probe.assert_not_called()
    recorded = session._right_press_efp[1]
    assert recorded is not None
    assert recorded["name"] == "貼上"
