"""Idle ElementFromPoint LED status (Phase 2: no recording FindAll)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.recorder.capture import (
    RecordingSession,
    _DeferredCaptureJob,
    _IDLE_EFP_DEBOUNCE_GREEN_S,
    _IDLE_EFP_DEBOUNCE_S,
)


def test_uia_path_ready_from_usable_efp_data_item() -> None:
    """Excel cell: usable EFP data_item greens without FindAll cell texts."""
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._idle_efp_payload = {
        "name": "G8",
        "value": "",
        "control_type": "data_item",
        "bbox": [466, 477, 72, 22],
    }
    assert session._uia_path_ready_at_cursor((500, 488)) is True


def test_uia_path_ready_from_usable_efp_button() -> None:
    """File-tab style labeled button greens from EFP alone."""
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._idle_efp_payload = {
        "name": "[檔案] 索引標籤",
        "value": "",
        "control_type": "button",
        "bbox": [40, 10, 60, 24],
    }
    assert session._uia_path_ready_at_cursor((50, 20)) is True


def test_uia_path_ready_from_position_thumb_efp_alone() -> None:
    """WinForms Position thumb + bbox opens the UIA hub (LED green)."""
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._idle_efp_payload = {
        "name": "Position",
        "value": "",
        "control_type": "data_item",
        "bbox": [1900, 400, 17, 200],
    }
    assert session._uia_path_ready_at_cursor((1908, 500)) is True
    session._idle_efp_payload = {
        "name": "",
        "value": "",
        "control_type": "button",
        "bbox": [40, 10, 60, 24],
    }
    assert session._uia_path_ready_at_cursor((50, 20)) is False


def test_led_and_press_efp_hub_gate_agree() -> None:
    """LED readiness matches analysis _press_efp_opens_uia_hub."""
    from src.recorder.vision_context import _press_efp_opens_uia_hub

    cases: list[tuple[dict[str, object] | None, bool]] = [
        (
            {
                "name": "G8",
                "value": "",
                "control_type": "data_item",
                "bbox": [466, 477, 72, 22],
            },
            True,
        ),
        (
            {
                "name": "Position",
                "value": "",
                "bbox": [1900, 400, 17, 200],
            },
            True,
        ),
        (
            {
                "name": "",
                "value": "",
                "control_type": "button",
                "bbox": [40, 10, 60, 24],
            },
            False,
        ),
        (None, False),
    ]
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    for payload, expected in cases:
        session._idle_efp_payload = payload
        assert _press_efp_opens_uia_hub(payload) is expected
        assert session._uia_path_ready_at_cursor((1, 1), efp_payload=payload) is expected


def test_emit_uia_path_readiness_greens_usable_efp() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._idle_efp_cursor_xy = (50, 20)
    session._idle_efp_probed_name = "[檔案] 索引標籤"
    session._idle_efp_probed_value = ""
    session._idle_efp_probed_control_type = "button"
    session._idle_efp_payload = {
        "name": "[檔案] 索引標籤",
        "value": "",
        "control_type": "button",
        "bbox": [40, 10, 60, 24],
        "parents": [
            {"control_type": "tool_bar", "name": "功能區"},
        ],
    }
    session._emit_uia_path_readiness()
    assert statuses[-1]["state"] == "green"
    assert statuses[-1]["efp_name"] == "[檔案] 索引標籤"
    assert statuses[-1]["efp_parent"] == "功能區"
    assert "findall_label" not in statuses[-1]


def test_emit_uia_path_readiness_red_when_empty() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._idle_efp_cursor_xy = (1, 1)
    session._idle_efp_payload = None
    session._emit_uia_path_readiness()
    assert statuses[-1]["state"] == "red"


def test_led_efp_parent_uses_parents_chain() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    parent = session._led_efp_parent(
        efp_payload={
            "name": "文件",
            "parents": [
                {"control_type": "pane", "name": ""},
                {"control_type": "tree", "name": "檔案", "automation_id": "ProperTreeHost"},
            ],
        }
    )
    assert parent == "檔案"
    legacy = session._led_efp_parent(
        efp_payload={
            "name": "文件",
            "parent": {
                "control_type": "tree",
                "name": "舊父層",
            },
        }
    )
    assert legacy == "舊父層"


def test_worker_idle_efp_greens_usable_leaf(tmp_path: Path) -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=tmp_path)  # type: ignore[arg-type]
    session._accepting_input = True
    session._run_dir = tmp_path
    session.set_on_uia_status(statuses.append)
    session._idle_efp_generation = 1

    with patch(
        "src.recorder.uia_probe.element_from_point_name_value",
        return_value={
            "name": "關閉",
            "value": "",
            "control_type": "button",
            "bbox": [0, 0, 20, 20],
            "parents": [{"control_type": "title_bar", "name": "視窗"}],
            "element_from_point_s": 0.01,
        },
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(10, 10),
                meta={"generation": 1},
            )
        )
    assert statuses[-1]["state"] == "green"
    assert statuses[-1]["efp_name"] == "關閉"
    assert statuses[-1]["efp_parent"] == "視窗"
    assert statuses[-1]["bg"] == []


def test_worker_idle_efp_no_hover_findall(tmp_path: Path) -> None:
    """Idle EFP must not call snapshot_uia_hover_subtree / snapshot_uia_screen."""
    session = RecordingSession(runs_root=tmp_path)  # type: ignore[arg-type]
    session._accepting_input = True
    session._run_dir = tmp_path
    session._idle_efp_generation = 1

    with (
        patch(
            "src.recorder.uia_probe.element_from_point_name_value",
            return_value={
                "name": "Launch 檔案總管",
                "value": "",
                "control_type": "hyperlink",
                "bbox": [10, 10, 80, 20],
                "element_from_point_s": 0.0,
            },
        ),
        patch(
            "src.recorder.uia_probe.snapshot_uia_hover_subtree",
        ) as hover,
        patch(
            "src.recorder.uia_probe.snapshot_uia_screen",
        ) as settle,
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(20, 15),
                meta={"generation": 1},
            )
        )
    hover.assert_not_called()
    settle.assert_not_called()


def test_worker_idle_efp_ignores_stale_generation() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._accepting_input = True
    session.set_on_uia_status(statuses.append)
    session._idle_efp_generation = 9

    with patch(
        "src.recorder.uia_probe.element_from_point_name_value",
        return_value={
            "name": "Stale",
            "value": "X",
            "control_type": "text",
            "bbox": [0, 0, 1, 1],
            "element_from_point_s": 0.0,
        },
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(1, 1),
                meta={"generation": 3},
            )
        )
    assert statuses == []


def test_worker_idle_efp_coalesces_to_newest() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._accepting_input = True
    session.set_on_uia_status(statuses.append)
    session._idle_efp_generation = 2
    calls: list[tuple[int, int]] = []

    def _probe(x: int, y: int):
        calls.append((int(x), int(y)))
        return {
            "name": f"N{x}",
            "value": "",
            "control_type": "button",
            "bbox": [0, 0, 10, 10],
            "element_from_point_s": 0.0,
        }

    session._enqueue(
        _DeferredCaptureJob(
            action="idle_efp",
            cursor_xy=(40, 40),
            meta={"generation": 2},
        )
    )
    with patch(
        "src.recorder.uia_probe.element_from_point_name_value",
        side_effect=_probe,
    ):
        session._worker_idle_efp(
            _DeferredCaptureJob(
                action="idle_efp",
                cursor_xy=(10, 10),
                meta={"generation": 1},
            )
        )
    assert calls == [(40, 40)]
    assert statuses[-1]["efp_name"] == "N40"
    assert statuses[-1]["state"] == "green"


def test_schedule_idle_efp_skips_inside_same_leaf_bbox() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._accepting_input = True
    session._idle_efp_payload = {
        "name": "G8",
        "value": "",
        "control_type": "data_item",
        "bbox": [100, 100, 50, 20],
    }
    session._schedule_idle_efp(110, 110)
    assert session._idle_efp_timer is None
    session._schedule_idle_efp(200, 200)
    assert session._idle_efp_timer is not None
    session._idle_efp_timer.cancel()


def test_schedule_idle_efp_skips_inside_position_thumb_bbox() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._accepting_input = True
    session._idle_efp_payload = {
        "name": "Position",
        "value": "",
        "bbox": [1900, 400, 17, 200],
    }
    session._schedule_idle_efp(1908, 500)
    assert session._idle_efp_timer is None
    session._schedule_idle_efp(10, 10)
    assert session._idle_efp_timer is not None
    session._idle_efp_timer.cancel()


def test_schedule_idle_efp_force_reprobes_inside_leaf() -> None:
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session._accepting_input = True
    session._idle_efp_payload = {
        "name": "G8",
        "value": "",
        "control_type": "data_item",
        "bbox": [100, 100, 50, 20],
    }
    session._schedule_idle_efp(110, 110, force=True)
    assert session._idle_efp_timer is not None
    session._idle_efp_timer.cancel()


def test_idle_efp_debounce_shorter_when_green() -> None:
    assert _IDLE_EFP_DEBOUNCE_GREEN_S < _IDLE_EFP_DEBOUNCE_S


def test_emit_uia_status_publishes_efp_parent() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)
    session._emit_uia_status(
        efp_name="文件",
        efp_value="",
        efp_control_type="list_item",
        efp_parent="ProperTreeHost",
        efp_pending=False,
    )
    assert statuses[-1]["efp_parent"] == "ProperTreeHost"
    assert statuses[-1]["efp_name"] == "文件"
    assert "findall_label" not in statuses[-1]


def test_emit_uia_status_tracks_bg_activities() -> None:
    statuses: list[dict] = []
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    session.set_on_uia_status(statuses.append)

    session._emit_uia_status(bg_add="EFP")
    assert statuses[-1]["bg"] == ["EFP"]
    session._emit_uia_status(bg_add="YOLO")
    assert statuses[-1]["bg"] == ["EFP", "YOLO"]
    session._emit_uia_status(bg_remove="EFP")
    assert statuses[-1]["bg"] == ["YOLO"]
    before = len(statuses)
    session._emit_uia_status(bg_remove="EFP")
    assert len(statuses) == before
    session._emit_uia_status(bg_clear=True)
    assert statuses[-1]["bg"] == []


def test_recording_uia_status_overlay_efp_only() -> None:
    from src.recorder import stop_overlay as so

    class _FakeLabel:
        def __init__(self) -> None:
            self.text = "—"

        def configure(self, **kwargs: object) -> None:
            if "text" in kwargs:
                self.text = str(kwargs["text"])

        def cget(self, key: str) -> str:
            assert key == "text"
            return self.text

        def pack(self, **_kwargs: object) -> None:
            return None

        def pack_forget(self) -> None:
            return None

    overlay = so.RecordingUiaStatusOverlay.__new__(so.RecordingUiaStatusOverlay)
    overlay._destroyed = False
    overlay._label = _FakeLabel()
    overlay._bg_label = _FakeLabel()
    overlay._bg_text = ""
    overlay._state = "red"
    overlay._led = None
    overlay._window = None
    overlay._pulse_after_id = None
    overlay.set_state = lambda _state: None  # type: ignore[method-assign]
    overlay.pulse = lambda _kind: None  # type: ignore[method-assign]
    overlay.ensure_visible = lambda force=False: None  # type: ignore[method-assign]

    overlay.apply_status(
        {
            "state": "green",
            "efp_name": "文件",
            "efp_value": "",
            "efp_control_type": "list_item",
            "efp_parent": "ProperTreeHost",
            "efp_pending": False,
            "bg": [],
        }
    )
    assert overlay._label.cget("text") == (
        "EFP list_item · 文件 · ⌂ProperTreeHost"
    )
    overlay.apply_status(
        {
            "state": "red",
            "efp_name": "",
            "efp_value": "",
            "efp_pending": True,
            "bg": ["EFP"],
        }
    )
    assert overlay._label.cget("text") == "EFP …"
    assert overlay._bg_label.cget("text") == "EFP"
    overlay.apply_status(
        {
            "state": "red",
            "efp_name": "Position",
            "efp_value": "",
            "efp_control_type": "data_item",
            "efp_scroll": "V37%",
            "efp_pending": False,
            "bg": [],
        }
    )
    assert overlay._label.cget("text") == (
        "EFP data_item · Position · V37%"
    )
    overlay.apply_status(
        {
            "state": "red",
            "efp_name": "",
            "efp_value": "",
            "efp_pending": False,
            "bg": [],
        }
    )
    assert overlay._label.cget("text") == "EFP (empty)"


def test_format_scroll_percent_label() -> None:
    from src.recorder.verify_signals import format_scroll_percent_label

    assert format_scroll_percent_label(37.2) == "37%"
    assert format_scroll_percent_label({"vertical": 37.2, "horizontal": 12.6}) == (
        "V37% H13%"
    )
    assert format_scroll_percent_label({"horizontal": 0}) == "H0%"
    assert format_scroll_percent_label(None) == ""
    assert format_scroll_percent_label({}) == ""


def test_efp_payload_looks_like_scrollbar() -> None:
    from src.recorder.uia_probe import efp_payload_looks_like_scrollbar

    assert efp_payload_looks_like_scrollbar(
        {"name": "Position", "control_type": "data_item"}
    )
    assert efp_payload_looks_like_scrollbar({"name": "Line down"})
    assert efp_payload_looks_like_scrollbar({"control_type": "thumb"})
    assert efp_payload_looks_like_scrollbar({"control_type": "scrollbar"})
    assert efp_payload_looks_like_scrollbar(
        {"parents": [{"automation_id": "VerticalScrollBar"}]}
    )
    assert not efp_payload_looks_like_scrollbar(
        {"name": "關閉", "control_type": "button"}
    )
    assert not efp_payload_looks_like_scrollbar(
        {"name": "G8", "control_type": "data_item"}
    )


def test_window_step_skips_settle_findall() -> None:
    """Window step must not call snapshot_uia_screen for settle FindAll."""
    session = RecordingSession(runs_root=None)  # type: ignore[arg-type]
    item = type(
        "Q",
        (),
        {
            "event_index": 1,
            "kind": "click",
            "windows_before": None,
            "signals_before": None,
            "window_after_sealed": False,
            "cursor_xy": (1, 1),
            "click_window": None,
            "end_click_window": None,
        },
    )()
    with (
        patch.object(session, "_resolve_window_change", return_value=(None, None, None)),
        patch.object(session, "_sample_debug_focus_for_item", return_value=(None, None)),
        patch.object(session, "_sleep_remaining_window_settle"),
        patch.object(session, "_notify_pending_text_input"),
        patch.object(session, "_release_drag_uia_notify"),
        patch("src.recorder.uia_probe.snapshot_uia_screen") as snap,
    ):
        session._window_step_body(item)  # type: ignore[arg-type]
    snap.assert_not_called()
