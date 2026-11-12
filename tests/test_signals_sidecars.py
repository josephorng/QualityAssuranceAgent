from __future__ import annotations

import json
from pathlib import Path

from src.recorder.models import (
    hydrate_window_snapshot_debug,
    peel_signals_to_sidecars,
    signals_sidecar_path,
    write_event_json,
)


def test_peel_write_and_hydrate_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    (run_dir / "events").mkdir(parents=True)
    debug = {
        "windows_after": [{"title": "A"}],
        "signals_before": {"focused": {"name": "box", "value": "pre"}},
        "signals_after": {"focused": {"name": "box", "value": "post"}},
    }
    write_event_json(
        run_dir,
        {
            "index": 1,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "kind": "click",
            "window_snapshot_debug": debug,
        },
    )
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    slim = raw["window_snapshot_debug"]
    assert slim["windows_after"] == [{"title": "A"}]
    assert slim["signals_before"] == "signals/event_001_before.json"
    assert slim["signals_after"] == "signals/event_001_after.json"
    assert signals_sidecar_path(run_dir, 1, "before").is_file()
    assert signals_sidecar_path(run_dir, 1, "after").is_file()

    hydrated = hydrate_window_snapshot_debug(run_dir, 1, slim)
    assert hydrated is not None
    assert hydrated["signals_before"]["focused"]["value"] == "pre"
    assert hydrated["signals_after"]["focused"]["value"] == "post"
    # Caller payload was not mutated.
    assert isinstance(debug["signals_after"], dict)


def test_hydrate_keeps_inline_signals_for_legacy_recordings(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    debug = {
        "signals_after": {"focused": {"name": "legacy", "value": "inline"}},
    }
    hydrated = hydrate_window_snapshot_debug(run_dir, 3, debug)
    assert hydrated is not None
    assert hydrated["signals_after"]["focused"]["value"] == "inline"


def test_appeared_foreground_reads_signal_sidecar(tmp_path: Path) -> None:
    """A new window stays in appeared when its foreground lives only in a sidecar."""
    from src.recorder.models import RecordedEvent
    from src.recorder.orchestrator import _foregrounds_after_appeared
    from src.recorder.window_snapshot import limit_appeared_to_later_foreground

    run_dir = tmp_path / "rec"
    (run_dir / "events").mkdir(parents=True)
    dialog = {
        "class_name": "#32770",
        "title": "另存新檔",
        "process_name": "Notepad.exe",
    }
    write_event_json(
        run_dir,
        {
            "index": 7,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "kind": "click",
            "window_snapshot_debug": {
                "signals_after": {"foreground": dialog},
            },
        },
    )
    raw = json.loads(
        (run_dir / "events" / "event_007.json").read_text(encoding="utf-8")
    )
    event = RecordedEvent.from_dict(raw)
    debug = event.window_snapshot_debug
    assert isinstance(debug, dict)
    assert isinstance(debug["signals_after"], str)

    found = _foregrounds_after_appeared([event], 0, run_dir=run_dir)
    assert found == [dialog]
    limited = limit_appeared_to_later_foreground(
        {
            "appeared": [dialog],
            "disappeared": [
                {
                    "class_name": "Microsoft.UI.Content.PopupWindowSiteBridge",
                    "title": "快顯主機",
                    "process_name": "Notepad.exe",
                }
            ],
        },
        found,
    )
    assert limited["appeared"] == [dialog]
    assert isinstance(debug["signals_after"], str)


def test_write_event_analysis_keeps_appeared_when_later_foregrounds_omitted(
    tmp_path: Path,
) -> None:
    """Empty later_foregrounds must not wipe appeared if this step's after-FG exists."""
    from src.common.io_utils import read_json
    from src.recorder.models import RecordedEvent
    from src.recorder.orchestrator import _write_event_analysis

    run_dir = tmp_path / "rec"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir(parents=True)
    dialog = {
        "hwnd": 42,
        "title": "登入系統",
        "pid": 1,
        "left": 10,
        "top": 10,
        "width": 200,
        "height": 100,
        "is_minimized": False,
        "is_maximized": False,
        "class_name": "WindowsForms10.Window.8.app.0.x",
        "process_name": "WM7Ldr.exe",
    }
    write_event_json(
        run_dir,
        {
            "index": 1,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "kind": "double_click",
            "window_snapshot_debug": {
                "windows_before": [],
                "windows_after": [dialog],
                "signals_before": {
                    "foreground": {
                        "class_name": "Progman",
                        "title": "Program Manager",
                        "process_name": "explorer.exe",
                    }
                },
                "signals_after": {
                    "foreground": {
                        "class_name": dialog["class_name"],
                        "title": dialog["title"],
                        "process_name": dialog["process_name"],
                    }
                },
                "signal_kind": "double_click",
            },
        },
    )
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    event = RecordedEvent.from_dict(raw)
    assert isinstance(event.window_snapshot_debug, dict)
    assert isinstance(event.window_snapshot_debug["signals_after"], str)

    analysis_path = run_dir / "analysis" / "event_001.json"
    _write_event_analysis(
        analysis_path,
        event=event,
        result={"instruction": "click", "use_char_target": False},
        instruction="click",
        vision={},
        expected_outcome=None,
        use_expected_outcome=False,
        elapsed_since_previous=None,
        text_resolution=None,
        later_foregrounds=[],
        next_windows_before=None,
        run_dir=run_dir,
    )
    analysis = read_json(analysis_path, {})
    assert isinstance(analysis, dict)
    assert analysis.get("window_verify", {}).get("appeared") == [
        {
            "class_name": dialog["class_name"],
            "title": dialog["title"],
            "process_name": dialog["process_name"],
        }
    ]


def test_write_event_analysis_keeps_foreground_when_appeared_filtered_out(
    tmp_path: Path,
) -> None:
    """Unrelated later_foregrounds must not erase a real foreground change.

    omit_foreground_when_structural used to run before appear filtering, so a
    dropped appear left window_verify empty even though focus had moved.
    """
    from src.common.io_utils import read_json
    from src.recorder.models import RecordedEvent
    from src.recorder.orchestrator import _write_event_analysis

    run_dir = tmp_path / "rec"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "analysis").mkdir(parents=True)
    dialog = {
        "hwnd": 42,
        "title": "登入系統",
        "pid": 1,
        "left": 10,
        "top": 10,
        "width": 200,
        "height": 100,
        "is_minimized": False,
        "is_maximized": False,
        "class_name": "WindowsForms10.Window.8.app.0.x",
        "process_name": "WM7Ldr.exe",
    }
    after_fg = {
        "class_name": dialog["class_name"],
        "title": dialog["title"],
        "process_name": dialog["process_name"],
    }
    write_event_json(
        run_dir,
        {
            "index": 1,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "kind": "double_click",
            "window_snapshot_debug": {
                "windows_before": [],
                "windows_after": [dialog],
                "signals_before": {
                    "foreground": {
                        "class_name": "Progman",
                        "title": "Program Manager",
                        "process_name": "explorer.exe",
                    }
                },
                "signals_after": {"foreground": after_fg},
                "signal_kind": "double_click",
            },
        },
    )
    raw = json.loads((run_dir / "events" / "event_001.json").read_text(encoding="utf-8"))
    event = RecordedEvent.from_dict(raw)
    analysis_path = run_dir / "analysis" / "event_001.json"
    _write_event_analysis(
        analysis_path,
        event=event,
        result={"instruction": "click", "use_char_target": False},
        instruction="click",
        vision={},
        expected_outcome=None,
        use_expected_outcome=False,
        elapsed_since_previous=None,
        text_resolution=None,
        # Force appear filtering to drop the dialog (no matching FG identity).
        later_foregrounds=[
            {
                "class_name": "Notepad",
                "title": "Untitled",
                "process_name": "Notepad.exe",
            }
        ],
        next_windows_before=None,
        run_dir=run_dir,
    )
    analysis = read_json(analysis_path, {})
    assert isinstance(analysis, dict)
    verify = analysis.get("window_verify")
    assert isinstance(verify, dict)
    assert "appeared" not in verify
    assert verify.get("foreground") == after_fg


def test_peel_does_not_mutate_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    (run_dir / "signals").mkdir(parents=True)
    original = {"signals_after": {"foreground": {"title": "X"}}}
    slim = peel_signals_to_sidecars(run_dir, 2, original)
    assert isinstance(slim["signals_after"], str)
    assert isinstance(original["signals_after"], dict)
