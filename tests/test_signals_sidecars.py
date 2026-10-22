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


def test_peel_does_not_mutate_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "rec"
    (run_dir / "signals").mkdir(parents=True)
    original = {"signals_after": {"foreground": {"title": "X"}}}
    slim = peel_signals_to_sidecars(run_dir, 2, original)
    assert isinstance(slim["signals_after"], str)
    assert isinstance(original["signals_after"], dict)
