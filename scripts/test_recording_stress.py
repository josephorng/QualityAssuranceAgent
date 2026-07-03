"""Stress: rapid start/stop and leftover listener cleanup."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recorder.capture import RecordingSession


def try_start(label: str, session: RecordingSession) -> bool:
    try:
        run_dir = session.start()
        ok = (
            session._mouse_listener is not None
            and session._mouse_listener.running
            and session._keyboard_listener is not None
            and session._keyboard_listener.running
        )
        print(f"{label}: start OK running={ok} dir={run_dir.name}")
        session.stop()
        return ok
    except Exception as exc:
        print(f"{label}: FAIL {type(exc).__name__}: {exc}")
        try:
            session.stop()
        except Exception:
            pass
        return False


runs = ROOT / "runs" / "_pynput_test"
session = RecordingSession(runs_root=runs)

print("=== rapid start/stop ===")
results = [try_start(f"cycle-{i}", session) for i in range(5)]
print(f"passed {sum(results)}/5")

print("\n=== start without stop, then second start ===")
s1 = RecordingSession(runs_root=runs)
d1 = s1.start()
print(f"first: {d1.name}")
try:
    s1.start()
    print("second start: unexpected success")
except RuntimeError as exc:
    print(f"second start blocked as expected: {exc}")
s1.stop()
