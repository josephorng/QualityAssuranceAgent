"""Test project RecordingSession + hotkey together (mirrors real app)."""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def main() -> int:
    from src.recorder.capture import RecordingSession
    from src.recorder.hotkey import RecordingHotkeyManager

    print("=== RecordingSession integration test ===")
    print(f"Admin: {is_admin()}")

    hotkey = RecordingHotkeyManager()
    session = RecordingSession(runs_root=ROOT / "recordings" / "_pynput_test")

    try:
        hotkey.register(lambda: print("hotkey fired"))
        print("Hotkey registered: OK")

        run_dir = session.start()
        print(f"Session started: {run_dir}")
        print(f"  mouse.running={session._mouse_listener.running if session._mouse_listener else None}")
        print(f"  keyboard.running={session._keyboard_listener.running if session._keyboard_listener else None}")

        time.sleep(1)
        stopped = session.stop()
        print(f"Session stopped: {stopped}")
        print("RESULT: OK — RecordingSession works in this environment")
        return 0
    except Exception as exc:
        print(f"RESULT: FAIL — {type(exc).__name__}: {exc}")
        return 1
    finally:
        hotkey.unregister()
        try:
            session.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
