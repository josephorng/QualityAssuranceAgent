"""Diagnose pynput global hook startup on Windows."""
from __future__ import annotations

import ctypes
import os
import sys
import time


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def check_listener(label: str, listener) -> bool:
    listener.start()
    time.sleep(0.3)
    running = listener.running
    print(f"  [{label}] running={running}")
    if not running:
        print(f"  [{label}] FAILED to start hook")
    listener.stop()
    return running


def main() -> int:
    print("=== pynput hook diagnostic ===")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"Admin: {is_admin()}")
    print(f"PID: {os.getpid()}")
    print()

    try:
        from pynput import keyboard, mouse
        import pynput

        print(f"pynput: {getattr(pynput, '__version__', 'unknown')}")
    except ImportError as exc:
        print(f"IMPORT ERROR: {exc}")
        return 1

    results: dict[str, bool] = {}

    print("Testing mouse.Listener ...")
    try:
        ml = mouse.Listener(on_click=lambda *a: None)
        results["mouse"] = check_listener("mouse", ml)
    except Exception as exc:
        print(f"  [mouse] EXCEPTION: {type(exc).__name__}: {exc}")
        results["mouse"] = False

    print("Testing keyboard.Listener ...")
    try:
        kl = keyboard.Listener(on_press=lambda k: None)
        results["keyboard"] = check_listener("keyboard", kl)
    except Exception as exc:
        print(f"  [keyboard] EXCEPTION: {type(exc).__name__}: {exc}")
        results["keyboard"] = False

    print("Testing keyboard.GlobalHotKeys ...")
    try:
        gh = keyboard.GlobalHotKeys({"<ctrl>+<shift>+r": lambda: None})
        gh.start()
        time.sleep(0.3)
        running = gh.running if hasattr(gh, "running") else True
        print(f"  [GlobalHotKeys] running={running}")
        gh.stop()
        results["hotkey"] = running
    except Exception as exc:
        print(f"  [GlobalHotKeys] EXCEPTION: {type(exc).__name__}: {exc}")
        results["hotkey"] = False

    print()
    print("=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")

    all_ok = all(results.values())
    if all_ok:
        print("\nAll hooks started successfully.")
        if not is_admin():
            print("Admin is NOT required on this machine (tested without admin).")
    else:
        print("\nSome hooks failed.")
        if not is_admin():
            print("Retry as Administrator to see if elevation fixes it.")
        else:
            print("Already running as admin — likely AV/security policy or environment block.")

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
