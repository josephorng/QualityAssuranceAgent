"""Test whether multiple pynput keyboard listeners conflict."""
from __future__ import annotations

import time
from pynput import keyboard


def start_pair(label: str) -> tuple[bool, bool]:
    a = keyboard.Listener(on_press=lambda k: None)
    b = keyboard.Listener(on_press=lambda k: None)
    a.start()
    b.start()
    time.sleep(0.2)
    ar, br = a.running, b.running
    print(f"{label}: listener_a={ar}, listener_b={br}")
    a.stop()
    b.stop()
    return ar, br


print("=== duplicate keyboard listeners ===")
start_pair("two keyboard.Listener")

gh = keyboard.GlobalHotKeys({"<ctrl>+<shift>+r": lambda: None})
kl = keyboard.Listener(on_press=lambda k: None)
gh.start()
kl.start()
time.sleep(0.2)
print(f"GlobalHotKeys + Listener: gh={getattr(gh, 'running', '?')}, kl={kl.running}")
gh.stop()
kl.stop()
