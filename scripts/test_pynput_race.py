"""Check if listener.running is briefly False right after start() (race)."""
from __future__ import annotations

import time
from pynput import keyboard, mouse


def probe(label, listener):
    listener.start()
    samples = []
    for ms in (0, 10, 50, 100, 200, 300):
        if ms:
            time.sleep(ms / 1000)
        samples.append((ms, listener.running))
    listener.stop()
    print(f"{label}: {samples}")
    return samples


print("=== running flag timing ===")
probe("mouse", mouse.Listener(on_click=lambda *a: None))
probe("keyboard", keyboard.Listener(on_press=lambda k: None))
