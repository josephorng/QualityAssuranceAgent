"""Interactive prompt to choose which monitor Eye captures (or all screens)."""

from __future__ import annotations

import os
import sys
from typing import Any


def _physical_monitors() -> list[dict[str, Any]]:
    import mss

    with mss.mss() as sct:
        out: list[dict[str, Any]] = []
        for idx in range(1, len(sct.monitors)):
            mon = sct.monitors[idx]
            out.append(
                {
                    "index": idx,
                    "left": int(mon["left"]),
                    "top": int(mon["top"]),
                    "width": int(mon["width"]),
                    "height": int(mon["height"]),
                }
            )
        return out


def _position_labels(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["only physical display"]
    if count == 2:
        return ["left", "right"]
    if count == 3:
        return ["left", "middle", "right"]
    return [f"position {i + 1} from left" for i in range(count)]


def _all_screens_entry() -> dict[str, Any]:
    import mss

    with mss.mss() as sct:
        m = sct.monitors[0]
        return {
            "index": 0,
            "left": int(m["left"]),
            "top": int(m["top"]),
            "width": int(m["width"]),
            "height": int(m["height"]),
            "name": "all_screens",
        }


def prompt_eye_monitor_index() -> int:
    """
    Ask the user which mss monitor index to capture.

    Returns mss index: 0 = all screens (virtual desktop), 1+ = a single monitor.
    If stdin is not a TTY, returns int(os.environ['EYE_MONITOR_INDEX']) if set,
    otherwise 1.
    """
    if not sys.stdin.isatty():
        preset = os.environ.get("EYE_MONITOR_INDEX")
        if preset is not None and preset.strip() != "":
            try:
                return int(preset)
            except ValueError:
                print(
                    f"[master] Invalid EYE_MONITOR_INDEX={preset!r}; "
                    "falling back to monitor index 1."
                )
        print(
            "[master] stdin is not interactive; using monitor index 1 "
            "(set EYE_MONITOR_INDEX to override)."
        )
        return 1

    all_entry = _all_screens_entry()
    physical = _physical_monitors()
    physical_sorted = sorted(physical, key=lambda d: (d["left"], d["top"]))
    labels = _position_labels(len(physical_sorted))
    index_to_label: dict[int, str] = {}
    for mon, label in zip(physical_sorted, labels):
        index_to_label[mon["index"]] = label

    print()
    print("Which screen should Eye capture? (coordinates from screenshots use this region.)")
    print()
    print(
        f"  [0]  All screens combined  —  {all_entry['width']}×{all_entry['height']} "
        f"at virtual origin ({all_entry['left']}, {all_entry['top']})"
    )
    print()

    if not physical:
        print("  No separate physical monitors detected; using [0] all screens.")
        return 0

    print("  Physical monitors (mss index; order is left → right by virtual desktop position):")
    for mon in physical_sorted:
        idx = mon["index"]
        label = index_to_label.get(idx, "")
        print(
            f"  [{idx}]  {label:22}  {mon['width']}×{mon['height']}  "
            f"at ({mon['left']}, {mon['top']})"
        )
    print()

    valid = {0} | {m["index"] for m in physical}
    while True:
        raw = input("Enter monitor index (0 = all screens, or a number from the list above): ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if choice in valid:
            return choice
        print(f"Invalid choice {choice!r}. Valid: {sorted(valid)}")


def read_eye_monitor_index_from_env(default: int = 1) -> int:
    """Parse EYE_MONITOR_INDEX for child processes (e.g. Eye server)."""
    raw = os.environ.get("EYE_MONITOR_INDEX")
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# if __name__ == "__main__":
#     print(prompt_eye_monitor_index())
#     print(read_eye_monitor_index_from_env())