"""Interactive prompt to choose which monitor Eye captures (or all screens)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EyeMonitorChoice:
    """One selectable mss monitor row for UI or CLI display."""

    index: int
    title: str
    detail: str


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


def list_eye_monitor_choices() -> list[EyeMonitorChoice]:
    """
    Build monitor rows in display order (all screens first, then physical left → right).

    Returns at least the all-screens row; if no physical monitors, only that row is returned.
    """
    all_entry = _all_screens_entry()
    physical = _physical_monitors()
    physical_sorted = sorted(physical, key=lambda d: (d["left"], d["top"]))
    primary_idx = _primary_monitor_index(physical)
    labels = _position_labels(len(physical_sorted))
    index_to_label: dict[int, str] = {}
    for mon, label in zip(physical_sorted, labels):
        index_to_label[mon["index"]] = label

    rows: list[EyeMonitorChoice] = [
        EyeMonitorChoice(
            index=0,
            title="All screens combined",
            detail=f"{all_entry['width']}×{all_entry['height']} at ({all_entry['left']}, {all_entry['top']})",
        )
    ]
    if not physical:
        return rows

    for mon in physical_sorted:
        idx = mon["index"]
        label = index_to_label.get(idx, "")
        primary_note = " [main]" if idx == primary_idx else ""
        title = f"Monitor {idx}: {label}{primary_note}".strip()
        detail = f"{mon['width']}×{mon['height']} at ({mon['left']}, {mon['top']})"
        rows.append(EyeMonitorChoice(index=idx, title=title, detail=detail))
    return rows


def _primary_monitor_index(physical: list[dict[str, Any]]) -> int | None:
    """
    Best-effort detection of the OS main display.

    In typical virtual-desktop layouts, the primary display is anchored at (0, 0).
    """
    candidates = [m["index"] for m in physical if m["left"] == 0 and m["top"] == 0]
    if len(candidates) == 1:
        return int(candidates[0])
    return None


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

    rows = list_eye_monitor_choices()
    physical = _physical_monitors()
    print()
    print("Which screen should Eye capture? (coordinates from screenshots use this region.)")
    print()
    for row in rows:
        print(f"  [{row.index}]  {row.title}  —  {row.detail}")
    print()

    if not physical:
        print("  No separate physical monitors detected; using [0] all screens.")
        return 0

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