"""Helpers for treating hub editor text as one smart-mode goal."""

from __future__ import annotations

from typing import Literal

RunMode = Literal["script", "runtime", "queue", "smart"]

_MODE_TAB_LABELS = {
    "單一腳本": "script",
    "佇列執行": "queue",
    "智能模式": "smart",
}


def normalize_smart_goal(text: str) -> str:
    """Return the whole editor text as one goal (strip outer whitespace only)."""
    return (text or "").strip()


def resolve_hub_run_mode(
    *,
    selected_tab: str,
    script_has_steps: bool,
) -> RunMode:
    """
    Map hub tab + script content to an explicit run mode.

    Smart and queue tabs win; single-script with no executable lines becomes runtime.
    """
    label = (selected_tab or "").strip()
    if label == "智能模式":
        return "smart"
    if label == "佇列執行":
        return "queue"
    if script_has_steps:
        return "script"
    return "runtime"


def mode_tab_label(mode: RunMode) -> str:
    for label, value in _MODE_TAB_LABELS.items():
        if value == mode or (mode == "runtime" and value == "script"):
            return label
    return "單一腳本"
