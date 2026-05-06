from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

# Allow running this file directly (python test_live/test_windows_chrome_enter_live.py)
# by ensuring the repository root is on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cua_mcp import tools as cua_tools


def _live_enabled() -> bool:
    return os.getenv("RUN_LIVE_DESKTOP_TESTS", "").strip().lower() in {"1", "true", "yes"}


def test_windows_chrome_enter_sequence_live() -> None:
    """
    Live desktop test: actually opens Start, types Chrome, and presses Enter.

    Safety gate: set RUN_LIVE_DESKTOP_TESTS=1 to enable.
    """
    if not _live_enabled():
        pytest.skip("Live desktop test disabled. Set RUN_LIVE_DESKTOP_TESTS=1 to run it.")

    # Give operator a brief moment to focus the desktop before keys are sent.
    time.sleep(2.0)

    cua_tools.press_key(
        key="win",
        instruction="Press the Windows key to open the Start Menu search bar.",
    )
    time.sleep(0.25)

    cua_tools.type_text(
        text="Chrome",
        instruction='Type "Chrome" into the search bar.',
    )
    time.sleep(0.25)

    cua_tools.press_key(
        key="Enter",
        instruction="Press Enter to launch Google Chrome.",
    )

if __name__ == "__main__":
    import os
    os.environ["RUN_LIVE_DESKTOP_TESTS"] = "true"
    test_windows_chrome_enter_sequence_live()