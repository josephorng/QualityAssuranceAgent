from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from computer_use_agent.main import run


if __name__ == "__main__":
    path = run("smoke test scaffold", dry_run=True, max_cycles_override=1)
    print(f"Smoke run complete: {path}")
