from __future__ import annotations

import json
from pathlib import Path

from computer_use_agent.core.session import utc_stamp


def write_thinking_record(
    thinking_dir: Path,
    image_name: str,
    thought: str,
    decision: dict,
    interrupted: bool = False,
) -> Path:
    path = thinking_dir / f"{utc_stamp()}-{image_name}.json"
    payload = {
        "timestamp": utc_stamp(),
        "image_name": image_name,
        "thought": thought,
        "decision": decision,
        "interrupted": interrupted,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path
