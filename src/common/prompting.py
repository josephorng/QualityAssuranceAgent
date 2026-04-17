from __future__ import annotations

import json
from pathlib import Path

from src.common.settings import ROOT_DIR


def load_prompts() -> dict:
    path = ROOT_DIR / "prompts.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_prompt(name: str) -> str:
    prompts = load_prompts()
    variants = prompts.get(name, [])
    if not variants:
        return ""
    return variants[0].get("prompt", "")
