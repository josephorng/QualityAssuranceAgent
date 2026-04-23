from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.settings import ROOT_DIR


@dataclass(frozen=True)
class PromptConfig:
    prompt: str
    instructions: list[str]
    models: list[str]


def load_prompts() -> dict[str, Any]:
    path = ROOT_DIR / "prompts.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_prompt(name: str) -> str:
    return get_prompt_config(name).prompt


def get_prompt_config(name: str) -> PromptConfig:
    prompts = load_prompts()
    variants = prompts.get(name, [])
    if not variants:
        return PromptConfig(prompt="", instructions=[], models=[])

    variant = variants[0]
    prompt = str(variant.get("prompt", ""))
    raw_instructions = variant.get("instructions", [])
    instructions = (
        [str(s) for s in raw_instructions] if isinstance(raw_instructions, list) else []
    )
    raw_models = variant.get("models", [])
    models = [str(model) for model in raw_models] if isinstance(raw_models, list) else []
    return PromptConfig(prompt=prompt, instructions=instructions, models=models)


def render_prompt_with_instructions(name: str) -> str:
    config = get_prompt_config(name)
    if not config.instructions:
        return config.prompt
    block = "\n".join(f"- {line}" for line in config.instructions)
    return f"{config.prompt}\n\nInstructions:\n{block}"
