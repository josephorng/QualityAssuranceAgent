from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.prompts import PROMPTS


@dataclass(frozen=True)
class PromptConfig:
    prompt: str
    instructions: list[str]
    models: list[str]


def load_prompts() -> dict[str, Any]:
    return PROMPTS


def get_prompt_config(name: str) -> PromptConfig:
    variants = PROMPTS.get(name, [])
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


def get_prompt(name: str) -> str:
    config = get_prompt_config(name)
    if not config.instructions:
        return config.prompt
    block = "\n".join(f"- {line}" for line in config.instructions)
    return f"{config.prompt}\n\n{block}"
