from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.settings import ROOT_DIR


@dataclass(frozen=True)
class PromptConfig:
    prompt: str
    skills: list[str]
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
        return PromptConfig(prompt="", skills=[], models=[])

    variant = variants[0]
    prompt = str(variant.get("prompt", ""))
    raw_skills = variant.get("skills", [])
    skills = [str(skill) for skill in raw_skills] if isinstance(raw_skills, list) else []
    raw_models = variant.get("models", [])
    models = [str(model) for model in raw_models] if isinstance(raw_models, list) else []
    return PromptConfig(prompt=prompt, skills=skills, models=models)


def render_prompt_with_skills(name: str) -> str:
    config = get_prompt_config(name)
    if not config.skills:
        return config.prompt
    skills_text = "\n".join(f"- {skill}" for skill in config.skills)
    return f"{config.prompt}\n\nSkills:\n{skills_text}"
