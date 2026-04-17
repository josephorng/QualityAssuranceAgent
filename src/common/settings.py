from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_constants() -> dict[str, Any]:
    path = ROOT_DIR / "constants.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_host: str = "http://localhost:11434"
    eye_vlm: str = "gemma4:e2b"
    brain_lm: str = "gemma4:e2b"
    runs_dir: str = "runs"
    log_level: str = "INFO"
    screenshot_interval_seconds: int = 2
    screenshot_similarity_threshold: float = 0.985
    brain_memory_max_chars: int = 16000
    eye_port: int = 8001
    brain_port: int = 8002
    hand_port: int = 8003
    debug: bool = True


def load_settings() -> Settings:
    constants = _load_constants()
    ports = constants.get("ports", {})
    data = {
        "ollama_host": constants.get("ollama_host", "http://localhost:11434"),
        "eye_vlm": constants.get("eye_vlm", "gemma4:e2b"),
        "brain_lm": constants.get("brain_lm", "gemma4:e2b"),
        "screenshot_interval_seconds": constants.get("screenshot_interval_seconds", 2),
        "screenshot_similarity_threshold": constants.get("screenshot_similarity_threshold", 0.985),
        "brain_memory_max_chars": constants.get("brain_memory_max_chars", 16000),
        "eye_port": ports.get("eye", 8001),
        "brain_port": ports.get("brain", 8002),
        "hand_port": ports.get("hand", 8003),
        "debug": constants.get("debug", True),
    }
    return Settings(**data)
