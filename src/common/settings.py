from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.io_utils import read_json, write_json

AGENT_SETTINGS_KEYS = (
    "llm_backend",
    "brain_lm",
    "ollama_host",
    "vllm_host",
    "debug",
)

AGENT_SETTINGS_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("llm_backend", "LLM 後端", "option"),
    ("debug", "除錯模式", "bool"),
)

# Fixed model/host pairs per backend (edited only via backend choice in the hub dialog).
BACKEND_PRESETS: dict[str, dict[str, str]] = {
    "ollama": {
        "llm_backend": "ollama",
        "brain_lm": "gemma4:e4b",
        "ollama_host": "http://localhost:11434",
        "vllm_host": "http://192.168.13.101:11434",
    },
    "vllm": {
        "llm_backend": "vllm",
        "brain_lm": "gemma4:26b",
        "ollama_host": "http://localhost:11434",
        "vllm_host": "http://192.168.13.101:11434",
    },
}

_LEGACY_CONSTANTS_PATH = "constants.json"
_AGENT_SETTINGS_FILENAME = "agent_settings.json"


def application_root() -> Path:
    """Project root in dev; directory containing the exe when frozen (Nuitka/PyInstaller)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


ROOT_DIR = application_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_backend: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    brain_lm: str = "gemma4:e4b"
    vllm_host: str = "http://192.168.13.101:11434"
    runs_dir: str = "runs"
    log_level: str = "INFO"
    debug: bool = True


def preset_for_backend(backend: str) -> dict[str, Any]:
    key = str(backend).strip().lower()
    if key not in BACKEND_PRESETS:
        raise ValueError("llm_backend 必須為 ollama 或 vllm")
    return dict(BACKEND_PRESETS[key])


def default_agent_settings_dict() -> dict[str, Any]:
    """Built-in defaults for agent settings (no file required)."""
    data = preset_for_backend("ollama")
    data["debug"] = Settings().debug
    return data


def _runs_dir_from_env() -> Path:
    """Resolve runs_dir from .env / env only (not from agent_settings.json)."""
    return Path(Settings().runs_dir)


def agent_settings_path() -> Path:
    return _runs_dir_from_env() / _AGENT_SETTINGS_FILENAME


def _legacy_constants_path() -> Path:
    return ROOT_DIR / _LEGACY_CONSTANTS_PATH


def _overlay_agent_keys(target: dict[str, Any], raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return target
    out = dict(target)
    for key in AGENT_SETTINGS_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key == "debug":
            out[key] = bool(value)
        elif key in ("llm_backend", "brain_lm", "ollama_host", "vllm_host"):
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
    return out


def normalize_agent_settings_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Apply fixed model/host preset for the selected backend; keep debug from input."""
    backend = str(data.get("llm_backend", "ollama")).strip().lower()
    out = preset_for_backend(backend)
    out["debug"] = bool(data.get("debug", True))
    return out


def load_agent_settings_dict() -> dict[str, Any]:
    """Load agent settings: defaults, then agent_settings.json, else legacy constants.json."""
    data = default_agent_settings_dict()
    path = agent_settings_path()
    if path.is_file():
        data = _overlay_agent_keys(data, read_json(path, {}))
        return normalize_agent_settings_dict(data)

    legacy = _legacy_constants_path()
    if legacy.is_file():
        data = _overlay_agent_keys(data, read_json(legacy, {}))
        data = normalize_agent_settings_dict(data)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, {k: data[k] for k in AGENT_SETTINGS_KEYS})
        except OSError:
            pass
    return data


def validate_agent_settings_dict(data: dict[str, Any]) -> dict[str, Any]:
    return normalize_agent_settings_dict(data)


def save_agent_settings_dict(data: dict[str, Any]) -> None:
    from src.common.llm_factory import reset_llm_client

    validated = validate_agent_settings_dict(data)
    path = agent_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, validated)
    reset_llm_client()


def load_settings() -> Settings:
    agent = load_agent_settings_dict()
    base = Settings()
    data = {
        "llm_backend": agent["llm_backend"],
        "ollama_host": agent["ollama_host"],
        "brain_lm": agent["brain_lm"],
        "vllm_host": agent["vllm_host"],
        "debug": agent["debug"],
        "runs_dir": base.runs_dir,
        "log_level": base.log_level,
    }
    return Settings(**data)
