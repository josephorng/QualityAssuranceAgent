from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.io_utils import read_json, write_json

AGENT_SETTINGS_KEYS = (
    "llm_backend",
    "brain_lm",
    "ollama_host",
    "debug",
)

AGENT_SETTINGS_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("llm_backend", "LLM 後端", "option"),
    ("debug", "除錯模式", "bool"),
)

# Fixed model/host pairs per backend (edited only via backend choice in the hub dialog).
BACKEND_PRESETS: dict[str, dict[str, str]] = {
    "ollama_local": {
        "llm_backend": "ollama_local",
        "brain_lm": "gemma4:e4b",
        "ollama_host": "http://localhost:11434",
    },
    "ollama_server": {
        "llm_backend": "ollama_server",
        "brain_lm": "gemma4:26b",
        "ollama_host": "http://192.168.13.101:11434",
    },
}

_LEGACY_LLM_BACKEND_ALIASES: dict[str, str] = {
    "ollama": "ollama_local",
    "vllm": "ollama_server",
}

_LEGACY_CONSTANTS_PATH = "constants.json"
_AGENT_SETTINGS_FILENAME = "agent_settings.json"

OLLAMA_PROBE_LOCAL_HOST = BACKEND_PRESETS["ollama_local"]["ollama_host"]
OLLAMA_PROBE_REMOTE_HOST = BACKEND_PRESETS["ollama_server"]["ollama_host"]
_OLLAMA_PROBE_TIMEOUT_SECONDS = 2.5


def application_root() -> Path:
    """Project root in dev; directory containing the exe when frozen (Nuitka/PyInstaller)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


ROOT_DIR = application_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_backend: str = "ollama_local"
    ollama_host: str = "http://localhost:11434"
    brain_lm: str = "gemma4:e4b"
    runs_dir: str = "runs"
    log_level: str = "INFO"
    debug: bool = True


def canonicalize_llm_backend(backend: str) -> str:
    """Map legacy ``ollama`` / ``vllm`` names to ``ollama_local`` / ``ollama_server``."""
    key = str(backend).strip().lower()
    return _LEGACY_LLM_BACKEND_ALIASES.get(key, key)


def preset_for_backend(backend: str) -> dict[str, Any]:
    key = canonicalize_llm_backend(backend)
    if key not in BACKEND_PRESETS:
        raise ValueError("llm_backend 必須為 ollama_local 或 ollama_server")
    return dict(BACKEND_PRESETS[key])


def default_agent_settings_dict() -> dict[str, Any]:
    """Built-in defaults for agent settings (no file required)."""
    data = preset_for_backend("ollama_local")
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
        elif key in ("llm_backend", "brain_lm", "ollama_host"):
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
    if isinstance(raw, dict):
        legacy_host = raw.get("vllm_host")
        if isinstance(legacy_host, str) and legacy_host.strip():
            backend = canonicalize_llm_backend(str(out.get("llm_backend", "ollama_local")))
            if backend == "ollama_server" or not str(out.get("ollama_host", "")).strip():
                out["ollama_host"] = legacy_host.strip()
    return out


def normalize_agent_settings_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Apply fixed model/host preset for the selected backend; keep debug and probed hosts."""
    backend = canonicalize_llm_backend(str(data.get("llm_backend", "ollama_local")))
    out = preset_for_backend(backend)
    out["debug"] = bool(data.get("debug", True))
    host = data.get("ollama_host") or data.get("vllm_host")
    if isinstance(host, str) and host.strip():
        out["ollama_host"] = host.strip()
    return out


def ollama_host_responds(host: str, *, timeout_seconds: float = _OLLAMA_PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True if an Ollama server responds at ``host`` (GET /api/tags)."""
    base = host.strip().rstrip("/")
    if not base:
        return False
    url = f"{base}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def select_reachable_ollama_host(
    *,
    local_host: str = OLLAMA_PROBE_LOCAL_HOST,
    remote_host: str = OLLAMA_PROBE_REMOTE_HOST,
) -> str | None:
    """Prefer ``local_host`` when it responds; else ``remote_host``; else None."""
    local = local_host.strip().rstrip("/")
    remote = remote_host.strip().rstrip("/")
    if ollama_host_responds(local):
        return local
    if ollama_host_responds(remote):
        return remote
    return None


def _ollama_host_probe_status_message(host: str) -> str:
    local = OLLAMA_PROBE_LOCAL_HOST.rstrip("/")
    chosen = host.rstrip("/")
    if chosen == local:
        return f"Ollama 主機：本機 ({host})"
    return f"Ollama 主機：公司主機 ({host})"


def apply_startup_ollama_host_probe() -> tuple[bool, str]:
    """Probe local then remote Ollama; persist chosen host when reachable."""
    chosen = select_reachable_ollama_host()
    if chosen is None:
        return (
            False,
            "錯誤：無法連線至 Ollama（本機與公司主機皆無回應）",
        )
    data = load_agent_settings_dict()
    data["ollama_host"] = chosen
    try:
        save_agent_settings_dict(data)
    except OSError:
        pass
    return True, _ollama_host_probe_status_message(chosen)


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
        "debug": agent["debug"],
        "runs_dir": base.runs_dir,
        "log_level": base.log_level,
    }
    return Settings(**data)
