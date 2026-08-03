from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.io_utils import read_json, write_json
from cua_mcp.vision_backend import normalize_triton_http_url

AGENT_SETTINGS_KEYS = (
    "llm_backend",
    "brain_lm",
    "ollama_host",
    "vision_backend",
    "triton_http_url",
)

AGENT_SETTINGS_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("llm_backend", "LLM 後端", "option"),
    ("vision_backend", "Vision 後端", "option"),
)

VISION_BACKEND_CHOICES = frozenset({"triton_local", "triton_192_168_0_17"})
_LEGACY_VISION_BACKEND_ALIASES = {
    "auto": "triton_local",
    "local": "triton_local",
    "triton": "triton_local",
}

# Fixed Triton host per vision preset (edited only via vision choice in the hub dialog).
VISION_BACKEND_PRESETS: dict[str, dict[str, str]] = {
    "triton_local": {
        "triton_http_url": "http://127.0.0.1:9000",
    },
    "triton_192_168_0_17": {
        "triton_http_url": "http://192.168.0.17:9000",
    },
}

# Fixed model/host pairs per backend (edited only via backend choice in the hub dialog).
BACKEND_PRESETS: dict[str, dict[str, str]] = {
    "ollama_local": {
        "llm_backend": "ollama_local",
        "brain_lm": "gemma4:e4b",
        "ollama_host": "http://localhost:11434",
    },
    "ollama_local_12b": {
        "llm_backend": "ollama_local_12b",
        "brain_lm": "gemma4:12b",
        "ollama_host": "http://localhost:11434",
    },
    "ollama_server": {
        "llm_backend": "ollama_server",
        "brain_lm": "gemma4:26b-a4b-it-q4_K_M",
        "ollama_host": "http://192.168.13.8:11434",
        # "ollama_host": "http://192.168.13.101:11434",
    },
    "vllm_server": {
        "llm_backend": "vllm_server",
        "brain_lm": "google/gemma-4-26B-A4B-it",
        "ollama_host": "http://192.168.4.134:8000",
    },
}

_LEGACY_LLM_BACKEND_ALIASES: dict[str, str] = {
    "ollama": "ollama_local",
    "vllm": "vllm_server",
}

_LEGACY_CONSTANTS_PATH = "constants.json"
_AGENT_SETTINGS_FILENAME = "agent_settings.json"

OLLAMA_PROBE_LOCAL_HOST = BACKEND_PRESETS["ollama_local"]["ollama_host"]
OLLAMA_PROBE_REMOTE_HOST = BACKEND_PRESETS["ollama_server"]["ollama_host"]
_OLLAMA_PROBE_TIMEOUT_SECONDS = 2.5


_USER_DATA_APP_NAME = "ComputerUseAgent"
_DEFAULT_RUNS_NAME = "runs"


def is_frozen_app() -> bool:
    """True when running as a Nuitka/PyInstaller binary."""
    return bool(getattr(sys, "frozen", False) or globals().get("__compiled__") is not None)


def application_root() -> Path:
    """Project root in dev; directory containing the exe when frozen (Nuitka/PyInstaller)."""
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def user_documents_dir() -> Path:
    """User Documents folder (Windows known folder when available)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", wintypes.BYTE * 8),
                ]

            # FOLDERID_Documents = {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
            folder_id = _GUID(
                0xFDD39AD0,
                0x238F,
                0x46AF,
                (wintypes.BYTE * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7),
            )
            path_ptr = ctypes.c_wchar_p()
            hr = ctypes.windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(folder_id), 0, None, ctypes.byref(path_ptr)
            )
            if hr == 0 and path_ptr.value:
                path = Path(path_ptr.value)
                ctypes.windll.ole32.CoTaskMemFree(path_ptr)
                return path
            if path_ptr:
                ctypes.windll.ole32.CoTaskMemFree(path_ptr)
        except (AttributeError, OSError, ValueError, TypeError):
            pass
    return Path.home() / "Documents"


def default_runs_dir() -> Path:
    """Frozen: Documents/<app>/runs; dev: <project>/runs."""
    if is_frozen_app():
        return user_documents_dir() / _USER_DATA_APP_NAME / _DEFAULT_RUNS_NAME
    return application_root() / _DEFAULT_RUNS_NAME


def resolve_runs_dir(configured: str | Path | None = None) -> Path:
    """Resolve ``runs_dir`` to an absolute path.

    Absolute configured paths are used as-is. Empty / ``runs`` uses
    :func:`default_runs_dir`. Other relative paths resolve against
    :func:`application_root`.
    """
    if configured is None:
        raw = str(Settings().runs_dir).strip()
    else:
        raw = str(configured).strip()
    if not raw or raw in (".", _DEFAULT_RUNS_NAME):
        return default_runs_dir().resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = application_root() / path
    return path.resolve()


ROOT_DIR = application_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_backend: str = "vllm_server"
    ollama_host: str = BACKEND_PRESETS["vllm_server"]["ollama_host"]
    brain_lm: str = BACKEND_PRESETS["vllm_server"]["brain_lm"]
    runs_dir: str = _DEFAULT_RUNS_NAME
    log_level: str = "INFO"
    triton_http_url: str = VISION_BACKEND_PRESETS["triton_192_168_0_17"]["triton_http_url"]
    vision_backend: str = "triton_192_168_0_17"
    triton_timeout_seconds: float = 20.0
    smart_max_cycles: int = 50
    smart_max_recovery_attempts: int = 3


def canonicalize_llm_backend(backend: str) -> str:
    """Map legacy ``ollama`` / ``vllm`` names to ``ollama_local`` / ``ollama_server``."""
    key = str(backend).strip().lower()
    return _LEGACY_LLM_BACKEND_ALIASES.get(key, key)


def canonicalize_vision_backend(backend: str) -> str:
    """Return a known vision preset key (e.g. ``triton_local``)."""
    key = str(backend).strip().lower()
    key = _LEGACY_VISION_BACKEND_ALIASES.get(key, key)
    if key not in VISION_BACKEND_CHOICES:
        known = ", ".join(sorted(VISION_BACKEND_CHOICES))
        raise ValueError(f"vision_backend 必須為已知後端之一：{known}")
    return key


def preset_for_vision_backend(backend: str) -> dict[str, Any]:
    key = canonicalize_vision_backend(backend)
    return dict(VISION_BACKEND_PRESETS[key])


def preset_for_backend(backend: str) -> dict[str, Any]:
    key = canonicalize_llm_backend(backend)
    if key not in BACKEND_PRESETS:
        known = ", ".join(sorted(BACKEND_PRESETS))
        raise ValueError(f"llm_backend 必須為已知後端之一：{known}")
    return dict(BACKEND_PRESETS[key])


def default_agent_settings_dict() -> dict[str, Any]:
    """Built-in defaults for agent settings (no file required)."""
    base = Settings()
    data = preset_for_backend(base.llm_backend)
    data["vision_backend"] = base.vision_backend
    data["triton_http_url"] = preset_for_vision_backend(base.vision_backend)["triton_http_url"]
    return data


def _runs_dir_from_env() -> Path:
    """Resolve runs_dir from .env / env only (not from agent_settings.json)."""
    return resolve_runs_dir(Settings().runs_dir)


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
        if key in ("llm_backend", "brain_lm", "ollama_host", "vision_backend", "triton_http_url"):
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
    if isinstance(raw, dict):
        legacy_host = raw.get("vllm_host")
        if isinstance(legacy_host, str) and legacy_host.strip():
            backend = canonicalize_llm_backend(str(out.get("llm_backend", Settings().llm_backend)))
            if backend == "ollama_server" or not str(out.get("ollama_host", "")).strip():
                out["ollama_host"] = legacy_host.strip()
    return out


def normalize_agent_settings_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Apply fixed model/host preset for the selected backend; keep probed hosts."""
    base = Settings()
    backend = canonicalize_llm_backend(str(data.get("llm_backend", base.llm_backend)))
    out = preset_for_backend(backend)
    vision_key = canonicalize_vision_backend(
        str(data.get("vision_backend", base.vision_backend))
    )
    vision_preset = preset_for_vision_backend(vision_key)
    out["vision_backend"] = vision_key
    out["triton_http_url"] = normalize_triton_http_url(vision_preset["triton_http_url"])
    if backend == "vllm_server":
        return out
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


def vllm_host_responds(host: str, *, timeout_seconds: float = _OLLAMA_PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True if an OpenAI-compatible vLLM server responds at ``host`` (GET /v1/models)."""
    base = host.strip().rstrip("/")
    if not base:
        return False
    url = f"{base}/v1/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def probe_llm_backend(backend: str) -> tuple[bool, str]:
    """Test connectivity for the selected LLM backend preset. Returns ``(ok, message)``."""
    key = canonicalize_llm_backend(backend)
    preset = preset_for_backend(key)
    host = str(preset["ollama_host"])
    model = str(preset["brain_lm"])
    if key == "vllm_server":
        if vllm_host_responds(host):
            return True, f"連線成功\n主機：{host}\n模型：{model}"
        return False, f"無法連線至 vLLM\n主機：{host}"
    if ollama_host_responds(host):
        return True, f"連線成功\n主機：{host}\n模型：{model}"
    return False, f"無法連線至 Ollama\n主機：{host}"


def triton_health_responds(
    triton_http_url: str,
    *,
    timeout_seconds: float = _OLLAMA_PROBE_TIMEOUT_SECONDS,
) -> bool:
    """Return True when Triton responds to ``GET /v2/health/ready``."""
    base = normalize_triton_http_url(triton_http_url)
    if not base:
        return False
    url = f"{base}/v2/health/ready"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def local_vision_models_available() -> tuple[bool, str]:
    """Deprecated: local ONNX models are no longer used for inference."""
    return False, "本機 ONNX 推論已停用"


def probe_vision_backend(
    vision_backend: str,
    *,
    triton_http_url: str | None = None,
) -> tuple[bool, str]:
    """Test connectivity for the Triton Vision backend. Returns ``(ok, message)``."""
    key = canonicalize_vision_backend(vision_backend)
    if triton_http_url is None:
        triton_url = normalize_triton_http_url(
            preset_for_vision_backend(key)["triton_http_url"]
        )
    else:
        triton_url = normalize_triton_http_url(triton_http_url)
    if triton_health_responds(triton_url):
        return True, f"Triton 連線成功\n主機：{triton_url}"
    return False, f"無法連線至 Triton\n主機：{triton_url}"


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


def apply_startup_triton_probe() -> tuple[bool, str]:
    """Probe Triton readiness at startup. Returns ``(ok, message)`` without raising."""
    settings = load_settings()
    triton_url = settings.triton_http_url
    ok, _detail = probe_vision_backend(settings.vision_backend, triton_http_url=triton_url)
    if ok:
        return True, f"Triton 主機：{triton_url}"
    return False, f"警告：無法連線 Triton（{triton_url}）"


def apply_startup_ollama_host_probe() -> tuple[bool, str]:
    """Probe local then remote Ollama; persist chosen host when reachable."""
    data = load_agent_settings_dict()
    backend = canonicalize_llm_backend(str(data.get("llm_backend", Settings().llm_backend)))
    if backend == "vllm_server":
        return True, f"vLLM 主機：{data['ollama_host']}"

    chosen = select_reachable_ollama_host()
    if chosen is None:
        return (
            False,
            "錯誤：無法連線至 Ollama（本機與公司主機皆無回應）",
        )
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
    apply_vision_env_from_settings()
    try:
        from cua_mcp.vision_triton import reset_triton_client

        reset_triton_client()
    except ImportError:
        pass


def load_settings() -> Settings:
    agent = load_agent_settings_dict()
    base = Settings()
    data = {
        "llm_backend": agent["llm_backend"],
        "ollama_host": agent["ollama_host"],
        "brain_lm": agent["brain_lm"],
        "runs_dir": str(resolve_runs_dir(base.runs_dir)),
        "log_level": base.log_level,
        "triton_http_url": agent["triton_http_url"],
        "vision_backend": agent["vision_backend"],
        "triton_timeout_seconds": base.triton_timeout_seconds,
    }
    return Settings(**data)


def apply_vision_env_from_settings() -> None:
    """Mirror vision-related Settings into ``os.environ`` for cua_mcp inference modules."""
    import os

    settings = load_settings()
    os.environ["TRITON_HTTP_URL"] = normalize_triton_http_url(settings.triton_http_url)
    os.environ["VISION_BACKEND"] = "triton"
    os.environ["TRITON_TIMEOUT_SECONDS"] = str(settings.triton_timeout_seconds)
