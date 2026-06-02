"""
Single source of truth for the LLM backend used across the project.

The active client is selected by ``llm_backend`` in ``runs/agent_settings.json``
(loaded by :func:`src.common.settings.load_settings`). Supported values:

* ``"ollama_local"`` (default) - :class:`src.common.ollama_client.OllamaClient`
* ``"ollama_server"``          - :class:`src.common.vllm_client.VLLMClient`

The client is constructed lazily on first use and cached as a process-wide
singleton; use the hub settings dialog or :func:`reset_llm_client` before the
next run to pick up changes.
"""

from __future__ import annotations

from threading import Lock

from src.common.llm_client import LLMClient
from src.common.settings import canonicalize_llm_backend, load_settings

__all__ = ["get_llm_client", "reset_llm_client"]


_client_singleton: LLMClient | None = None
_client_lock = Lock()


def _build_client() -> LLMClient:
    settings = load_settings()
    backend = canonicalize_llm_backend(settings.llm_backend or "ollama_local")
    if backend == "ollama_local":
        from src.common.ollama_client import OllamaClient

        return OllamaClient(settings.ollama_host)
    if backend == "ollama_server":
        from src.common.vllm_client import VLLMClient

        return VLLMClient(settings.ollama_host)
    raise ValueError(
        f"Unknown llm_backend {backend!r} in agent settings; "
        "expected 'ollama_local' or 'ollama_server'"
    )


def get_llm_client() -> LLMClient:
    """Return the process-wide LLM client, building it on first access."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    with _client_lock:
        if _client_singleton is None:
            _client_singleton = _build_client()
    return _client_singleton


def reset_llm_client() -> None:
    """Drop the cached singleton (useful for tests or runtime reconfiguration)."""
    global _client_singleton
    with _client_lock:
        _client_singleton = None
