"""Vision inference backend configuration (Triton only)."""

from __future__ import annotations

import os

VISION_BACKENDS = frozenset({"triton"})
_LEGACY_VISION_BACKEND_ALIASES = {
    "auto": "triton",
    "local": "triton",
}
DEFAULT_TRITON_HTTP_URL = "http://127.0.0.1:9000"
DEFAULT_VISION_BACKEND = "triton"
DEFAULT_TRITON_TIMEOUT_SECONDS = 20.0


def normalize_triton_http_url(url: str) -> str:
    """Prefer IPv4 loopback; ``localhost`` often hits broken IPv6 on Windows Docker."""
    base = url.strip().rstrip("/")
    if not base:
        return DEFAULT_TRITON_HTTP_URL
    for scheme in ("https://", "http://"):
        prefix = scheme + "localhost"
        if base.startswith(prefix):
            return scheme + "127.0.0.1" + base[len(prefix) :]
    return base


def vision_backend() -> str:
    """Return normalized ``VISION_BACKEND`` (always ``triton``)."""
    raw = os.environ.get("VISION_BACKEND", DEFAULT_VISION_BACKEND).strip().lower()
    raw = _LEGACY_VISION_BACKEND_ALIASES.get(raw, raw)
    if raw not in VISION_BACKENDS:
        return DEFAULT_VISION_BACKEND
    return raw


def triton_http_url() -> str:
    """Triton HTTP base URL without trailing slash."""
    return normalize_triton_http_url(
        os.environ.get("TRITON_HTTP_URL", DEFAULT_TRITON_HTTP_URL)
    )


def triton_timeout_seconds() -> float:
    """HTTP/infer timeout for Triton client requests."""
    raw = os.environ.get("TRITON_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_TRITON_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TRITON_TIMEOUT_SECONDS
    return max(0.1, value)


def triton_client_address() -> str:
    """``host:port`` for ``tritonclient`` (scheme must not be included)."""
    url = triton_http_url()
    if url.startswith("https://"):
        return url[len("https://") :]
    if url.startswith("http://"):
        return url[len("http://") :]
    return url


def triton_client_use_ssl() -> bool:
    """Return True when ``TRITON_HTTP_URL`` uses HTTPS."""
    return triton_http_url().startswith("https://")


def triton_yolo_model_name() -> str:
    return os.environ.get("TRITON_YOLO_MODEL", "yolo_ui").strip() or "yolo_ui"


def triton_crnn_model_name() -> str:
    return os.environ.get("TRITON_CRNN_MODEL", "crnn_ocr").strip() or "crnn_ocr"


def should_try_triton() -> bool:
    """Vision inference always uses Triton."""
    return True
