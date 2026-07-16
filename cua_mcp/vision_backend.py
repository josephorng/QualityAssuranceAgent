"""Vision inference backend selection (Triton vs local ONNX Runtime)."""

from __future__ import annotations

import os

VISION_BACKENDS = frozenset({"auto", "triton", "local"})
DEFAULT_TRITON_HTTP_URL = "http://localhost:9000"
DEFAULT_VISION_BACKEND = "auto"


def vision_backend() -> str:
    """Return normalized ``VISION_BACKEND`` (``auto``, ``triton``, or ``local``)."""
    raw = os.environ.get("VISION_BACKEND", DEFAULT_VISION_BACKEND).strip().lower()
    if raw not in VISION_BACKENDS:
        return DEFAULT_VISION_BACKEND
    return raw


def triton_http_url() -> str:
    """Triton HTTP base URL without trailing slash."""
    return os.environ.get("TRITON_HTTP_URL", DEFAULT_TRITON_HTTP_URL).strip().rstrip("/")


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
    """Return True when the caller should attempt a Triton infer before local ORT."""
    return vision_backend() != "local"


def allow_local_ort_fallback() -> bool:
    """Return True when Triton failures may fall back to local ONNX Runtime."""
    return vision_backend() == "auto"
