"""HTTP client for YOLO and CRNN models served by NVIDIA Triton."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

import numpy as np

from cua_mcp.vision_backend import (
    triton_client_address,
    triton_client_use_ssl,
    triton_crnn_model_name,
    triton_http_url,
    triton_yolo_model_name,
)

if TYPE_CHECKING:
    import tritonclient.http as httpclient

__all__ = [
    "TritonUnavailableError",
    "infer_crnn",
    "infer_yolo",
    "triton_configured",
    "triton_ready",
]

YOLO_INPUT_NAME = "images"
YOLO_OUTPUT_NAME = "output0"
CRNN_INPUT_NAME = "input"
CRNN_OUTPUT_NAME = "output"

_CLIENT: httpclient.InferenceServerClient | None = None


def _log_triton_profile(message: str) -> None:
    """Write Triton timing lines to the run log when available."""
    try:
        from src.common.run_state import get_run_state_manager

        get_run_state_manager().log_info(f"[vision_triton] {message}")
    except RuntimeError:
        pass


class TritonUnavailableError(RuntimeError):
    """Raised when Triton cannot be reached or returns an infer error."""


def triton_configured() -> bool:
    """Return True when a Triton HTTP URL is configured (always true with default localhost)."""
    return bool(triton_http_url())


def triton_ready(*, timeout_seconds: float = 2.5) -> bool:
    """Return True when Triton responds to ``/v2/health/ready``."""
    if not triton_configured():
        return False
    url = f"{triton_http_url()}/v2/health/ready"
    req = urllib.request.Request(url, method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            ok = 200 <= int(resp.status) < 300
            elapsed = time.perf_counter() - started
            _log_triton_profile(
                f"health_ready ok={ok} url={url} elapsed_s={elapsed:.3f}"
            )
            return ok
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        elapsed = time.perf_counter() - started
        _log_triton_profile(
            f"health_ready ok=False url={url} elapsed_s={elapsed:.3f} "
            f"error={type(exc).__name__}: {exc}"
        )
        return False


def _get_client() -> httpclient.InferenceServerClient:
    global _CLIENT
    if _CLIENT is None:
        import tritonclient.http as httpclient

        started = time.perf_counter()
        try:
            _CLIENT = httpclient.InferenceServerClient(
                url=triton_client_address(),
                verbose=False,
                ssl=triton_client_use_ssl(),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            _log_triton_profile(
                f"client_create failed address={triton_client_address()!r} "
                f"elapsed_s={elapsed:.3f} error={type(exc).__name__}: {exc}"
            )
            raise TritonUnavailableError(
                f"Failed to create Triton client for {triton_http_url()!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        elapsed = time.perf_counter() - started
        _log_triton_profile(
            f"client_create ok address={triton_client_address()!r} "
            f"ssl={triton_client_use_ssl()} elapsed_s={elapsed:.3f}"
        )
    return _CLIENT


def _infer(
    model_name: str,
    input_name: str,
    input_array: np.ndarray,
    output_name: str,
) -> np.ndarray:
    import tritonclient.http as httpclient

    shape = list(input_array.shape)
    started = time.perf_counter()
    try:
        client = _get_client()
        infer_input = httpclient.InferInput(
            input_name,
            shape,
            "FP32" if input_array.dtype != np.int64 else "INT64",
        )
        if input_array.dtype == np.float64:
            input_array = input_array.astype(np.float32)
        infer_input.set_data_from_numpy(input_array)
        infer_output = httpclient.InferRequestedOutput(output_name)
        result = client.infer(
            model_name=model_name,
            inputs=[infer_input],
            outputs=[infer_output],
        )
        out = result.as_numpy(output_name)
        if out is None:
            raise TritonUnavailableError(
                f"Triton model {model_name!r} returned no output {output_name!r}"
            )
        elapsed = time.perf_counter() - started
        _log_triton_profile(
            f"infer ok model={model_name} input={input_name} shape={shape} "
            f"output={output_name} out_shape={list(out.shape)} elapsed_s={elapsed:.3f}"
        )
        return out
    except TritonUnavailableError as exc:
        elapsed = time.perf_counter() - started
        _log_triton_profile(
            f"infer failed model={model_name} input={input_name} shape={shape} "
            f"elapsed_s={elapsed:.3f} error={exc}"
        )
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - started
        _log_triton_profile(
            f"infer failed model={model_name} input={input_name} shape={shape} "
            f"elapsed_s={elapsed:.3f} error={type(exc).__name__}: {exc}"
        )
        raise TritonUnavailableError(
            f"Triton infer failed for model {model_name!r}: {type(exc).__name__}: {exc}"
        ) from exc


def infer_yolo(nchw_batch: np.ndarray) -> np.ndarray:
    """Run YOLO end2end ONNX on Triton; return raw ``output0`` array."""
    batch = np.asarray(nchw_batch, dtype=np.float32)
    if batch.ndim != 4:
        raise ValueError(f"YOLO input must be NCHW batch, got shape {batch.shape}")
    return _infer(
        triton_yolo_model_name(),
        YOLO_INPUT_NAME,
        batch,
        YOLO_OUTPUT_NAME,
    )


def infer_crnn(batch: np.ndarray) -> np.ndarray:
    """Run CRNN OCR ONNX on Triton; return raw ``output`` token index array."""
    arr = np.asarray(batch, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"CRNN input must be [batch, H, W], got shape {arr.shape}")
    return _infer(
        triton_crnn_model_name(),
        CRNN_INPUT_NAME,
        arr,
        CRNN_OUTPUT_NAME,
    )
