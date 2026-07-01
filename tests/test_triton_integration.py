from __future__ import annotations

import os

import numpy as np
import pytest

from cua_mcp.vision_triton import infer_crnn, infer_yolo, triton_ready


pytestmark = pytest.mark.triton


@pytest.fixture
def triton_required() -> None:
    if os.environ.get("VISION_BACKEND", "auto") == "local":
        pytest.skip("Set VISION_BACKEND=auto or triton for Triton integration tests")
    if not triton_ready(timeout_seconds=1.0):
        pytest.skip("Triton server not reachable at TRITON_HTTP_URL")


def test_triton_yolo_infer_shape(triton_required: None) -> None:
    batch = np.zeros((1, 3, 1280, 1280), dtype=np.float32)
    out = infer_yolo(batch)
    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[-1] == 6


def test_triton_crnn_infer_shape(triton_required: None) -> None:
    batch = np.zeros((2, 32, 16), dtype=np.float32)
    out = infer_crnn(batch)
    assert out.ndim == 2
    assert out.shape[0] == 2
