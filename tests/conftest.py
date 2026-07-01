from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "triton: requires a running NVIDIA Triton server")


@pytest.fixture(autouse=True)
def _vision_backend_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default tests to local ONNX Runtime unless a test opts into Triton."""
    if "VISION_BACKEND" not in os.environ:
        monkeypatch.setenv("VISION_BACKEND", "local")
