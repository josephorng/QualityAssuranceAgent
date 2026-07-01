from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_vision_backend_defaults_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISION_BACKEND", raising=False)
    from cua_mcp import vision_backend as vb

    monkeypatch.setattr(vb, "DEFAULT_VISION_BACKEND", "auto")
    assert vb.vision_backend() == "auto"


def test_should_try_triton_respects_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "local")
    from cua_mcp.vision_backend import allow_local_ort_fallback, should_try_triton

    assert should_try_triton() is False
    assert allow_local_ort_fallback() is False


def test_should_try_triton_for_auto_and_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp.vision_backend import allow_local_ort_fallback, should_try_triton

    monkeypatch.setenv("VISION_BACKEND", "auto")
    assert should_try_triton() is True
    assert allow_local_ort_fallback() is True

    monkeypatch.setenv("VISION_BACKEND", "triton")
    assert should_try_triton() is True
    assert allow_local_ort_fallback() is False


def test_triton_client_address_strips_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp.vision_backend import triton_client_address, triton_client_use_ssl

    monkeypatch.setenv("TRITON_HTTP_URL", "http://localhost:8000")
    assert triton_client_address() == "localhost:8000"
    assert triton_client_use_ssl() is False

    monkeypatch.setenv("TRITON_HTTP_URL", "https://gpu-host:8000")
    assert triton_client_address() == "gpu-host:8000"
    assert triton_client_use_ssl() is True


def test_triton_ready_false_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRITON_HTTP_URL", "http://127.0.0.1:1")
    from cua_mcp.vision_triton import triton_ready

    assert triton_ready(timeout_seconds=0.2) is False


def test_infer_yolo_calls_triton_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    batch = np.zeros((1, 3, 1280, 1280), dtype=np.float32)
    expected = np.zeros((1, 300, 6), dtype=np.float32)

    mock_result = MagicMock()
    mock_result.as_numpy.return_value = expected
    mock_client = MagicMock()
    mock_client.infer.return_value = mock_result

    with patch("tritonclient.http.InferenceServerClient", return_value=mock_client):
        from cua_mcp import vision_triton as vt

        vt._CLIENT = None
        out = vt.infer_yolo(batch)

    assert out.shape == expected.shape
    mock_client.infer.assert_called_once()
    call_kwargs = mock_client.infer.call_args.kwargs
    assert call_kwargs["model_name"] == "yolo_ui"


def test_yolo_falls_back_to_ort_on_triton_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "auto")
    from cua_mcp.vision_triton import TritonUnavailableError
    from cua_mcp.yolo_onnx import _run_yolo_raw_output, DEFAULT_YOLO_ONNX_PATH

    img = np.zeros((1, 3, 1280, 1280), dtype=np.float32)
    ort_out = np.zeros((1, 300, 6), dtype=np.float32)

    with patch("cua_mcp.vision_triton.infer_yolo", side_effect=TritonUnavailableError("down")):
        with patch("cua_mcp.yolo_onnx._local_ort_infer_yolo", return_value=ort_out) as local:
            out = _run_yolo_raw_output(img, model_path=DEFAULT_YOLO_ONNX_PATH)

    assert np.array_equal(out, ort_out)
    local.assert_called_once()


def test_get_ocr_predictor_skips_onnx_file_when_triton_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    from cua_mcp.read_screen_text import ocr_image

    ocr_image._CRNN_PREDICTOR = None
    missing = tmp_path / "missing.onnx"

    with patch.object(ocr_image, "TextPredictor") as mock_predictor:
        ocr_image._get_ocr_predictor(str(missing), quiet=True)

    mock_predictor.assert_called_once_with(str(missing), quiet=True)
    ocr_image._CRNN_PREDICTOR = None


def test_get_ocr_predictor_requires_onnx_file_for_local_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VISION_BACKEND", "local")
    from cua_mcp.read_screen_text import ocr_image

    ocr_image._CRNN_PREDICTOR = None
    missing = tmp_path / "missing.onnx"

    with pytest.raises(FileNotFoundError, match="ONNX CRNN model not found"):
        ocr_image._get_ocr_predictor(str(missing), quiet=True)

    ocr_image._CRNN_PREDICTOR = None


def test_text_predictor_falls_back_to_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VISION_BACKEND", "auto")
    model = ROOT / "cua_mcp" / "read_screen_text" / "ocr_model_finetuned.onnx"
    if not model.is_file():
        pytest.skip("CRNN ONNX model not present")

    from cua_mcp.read_screen_text.inference_onnx import TextPredictor
    from cua_mcp.vision_triton import TritonUnavailableError

    predictor = TextPredictor(str(model), quiet=True)
    batch = np.zeros((1, 32, 8), dtype=np.float32)
    fake_out = np.full((1, 4), 9999, dtype=np.int64)

    with patch("cua_mcp.vision_triton.infer_crnn", side_effect=TritonUnavailableError("down")):
        with patch.object(predictor, "_local_ort_predict", return_value=fake_out) as local:
            preds = predictor.predict_images(batch)

    local.assert_called_once()
    assert preds == [""]
