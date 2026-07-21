from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_vision_backend_defaults_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISION_BACKEND", raising=False)
    from cua_mcp import vision_backend as vb

    assert vb.vision_backend() == "triton"


def test_legacy_auto_and_local_map_to_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp import vision_backend as vb

    monkeypatch.setenv("VISION_BACKEND", "auto")
    assert vb.vision_backend() == "triton"
    monkeypatch.setenv("VISION_BACKEND", "local")
    assert vb.vision_backend() == "triton"


def test_should_try_triton_always_true(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp.vision_backend import should_try_triton

    monkeypatch.setenv("VISION_BACKEND", "triton")
    assert should_try_triton() is True


def test_triton_timeout_seconds_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp.vision_backend import triton_timeout_seconds

    monkeypatch.delenv("TRITON_TIMEOUT_SECONDS", raising=False)
    assert triton_timeout_seconds() == 20.0
    monkeypatch.setenv("TRITON_TIMEOUT_SECONDS", "15")
    assert triton_timeout_seconds() == 15.0


def test_triton_client_address_strips_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    from cua_mcp.vision_backend import triton_client_address, triton_client_use_ssl

    monkeypatch.setenv("TRITON_HTTP_URL", "http://localhost:9000")
    assert triton_client_address() == "127.0.0.1:9000"
    assert triton_client_use_ssl() is False

    monkeypatch.setenv("TRITON_HTTP_URL", "https://gpu-host:9000")
    assert triton_client_address() == "gpu-host:9000"
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

    with patch("tritonclient.http.InferenceServerClient", return_value=mock_client) as client_cls:
        from cua_mcp import vision_triton as vt

        vt.reset_triton_client()
        out = vt.infer_yolo(batch)

    assert out.shape == expected.shape
    mock_client.infer.assert_called_once()
    call_kwargs = mock_client.infer.call_args.kwargs
    assert call_kwargs["model_name"] == "yolo_ui"
    assert call_kwargs["timeout"] == 20
    client_kwargs = client_cls.call_args.kwargs
    assert client_kwargs["connection_timeout"] == 20.0
    assert client_kwargs["network_timeout"] == 20.0


def test_get_client_recreates_on_different_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    batch = np.zeros((1, 3, 1280, 1280), dtype=np.float32)
    expected = np.zeros((1, 300, 6), dtype=np.float32)
    clients: list[MagicMock] = []

    def make_client(*_args: object, **_kwargs: object) -> MagicMock:
        mock_result = MagicMock()
        mock_result.as_numpy.return_value = expected
        mock_client = MagicMock()
        mock_client.infer.return_value = mock_result
        clients.append(mock_client)
        return mock_client

    with patch("tritonclient.http.InferenceServerClient", side_effect=make_client):
        from cua_mcp import vision_triton as vt

        vt.reset_triton_client()
        first_thread_id: list[int] = []
        second_thread_id: list[int] = []

        def first_thread() -> None:
            first_thread_id.append(threading.get_ident())
            vt.infer_yolo(batch)

        def second_thread() -> None:
            second_thread_id.append(threading.get_ident())
            vt.infer_yolo(batch)

        t1 = threading.Thread(target=first_thread)
        t2 = threading.Thread(target=second_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    assert len(first_thread_id) == 1
    assert len(second_thread_id) == 1
    assert first_thread_id[0] != second_thread_id[0]
    assert len(clients) == 2
    assert clients[0] is not clients[1]
    vt.reset_triton_client()


def test_infer_retries_once_on_thread_affinity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    batch = np.zeros((1, 3, 1280, 1280), dtype=np.float32)
    expected = np.zeros((1, 300, 6), dtype=np.float32)

    mock_result = MagicMock()
    mock_result.as_numpy.return_value = expected
    mock_client = MagicMock()
    mock_client.infer.side_effect = [
        ValueError("error: cannot switch to a different thread (which happens to have exited)"),
        mock_result,
    ]

    with patch("tritonclient.http.InferenceServerClient", return_value=mock_client):
        from cua_mcp import vision_triton as vt

        vt.reset_triton_client()
        out = vt.infer_yolo(batch)

    assert out.shape == expected.shape
    assert mock_client.infer.call_count == 2
    vt.reset_triton_client()


def test_yolo_raises_on_triton_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    from cua_mcp.vision_triton import TritonUnavailableError
    from cua_mcp.yolo_onnx import _run_yolo_raw_output, DEFAULT_YOLO_ONNX_PATH

    img = np.zeros((1, 3, 1280, 1280), dtype=np.float32)

    with patch("cua_mcp.vision_triton.infer_yolo", side_effect=TritonUnavailableError("down")):
        with pytest.raises(TritonUnavailableError, match="down"):
            _run_yolo_raw_output(img, model_path=DEFAULT_YOLO_ONNX_PATH)


def test_get_ocr_predictor_does_not_require_onnx_file(
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


def test_text_predictor_raises_on_triton_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    from cua_mcp.read_screen_text.inference_onnx import TextPredictor
    from cua_mcp.vision_triton import TritonUnavailableError

    predictor = TextPredictor(quiet=True)
    batch = np.zeros((1, 32, 8), dtype=np.float32)

    with patch("cua_mcp.vision_triton.infer_crnn", side_effect=TritonUnavailableError("down")):
        with pytest.raises(TritonUnavailableError, match="down"):
            predictor.predict_images(batch)


def test_infer_timeout_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    batch = np.zeros((1, 3, 1280, 1280), dtype=np.float32)
    mock_client = MagicMock()
    mock_client.infer.side_effect = TimeoutError("timed out")

    with patch("tritonclient.http.InferenceServerClient", return_value=mock_client):
        from cua_mcp import vision_triton as vt
        from cua_mcp.vision_triton import TritonUnavailableError

        vt.reset_triton_client()
        with pytest.raises(TritonUnavailableError, match="timed out after 20s"):
            vt.infer_yolo(batch)
        vt.reset_triton_client()
