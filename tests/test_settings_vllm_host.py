from pathlib import Path

from src.common.settings import (
    apply_startup_ollama_host_probe,
    apply_startup_triton_probe,
    normalize_agent_settings_dict,
    probe_llm_backend,
    probe_vision_backend,
)


def test_normalize_vllm_server_ignores_probed_ollama_host() -> None:
    out = normalize_agent_settings_dict(
        {
            "llm_backend": "vllm_server",
            "brain_lm": "google/gemma-4-26B-A4B-it",
            "ollama_host": "http://localhost:11434",
        }
    )
    assert out["ollama_host"] == "http://192.168.4.134:8000"
    assert out["brain_lm"] == "google/gemma-4-26B-A4B-it"
    assert "debug" not in out


def test_startup_probe_skips_ollama_for_vllm_server(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.common.settings.load_agent_settings_dict",
        lambda: {
            "llm_backend": "vllm_server",
            "brain_lm": "google/gemma-4-26B-A4B-it",
            "ollama_host": "http://192.168.4.134:8000",
        },
    )
    saved: list[dict] = []
    monkeypatch.setattr(
        "src.common.settings.save_agent_settings_dict",
        lambda data: saved.append(data),
    )

    ok, message = apply_startup_ollama_host_probe()

    assert saved == []
    assert ok is True
    assert "vLLM" in message
    assert "192.168.4.134:8000" in message


def test_probe_llm_backend_ollama_success(monkeypatch) -> None:
    monkeypatch.setattr("src.common.settings.ollama_host_responds", lambda host: True)
    ok, message = probe_llm_backend("ollama_local")
    assert ok is True
    assert "連線成功" in message
    assert "localhost:11434" in message
    assert "gemma4:e4b" in message


def test_probe_llm_backend_vllm_failure(monkeypatch) -> None:
    monkeypatch.setattr("src.common.settings.vllm_host_responds", lambda host: False)
    ok, message = probe_llm_backend("vllm_server")
    assert ok is False
    assert "無法連線至 vLLM" in message
    assert "192.168.4.134:8000" in message


def test_normalize_agent_settings_includes_vision_fields() -> None:
    out = normalize_agent_settings_dict(
        {
            "llm_backend": "ollama_local",
            "vision_backend": "triton_local",
            "triton_http_url": "http://localhost:9000/",
        }
    )
    assert out["vision_backend"] == "triton_local"
    assert out["triton_http_url"] == "http://127.0.0.1:9000"


def test_normalize_agent_settings_remote_vision_preset() -> None:
    out = normalize_agent_settings_dict(
        {
            "llm_backend": "ollama_local",
            "vision_backend": "triton_192_168_0_17",
        }
    )
    assert out["vision_backend"] == "triton_192_168_0_17"
    assert out["triton_http_url"] == "http://192.168.0.17:9000"


def test_probe_vision_backend_triton_success(monkeypatch) -> None:
    monkeypatch.setattr("src.common.settings.triton_health_responds", lambda url: True)
    ok, message = probe_vision_backend("triton_local")
    assert ok is True
    assert "Triton 連線成功" in message
    assert "127.0.0.1:9000" in message


def test_probe_vision_backend_remote_preset(monkeypatch) -> None:
    monkeypatch.setattr("src.common.settings.triton_health_responds", lambda url: True)
    ok, message = probe_vision_backend("triton_192_168_0_17")
    assert ok is True
    assert "192.168.0.17:9000" in message


def test_probe_vision_backend_triton_failure(monkeypatch) -> None:
    monkeypatch.setattr("src.common.settings.triton_health_responds", lambda url: False)
    ok, message = probe_vision_backend("triton_local")
    assert ok is False
    assert "無法連線至 Triton" in message


def test_apply_startup_triton_probe_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.common.settings.load_settings",
        lambda: type(
            "S",
            (),
            {
                "triton_http_url": "http://localhost:9000",
                "vision_backend": "triton_local",
            },
        )(),
    )
    monkeypatch.setattr(
        "src.common.settings.probe_vision_backend",
        lambda backend, triton_http_url=None: (True, "ok"),
    )
    ok, message = apply_startup_triton_probe()
    assert ok is True
    assert "Triton 主機" in message


def test_warm_vision_models_returns_status_without_raising(monkeypatch) -> None:
    from cua_mcp.read_screen_text import ocr_image

    monkeypatch.setattr("cua_mcp.vision_triton.triton_ready", lambda **kw: False)
    ok, message = ocr_image.warm_vision_models(quiet=True, timeout_seconds=0.2)
    assert ok is False
    assert "Triton 無回應" in message


def test_default_runs_dir_dev_uses_project_root(monkeypatch) -> None:
    from src.common.settings import application_root, default_runs_dir

    monkeypatch.setattr("src.common.settings.is_frozen_app", lambda: False)
    assert default_runs_dir() == application_root() / "runs"


def test_default_runs_dir_frozen_uses_documents(monkeypatch, tmp_path: Path) -> None:
    from src.common.settings import default_runs_dir

    monkeypatch.setattr("src.common.settings.is_frozen_app", lambda: True)
    monkeypatch.setattr("src.common.settings.user_documents_dir", lambda: tmp_path)
    assert default_runs_dir() == tmp_path / "QualityAssuranceAgent" / "runs"


def test_resolve_runs_dir_absolute_and_relative(monkeypatch, tmp_path: Path) -> None:
    from src.common.settings import application_root, resolve_runs_dir

    monkeypatch.setattr("src.common.settings.is_frozen_app", lambda: False)
    absolute = tmp_path / "custom_runs"
    assert resolve_runs_dir(absolute) == absolute.resolve()
    assert resolve_runs_dir("runs") == (application_root() / "runs").resolve()
    assert resolve_runs_dir("alt_runs") == (application_root() / "alt_runs").resolve()


def test_load_settings_runs_dir_is_absolute(monkeypatch) -> None:
    from src.common.settings import application_root, load_settings

    monkeypatch.setattr("src.common.settings.is_frozen_app", lambda: False)
    monkeypatch.setattr(
        "src.common.settings.load_agent_settings_dict",
        lambda: {
            "llm_backend": "vllm_server",
            "brain_lm": "google/gemma-4-26B-A4B-it",
            "ollama_host": "http://192.168.4.134:8000",
            "vision_backend": "triton_192_168_0_17",
            "triton_http_url": "http://192.168.0.17:9000",
        },
    )
    settings = load_settings()
    assert Path(settings.runs_dir).is_absolute()
    assert Path(settings.runs_dir) == (application_root() / "runs").resolve()
