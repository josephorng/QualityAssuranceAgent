from src.common.settings import (
    apply_startup_ollama_host_probe,
    normalize_agent_settings_dict,
    probe_llm_backend,
)


def test_normalize_vllm_server_ignores_probed_ollama_host() -> None:
    out = normalize_agent_settings_dict(
        {
            "llm_backend": "vllm_server",
            "brain_lm": "google/gemma-4-26B-A4B-it",
            "ollama_host": "http://localhost:11434",
            "debug": False,
        }
    )
    assert out["ollama_host"] == "http://192.168.4.134:8000"
    assert out["brain_lm"] == "google/gemma-4-26B-A4B-it"
    assert out["debug"] is False


def test_startup_probe_skips_ollama_for_vllm_server(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.common.settings.load_agent_settings_dict",
        lambda: {
            "llm_backend": "vllm_server",
            "brain_lm": "google/gemma-4-26B-A4B-it",
            "ollama_host": "http://192.168.4.134:8000",
            "debug": True,
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
