from src.common.settings import (
    apply_startup_ollama_host_probe,
    normalize_agent_settings_dict,
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
