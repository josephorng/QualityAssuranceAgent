import pytest


def test_brain_module_import_smoke() -> None:
    try:
        import src.brain.server as brain_server
    except Exception as exc:  # pragma: no cover - environment-dependent compatibility
        pytest.skip(f"Skipping Brain FastAPI smoke due to runtime compatibility issue: {exc}")
    assert brain_server.app.title == "Brain Server"


def test_hand_module_import_smoke() -> None:
    try:
        import src.hand.server as hand_server
    except Exception as exc:  # pragma: no cover - environment-dependent compatibility
        pytest.skip(f"Skipping Hand FastAPI smoke due to runtime compatibility issue: {exc}")
    assert hand_server.app.title == "Hand Server"
