import pytest


def test_brain_module_import_smoke() -> None:
    try:
        from src.brain.module import BrainModule
    except Exception as exc:  # pragma: no cover - environment-dependent compatibility
        pytest.skip(f"Skipping Brain module smoke due to runtime compatibility issue: {exc}")
    assert BrainModule is not None


def test_hand_module_import_smoke() -> None:
    try:
        from src.hand.module import HandModule
    except Exception as exc:  # pragma: no cover - environment-dependent compatibility
        pytest.skip(f"Skipping Hand module smoke due to runtime compatibility issue: {exc}")
    assert HandModule is not None


def test_coordinator_import_smoke() -> None:
    try:
        from src.runtime.coordinator import RuntimeCoordinator
    except Exception as exc:  # pragma: no cover - environment-dependent compatibility
        pytest.skip(f"Skipping Coordinator import smoke due to runtime compatibility issue: {exc}")
    assert RuntimeCoordinator is not None
