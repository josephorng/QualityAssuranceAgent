from pathlib import Path

from src.common.run_state import RunStateManager


def test_init_run_creates_expected_paths(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path, memory_max_chars=100)
    paths = mgr.init_run("demo task", "demo_run")
    assert paths.eye_dir.exists()
    assert paths.thinking_dir.exists()
    assert paths.storage_dir.exists()
    assert paths.long_term_memory_txt.exists()
    assert paths.hand_csv.exists()
    assert paths.storage_json.exists()
    assert paths.steps_json.exists()
    assert paths.info_log.exists()


def test_brain_memory_capped(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path, memory_max_chars=20)
    mgr.init_run("task", "run")
    mgr.append_brain_memory("a" * 50)
    text = mgr.require_paths().long_term_memory_txt.read_text(encoding="utf-8")
    assert len(text) <= 20
