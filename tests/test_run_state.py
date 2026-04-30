from pathlib import Path

from src.common.run_state import RunStateManager


def test_init_run_creates_expected_paths(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    paths = mgr.init_run("demo task", "demo_run")
    assert paths.eye_dir.exists()
    assert paths.storage_dir.exists()
    assert paths.hand_csv.exists()
    assert paths.storage_json.exists()
    assert paths.info_log.exists()
