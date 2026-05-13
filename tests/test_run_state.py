from pathlib import Path

from src.common.run_state import RunStateManager, sanitize_log_text


def test_init_run_creates_expected_paths(tmp_path: Path) -> None:
    mgr = RunStateManager(tmp_path)
    paths = mgr.init_run("demo task", "demo_run")
    assert paths.eye_dir.exists()
    assert paths.storage_dir.exists()
    assert paths.hand_csv.exists()
    assert paths.storage_json.exists()
    assert paths.info_log.exists()


def test_sanitize_log_text_redacts_long_data_urls() -> None:
    b64 = "A" * 200
    raw = f"payload={{'url': 'data:image/png;base64,{b64}'}}"
    out = sanitize_log_text(raw)
    assert "AAA" in out
    assert "omitted 200 base64" in out
    assert len(out) < len(raw)


def test_sanitize_log_text_skips_short_data_urls() -> None:
    raw = "data:image/png;base64,QUJD"  # "ABC" — too short to redact
    assert sanitize_log_text(raw) == raw


def test_sanitize_log_text_avoids_metadata_false_positive() -> None:
    raw = "metadata:foo"
    assert sanitize_log_text(raw) == raw
