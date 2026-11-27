"""Smoke tests for ComputerAgent_fault.log / faulthandler install."""

from __future__ import annotations

import faulthandler
from pathlib import Path

import main as main_mod


def test_install_faulthandler_writes_session_start(tmp_path: Path, monkeypatch) -> None:
    fault_path = tmp_path / "ComputerAgent_fault.log"
    monkeypatch.setattr(main_mod, "_fault_log_path", lambda: fault_path)
    monkeypatch.setattr(main_mod, "_FAULT_HEARTBEAT_INTERVAL_S", 3600.0)
    # Avoid stacking VEHs across repeated test runs on Windows.
    monkeypatch.setattr(main_mod, "_install_nonfatal_com_veh", lambda: None)
    main_mod._fault_heartbeat_stop.set()
    main_mod._fault_log_path_active = None

    main_mod._install_faulthandler()

    assert main_mod._fault_log_path_active == fault_path
    assert fault_path.is_file()
    text = fault_path.read_text(encoding="utf-8", errors="replace")
    assert "kind=session_start" in text
    assert faulthandler.is_enabled()

    main_mod._fault_heartbeat_stop.set()


def test_nonfatal_com_exception_codes_cover_cantcallout() -> None:
    assert 0x8001010D in main_mod._NONFATAL_COM_EXCEPTION_CODES
    assert 0x8001010A in main_mod._NONFATAL_COM_EXCEPTION_CODES
    assert 0x80010001 in main_mod._NONFATAL_COM_EXCEPTION_CODES
    # NTSTATUS RPC forms seen as SEH during cross-process UIA (e.g. Excel).
    assert 0xC0020043 in main_mod._NONFATAL_COM_EXCEPTION_CODES
    assert 0xC0020017 in main_mod._NONFATAL_COM_EXCEPTION_CODES
    assert 0xC0020018 in main_mod._NONFATAL_COM_EXCEPTION_CODES
    assert 0xC0020015 in main_mod._NONFATAL_COM_EXCEPTION_CODES
