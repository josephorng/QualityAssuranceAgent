from __future__ import annotations

from pathlib import Path

from cua_mcp import active_monitor_capture
from cua_mcp import hand_tools
from src.eye import capture


def test_active_monitor_capture_module_reuses_eye_functions() -> None:
    assert active_monitor_capture.active_monitor_index is capture.active_monitor_index
    assert active_monitor_capture.active_monitor_offset is capture.active_monitor_offset


def test_hand_tools_screenshot_uses_eye_capture(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    def _fake_capture(path: Path, default_monitor_index: int = 1) -> int:
        called["path"] = path
        called["default_monitor_index"] = default_monitor_index
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-png")
        return 1

    monkeypatch.setattr(hand_tools, "capture_active_monitor_to_file", _fake_capture)

    out = hand_tools.screenshot_to_file(str(tmp_path / "shot.png"))
    assert out["path"].endswith("shot.png")
    assert called["path"] == tmp_path / "shot.png"


def test_hand_tools_screenshot_appends_png_when_path_has_no_extension(
    monkeypatch, tmp_path: Path
) -> None:
    called: dict[str, object] = {}

    def _fake_capture(path: Path, default_monitor_index: int = 1) -> int:
        called["path"] = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-png")
        return 1

    monkeypatch.setattr(hand_tools, "capture_active_monitor_to_file", _fake_capture)

    out = hand_tools.screenshot_to_file(str(tmp_path / "chrome_status_capture"))
    assert out["path"].endswith("chrome_status_capture.png")
    assert called["path"] == tmp_path / "chrome_status_capture.png"


def test_capture_active_monitor_to_file_clamps_requested_index(monkeypatch, tmp_path: Path) -> None:
    class _FakeShot:
        size = (10, 8)
        rgb = b"\x00" * (10 * 8 * 3)

    class _FakeSct:
        def __init__(self) -> None:
            self.monitors = [
                {"left": 0, "top": 0, "width": 20, "height": 10},
                {"left": 10, "top": 20, "width": 10, "height": 8},
            ]
            self.grabbed: dict[str, int] | None = None

        def grab(self, monitor: dict[str, int]) -> _FakeShot:
            self.grabbed = monitor
            return _FakeShot()

    class _FakeMssCtx:
        def __init__(self) -> None:
            self.obj = _FakeSct()

        def __enter__(self) -> _FakeSct:
            return self.obj

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

    class _FakeImage:
        def save(self, dest: Path) -> None:
            Path(dest).write_bytes(b"saved")

    monkeypatch.setattr(capture, "active_monitor_index", lambda default=1: 99)
    monkeypatch.setattr(capture.mss, "mss", _FakeMssCtx)
    monkeypatch.setattr(capture.Image, "frombytes", lambda *_args, **_kwargs: _FakeImage())

    saved_idx = capture.capture_active_monitor_to_file(tmp_path / "capture.png")
    assert saved_idx == 1
    assert (tmp_path / "capture.png").exists()
