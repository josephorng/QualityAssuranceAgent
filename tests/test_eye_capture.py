from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cua_mcp import active_monitor_capture
from cua_mcp import hand_tools
from src.common.run_state import RunStateManager
from src.common.runtime_context import set_runtime_env
from src.eye import capture
from src.eye.module import EyeModule


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


def test_overlay_mouse_cursor_draws_when_cursor_is_in_bounds(monkeypatch) -> None:
    img = capture.Image.new("RGB", (100, 100), "gray")
    monkeypatch.setattr(capture.pyautogui, "position", lambda: type("P", (), {"x": 20, "y": 30})())

    before = img.getpixel((20, 30))
    capture.overlay_mouse_cursor(img, origin_left=0, origin_top=0)
    after = img.getpixel((20, 30))

    assert before != after


def test_overlay_mouse_cursor_skips_when_cursor_is_outside_image(monkeypatch) -> None:
    img = capture.Image.new("RGB", (100, 100), "gray")
    monkeypatch.setattr(capture.pyautogui, "position", lambda: type("P", (), {"x": 200, "y": 30})())

    before = img.getpixel((50, 50))
    capture.overlay_mouse_cursor(img, origin_left=0, origin_top=0)
    after = img.getpixel((50, 50))

    assert before == after


def test_capture_all_screens_to_file_uses_virtual_desktop(monkeypatch, tmp_path: Path) -> None:
    class _FakeShot:
        size = (20, 10)
        rgb = b"\x00" * (20 * 10 * 3)

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
        width = 20
        height = 10

        def save(self, dest: Path) -> None:
            Path(dest).write_bytes(b"saved")

    overlay_calls: list[tuple[int, int]] = []

    def _fake_overlay(img: object, origin_left: int, origin_top: int) -> object:
        overlay_calls.append((origin_left, origin_top))
        return img

    monkeypatch.setattr(capture.mss, "mss", _FakeMssCtx)
    monkeypatch.setattr(capture.Image, "frombytes", lambda *_args, **_kwargs: _FakeImage())
    monkeypatch.setattr(capture, "overlay_mouse_cursor", _fake_overlay)

    capture.capture_all_screens_to_file(tmp_path / "all.png")
    assert (tmp_path / "all.png").exists()
    assert overlay_calls == [(0, 0)]


def test_capture_once_writes_combined_screenshot_when_multiple_monitors_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = RunStateManager(runs_root=tmp_path)
    paths = manager.init_run("combined-after-action-test", "test_combined_after_action")
    set_runtime_env(paths.root, paths.root.name)

    captured: list[Path] = []
    single_monitor_calls: list[int] = []

    def _fake_capture_all_screens(dest: Path) -> None:
        captured.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"combined")

    def _fake_capture_monitor(
        *,
        dest: Path,
        monitor_index: int,
        include_cursor: bool = False,
    ) -> int:
        single_monitor_calls.append(monitor_index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"single")
        return monitor_index

    import src.eye.module as eye_module

    monkeypatch.setattr(eye_module, "capture_all_screens_to_file", _fake_capture_all_screens)
    monkeypatch.setattr(eye_module, "capture_monitor_to_file", _fake_capture_monitor)
    monkeypatch.setenv("EYE_MONITOR_INDICES", "1,2")

    eye = EyeModule()
    event = asyncio.run(eye.capture_once())

    assert len(captured) == 1
    assert single_monitor_calls == []
    assert captured[0].parent == paths.eye_dir
    assert event.screenshot_path == str(captured[0])


def test_capture_once_writes_single_monitor_when_only_one_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = RunStateManager(runs_root=tmp_path)
    paths = manager.init_run("single-after-action-test", "test_single_after_action")
    set_runtime_env(paths.root, paths.root.name)

    captured: list[Path] = []
    single_monitor_calls: list[int] = []

    def _fake_capture_all_screens(dest: Path) -> None:
        captured.append(dest)

    def _fake_capture_monitor(
        *,
        dest: Path,
        monitor_index: int,
        include_cursor: bool = False,
    ) -> int:
        single_monitor_calls.append(monitor_index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"single")
        return monitor_index

    import src.eye.module as eye_module

    monkeypatch.setattr(eye_module, "capture_all_screens_to_file", _fake_capture_all_screens)
    monkeypatch.setattr(eye_module, "capture_monitor_to_file", _fake_capture_monitor)
    monkeypatch.setenv("EYE_MONITOR_INDEX", "2")
    monkeypatch.delenv("EYE_MONITOR_INDICES", raising=False)

    eye = EyeModule()
    event = asyncio.run(eye.capture_once())

    assert captured == []
    assert single_monitor_calls == [2]
    assert event.screenshot_path.startswith(str(paths.eye_dir))


def test_capture_separated_images_writes_single_monitor_when_only_one_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = RunStateManager(runs_root=tmp_path)
    paths = manager.init_run("single-brain-test", "test_single_brain")
    set_runtime_env(paths.root, paths.root.name)

    captured: list[Path] = []
    single_monitor_calls: list[int] = []

    def _fake_capture_all_screens(dest: Path) -> None:
        captured.append(dest)

    def _fake_capture_monitor(
        *,
        dest: Path,
        monitor_index: int,
        include_cursor: bool = False,
    ) -> int:
        single_monitor_calls.append(monitor_index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"single")
        return monitor_index

    import src.eye.module as eye_module

    monkeypatch.setattr(eye_module, "capture_all_screens_to_file", _fake_capture_all_screens)
    monkeypatch.setattr(eye_module, "capture_monitor_to_file", _fake_capture_monitor)
    monkeypatch.setenv("EYE_MONITOR_INDEX", "2")
    monkeypatch.delenv("EYE_MONITOR_INDICES", raising=False)

    eye = EyeModule()
    image_paths = asyncio.run(eye.capture_separated_images())

    assert len(image_paths) == 1
    assert captured == []
    assert single_monitor_calls == [2]
    assert image_paths[0].startswith(str(paths.eye_dir))


def test_capture_separated_images_writes_one_combined_eye_screenshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = RunStateManager(runs_root=tmp_path)
    paths = manager.init_run("combined-eye-test", "test_combined_eye")
    set_runtime_env(paths.root, paths.root.name)

    captured: list[Path] = []

    def _fake_capture_all_screens(dest: Path) -> None:
        captured.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"combined")

    import src.eye.module as eye_module

    monkeypatch.setattr(eye_module, "capture_all_screens_to_file", _fake_capture_all_screens)
    monkeypatch.setenv("EYE_MONITOR_INDICES", "1,2")

    eye = EyeModule()
    image_paths = asyncio.run(eye.capture_separated_images())

    assert len(image_paths) == 1
    assert len(captured) == 1
    assert captured[0].parent == paths.eye_dir
    assert image_paths[0] == str(captured[0])
