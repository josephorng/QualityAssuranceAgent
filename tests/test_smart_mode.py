"""Tests for smart-mode helpers, multimodal translation, and Plan→Act→Verify loop."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.models import SmartPlannerDecision, SmartRuntimeState, SmartVerifierDecision
from src.common.prompting import get_prompt
from src.common.runtime_context import (
    SMART_GOAL_ENV,
    SMART_MODE_ENV,
    USE_TOOL_CACHE_ENV,
    is_smart_mode,
    get_smart_goal,
    use_tool_cache_enabled,
)
from src.common.smart_mode import normalize_smart_goal, resolve_hub_run_mode
from src.common.vllm_client import _content_with_images, _encode_image_data_url, _translate_messages_to_openai
from src.runtime.smart_coordinator import (
    SmartCoordinator,
    _format_available_tools,
    _parse_planner,
    _parse_verifier,
)


def test_normalize_smart_goal_keeps_multiline() -> None:
    text = "  Open Edge\nThen search for docs  \n"
    assert normalize_smart_goal(text) == "Open Edge\nThen search for docs"


def test_smart_prompts_format_literal_json_examples() -> None:
    plan = get_prompt("brain_smart_plan").format(
        goal="goal",
        current_state="state",
        history="history",
        available_tools="- click(button?, instruction?)",
        ocr_text="ocr",
    )
    verify = get_prompt("brain_smart_verify").format(
        goal="goal",
        current_state="state",
        instruction="instruction",
        expected_outcome="expected",
        actor_result="result",
        ocr_text="ocr",
    )

    assert '{"status":"continue"' in plan
    assert "click(button?, instruction?)" in plan
    assert '{"outcome":"succeeded"' in verify


def test_format_available_tools_uses_live_mcp_metadata() -> None:
    tools = [
        SimpleNamespace(
            name="move_mouse",
            description="Move to a visible target.",
            inputSchema={
                "properties": {
                    "instruction": {"type": "string"},
                    "nearby_objects": {"type": "array"},
                },
                "required": ["instruction"],
            },
        ),
        SimpleNamespace(
            name="click",
            description="\n Single click at the current cursor. \n",
            inputSchema={"properties": {"button": {"type": "string"}}},
        ),
    ]

    catalog = _format_available_tools(tools)

    assert "- move_mouse(instruction, nearby_objects?): Move to a visible target." in catalog
    assert "- click(button?): Single click at the current cursor." in catalog


@pytest.mark.parametrize(
    ("tab", "has_steps", "expected"),
    [
        ("智能模式", False, "smart"),
        ("智能模式", True, "smart"),
        ("佇列執行", True, "queue"),
        ("單一腳本", True, "script"),
        ("單一腳本", False, "runtime"),
    ],
)
def test_resolve_hub_run_mode(tab: str, has_steps: bool, expected: str) -> None:
    assert resolve_hub_run_mode(selected_tab=tab, script_has_steps=has_steps) == expected


def test_is_smart_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    assert is_smart_mode() is False
    monkeypatch.setenv(SMART_MODE_ENV, "1")
    assert is_smart_mode() is True
    monkeypatch.setenv(SMART_GOAL_ENV, "do the thing")
    assert get_smart_goal() == "do the thing"


def test_use_tool_cache_disabled_in_smart_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USE_TOOL_CACHE_ENV, "1")
    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    assert use_tool_cache_enabled() is True
    monkeypatch.setenv(SMART_MODE_ENV, "1")
    assert use_tool_cache_enabled() is False


def test_parse_planner_and_verifier() -> None:
    plan = _parse_planner(
        json.dumps(
            {
                "status": "continue",
                "instruction": "Click Start",
                "expected_outcome": "Start menu opens",
                "rationale": "Need the menu",
            }
        )
    )
    assert isinstance(plan, SmartPlannerDecision)
    assert plan.instruction == "Click Start"

    verify = _parse_verifier(
        json.dumps(
            {
                "outcome": "succeeded",
                "updated_state": "Start menu visible",
                "branch": "advance",
                "reason": "Menu is open",
                "corrected_instruction": None,
            }
        )
    )
    assert isinstance(verify, SmartVerifierDecision)
    assert verify.branch == "advance"


def test_planner_continue_requires_instruction() -> None:
    with pytest.raises(Exception):
        SmartPlannerDecision(status="continue", instruction=None, rationale="x")


def test_vllm_image_data_url(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    # Minimal valid-ish PNG header bytes; encoder only needs readable bytes.
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    url = _encode_image_data_url(image)
    assert url.startswith("data:image/png;base64,")
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload).startswith(b"\x89PNG")


def test_vllm_translate_messages_embeds_images(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(b"jpeg-bytes")
    messages = [
        {
            "role": "user",
            "content": "Describe",
            "images": [str(image)],
        }
    ]
    out = _translate_messages_to_openai(messages)
    assert len(out) == 1
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_content_with_images_without_paths_is_plain_text() -> None:
    assert _content_with_images("hello", None) == "hello"
    assert _content_with_images("hello", []) == "hello"


class _FakeManager:
    def __init__(self, root: Path) -> None:
        self.session_end_reason: str | None = None
        self.logs: list[str] = []
        self._root = root

    def log_info(self, message: str) -> None:
        self.logs.append(message)

    def log_error(self, message: str) -> None:
        self.logs.append(f"ERROR:{message}")

    def set_session_end_reason(self, reason: str) -> None:
        self.session_end_reason = reason

    def require_paths(self) -> Any:
        class _Paths:
            def __init__(self, root: Path) -> None:
                self.root = root

        return _Paths(self._root)


class _FakeBrain:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute_instruction(self, instruction: str) -> bool:
        self.calls.append(instruction)
        return True


class _FakeContext:
    def __init__(self) -> None:
        self.screenshot_paths = ["shot.png"]
        self.ocr_text = "[index 0] class=文字 text='Start'"
        self.candidate_count = 1
        self.monitor_indices = [1]

    def to_log_dict(self) -> dict[str, Any]:
        return {"screenshot_paths": self.screenshot_paths, "candidate_count": 1}


def test_smart_coordinator_plan_act_verify_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SMART_MODE_ENV, "1")
    monkeypatch.setenv(SMART_GOAL_ENV, "Open Start and finish")

    manager = _FakeManager(tmp_path)
    brain = _FakeBrain()
    coordinator = SmartCoordinator.__new__(SmartCoordinator)
    coordinator.settings = type("S", (), {"smart_max_cycles": 5, "smart_max_recovery_attempts": 3})()
    coordinator.manager = manager
    coordinator.brain = brain
    coordinator.max_cycles = 5
    coordinator.max_recovery = 3
    coordinator.state = SmartRuntimeState(goal="Open Start and finish", current_state="(initial)")

    plan_calls = {"n": 0}

    async def fake_capture() -> _FakeContext:
        return _FakeContext()

    async def fake_plan(_context: _FakeContext) -> SmartPlannerDecision:
        plan_calls["n"] += 1
        if plan_calls["n"] == 1:
            return SmartPlannerDecision(
                status="continue",
                instruction="Click Start",
                expected_outcome="Start menu opens",
                rationale="Need start",
            )
        return SmartPlannerDecision(
            status="completed",
            instruction=None,
            expected_outcome="",
            rationale="Goal done",
        )

    async def fake_verify(**_kwargs: Any) -> SmartVerifierDecision:
        return SmartVerifierDecision(
            outcome="succeeded",
            updated_state="Start menu is open",
            branch="advance",
            reason="Visible",
            corrected_instruction=None,
        )

    async def fake_wait() -> None:
        return None

    monkeypatch.setattr(coordinator, "_capture_context", fake_capture)
    monkeypatch.setattr(coordinator, "_plan", fake_plan)
    monkeypatch.setattr(coordinator, "_verify", fake_verify)
    monkeypatch.setattr(coordinator, "_wait_if_paused", fake_wait)

    asyncio.run(coordinator.run())

    assert brain.calls == ["Click Start"]
    assert manager.session_end_reason == "completed"
    assert coordinator.state.current_state == "Start menu is open"
    assert coordinator.state.terminal_reason == "completed"
    assert (tmp_path / "smart_state.json").is_file()
    events = (tmp_path / "smart_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) >= 3
    phases = [json.loads(line)["phase"] for line in events]
    assert "plan" in phases and "act" in phases and "verify" in phases


def test_smart_coordinator_recovery_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeManager(tmp_path)
    brain = _FakeBrain()
    coordinator = SmartCoordinator.__new__(SmartCoordinator)
    coordinator.settings = type("S", (), {"smart_max_cycles": 10, "smart_max_recovery_attempts": 2})()
    coordinator.manager = manager
    coordinator.brain = brain
    coordinator.max_cycles = 10
    coordinator.max_recovery = 2
    coordinator.state = SmartRuntimeState(goal="Impossible", current_state="(initial)")

    async def fake_capture() -> _FakeContext:
        return _FakeContext()

    async def fake_plan(_context: _FakeContext) -> SmartPlannerDecision:
        return SmartPlannerDecision(
            status="continue",
            instruction="Try again",
            expected_outcome="success",
            rationale="retrying",
        )

    async def fake_verify(**_kwargs: Any) -> SmartVerifierDecision:
        return SmartVerifierDecision(
            outcome="failed",
            updated_state="",
            branch="replan",
            reason="still broken",
            corrected_instruction=None,
        )

    async def fake_wait() -> None:
        return None

    monkeypatch.setattr(coordinator, "_capture_context", fake_capture)
    monkeypatch.setattr(coordinator, "_plan", fake_plan)
    monkeypatch.setattr(coordinator, "_verify", fake_verify)
    monkeypatch.setattr(coordinator, "_wait_if_paused", fake_wait)

    asyncio.run(coordinator.run())
    assert manager.session_end_reason == "step_failed"
    assert coordinator.state.terminal_reason == "recovery_budget_exhausted"
    assert coordinator.state.recovery_attempts > 2


def test_session_report_includes_smart_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.common.runtime_context import SCRIPT_PATH_ENV, SMART_GOAL_ENV, SMART_MODE_ENV
    from src.common.session_report import build_session_report

    monkeypatch.delenv(SMART_MODE_ENV, raising=False)
    monkeypatch.delenv(SMART_GOAL_ENV, raising=False)
    monkeypatch.delenv(SCRIPT_PATH_ENV, raising=False)

    (tmp_path / "smart_state.json").write_text(
        json.dumps({"goal": "Do X", "terminal_reason": "completed", "cycle": 1}),
        encoding="utf-8",
    )
    events = [
        {"phase": "plan", "cycle": 1, "instruction": "Click A", "rationale": "go"},
        {"phase": "act", "cycle": 1, "ok": True, "reason": "done"},
        {"phase": "verify", "cycle": 1, "branch": "advance", "outcome": "succeeded"},
    ]
    (tmp_path / "smart_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )
    report = build_session_report(tmp_path, session_end_reason="completed")
    assert report.get("run_mode") == "smart"
    assert report.get("smart_goal") == "Do X"
    assert len(report.get("smart_cycles", [])) == 1
    assert report["summary"].get("smart_cycle_count") == 1


def test_session_report_records_smart_goal_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.common.runtime_context import SCRIPT_PATH_ENV, SMART_GOAL_ENV, SMART_MODE_ENV
    from src.common.session_report import build_session_report

    goal_file = tmp_path / "open_outlook.txt"
    goal_file.write_text("Open Outlook\n", encoding="utf-8")
    monkeypatch.setenv(SMART_MODE_ENV, "1")
    monkeypatch.setenv(SMART_GOAL_ENV, "Open Outlook")
    monkeypatch.setenv(SCRIPT_PATH_ENV, str(goal_file))

    report = build_session_report(tmp_path, session_end_reason="completed")
    assert report.get("run_mode") == "smart"
    assert report.get("script_name") == "open_outlook.txt"
    assert report.get("script_path") == str(goal_file)
    assert report.get("smart_goal") == "Open Outlook"
