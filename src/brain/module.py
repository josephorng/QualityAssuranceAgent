from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from cua_mcp.hand_tools import close_hwnd
from cua_mcp.select_mouse_target import (
    RECORDED_ANCHOR_MISS_MESSAGE,
    require_similarity_match,
)
from cua_mcp.tools import (
    TOOL_FUNCTIONS,
    get_mode_tool_functions,
    mcp_server,
)
from ollama import Message
from pydantic import ValidationError
from src.common.io_utils import write_json
from cua_mcp.llm_json import extract_json_object_string, parse_json_object, repair_json_object_text
from src.common.models import (
    BrainStepOutcome,
    BrainTaskState,
    ExecutionResult,
    ScriptStepVerifyResult,
    ToolCommand,
)
from src.common.instruction_tool_cache import (
    extract_tool_calls_from_messages,
    lookup_tool_calls,
    upsert_tool_calls,
)
from src.common.llm_factory import get_llm_client
from src.common.llm_text import strip_llm_quote_wrappers
from src.common.nearby_side import enrich_tool_arguments_from_goal
from src.common.prompting import get_prompt
from src.common.run_control import (
    notify_active_step,
    publish_script_steps,
    take_recording_replay_dirty,
)
from src.common.run_state import (
    get_run_state_manager,
    parse_step_artifact_stem,
    step_artifact_name,
    step_filenames_are_one_based,
)
from src.common.runtime_context import (
    SCRIPT_BASELINE_AFTER_ENV,
    SCRIPT_LINES_ENV,
    SCRIPT_OUTCOMES_ENV,
    SCRIPT_PATH_ENV,
    SCRIPT_SETTLE_AFTER_ENV,
    SCRIPT_WINDOW_VERIFY_ENV,
    get_runtime_env,
    is_runtime_command_mode,
    is_smart_mode,
    skip_verification_enabled,
    use_tool_cache_enabled,
)
from src.common.settings import load_settings
from src.recorder.verify_signals import (
    CLICK_KINDS,
    capture_replay_after_signals,
    predicate_has_signal_assertions,
)
from src.recorder.window_snapshot import (
    WindowInfo,
    build_window_verify_predicate,
    disappeared_windows_still_present,
    omit_foreground_when_structural,
    snapshot_top_level_windows,
    window_verify_has_assertions,
    window_verify_satisfied,
)

# Window-verify polls continuously after actions; deadline is 2.0x of settle when
# settle is at least ``_SETTLE_POLL_MULTI_MIN_SECONDS``, else 1.0x.
_SETTLE_POLL_MULTI_MIN_SECONDS = 2.0
_WINDOW_VERIFY_POLL_INTERVAL_SECONDS = 0.1
# Appeared asserts floor the poll settle so 2.0x lands at least at 10s (slow launches).
_APPEARED_POLL_MIN_SECONDS = 5.0
# Disappeared asserts floor a short poll so async closes (e.g. Chrome WM_CLOSE)
# are not judged on a single immediate after-snapshot when settle is 0/None.
_DISAPPEARED_POLL_MIN_SECONDS = 1.0
# After a successful close_windows, a disappeared miss re-closes the matching
# window until the check passes. The wait is at least this long, and stays the
# settle poll deadline when that deadline is already longer.
_DISAPPEARED_CLOSE_RETRY_MIN_SECONDS = 10.0
# Minimum gap between those close attempts, measured in poll sleep time.
_DISAPPEARED_CLOSE_RETRY_INTERVAL_SECONDS = 0.5


def _reason_is_disappeared_miss(reason: str) -> bool:
    return reason.startswith("window verify missed disappeared")


def _predicate_has_disappeared(recorded: dict[str, Any]) -> bool:
    disappeared = recorded.get("disappeared")
    return isinstance(disappeared, list) and bool(disappeared)


def _close_retry_due(
    elapsed: float,
    last_close_elapsed: float | None,
    interval: float,
) -> bool:
    """True when no close has been tried, or the last try was ``interval`` ago."""
    if last_close_elapsed is None:
        return True
    return (elapsed - last_close_elapsed) + 1e-9 >= interval
# Signal asserts (foreground, click_window, clipboard, caret, focused, scroll)
# share the same short floor: Start/SearchHost and similar focus changes often
# lag the click by tens–hundreds of ms, and a tiny recording settle must not
# collapse to a single immediate sample. Kept under ``_SETTLE_POLL_MULTI_MIN_SECONDS``
# so the deadline stays 1.0x (1s), not 2.0x (4s).
_SIGNAL_POLL_MIN_SECONDS = 1.0


ROLE_USER = "user"
ROLE_TOOL = "tool"
ROLE_ASSISTANT = "assistant"
ROLE_THINKING = "thinking"

if TYPE_CHECKING:
    from src.eye.module import EyeModule
    from src.hand.module import HandModule

_MAX_INNER_DECIDE_STEPS = 10
# Models sometimes invent these as tool calls when asked to end the step with JSON.
_PSEUDO_END_TOOL_NAMES = frozenset({"finish", "done", "complete", "end", "end_step"})


# Remove key-value pairs with None values in each message
def prune_nulls(d):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if v is not None}


def stamp_message(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a chat message with `timestamp_utc` set if missing."""
    stamped = prune_nulls(message)
    if "timestamp_utc" not in stamped:
        stamped["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return stamped

@dataclass
class BrainRuntime:
    """Lightweight flags for brain task lifecycle (reserved for future use)."""

    active: BrainTaskState | None = None
    finished: bool = False
    processing: bool = False


@dataclass
class BrainStepResult:
    """Outcome of one `BrainModule.process_step()` call."""

    reason: str = ""
    step_finished: bool = False
    run_complete: bool = False
    step_index: int | None = None


class BrainModule:
    """Orchestrates scripted tasks: vision + LLM tool loop, then window/signal verification and script branching."""

    def __init__(
        self,
        hand: HandModule | None = None,
        eye: EyeModule | None = None,
    ) -> None:
        """Initialize run state, load script lines from the environment, and retain hand/eye modules."""
        self.settings = load_settings()
        self.ollama = get_llm_client()
        self.run_root, self.run_id = get_runtime_env()
        self.manager = get_run_state_manager()
        self.manager.init_run(self.run_id, self.run_root.name)
        self.runtime = BrainRuntime()
        self.script_lines = (
            []
            if is_runtime_command_mode() or is_smart_mode()
            else self._script_seed_steps()
        )
        self.script_expected_outcomes = (
            []
            if is_runtime_command_mode() or is_smart_mode()
            else self._script_seed_outcomes(len(self.script_lines))
        )
        self.script_baseline_after_paths = (
            []
            if is_runtime_command_mode() or is_smart_mode()
            else self._script_seed_baseline_after_paths(len(self.script_lines))
        )
        self.script_settle_after_seconds = (
            []
            if is_runtime_command_mode() or is_smart_mode()
            else self._script_seed_settle_after_seconds(len(self.script_lines))
        )
        self.script_window_verifies = (
            []
            if is_runtime_command_mode() or is_smart_mode()
            else self._script_seed_window_verifies(len(self.script_lines))
        )
        self._script_step_index = 0
        # Recording event id for each script step. Filled on the first disk reload.
        self._script_event_indices: list[int | None] = []
        self._hand = hand
        self._eye = eye
        # Last-tool after-screenshot pending until the next step's before seals it
        # (or a final capture at run end / failure).
        self._pending_after_capture: bool = False
        self._replay_press_point_window: dict[str, Any] | None = None
        # Set when cache-replay move_mouse fails. process_step holds the step
        # so the overlay can count down and retry, instead of ending the run.
        self._hold_step_retry_reason: str | None = None
        # Set when this step's close_windows succeeded, so a disappeared miss
        # can re-close that window during verify.
        self._step_close_windows_ok = False
        self._step_transcript_counter = (
            self._resume_step_transcript_counter()
            if is_runtime_command_mode() or is_smart_mode()
            else 0
        )
        self.manager.log_info(f"Brain module initialized run_id={self.run_id}")

    def _resume_step_transcript_counter(self) -> int:
        """Continue numbering runtime steps after script steps in the same run folder."""
        steps_dir = self.manager.require_paths().root / "steps"
        if not steps_dir.is_dir():
            return 0
        one_based = step_filenames_are_one_based(steps_dir)
        max_tc = -1
        for path in steps_dir.iterdir():
            if path.suffix not in (".json", ".log"):
                continue
            parsed = parse_step_artifact_stem(path.stem, one_based=one_based)
            if parsed is None:
                continue
            max_tc = max(max_tc, parsed[0])
        return max_tc + 1

    def _script_step_attempt_number(self, script_step_index: int) -> int:
        """1-based attempt number for ``script_step_index`` (counts existing ``steps/*_<index>.json``)."""
        steps_dir = self.manager.require_paths().root / "steps"
        if not steps_dir.is_dir():
            return 1
        one_based = step_filenames_are_one_based(steps_dir)
        count = 0
        for path in steps_dir.iterdir():
            if path.suffix != ".json":
                continue
            parsed = parse_step_artifact_stem(path.stem, one_based=one_based)
            if parsed is not None and parsed[1] == script_step_index:
                count += 1
        return count + 1

    def _script_step_smart_recovery_count(self, script_step_index: int) -> int:
        """Count prior transcripts for this line whose verify branch was ``smart``."""
        steps_dir = self.manager.require_paths().root / "steps"
        if not steps_dir.is_dir():
            return 0
        one_based = step_filenames_are_one_based(steps_dir)
        count = 0
        for path in steps_dir.iterdir():
            if path.suffix != ".json":
                continue
            parsed = parse_step_artifact_stem(path.stem, one_based=one_based)
            if parsed is None or parsed[1] != script_step_index:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            timing = payload.get("step_timing")
            if not isinstance(timing, dict):
                continue
            verify = timing.get("verify")
            if isinstance(verify, dict) and verify.get("branch") == "smart":
                count += 1
        return count

    def _script_step_retry_limit_reached(self, script_step_index: int) -> bool:
        max_attempts = int(self.settings.script_max_step_attempts)
        if max_attempts <= 0:
            return False
        return self._script_step_attempt_number(script_step_index) >= max_attempts

    def _save_step_messages(self, messages: list[dict[str, Any]]) -> None:
        """Save or update the decide-loop transcript under `steps/<tc>_<si>.json` (1-based names)."""
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / step_artifact_name(
            self._step_transcript_counter, self._script_step_index
        )
        payload: dict[str, Any] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
        payload["messages"] = [stamp_message(msg) for msg in messages]
        write_json(out_path, payload)

    def _update_step_metadata(
        self,
        transcript_counter: int,
        script_step_index: int,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert `step_timing` metadata for one step transcript file."""
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / step_artifact_name(transcript_counter, script_step_index)

        payload: dict[str, Any] = {}
        existing_metadata: dict[str, Any] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
                    existing_timing = existing.get("step_timing")
                    if isinstance(existing_timing, dict):
                        existing_metadata = dict(existing_timing)
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
                existing_metadata = {}

        existing_metadata.update(metadata)
        payload["step_timing"] = existing_metadata
        write_json(out_path, payload)

    def _save_interrupted_step_record(
        self,
        transcript_counter: int,
        script_step_index: int,
        started_iso: str,
        started_at: float,
        reason: str,
    ) -> None:
        """Write goal and timing when a step stops before its normal metadata update."""
        finished_iso = datetime.now(timezone.utc).isoformat()
        duration_seconds = round(perf_counter() - started_at, 3)
        self.manager.log_info(reason)
        goal = ""
        try:
            goal = self._current_goal()
        except Exception:
            goal = ""
        expected = ""
        try:
            expected = self._current_expected_outcome()
        except Exception:
            expected = ""
        self._update_step_metadata(
            transcript_counter,
            script_step_index,
            {
                "started_at_utc": started_iso,
                "finished_at_utc": finished_iso,
                "duration_seconds": duration_seconds,
                "status": "failed",
                "step_index": script_step_index,
                "goal": goal,
                "expected_outcome": expected or None,
            },
        )

    def _append_step_messages(
        self,
        messages: list[dict[str, Any]],
        transcript_counter: int,
        script_step_index: int,
        attribute_name: str = "messages",
    ) -> None:
        """Append messages under `attribute_name` in a step transcript file."""
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / step_artifact_name(transcript_counter, script_step_index)

        merged_messages: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
                    existing_messages = existing.get(attribute_name)
                    if isinstance(existing_messages, list):
                        merged_messages.extend(existing_messages)
            except (OSError, ValueError, json.JSONDecodeError):
                # If the existing transcript is malformed/unreadable, keep only new messages
                # under the requested attribute.
                merged_messages = []
                payload = {}

        merged_messages.extend([stamp_message(msg) for msg in messages])
 
        payload[attribute_name] = merged_messages
        write_json(out_path, payload)

    def _append_failed_tool_call(
        self,
        tool_name: str,
        transcript_counter: int,
        script_step_index: int,
    ) -> None:
        """Append a failed tool name to `failed_tool_calls` in a step transcript file."""
        steps_dir = self.manager.require_paths().root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = steps_dir / step_artifact_name(transcript_counter, script_step_index)

        payload: dict[str, Any] = {}
        failed_tool_calls: list[str] = []
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
                    existing_failed = existing.get("failed_tool_calls")
                    if isinstance(existing_failed, list):
                        failed_tool_calls.extend(
                            item for item in existing_failed if isinstance(item, str)
                        )
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
                failed_tool_calls = []

        failed_tool_calls.append(tool_name)
        payload["failed_tool_calls"] = failed_tool_calls
        write_json(out_path, payload)

    def reload_pending_recording_steps(self) -> None:
        """Replace steps that have not started with the recording's saved settings.

        The step already inside ``process_step`` keeps the copy it started with.
        Pause is checked between steps, so edits made while paused apply to the
        next step and every step after it. Skips disk when the recording is not
        marked dirty (hub edit / run start).
        """
        if is_runtime_command_mode() or is_smart_mode():
            return
        if not take_recording_replay_dirty():
            return
        script_path_raw = (os.environ.get(SCRIPT_PATH_ENV) or "").strip()
        if not script_path_raw:
            return
        from src.common.script_helper import (
            load_recording_replay_steps,
            merge_unstarted_recording_steps,
            recording_run_dir,
        )

        recording_dir = recording_run_dir(Path(script_path_raw))
        if recording_dir is None:
            return
        try:
            fresh = load_recording_replay_steps(recording_dir)
        except Exception as exc:
            self.manager.log_info(f"Recording reload skipped: {exc}")
            return
        merged = merge_unstarted_recording_steps(
            step_index=self._script_step_index,
            current_lines=list(self.script_lines),
            current_outcomes=list(self.script_expected_outcomes),
            current_baselines=list(self.script_baseline_after_paths),
            current_settles=list(self.script_settle_after_seconds),
            current_verifies=[
                dict(item) if isinstance(item, dict) else {}
                for item in self.script_window_verifies
            ],
            current_event_indices=list(self._script_event_indices),
            fresh_lines=fresh[0],
            fresh_outcomes=fresh[1],
            fresh_baselines=fresh[2],
            fresh_settles=fresh[3],
            fresh_verifies=fresh[4],
            fresh_event_indices=fresh[5],
        )
        content_same = (
            merged[0] == list(self.script_lines)
            and merged[1] == list(self.script_expected_outcomes)
            and merged[2] == list(self.script_baseline_after_paths)
            and merged[3] == list(self.script_settle_after_seconds)
            and merged[4] == [
                dict(item) if isinstance(item, dict) else {}
                for item in self.script_window_verifies
            ]
        )
        self._script_event_indices = list(merged[5])
        if content_same:
            return
        self.script_lines = list(merged[0])
        self.script_expected_outcomes = list(merged[1])
        self.script_baseline_after_paths = list(merged[2])
        self.script_settle_after_seconds = list(merged[3])
        self.script_window_verifies = list(merged[4])
        pending = max(0, len(self.script_lines) - self._script_step_index)
        self.manager.log_info(
            f"Applied recording edits to {pending} unstarted step(s); "
            f"next step {self._script_step_index + 1}/{max(len(self.script_lines), 1)}"
        )

    def jump_to_script_step(self, index: int) -> None:
        """Run ``index`` next. The step already in progress is left to finish."""
        if not self.script_lines:
            return
        target = max(0, min(int(index), len(self.script_lines) - 1))
        if target == self._script_step_index:
            return
        self.manager.log_info(
            f"Replay jump: step {self._script_step_index + 1} -> {target + 1}"
        )
        self._script_step_index = target

    def _script_seed_steps(self) -> list[str]:
        """Load non-empty script lines from `SCRIPT_LINES_ENV` (JSON array of strings)."""
        raw = os.environ.get(SCRIPT_LINES_ENV, "")
        payload: list[str] | None = None
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    payload = [item for item in parsed if isinstance(item, str)]
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            raise RuntimeError("No script steps found")
        lines: list[str] = []
        for item in payload:
            cleaned = item.strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    def _script_seed_outcomes(self, step_count: int) -> list[str | None]:
        """Load optional expected outcomes aligned with script steps."""
        raw = os.environ.get(SCRIPT_OUTCOMES_ENV, "")
        outcomes: list[str | None] = [None] * step_count
        if not raw:
            return outcomes
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return outcomes
        if not isinstance(parsed, list):
            return outcomes
        for index in range(min(step_count, len(parsed))):
            item = parsed[index]
            if isinstance(item, str) and item.strip():
                outcomes[index] = item.strip()
        return outcomes

    def _script_seed_baseline_after_paths(self, step_count: int) -> list[str | None]:
        """Load optional recording after-screenshot paths aligned with script steps."""
        raw = os.environ.get(SCRIPT_BASELINE_AFTER_ENV, "")
        baselines: list[str | None] = [None] * step_count
        if not raw:
            return baselines
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return baselines
        if not isinstance(parsed, list):
            return baselines
        for index in range(min(step_count, len(parsed))):
            item = parsed[index]
            if isinstance(item, str) and item.strip():
                path = Path(item.strip())
                baselines[index] = str(path) if path.is_file() else None
        return baselines

    def _script_seed_settle_after_seconds(self, step_count: int) -> list[float | None]:
        """Load optional post-action settle delays aligned with script steps."""
        raw = os.environ.get(SCRIPT_SETTLE_AFTER_ENV, "")
        settles: list[float | None] = [None] * step_count
        if not raw:
            return settles
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return settles
        if not isinstance(parsed, list):
            return settles
        for index in range(min(step_count, len(parsed))):
            item = parsed[index]
            if isinstance(item, bool) or item is None:
                continue
            if isinstance(item, (int, float)):
                value = float(item)
                if value > 0:
                    settles[index] = value
        return settles

    def _script_seed_window_verifies(self, step_count: int) -> list[dict[str, Any]]:
        """Load recorded window-diff predicates aligned with script steps."""
        verifies: list[dict[str, Any]] = [{} for _ in range(step_count)]
        raw = os.environ.get(SCRIPT_WINDOW_VERIFY_ENV, "")
        if not raw:
            return verifies
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return verifies
        if not isinstance(parsed, list):
            return verifies
        for index in range(min(step_count, len(parsed))):
            item = parsed[index]
            if isinstance(item, dict) and window_verify_has_assertions(item):
                verifies[index] = omit_foreground_when_structural(item)
        return verifies

    def _current_expected_outcome(self) -> str:
        if not self.script_expected_outcomes:
            return ""
        if self._script_step_index >= len(self.script_expected_outcomes):
            return ""
        value = self.script_expected_outcomes[self._script_step_index]
        return value.strip() if isinstance(value, str) else ""

    def _current_baseline_after_path(self) -> str | None:
        if not self.script_baseline_after_paths:
            return None
        if self._script_step_index >= len(self.script_baseline_after_paths):
            return None
        value = self.script_baseline_after_paths[self._script_step_index]
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value.strip())
        return str(path) if path.is_file() else None

    def _current_settle_after_seconds(self) -> float | None:
        if not self.script_settle_after_seconds:
            return None
        if self._script_step_index >= len(self.script_settle_after_seconds):
            return None
        value = self.script_settle_after_seconds[self._script_step_index]
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
        return None

    def _current_window_verify(self) -> dict[str, Any]:
        verifies = getattr(self, "script_window_verifies", None)
        if not verifies:
            return {}
        if self._script_step_index >= len(verifies):
            return {}
        value = verifies[self._script_step_index]
        if isinstance(value, dict) and window_verify_has_assertions(value):
            return value
        return {}

    async def _sleep_settle_after(self, settle_after: float) -> None:
        """Block on this step's recording settle gap (when window-verify does not own it)."""
        seconds = float(settle_after)
        if seconds <= 0:
            return
        waited = round(seconds, 3)
        self.manager.log_info(f"Settle after step: waiting {waited:.3f}s")
        await asyncio.sleep(seconds)

    async def _flush_pending_after_capture(self) -> None:
        """Capture a dedicated after-screenshot when no next-step before will seal it."""
        if not getattr(self, "_pending_after_capture", False):
            return
        self._pending_after_capture = False
        hand = getattr(self, "_hand", None)
        if hand is None:
            return
        try:
            await hand.capture_after_for_last_action()
        except Exception as exc:
            self.manager.log_error(
                f"Failed to capture deferred after-action screenshot: {exc}"
            )

    async def _seal_pending_after_with_before(
        self,
        before_screenshot: str | None,
    ) -> None:
        """Reuse this step's before shot as the previous last-tool after frame."""
        if not getattr(self, "_pending_after_capture", False):
            return
        self._pending_after_capture = False
        hand = getattr(self, "_hand", None)
        if hand is None or not before_screenshot:
            return
        try:
            await hand.assign_after_for_last_action(before_screenshot)
        except Exception as exc:
            self.manager.log_error(
                f"Failed to assign next-before as previous after-screenshot: {exc}"
            )

    def _mark_pending_after_capture(self) -> None:
        self._pending_after_capture = True

    def prepare_runtime_step(self, command: str) -> None:
        """Set a single script line for the next `process_step()` (runtime command mode)."""
        cleaned = command.strip()
        if not cleaned:
            raise ValueError("runtime step command must be non-empty")
        self.script_lines = [cleaned]
        self.script_expected_outcomes = [None]
        self.script_baseline_after_paths = [None]
        self.script_settle_after_seconds = [None]
        self.script_window_verifies = [{}]
        self._script_step_index = 0

    async def execute_instruction(self, instruction: str) -> bool:
        """
        Run the existing multi-tool decide/act loop against one bounded instruction.

        Used by smart mode as the Act step. Preserves surrounding script state and
        advances the step transcript counter after the actor finishes.
        """
        cleaned = instruction.strip()
        if not cleaned:
            raise ValueError("instruction must be non-empty")

        saved_lines = list(self.script_lines)
        saved_outcomes = list(self.script_expected_outcomes)
        saved_baselines = list(self.script_baseline_after_paths)
        saved_settles = list(self.script_settle_after_seconds)
        saved_verifies = list(getattr(self, "script_window_verifies", []))
        saved_index = self._script_step_index
        transcript_counter = self._step_transcript_counter
        script_step_index = 0
        self.script_lines = [cleaned]
        self.script_expected_outcomes = [None]
        self.script_baseline_after_paths = [None]
        self.script_settle_after_seconds = [None]
        self.script_window_verifies = [{}]
        self._script_step_index = 0
        self.manager.set_step_log_context(transcript_counter, script_step_index)
        started_iso = datetime.now(timezone.utc).isoformat()
        started_at = perf_counter()
        try:
            step_succeeded = await self.loop()
            finished_iso = datetime.now(timezone.utc).isoformat()
            duration_seconds = round(perf_counter() - started_at, 3)
            self._update_step_metadata(
                transcript_counter,
                script_step_index,
                {
                    "started_at_utc": started_iso,
                    "finished_at_utc": finished_iso,
                    "duration_seconds": duration_seconds,
                    "status": "completed" if step_succeeded else "failed",
                    "step_index": script_step_index,
                    "goal": cleaned,
                },
            )
            self._step_transcript_counter += 1
            return step_succeeded
        finally:
            self.script_lines = saved_lines
            self.script_expected_outcomes = saved_outcomes
            self.script_baseline_after_paths = saved_baselines
            self.script_settle_after_seconds = saved_settles
            self.script_window_verifies = saved_verifies
            self._script_step_index = saved_index
            self.manager.clear_step_log_context()

    def undo_last_runtime_step(self) -> bool:
        """Remove the last completed step transcript and rewind the step counter."""
        if self._step_transcript_counter <= 0:
            return False
        tc = self._step_transcript_counter - 1
        si = 0
        steps_dir = self.manager.require_paths().root / "steps"
        for ext in (".json", ".log"):
            path = steps_dir / step_artifact_name(tc, si, suffix=ext)
            if path.exists():
                path.unlink()
        self._step_transcript_counter = tc
        self._script_step_index = 0
        return True

    def _current_goal(self) -> str:
        """Return the goal text for `_script_step_index`, or the last line if the index is past the end."""
        if not self.script_lines:
            raise RuntimeError("No script steps found")
        if self._script_step_index >= len(self.script_lines):
            return self.script_lines[-1]
        return self.script_lines[self._script_step_index]

    @staticmethod
    def _primary_decision_screenshot(image_paths: list[str]) -> str | None:
        if not image_paths:
            return None
        return image_paths[0]

    async def _flush_eye_pending_saves(self) -> None:
        """Await background eye PNG encodes so transcript paths exist on disk."""
        import inspect

        eye = getattr(self, "_eye", None)
        if eye is None:
            return
        flush = getattr(eye, "flush_pending_saves", None)
        if not callable(flush):
            return
        result = flush()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _normalize_tool_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Copy tool args and strip model quote-wrapper tokens like ``<|"|>``."""
        raw = dict(arguments) if isinstance(arguments, dict) else {}
        cleaned = strip_llm_quote_wrappers(raw)
        return cleaned if isinstance(cleaned, dict) else raw

    def _enrich_tool_arguments(
        self, tool_name: str, arguments: dict[str, Any], goal: str
    ) -> dict[str, Any]:
        """Restore directed nearby sides from the step goal when the model stripped them."""
        enriched = enrich_tool_arguments_from_goal(tool_name, arguments, goal)
        for key in (
            "nearby_objects",
            "start_nearby_objects",
            "destination_nearby_objects",
        ):
            before = arguments.get(key)
            after = enriched.get(key)
            if after is not None and after != before:
                self.manager.log_info(
                    f"{tool_name}: restored nearby sides from goal "
                    f"{key}={before!r} -> {after!r}"
                )
        return enriched

    async def _normalize_tool_name(self, tool_name: str, arguments: dict | None = None) -> str:
        """
        Normalize model-emitted tool names to canonical callable names.

        Some model/tool providers return names like `action:click` while local
        tool registration uses `click`.
        """
        candidate = tool_name.strip()
        tools = await mcp_server.list_tools()        
        tool_names = [tool.name for tool in tools]
        if candidate in tool_names:
            return candidate
        if ":" in candidate:
            suffix = candidate.rsplit(":", 1)[-1].strip()
            if suffix in tool_names:
                return suffix
        # Some providers return wrapper tokens (e.g. "type") and place the
        # real action in arguments.
        args = arguments if isinstance(arguments, dict) else {}
        for key in ("action", "tool", "name", "type"):
            value = args.get(key)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value in tool_names:
                return value
            if ":" in value:
                suffix = value.rsplit(":", 1)[-1].strip()
                if suffix in tool_names:
                    return suffix
        return candidate

    async def _validate_tool_functions_match_mcp(self) -> None:
        """Ensure MCP-registered tool names match `TOOL_FUNCTIONS`; raises if they diverge."""
        tool_names = {tool.name for tool in await mcp_server.list_tools()}
        tool_functions = {tool_function.__name__ for tool_function in TOOL_FUNCTIONS}
        if tool_names != tool_functions:
            only_in_tool_names = tool_names - tool_functions
            only_in_tool_functions = tool_functions - tool_names
            self.manager.log_error(
                f"Tool names and tool functions do not match.\n"
                f"Only in tool names: {only_in_tool_names}\n"
                f"Only in tool functions: {only_in_tool_functions}"
            )
            raise RuntimeError("Tool names and tool functions do not match")

    def sanitize_execution_result(self, result: ExecutionResult) -> dict[str, Any]:
        """Strip noisy or redundant fields from tool execution results before logging into chat."""
        result_dict = result.model_dump()
        result_dict.pop("timestamp")
        result_dict.pop("screenshot_name", None)
        result_dict.pop("screenshot_before_path", None)
        result_dict.pop("screenshot_after_path", None)
        result_dict = prune_nulls(result_dict)       
        return result_dict
    
    def sanitize_message(self, message: Message) -> dict[str, Any]:
        """Prepare an Ollama assistant message dict for persistence (drops empty keys and thinking role)."""
        message_dict = message.model_dump()
        message_dict.pop(ROLE_THINKING)
        pop_list = []
        for key, value in message_dict.items():
            if not value:
                pop_list.append(key)
        for key in pop_list:
                message_dict.pop(key)
        return message_dict

    @staticmethod
    def _parse_json_object_from_model_content(content: str) -> dict[str, Any]:
        """Parse a JSON object from model text, stripping optional markdown code fences."""
        return parse_json_object(
            content,
            empty_error="Model returned empty content; expected a JSON object",
            decode_error_prefix="Model JSON decode failed",
        )

    @staticmethod
    def _recover_step_outcome_payload(content: str) -> dict[str, Any] | None:
        """Best-effort extract of status/reason when model JSON is malformed.

        Handles common LLM mistakes such as unescaped quotes inside ``reason``.
        """
        text = repair_json_object_text(content) or extract_json_object_string(content)
        status_match = re.search(r'"status"\s*:\s*"(completed|failed)"', text)
        if not status_match:
            return None
        reason = ""
        reason_match = re.search(r'"reason"\s*:\s*"(.*)"\s*}\s*$', text, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1)
        return {"status": status_match.group(1), "reason": reason}

    @staticmethod
    def _is_pseudo_end_tool_name(tool_name: str | None) -> bool:
        """True for invented end-of-step tool names (not registered MCP tools)."""
        if not tool_name:
            return False
        return tool_name.strip().lower() in _PSEUDO_END_TOOL_NAMES

    def _parse_step_outcome(self, content: str | None) -> BrainStepOutcome | None:
        """Parse a structured step finish reply, or None if content is missing/invalid."""
        if not (content or "").strip():
            return None
        try:
            payload = self._parse_json_object_from_model_content(content or "")
            return BrainStepOutcome.model_validate(payload)
        except (ValueError, ValidationError, TypeError) as e:
            recovered = self._recover_step_outcome_payload(content or "")
            if recovered is not None:
                try:
                    outcome = BrainStepOutcome.model_validate(recovered)
                    self.manager.log_info(
                        "Step outcome JSON was malformed; recovered status/reason via fallback parser"
                    )
                    return outcome
                except (ValidationError, TypeError):
                    pass
            self.manager.log_error(f"Step outcome JSON parse/validation failed: {e}")
            return None

    def _parse_step_outcome_from_arguments(
        self, arguments: dict[str, Any] | None
    ) -> BrainStepOutcome | None:
        """Parse step outcome from a pseudo end-tool's arguments (status/reason)."""
        if not isinstance(arguments, dict):
            return None
        try:
            return BrainStepOutcome.model_validate(arguments)
        except (ValidationError, TypeError):
            status = arguments.get("status")
            if status not in ("completed", "failed"):
                return None
            reason = arguments.get("reason", "")
            if reason is None:
                reason = ""
            try:
                return BrainStepOutcome(status=status, reason=str(reason))
            except (ValidationError, TypeError):
                return None

    def _resolve_step_success_from_outcome(
        self,
        outcome: BrainStepOutcome | None,
        unresolved_tool_failures: set[str],
        *,
        missing_outcome_message: str,
    ) -> bool:
        """Apply a parsed step outcome to success/failure, logging the decision."""
        if outcome is None:
            self.manager.log_error(missing_outcome_message)
            return False
        if outcome.status == "completed":
            if unresolved_tool_failures:
                failed = ", ".join(sorted(unresolved_tool_failures))
                self.manager.log_info(
                    f"Model marked completed but unresolved tool failure(s) remain "
                    f"({failed}); treating step as failed. Model reason: {outcome.reason}"
                )
                return False
            self.manager.log_info(f"Step marked completed by model: {outcome.reason}")
            return True
        self.manager.log_info(f"Step marked failed by model: {outcome.reason}")
        return False

    def _apply_verify_branch(self, result: ScriptStepVerifyResult) -> bool:
        """Apply verification `branch` to `_script_step_index`. Returns whether all script lines are done.

        ``abort`` leaves the index unchanged; callers must treat it as a hard stop
        (``step_finished=False``) rather than continuing the script.
        """
        n = len(self.script_lines)
        idx = self._script_step_index

        if result.branch == "abort":
            self.manager.log_info(
                f"Verify: branch=abort; holding step and stopping run. {result.reason}"
            )
        elif result.branch == "smart":
            # Primary handling is in process_step (nested recovery); hold index like retry.
            self.manager.log_info(
                f"Verify: branch=smart; holding step for nested recovery. {result.reason}"
            )
        elif result.accomplished:
            if result.branch != "advance":
                self.manager.log_info(
                    f"Verify: accomplished with branch={result.branch}; advancing to next line. {result.reason}"
                )
            self._script_step_index = idx + 1
        elif result.branch == "retry":
            pass
        elif result.branch == "skip":
            self._script_step_index = idx + 1
        elif result.branch == "goto":
            assert result.target_step is not None
            target_0 = result.target_step - 1
            self._script_step_index = max(0, min(target_0, n - 1))
        elif result.branch == "advance":
            self.manager.log_info(
                f"Verify: not accomplished but branch=advance; holding step. {result.reason}"
            )

        run_complete = self._script_step_index >= n
        self.manager.log_info(
            f"Verify branch applied: index={self._script_step_index}/{n} "
            f"accomplished={result.accomplished} branch={result.branch} run_complete={run_complete} "
            f"reason={result.reason}"
        )
        return run_complete

    @staticmethod
    def _is_recorded_anchor_miss(result: ExecutionResult) -> bool:
        message = result.message or ""
        return RECORDED_ANCHOR_MISS_MESSAGE in message

    def _consume_hold_step_retry(self) -> ScriptStepVerifyResult | None:
        """Take a move_mouse replay failure that should pause and retry this step."""
        reason = getattr(self, "_hold_step_retry_reason", None)
        self._hold_step_retry_reason = None
        if not isinstance(reason, str) or not reason.strip():
            return None
        return ScriptStepVerifyResult(
            accomplished=False,
            branch="retry",
            target_step=None,
            clearly_unmet=True,
            reason=reason.strip(),
        )

    async def _try_replay_cached_tools(
        self,
        goal: str,
        cached_calls: list[dict[str, Any]],
    ) -> Literal["ok", "fallback", "anchor_miss", "hold_retry"]:
        """Execute cached tool calls in order.

        ``ok`` means every tool succeeded. ``hold_retry`` means ``move_mouse``
        failed; the caller holds the step so the overlay counts down and the
        same step runs again, without asking the model to pick another window.
        ``anchor_miss`` means some other recorded target had no similarity
        match; the caller must fail the step instead of asking the model to
        re-pick visually. ``fallback`` is any other replay failure, which still
        goes to the decide loop.
        """
        all_image_paths = await self._eye.capture_separated_images()
        before_screenshot = self._primary_decision_screenshot(all_image_paths)
        # Matching recording: this step's before is the previous step's after.
        await self._seal_pending_after_with_before(before_screenshot)
        messages: list[dict[str, Any]] = [
            stamp_message(
                {
                    "role": ROLE_USER,
                    "content": f"Cache replay for task: {goal}",
                    "images": all_image_paths,
                    "cache_replay": True,
                }
            ),
            stamp_message(
                {
                    "role": ROLE_ASSISTANT,
                    "tool_calls": [
                        {
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            }
                        }
                        for call in cached_calls
                    ],
                    "cache_replay": True,
                }
            ),
        ]

        repair_binding = nullcontext()
        fallback_binding = nullcontext()
        script_path_raw = (os.environ.get(SCRIPT_PATH_ENV) or "").strip()
        if script_path_raw:
            try:
                from src.common.script_helper import recording_run_dir
                from src.recorder.uia_replay_repair import (
                    bind_uia_replay_repairs,
                    bind_yolo_ocr_fallbacks,
                    replay_repair_slots,
                    yolo_ocr_fallback_slots,
                )

                recording_dir = recording_run_dir(Path(script_path_raw))
                event_ids = self._script_event_indices
                step_index = self._script_step_index
                event_index = (
                    event_ids[step_index]
                    if 0 <= step_index < len(event_ids)
                    and isinstance(event_ids[step_index], int)
                    else None
                )
                if recording_dir is not None and event_index is not None:
                    slots = replay_repair_slots(
                        recording_dir, int(event_index), cached_calls
                    )
                    if slots:
                        repair_binding = bind_uia_replay_repairs(slots)
                    fallback_slots = yolo_ocr_fallback_slots(
                        recording_dir, int(event_index), cached_calls
                    )
                    if fallback_slots:
                        fallback_binding = bind_yolo_ocr_fallbacks(fallback_slots)
            except Exception as exc:
                self.manager.log_info(f"UIA replay repair setup skipped: {exc}")

        try:
            with repair_binding, fallback_binding:
                with require_similarity_match():
                    for call in cached_calls:
                        arguments = self._normalize_tool_arguments(call.get("arguments"))
                        try:
                            normalized_name = await self._normalize_tool_name(
                                call["name"], arguments
                            )
                        except Exception as e:
                            self.manager.log_error(
                                f"Cache replay: error normalizing tool name: {e}"
                            )
                            self._save_step_messages(messages)
                            return "fallback"
                        arguments = self._enrich_tool_arguments(
                            normalized_name, arguments, goal
                        )
                        result = await self._hand.execute_tool_command(
                            ToolCommand(
                                action=normalized_name,
                                args=arguments,
                                screenshot_before_path=before_screenshot,
                            )
                        )
                        self._mark_pending_after_capture()
                        self._remember_replay_press_point(result)
                        self._remember_close_windows_success(result)
                        messages.append(
                            stamp_message(
                                {
                                    "role": ROLE_TOOL,
                                    "content": json.dumps(
                                        self.sanitize_execution_result(result),
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                        )
                        if not result.ok:
                            self._append_failed_tool_call(
                                result.action,
                                self._step_transcript_counter,
                                self._script_step_index,
                            )
                            self._save_step_messages(messages)
                            if result.action == "move_mouse":
                                message = (result.message or "").strip()
                                self._hold_step_retry_reason = (
                                    message or "move_mouse failed"
                                )
                                return "hold_retry"
                            if self._is_recorded_anchor_miss(result):
                                return "anchor_miss"
                            return "fallback"

            self._save_step_messages(messages)
            return "ok"
        finally:
            await self._flush_eye_pending_saves()

    async def loop(self) -> bool:
        """Run the capture → LLM (with tools) → execute tools loop until the model returns no tool calls or cap is hit."""
        if self._hand is None:
            raise RuntimeError("BrainModule requires hand=HandModule(...) for the decide/execute loop")
        if self._eye is None:
            raise RuntimeError("BrainModule requires eye=EyeModule(...) to capture screenshots for the decide loop")

        self._hold_step_retry_reason = None
        goal = self._current_goal()
        first_prompt = get_prompt("brain_decide_action").format(task=goal)
        second_prompt = get_prompt("brain_decide_action_2").format(task=goal)
        tool_functions = get_mode_tool_functions()
        tool_names = {tool.__name__ for tool in tool_functions}
        if "move_mouse_visual" in tool_names and "move_mouse" not in tool_names:
            mode_tool_policy = (
                "\n\nMode tool policy: move_mouse is unavailable. Use move_mouse_visual "
                "for movement to a named on-screen target."
            )
        elif "move_mouse" in tool_names and "move_mouse_visual" not in tool_names:
            mode_tool_policy = (
                "\n\nMode tool policy: move_mouse_visual is unavailable. Use move_mouse "
                "for movement to a named on-screen target."
            )
        else:
            mode_tool_policy = ""
        first_prompt += mode_tool_policy
        second_prompt += mode_tool_policy

        # Recording runs always prefer compiled tool calls from the recording folder.
        # The hub "use tool cache" flag only gates the global runs/instruction cache.
        # Do not call ensure_recording_tool_cache here: main.py ensures once at run
        # start, and hub edits rebuild the cache on save. Re-ensuring every step
        # re-parses huge analysis JSON (window_snapshot_debug) for no benefit.
        cached_calls = None
        script_path_raw = (os.environ.get(SCRIPT_PATH_ENV) or "").strip()
        if script_path_raw:
            from src.common.script_helper import recording_run_dir
            from src.recorder.compile_tool_calls import recording_tool_cache_path

            recording_dir = recording_run_dir(Path(script_path_raw))
            if recording_dir is not None:
                cached_calls = lookup_tool_calls(
                    goal, path=recording_tool_cache_path(recording_dir)
                )
        if not cached_calls and use_tool_cache_enabled():
            cached_calls = lookup_tool_calls(goal)
        if cached_calls:
            self.manager.log_info(
                f"Instruction tool cache hit ({len(cached_calls)} tool call(s)); replaying"
            )
            replay = await self._try_replay_cached_tools(goal, cached_calls)
            if replay == "ok":
                return True
            if replay == "hold_retry":
                self.manager.log_info(
                    "Cache replay move_mouse failed; holding step for overlay retry"
                )
                return False
            if replay == "anchor_miss":
                self.manager.log_info(
                    "Cache replay found no similarity match for the recorded target; "
                    "failing the step"
                )
                return False
            self.manager.log_info("Cache replay failed; falling back to LLM decide loop")

        messages: list[dict[str, Any]] = []
        step_succeeded = False
        llm_path_used = False
        # Actions that returned ok=false and have not succeeded on a later retry.
        unresolved_tool_failures: set[str] = set()

        for _ in range(_MAX_INNER_DECIDE_STEPS):
            try:
                llm_path_used = True
                all_image_paths = await self._eye.capture_separated_images()
                before_screenshot = self._primary_decision_screenshot(all_image_paths)
                # Matching recording: this decide-frame seals the previous last-tool after.
                await self._seal_pending_after_with_before(before_screenshot)
                user_content = first_prompt if not messages else second_prompt
                messages.append(
                    stamp_message(
                        {
                            "role": ROLE_USER,
                            "content": user_content,
                            "images": all_image_paths,
                        }
                    )
                )
                response_message = await self.ollama.chat_messages(
                    self.settings.brain_lm,
                    messages=messages,
                    tools=tool_functions,
                )
                if not response_message:
                    self.manager.log_error("LLM returned empty response")
                    break
                response_message_dict = stamp_message(self.sanitize_message(response_message))
                messages.append(response_message_dict)

                raw_tool_calls = list(response_message.tool_calls or [])
                real_tool_calls = [
                    call
                    for call in raw_tool_calls
                    if not self._is_pseudo_end_tool_name(
                        getattr(getattr(call, "function", None), "name", None)
                    )
                ]
                pseudo_end_calls = [
                    call
                    for call in raw_tool_calls
                    if self._is_pseudo_end_tool_name(
                        getattr(getattr(call, "function", None), "name", None)
                    )
                ]

                if not raw_tool_calls:
                    outcome = self._parse_step_outcome(
                        getattr(response_message, "content", None)
                    )
                    step_succeeded = self._resolve_step_success_from_outcome(
                        outcome,
                        unresolved_tool_failures,
                        missing_outcome_message=(
                            "Decide loop ended without tools but step outcome JSON was "
                            "missing/invalid"
                        ),
                    )
                    break

                if not real_tool_calls and pseudo_end_calls:
                    end_call = pseudo_end_calls[0]
                    end_name = getattr(end_call.function, "name", "finish")
                    end_args = dict(getattr(end_call.function, "arguments", {}) or {})
                    outcome = self._parse_step_outcome_from_arguments(end_args)
                    if outcome is None:
                        outcome = self._parse_step_outcome(
                            getattr(response_message, "content", None)
                        )
                    self.manager.log_info(
                        f"Model emitted pseudo end tool '{end_name}'; treating arguments "
                        "as step outcome JSON (not executing)"
                    )
                    step_succeeded = self._resolve_step_success_from_outcome(
                        outcome,
                        unresolved_tool_failures,
                        missing_outcome_message=(
                            f"Pseudo end tool '{end_name}' lacked valid status/reason"
                        ),
                    )
                    break

                if pseudo_end_calls:
                    ignored = ", ".join(
                        sorted(
                            {
                                str(getattr(call.function, "name", "finish"))
                                for call in pseudo_end_calls
                            }
                        )
                    )
                    self.manager.log_info(
                        f"Ignoring pseudo end tool(s) mixed with real tools: {ignored}"
                    )

                abort_step = False
                for tool_call in real_tool_calls:
                    arguments = self._normalize_tool_arguments(
                        tool_call.function.arguments
                    )
                    try:
                        normalized_name = await self._normalize_tool_name(
                            tool_call.function.name, arguments
                        )
                    except Exception as e:
                        self.manager.log_error(f"Error normalizing tool name: {e}")
                        step_succeeded = False
                        abort_step = True
                        break
                    arguments = self._enrich_tool_arguments(
                        normalized_name, arguments, goal
                    )
                    result = await self._hand.execute_tool_command(
                        ToolCommand(
                            action=normalized_name,
                            args=arguments,
                            screenshot_before_path=before_screenshot,
                        )
                    )
                    self._mark_pending_after_capture()
                    self._remember_replay_press_point(result)
                    self._remember_close_windows_success(result)
                    messages.append(
                        stamp_message(
                            {
                                "role": ROLE_TOOL,
                                "content": json.dumps(
                                    self.sanitize_execution_result(result),
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    )
                    if not result.ok:
                        self._append_failed_tool_call(
                            result.action,
                            self._step_transcript_counter,
                            self._script_step_index,
                        )
                        unresolved_tool_failures.add(result.action)
                        # Do not run dependent follow-ups (e.g. click after failed move_mouse).
                        # Continue the decide loop so the model can retry or end as failed.
                        break
                    unresolved_tool_failures.discard(result.action)
                if abort_step:
                    break
                continue
            finally:
                self._save_step_messages(messages)
        else:
            self.manager.log_info(
                f"Brain inner loop reached max steps ({_MAX_INNER_DECIDE_STEPS}) without model completion"
            )
            step_succeeded = False

        if step_succeeded and llm_path_used:
            tool_calls = extract_tool_calls_from_messages(messages)
            if tool_calls:
                upsert_tool_calls(goal, tool_calls, source_run_id=self.run_id)
                self.manager.log_info(
                    f"Instruction tool cache updated ({len(tool_calls)} tool call(s))"
                )
                script_path_raw = (os.environ.get(SCRIPT_PATH_ENV) or "").strip()
                if script_path_raw:
                    from src.common.script_helper import recording_run_dir
                    from src.recorder.compile_tool_calls import (
                        mirror_tool_calls_into_recording,
                    )

                    recording_dir = recording_run_dir(Path(script_path_raw))
                    if recording_dir is not None:
                        mirror_tool_calls_into_recording(
                            recording_dir,
                            goal,
                            tool_calls,
                            source_run_id=self.run_id,
                        )
                        self.manager.log_info(
                            "Recording instruction tool cache updated "
                            f"({len(tool_calls)} tool call(s))"
                        )

        await self._flush_eye_pending_saves()
        return step_succeeded

    @staticmethod
    def _auto_advance_verify_result(
        *,
        window_verify_passed: bool = False,
    ) -> ScriptStepVerifyResult:
        if window_verify_passed:
            reason = (
                "Actor completed the step with all tools ok; window verify passed."
            )
        else:
            reason = (
                "Actor completed the step with all tools ok; "
                "no window verify required."
            )
        return ScriptStepVerifyResult(
            accomplished=True,
            branch="advance",
            target_step=None,
            reason=reason,
        )

    def _remember_replay_press_point(self, result: ExecutionResult) -> None:
        """Keep the root window read immediately before this step's click."""
        if not result.ok or result.action not in CLICK_KINDS:
            return
        point = result.args.get("point_window") if isinstance(result.args, dict) else None
        self._replay_press_point_window = point if isinstance(point, dict) and point else None

    def _remember_close_windows_success(self, result: ExecutionResult) -> None:
        """Remember a successful close so verify can re-close a window that stays up."""
        if result.ok and result.action == "close_windows":
            self._step_close_windows_ok = True

    async def _snapshot_top_level_windows(self) -> list[WindowInfo] | None:
        """Enumerate top-level windows off the event loop. None when the snapshot fails."""
        try:
            windows = await asyncio.to_thread(snapshot_top_level_windows)
        except Exception as exc:
            self.manager.log_info(f"window snapshot failed: {exc}")
            return None
        return list(windows)

    def _window_verify_poll_settle(
        self,
        settle_after: float | None,
        recorded: dict[str, Any],
    ) -> float | None:
        """Settle budget base for window-verify continuous polling.

        Prefer the recording ``settle_after``. When the step expects a new window
        (``appeared``), floor the budget at ``_APPEARED_POLL_MIN_SECONDS`` so a
        slow launch keeps a long enough deadline (2.0x of this base is at least
        10s). When it expects a window to leave (``disappeared``), floor at
        ``_DISAPPEARED_POLL_MIN_SECONDS`` so async closes are not judged on a
        single immediate after-snapshot. When it expects signal end-state
        (``foreground``, ``click_window``, clipboard, caret, focused, scroll),
        floor at ``_SIGNAL_POLL_MIN_SECONDS`` for the same reason. Floors
        combine with ``max``. Polling starts immediately after actions and
        early-exits on match.
        """
        appeared = recorded.get("appeared")
        disappeared = recorded.get("disappeared")
        has_appeared = isinstance(appeared, list) and bool(appeared)
        has_disappeared = isinstance(disappeared, list) and bool(disappeared)
        has_signals = predicate_has_signal_assertions(recorded)
        floor = 0.0
        if has_appeared:
            floor = max(floor, float(_APPEARED_POLL_MIN_SECONDS))
        if has_disappeared:
            floor = max(floor, float(_DISAPPEARED_POLL_MIN_SECONDS))
        if has_signals:
            floor = max(floor, float(_SIGNAL_POLL_MIN_SECONDS))
        if settle_after is not None and float(settle_after) > 0:
            base = float(settle_after)
            if floor > 0:
                return max(base, floor)
            return base
        if floor > 0:
            return floor
        return None

    async def _window_verify_miss(
        self,
        windows_before: list[WindowInfo] | None,
        recorded: dict[str, Any],
        *,
        log_result: bool = True,
    ) -> ScriptStepVerifyResult | None:
        """Return a clearly-unmet retry when the live delta misses the recorded one."""
        miss, _windows_after = await self._window_verify_observation(
            windows_before,
            recorded,
            log_result=log_result,
        )
        return miss

    async def _window_verify_observation(
        self,
        windows_before: list[WindowInfo] | None,
        recorded: dict[str, Any],
        *,
        log_result: bool = True,
    ) -> tuple[ScriptStepVerifyResult | None, list[WindowInfo] | None]:
        """Verify once. ``None`` miss means the recorded end state holds.

        The after-window list is returned with the miss so a disappeared close
        retry can target the same sample that failed the check.
        """
        if windows_before is None:
            return (
                ScriptStepVerifyResult(
                    accomplished=False,
                    branch="retry",
                    target_step=None,
                    clearly_unmet=True,
                    reason="Window snapshot before the step failed.",
                ),
                None,
            )
        windows_after = await self._snapshot_top_level_windows()
        if windows_after is None:
            return (
                ScriptStepVerifyResult(
                    accomplished=False,
                    branch="retry",
                    target_step=None,
                    clearly_unmet=True,
                    reason="Window snapshot after the step failed.",
                ),
                None,
            )
        live = build_window_verify_predicate(windows_before, windows_after)
        live_after: dict[str, Any] | None = None
        if predicate_has_signal_assertions(recorded):
            try:
                press_point = self._replay_press_point_window
                live_after = await asyncio.to_thread(
                    capture_replay_after_signals,
                    recorded,
                    press_point,
                )
            except Exception as exc:
                self.manager.log_info(f"signal snapshot failed: {exc}")
                return (
                    ScriptStepVerifyResult(
                        accomplished=False,
                        branch="retry",
                        target_step=None,
                        clearly_unmet=True,
                        reason="Signal snapshot after the step failed.",
                    ),
                    windows_after,
                )
        ok, reason = window_verify_satisfied(
            recorded,
            live,
            live_after=live_after,
            live_windows_after=windows_after,
        )
        if ok:
            if log_result:
                self.manager.log_info("Window verify matched recorded end state")
            return None, windows_after
        if log_result:
            self.manager.log_info(f"Window verify failed: {reason}")
        return (
            ScriptStepVerifyResult(
                accomplished=False,
                branch="retry",
                target_step=None,
                clearly_unmet=True,
                reason=reason,
            ),
            windows_after,
        )

    def _record_disappeared_close_retry(self, rows: list[dict[str, Any]]) -> None:
        """Write one re-close attempt into the step transcript, not the tool cache."""
        if not rows:
            return
        try:
            self._append_step_messages(
                [
                    {
                        "role": ROLE_ASSISTANT,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "close_windows",
                                    "arguments": {
                                        "hwnd": row.get("hwnd"),
                                        "matched_title": row.get("matched_title"),
                                        "disappeared_close_retry": True,
                                    },
                                }
                            }
                            for row in rows
                        ],
                        "disappeared_close_retry": True,
                    },
                    {
                        "role": ROLE_TOOL,
                        "content": json.dumps(
                            {
                                "ok": all(row.get("status") == "closed" for row in rows),
                                "action": "close_windows",
                                "disappeared_close_retry": True,
                                "targets": rows,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                self._step_transcript_counter,
                self._script_step_index,
            )
        except Exception as exc:
            self.manager.log_info(
                f"disappeared close retry transcript append failed: {exc}"
            )

    async def _retry_close_disappeared_windows(
        self,
        windows: list[WindowInfo],
    ) -> None:
        """Close hwnds that still match the unmet disappeared entries."""
        rows: list[dict[str, Any]] = []
        for win in windows:
            try:
                row = await asyncio.to_thread(close_hwnd, int(win.hwnd))
            except Exception as exc:
                self.manager.log_info(
                    f"disappeared close retry failed hwnd={win.hwnd} "
                    f"title={win.title!r}: {exc}"
                )
                row = {
                    "hwnd": int(win.hwnd),
                    "matched_title": win.title,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                self.manager.log_info(
                    f"disappeared close retry closed hwnd={win.hwnd} "
                    f"title={win.title!r}"
                )
            rows.append(row)
        self._record_disappeared_close_retry(rows)

    async def _poll_window_verify(
        self,
        windows_before: list[WindowInfo] | None,
        recorded: dict[str, Any],
        *,
        settle_after: float,
        script_step_index: int,
    ) -> ScriptStepVerifyResult | None:
        """Poll window verify continuously until match or deadline.

        Deadline is ``2.0x`` of ``settle_after`` when settle is at least
        ``_SETTLE_POLL_MULTI_MIN_SECONDS``, otherwise ``1.0x``. Checks start
        immediately after step actions and repeat every
        ``_WINDOW_VERIFY_POLL_INTERVAL_SECONDS``. Returns ``None`` on the first
        pass; failure is only returned after the deadline.

        When this step's ``close_windows`` succeeded and the miss is a
        disappeared window, re-close the windows that still match that entry.
        That retry uses the same deadline, extended to
        ``_DISAPPEARED_CLOSE_RETRY_MIN_SECONDS`` when the settle deadline is
        shorter. Other misses keep the settle deadline and do not close again.
        """
        base_deadline = (
            settle_after * 2.0
            if settle_after >= _SETTLE_POLL_MULTI_MIN_SECONDS
            else settle_after
        )
        close_retry = bool(self._step_close_windows_ok) and _predicate_has_disappeared(
            recorded
        )
        extended_deadline = max(
            base_deadline, float(_DISAPPEARED_CLOSE_RETRY_MIN_SECONDS)
        )
        interval = float(_WINDOW_VERIFY_POLL_INTERVAL_SECONDS)
        close_interval = float(_DISAPPEARED_CLOSE_RETRY_INTERVAL_SECONDS)
        if close_retry and extended_deadline > base_deadline:
            self.manager.log_info(
                f"Script step {script_step_index + 1} window verify polling "
                f"every {interval:.1f}s up to {base_deadline:.3f}s; "
                f"disappeared close retry extends that to {extended_deadline:.3f}s"
            )
        elif close_retry:
            self.manager.log_info(
                f"Script step {script_step_index + 1} window verify polling "
                f"every {interval:.1f}s up to {base_deadline:.3f}s "
                "with disappeared close retry"
            )
        else:
            self.manager.log_info(
                f"Script step {script_step_index + 1} window verify polling "
                f"every {interval:.1f}s up to {base_deadline:.3f}s"
            )
        elapsed = 0.0
        last_close_elapsed: float | None = None
        last_miss: ScriptStepVerifyResult | None = None
        while True:
            miss, windows_after = await self._window_verify_observation(
                windows_before,
                recorded,
                log_result=False,
            )
            if miss is None:
                self.manager.log_info(
                    f"Script step {script_step_index + 1} window verify "
                    f"matched after {elapsed:.3f}s"
                )
                return None
            last_miss = miss
            disappeared_miss = close_retry and _reason_is_disappeared_miss(miss.reason)
            deadline = extended_deadline if disappeared_miss else base_deadline
            if elapsed >= deadline:
                self.manager.log_info(
                    f"Script step {script_step_index + 1} window verify "
                    f"still unmet after {elapsed:.3f}s "
                    f"(deadline {deadline:.3f}s); {miss.reason}"
                )
                return last_miss
            if (
                disappeared_miss
                and windows_after is not None
                and _close_retry_due(elapsed, last_close_elapsed, close_interval)
            ):
                targets = disappeared_windows_still_present(recorded, windows_after)
                if targets:
                    await self._retry_close_disappeared_windows(targets)
                    last_close_elapsed = elapsed
            delay = min(interval, max(0.0, deadline - elapsed))
            if delay > 0:
                await asyncio.sleep(delay)
                elapsed += delay
            else:
                return last_miss

    def _verify_result_metadata(
        self,
        verify_result: ScriptStepVerifyResult,
        *,
        baseline_after_path: str | None = None,
    ) -> dict[str, Any]:
        return {
            "accomplished": verify_result.accomplished,
            "branch": verify_result.branch,
            "target_step": verify_result.target_step,
            "clearly_unmet": verify_result.clearly_unmet,
            "reason": verify_result.reason,
            "baseline_after_path": baseline_after_path,
        }

    async def process_step(self) -> BrainStepResult:
        """Run one script step: tool loop, then window/signal verification and branching.

        Happy path: no recorded window/signal assert and actor success (all tools ok)
        auto-advances without a screenshot or verifier LLM. Any recording
        ``settle_after`` sleeps on this step when window-verify does not own settle.
        A recorded window diff polls continuously after actions (early-exit on match;
        fail only at 2.0x of settle, or 1.0x when settle is under 2s; ``appeared`` /
        ``disappeared`` / signal asserts floor settle so small recording gaps still
        poll). When ``close_windows`` succeeded and a disappeared window is still
        present, that wait re-closes the matching window until the check passes
        or ``max(deadline, 10s)``. A miss retries with ``clearly_unmet``.

        Report after-frames match recording: the next step's before shot seals this
        step's last-tool after. The final step (or a failed stop) captures a
        dedicated after instead.

        Screenshot/LLM vision verification is not used. Actor failure without a
        window miss fails the step with no image-based recovery. Single-script and
        queue runs can skip every check (``CUA_SKIP_VERIFICATION``): a successful
        actor advances, and a failed actor stops the step. Branch ``smart`` (if
        produced) still runs bounded nested Plan→Act→Verify recovery. Verify
        ``branch=abort`` stops the scripted run (``step_finished=False``).
        Sets ``run_complete`` when the script is exhausted.
        """
        # await self._validate_tool_functions_match_mcp()

        transcript_counter = self._step_transcript_counter
        script_step_index = self._script_step_index
        self.manager.set_step_log_context(transcript_counter, script_step_index)
        event_ids = self._script_event_indices
        event_index = (
            event_ids[script_step_index]
            if 0 <= script_step_index < len(event_ids)
            and isinstance(event_ids[script_step_index], int)
            else None
        )
        instruction = (
            self.script_lines[script_step_index]
            if 0 <= script_step_index < len(self.script_lines)
            else None
        )
        notify_active_step(
            script_step_index,
            instruction=instruction,
            event_index=event_index,
        )
        started_iso = datetime.now(timezone.utc).isoformat()
        started_at = perf_counter()
        # Attribute rare dirty reloads to this step's duration (not the inter-step gap).
        self.reload_pending_recording_steps()
        publish_script_steps(list(self.script_lines), list(self._script_event_indices))
        if self._script_step_index >= len(self.script_lines):
            return BrainStepResult(
                reason="All script steps complete",
                step_finished=True,
                run_complete=True,
            )

        # When True, leave last-tool after pending for the next step's before shot.
        # Otherwise finally captures a dedicated after (last step / failure).
        defer_after_capture = False
        try:
            self._replay_press_point_window = None
            self._step_close_windows_ok = False
            skip_verification = skip_verification_enabled()
            window_verify = {} if skip_verification else self._current_window_verify()
            windows_before: list[WindowInfo] | None = None
            if window_verify:
                windows_before = await self._snapshot_top_level_windows()

            step_succeeded = await self.loop()
            settle_after = None
            if step_succeeded:
                settle_after = self._current_settle_after_seconds()
            # Window verify polls continuously when present; otherwise sleep the
            # recording gap on this step before finished_at is stamped.
            run_window_verify = step_succeeded and bool(window_verify)
            window_poll_settle = (
                self._window_verify_poll_settle(settle_after, window_verify)
                if run_window_verify
                else None
            )
            window_miss: ScriptStepVerifyResult | None = None
            if run_window_verify:
                if window_poll_settle is not None:
                    window_miss = await self._poll_window_verify(
                        windows_before,
                        window_verify,
                        settle_after=window_poll_settle,
                        script_step_index=script_step_index,
                    )
                else:
                    window_miss = await self._window_verify_miss(
                        windows_before, window_verify
                    )
            elif (
                step_succeeded
                and settle_after is not None
                and settle_after > 0
            ):
                await self._sleep_settle_after(settle_after)
            if window_miss is not None:
                verify_result = window_miss
            elif step_succeeded:
                if skip_verification:
                    self.manager.log_info(
                        f"Script step {script_step_index + 1} skipping verification"
                    )
                    verify_result = ScriptStepVerifyResult(
                        accomplished=True,
                        branch="advance",
                        target_step=None,
                        reason="Verification skipped for this queue run.",
                    )
                else:
                    if run_window_verify:
                        self.manager.log_info(
                            f"Script step {script_step_index + 1} auto-advancing "
                            "(actor tools succeeded; window verify passed)"
                        )
                    else:
                        self.manager.log_info(
                            f"Script step {script_step_index + 1} auto-advancing "
                            "(actor tools succeeded; no window verify required)"
                        )
                    verify_result = self._auto_advance_verify_result(
                        window_verify_passed=run_window_verify
                    )
            else:
                hold_retry = self._consume_hold_step_retry()
                if hold_retry is not None:
                    self.manager.log_info(
                        f"Script step {script_step_index + 1} move_mouse failed; "
                        "holding step for overlay retry"
                    )
                    verify_result = hold_retry
                else:
                    if skip_verification:
                        self.manager.log_info(
                            f"Script step {script_step_index + 1} skipping verification; "
                            "actor failed"
                        )
                    else:
                        self.manager.log_info(
                            f"Script step {script_step_index + 1} actor failed; "
                            "no screenshot verification recovery"
                        )
                    verify_result = None
            finished_iso = datetime.now(timezone.utc).isoformat()
            duration_seconds = round(perf_counter() - started_at, 3)
            self._step_transcript_counter += 1
            step_baseline_after = self._current_baseline_after_path()
            if verify_result is None:
                self._update_step_metadata(
                    transcript_counter,
                    script_step_index,
                    {
                        "started_at_utc": started_iso,
                        "finished_at_utc": finished_iso,
                        "duration_seconds": duration_seconds,
                        "status": "failed",
                        "step_index": script_step_index,
                        "goal": self._current_goal(),
                        "settle_after_seconds": settle_after,
                        "verify": None,
                    },
                )
                return BrainStepResult(
                    reason=f"Script step {script_step_index + 1} failed",
                    step_finished=False,
                    step_index=script_step_index,
                )

            step_goal = self._current_goal()
            step_expected_outcome = self._current_expected_outcome() or None
            attempt_number = self._script_step_attempt_number(script_step_index)
            max_attempts = int(self.settings.script_max_step_attempts)
            verify_meta = self._verify_result_metadata(
                verify_result,
                baseline_after_path=step_baseline_after,
            )
            if verify_result.branch == "smart":
                prior_smart = self._script_step_smart_recovery_count(script_step_index)
                max_smart = max(0, int(self.settings.script_smart_recovery_max_per_step))
                if max_smart > 0 and prior_smart >= max_smart:
                    reason = (
                        f"Script step {script_step_index + 1} smart recovery cap reached "
                        f"({prior_smart}/{max_smart}): {verify_result.reason}"
                    )
                    self.manager.log_info(reason)
                    self._update_step_metadata(
                        transcript_counter,
                        script_step_index,
                        {
                            "started_at_utc": started_iso,
                            "finished_at_utc": finished_iso,
                            "duration_seconds": duration_seconds,
                            "status": "failed",
                            "step_index": script_step_index,
                            "goal": step_goal,
                            "expected_outcome": step_expected_outcome,
                            "attempt_number": attempt_number,
                            "max_attempts": max_attempts,
                            "verify": verify_meta,
                            "smart_recovery": {
                                "ok": False,
                                "reason": reason,
                                "prior_count": prior_smart,
                                "max_per_step": max_smart,
                                "events": [],
                            },
                        },
                    )
                    return BrainStepResult(
                        reason=reason,
                        step_finished=False,
                        step_index=script_step_index,
                    )
                from src.runtime.script_smart_recovery import run_script_smart_recovery

                recovery = await run_script_smart_recovery(
                    brain=self,
                    script_goal=step_goal,
                    verify_reason=verify_result.reason,
                    max_cycles=int(self.settings.script_smart_recovery_max_cycles),
                )
                finished_iso = datetime.now(timezone.utc).isoformat()
                duration_seconds = round(perf_counter() - started_at, 3)
                smart_meta = {
                    "ok": recovery.ok,
                    "reason": recovery.reason,
                    "prior_count": prior_smart,
                    "max_per_step": max_smart,
                    "events": recovery.events,
                }
                if recovery.ok:
                    self.manager.log_info(
                        f"Script step {script_step_index + 1} smart recovery ok; "
                        "holding step for retry"
                    )
                    self._update_step_metadata(
                        transcript_counter,
                        script_step_index,
                        {
                            "started_at_utc": started_iso,
                            "finished_at_utc": finished_iso,
                            "duration_seconds": duration_seconds,
                            "status": "smart_recovered",
                            "step_index": script_step_index,
                            "goal": step_goal,
                            "expected_outcome": step_expected_outcome,
                            "attempt_number": attempt_number,
                            "max_attempts": max_attempts,
                            "verify": verify_meta,
                            "smart_recovery": smart_meta,
                        },
                    )
                    return BrainStepResult(
                        reason=f"Smart recovery: {recovery.reason}",
                        step_finished=True,
                        run_complete=False,
                        step_index=script_step_index,
                    )
                reason = (
                    f"Script step {script_step_index + 1} smart recovery failed: "
                    f"{recovery.reason}"
                )
                self.manager.log_info(reason)
                self._update_step_metadata(
                    transcript_counter,
                    script_step_index,
                    {
                        "started_at_utc": started_iso,
                        "finished_at_utc": finished_iso,
                        "duration_seconds": duration_seconds,
                        "status": "failed",
                        "step_index": script_step_index,
                        "goal": step_goal,
                        "expected_outcome": step_expected_outcome,
                        "attempt_number": attempt_number,
                        "max_attempts": max_attempts,
                        "verify": verify_meta,
                        "smart_recovery": smart_meta,
                    },
                )
                return BrainStepResult(
                    reason=reason,
                    step_finished=False,
                    step_index=script_step_index,
                )
            if verify_result.branch == "abort":
                reason = (
                    f"Script step {script_step_index + 1} aborted: {verify_result.reason}"
                )
                self.manager.log_info(
                    f"Script step {script_step_index + 1} verify branch=abort; stopping run"
                )
                self._update_step_metadata(
                    transcript_counter,
                    script_step_index,
                    {
                        "started_at_utc": started_iso,
                        "finished_at_utc": finished_iso,
                        "duration_seconds": duration_seconds,
                        "status": "failed",
                        "step_index": script_step_index,
                        "goal": step_goal,
                        "expected_outcome": step_expected_outcome,
                        "attempt_number": attempt_number,
                        "max_attempts": max_attempts,
                        "verify": verify_meta,
                    },
                )
                return BrainStepResult(
                    reason=reason,
                    step_finished=False,
                    step_index=script_step_index,
                )
            if (
                max_attempts > 0
                and not verify_result.accomplished
                and verify_result.branch == "retry"
                and attempt_number >= max_attempts
            ):
                reason = (
                    f"Script step {script_step_index + 1} failed after "
                    f"{attempt_number}/{max_attempts} attempt(s): {verify_result.reason}"
                )
                self.manager.log_info(
                    f"Script step retry limit reached ({attempt_number}/{max_attempts}); stopping run"
                )
                self._update_step_metadata(
                    transcript_counter,
                    script_step_index,
                    {
                        "started_at_utc": started_iso,
                        "finished_at_utc": finished_iso,
                        "duration_seconds": duration_seconds,
                        "status": "failed",
                        "step_index": script_step_index,
                        "goal": step_goal,
                        "expected_outcome": step_expected_outcome,
                        "attempt_number": attempt_number,
                        "max_attempts": max_attempts,
                        "verify": verify_meta,
                    },
                )
                return BrainStepResult(
                    reason=reason,
                    step_finished=False,
                    step_index=script_step_index,
                )
            run_complete = self._apply_verify_branch(verify_result)
            if run_complete:
                # Last step: finally captures a dedicated after.
                defer_after_capture = False
            elif verify_result.accomplished:
                # Next step's before seals this step's after (no extra capture).
                defer_after_capture = True
            else:
                defer_after_capture = False
            if verify_result.accomplished:
                status = "completed"
            elif not step_succeeded:
                status = "failed"
            else:
                status = "verify_failed"
            self._update_step_metadata(
                transcript_counter,
                script_step_index,
                {
                    "started_at_utc": started_iso,
                    "finished_at_utc": finished_iso,
                    "duration_seconds": duration_seconds,
                    "status": status,
                    "step_index": script_step_index,
                    "goal": step_goal,
                    "expected_outcome": step_expected_outcome,
                    "settle_after_seconds": settle_after,
                    "verify": verify_meta,
                },
            )
            return BrainStepResult(
                reason=f"Verify: {verify_result.reason}",
                step_finished=True,
                run_complete=run_complete,
                step_index=script_step_index,
            )
        except asyncio.CancelledError:
            defer_after_capture = False
            self._save_interrupted_step_record(
                transcript_counter,
                script_step_index,
                started_iso,
                started_at,
                f"Script step {script_step_index + 1} cancelled",
            )
            raise
        except Exception as exc:
            defer_after_capture = False
            reason = f"Script step {script_step_index + 1} failed: {exc}"
            self._save_interrupted_step_record(
                transcript_counter,
                script_step_index,
                started_iso,
                started_at,
                reason,
            )
            return BrainStepResult(
                reason=reason,
                step_finished=False,
                step_index=script_step_index,
            )
        finally:
            try:
                await self._flush_eye_pending_saves()
            except Exception as exc:
                self.manager.log_error(f"Eye pending PNG flush failed: {exc}")
            if not defer_after_capture:
                await self._flush_pending_after_capture()
            self.manager.clear_step_log_context()
