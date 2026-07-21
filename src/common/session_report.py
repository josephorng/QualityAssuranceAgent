from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.io_utils import write_json
from src.common.runtime_context import SCRIPT_PATH_ENV

_REPORT_VERSION = 1
_RUNTIME_COMMANDS_NAME = "runtime_commands.txt"
_QUEUE_SCRIPT_LOG_MARKER = "Queue starting coordinator for "
_RUN_FOLDER_TS_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_(\d{8})_(\d{6})_\d+$")
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"
_ROLE_TOOL = "tool"


def should_write_session_report(
    *,
    script_finished: bool,
    user_continues_runtime: bool,
) -> bool:
    """Return whether the hub should write ``report.json`` after a worker finishes."""
    if script_finished and user_continues_runtime:
        return False
    return True


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    return round(max(0.0, delta), 3)


def _parse_step_filename(path: Path) -> tuple[int, int] | None:
    stem = path.stem
    if "_" not in stem:
        return None
    left, right = stem.split("_", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _load_runtime_goals(run_root: Path) -> list[str]:
    path = run_root / _RUNTIME_COMMANDS_NAME
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_step_files(run_root: Path) -> list[tuple[int, int, dict[str, Any]]]:
    steps_dir = run_root / "steps"
    if not steps_dir.is_dir():
        return []
    loaded: list[tuple[int, int, dict[str, Any]]] = []
    for path in sorted(steps_dir.glob("*.json")):
        key = _parse_step_filename(path)
        if key is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            loaded.append((key[0], key[1], payload))
    loaded.sort(key=lambda item: (item[0], item[1]))
    return loaded


def _resolve_goal(
    transcript_counter: int,
    script_step_index: int,
    step_timing: dict[str, Any],
    runtime_goals: list[str],
) -> str:
    goal = step_timing.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    if script_step_index == 0 and 0 <= transcript_counter < len(runtime_goals):
        return runtime_goals[transcript_counter]
    return ""


def _tool_payload_from_message(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_assistant_tool_names(message: dict[str, Any]) -> list[str]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        func = call.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _describe_time_profile_entry(message: dict[str, Any]) -> dict[str, Any]:
    """
    Map a stamped transcript message to a human-readable phase description.

    Each profile row covers the interval starting at this message's timestamp until
    the next message (or step end):
    - user: screenshots were sent; duration is LLM inference.
    - assistant with tool_calls: LLM chose tools; duration is hand execution.
    - assistant without tool_calls: LLM declared step done; duration is wrap-up.
    - tool: tool result recorded; duration is post-action wait and next capture.
    """
    role = message.get("role")
    if role == _ROLE_USER:
        return {
            "kind": "llm_inference",
            "label": "LLM response generation after prompt and screenshots were sent",
        }
    if role == _ROLE_ASSISTANT:
        tool_names = _extract_assistant_tool_names(message)
        if tool_names:
            joined = ", ".join(tool_names)
            return {
                "kind": "tool_execution",
                "label": f"Hand tool execution: {joined}",
                "actions": tool_names,
            }
        return {
            "kind": "step_completion",
            "label": "Step completion after final LLM response",
        }
    if role == _ROLE_TOOL:
        payload = _tool_payload_from_message(message.get("content"))
        action = payload.get("action")
        entry: dict[str, Any] = {
            "kind": "screenshot_capture",
            "label": "Screenshot capture and prompt prep for next decision",
        }
        if isinstance(action, str) and action:
            entry["action"] = action
        if "ok" in payload:
            entry["ok"] = bool(payload.get("ok"))
        return entry
    role_label = str(role) if role else "unknown"
    return {"kind": role_label, "label": f"Unhandled message role: {role_label}"}


def _build_time_profile(
    messages: list[dict[str, Any]],
    finished_at_utc: str | None,
) -> list[dict[str, Any]]:
    if not messages:
        return []

    end_boundary = _parse_iso(finished_at_utc)
    profile: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        started = _parse_iso(message.get("timestamp_utc"))
        if started is None:
            continue

        if index + 1 < len(messages):
            next_started = _parse_iso(messages[index + 1].get("timestamp_utc"))
        else:
            next_started = end_boundary

        duration = _duration_seconds(started, next_started)
        entry: dict[str, Any] = {
            "started_at_utc": started.isoformat(),
            **_describe_time_profile_entry(message),
        }
        if duration is not None:
            entry["duration_seconds"] = duration

        profile.append(entry)

    return profile


def _build_step_records(
    step_files: list[tuple[int, int, dict[str, Any]]],
    runtime_goals: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for transcript_counter, script_step_index, payload in step_files:
        step_timing_raw = payload.get("step_timing")
        step_timing = dict(step_timing_raw) if isinstance(step_timing_raw, dict) else {}
        messages_raw = payload.get("messages")
        messages = [msg for msg in messages_raw if isinstance(msg, dict)] if isinstance(messages_raw, list) else []

        timing = {
            key: step_timing[key]
            for key in ("started_at_utc", "finished_at_utc", "duration_seconds", "status")
            if key in step_timing
        }

        record: dict[str, Any] = {
            "transcript_counter": transcript_counter,
            "script_step_index": script_step_index,
            "goal": _resolve_goal(transcript_counter, script_step_index, step_timing, runtime_goals),
            "timing": timing,
            "time_profile": _build_time_profile(messages, step_timing.get("finished_at_utc")),
        }
        records.append(record)
    return records


def _parse_csv_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _parse_csv_args(value: str | None) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _find_step_for_timestamp(
    tool_time: datetime,
    step_records: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    for record in step_records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        started = _parse_iso(timing.get("started_at_utc"))
        finished = _parse_iso(timing.get("finished_at_utc"))
        if started is None:
            continue
        if finished is None:
            if tool_time >= started:
                return record.get("transcript_counter"), record.get("script_step_index")
            continue
        if started <= tool_time <= finished:
            return record.get("transcript_counter"), record.get("script_step_index")

    if not step_records:
        return None, None

    nearest = min(
        step_records,
        key=lambda record: abs(
            (
                _parse_iso((record.get("timing") or {}).get("started_at_utc"))
                or datetime.min.replace(tzinfo=timezone.utc)
            )
            - tool_time
        ).total_seconds(),
    )
    return nearest.get("transcript_counter"), nearest.get("script_step_index")


def _load_tool_results(
    run_root: Path,
    step_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hand_csv = run_root / "hand.csv"
    if not hand_csv.is_file() or hand_csv.stat().st_size == 0:
        return []

    results: list[dict[str, Any]] = []
    with hand_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp_raw = row.get("timestamp")
            tool_time = _parse_iso(timestamp_raw if isinstance(timestamp_raw, str) else None)
            if tool_time is None:
                continue
            transcript_counter, script_step_index = _find_step_for_timestamp(tool_time, step_records)
            entry: dict[str, Any] = {
                "timestamp_utc": tool_time.isoformat(),
                "action": row.get("action") or "",
                "args": _parse_csv_args(row.get("args")),
                "ok": _parse_csv_bool(row.get("ok")),
                "message": row.get("message") or "",
                "screenshot_name": row.get("screenshot_name") or "",
            }
            if transcript_counter is not None:
                entry["transcript_counter"] = transcript_counter
            if script_step_index is not None:
                entry["script_step_index"] = script_step_index
            results.append(entry)

    results.sort(key=lambda item: item["timestamp_utc"])
    return results


def _build_summary(
    step_records: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_steps = 0
    total_duration = 0.0
    has_duration = False
    for record in step_records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        if timing.get("status") == "failed":
            failed_steps += 1
        duration = timing.get("duration_seconds")
        if isinstance(duration, (int, float)):
            total_duration += float(duration)
            has_duration = True

    failed_tools = sum(1 for item in tool_results if not item.get("ok", False))
    summary: dict[str, Any] = {
        "step_count": len(step_records),
        "tool_call_count": len(tool_results),
        "failed_step_count": failed_steps,
        "failed_tool_count": failed_tools,
    }
    if has_duration:
        summary["total_duration_seconds"] = round(total_duration, 3)
    return summary


def _resolve_script_metadata(run_root: Path) -> dict[str, str]:
    script_path_raw = os.environ.get(SCRIPT_PATH_ENV, "").strip()
    if script_path_raw:
        path = Path(script_path_raw)
        return {"script_path": str(path), "script_name": path.name}

    log_path = run_root / "run.log"
    if log_path.is_file():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if _QUEUE_SCRIPT_LOG_MARKER in line:
                    name = line.split(_QUEUE_SCRIPT_LOG_MARKER, 1)[1].strip()
                    if name:
                        return {"script_name": name}
        except OSError:
            pass

    if (run_root / _RUNTIME_COMMANDS_NAME).is_file():
        return {"script_name": _RUNTIME_COMMANDS_NAME}

    return {}


def _resolve_started_at_utc(run_root: Path, step_records: list[dict[str, Any]]) -> str | None:
    earliest: datetime | None = None
    for record in step_records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        started = _parse_iso(timing.get("started_at_utc"))
        if started is None:
            continue
        if earliest is None or started < earliest:
            earliest = started
    if earliest is not None:
        return earliest.isoformat()

    match = _RUN_FOLDER_TS_RE.match(run_root.name)
    if match is None:
        return None
    date_part, time_part = match.groups()
    try:
        return (
            datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
            .replace(tzinfo=timezone.utc)
            .isoformat()
        )
    except ValueError:
        return None


def build_session_report(run_root: Path, *, session_end_reason: str) -> dict[str, Any]:
    """Aggregate step timing and tool results from a run folder into a report dict."""
    step_files = _load_step_files(run_root)
    runtime_goals = _load_runtime_goals(run_root)
    step_records = _build_step_records(step_files, runtime_goals)
    tool_results = _load_tool_results(run_root, step_records)
    script_meta = _resolve_script_metadata(run_root)
    started_at_utc = _resolve_started_at_utc(run_root, step_records)

    report: dict[str, Any] = {
        "version": _REPORT_VERSION,
        "run_id": run_root.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_end_reason": session_end_reason,
        "summary": _build_summary(step_records, tool_results),
        "steps": step_records,
        "tool_results": tool_results,
    }
    if script_meta:
        report.update(script_meta)
    if started_at_utc is not None:
        report["started_at_utc"] = started_at_utc
    return report


def write_session_report(run_root: Path, *, session_end_reason: str) -> Path:
    """Write ``report.json`` under ``run_root`` and rebuild ``session_steps.html``; return report path."""
    report_path = run_root / "report.json"
    write_json(report_path, build_session_report(run_root, session_end_reason=session_end_reason))

    from src.common.session_html import write_session_html_from_run

    write_session_html_from_run(run_root)
    return report_path
