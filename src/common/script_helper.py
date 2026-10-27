from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from src.recorder.analyze import (
    after_screenshot_for_outcome,
    typed_text_from_instruction,
    use_expected_outcome_enabled,
)
from src.recorder.models import RecordedEvent
from src.recorder.verify_signals import omit_focused_value_unrelated_to_typed_text
from src.recorder.window_snapshot import (
    analysis_uses_chrome_window_tool,
    filter_disabled_window_verify,
    omit_click_window_for_chrome_tools,
    omit_foreground_when_structural,
)

_EXPECTED_OUTCOME_PREFIX = "# expected_outcome:"
_LEGACY_RECORDING_SCRIPT_FILENAME = "script.txt"

_ReplaySteps = tuple[
    list[str],
    list[str | None],
    list[str | None],
    list[float | None],
    list[dict[str, Any]],
    list[int | None],
]
_InstructionRows = list[tuple[RecordedEvent, dict[str, Any]]]

# ``run_dir.resolve()`` → ``(fingerprint, payload)``
_replay_steps_cache: dict[str, tuple[tuple[float, int], _ReplaySteps]] = {}
_instruction_rows_cache: dict[str, tuple[tuple[float, int], _InstructionRows]] = {}


def _safe_is_dir(path: Path) -> bool:
    """Like ``Path.is_dir`` but treat permission / IO failures as missing."""
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    """Like ``Path.is_file`` but treat permission / IO failures as missing."""
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_resolve_key(path: Path) -> str | None:
    """Return a resolved path key, or ``None`` when the path is inaccessible."""
    try:
        return str(Path(path).resolve())
    except OSError:
        return None


def is_recording_dir(path: Path) -> bool:
    """True when ``path`` is a recording folder (has ``session.json``)."""
    candidate = Path(path)
    return _safe_is_dir(candidate) and _safe_is_file(candidate / "session.json")


def partition_recording_dirs(
    picked: Sequence[Path],
    *,
    existing: Sequence[Path] = (),
) -> tuple[list[Path], list[Path]]:
    """Split picked folders into new recordings vs invalid paths.

    Duplicates (including folders already in ``existing``) are omitted from both lists.
    """
    seen: set[str] = set()
    for path in existing:
        key = _safe_resolve_key(path)
        if key is not None:
            seen.add(key)
    added: list[Path] = []
    invalid: list[Path] = []
    for raw in picked:
        folder = Path(raw)
        if not is_recording_dir(folder):
            invalid.append(folder)
            continue
        key = _safe_resolve_key(folder)
        if key is None:
            invalid.append(folder)
            continue
        if key in seen:
            continue
        seen.add(key)
        added.append(folder)
    return added, invalid


def recording_run_dir(path: Path) -> Path | None:
    """Return the recording folder for a dir or legacy ``script.txt`` inside one."""
    candidate = Path(path)
    if is_recording_dir(candidate):
        return candidate
    if (
        _safe_is_file(candidate)
        and candidate.name == _LEGACY_RECORDING_SCRIPT_FILENAME
        and is_recording_dir(candidate.parent)
    ):
        return candidate.parent
    return None


def is_recording_script_path(path: Path) -> bool:
    """True when ``path`` is a recording folder (or legacy ``script.txt`` inside one)."""
    return recording_run_dir(path) is not None


def is_runnable_script_path(path: Path) -> bool:
    """True when ``path`` is an existing script file or recording folder."""
    resolved = resolve_runnable_script_path(path)
    return _safe_is_file(resolved) or is_recording_dir(resolved)


def resolve_runnable_script_path(path: Path) -> Path:
    """Resolve a recording folder (or legacy ``script.txt``) to the recording dir.

    Plain ``.txt`` script files are returned unchanged.
    """
    rec = recording_run_dir(path)
    if rec is not None:
        return rec
    return Path(path)


def script_display_name(path: Path) -> str:
    """Human-facing name: recording folder name, else file name."""
    rec = recording_run_dir(path)
    if rec is not None:
        return rec.name
    return Path(path).name


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def recording_event_json_paths(run_dir: Path) -> list[Path]:
    """Event JSON paths in ``session.json`` order, or filename order as fallback."""
    run_dir = Path(run_dir)
    manifest = _load_json_dict(run_dir / "session.json")
    if isinstance(manifest, dict) and isinstance(manifest.get("events"), list):
        event_paths: list[Path] = []
        seen: set[str] = set()
        for item in manifest["events"]:
            if not isinstance(item, str) or not item.strip():
                continue
            path = run_dir / item
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            event_paths.append(path)
        return event_paths
    events_dir = run_dir / "events"
    if events_dir.is_dir():
        return sorted(path for path in events_dir.glob("event_*.json") if path.is_file())
    return []


_recording_event_json_paths = recording_event_json_paths


def _resolve_recording_media_path(run_dir: Path, raw: str | None) -> str | None:
    """Return an existing media path under ``run_dir``.

    Handles relative paths, absolute paths that still exist, and relocated /
    renamed recordings whose JSON still points at a pre-rename absolute path
    (same basename under ``run_dir/screenshots/``).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip())
    run_dir = Path(run_dir)
    candidates: list[Path] = []
    if path.is_file():
        candidates.append(path)
    if not path.is_absolute():
        candidates.append(run_dir / path)
    # Renamed/copied recording folders leave absolute paths to the old dir.
    if path.name:
        candidates.append(run_dir / "screenshots" / path.name)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _recorded_event_with_resolved_shots(
    run_dir: Path, raw: dict[str, Any]
) -> RecordedEvent | None:
    """Build a ``RecordedEvent`` with screenshot paths resolved under ``run_dir``."""
    try:
        event = RecordedEvent.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None
    shot = _resolve_recording_media_path(run_dir, event.screenshot_path)
    end_shot = _resolve_recording_media_path(run_dir, event.end_screenshot_path)
    # Always rewrite (or clear) so callers never keep a stale absolute path.
    event.screenshot_path = shot or ""
    event.end_screenshot_path = end_shot or ""
    return event


def _recording_replay_cache_key(run_dir: Path) -> str:
    try:
        return str(Path(run_dir).resolve())
    except OSError:
        return str(Path(run_dir))


def _recording_replay_fingerprint(run_dir: Path) -> tuple[float, int]:
    """Max mtime + file count for session/events/analysis JSON (cache invalidation)."""
    run_dir = Path(run_dir)
    paths: list[Path] = [run_dir / "session.json"]
    events_dir = run_dir / "events"
    if events_dir.is_dir():
        paths.extend(path for path in events_dir.glob("event_*.json") if path.is_file())
    analysis_dir = run_dir / "analysis"
    if analysis_dir.is_dir():
        paths.extend(path for path in analysis_dir.glob("event_*.json") if path.is_file())
    mtimes: list[float] = []
    count = 0
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
            count += 1
        except OSError:
            continue
    return (max(mtimes) if mtimes else 0.0, count)


def invalidate_recording_replay_cache(run_dir: Path | None = None) -> None:
    """Drop cached replay steps/rows for ``run_dir``, or all recordings when omitted."""
    if run_dir is None:
        _replay_steps_cache.clear()
        _instruction_rows_cache.clear()
        return
    key = _recording_replay_cache_key(run_dir)
    _replay_steps_cache.pop(key, None)
    _instruction_rows_cache.pop(key, None)


def _load_instruction_event_rows_uncached(run_dir: Path) -> _InstructionRows:
    """Single pass: instruction events as ``(RecordedEvent, analysis)`` pairs."""
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    rows: _InstructionRows = []
    for event_path in _recording_event_json_paths(run_dir):
        raw = _load_json_dict(event_path)
        if raw is None:
            continue
        raw_index = raw.get("index")
        if not isinstance(raw_index, int):
            continue
        analysis = _load_json_dict(analysis_dir / f"event_{raw_index:03d}.json")
        if analysis is None:
            continue
        instruction = analysis.get("instruction")
        if not (isinstance(instruction, str) and instruction.strip()):
            continue
        event = _recorded_event_with_resolved_shots(run_dir, raw)
        if event is None:
            continue
        rows.append((event, analysis))
    return rows


def _load_instruction_event_rows(run_dir: Path) -> _InstructionRows:
    """Cached instruction-event rows (shared by collectors and full replay load)."""
    run_dir = Path(run_dir)
    key = _recording_replay_cache_key(run_dir)
    fingerprint = _recording_replay_fingerprint(run_dir)
    cached = _instruction_rows_cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    rows = _load_instruction_event_rows_uncached(run_dir)
    _instruction_rows_cache[key] = (fingerprint, rows)
    return rows


def _iter_recording_instructions(
    run_dir: Path,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Instruction events in replay order: ``(event index, instruction, analysis)``.

    Virtual ``wait_instruction`` fields are ignored; manual ``kind=wait`` events
    still appear via their own analysis ``instruction``. Events with no analysis
    or a blank instruction are skipped.
    """
    found: list[tuple[int, str, dict[str, Any]]] = []
    for event, analysis in _load_instruction_event_rows(run_dir):
        instruction = analysis.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            found.append((event.index, instruction.strip(), analysis))
    return found


def collect_recording_instruction_event_indices(run_dir: Path) -> list[int]:
    """Recording event ids aligned with ``collect_recording_instructions``.

    These are the ``id="event-N"`` anchors in ``recording_steps.html``.
    """
    return [event.index for event, _analysis in _load_instruction_event_rows(run_dir)]


def collect_recording_instructions(run_dir: Path) -> tuple[list[str], list[str | None]]:
    """Collect hub-script lines from recording analysis files.

    Virtual ``wait_instruction`` fields are ignored; manual ``kind=wait`` events
    still appear via their own analysis ``instruction``.
    """
    instructions: list[str] = []
    expected_outcomes: list[str | None] = []
    for _event, analysis in _load_instruction_event_rows(run_dir):
        instruction = analysis.get("instruction")
        instructions.append(
            instruction.strip() if isinstance(instruction, str) else ""
        )
        if use_expected_outcome_enabled(analysis):
            outcome = analysis.get("expected_outcome")
            if isinstance(outcome, str) and outcome.strip():
                expected_outcomes.append(outcome.strip())
            else:
                expected_outcomes.append(None)
        else:
            expected_outcomes.append(None)
    return instructions, expected_outcomes


def _final_after_screenshot_path(run_dir: Path) -> str | None:
    session = _load_json_dict(Path(run_dir) / "session.json") or {}
    final_after = _resolve_recording_media_path(
        run_dir, session.get("final_after_screenshot") if isinstance(session, dict) else None
    )
    if final_after is None:
        final_after = _resolve_recording_media_path(
            run_dir, str(Path(run_dir) / "screenshots" / "final_after.jpeg")
        )
    return final_after


def _baselines_from_rows(
    run_dir: Path,
    rows: _InstructionRows,
    *,
    include_disabled: bool,
) -> list[str | None]:
    final_after = _final_after_screenshot_path(run_dir)
    baselines: list[str | None] = []
    for index, (event, analysis) in enumerate(rows):
        if not include_disabled and not use_expected_outcome_enabled(analysis):
            baselines.append(None)
            continue
        next_event = rows[index + 1][0] if index + 1 < len(rows) else None
        baselines.append(
            after_screenshot_for_outcome(
                event,
                next_event,
                final_after_screenshot=final_after,
            )
        )
    return baselines


def collect_recording_baseline_after_paths(
    run_dir: Path,
    *,
    include_disabled: bool = False,
) -> list[str | None]:
    """Collect recording after-screenshot paths aligned with ``collect_recording_instructions``.

    Action lines use ``after_screenshot_for_outcome``
    (next event before → typing/drag end → session ``final_after``).
    ``include_disabled`` keeps baselines for steps whose verification checkbox is off,
    so reports can show the recorded screen without turning verification on.
    """
    return _baselines_from_rows(
        Path(run_dir),
        _load_instruction_event_rows(run_dir),
        include_disabled=include_disabled,
    )


def _settles_from_rows(rows: _InstructionRows) -> list[float | None]:
    settles: list[float | None] = []
    for index, (event, analysis) in enumerate(rows):
        if event.kind == "wait":
            settles.append(None)
            continue
        analysis_settle = analysis.get("settle_after_seconds")
        if (
            isinstance(analysis_settle, (int, float))
            and not isinstance(analysis_settle, bool)
            and float(analysis_settle) >= 0
        ):
            settles.append(float(analysis_settle))
            continue
        observed = event.observed_settle_seconds
        if (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and float(observed) >= 0
        ):
            settles.append(float(observed))
            continue
        if index + 1 >= len(rows):
            settles.append(None)
            continue
        end_ts = rows[index + 1][0].timestamp_utc
        settle = _elapsed_seconds_between(event.timestamp_utc, end_ts)
        if settle is not None and settle >= 1.0:
            settles.append(settle)
        else:
            settles.append(None)
    return settles


def collect_recording_settle_after_seconds(run_dir: Path) -> list[float | None]:
    """Collect post-action settle seconds aligned with ``collect_recording_instructions``.

    Preference order per instruction event:
    1. ``analysis/event_NNN.json`` ``settle_after_seconds`` (including 0)
    2. event ``observed_settle_seconds`` (recording settle probe, including 0)
    3. timestamp gap to next instruction event when ≥ 1s

    The last event does not fall back to session ``stopped_at`` (stop-UI latency).
    Gaps under 1s (legacy path) and ``kind=wait`` events yield ``None``.
    """
    return _settles_from_rows(_load_instruction_event_rows(run_dir))


def _replay_typed_text(analysis: dict[str, Any]) -> str | None:
    """Text ``type_text`` will enter for this step, when the analysis records it."""
    calls = analysis.get("tool_calls")
    if isinstance(calls, list):
        chunks: list[str] = []
        for call in calls:
            if not isinstance(call, dict) or call.get("name") != "type_text":
                continue
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                continue
            text = arguments.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return "".join(chunks)
    instruction = analysis.get("instruction")
    if isinstance(instruction, str):
        return typed_text_from_instruction(instruction)
    return None


def _window_verify_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    raw = analysis.get("window_verify")
    if not isinstance(raw, dict):
        return {}
    active = filter_disabled_window_verify(
        raw, analysis.get("window_verify_disabled")
    )
    active = omit_foreground_when_structural(active)
    if analysis_uses_chrome_window_tool(analysis):
        active = omit_click_window_for_chrome_tools(active)
    return omit_focused_value_unrelated_to_typed_text(
        active,
        _replay_typed_text(analysis),
    )


def collect_recording_window_verifies(run_dir: Path) -> list[dict[str, Any]]:
    """Collect window-diff predicates aligned with ``collect_recording_instructions``.

    An empty dict means the step has no window assertion.
    """
    return [
        _window_verify_from_analysis(analysis)
        for _event, analysis in _load_instruction_event_rows(run_dir)
    ]


def recording_replay_steps(
    lines: list[str],
    outcomes: list[str | None],
    baselines: list[str | None],
    settles: list[float | None],
    verifies: list[dict[str, Any]],
    event_indices: list[int | None],
) -> tuple[
    list[str],
    list[str | None],
    list[str | None],
    list[float | None],
    list[dict[str, Any]],
    list[int | None],
]:
    """Pad collectors to ``len(lines)`` and copy verify dicts."""
    count = len(lines)
    padded_outcomes: list[str | None] = []
    padded_baselines: list[str | None] = []
    padded_settles: list[float | None] = []
    padded_verifies: list[dict[str, Any]] = []
    padded_events: list[int | None] = []
    for index in range(count):
        outcome = outcomes[index] if index < len(outcomes) else None
        padded_outcomes.append(outcome if isinstance(outcome, str) and outcome.strip() else None)
        baseline = baselines[index] if index < len(baselines) else None
        padded_baselines.append(baseline if isinstance(baseline, str) and baseline.strip() else None)
        settle = settles[index] if index < len(settles) else None
        if isinstance(settle, bool) or settle is None:
            padded_settles.append(None)
        elif isinstance(settle, (int, float)):
            padded_settles.append(float(settle))
        else:
            padded_settles.append(None)
        verify = verifies[index] if index < len(verifies) else {}
        padded_verifies.append(dict(verify) if isinstance(verify, dict) else {})
        event_index = event_indices[index] if index < len(event_indices) else None
        padded_events.append(event_index if isinstance(event_index, int) else None)
    return (
        list(lines),
        padded_outcomes,
        padded_baselines,
        padded_settles,
        padded_verifies,
        padded_events,
    )


def _build_recording_replay_steps_from_rows(
    run_dir: Path,
    rows: _InstructionRows,
) -> _ReplaySteps:
    instructions: list[str] = []
    outcomes: list[str | None] = []
    event_indices: list[int | None] = []
    verifies: list[dict[str, Any]] = []
    for event, analysis in rows:
        instruction = analysis.get("instruction")
        instructions.append(
            instruction.strip() if isinstance(instruction, str) else ""
        )
        if use_expected_outcome_enabled(analysis):
            outcome = analysis.get("expected_outcome")
            if isinstance(outcome, str) and outcome.strip():
                outcomes.append(outcome.strip())
            else:
                outcomes.append(None)
        else:
            outcomes.append(None)
        event_indices.append(event.index)
        verifies.append(_window_verify_from_analysis(analysis))
    return recording_replay_steps(
        instructions,
        outcomes,
        _baselines_from_rows(run_dir, rows, include_disabled=False),
        _settles_from_rows(rows),
        verifies,
        event_indices,
    )


def load_recording_replay_steps(
    run_dir: Path,
) -> tuple[
    list[str],
    list[str | None],
    list[str | None],
    list[float | None],
    list[dict[str, Any]],
    list[int | None],
]:
    """Read the recording's current replay steps from disk (single-pass, mtime-cached)."""
    run_dir = Path(run_dir)
    key = _recording_replay_cache_key(run_dir)
    fingerprint = _recording_replay_fingerprint(run_dir)
    cached = _replay_steps_cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    rows = _load_instruction_event_rows(run_dir)
    steps = _build_recording_replay_steps_from_rows(run_dir, rows)
    _replay_steps_cache[key] = (fingerprint, steps)
    return steps


def merge_unstarted_recording_steps(
    *,
    step_index: int,
    current_lines: list[str],
    current_outcomes: list[str | None],
    current_baselines: list[str | None],
    current_settles: list[float | None],
    current_verifies: list[dict[str, Any]],
    current_event_indices: list[int | None],
    fresh_lines: list[str],
    fresh_outcomes: list[str | None],
    fresh_baselines: list[str | None],
    fresh_settles: list[float | None],
    fresh_verifies: list[dict[str, Any]],
    fresh_event_indices: list[int | None],
) -> tuple[
    list[str],
    list[str | None],
    list[str | None],
    list[float | None],
    list[dict[str, Any]],
    list[int | None],
]:
    """Keep steps that already started, and replace the rest from ``fresh``.

    Completed steps stay put so the step cursor does not shift. Unstarted steps
    are matched by recording event id when those ids are known, so an edit,
    insert, or delete of a step that has not started is picked up. Without ids,
    the unstarted tail is taken by position.
    """
    fresh = recording_replay_steps(
        fresh_lines,
        fresh_outcomes,
        fresh_baselines,
        fresh_settles,
        fresh_verifies,
        fresh_event_indices,
    )
    prefix_len = max(0, min(step_index, len(current_lines)))
    if prefix_len == 0:
        return fresh

    prefix_ids: list[int | None] = []
    for index in range(prefix_len):
        event_index = (
            current_event_indices[index] if index < len(current_event_indices) else None
        )
        prefix_ids.append(event_index if isinstance(event_index, int) else None)
    id_aligned = all(isinstance(event_index, int) for event_index in prefix_ids)

    if id_aligned:
        consumed = {event_index for event_index in prefix_ids if isinstance(event_index, int)}
        tail_indexes = [
            index
            for index, event_index in enumerate(fresh[5])
            if isinstance(event_index, int) and event_index not in consumed
        ]
    else:
        tail_indexes = list(range(prefix_len, len(fresh[0])))

    lines = [current_lines[index] for index in range(prefix_len)]
    outcomes: list[str | None] = []
    baselines: list[str | None] = []
    settles: list[float | None] = []
    verifies: list[dict[str, Any]] = []
    event_indices: list[int | None] = list(prefix_ids)
    for index in range(prefix_len):
        outcome = current_outcomes[index] if index < len(current_outcomes) else None
        outcomes.append(outcome if isinstance(outcome, str) and outcome.strip() else None)
        baseline = current_baselines[index] if index < len(current_baselines) else None
        baselines.append(baseline if isinstance(baseline, str) and baseline.strip() else None)
        settle = current_settles[index] if index < len(current_settles) else None
        if isinstance(settle, bool) or settle is None:
            settles.append(None)
        elif isinstance(settle, (int, float)):
            settles.append(float(settle))
        else:
            settles.append(None)
        verify = current_verifies[index] if index < len(current_verifies) else {}
        verifies.append(dict(verify) if isinstance(verify, dict) else {})

    fresh_lines_aligned = fresh[0]
    fresh_outcomes_aligned = fresh[1]
    fresh_baselines_aligned = fresh[2]
    fresh_settles_aligned = fresh[3]
    fresh_verifies_aligned = fresh[4]
    fresh_events_aligned = fresh[5]
    for index in tail_indexes:
        lines.append(fresh_lines_aligned[index])
        outcomes.append(fresh_outcomes_aligned[index])
        baselines.append(fresh_baselines_aligned[index])
        settles.append(fresh_settles_aligned[index])
        verifies.append(dict(fresh_verifies_aligned[index]))
        event_indices.append(fresh_events_aligned[index])
    return (lines, outcomes, baselines, settles, verifies, event_indices)


def _elapsed_seconds_between(
    previous_timestamp_utc: str, current_timestamp_utc: str | None
) -> float | None:
    """Return non-negative elapsed seconds between two timezone-aware ISO timestamps."""
    if not isinstance(current_timestamp_utc, str) or not current_timestamp_utc.strip():
        return None
    try:
        previous = datetime.fromisoformat(previous_timestamp_utc.replace("Z", "+00:00"))
        current = datetime.fromisoformat(
            current_timestamp_utc.strip().replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None
    if previous.tzinfo is None or current.tzinfo is None:
        return None
    elapsed = (current - previous).total_seconds()
    return elapsed if elapsed >= 0 else None


def collect_recording_script_text(run_dir: Path) -> str:
    """Format collected recording instructions as hub-script text."""
    instructions, outcomes = collect_recording_instructions(run_dir)
    lines = format_script_lines_with_outcomes(instructions, outcomes)
    return ("\n".join(lines).rstrip() + "\n") if lines else ""


def load_runnable_script_text(path: Path) -> str:
    """Read a ``.txt`` script, or collect instructions from a recording folder."""
    rec = recording_run_dir(path)
    if rec is not None:
        return collect_recording_script_text(rec)
    return Path(path).read_text(encoding="utf-8")


def resolve_task(cli_task: str | None) -> str:
    """Return CLI task text when provided, otherwise prompt interactively until non-empty."""
    if cli_task and cli_task.strip():
        return cli_task.strip()
    while True:
        task = input("Enter task: ").strip()
        if task:
            return task
        print("Task cannot be empty.")


def list_script_files(scripts_dir: Path) -> list[Path]:
    """List `.txt` script files under the scripts directory in stable sorted order."""
    if not scripts_dir.exists():
        return []
    return sorted([path for path in scripts_dir.glob("*.txt") if path.is_file()])


def parse_executable_lines_from_text(raw: str) -> list[str]:
    """Parse executable script lines from in-memory text (same rules as ``parse_script_lines``)."""
    steps, _outcomes = parse_script_steps_with_outcomes(raw)
    return steps


def parse_expected_outcome_comment(line: str) -> str | None:
    """Return the outcome text from a ``# expected_outcome:`` comment, else None."""
    cleaned = line.strip()
    if not cleaned.lower().startswith(_EXPECTED_OUTCOME_PREFIX):
        return None
    outcome = cleaned[len(_EXPECTED_OUTCOME_PREFIX) :].strip()
    return outcome or None


def parse_script_steps_with_outcomes(raw: str) -> tuple[list[str], list[str | None]]:
    """Parse executable steps and optional following ``# expected_outcome:`` comments."""
    steps: list[str] = []
    outcomes: list[str | None] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        outcome = parse_expected_outcome_comment(cleaned)
        if outcome is not None:
            if outcomes:
                outcomes[-1] = outcome
            continue
        if cleaned.startswith("#"):
            continue
        steps.append(cleaned)
        outcomes.append(None)
    return steps, outcomes


def format_script_lines_with_outcomes(
    instructions: list[str],
    expected_outcomes: list[str | None] | None = None,
) -> list[str]:
    """Build script text lines with optional ``# expected_outcome:`` comments after steps."""
    lines: list[str] = []
    outcomes = expected_outcomes or []
    for index, instruction in enumerate(instructions):
        text = instruction.strip()
        if not text:
            continue
        lines.append(text)
        outcome = outcomes[index] if index < len(outcomes) else None
        if isinstance(outcome, str) and outcome.strip():
            lines.append(f"{_EXPECTED_OUTCOME_PREFIX} {outcome.strip()}")
    return lines


def executable_source_line_numbers(raw: str) -> list[int]:
    """Return 1-based source line numbers for executable script lines (skip blanks/comments)."""
    numbers: list[int] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        numbers.append(index)
    return numbers


def parse_script_lines(script_path: Path) -> list[str]:
    """Parse executable script lines, skipping blanks and comment lines starting with `#`."""
    return parse_executable_lines_from_text(script_path.read_text(encoding="utf-8"))


def save_plain_task_script(task: str, scripts_dir: Path) -> Path:
    """Persist a one-line ad-hoc task as a timestamped script file and return its path."""
    scripts_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = scripts_dir / f"adhoc_{stamp}.txt"
    script_path.write_text(task.strip() + "\n", encoding="utf-8")
    return script_path


def resolve_task_and_script(cli_task: str | None, root_dir: Path) -> tuple[str, Path, list[str]]:
    """Resolve user input into task text, backing script path, and ordered step lines."""
    scripts_dir = root_dir / "scripts"
    if cli_task and cli_task.strip():
        task = resolve_task(cli_task)
        script_path = save_plain_task_script(task, scripts_dir)
        return task, script_path, [task]

    scripts = list_script_files(scripts_dir)
    while True:
        print("Choose input mode:")
        print("  1) Type task text")
        if scripts:
            print("  2) Choose script file from scripts/")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            task = resolve_task(None)
            script_path = save_plain_task_script(task, scripts_dir)
            return task, script_path, [task]

        if choice == "2" and scripts:
            print("Available scripts:")
            for idx, script in enumerate(scripts, start=1):
                print(f"  {idx}) {script.name}")
            selected = input("Select script number: ").strip()
            if not selected.isdigit():
                print("Please enter a valid number.")
                continue
            selected_index = int(selected) - 1
            if selected_index < 0 or selected_index >= len(scripts):
                print("Selected number is out of range.")
                continue
            script_path = scripts[selected_index]
            script_steps = parse_script_lines(script_path)
            if not script_steps:
                print("Selected script has no executable lines. Add steps and try again.")
                continue
            task = script_steps[0]
            return task, script_path, script_steps

        print("Invalid choice. Please try again.")
