from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from src.recorder.analyze import after_screenshot_for_outcome, use_expected_outcome_enabled
from src.recorder.models import RecordedEvent

_EXPECTED_OUTCOME_PREFIX = "# expected_outcome:"
_LEGACY_RECORDING_SCRIPT_FILENAME = "script.txt"


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


def _recording_event_json_paths(run_dir: Path) -> list[Path]:
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


def collect_recording_instructions(run_dir: Path) -> tuple[list[str], list[str | None]]:
    """Collect hub-script lines from recording analysis files.

    Virtual ``wait_instruction`` fields are ignored; manual ``kind=wait`` events
    still appear via their own analysis ``instruction``.
    """
    analysis_dir = Path(run_dir) / "analysis"
    instructions: list[str] = []
    expected_outcomes: list[str | None] = []
    for event_path in _recording_event_json_paths(run_dir):
        event = _load_json_dict(event_path)
        if event is None:
            continue
        raw_index = event.get("index")
        if not isinstance(raw_index, int):
            continue
        analysis = _load_json_dict(analysis_dir / f"event_{raw_index:03d}.json")
        if analysis is None:
            continue
        instruction = analysis.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            instructions.append(instruction.strip())
            if use_expected_outcome_enabled(analysis):
                outcome = analysis.get("expected_outcome")
                if isinstance(outcome, str) and outcome.strip():
                    expected_outcomes.append(outcome.strip())
                else:
                    expected_outcomes.append(None)
            else:
                expected_outcomes.append(None)
    return instructions, expected_outcomes


def collect_recording_baseline_after_paths(run_dir: Path) -> list[str | None]:
    """Collect recording after-screenshot paths aligned with ``collect_recording_instructions``.

    Action lines use ``after_screenshot_for_outcome``
    (next event before → typing/drag end → session ``final_after``).
    """
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    session = _load_json_dict(run_dir / "session.json") or {}
    final_after = _resolve_recording_media_path(
        run_dir, session.get("final_after_screenshot") if isinstance(session, dict) else None
    )
    if final_after is None:
        final_after = _resolve_recording_media_path(
            run_dir, str(run_dir / "screenshots" / "final_after.jpeg")
        )

    event_paths = _recording_event_json_paths(run_dir)
    loaded: list[tuple[RecordedEvent, dict[str, Any]]] = []
    for event_path in event_paths:
        raw = _load_json_dict(event_path)
        if raw is None:
            continue
        raw_index = raw.get("index")
        if not isinstance(raw_index, int):
            continue
        analysis = _load_json_dict(analysis_dir / f"event_{raw_index:03d}.json")
        if analysis is None:
            continue
        event = _recorded_event_with_resolved_shots(run_dir, raw)
        if event is None:
            continue
        instruction = analysis.get("instruction")
        if not (isinstance(instruction, str) and instruction.strip()):
            continue
        loaded.append((event, analysis))

    baselines: list[str | None] = []
    for index, (event, analysis) in enumerate(loaded):
        if not use_expected_outcome_enabled(analysis):
            # Opt-in verification: no baseline when the step's checkbox is off.
            baselines.append(None)
            continue
        next_event = loaded[index + 1][0] if index + 1 < len(loaded) else None
        baselines.append(
            after_screenshot_for_outcome(
                event,
                next_event,
                final_after_screenshot=final_after,
            )
        )
    return baselines


def collect_recording_settle_after_seconds(run_dir: Path) -> list[float | None]:
    """Collect post-action settle seconds aligned with ``collect_recording_instructions``.

    Derived from event timestamps (this → next instruction event). The last action
    uses ``session.stopped_at_utc`` when present. Gaps under 1s and ``kind=wait``
    events yield ``None``.
    """
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    session = _load_json_dict(run_dir / "session.json") or {}
    stopped_at_utc = (
        session.get("stopped_at_utc") if isinstance(session, dict) else None
    )
    if not isinstance(stopped_at_utc, str):
        stopped_at_utc = None
    loaded: list[RecordedEvent] = []
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
        loaded.append(event)

    settles: list[float | None] = []
    for index, event in enumerate(loaded):
        if event.kind == "wait":
            settles.append(None)
            continue
        if index + 1 < len(loaded):
            end_ts = loaded[index + 1].timestamp_utc
        else:
            end_ts = stopped_at_utc
        settle = _elapsed_seconds_between(event.timestamp_utc, end_ts)
        if settle is not None and settle >= 1.0:
            settles.append(settle)
        else:
            settles.append(None)
    return settles


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
