from __future__ import annotations

from datetime import datetime
from pathlib import Path

_EXPECTED_OUTCOME_PREFIX = "# expected_outcome:"


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
