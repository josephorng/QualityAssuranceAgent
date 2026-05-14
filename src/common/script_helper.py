from __future__ import annotations

from datetime import datetime
from pathlib import Path


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
    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        lines.append(cleaned)
    return lines


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
