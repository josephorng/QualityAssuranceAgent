from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.common.instruction_tool_cache import upsert_tool_calls
from src.common.io_utils import append_text, read_json, write_json
from src.common.run_state import get_run_state_manager, reset_run_state_manager
from src.common.runtime_context import set_runtime_env
from src.recorder.analyze import analyze_event_to_cache
from src.recorder.models import RecordedEvent, SessionManifest
from src.recorder.coalesce import coalesce_consecutive_text_inputs
from src.recorder.to_cache import validate_tool_calls
from src.recorder.vision_context import build_vision_context


def _load_events(run_dir: Path) -> list[RecordedEvent]:
    manifest_raw = read_json(run_dir / "session.json", {})
    events: list[RecordedEvent] = []
    if isinstance(manifest_raw, dict):
        paths = manifest_raw.get("events")
        if isinstance(paths, list):
            for rel in paths:
                if not isinstance(rel, str):
                    continue
                raw = read_json(run_dir / rel, None)
                if isinstance(raw, dict):
                    events.append(RecordedEvent.from_dict(raw))
            return events

    events_dir = run_dir / "events"
    if not events_dir.is_dir():
        return []
    for path in sorted(events_dir.glob("event_*.json")):
        raw = read_json(path, None)
        if isinstance(raw, dict):
            events.append(RecordedEvent.from_dict(raw))
    return events


def _bind_run_state_for_analysis(run_dir: Path, run_id: str) -> None:
    """Attach the recording folder as the active run so vision/LLM logging can write run.log."""
    run_dir = run_dir.resolve()
    set_runtime_env(run_dir, run_id)
    reset_run_state_manager()
    manager = get_run_state_manager()
    if manager.paths is None:
        manager.init_run("screen_recording_analysis", run_folder_name=run_dir.name)


async def analyze_recording_session(
    run_dir: Path,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Analyze all events in a recording session and write instruction_tool_cache entries."""
    run_dir = Path(run_dir)
    log_path = run_dir / "import.log"
    manifest_raw = read_json(run_dir / "session.json", {})
    run_id = run_dir.name
    if isinstance(manifest_raw, dict) and isinstance(manifest_raw.get("run_id"), str):
        run_id = manifest_raw["run_id"]

    def log_info(text: str) -> None:
        append_text(log_path, f"{text}\n")

    _bind_run_state_for_analysis(run_dir, run_id)
    try:
        events = coalesce_consecutive_text_inputs(_load_events(run_dir))
        log_info(f"analyze_recording_session start events={len(events)} run_id={run_id}")

        cached = 0
        skipped = 0
        cancelled = False
        processed = 0
        errors: list[dict[str, Any]] = []
        instructions: list[str] = []
        total = len(events)

        if on_progress is not None:
            on_progress(0, total)

        for event in events:
            if should_cancel is not None and should_cancel():
                cancelled = True
                log_info("analyze_recording_session cancelled by user")
                break
            log_info(f"processing event {event.index} kind={event.kind}")
            vision = build_vision_context(event, run_dir=run_dir)
            analysis_path = run_dir / "analysis" / f"event_{event.index:03d}.json"
            analysis_path.parent.mkdir(parents=True, exist_ok=True)

            result = await analyze_event_to_cache(
                event,
                run_dir=run_dir,
                vision=vision,
                log_info=log_info,
            )
            if result is None:
                skipped += 1
                errors.append({"event_index": event.index, "error": "llm_analysis_failed"})
            else:
                instruction = result["instruction"]
                tool_calls = result["tool_calls"]
                err = validate_tool_calls(tool_calls)
                if err:
                    skipped += 1
                    errors.append({"event_index": event.index, "error": err})
                else:
                    upsert_tool_calls(instruction, tool_calls, source_run_id=run_id)
                    cached += 1
                    instructions.append(instruction)
                    write_json(
                        analysis_path,
                        {
                            "event_index": event.index,
                            "instruction": instruction,
                            "tool_calls": tool_calls,
                            "vision": {
                                "used_vision": vision.get("used_vision"),
                                "candidate_text": vision.get("candidate_text"),
                            },
                        },
                    )
                    log_info(f"cached event {event.index}: {instruction}")

            processed += 1
            if on_progress is not None:
                on_progress(processed, total)

        report = {
            "run_id": run_id,
            "recorded": len(events),
            "processed": processed,
            "cached": cached,
            "skipped": skipped,
            "cancelled": cancelled,
            "errors": errors,
            "instructions": instructions,
        }
        write_json(run_dir / "report.json", report)
        log_info(
            f"analyze_recording_session done recorded={len(events)} cached={cached} skipped={skipped}"
        )
        return report
    finally:
        reset_run_state_manager()
