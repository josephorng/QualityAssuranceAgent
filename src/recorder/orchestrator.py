from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

from src.common.io_utils import append_text, read_json, write_json
from src.common.run_state import get_run_state_manager, reset_run_state_manager
from src.common.runtime_context import set_runtime_env
from src.common.settings import load_settings
from src.recorder.analyze import analyze_event_to_cache
from src.recorder.models import RecordedEvent, SessionManifest
from src.recorder.coalesce import coalesce_consecutive_text_inputs
from src.recorder.text_resolve import event_with_resolved_text, resolve_text_input_text
from src.recorder.vision_context import build_vision_context
from src.recorder.window_snapshot import is_agent_app_restore, resolve_window_change


_WAIT_THRESHOLD_SECONDS = 10.0


def _elapsed_seconds(previous_timestamp_utc: str, current_timestamp_utc: str) -> float | None:
    """Return the non-negative elapsed time between two timezone-aware ISO timestamps."""
    try:
        previous = datetime.fromisoformat(previous_timestamp_utc.replace("Z", "+00:00"))
        current = datetime.fromisoformat(current_timestamp_utc.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if previous.tzinfo is None or current.tzinfo is None:
        return None
    elapsed = (current - previous).total_seconds()
    return elapsed if elapsed >= 0 else None


def _wait_instruction(elapsed_seconds: float) -> str:
    return f"等待 {ceil(elapsed_seconds)} 秒"


def _drop_trailing_agent_restore(
    events: list[RecordedEvent],
    *,
    log_info: Callable[[str], None] | None = None,
) -> list[RecordedEvent]:
    """Omit a final restore of the hub window (common stop-recording artifact)."""
    if not events:
        return events
    last = events[-1]
    change = resolve_window_change(
        last.window_change,
        last.window_snapshot_debug,
        last.cursor_xy,
    )
    if not is_agent_app_restore(change):
        return events
    if log_info is not None:
        log_info(
            f"dropping trailing agent restore event index={last.index} "
            f"title={change.get('title')!r}"
        )
    return events[:-1]


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
    """Analyze all events in a recording session and write hub-script instructions."""
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
        settings = load_settings()
        log_info(
            "analyze_recording_session llm "
            f"backend={settings.llm_backend} model={settings.brain_lm} host={settings.ollama_host}"
        )
        events = coalesce_consecutive_text_inputs(_load_events(run_dir))
        events = _drop_trailing_agent_restore(events, log_info=log_info)
        log_info(f"analyze_recording_session start events={len(events)} run_id={run_id}")

        cached = 0
        skipped = 0
        cancelled = False
        processed = 0
        errors: list[dict[str, Any]] = []
        instructions: list[str] = []
        previous_instruction_event: RecordedEvent | None = None
        total = len(events)

        if on_progress is not None:
            on_progress(0, total)

        for event in events:
            if should_cancel is not None and should_cancel():
                cancelled = True
                log_info("analyze_recording_session cancelled by user")
                break
            log_info(f"processing event {event.index} kind={event.kind}")
            text_resolution: dict[str, Any] | None = None
            event_for_llm = event
            if event.kind == "text_input":
                resolved = await resolve_text_input_text(
                    event,
                    run_dir=run_dir,
                    log_info=log_info,
                )
                text_resolution = {
                    "recorded_text": resolved.get("recorded_text"),
                    "resolved_text": resolved.get("text"),
                    "source": resolved.get("source"),
                    "meaningful": resolved.get("meaningful"),
                    "reason": resolved.get("reason"),
                }
                event_for_llm = event_with_resolved_text(event, resolved)
                vision = resolved.get("vision") or await build_vision_context(
                    event_for_llm,
                    run_dir=run_dir,
                    log_info=log_info,
                )
            else:
                vision = await build_vision_context(
                    event,
                    run_dir=run_dir,
                    log_info=log_info,
                )
            analysis_path = run_dir / "analysis" / f"event_{event.index:03d}.json"
            analysis_path.parent.mkdir(parents=True, exist_ok=True)

            result = await analyze_event_to_cache(
                event_for_llm,
                run_dir=run_dir,
                vision=vision,
                log_info=log_info,
            )
            if result is None:
                skipped += 1
                errors.append({"event_index": event.index, "error": "llm_analysis_failed"})
            else:
                instruction = result["instruction"]
                cached += 1
                elapsed_since_previous: float | None = None
                wait_instruction: str | None = None
                if previous_instruction_event is not None:
                    elapsed_since_previous = _elapsed_seconds(
                        previous_instruction_event.timestamp_utc,
                        event.timestamp_utc,
                    )
                    if (
                        elapsed_since_previous is not None
                        and elapsed_since_previous > _WAIT_THRESHOLD_SECONDS
                    ):
                        wait_instruction = _wait_instruction(elapsed_since_previous)
                        instructions.append(wait_instruction)
                instructions.append(instruction)
                previous_instruction_event = event
                write_json(
                    analysis_path,
                    {
                        "event_index": event.index,
                        "instruction": instruction,
                        **(
                            {"elapsed_since_previous_seconds": elapsed_since_previous}
                            if elapsed_since_previous is not None
                            else {}
                        ),
                        **(
                            {"wait_instruction": wait_instruction}
                            if wait_instruction is not None
                            else {}
                        ),
                        "vision": {
                            "used_vision": vision.get("used_vision"),
                            "candidate_text": vision.get("candidate_text"),
                        },
                        **(
                            {"text_resolution": text_resolution}
                            if text_resolution is not None
                            else {}
                        ),
                        **(
                            {"window_change": event.window_change}
                            if event.window_change is not None
                            else {}
                        ),
                        **(
                            {"window_snapshot_debug": event.window_snapshot_debug}
                            if event.window_snapshot_debug is not None
                            else {}
                        ),
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
        try:
            from src.common.session_html import write_recording_html_from_run

            write_recording_html_from_run(run_dir)
        except Exception as exc:
            log_info(f"analyze_recording_session html write failed: {exc}")
        log_info(
            f"analyze_recording_session done recorded={len(events)} cached={cached} skipped={skipped}"
        )
        return report
    finally:
        reset_run_state_manager()
