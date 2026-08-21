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
from src.recorder.analyze import (
    after_screenshot_for_outcome,
    analyze_event_to_cache,
    before_screenshot_for_outcome,
    infer_expected_outcome,
)
from src.recorder.models import RecordedEvent, final_after_screenshot_path
from src.recorder.coalesce import (
    coalesce_consecutive_same_location_clicks,
    coalesce_consecutive_text_inputs,
)
from src.recorder.text_resolve import event_with_resolved_text, resolve_text_input_text
from src.recorder.vision_context import build_vision_context
from src.recorder.window_snapshot import (
    expected_outcome_for_window_change,
    format_window_change_hint,
    is_agent_app_restore,
    resolve_window_change,
)


_WAIT_THRESHOLD_SECONDS = 10.0
_FINAL_AFTER_RELATIVE = "screenshots/final_after.jpeg"


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


def _is_trailing_agent_restore(event: RecordedEvent) -> bool:
    change = resolve_window_change(
        event.window_change,
        event.window_snapshot_debug,
        event.cursor_xy,
    )
    return is_agent_app_restore(change)


def _write_final_after_from_source(
    run_dir: Path,
    source_screenshot: str,
    *,
    log_info: Callable[[str], None] | None = None,
) -> bool:
    """Copy ``source_screenshot`` to the session final-after path and update session.json."""
    src = Path(source_screenshot)
    if not src.is_file():
        return False
    dest = final_after_screenshot_path(run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != src.resolve():
        dest.write_bytes(src.read_bytes())
    session_path = run_dir / "session.json"
    session = read_json(session_path, {})
    if not isinstance(session, dict):
        session = {}
    session["final_after_screenshot"] = _FINAL_AFTER_RELATIVE
    write_json(session_path, session)
    if log_info is not None:
        log_info(f"final after screenshot set from restore event path={dest}")
    return True


def _drop_trailing_agent_restore(
    events: list[RecordedEvent],
    *,
    run_dir: Path | None = None,
    log_info: Callable[[str], None] | None = None,
) -> list[RecordedEvent]:
    """Omit trailing restores of the hub window (common stop-recording artifacts).

    Drops every consecutive trailing agent-restore event. When ``run_dir`` is set,
    also deletes those events' raw files and prefers the earliest dropped restore
    screenshot as the last action's after-frame (UI just before the hub restored).
    """
    if not events:
        return events

    dropped: list[RecordedEvent] = []
    while events and _is_trailing_agent_restore(events[-1]):
        last = events[-1]
        change = resolve_window_change(
            last.window_change,
            last.window_snapshot_debug,
            last.cursor_xy,
        )
        if log_info is not None:
            title = change.get("title") if isinstance(change, dict) else None
            log_info(
                f"dropping trailing agent restore event index={last.index} "
                f"title={title!r}"
            )
        dropped.append(last)
        events = events[:-1]

    if dropped and run_dir is not None:
        # dropped[0] is the last restore; dropped[-1] is the earliest (first restore click).
        # Copy before purge — purge deletes the restore event's screenshot files.
        first_restore = dropped[-1]
        if first_restore.screenshot_path:
            _write_final_after_from_source(
                run_dir,
                first_restore.screenshot_path,
                log_info=log_info,
            )
        for last in dropped:
            try:
                from src.common.runs_report_server import purge_recording_event_from_session

                remaining = purge_recording_event_from_session(run_dir, last.index)
                if log_info is not None:
                    log_info(
                        f"purged trailing agent restore event index={last.index} "
                        f"remaining={remaining}"
                    )
            except ValueError as exc:
                if log_info is not None:
                    log_info(
                        f"purge trailing agent restore event index={last.index} "
                        f"skipped: {exc}"
                    )
    return events


def _resolve_final_after_screenshot(run_dir: Path) -> str | None:
    """Return an existing final-after screenshot path for the last analyzed action."""
    session = read_json(run_dir / "session.json", {})
    if isinstance(session, dict):
        raw = session.get("final_after_screenshot")
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw.strip())
            if not candidate.is_file():
                candidate = run_dir / raw.strip()
            if candidate.is_file():
                return str(candidate)
    fallback = final_after_screenshot_path(run_dir)
    return str(fallback) if fallback.is_file() else None


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
        events = coalesce_consecutive_same_location_clicks(
            coalesce_consecutive_text_inputs(_load_events(run_dir))
        )
        events = _drop_trailing_agent_restore(
            events,
            run_dir=run_dir,
            log_info=log_info,
        )
        try:
            from src.common.runs_report_server import sync_recording_events

            sync_result = sync_recording_events(run_dir, events)
            purged = sync_result.get("purged") or []
            if purged:
                log_info(
                    "persisted coalesced events "
                    f"kept={sync_result.get('kept')} purged={purged}"
                )
            else:
                log_info(
                    f"persisted coalesced events kept={sync_result.get('kept')} purged=[]"
                )
        except Exception as exc:
            log_info(f"persist coalesced events failed: {exc}")
        final_after_screenshot = _resolve_final_after_screenshot(run_dir)
        if final_after_screenshot is not None:
            log_info(f"final after screenshot ready path={final_after_screenshot}")
        log_info(f"analyze_recording_session start events={len(events)} run_id={run_id}")

        cached = 0
        skipped = 0
        cancelled = False
        processed = 0
        errors: list[dict[str, Any]] = []
        instructions: list[str] = []
        expected_outcomes: list[str | None] = []
        previous_instruction_event: RecordedEvent | None = None
        total = len(events)

        if on_progress is not None:
            on_progress(0, total)

        for event_pos, event in enumerate(events):
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
                    "ocr_text": resolved.get("ocr_text"),
                    "ocr_options": list(resolved.get("ocr_options") or []),
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
                        expected_outcomes.append(None)
                next_event = (
                    events[event_pos + 1] if event_pos + 1 < len(events) else None
                )
                before_shot = before_screenshot_for_outcome(event)
                after_shot = after_screenshot_for_outcome(
                    event,
                    next_event,
                    final_after_screenshot=final_after_screenshot,
                )
                window_change = resolve_window_change(
                    event.window_change,
                    event.window_snapshot_debug,
                    event.cursor_xy,
                )
                expected_outcome = expected_outcome_for_window_change(window_change)
                if (
                    expected_outcome is None
                    and before_shot is not None
                    and after_shot is not None
                ):
                    expected_outcome = await infer_expected_outcome(
                        instruction=instruction,
                        before_screenshot=before_shot,
                        after_screenshot=after_shot,
                        window_change_hint=format_window_change_hint(window_change),
                        log_info=log_info,
                    )
                instructions.append(instruction)
                expected_outcomes.append(expected_outcome)
                previous_instruction_event = event
                write_json(
                    analysis_path,
                    {
                        "event_index": event.index,
                        "instruction": instruction,
                        **(
                            {"expected_outcome": expected_outcome}
                            if expected_outcome is not None
                            else {}
                        ),
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
                if expected_outcome:
                    log_info(
                        f"cached event {event.index}: {instruction} "
                        f"| expected_outcome={expected_outcome}"
                    )
                else:
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
            "expected_outcomes": expected_outcomes,
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
