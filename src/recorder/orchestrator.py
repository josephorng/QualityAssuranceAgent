from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
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
from src.recorder.vision_context import (
    build_vision_context,
    save_text_resolution_cache,
    try_rebuild_text_input_from_cache,
    try_rebuild_vision_from_cache,
)
from src.recorder.window_snapshot import (
    expected_outcome_for_window_change,
    format_window_change_hint,
    is_agent_app_restore,
    resolve_window_change,
)


_WAIT_THRESHOLD_SECONDS = 10.0
_FINAL_AFTER_RELATIVE = "screenshots/final_after.jpeg"
# Cap concurrent Triton YOLO+OCR jobs so the GPU is not flooded.
_DEFAULT_VISION_WORKERS = 4
# Cap concurrent LLM calls so local Ollama/vLLM is not flooded.
_DEFAULT_LLM_WORKERS = 4
_UNSET: Any = object()
# Each event contributes vision + instruction + expected-outcome work units.
_PROGRESS_UNITS_PER_EVENT = 3


def _env_max_workers(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return default


def _vision_max_workers() -> int:
    return _env_max_workers("RECORDING_VISION_WORKERS", _DEFAULT_VISION_WORKERS)


def _llm_max_workers() -> int:
    return _env_max_workers("RECORDING_LLM_WORKERS", _DEFAULT_LLM_WORKERS)


class _AnalysisProgress:
    """Thread-safe multi-phase progress: vision + instruction + outcome per event."""

    def __init__(
        self,
        event_count: int,
        on_progress: Callable[[int, int], None] | None,
    ) -> None:
        self._lock = threading.Lock()
        self._done = 0
        self._total = max(1, event_count * _PROGRESS_UNITS_PER_EVENT)
        self._on_progress = on_progress
        self._emit_unlocked()

    @property
    def total(self) -> int:
        return self._total

    @property
    def done(self) -> int:
        with self._lock:
            return self._done

    def _emit_unlocked(self) -> None:
        if self._on_progress is not None:
            self._on_progress(self._done, self._total)

    def bump(self, units: int = 1) -> None:
        if units <= 0:
            return
        with self._lock:
            self._done = min(self._total, self._done + units)
            self._emit_unlocked()

    def complete(self) -> None:
        with self._lock:
            self._done = self._total
            self._emit_unlocked()


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


@dataclass
class _PreparedEvent:
    event: RecordedEvent
    event_for_llm: RecordedEvent
    vision: dict[str, Any]
    text_resolution: dict[str, Any] | None


async def prepare_event_vision(
    event: RecordedEvent,
    *,
    run_dir: Path,
    log_info: Callable[[str], None],
) -> _PreparedEvent:
    """Resolve typing text (if needed) and run YOLO+OCR for one event.

    Reuses fingerprinted ``yolo_ocr`` / text-resolution caches when present.
    """
    text_resolution: dict[str, Any] | None = None
    event_for_llm = event
    if event.kind == "text_input":
        cached_text = try_rebuild_text_input_from_cache(event, run_dir)
        if cached_text is not None:
            text_resolution, vision = cached_text
            resolved_text = text_resolution.get("resolved_text")
            if resolved_text is None:
                resolved_text = text_resolution.get("recorded_text") or event.text or ""
            event_for_llm = event_with_resolved_text(event, {"text": resolved_text})
            log_info(f"vision cache hit event={event.index} kind=text_input")
            return _PreparedEvent(
                event=event,
                event_for_llm=event_for_llm,
                vision=vision,
                text_resolution=text_resolution,
            )
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
        save_text_resolution_cache(run_dir, event, text_resolution)
        event_for_llm = event_with_resolved_text(event, resolved)
        vision = resolved.get("vision") or await build_vision_context(
            event_for_llm,
            run_dir=run_dir,
            log_info=log_info,
        )
    else:
        cached_vision = try_rebuild_vision_from_cache(event, run_dir)
        if cached_vision is not None:
            log_info(f"vision cache hit event={event.index} kind={event.kind}")
            return _PreparedEvent(
                event=event,
                event_for_llm=event,
                vision=cached_vision,
                text_resolution=None,
            )
        vision = await build_vision_context(
            event,
            run_dir=run_dir,
            log_info=log_info,
        )
    return _PreparedEvent(
        event=event,
        event_for_llm=event_for_llm,
        vision=vision,
        text_resolution=text_resolution,
    )


# Backward-compatible private alias.
_prepare_event_vision = prepare_event_vision


async def _prepare_all_event_visions(
    events: list[RecordedEvent],
    *,
    run_dir: Path,
    log_info: Callable[[str], None],
    should_cancel: Callable[[], bool] | None,
    max_workers: int,
    on_vision_done: Callable[[], None] | None = None,
) -> tuple[list[_PreparedEvent | None], bool]:
    """Run vision for events in parallel (bounded). Returns (results, cancelled)."""
    sem = asyncio.Semaphore(max_workers)
    cancelled = False
    results: list[_PreparedEvent | None] = [None] * len(events)

    async def _one(pos: int, event: RecordedEvent) -> None:
        nonlocal cancelled
        if should_cancel is not None and should_cancel():
            cancelled = True
            return
        async with sem:
            if should_cancel is not None and should_cancel():
                cancelled = True
                return
            log_info(f"vision event {event.index} kind={event.kind}")
            results[pos] = await _prepare_event_vision(
                event,
                run_dir=run_dir,
                log_info=log_info,
            )
            if on_vision_done is not None:
                on_vision_done()

    await asyncio.gather(*[_one(pos, event) for pos, event in enumerate(events)])
    if cancelled:
        log_info("analyze_recording_session cancelled during vision phase")
    return results, cancelled


async def _analyze_all_event_instructions(
    prepared_list: list[_PreparedEvent | None],
    *,
    run_dir: Path,
    log_info: Callable[[str], None],
    should_cancel: Callable[[], bool] | None,
    max_workers: int,
    on_instruction_done: Callable[[], None] | None = None,
) -> tuple[list[Any], bool]:
    """Run instruction LLM for prepared events in parallel (bounded).

    Returns a list aligned with ``prepared_list``: ``_UNSET`` if never started
    (cancel), ``None`` if LLM failed, or the analyze_event_to_cache payload.
    """
    sem = asyncio.Semaphore(max_workers)
    cancelled = False
    results: list[Any] = [_UNSET] * len(prepared_list)

    async def _one(pos: int, prepared: _PreparedEvent) -> None:
        nonlocal cancelled
        if should_cancel is not None and should_cancel():
            cancelled = True
            return
        async with sem:
            if should_cancel is not None and should_cancel():
                cancelled = True
                return
            event = prepared.event
            log_info(f"processing event {event.index} kind={event.kind}")
            results[pos] = await analyze_event_to_cache(
                prepared.event_for_llm,
                run_dir=run_dir,
                vision=prepared.vision,
                log_info=log_info,
            )
            if on_instruction_done is not None:
                on_instruction_done()

    tasks = []
    for pos, prepared in enumerate(prepared_list):
        if prepared is None:
            continue
        tasks.append(_one(pos, prepared))
    if tasks:
        await asyncio.gather(*tasks)
    if cancelled:
        log_info("analyze_recording_session cancelled during instruction LLM phase")
    return results, cancelled


async def _infer_all_expected_outcomes(
    jobs: list[tuple[int, str, str, str, str]],
    *,
    log_info: Callable[[str], None],
    should_cancel: Callable[[], bool] | None,
    max_workers: int,
    on_outcome_done: Callable[[], None] | None = None,
) -> tuple[dict[int, str | None], bool]:
    """Run expected-outcome LLM jobs in parallel. ``jobs`` are (pos, instruction, before, after, hint)."""
    sem = asyncio.Semaphore(max_workers)
    cancelled = False
    outcomes: dict[int, str | None] = {}

    async def _one(
        pos: int,
        instruction: str,
        before_shot: str,
        after_shot: str,
        window_change_hint: str,
    ) -> None:
        nonlocal cancelled
        if should_cancel is not None and should_cancel():
            cancelled = True
            return
        async with sem:
            if should_cancel is not None and should_cancel():
                cancelled = True
                return
            outcomes[pos] = await infer_expected_outcome(
                instruction=instruction,
                before_screenshot=before_shot,
                after_screenshot=after_shot,
                window_change_hint=window_change_hint,
                log_info=log_info,
            )
            if on_outcome_done is not None:
                on_outcome_done()

    if jobs:
        await asyncio.gather(
            *[
                _one(pos, instruction, before_shot, after_shot, hint)
                for pos, instruction, before_shot, after_shot, hint in jobs
            ]
        )
    if cancelled:
        log_info("analyze_recording_session cancelled during expected-outcome LLM phase")
    return outcomes, cancelled


def _write_event_analysis(
    analysis_path: Path,
    *,
    event: RecordedEvent,
    result: dict[str, Any],
    instruction: str,
    vision: dict[str, Any],
    expected_outcome: str | None,
    elapsed_since_previous: float | None,
    wait_instruction: str | None,
    text_resolution: dict[str, Any] | None,
) -> None:
    write_json(
        analysis_path,
        {
            "event_index": event.index,
            "instruction": instruction,
            **(
                {"use_char_target": result["use_char_target"]}
                if "use_char_target" in result
                else {}
            ),
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

    log_lock = threading.Lock()

    def log_info(text: str) -> None:
        with log_lock:
            append_text(log_path, f"{text}\n")

    _bind_run_state_for_analysis(run_dir, run_id)
    try:
        settings = load_settings()
        vision_workers = _vision_max_workers()
        llm_workers = _llm_max_workers()
        log_info(
            "analyze_recording_session llm "
            f"backend={settings.llm_backend} model={settings.brain_lm} host={settings.ollama_host}"
        )
        log_info(
            f"analyze_recording_session vision_workers={vision_workers} "
            f"llm_workers={llm_workers}"
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
        progress = _AnalysisProgress(len(events), on_progress)

        prepared_list, vision_cancelled = await _prepare_all_event_visions(
            events,
            run_dir=run_dir,
            log_info=log_info,
            should_cancel=should_cancel,
            max_workers=vision_workers,
            on_vision_done=progress.bump,
        )
        if vision_cancelled:
            cancelled = True

        instruction_results, llm_cancelled = await _analyze_all_event_instructions(
            prepared_list,
            run_dir=run_dir,
            log_info=log_info,
            should_cancel=should_cancel,
            max_workers=llm_workers,
            on_instruction_done=progress.bump,
        )
        if llm_cancelled:
            cancelled = True

        # Build expected-outcome jobs for events that got an instruction.
        outcome_jobs: list[tuple[int, str, str, str, str]] = []
        deterministic_outcomes: dict[int, str | None] = {}
        for event_pos, event in enumerate(events):
            prepared = prepared_list[event_pos]
            result = instruction_results[event_pos]
            if prepared is None or result is _UNSET:
                # Cancelled before vision/instruction: credit remaining units for this event.
                if prepared is None:
                    progress.bump(3)  # vision + instruction + outcome never ran
                else:
                    progress.bump(2)  # instruction + outcome never ran
                continue
            if result is None:
                # Instruction failed (already bumped); count the outcome slot as done.
                progress.bump(1)
                continue
            instruction = result["instruction"]
            next_event = events[event_pos + 1] if event_pos + 1 < len(events) else None
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
            if expected_outcome is not None:
                deterministic_outcomes[event_pos] = expected_outcome
                progress.bump(1)
            elif before_shot is not None and after_shot is not None:
                outcome_jobs.append(
                    (
                        event_pos,
                        instruction,
                        before_shot,
                        after_shot,
                        format_window_change_hint(window_change),
                    )
                )
            else:
                deterministic_outcomes[event_pos] = None
                progress.bump(1)

        inferred_outcomes, outcome_cancelled = await _infer_all_expected_outcomes(
            outcome_jobs,
            log_info=log_info,
            should_cancel=should_cancel,
            max_workers=llm_workers,
            on_outcome_done=progress.bump,
        )
        if outcome_cancelled:
            cancelled = True
            # Credit outcome slots that never started due to cancel.
            for pos, *_rest in outcome_jobs:
                if pos not in inferred_outcomes and pos not in deterministic_outcomes:
                    progress.bump(1)

        # Ordered assemble: waits, report lists, per-event analysis JSON.
        previous_instruction_event: RecordedEvent | None = None
        for event_pos, event in enumerate(events):
            if should_cancel is not None and should_cancel():
                cancelled = True
                log_info("analyze_recording_session cancelled by user")
                break

            prepared = prepared_list[event_pos]
            if prepared is None:
                cancelled = True
                log_info(
                    f"skipping event {event.index}: vision not prepared (cancelled)"
                )
                break

            result = instruction_results[event_pos]
            if result is _UNSET:
                cancelled = True
                log_info(
                    f"skipping event {event.index}: instruction LLM not started (cancelled)"
                )
                break

            vision = prepared.vision
            text_resolution = prepared.text_resolution
            analysis_path = run_dir / "analysis" / f"event_{event.index:03d}.json"
            analysis_path.parent.mkdir(parents=True, exist_ok=True)

            if result is None:
                skipped += 1
                processed += 1
                errors.append({"event_index": event.index, "error": "llm_analysis_failed"})
                continue

            instruction = result["instruction"]
            cached += 1
            processed += 1
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

            if event_pos in deterministic_outcomes:
                expected_outcome = deterministic_outcomes[event_pos]
            else:
                expected_outcome = inferred_outcomes.get(event_pos)

            instructions.append(instruction)
            expected_outcomes.append(expected_outcome)
            previous_instruction_event = event
            _write_event_analysis(
                analysis_path,
                event=event,
                result=result,
                instruction=instruction,
                vision=vision,
                expected_outcome=expected_outcome,
                elapsed_since_previous=elapsed_since_previous,
                wait_instruction=wait_instruction,
                text_resolution=text_resolution,
            )
            if expected_outcome:
                log_info(
                    f"cached event {event.index}: {instruction} "
                    f"| expected_outcome={expected_outcome}"
                )
            else:
                log_info(f"cached event {event.index}: {instruction}")

        if not cancelled:
            progress.complete()

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
