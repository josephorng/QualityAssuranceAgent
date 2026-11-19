"""Mid-replay multi-action re-record: capture gestures into a temp session, splice over event N."""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pynput import keyboard, mouse

from src.common.io_utils import read_json, write_json
from src.common.run_control import mark_recording_replay_dirty
from src.common.script_helper import invalidate_recording_replay_cache, recording_event_json_paths
from src.recorder.models import (
    RecordedEvent,
    event_json_path,
    hydrate_window_snapshot_debug,
    next_recording_event_index,
    screenshot_path_for_event,
    screenshot_path_for_event_end,
    signals_sidecar_path,
    utc_now_iso,
    write_event_json,
)
from src.recorder.verify_signals import (
    capture_fast_signals,
    capture_step_signals,
    identity_from_click_window,
)
from src.recorder.window_snapshot import (
    ClickWindowInfo,
    diff_snapshots_with_debug,
    resolve_click_window,
    snapshot_top_level_windows,
)

RerecordMode = Literal["replace", "insert_after"]
IgnoreRect = tuple[int, int, int, int]
IgnoreRectProvider = Callable[[], IgnoreRect | list[IgnoreRect] | None]

RERECORDABLE_KINDS = frozenset(
    {
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "hold",
        "drag",
        "scroll",
        "text_input",
        "key_press",
        "hotkey",
    }
)

_DOUBLE_CLICK_INTERVAL_S = 0.35
_DOUBLE_CLICK_MAX_DIST_PX = 8
_WINDOW_AFTER_SETTLE_S = 0.45
_ARM_GRACE_S = 0.35
_TEMP_DIR_NAME = ".rerecord_tmp"


class EmptyRerecordError(RuntimeError):
    """Raised when 完成錄製 is pressed with no captured gestures."""


def event_kind_is_rerecordable(kind: str | None) -> bool:
    """True when mid-replay re-record supports this event kind."""
    return str(kind or "").strip() in RERECORDABLE_KINDS


def _normalize_ignore_rects(
    value: IgnoreRect | list[IgnoreRect] | None,
) -> list[IgnoreRect]:
    if value is None:
        return []
    if isinstance(value, tuple) and len(value) == 4:
        return [value]
    if isinstance(value, list):
        out: list[IgnoreRect] = []
        for item in value:
            if isinstance(item, tuple) and len(item) == 4:
                out.append(item)
        return out
    return []


def _point_in_any_rect(x: int, y: int, rects: list[IgnoreRect]) -> bool:
    for left, top, width, height in rects:
        if left <= x < left + width and top <= y < top + height:
            return True
    return False


def _normalize_button(button: mouse.Button) -> str:
    if button == mouse.Button.right:
        return "right"
    if button == mouse.Button.middle:
        return "middle"
    return "left"


@dataclass
class _PressSample:
    button: str
    cursor_xy: tuple[int, int]
    down_at: float
    timestamp_utc: str
    screenshot_path: str
    monitor_index: int
    monitor_offset: tuple[int, int]
    click_window: ClickWindowInfo | None
    windows_before: tuple[Any, ...]
    signals_before: dict[str, Any]


def purge_rerecord_analysis_artifacts(run_dir: Path, event_index: int) -> None:
    """Delete YOLO / end-shot / signals for ``event_index`` before overwrite.

    Leaves ``events/event_NNN.json``, the primary screenshot, and
    ``analysis/event_NNN.json`` so reanalyze can preserve verification fields
    before rewriting that analysis file. Does not change ``session.json``.
    """
    run_dir = Path(run_dir)
    if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 1:
        raise ValueError("invalid event index")

    for path in (
        screenshot_path_for_event_end(run_dir, event_index),
        signals_sidecar_path(run_dir, event_index, "before"),
        signals_sidecar_path(run_dir, event_index, "after"),
    ):
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass

    yolo_dir = run_dir / "yolo_ocr"
    if yolo_dir.is_dir():
        prefix = f"event_{event_index:03d}"
        for path in yolo_dir.glob(f"{prefix}*"):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


def _capture_before_shot(
    run_dir: Path,
    event_index: int,
    cursor_xy: tuple[int, int],
) -> tuple[str, int, tuple[int, int]]:
    from src.recorder.capture import _capture_screenshot_at_point

    dest = screenshot_path_for_event(run_dir, event_index)
    return _capture_screenshot_at_point(cursor_xy[0], cursor_xy[1], dest)


def _click_window_payload(
    click_window: ClickWindowInfo | None,
    monitor_offset: tuple[int, int],
) -> dict[str, Any] | None:
    if click_window is None:
        return None
    return click_window.to_local_payload(monitor_offset)


def overwrite_pointer_event(
    run_dir: Path,
    event_index: int,
    *,
    kind: str,
    cursor_xy: tuple[int, int],
    button: str,
    timestamp_utc: str,
    screenshot_path: str,
    monitor_index: int,
    monitor_offset: tuple[int, int],
    click_window: ClickWindowInfo | None,
    windows_before: tuple[Any, ...] | list[Any],
    signals_before: dict[str, Any],
    click_count: int | None = None,
) -> RecordedEvent:
    """Overwrite ``events/event_NNN.json`` with a newly captured pointer gesture."""
    run_dir = Path(run_dir)
    if kind not in RERECORDABLE_KINDS:
        raise ValueError(f"unsupported kind: {kind}")
    if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 1:
        raise ValueError("invalid event index")
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    (run_dir / "screenshots").mkdir(parents=True, exist_ok=True)

    if click_count is None:
        if kind == "double_click":
            click_count = 2
        elif kind == "click":
            click_count = 1
        else:
            click_count = None

    windows_after: list[Any] = []
    signals_after: dict[str, Any] = {}
    time.sleep(_WINDOW_AFTER_SETTLE_S)
    try:
        windows_after = snapshot_top_level_windows()
    except Exception:
        windows_after = []
    try:
        signals_after = capture_step_signals(
            cursor_xy=cursor_xy,
            kind=kind,
            uia_reader=None,
        )
    except Exception:
        signals_after = dict(capture_fast_signals() or {})

    before_signals = dict(signals_before or {})
    point_before = identity_from_click_window(click_window)
    if point_before is not None:
        before_signals["point_window"] = point_before

    change: dict[str, Any] | None = None
    title: str | None = None
    debug: dict[str, Any] = {
        "signals_before": before_signals,
        "signals_after": signals_after,
        "signal_kind": kind,
        "rerecord": True,
    }
    if windows_before:
        try:
            result = diff_snapshots_with_debug(
                list(windows_before),
                windows_after,
                click_xy=cursor_xy,
            )
            debug.update(dict(result.debug))
            debug["signals_before"] = before_signals
            debug["signals_after"] = signals_after
            debug["signal_kind"] = kind
            debug["rerecord"] = True
            if result.change is not None:
                change = result.change.to_dict()
                title = result.change.title or None
        except Exception:
            pass

    try:
        rel_shot = str(
            Path(screenshot_path).resolve().relative_to(run_dir.resolve())
        ).replace("\\", "/")
    except ValueError:
        rel_shot = f"screenshots/event_{event_index:03d}.jpeg"

    event = RecordedEvent(
        index=event_index,
        timestamp_utc=timestamp_utc,
        kind=kind,
        cursor_xy=cursor_xy,
        button=button,
        click_count=click_count,
        screenshot_path=rel_shot,
        monitor_index=monitor_index,
        monitor_offset=monitor_offset,
        click_window=_click_window_payload(click_window, monitor_offset),
        window_change=change,
        target_window_title=title,
        window_snapshot_debug=debug,
    )
    end_shot = screenshot_path_for_event_end(run_dir, event_index)
    if end_shot.is_file():
        try:
            end_shot.unlink()
        except OSError:
            pass
    write_event_json(run_dir, event.to_dict())
    return event


def _rebuild_report_lists(run_dir: Path) -> None:
    from src.common.script_helper import collect_recording_instructions

    instructions, outcomes = collect_recording_instructions(run_dir)
    report_path = run_dir / "report.json"
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        report = {}
    report["instructions"] = instructions
    report["expected_outcomes"] = outcomes
    report["cached"] = len(instructions)
    write_json(report_path, report)


def _session_event_rel_paths(run_dir: Path) -> list[str]:
    paths = recording_event_json_paths(run_dir)
    out: list[str] = []
    for path in paths:
        try:
            out.append(path.relative_to(run_dir).as_posix())
        except ValueError:
            out.append(path.as_posix())
    return out


def _rewrite_session_events(run_dir: Path, rel_paths: list[str]) -> None:
    session_path = run_dir / "session.json"
    session = read_json(session_path, {})
    if not isinstance(session, dict):
        session = {}
    session["events"] = list(rel_paths)
    session["event_count"] = len(rel_paths)
    write_json(session_path, session)


def _copy_media_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)


def _copy_temp_event_into_run(
    temp_dir: Path,
    run_dir: Path,
    *,
    src_index: int,
    dst_index: int,
) -> RecordedEvent:
    """Copy one temp event (+ screenshots / signals / yolo) into ``run_dir`` as ``dst_index``."""
    raw = read_json(event_json_path(temp_dir, src_index), None)
    if not isinstance(raw, dict):
        raise ValueError(f"temp event not found: {src_index}")
    event = RecordedEvent.from_dict(raw)
    event.index = dst_index

    debug = hydrate_window_snapshot_debug(
        temp_dir, src_index, event.window_snapshot_debug
    )
    if isinstance(debug, dict):
        debug = dict(debug)
        debug["rerecord"] = True
        event.window_snapshot_debug = debug

    # Screenshots
    src_shot = screenshot_path_for_event(temp_dir, src_index)
    if not src_shot.is_file() and event.screenshot_path:
        candidate = Path(event.screenshot_path)
        if not candidate.is_absolute():
            candidate = temp_dir / candidate
        if candidate.is_file():
            src_shot = candidate
    dst_shot = screenshot_path_for_event(run_dir, dst_index)
    if src_shot.is_file():
        _copy_media_file(src_shot, dst_shot)
        event.screenshot_path = f"screenshots/event_{dst_index:03d}.jpeg"
    else:
        event.screenshot_path = event.screenshot_path or ""

    src_end = screenshot_path_for_event_end(temp_dir, src_index)
    if not src_end.is_file() and event.end_screenshot_path:
        candidate = Path(event.end_screenshot_path)
        if not candidate.is_absolute():
            candidate = temp_dir / candidate
        if candidate.is_file():
            src_end = candidate
    if src_end.is_file():
        dst_end = screenshot_path_for_event_end(run_dir, dst_index)
        _copy_media_file(src_end, dst_end)
        event.end_screenshot_path = f"screenshots/event_{dst_index:03d}_end.jpeg"
    else:
        event.end_screenshot_path = ""

    # YOLO / OCR caches
    src_yolo = temp_dir / "yolo_ocr"
    dst_yolo = run_dir / "yolo_ocr"
    if src_yolo.is_dir():
        dst_yolo.mkdir(parents=True, exist_ok=True)
        prefix = f"event_{src_index:03d}"
        for path in src_yolo.glob(f"{prefix}*"):
            if not path.is_file():
                continue
            suffix = path.name[len(prefix) :]
            _copy_media_file(path, dst_yolo / f"event_{dst_index:03d}{suffix}")

    write_event_json(run_dir, event.to_dict())
    return event


def _delete_event_files(run_dir: Path, event_index: int) -> None:
    from src.common.runs_report_server import _delete_recording_event_files

    raw = read_json(event_json_path(run_dir, event_index), None)
    if not isinstance(raw, dict):
        raw = {"index": event_index}
    _delete_recording_event_files(run_dir, event_index, raw)


def _coalesce_temp_events(temp_dir: Path, log_info: Callable[[str], None] | None) -> None:
    from src.recorder.coalesce import (
        coalesce_consecutive_same_location_clicks,
        coalesce_consecutive_scrolls,
        coalesce_consecutive_text_inputs,
        coalesce_text_inputs_bridged_by_edit_keys,
        reclassify_flyout_pick_drags_as_clicks,
        reclassify_negligible_drags_as_clicks,
        retarget_ime_candidate_end_screenshots,
    )
    from src.recorder.orchestrator import _load_events, _persist_coalesced_events

    def _log(text: str) -> None:
        if log_info is not None:
            log_info(text)

    events = coalesce_consecutive_scrolls(
        coalesce_consecutive_same_location_clicks(
            retarget_ime_candidate_end_screenshots(
                coalesce_consecutive_text_inputs(
                    coalesce_text_inputs_bridged_by_edit_keys(
                        reclassify_flyout_pick_drags_as_clicks(
                            reclassify_negligible_drags_as_clicks(
                                _load_events(temp_dir)
                            )
                        )
                    )
                )
            )
        )
    )
    _persist_coalesced_events(temp_dir, events, log_info=_log)


def splice_rerecord_events(
    run_dir: Path,
    anchor_event_index: int,
    temp_dir: Path,
    *,
    mode: RerecordMode = "replace",
    log_info: Callable[[str], None] | None = None,
) -> list[int]:
    """Splice temp capture events at ``anchor_event_index`` in session order.

    ``mode="replace"`` replaces the anchor event with the capture (1→N).
    ``mode="insert_after"`` keeps the anchor and inserts the capture after it.

    Returns the new event indices in capture order. Raises ``EmptyRerecordError``
    when the temp session has no events (recording left untouched).
    """
    run_dir = Path(run_dir)
    temp_dir = Path(temp_dir)
    if mode not in ("replace", "insert_after"):
        raise ValueError(f"unsupported splice mode: {mode}")
    if not isinstance(anchor_event_index, int) or isinstance(anchor_event_index, bool):
        raise ValueError("invalid event index")
    if anchor_event_index < 1:
        raise ValueError("invalid event index")

    def _log(text: str) -> None:
        if log_info is not None:
            log_info(text)

    _coalesce_temp_events(temp_dir, log_info)
    from src.recorder.orchestrator import _load_events

    temp_events = _load_events(temp_dir)
    if not temp_events:
        raise EmptyRerecordError("沒有錄到任何動作，請再試一次或按取消。")

    rel_paths = _session_event_rel_paths(run_dir)
    old_rel = f"events/event_{anchor_event_index:03d}.json"
    try:
        pos = rel_paths.index(old_rel)
    except ValueError:
        # Fall back to matching by loaded index order.
        ordered = recording_event_json_paths(run_dir)
        pos = None
        rebuilt: list[str] = []
        for i, path in enumerate(ordered):
            try:
                rebuilt.append(path.relative_to(run_dir).as_posix())
            except ValueError:
                rebuilt.append(path.as_posix())
            raw = read_json(path, None)
            if isinstance(raw, dict) and raw.get("index") == anchor_event_index:
                pos = i
        if pos is None:
            raise ValueError(f"event not found in session: {anchor_event_index}")
        rel_paths = rebuilt

    next_index = next_recording_event_index(run_dir)
    new_indices: list[int] = []
    new_rels: list[str] = []
    for offset, src in enumerate(temp_events):
        dst_index = next_index + offset
        _copy_temp_event_into_run(
            temp_dir,
            run_dir,
            src_index=src.index,
            dst_index=dst_index,
        )
        new_indices.append(dst_index)
        new_rels.append(f"events/event_{dst_index:03d}.json")
        _log(
            f"rerecord({mode}) copied temp event={src.index} -> {dst_index} "
            f"kind={src.kind}"
        )

    updated = list(rel_paths)
    if mode == "insert_after":
        updated[pos + 1 : pos + 1] = new_rels
    else:
        _delete_event_files(run_dir, anchor_event_index)
        updated[pos : pos + 1] = new_rels
    _rewrite_session_events(run_dir, updated)
    return new_indices


async def reanalyze_recording_event(
    run_dir: Path,
    event_index: int,
    *,
    log_info: Callable[[str], None] | None = None,
    preserve_from_analysis: dict[str, Any] | None = None,
    rebuild_caches: bool = True,
) -> dict[str, Any]:
    """Vision + instruction for one event; optionally refresh caches and HTML."""
    from src.recorder.analyze import (
        analyze_event_to_cache,
        use_expected_outcome_enabled,
    )
    from src.recorder.compile_tool_calls import rebuild_recording_instruction_tool_cache
    from src.recorder.coalesce import _elapsed_seconds
    from src.recorder.models import hydrate_window_snapshot_debug
    from src.recorder.orchestrator import (
        _bind_run_state_for_analysis,
        _foregrounds_after_appeared,
        _load_events,
        _next_step_windows_before,
        _write_event_analysis,
        prepare_event_vision,
    )
    from src.recorder.text_choose import (
        apply_text_choice_hard_rules,
        choose_text_input_with_llm,
        needs_llm_text_choice,
    )
    from src.recorder.text_resolve import event_with_resolved_text
    from src.recorder.vision_context import save_text_resolution_cache
    from src.common.session_html import write_recording_html_from_run

    run_dir = Path(run_dir)
    if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 1:
        raise ValueError("invalid event index")

    def _log(text: str) -> None:
        if log_info is not None:
            log_info(text)

    events = _load_events(run_dir)
    event = next((item for item in events if item.index == event_index), None)
    if event is None:
        raise ValueError("event not found")
    if event.kind not in RERECORDABLE_KINDS:
        raise ValueError(f"unsupported kind: {event.kind}")
    event.window_snapshot_debug = hydrate_window_snapshot_debug(
        run_dir,
        event.index,
        event.window_snapshot_debug,
    )

    analysis_path = run_dir / "analysis" / f"event_{event_index:03d}.json"
    previous_analysis = preserve_from_analysis
    if not isinstance(previous_analysis, dict):
        previous_analysis = read_json(analysis_path, None)
    if not isinstance(previous_analysis, dict):
        previous_analysis = {}

    preserved_outcome = previous_analysis.get("expected_outcome")
    if isinstance(preserved_outcome, str):
        preserved_outcome = preserved_outcome.strip() or None
    else:
        preserved_outcome = None
    preserved_use = (
        bool(previous_analysis.get("use_expected_outcome"))
        if "use_expected_outcome" in previous_analysis
        else use_expected_outcome_enabled(previous_analysis)
    )
    _bind_run_state_for_analysis(run_dir, run_dir.name)
    prepared = await prepare_event_vision(event, run_dir=run_dir, log_info=_log)

    if event.kind == "text_input" and isinstance(prepared.text_resolution, dict):
        text_resolution = dict(prepared.text_resolution)
        if needs_llm_text_choice(text_resolution):
            text_resolution = await choose_text_input_with_llm(
                text_resolution,
                log_info=_log,
            )
        else:
            text_resolution = apply_text_choice_hard_rules(text_resolution)
        prepared.text_resolution = text_resolution
        prepared.event_for_llm = event_with_resolved_text(
            prepared.event,
            {"text": text_resolution.get("resolved_text") or ""},
        )
        save_text_resolution_cache(run_dir, prepared.event, text_resolution)

    result = await analyze_event_to_cache(
        prepared.event_for_llm,
        run_dir=run_dir,
        vision=prepared.vision,
        log_info=_log,
    )
    if result is None or not isinstance(result.get("instruction"), str):
        raise RuntimeError("instruction analysis failed")

    positions = {item.index: pos for pos, item in enumerate(events)}
    event_pos = positions[event_index]
    previous_instruction_event = None
    for earlier in reversed(events[:event_pos]):
        previous_instruction_event = earlier
        break
    elapsed_since_previous = None
    if previous_instruction_event is not None:
        elapsed_since_previous = _elapsed_seconds(
            previous_instruction_event.timestamp_utc,
            event.timestamp_utc,
        )

    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    _write_event_analysis(
        analysis_path,
        event=prepared.event_for_llm,
        result=result,
        instruction=str(result["instruction"]),
        vision=prepared.vision,
        expected_outcome=preserved_outcome,
        use_expected_outcome=bool(preserved_use and preserved_outcome),
        elapsed_since_previous=elapsed_since_previous,
        text_resolution=prepared.text_resolution,
        later_foregrounds=_foregrounds_after_appeared(
            events, event_pos, run_dir=run_dir
        ),
        next_windows_before=_next_step_windows_before(events, event_pos),
        time_profile=None,
        run_dir=run_dir,
    )

    disabled = previous_analysis.get("window_verify_disabled")
    if isinstance(disabled, list) and disabled:
        analysis = read_json(analysis_path, {})
        if isinstance(analysis, dict):
            analysis["window_verify_disabled"] = list(disabled)
            write_json(analysis_path, analysis)

    if rebuild_caches:
        rebuild_recording_instruction_tool_cache(run_dir)
        _rebuild_report_lists(run_dir)
        write_recording_html_from_run(run_dir, update_index=False)
        invalidate_recording_replay_cache(run_dir)
        mark_recording_replay_dirty()
    _log(
        f"rerecord reanalyze event={event_index} "
        f"instruction={result['instruction']!r}"
    )
    return {
        "event_index": event_index,
        "instruction": str(result["instruction"]),
        "kind": event.kind,
    }


def reanalyze_recording_event_sync(
    run_dir: Path,
    event_index: int,
    *,
    log_info: Callable[[str], None] | None = None,
    preserve_from_analysis: dict[str, Any] | None = None,
    rebuild_caches: bool = True,
) -> dict[str, Any]:
    """Sync wrapper around :func:`reanalyze_recording_event`."""
    return asyncio.run(
        reanalyze_recording_event(
            run_dir,
            event_index,
            log_info=log_info,
            preserve_from_analysis=preserve_from_analysis,
            rebuild_caches=rebuild_caches,
        )
    )


async def _reanalyze_new_indices(
    run_dir: Path,
    new_indices: list[int],
    *,
    preserved_analysis: dict[str, Any] | None,
    log_info: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    from src.recorder.compile_tool_calls import rebuild_recording_instruction_tool_cache
    from src.common.session_html import write_recording_html_from_run

    results: list[dict[str, Any]] = []
    last = len(new_indices) - 1
    for i, event_index in enumerate(new_indices):
        preserve = preserved_analysis if i == last else {}
        results.append(
            await reanalyze_recording_event(
                run_dir,
                event_index,
                log_info=log_info,
                preserve_from_analysis=preserve,
                rebuild_caches=False,
            )
        )
    rebuild_recording_instruction_tool_cache(run_dir)
    _rebuild_report_lists(run_dir)
    write_recording_html_from_run(run_dir, update_index=False)
    invalidate_recording_replay_cache(run_dir)
    mark_recording_replay_dirty()
    return results


def replace_and_reanalyze_pointer_event(
    run_dir: Path,
    event_index: int,
    *,
    kind: str,
    cursor_xy: tuple[int, int],
    button: str,
    timestamp_utc: str,
    screenshot_path: str,
    monitor_index: int,
    monitor_offset: tuple[int, int],
    click_window: ClickWindowInfo | None,
    windows_before: tuple[Any, ...] | list[Any],
    signals_before: dict[str, Any],
    click_count: int | None = None,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Overwrite event artifacts, purge stale analysis, then reanalyze."""
    purge_rerecord_analysis_artifacts(run_dir, event_index)
    overwrite_pointer_event(
        run_dir,
        event_index,
        kind=kind,
        cursor_xy=cursor_xy,
        button=button,
        timestamp_utc=timestamp_utc,
        screenshot_path=screenshot_path,
        monitor_index=monitor_index,
        monitor_offset=monitor_offset,
        click_window=click_window,
        windows_before=windows_before,
        signals_before=signals_before,
        click_count=click_count,
    )
    return reanalyze_recording_event_sync(
        run_dir, event_index, log_info=log_info
    )


def splice_and_reanalyze_rerecord_capture(
    run_dir: Path,
    anchor_event_index: int,
    temp_dir: Path,
    *,
    mode: RerecordMode = "replace",
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Splice temp capture at ``anchor_event_index`` and analyze the new events.

    Replace mode preserves verification fields from the replaced analysis onto
    the last new event. Insert-after starts fresh analysis for every new event.
    """
    run_dir = Path(run_dir)
    if mode not in ("replace", "insert_after"):
        raise ValueError(f"unsupported splice mode: {mode}")
    preserved: dict[str, Any] = {}
    if mode == "replace":
        analysis_path = run_dir / "analysis" / f"event_{anchor_event_index:03d}.json"
        raw = read_json(analysis_path, None)
        if isinstance(raw, dict):
            preserved = raw

    new_indices = splice_rerecord_events(
        run_dir,
        anchor_event_index,
        temp_dir,
        mode=mode,
        log_info=log_info,
    )
    results = asyncio.run(
        _reanalyze_new_indices(
            run_dir,
            new_indices,
            preserved_analysis=preserved,
            log_info=log_info,
        )
    )
    return {
        "mode": mode,
        "anchor_event_index": anchor_event_index,
        "replaced_event_index": anchor_event_index if mode == "replace" else None,
        "event_indices": new_indices,
        "instructions": [str(item.get("instruction") or "") for item in results],
        "kinds": [str(item.get("kind") or "") for item in results],
        "results": results,
    }


def _rerecord_temp_dir(run_dir: Path) -> Path:
    return Path(run_dir) / _TEMP_DIR_NAME


def _reset_temp_dir(temp_dir: Path) -> Path:
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "events").mkdir(exist_ok=True)
    (temp_dir / "screenshots").mkdir(exist_ok=True)
    (temp_dir / "yolo_ocr").mkdir(exist_ok=True)
    write_json(
        temp_dir / "session.json",
        {
            "run_id": temp_dir.name,
            "started_at_utc": utc_now_iso(),
            "event_count": 0,
            "events": [],
        },
    )
    return temp_dir


class StepRerecordSession:
    """Arm a temp RecordingSession until finish/cancel; splice at one event."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._armed = False
        self._finishing = False
        self._run_dir: Path | None = None
        self._event_index: int | None = None
        self._mode: RerecordMode = "replace"
        self._temp_dir: Path | None = None
        self._ignore_rect_provider: IgnoreRectProvider | None = None
        self._on_done: Callable[[dict[str, Any]], None] | None = None
        self._on_error: Callable[[BaseException], None] | None = None
        self._on_cancelled: Callable[[], None] | None = None
        self._capture = None
        self._esc_listener: keyboard.Listener | None = None
        self._armed_at = 0.0

    def is_armed(self) -> bool:
        with self._lock:
            return self._armed and not self._finishing

    def is_active(self) -> bool:
        """True while capturing or while finish/analyze is in flight."""
        with self._lock:
            return self._armed or self._finishing

    def capture_mode(self) -> RerecordMode:
        with self._lock:
            return self._mode

    def captured_event_count(self) -> int:
        capture = None
        with self._lock:
            capture = self._capture
        if capture is None:
            return 0
        try:
            return int(capture.event_count())
        except Exception:
            return 0

    def arm(
        self,
        *,
        run_dir: Path,
        event_index: int,
        mode: RerecordMode = "replace",
        ignore_rect_provider: IgnoreRectProvider | None = None,
        on_uia_status: Callable[[dict[str, Any]], None] | None = None,
        on_done: Callable[[dict[str, Any]], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
    ) -> None:
        """Start multi-gesture capture into a temp folder under ``run_dir``."""
        from src.recorder.capture import RecordingSession

        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise RuntimeError(f"Recording folder not found: {run_dir}")
        if mode not in ("replace", "insert_after"):
            raise ValueError(f"unsupported splice mode: {mode}")
        if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 1:
            raise ValueError("invalid event index")
        with self._lock:
            if self._armed or self._finishing:
                raise RuntimeError("Re-record is already armed")
            self._armed = True
            self._finishing = False
            self._run_dir = run_dir
            self._event_index = event_index
            self._mode = mode
            self._ignore_rect_provider = ignore_rect_provider
            self._on_done = on_done
            self._on_error = on_error
            self._on_cancelled = on_cancelled
            self._armed_at = time.monotonic()

        temp_dir = _reset_temp_dir(_rerecord_temp_dir(run_dir))
        capture = RecordingSession(runs_root=temp_dir.parent)
        if on_uia_status is not None:
            capture.set_on_uia_status(on_uia_status)
        try:
            capture.start(
                ignore_rect_provider=ignore_rect_provider,
                existing_run_dir=temp_dir,
            )
        except Exception:
            with self._lock:
                self._armed = False
                self._run_dir = None
                self._event_index = None
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        esc_listener = keyboard.Listener(on_press=self._on_esc_press)
        with self._lock:
            self._temp_dir = temp_dir
            self._capture = capture
            self._esc_listener = esc_listener
        esc_listener.start()

    def finish(self) -> None:
        """Stop capture, splice into the live recording, and invoke ``on_done``."""
        with self._lock:
            if not self._armed or self._finishing:
                return
            self._finishing = True
            self._armed = False
            run_dir = self._run_dir
            event_index = self._event_index
            mode = self._mode
            temp_dir = self._temp_dir
            capture = self._capture
            on_done = self._on_done
            on_error = self._on_error
            esc_listener = self._esc_listener
            self._esc_listener = None
            self._capture = None

        self._stop_esc_listener(esc_listener)
        if capture is not None:
            try:
                capture.begin_stop()
            except Exception:
                pass
            try:
                capture.finalize_stop()
            except Exception:
                pass

        if run_dir is None or event_index is None or temp_dir is None:
            self._clear_finish_state()
            return

        def worker() -> None:
            try:
                result = splice_and_reanalyze_rerecord_capture(
                    run_dir,
                    event_index,
                    temp_dir,
                    mode=mode,
                )
            except Exception as exc:
                self._cleanup_temp(temp_dir)
                self._clear_finish_state()
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        pass
                return
            self._cleanup_temp(temp_dir)
            self._clear_finish_state()
            if on_done is not None:
                try:
                    on_done(result)
                except Exception:
                    pass

        threading.Thread(
            target=worker,
            name=f"rerecord-splice-{mode}-{event_index}",
            daemon=True,
        ).start()

    def cancel(self) -> None:
        """Disarm without writing. Invokes ``on_cancelled`` when it was armed."""
        callback: Callable[[], None] | None = None
        with self._lock:
            was_armed = self._armed and not self._finishing
            callback = self._on_cancelled if was_armed else None
            self._armed = False
            finishing = self._finishing
            capture = self._capture
            temp_dir = self._temp_dir
            esc_listener = self._esc_listener
            self._esc_listener = None
            self._capture = None
            self._temp_dir = None
            self._run_dir = None
            self._event_index = None
            if not finishing:
                self._finishing = False

        self._stop_esc_listener(esc_listener)
        if was_armed and capture is not None:
            try:
                capture.begin_stop()
            except Exception:
                pass
            try:
                capture.finalize_stop()
            except Exception:
                pass
        if was_armed:
            self._cleanup_temp(temp_dir)
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _clear_finish_state(self) -> None:
        with self._lock:
            self._finishing = False
            self._armed = False
            self._run_dir = None
            self._event_index = None
            self._mode = "replace"
            self._temp_dir = None
            self._capture = None
            self._on_done = None
            self._on_error = None
            self._on_cancelled = None

    def _cleanup_temp(self, temp_dir: Path | None) -> None:
        if temp_dir is None:
            return
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    def _stop_esc_listener(self, listener: keyboard.Listener | None) -> None:
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:
            pass

    def _on_esc_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.esc:
            self.cancel()
