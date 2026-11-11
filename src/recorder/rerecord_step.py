"""Mid-replay single-step re-record: capture one pointer gesture and overwrite event N."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pynput import keyboard, mouse

from src.common.io_utils import read_json, write_json
from src.common.run_control import mark_recording_replay_dirty
from src.common.script_helper import invalidate_recording_replay_cache
from src.recorder.models import (
    RecordedEvent,
    event_json_path,
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

IgnoreRect = tuple[int, int, int, int]
IgnoreRectProvider = Callable[[], IgnoreRect | list[IgnoreRect] | None]

RERECORDABLE_KINDS = frozenset({"click", "double_click", "right_click"})

_DOUBLE_CLICK_INTERVAL_S = 0.35
_DOUBLE_CLICK_MAX_DIST_PX = 8
_WINDOW_AFTER_SETTLE_S = 0.45
_ARM_GRACE_S = 0.35


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


async def reanalyze_recording_event(
    run_dir: Path,
    event_index: int,
    *,
    log_info: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Vision + instruction for one overwritten event; refresh caches and HTML."""
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

    # Keep prior verify-disabled selectors when still meaningful.
    disabled = previous_analysis.get("window_verify_disabled")
    if isinstance(disabled, list) and disabled:
        analysis = read_json(analysis_path, {})
        if isinstance(analysis, dict):
            analysis["window_verify_disabled"] = list(disabled)
            write_json(analysis_path, analysis)

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
) -> dict[str, Any]:
    """Sync wrapper around :func:`reanalyze_recording_event`."""
    return asyncio.run(
        reanalyze_recording_event(run_dir, event_index, log_info=log_info)
    )


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


class StepRerecordSession:
    """Arm listeners, capture one click/double/right-click, then callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._armed = False
        self._run_dir: Path | None = None
        self._event_index: int | None = None
        self._ignore_rect_provider: IgnoreRectProvider | None = None
        self._on_done: Callable[[dict[str, Any]], None] | None = None
        self._on_error: Callable[[BaseException], None] | None = None
        self._on_cancelled: Callable[[], None] | None = None
        self._mouse_listener: mouse.Listener | None = None
        self._keyboard_listener: keyboard.Listener | None = None
        self._press: _PressSample | None = None
        self._pending_left: _PressSample | None = None
        self._pending_left_timer: threading.Timer | None = None
        self._armed_at = 0.0
        self._finishing = False

    def is_armed(self) -> bool:
        with self._lock:
            return self._armed

    def arm(
        self,
        *,
        run_dir: Path,
        event_index: int,
        ignore_rect_provider: IgnoreRectProvider | None = None,
        on_done: Callable[[dict[str, Any]], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
    ) -> None:
        """Start listening for one pointer gesture that overwrites ``event_index``."""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise RuntimeError(f"Recording folder not found: {run_dir}")
        if not isinstance(event_index, int) or isinstance(event_index, bool) or event_index < 1:
            raise ValueError("invalid event index")
        with self._lock:
            if self._armed:
                raise RuntimeError("Re-record is already armed")
            self._armed = True
            self._finishing = False
            self._run_dir = run_dir
            self._event_index = event_index
            self._ignore_rect_provider = ignore_rect_provider
            self._on_done = on_done
            self._on_error = on_error
            self._on_cancelled = on_cancelled
            self._press = None
            self._pending_left = None
            self._armed_at = time.monotonic()

        mouse_listener = mouse.Listener(
            on_click=self._on_click,
        )
        keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        with self._lock:
            self._mouse_listener = mouse_listener
            self._keyboard_listener = keyboard_listener
        mouse_listener.start()
        keyboard_listener.start()

    def cancel(self) -> None:
        """Disarm without writing. Invokes ``on_cancelled`` when it was armed."""
        callback: Callable[[], None] | None = None
        with self._lock:
            was_armed = self._armed and not self._finishing
            callback = self._on_cancelled if was_armed else None
            self._armed = False
            self._finishing = False
            self._clear_pending_left_locked()
            mouse_listener = self._mouse_listener
            keyboard_listener = self._keyboard_listener
            self._mouse_listener = None
            self._keyboard_listener = None
            self._press = None
            self._run_dir = None
            self._event_index = None
        self._stop_listeners(mouse_listener, keyboard_listener)
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _clear_pending_left_locked(self) -> None:
        timer = self._pending_left_timer
        self._pending_left_timer = None
        self._pending_left = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _stop_listeners(
        self,
        mouse_listener: mouse.Listener | None,
        keyboard_listener: keyboard.Listener | None,
    ) -> None:
        for listener in (mouse_listener, keyboard_listener):
            if listener is None:
                continue
            try:
                listener.stop()
            except Exception:
                pass

    def _current_ignore_rects(self) -> list[IgnoreRect]:
        provider = self._ignore_rect_provider
        if provider is None:
            return []
        try:
            return _normalize_ignore_rects(provider())
        except Exception:
            return []

    def _should_ignore(self, x: int, y: int) -> bool:
        with self._lock:
            if not self._armed or self._finishing:
                return True
            if time.monotonic() - self._armed_at < _ARM_GRACE_S:
                return True
        return _point_in_any_rect(x, y, self._current_ignore_rects())

    def _on_key_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.esc:
            self.cancel()

    def _on_click(
        self,
        x: int,
        y: int,
        button: mouse.Button,
        pressed: bool,
    ) -> None:
        ix, iy = int(x), int(y)
        name = _normalize_button(button)
        if name == "middle":
            return
        if pressed:
            self._handle_press(ix, iy, name)
        else:
            self._handle_release(ix, iy, name)

    def _handle_press(self, x: int, y: int, button: str) -> None:
        if self._should_ignore(x, y):
            return
        with self._lock:
            if not self._armed or self._finishing:
                return
            run_dir = self._run_dir
            event_index = self._event_index
            if run_dir is None or event_index is None:
                return
            # A second left press while a pending single is waiting → double.
            pending = self._pending_left
            if (
                button == "left"
                and pending is not None
                and time.monotonic() - pending.down_at <= _DOUBLE_CLICK_INTERVAL_S
                and abs(pending.cursor_xy[0] - x) <= _DOUBLE_CLICK_MAX_DIST_PX
                and abs(pending.cursor_xy[1] - y) <= _DOUBLE_CLICK_MAX_DIST_PX
            ):
                self._clear_pending_left_locked()
                sample = pending
                self._finishing = True
                mouse_listener = self._mouse_listener
                keyboard_listener = self._keyboard_listener
                self._mouse_listener = None
                self._keyboard_listener = None
                self._armed = False
            else:
                sample = None
                mouse_listener = None
                keyboard_listener = None

        if sample is not None:
            self._stop_listeners(mouse_listener, keyboard_listener)
            self._finish(
                sample,
                kind="double_click",
                button="left",
                click_count=2,
            )
            return

        try:
            shot_path, monitor_index, monitor_offset = _capture_before_shot(
                run_dir, event_index, (x, y)
            )
        except Exception as exc:
            self._fail(exc)
            return
        try:
            click_window = resolve_click_window(x, y)
        except Exception:
            click_window = None
        try:
            windows_before = tuple(snapshot_top_level_windows())
        except Exception:
            windows_before = ()
        try:
            signals_before = capture_fast_signals()
        except Exception:
            signals_before = {}

        sample = _PressSample(
            button=button,
            cursor_xy=(x, y),
            down_at=time.monotonic(),
            timestamp_utc=utc_now_iso(),
            screenshot_path=shot_path,
            monitor_index=monitor_index,
            monitor_offset=monitor_offset,
            click_window=click_window,
            windows_before=windows_before,
            signals_before=dict(signals_before or {}),
        )
        with self._lock:
            if not self._armed or self._finishing:
                return
            self._press = sample

    def _handle_release(self, x: int, y: int, button: str) -> None:
        with self._lock:
            if not self._armed or self._finishing:
                return
            sample = self._press
            self._press = None
            if sample is None or sample.button != button:
                return
            if button == "right":
                self._finishing = True
                mouse_listener = self._mouse_listener
                keyboard_listener = self._keyboard_listener
                self._mouse_listener = None
                self._keyboard_listener = None
                self._armed = False
            else:
                mouse_listener = None
                keyboard_listener = None
                # Wait for a possible second click before emitting single click.
                self._clear_pending_left_locked()
                self._pending_left = sample
                delay = max(
                    0.0,
                    _DOUBLE_CLICK_INTERVAL_S - (time.monotonic() - sample.down_at),
                )
                timer = threading.Timer(delay, self._emit_pending_left)
                self._pending_left_timer = timer
                timer.daemon = True
                timer.start()
                return

        self._stop_listeners(mouse_listener, keyboard_listener)
        self._finish(sample, kind="right_click", button="right", click_count=None)

    def _emit_pending_left(self) -> None:
        with self._lock:
            sample = self._pending_left
            self._pending_left = None
            self._pending_left_timer = None
            if sample is None or not self._armed or self._finishing:
                return
            self._finishing = True
            mouse_listener = self._mouse_listener
            keyboard_listener = self._keyboard_listener
            self._mouse_listener = None
            self._keyboard_listener = None
            self._armed = False
        self._stop_listeners(mouse_listener, keyboard_listener)
        self._finish(sample, kind="click", button="left", click_count=1)

    def _finish(
        self,
        sample: _PressSample,
        *,
        kind: str,
        button: str,
        click_count: int | None,
    ) -> None:
        with self._lock:
            run_dir = self._run_dir
            event_index = self._event_index
            on_done = self._on_done
            on_error = self._on_error
            self._run_dir = None
            self._event_index = None
        if run_dir is None or event_index is None:
            return

        def worker() -> None:
            try:
                result = replace_and_reanalyze_pointer_event(
                    run_dir,
                    event_index,
                    kind=kind,
                    cursor_xy=sample.cursor_xy,
                    button=button,
                    timestamp_utc=sample.timestamp_utc,
                    screenshot_path=sample.screenshot_path,
                    monitor_index=sample.monitor_index,
                    monitor_offset=sample.monitor_offset,
                    click_window=sample.click_window,
                    windows_before=sample.windows_before,
                    signals_before=sample.signals_before,
                    click_count=click_count,
                )
            except Exception as exc:
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        pass
                return
            if on_done is not None:
                try:
                    on_done(result)
                except Exception:
                    pass

        threading.Thread(
            target=worker,
            name=f"rerecord-event-{event_index}",
            daemon=True,
        ).start()

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            on_error = self._on_error
            self._armed = False
            self._finishing = False
            self._clear_pending_left_locked()
            mouse_listener = self._mouse_listener
            keyboard_listener = self._keyboard_listener
            self._mouse_listener = None
            self._keyboard_listener = None
            self._press = None
            self._run_dir = None
            self._event_index = None
        self._stop_listeners(mouse_listener, keyboard_listener)
        if on_error is not None:
            try:
                on_error(exc)
            except Exception:
                pass
