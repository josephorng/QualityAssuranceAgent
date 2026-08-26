"""Background YOLO/OCR prefetch while a recording session is active."""

from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path
from typing import Any

from src.common.io_utils import append_text
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent, utc_now_iso

_PREFETCH_KINDS = POINTER_EVENT_KINDS | {"text_input"}
_SENTINEL = object()


class VisionPrefetchWorker:
    """Single-worker queue that prefetches vision for persisted recording events."""

    def __init__(self) -> None:
        self._run_dir: Path | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, run_dir: Path) -> None:
        """Start (or restart) the prefetch worker for ``run_dir``."""
        self.drain_and_stop(timeout=0.1)
        with self._lock:
            self._run_dir = Path(run_dir)
            self._queue = queue.Queue()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="recording-vision-prefetch",
                daemon=True,
            )
            self._thread.start()

    def enqueue(self, event: RecordedEvent) -> None:
        """Queue a persisted event for vision prefetch (no-op if not started)."""
        with self._lock:
            if self._thread is None or self._run_dir is None:
                return
            if event.kind not in _PREFETCH_KINDS:
                return
            self._queue.put(event)

    def drain_and_stop(self, timeout: float = 120.0) -> None:
        """Finish queued jobs (best-effort) and stop the worker thread."""
        with self._lock:
            thread = self._thread
            q = self._queue
            run_dir = self._run_dir
        if thread is None:
            return
        q.put(_SENTINEL)
        thread.join(timeout=timeout)
        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._run_dir = None
        if run_dir is not None and thread.is_alive():
            self._log(run_dir, "vision prefetch drain timed out; abandoning worker")

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                if not isinstance(item, RecordedEvent):
                    continue
                with self._lock:
                    run_dir = self._run_dir
                if run_dir is None:
                    continue
                self._run_one(item, run_dir)
            finally:
                self._queue.task_done()

    def _run_one(self, event: RecordedEvent, run_dir: Path) -> None:
        from src.recorder.orchestrator import prepare_event_vision

        def log_info(text: str) -> None:
            self._log(run_dir, text)

        try:
            log_info(f"vision prefetch start event={event.index} kind={event.kind}")
            asyncio.run(
                prepare_event_vision(
                    event,
                    run_dir=run_dir,
                    log_info=log_info,
                )
            )
            log_info(f"vision prefetch done event={event.index}")
        except Exception as exc:
            log_info(f"vision prefetch failed event={event.index}: {exc}")

    @staticmethod
    def _log(run_dir: Path, text: str) -> None:
        try:
            append_text(run_dir / "record.log", f"{utc_now_iso()} {text}\n")
        except Exception:
            pass
