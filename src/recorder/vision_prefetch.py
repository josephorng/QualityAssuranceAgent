"""Background YOLO/OCR prefetch while a recording session is active."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from src.common.io_utils import append_text
from src.recorder.models import POINTER_EVENT_KINDS, RecordedEvent, utc_now_iso
from src.recorder.orchestrator import _vision_max_workers, prepare_event_vision

_PREFETCH_KINDS = POINTER_EVENT_KINDS | {"text_input"}


class VisionPrefetchWorker:
    """Bounded thread pool that prefetches vision for persisted recording events."""

    def __init__(self, *, max_workers: int | None = None) -> None:
        self._max_workers = max_workers
        self._run_dir: Path | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future[Any]] = set()
        self._lock = threading.Lock()
        self._accepting = False
        self._drain_progress: Callable[[int, int], None] | None = None
        self._drain_total = 0
        self._drain_done = 0

    def _worker_count(self) -> int:
        if self._max_workers is not None:
            return max(1, self._max_workers)
        return _vision_max_workers()

    def start(self, run_dir: Path) -> None:
        """Start (or restart) the prefetch worker for ``run_dir``."""
        self.drain_and_stop(timeout=0.1)
        with self._lock:
            self._run_dir = Path(run_dir)
            self._futures = set()
            self._accepting = True
            self._executor = ThreadPoolExecutor(
                max_workers=self._worker_count(),
                thread_name_prefix="recording-vision-prefetch",
            )

    def enqueue(self, event: RecordedEvent) -> None:
        """Queue a persisted event for vision prefetch (no-op if not started)."""
        with self._lock:
            if not self._accepting or self._executor is None or self._run_dir is None:
                return
            if event.kind not in _PREFETCH_KINDS:
                return
            executor = self._executor
            run_dir = self._run_dir
            future = executor.submit(self._run_one, event, run_dir)
            self._futures.add(future)
            future.add_done_callback(self._on_future_done)

    def drain_and_stop(
        self,
        timeout: float = 120.0,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Finish queued jobs (best-effort) and stop the worker thread pool."""
        with self._lock:
            executor = self._executor
            run_dir = self._run_dir
            futures = list(self._futures)
            self._accepting = False
            self._drain_progress = on_progress
            self._drain_total = len(futures)
            self._drain_done = 0
            if executor is None:
                if on_progress is not None:
                    on_progress(0, 0)
                self._drain_progress = None
                return
            if on_progress is not None:
                on_progress(0, self._drain_total)

        done, not_done = wait(futures, timeout=timeout)
        executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            if self._executor is executor:
                self._executor = None
                self._run_dir = None
            self._futures.difference_update(done)
            self._futures.difference_update(not_done)
            self._drain_progress = None
            self._drain_total = 0
            self._drain_done = 0
        if run_dir is not None and not_done:
            self._log(
                run_dir,
                f"vision prefetch drain timed out; abandoned {len(not_done)} job(s)",
            )

    def _on_future_done(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)
            progress = self._drain_progress
            if progress is not None and self._drain_total > 0:
                self._drain_done += 1
                done = self._drain_done
                total = self._drain_total
            else:
                progress = None
        if progress is not None:
            progress(done, total)

    def _run_one(self, event: RecordedEvent, run_dir: Path) -> None:
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
