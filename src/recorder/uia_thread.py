"""Dedicated UI Automation worker thread for recorder COM calls.

FindAll/budgeted walks, ElementFromPoint, and verify UIA reads share one
MTA thread so providers are not hit concurrently from settle, hover, and
pre-click loops (avoids RPC_E_WRONG_THREAD / AV storms).
"""

from __future__ import annotations

import ctypes
import queue
import threading
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")

_UIA_CALL_TIMEOUT_S = 2.5
_COINIT_MULTITHREADED = 0x0


class _UiaJob:
    __slots__ = ("fn", "args", "kwargs", "done", "result", "error")

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class UiaWorker:
    """Lazy singleton thread that runs callables with a shared COM apartment."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[_UiaJob | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._thread_id: int | None = None

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(
                target=self._loop,
                name="cua-uia-worker",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def _loop(self) -> None:
        self._thread_id = threading.get_ident()
        # Pin an MTA for this thread's lifetime so _impl CoInitializeEx sees
        # S_FALSE and must not CoUninitialize between jobs.
        try:
            ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
        except Exception:
            pass
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                job.result = job.fn(*job.args, **job.kwargs)
            except BaseException as exc:  # noqa: BLE001 — deliver to waiter
                job.error = exc
            finally:
                job.done.set()

    def is_worker_thread(self) -> bool:
        return self._thread_id is not None and threading.get_ident() == self._thread_id

    def call(
        self,
        fn: Callable[..., _T],
        /,
        *args: Any,
        timeout_s: float = _UIA_CALL_TIMEOUT_S,
        **kwargs: Any,
    ) -> _T | None:
        """Run ``fn`` on the UIA worker; ``None`` on timeout.

        Re-entrant when already on the worker thread (runs inline). Exceptions
        from ``fn`` propagate to the caller.
        """
        if self.is_worker_thread():
            return fn(*args, **kwargs)
        self._ensure_started()
        job = _UiaJob(fn, args, kwargs)
        self._jobs.put(job)
        if not job.done.wait(timeout=max(0.05, float(timeout_s))):
            return None
        if job.error is not None:
            raise job.error
        return job.result  # type: ignore[return-value]


_WORKER = UiaWorker()


def uia_call(
    fn: Callable[..., _T],
    /,
    *args: Any,
    timeout_s: float = _UIA_CALL_TIMEOUT_S,
    **kwargs: Any,
) -> _T | None:
    """Marshal ``fn`` onto the process-wide UIA worker thread."""
    return _WORKER.call(fn, *args, timeout_s=timeout_s, **kwargs)


def uia_worker_is_current_thread() -> bool:
    return _WORKER.is_worker_thread()
