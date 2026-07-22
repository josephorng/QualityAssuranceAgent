"""Localhost HTTP server for browsing/deleting run reports under ``runs_dir``.

Browsers cannot delete folders from a ``file://`` page, so the hub opens the
reports index via this loopback server and the page POSTs to ``/api/runs/<id>/delete``.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.common.session_html import write_runs_index_html

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_DELETE_PATH_RE = re.compile(r"^/api/runs/([^/]+)/delete/?$")

_server_lock = threading.Lock()
_active_server: RunsReportServer | None = None


class RunsReportServer:
    """Serve ``runs_root`` on ``127.0.0.1`` and accept report-folder deletes."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("runs report server is not running")
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def is_running(self) -> bool:
        return (
            self._httpd is not None
            and self._thread is not None
            and self._thread.is_alive()
        )

    def start(self) -> str:
        if self.is_running():
            return self.base_url

        handler = _make_handler(self.runs_root)
        # Bind loopback only; port 0 picks a free ephemeral port.
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        httpd.daemon_threads = True
        self._httpd = httpd

        thread = threading.Thread(
            target=httpd.serve_forever,
            name="runs-report-server",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return self.base_url

    def stop(self) -> None:
        httpd = self._httpd
        self._httpd = None
        thread = self._thread
        self._thread = None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:
                pass
            try:
                httpd.server_close()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


def ensure_runs_report_server(runs_root: Path) -> RunsReportServer:
    """Return a running server for ``runs_root``, replacing any other active server."""
    global _active_server
    runs_root = Path(runs_root).resolve()
    with _server_lock:
        if (
            _active_server is not None
            and _active_server.runs_root == runs_root
            and _active_server.is_running()
        ):
            return _active_server
        if _active_server is not None:
            _active_server.stop()
            _active_server = None
        server = RunsReportServer(runs_root)
        server.start()
        _active_server = server
        return server


def stop_runs_report_server() -> None:
    global _active_server
    with _server_lock:
        if _active_server is not None:
            _active_server.stop()
            _active_server = None


def delete_run_report_folder(runs_root: Path, run_id: str) -> Path:
    """Delete one direct child run folder under ``runs_root`` and rebuild the index.

    Raises ``ValueError`` for invalid ids / path traversal / missing folders.
    """
    target = resolve_deletable_run_folder(runs_root, run_id)
    shutil.rmtree(target)
    write_runs_index_html(Path(runs_root).resolve())
    return target


def resolve_deletable_run_folder(runs_root: Path, run_id: str) -> Path:
    """Validate ``run_id`` and return the absolute path of a deletable run folder."""
    runs_root = Path(runs_root).resolve()
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run id")
    if Path(run_id).name != run_id:
        raise ValueError("invalid run id")

    target = (runs_root / run_id).resolve()
    try:
        target.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("run folder is outside runs root") from exc
    if target.parent != runs_root:
        raise ValueError("run folder must be a direct child of runs root")
    if not target.is_dir():
        raise ValueError("run folder not found")
    return target


def _make_handler(runs_root: Path) -> type[SimpleHTTPRequestHandler]:
    root = runs_root.resolve()

    class RunsReportHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            # Keep hub console quiet during normal browsing.
            return

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            match = _DELETE_PATH_RE.fullmatch(unquote(parsed.path))
            if match is None:
                self._send_json(404, {"ok": False, "error": "not found"})
                return

            run_id = match.group(1)
            try:
                deleted = delete_run_report_folder(root, run_id)
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            except OSError as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return

            self._send_json(200, {"ok": True, "deleted": deleted.name})

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return RunsReportHandler
