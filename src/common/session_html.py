from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from src.common.session_report import _resolve_script_metadata, _resolve_started_at_utc

_HTML_NAME = "session_steps.html"
_INDEX_HTML_NAME = "index.html"
_RUN_FOLDER_TS_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_(\d{8})_(\d{6})_\d+$")
_QUEUE_SCRIPT_LOG_MARKER = "Queue starting coordinator for "
_HAND_OP_PREFIX = "動作 "
_UNGROUPED_GOAL = "未分類動作"
_FALLBACK_GOAL = "手部動作"

# Column order written by ``src.hand.module`` via ``append_csv_row``; used as a fallback for
# older ``hand.csv`` files that were saved without a header row.
_HAND_CSV_FIELDS = [
    "timestamp",
    "action",
    "args",
    "ok",
    "screenshot_name",
    "screenshot_before_path",
    "screenshot_after_path",
    "message",
]

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Microsoft JhengHei", system-ui, sans-serif;
  margin: 0; padding: 2rem; line-height: 1.5;
  background: #f5f6f8; color: #1f2328;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
.intro { color: #57606a; margin: 0 0 1.5rem; }
.nav { margin: 0 0 1rem; }
.nav a { color: #0969da; text-decoration: none; font-weight: 600; }
.nav a:hover { text-decoration: underline; }
.instruction-group {
  background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
  padding: 0; margin: 0 0 1.25rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.instruction-group > summary {
  display: flex; align-items: center; justify-content: space-between; gap: .75rem;
  cursor: pointer; user-select: none; list-style: none;
  padding: 1rem 1.5rem; font-size: 1.05rem; font-weight: 600;
}
.instruction-group > summary::-webkit-details-marker { display: none; }
.instruction-group > summary::before {
  content: "▶"; flex: 0 0 auto; font-size: .7rem; color: #57606a;
  transition: transform .15s ease;
}
.instruction-group[open] > summary::before { transform: rotate(90deg); }
.instruction-group > summary .instruction-title { flex: 1 1 auto; min-width: 0; }
.instruction-group[open] > summary { border-bottom: 1px solid #d0d7de; }
.hand-ops {
  list-style: disc; margin: 0; padding: 1rem 1.5rem 1.25rem 2.5rem;
}
.hand-op { margin: 0 0 1.25rem; }
.hand-op:last-child { margin-bottom: 0; }
.hand-op-title {
  display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
  font-weight: 600; margin: 0 0 .75rem;
}
.hand-op-action {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}
.meta { margin: 0 0 1rem; }
.meta dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; margin: 0; }
.meta dt { color: #57606a; font-weight: 600; }
.meta dd { margin: 0; }
.badge { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.badge.ok { background: #dafbe1; color: #116329; }
.badge.fail { background: #ffebe9; color: #cf222e; }
.badge.neutral { background: #eaeef2; color: #57606a; }
.args { margin: 0 0 1rem; border: 1px solid #d0d7de; border-radius: 6px; background: #f6f8fa; }
.args > summary { cursor: pointer; padding: .5rem .8rem; font-weight: 600; color: #57606a; user-select: none; }
.args[open] > summary { border-bottom: 1px solid #d0d7de; }
.args-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.args-table th, .args-table td { text-align: left; vertical-align: top; padding: .35rem .8rem; border-top: 1px solid #eaeef2; }
.args-table tr:first-child th, .args-table tr:first-child td { border-top: none; }
.args-table th { width: 30%; color: #57606a; font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-weight: 600; word-break: break-word; }
.args-table td { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; word-break: break-word; white-space: pre-wrap; }
.args-empty { margin: 0; padding: .5rem .8rem; color: #8c959f; }
.shots { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
.shot { min-width: 0; }
.shot .label { font-weight: 600; color: #57606a; margin: 0 0 .4rem; font-size: .9rem; }
.shot a { display: block; }
.shot img { width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 6px; background: #fff; }
.shot .missing { color: #8c959f; font-style: italic; }
@media (max-width: 720px) { .shots { grid-template-columns: 1fr; } }
""".strip()

_INDEX_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Microsoft JhengHei", system-ui, sans-serif;
  margin: 0; padding: 2rem; line-height: 1.5;
  background: #f5f6f8; color: #1f2328;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
.intro { color: #57606a; margin: 0 0 1.5rem; }
.empty { color: #8c959f; font-style: italic; }
.reports {
  width: 100%; border-collapse: collapse; background: #fff;
  border: 1px solid #d0d7de; border-radius: 10px; overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.reports th, .reports td {
  text-align: left; vertical-align: top; padding: .75rem 1rem;
  border-top: 1px solid #d0d7de;
}
.reports thead th {
  border-top: none; background: #f6f8fa; color: #57606a; font-size: .85rem;
}
.reports tbody tr:hover { background: #f6f8fa; }
.reports a { color: #0969da; text-decoration: none; font-weight: 600; word-break: break-all; }
.reports a:hover { text-decoration: underline; }
.badge { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.badge.ok { background: #dafbe1; color: #116329; }
.badge.fail { background: #ffebe9; color: #cf222e; }
.badge.neutral { background: #eaeef2; color: #57606a; }
.mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: .9rem; }
@media (max-width: 720px) {
  .reports, .reports thead, .reports tbody, .reports th, .reports td, .reports tr { display: block; }
  .reports thead { display: none; }
  .reports tr { border-top: 1px solid #d0d7de; padding: .5rem 0; }
  .reports td { border-top: none; padding: .25rem 1rem; }
  .reports td::before {
    content: attr(data-label); display: block; color: #57606a;
    font-size: .75rem; font-weight: 600; margin-bottom: .1rem;
  }
}
""".strip()


def session_html_path(run_root: Path) -> Path:
    return run_root / _HTML_NAME


def runs_index_html_path(runs_root: Path) -> Path:
    return Path(runs_root) / _INDEX_HTML_NAME


def _hand_csv_path(run_root: Path) -> Path:
    return run_root / "hand.csv"


def _coerce_args(value: Any) -> Any:
    """Parse an args value that may be a JSON string, a Python ``repr`` dict, or already structured.

    ``hand.csv`` stores args via ``csv.DictWriter`` (Python repr with single quotes), so plain
    ``json.loads`` fails. Try JSON first, then ``ast.literal_eval``.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _flatten_args_pairs(args: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten nested args into ``(key, value)`` pairs for table rendering."""
    pairs: list[tuple[str, str]] = []

    if isinstance(args, dict):
        for key, value in args.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                pairs.extend(_flatten_args_pairs(value, full_key))
            elif isinstance(value, list):
                if not value:
                    pairs.append((full_key, "（空）"))
                else:
                    for index, item in enumerate(value):
                        item_key = f"{full_key}[{index}]"
                        if isinstance(item, (dict, list)):
                            pairs.extend(_flatten_args_pairs(item, item_key))
                        else:
                            pairs.append((item_key, str(item)))
            else:
                pairs.append((full_key, str(value)))
        return pairs

    if isinstance(args, list):
        for index, item in enumerate(args):
            item_key = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if isinstance(item, (dict, list)):
                pairs.extend(_flatten_args_pairs(item, item_key))
            else:
                pairs.append((item_key, str(item)))
        return pairs

    if args is None:
        return []
    return [("值", str(args))]


def _render_args_html(args: dict[str, Any] | Any) -> str:
    """Render args as a key/value table inside a collapsed ``<details>`` block."""
    pairs = _flatten_args_pairs(args)
    if pairs:
        rows = "".join(
            f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>" for key, value in pairs
        )
        body = f'<table class="args-table"><tbody>{rows}</tbody></table>'
    else:
        body = '<p class="args-empty">（無）</p>'
    return f'<details class="args"><summary>參數</summary>{body}</details>'


def _timestamp_text(timestamp: datetime | str | None) -> str:
    """Format a timestamp for display, e.g. ``2026-07-17 14:30:32 (UTC+08:00)``.

    Parses ISO 8601 input, converts timezone-aware values to local time, and drops microseconds.
    Falls back to the raw string when it cannot be parsed.
    """
    if timestamp is None:
        return ""
    if isinstance(timestamp, datetime):
        dt: datetime | None = timestamp
    else:
        text = str(timestamp).strip()
        if not text:
            return ""
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text

    if dt.tzinfo is not None:
        dt = dt.astimezone()
        offset = dt.strftime("%z")
        formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
        if offset:
            return f"{formatted} (UTC{offset[:3]}:{offset[3:]})"
        return formatted
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _status_text(ok: bool) -> str:
    return "成功" if ok else "失敗"


def _parse_csv_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _resolve_run_screenshot(raw: str | None, run_root: Path) -> Path | None:
    """Resolve a ``hand.csv`` screenshot path to an existing file, regardless of cwd.

    Stored paths may be absolute or relative to the repo root (the app's working directory),
    e.g. ``runs/<run_id>/eye/<file>.png``. Returns ``None`` when no candidate exists.
    """
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    repo_root = run_root.parent.parent
    for resolved in (repo_root / candidate, run_root / "eye" / candidate.name):
        if resolved.is_file():
            return resolved
    return None


def _relative_img_src(screenshot: Path | None, run_root: Path) -> str | None:
    """Return an ``<img src>`` value relative to the HTML file (which lives in ``run_root``)."""
    if screenshot is None:
        return None
    resolved = screenshot.resolve()
    try:
        rel = resolved.relative_to(run_root.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        # Screenshot lives outside the run folder; fall back to an absolute file URI.
        return resolved.as_uri()


def _render_shot_html(label: str, screenshot: Path | None, run_root: Path) -> str:
    src = _relative_img_src(screenshot, run_root)
    if src is None:
        body = '<p class="missing">無螢幕截圖</p>'
    else:
        esc = escape(src, quote=True)
        body = f'<a href="{esc}" target="_blank" rel="noopener"><img src="{esc}" alt="{escape(label)}" loading="lazy"></a>'
    return f'<div class="shot"><p class="label">{escape(label)}</p>{body}</div>'


def _normalize_timestamp_key(timestamp: datetime | str | None) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    text = str(timestamp).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return text


def _load_session_report_data(run_root: Path) -> dict[str, Any]:
    report = _load_run_report(run_root)
    if report is not None:
        return report
    from src.common.session_report import build_session_report

    return build_session_report(run_root, session_end_reason="")


def _iter_hand_csv_rows(run_root: Path) -> list[dict[str, str]]:
    hand_csv = _hand_csv_path(run_root)
    if not hand_csv.is_file() or hand_csv.stat().st_size == 0:
        return []

    rows: list[dict[str, str]] = []
    with hand_csv.open(newline="", encoding="utf-8") as handle:
        first_line = handle.readline()
        handle.seek(0)
        has_header = first_line.split(",", 1)[0].strip() == "timestamp"
        reader = (
            csv.DictReader(handle)
            if has_header
            else csv.DictReader(handle, fieldnames=_HAND_CSV_FIELDS)
        )
        for row in reader:
            rows.append({key: (value or "") for key, value in row.items()})
    return rows


def _hand_operation_from_row(
    *,
    run_root: Path,
    operation_number: int,
    row: dict[str, str],
) -> dict[str, Any]:
    before = _resolve_run_screenshot(
        row.get("screenshot_before_path") or row.get("screenshot_name"),
        run_root,
    )
    after = _resolve_run_screenshot(row.get("screenshot_after_path"), run_root)
    return {
        "operation_number": operation_number,
        "action": row.get("action") or "",
        "args": _coerce_args(row.get("args")),
        "ok": _parse_csv_bool(row.get("ok")),
        "message": row.get("message") or "",
        "timestamp": row.get("timestamp") or None,
        "before": before,
        "after": after,
    }


def _load_instruction_groups(run_root: Path) -> list[dict[str, Any]]:
    hand_rows = _iter_hand_csv_rows(run_root)
    if not hand_rows:
        return []

    report = _load_session_report_data(run_root)
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    tool_results = (
        report.get("tool_results") if isinstance(report.get("tool_results"), list) else []
    )

    order: list[tuple[int, int]] = []
    goals: dict[tuple[int, int], str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        transcript_counter = step.get("transcript_counter")
        script_step_index = step.get("script_step_index")
        if not isinstance(transcript_counter, int) or not isinstance(script_step_index, int):
            continue
        key = (transcript_counter, script_step_index)
        if key not in goals:
            order.append(key)
        goal = step.get("goal")
        if isinstance(goal, str) and goal.strip():
            goals[key] = goal.strip()

    operations = [
        _hand_operation_from_row(run_root=run_root, operation_number=index, row=row)
        for index, row in enumerate(hand_rows, start=1)
    ]

    if not order:
        return [{"goal": _FALLBACK_GOAL, "operations": operations}]

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in order}
    ungrouped: list[dict[str, Any]] = []

    tool_key_by_timestamp: dict[str, tuple[int, int]] = {}
    for tool in tool_results:
        if not isinstance(tool, dict):
            continue
        transcript_counter = tool.get("transcript_counter")
        script_step_index = tool.get("script_step_index")
        timestamp_key = _normalize_timestamp_key(tool.get("timestamp_utc"))
        if (
            isinstance(transcript_counter, int)
            and isinstance(script_step_index, int)
            and timestamp_key is not None
        ):
            tool_key_by_timestamp[timestamp_key] = (transcript_counter, script_step_index)

    for operation in operations:
        timestamp_key = _normalize_timestamp_key(operation.get("timestamp"))
        key = tool_key_by_timestamp.get(timestamp_key or "")
        if key is not None and key in groups:
            groups[key].append(operation)
        else:
            ungrouped.append(operation)

    grouped: list[dict[str, Any]] = []
    for key in order:
        goal = goals.get(key) or f"指令 {key[0] + 1}"
        grouped.append({"goal": goal, "operations": groups[key]})

    if ungrouped:
        grouped.append({"goal": _UNGROUPED_GOAL, "operations": ungrouped})

    return grouped


def _render_hand_operation_html(*, run_root: Path, operation: dict[str, Any]) -> str:
    action = operation["action"]
    args = operation["args"]
    ok = operation["ok"]
    message = operation["message"]
    timestamp = operation["timestamp"]
    before = operation["before"]
    after = operation["after"]
    operation_number = operation["operation_number"]

    status_class = "ok" if ok else "fail"
    status_label = escape(_status_text(ok))
    action_label = escape(f"{_HAND_OP_PREFIX}{operation_number}：{action}")
    time_text = escape(_timestamp_text(timestamp)) or "—"
    message_text = escape(message or "（無）")

    shots = _render_shot_html("動作前截圖", before, run_root) + _render_shot_html(
        "動作後截圖", after, run_root
    )

    return (
        f'<li class="hand-op">'
        f'<div class="hand-op-title">'
        f'<span class="hand-op-action">{action_label}</span>'
        f'<span class="badge {status_class}">{status_label}</span>'
        f"</div>"
        f'<div class="meta"><dl>'
        f"<dt>狀態</dt><dd><span class=\"badge {status_class}\">{status_label}</span></dd>"
        f"<dt>時間</dt><dd>{time_text}</dd>"
        f"<dt>訊息</dt><dd>{message_text}</dd>"
        f"</dl></div>"
        f"{_render_args_html(args)}"
        f'<div class="shots">{shots}</div>'
        f"</li>"
    )


def _render_instruction_group_html(*, run_root: Path, goal: str, operations: list[dict[str, Any]]) -> str:
    operation_count = len(operations)
    count_label = escape(f"{operation_count} 個動作")
    has_failure = any(not operation.get("ok", False) for operation in operations)
    summary_badge_class = "fail" if has_failure else "neutral"

    if operations:
        items = "".join(
            _render_hand_operation_html(run_root=run_root, operation=operation)
            for operation in operations
        )
        body = f'<ul class="hand-ops">{items}</ul>'
    else:
        body = '<p class="args-empty" style="padding: 1rem 1.5rem;">（無手部動作）</p>'

    return (
        f'<details class="instruction-group">'
        f"<summary>"
        f'<span class="instruction-title">{escape(goal)}</span>'
        f'<span class="badge {summary_badge_class}">{count_label}</span>'
        f"</summary>"
        f"{body}"
        f"</details>"
    )


def _format_duration_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    seconds = max(0.0, float(value))
    if seconds >= 60:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes}m {rem:.0f}s"
    return f"{seconds:.1f}s"


def _load_run_report(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "report.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _reason_badge_class(reason: str, *, has_failures: bool) -> str:
    if has_failures:
        return "fail"
    normalized = reason.strip().lower()
    if normalized in {"completed", "success", "ok"}:
        return "ok"
    if normalized:
        return "neutral"
    return "neutral"


def _resolve_index_script_name(run_root: Path, report: dict[str, Any] | None) -> str:
    if isinstance(report, dict):
        name = report.get("script_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        path = report.get("script_path")
        if isinstance(path, str) and path.strip():
            return Path(path.strip()).name

    meta = _resolve_script_metadata(run_root)
    if meta.get("script_name"):
        return meta["script_name"]

    log_path = run_root / "run.log"
    if log_path.is_file():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if _QUEUE_SCRIPT_LOG_MARKER in line:
                    name = line.split(_QUEUE_SCRIPT_LOG_MARKER, 1)[1].strip()
                    if name:
                        return name
        except OSError:
            pass

    return run_root.name


def _resolve_index_run_datetime(run_root: Path, report: dict[str, Any] | None) -> str:
    if isinstance(report, dict):
        for key in ("started_at_utc", "generated_at_utc"):
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                formatted = _timestamp_text(value)
                if formatted:
                    return formatted
        steps = report.get("steps")
        if isinstance(steps, list):
            started = _resolve_started_at_utc(run_root, steps)
            if started:
                formatted = _timestamp_text(started)
                if formatted:
                    return formatted

    started = _resolve_started_at_utc(run_root, [])
    if started:
        formatted = _timestamp_text(started)
        if formatted:
            return formatted

    match = _RUN_FOLDER_TS_RE.match(run_root.name)
    if match is not None:
        date_part, time_part = match.groups()
        try:
            dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
            formatted = _timestamp_text(dt)
            if formatted:
                return formatted
        except ValueError:
            pass

    return "—"


def _render_index_row(run_root: Path) -> str:
    run_id = run_root.name
    href = escape(f"{run_id}/{_HTML_NAME}", quote=True)
    report = _load_run_report(run_root)
    script_name = escape(_resolve_index_script_name(run_root, report))
    run_time = escape(_resolve_index_run_datetime(run_root, report))
    run_id_title = escape(run_id, quote=True)
    summary = report.get("summary") if isinstance(report, dict) else None
    if not isinstance(summary, dict):
        summary = {}

    reason = ""
    if isinstance(report, dict):
        raw_reason = report.get("session_end_reason")
        if isinstance(raw_reason, str):
            reason = raw_reason.strip()

    step_count = summary.get("step_count")
    tool_count = summary.get("tool_call_count")
    failed_steps = summary.get("failed_step_count")
    failed_tools = summary.get("failed_tool_count")
    duration = _format_duration_seconds(summary.get("total_duration_seconds"))

    failed_step_n = failed_steps if isinstance(failed_steps, int) else 0
    failed_tool_n = failed_tools if isinstance(failed_tools, int) else 0
    has_failures = failed_step_n > 0 or failed_tool_n > 0

    reason_label = escape(reason) if reason else "—"
    reason_class = _reason_badge_class(reason, has_failures=has_failures)
    reason_html = (
        f'<span class="badge {reason_class}">{reason_label}</span>'
        if reason
        else "—"
    )

    def _count_text(value: Any) -> str:
        return escape(str(value)) if isinstance(value, int) else "—"

    fail_text = (
        f"{failed_step_n} / {failed_tool_n}"
        if isinstance(failed_steps, int) or isinstance(failed_tools, int)
        else "—"
    )

    return (
        "<tr>"
        f'<td data-label="執行"><a href="{href}" title="{run_id_title}">{script_name}</a></td>'
        f'<td data-label="時間">{run_time}</td>'
        f'<td data-label="結束原因">{reason_html}</td>'
        f'<td data-label="步驟">{_count_text(step_count)}</td>'
        f'<td data-label="工具">{_count_text(tool_count)}</td>'
        f'<td data-label="失敗（步驟/工具）">{escape(fail_text)}</td>'
        f'<td data-label="耗時">{escape(duration)}</td>'
        "</tr>"
    )


def _iter_report_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    found: list[Path] = []
    for child in runs_root.iterdir():
        if child.is_dir() and (child / _HTML_NAME).is_file():
            found.append(child)
    found.sort(key=lambda path: (path.name, path.stat().st_mtime), reverse=True)
    return found


def write_runs_index_html(runs_root: Path) -> Path:
    """Build ``index.html`` listing every child run that has ``session_steps.html``."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _iter_report_run_dirs(runs_root)
    if run_dirs:
        rows = "".join(_render_index_row(run_dir) for run_dir in run_dirs)
        body = (
            '<table class="reports">'
            "<thead><tr>"
            "<th>執行</th><th>時間</th><th>結束原因</th><th>步驟</th><th>工具</th>"
            "<th>失敗（步驟/工具）</th><th>耗時</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
        )
    else:
        body = '<p class="empty">尚無報告。完成一次執行後，報告會出現在此列表。</p>'

    title = "工作階段報告列表"
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n"
        f"<style>\n{_INDEX_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        f"<h1>{escape(title)}</h1>\n"
        f'<p class="intro">共 {len(run_dirs)} 筆報告。點選執行名稱開啟步驟紀錄。</p>\n'
        f"{body}\n"
        "</body>\n</html>\n"
    )

    path = runs_index_html_path(runs_root)
    path.write_text(html, encoding="utf-8")
    return path


def write_session_html_from_run(run_root: Path) -> Path:
    """Build ``session_steps.html`` from ``hand.csv`` in a single pass (O(n)).

    Screenshots are referenced relatively (e.g. ``eye/<file>.png``) so the report stays tiny; keep
    the run folder together when sharing. Safe to call repeatedly and for rebuilding old runs, and
    handles both headered ``hand.csv`` files and legacy header-less ones.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    instruction_groups = _load_instruction_groups(run_root)
    groups_html = [
        _render_instruction_group_html(
            run_root=run_root,
            goal=group["goal"],
            operations=group["operations"],
        )
        for group in instruction_groups
    ]

    title = escape(f"工作階段步驟紀錄：{run_root.name}")
    body = "\n".join(groups_html)
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        '<p class="nav"><a href="../index.html">← 報告列表</a></p>\n'
        f"<h1>{title}</h1>\n"
        '<p class="intro">依使用者指令分組的手部動作紀錄。點選指令可展開底下的動作列表。</p>\n'
        f"{body}\n"
        "</body>\n</html>\n"
    )

    path = session_html_path(run_root)
    path.write_text(html, encoding="utf-8")
    write_runs_index_html(run_root.parent)
    return path
