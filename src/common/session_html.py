from __future__ import annotations

import ast
import csv
import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

_HTML_NAME = "session_steps.html"
_STEP_HEADING_PREFIX = "步驟 "

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
.step {
  background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
  padding: 1.25rem 1.5rem; margin: 0 0 1.25rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.step h2 { font-size: 1.15rem; margin: 0 0 .75rem; }
.instruction {
  font-size: 1rem; font-weight: 600; color: #1f2328;
  border-left: 3px solid #0969da; background: #ddf4ff;
  padding: .5rem .75rem; border-radius: 0 6px 6px 0; margin: 0 0 1rem;
}
.meta { margin: 0 0 1rem; }
.meta dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; margin: 0; }
.meta dt { color: #57606a; font-weight: 600; }
.meta dd { margin: 0; }
.badge { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.badge.ok { background: #dafbe1; color: #116329; }
.badge.fail { background: #ffebe9; color: #cf222e; }
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


def session_html_path(run_root: Path) -> Path:
    return run_root / _HTML_NAME


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


def _extract_instruction(args: Any) -> str:
    """Return the human-readable ``instruction`` from args, or empty when absent."""
    if isinstance(args, dict):
        value = args.get("instruction")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _render_action_html(
    *,
    run_root: Path,
    step_number: int,
    action: str,
    args: dict[str, Any] | Any,
    ok: bool,
    message: str,
    timestamp: datetime | str | None,
    before: Path | None,
    after: Path | None,
) -> str:
    status_class = "ok" if ok else "fail"
    heading = escape(f"{_STEP_HEADING_PREFIX}{step_number}：{action}")
    time_text = escape(_timestamp_text(timestamp)) or "—"
    message_text = escape(message or "（無）")

    instruction = _extract_instruction(args)
    instruction_html = (
        f'<p class="instruction">{escape(instruction)}</p>' if instruction else ""
    )

    shots = _render_shot_html("動作前截圖", before, run_root) + _render_shot_html(
        "動作後截圖", after, run_root
    )

    return (
        f'<section class="step">'
        f"<h2>{heading}</h2>"
        f"{instruction_html}"
        f'<div class="meta"><dl>'
        f"<dt>狀態</dt><dd><span class=\"badge {status_class}\">{escape(_status_text(ok))}</span></dd>"
        f"<dt>時間</dt><dd>{time_text}</dd>"
        f"<dt>訊息</dt><dd>{message_text}</dd>"
        f"</dl></div>"
        f"{_render_args_html(args)}"
        f'<div class="shots">{shots}</div>'
        f"</section>"
    )


def write_session_html_from_run(run_root: Path) -> Path:
    """Build ``session_steps.html`` from ``hand.csv`` in a single pass (O(n)).

    Screenshots are referenced relatively (e.g. ``eye/<file>.png``) so the report stays tiny; keep
    the run folder together when sharing. Safe to call repeatedly and for rebuilding old runs, and
    handles both headered ``hand.csv`` files and legacy header-less ones.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    steps_html: list[str] = []
    hand_csv = _hand_csv_path(run_root)
    if hand_csv.is_file() and hand_csv.stat().st_size > 0:
        with hand_csv.open(newline="", encoding="utf-8") as handle:
            first_line = handle.readline()
            handle.seek(0)
            has_header = first_line.split(",", 1)[0].strip() == "timestamp"
            reader = (
                csv.DictReader(handle)
                if has_header
                else csv.DictReader(handle, fieldnames=_HAND_CSV_FIELDS)
            )
            for step_number, row in enumerate(reader, start=1):
                before = _resolve_run_screenshot(
                    row.get("screenshot_before_path") or row.get("screenshot_name"),
                    run_root,
                )
                after = _resolve_run_screenshot(row.get("screenshot_after_path"), run_root)
                steps_html.append(
                    _render_action_html(
                        run_root=run_root,
                        step_number=step_number,
                        action=row.get("action") or "",
                        args=_coerce_args(row.get("args")),
                        ok=_parse_csv_bool(row.get("ok")),
                        message=row.get("message") or "",
                        timestamp=row.get("timestamp") or None,
                        before=before,
                        after=after,
                    )
                )

    title = escape(f"工作階段步驟紀錄：{run_root.name}")
    body = "\n".join(steps_html)
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        f"<h1>{title}</h1>\n"
        '<p class="intro">即時手部動作紀錄。每一節為一次工具執行。</p>\n'
        f"{body}\n"
        "</body>\n</html>\n"
    )

    path = session_html_path(run_root)
    path.write_text(html, encoding="utf-8")
    return path
