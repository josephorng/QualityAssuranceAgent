#!/usr/bin/env python3
"""Post-success release helper for nuitka.bat.

After a successful RELEASE build:
  1. Collect git commits since ``last_logged_commit`` (plus any pending drafts).
  2. Ask Gemma (vLLM) for a Traditional Chinese changelog.
  3. Prepend / rewrite entries in VERSION_LOG.md.
  4. Persist VERSION and version_state.json.

If Gemma is unreachable, write raw commit text with a draft marker and keep
those commits in ``pending`` for automatic retry on the next successful build.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = ROOT / "VERSION"
VERSION_LOG_PATH = ROOT / "VERSION_LOG.md"
STATE_PATH = ROOT / "version_state.json"
AGENT_SETTINGS_PATH = ROOT / "runs" / "agent_settings.json"

DEFAULT_HOST = "http://192.168.4.134:8000"
DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
FIRST_RUN_COMMIT_LIMIT = 30
GEMMA_TIMEOUT_SECONDS = 60

DRAFT_MARKER = "（草稿／原文）"
SECTION_HEADER_RE = re.compile(
    r"^## v(?P<version>\d+\.\d+\.\d+\.\d+) — (?P<date>\d{4}-\d{2}-\d{2})"
    r"(?P<draft>\s*（草稿／原文）)?\s*$"
)


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {err}")
    return result.stdout


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"last_logged_commit": None, "pending": {}}
    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    pending = raw.get("pending") or {}
    if not isinstance(pending, dict):
        pending = {}
    return {
        "last_logged_commit": raw.get("last_logged_commit") or None,
        "pending": {str(k): list(v) for k, v in pending.items() if isinstance(v, list)},
    }


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_llm_config() -> tuple[str, str]:
    host = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST
    model = os.environ.get("BRAIN_LM", "").strip() or DEFAULT_MODEL
    if AGENT_SETTINGS_PATH.is_file():
        try:
            data = json.loads(AGENT_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            host = str(data.get("ollama_host") or host).strip() or host
            model = str(data.get("brain_lm") or model).strip() or model
    return host.rstrip("/"), model


def _collect_new_commits(last_logged_commit: str | None) -> list[str]:
    fmt = "%s%n%b%n---"
    if last_logged_commit:
        # Confirm the SHA still exists; if not, fall back to a bounded window.
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{last_logged_commit}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            raw = _run_git("log", f"{last_logged_commit}..HEAD", f"--pretty=format:{fmt}")
        else:
            print(
                f"warning: last_logged_commit {last_logged_commit} missing; "
                f"using last {FIRST_RUN_COMMIT_LIMIT} commits",
                file=sys.stderr,
            )
            raw = _run_git(
                "log", f"-{FIRST_RUN_COMMIT_LIMIT}", f"--pretty=format:{fmt}"
            )
    else:
        # First release log: bounded window so we do not dump entire history.
        raw = _run_git("log", f"-{FIRST_RUN_COMMIT_LIMIT}", f"--pretty=format:{fmt}")

    commits: list[str] = []
    for block in raw.split("\n---"):
        text = "\n".join(line.rstrip() for line in block.strip().splitlines()).strip()
        if text:
            commits.append(text)
    return commits


def _format_raw_bullets(commits: list[str]) -> str:
    lines: list[str] = []
    for commit in commits:
        subject = commit.splitlines()[0].strip()
        if subject:
            lines.append(f"- {subject}")
    return "\n".join(lines) if lines else "- （無新的提交說明）"


def _ask_gemma(host: str, model: str, commits: list[str]) -> str | None:
    """Return Chinese bullet markdown, or None if unreachable / invalid."""
    commit_block = "\n\n".join(commits)
    prompt = (
        "你是發行說明撰寫助手。請根據下列 git commit 訊息，"
        "產出簡潔的繁體中文使用者向變更說明。\n"
        "規則：\n"
        "1. 只輸出 Markdown 項目符號清單（每行以「- 」開頭）。\n"
        "2. 不要前言、標題、編號或結尾說明。\n"
        "3. 合併語意相近的提交；略過明顯的 merge / 純錯字修正噪音。\n"
        "4. 語氣精簡、面向使用者。\n\n"
        f"Commit 訊息：\n{commit_block}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 1024,
    }
    url = f"{host}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GEMMA_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"warning: Gemma unreachable ({exc})", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"warning: Gemma returned invalid JSON ({exc})", file=sys.stderr)
        return None

    choices = body.get("choices") or []
    if not choices:
        print("warning: Gemma returned no choices", file=sys.stderr)
        return None
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        print("warning: Gemma returned empty content", file=sys.stderr)
        return None
    return _normalize_bullet_markdown(content)


def _normalize_bullet_markdown(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ")):
            lines.append(f"- {stripped[2:].strip()}")
        elif re.match(r"^\d+\.\s+", stripped):
            numbered = re.sub(r"^\d+\.\s+", "", stripped).strip()
            lines.append(f"- {numbered}")
        else:
            # Keep stray content as a bullet rather than dropping it.
            lines.append(f"- {stripped}")
    return "\n".join(lines) if lines else "- （無說明）"


def _parse_version_log_sections(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split VERSION_LOG.md into preamble + ordered section dicts."""
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    preamble_lines: list[str] = []
    current: dict[str, Any] | None = None

    for line in lines:
        match = SECTION_HEADER_RE.match(line)
        if match:
            if current is not None:
                sections.append(current)
            current = {
                "version": match.group("version"),
                "date": match.group("date"),
                "draft": bool(match.group("draft")),
                "body_lines": [],
            }
            continue
        if current is None:
            preamble_lines.append(line)
        else:
            current["body_lines"].append(line)

    if current is not None:
        sections.append(current)

    preamble = "\n".join(preamble_lines).rstrip()
    if preamble and not preamble.endswith("\n"):
        preamble += "\n"
    return preamble, sections


def _render_section(version: str, day: str, body: str, *, draft: bool) -> str:
    marker = f" {DRAFT_MARKER}" if draft else ""
    body = body.rstrip()
    return f"## v{version} — {day}{marker}\n\n{body}\n"


def _write_version_log(
    *,
    new_version: str,
    new_body: str,
    new_draft: bool,
    rewritten: dict[str, str],
) -> None:
    """Prepend the new release and replace any successfully translated drafts."""
    existing = (
        VERSION_LOG_PATH.read_text(encoding="utf-8")
        if VERSION_LOG_PATH.is_file()
        else "# ComputerAgent 版本紀錄\n"
    )
    preamble, sections = _parse_version_log_sections(existing)
    if not preamble.strip():
        preamble = "# ComputerAgent 版本紀錄\n"

    today = date.today().isoformat()
    # Drop an existing section for this version if we are rewriting mid-run
    # (should not normally happen), then rebuild.
    sections = [s for s in sections if s["version"] != new_version]

    updated: list[dict[str, Any]] = []
    for section in sections:
        ver = section["version"]
        if ver in rewritten:
            updated.append(
                {
                    "version": ver,
                    "date": section["date"],
                    "draft": False,
                    "body_lines": rewritten[ver].splitlines(),
                }
            )
        else:
            updated.append(section)

    parts: list[str] = [preamble.rstrip(), ""]
    parts.append(
        _render_section(new_version, today, new_body, draft=new_draft).rstrip()
    )
    for section in updated:
        body = "\n".join(section["body_lines"]).strip() or "- （無說明）"
        parts.append("")
        parts.append(
            _render_section(
                section["version"],
                section["date"],
                body,
                draft=bool(section["draft"]),
            ).rstrip()
        )

    VERSION_LOG_PATH.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _head_sha() -> str:
    return _run_git("rev-parse", "HEAD").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="New release version that was just built (e.g. 1.0.0.1)",
    )
    args = parser.parse_args(argv)
    new_version = args.version.strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", new_version):
        print(f"error: invalid version {new_version!r}", file=sys.stderr)
        return 2

    state = _load_state()
    new_commits = _collect_new_commits(state.get("last_logged_commit"))
    pending: dict[str, list[str]] = dict(state.get("pending") or {})

    host, model = _resolve_llm_config()
    print(f"release_version_log: version={new_version} host={host} model={model}")
    print(f"release_version_log: new_commits={len(new_commits)} pending_versions={len(pending)}")

    rewritten: dict[str, str] = {}
    still_pending: dict[str, list[str]] = {}
    gemma_up = True

    if new_commits:
        translated = _ask_gemma(host, model, new_commits)
        if translated is None:
            new_body = _format_raw_bullets(new_commits)
            new_draft = True
            still_pending[new_version] = new_commits
            gemma_up = False
            print(
                "release_version_log: wrote raw draft for new version; "
                "will retry next build"
            )
        else:
            new_body = translated
            new_draft = False
    else:
        new_body = "- （此版本無新的提交）"
        new_draft = False

    # Retry older draft sections. Stop further calls after the first failure
    # (server is likely still down).
    if gemma_up:
        for ver, commits in pending.items():
            if ver == new_version or not commits:
                continue
            zh = _ask_gemma(host, model, commits)
            if zh is None:
                still_pending[ver] = commits
                for rest_ver, rest_commits in pending.items():
                    if (
                        rest_ver != new_version
                        and rest_ver not in rewritten
                        and rest_ver not in still_pending
                    ):
                        still_pending[rest_ver] = rest_commits
                break
            rewritten[ver] = zh
    else:
        for ver, commits in pending.items():
            if ver != new_version:
                still_pending[ver] = commits

    _write_version_log(
        new_version=new_version,
        new_body=new_body,
        new_draft=new_draft,
        rewritten=rewritten,
    )

    VERSION_PATH.write_text(new_version + "\n", encoding="utf-8")
    _save_state(
        {
            "last_logged_commit": _head_sha(),
            "pending": still_pending,
        }
    )

    if rewritten:
        print(f"release_version_log: translated pending drafts: {', '.join(rewritten)}")
    if still_pending:
        print(
            "release_version_log: still pending: "
            + ", ".join(still_pending.keys())
        )
    print(f"release_version_log: VERSION -> {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
