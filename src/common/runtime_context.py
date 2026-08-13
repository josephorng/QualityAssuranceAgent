from __future__ import annotations

import os
from pathlib import Path


RUN_ROOT_ENV = "CUA_RUN_ROOT"
RUN_ID_ENV = "CUA_RUN_ID"
SCRIPT_PATH_ENV = "CUA_SCRIPT_PATH"
SCRIPT_LINES_ENV = "CUA_SCRIPT_LINES_JSON"
SCRIPT_OUTCOMES_ENV = "CUA_SCRIPT_OUTCOMES_JSON"
RUNTIME_COMMAND_MODE_ENV = "CUA_RUNTIME_COMMAND_MODE"
SMART_MODE_ENV = "CUA_SMART_MODE"
SMART_GOAL_ENV = "CUA_SMART_GOAL"
USE_TOOL_CACHE_ENV = "CUA_USE_TOOL_CACHE"


def is_runtime_command_mode() -> bool:
    raw = os.getenv(RUNTIME_COMMAND_MODE_ENV, "").strip().lower()
    return raw in ("1", "true", "yes")


def is_smart_mode() -> bool:
    raw = os.getenv(SMART_MODE_ENV, "").strip().lower()
    return raw in ("1", "true", "yes")


def get_smart_goal() -> str:
    return (os.getenv(SMART_GOAL_ENV) or "").strip()


def use_tool_cache_enabled() -> bool:
    # Smart mode plans dynamic instructions; instruction-tool cache is not applicable.
    if is_smart_mode():
        return False
    raw = os.getenv(USE_TOOL_CACHE_ENV, "").strip().lower()
    return raw in ("1", "true", "yes")


def set_runtime_env(run_root: Path, run_id: str) -> None:
    os.environ[RUN_ROOT_ENV] = str(run_root)
    os.environ[RUN_ID_ENV] = run_id


def get_runtime_env() -> tuple[Path, str]:
    run_root = os.getenv(RUN_ROOT_ENV)
    run_id = os.getenv(RUN_ID_ENV, "")
    if not run_root:
        # Test/dev fallback so single services can boot directly.
        from src.common.settings import resolve_runs_dir

        fallback_root = resolve_runs_dir() / "default_run"
        fallback_root.mkdir(parents=True, exist_ok=True)
        return fallback_root, run_id or "default_run"
    return Path(run_root), run_id or "default_run"
