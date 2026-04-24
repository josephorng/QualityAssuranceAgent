from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.common.run_state import get_run_state_manager

StepPath = str

_DONE_RESULTS = {"done"}
_PENDING_RESULTS = {"pending"}
_NON_ACTIONABLE_RESULTS = {"done", "pending", "failed", "interrupted", "splitted"}


def _manager():
    """Return the process-wide run-state manager."""
    return get_run_state_manager()


def _normalize_result(value: Any) -> str:
    """Normalize step result values for case-insensitive state checks."""
    return str(value or "").strip().lower()


def _default_step(step: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize an input step payload into the runtime step schema."""
    src = step or {}
    out = {
        "image": str(src.get("image", "")),
        "goal": str(src.get("goal", "")),
        "instruction": str(src.get("instruction", "")),
        "result": str(src.get("result", "")),
    }
    if "tool" in src and src.get("tool") not in (None, ""):
        out["tool"] = str(src.get("tool"))
    if "arguments" in src and src.get("arguments") not in (None, ""):
        out["arguments"] = src.get("arguments")
    if "steps" in src:
        raw_children = src.get("steps")
        if isinstance(raw_children, list):
            out["steps"] = [_default_step(child) for child in raw_children if isinstance(child, dict)]
    return out


def _path_parts(path: StepPath) -> list[int]:
    """Convert dot-path notation like '1.2.0' into integer indices."""
    if not path:
        return []
    return [int(part) for part in path.split(".")]


def _locate_step(tree: dict[str, Any], path: StepPath) -> dict[str, Any]:
    """Resolve a step node from the task tree by dot-path."""
    node = tree
    for index in _path_parts(path):
        children = node.get("steps", [])
        if not isinstance(children, list) or index < 0 or index >= len(children):
            raise IndexError(f"invalid step path: {path}")
        child = children[index]
        if not isinstance(child, dict):
            raise ValueError(f"invalid step node at path: {path}")
        node = child
    return node


def _locate_parent(tree: dict[str, Any], path: StepPath) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Resolve parent node and sibling list for a non-root step path."""
    parts = _path_parts(path)
    if not parts:
        raise ValueError("root path has no parent")
    parent = tree
    for index in parts[:-1]:
        children = parent.get("steps", [])
        if not isinstance(children, list) or index < 0 or index >= len(children):
            raise IndexError(f"invalid step path: {path}")
        candidate = children[index]
        if not isinstance(candidate, dict):
            raise ValueError(f"invalid step node at path: {path}")
        parent = candidate
    siblings = parent.get("steps", [])
    if not isinstance(siblings, list):
        raise ValueError(f"invalid siblings at path: {path}")
    return parent, siblings, parts[-1]


def _is_leaf(step: dict[str, Any]) -> bool:
    """Return True when a step has no child `steps`."""
    children = step.get("steps", [])
    return not isinstance(children, list) or len(children) == 0


def init_root_task(goal: str, instruction: str = "", image: str = "") -> dict[str, Any]:
    """Initialize and persist a new root task tree in `steps.json`."""
    tree = {
        "image": image,
        "goal": goal,
        "instruction": instruction or goal,
        "result": "splitted",
        "steps": [],
    }
    _manager().write_steps_tree(tree)
    return tree


def read_steps_tree() -> dict[str, Any]:
    """Load the current step tree from run storage."""
    return _manager().read_steps_tree()


def write_steps_tree(tree: dict[str, Any]) -> None:
    """Persist the full step tree to run storage."""
    _manager().write_steps_tree(tree)


def mark_step_result(path: StepPath, result: str, message: str = "") -> dict[str, Any]:
    """Update one step's `result` (and optional message) by path."""
    tree = read_steps_tree()
    node = _locate_step(tree, path)
    node["result"] = result
    if message:
        node["message"] = message
    write_steps_tree(tree)
    return node


def set_step_image(path: StepPath, image: str) -> dict[str, Any]:
    """Set one step's `image` field by path."""
    tree = read_steps_tree()
    node = _locate_step(tree, path)
    node["image"] = str(image)
    write_steps_tree(tree)
    return node


def divide_step(path: StepPath, new_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace a step with child steps and mark the parent as `splitted`."""
    tree = read_steps_tree()
    node = _locate_step(tree, path)
    normalized_children = [_default_step(step) for step in new_steps]
    node["result"] = "splitted"
    node["steps"] = normalized_children
    write_steps_tree(tree)
    first_child_path = f"{path}.0" if normalized_children else ""
    return {
        "status": "ok",
        "path": path,
        "child_count": len(normalized_children),
        "first_child_path": first_child_path,
    }


def create_new_steps(target_path: StepPath, new_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Insert one or more sibling steps after `target_path` (or append at root)."""
    tree = read_steps_tree()
    if target_path == "":
        siblings = tree.get("steps", [])
        if not isinstance(siblings, list):
            siblings = []
            tree["steps"] = siblings
        insert_at = len(siblings)
        parent_prefix = ""
    else:
        _, siblings, index = _locate_parent(tree, target_path)
        insert_at = index + 1
        parent_prefix = ".".join(target_path.split(".")[:-1])
    created_paths: list[str] = []
    for offset, step in enumerate(new_steps):
        normalized = _default_step(step)
        siblings.insert(insert_at + offset, normalized)
        created_index = insert_at + offset
        created_paths.append(f"{parent_prefix}.{created_index}" if parent_prefix else str(created_index))
    write_steps_tree(tree)
    return {"status": "ok", "created_paths": created_paths, "count": len(created_paths)}


def _dfs_paths(node: dict[str, Any], prefix: str = "") -> list[str]:
    """Collect all leaf step paths in depth-first order."""
    paths: list[str] = []
    children = node.get("steps", [])
    if not isinstance(children, list) or len(children) == 0:
        if prefix != "":
            paths.append(prefix)
        return paths
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
        paths.extend(_dfs_paths(child, next_prefix))
    return paths


def get_pending_step_path() -> str | None:
    """Return the first leaf step path currently marked as pending."""
    tree = read_steps_tree()
    for path in _dfs_paths(tree):
        node = _locate_step(tree, path)
        if _normalize_result(node.get("result")) in _PENDING_RESULTS:
            return path
    return None


def get_next_actionable_step_path() -> str | None:
    """Return the next leaf step path that is not in a terminal/non-actionable state."""
    tree = read_steps_tree()
    for path in _dfs_paths(tree):
        node = _locate_step(tree, path)
        result = _normalize_result(node.get("result"))
        if result in _NON_ACTIONABLE_RESULTS:
            continue
        if _is_leaf(node):
            return path
    return None


def get_step(path: StepPath) -> dict[str, Any]:
    """Return a deep copy of one step node by path."""
    tree = read_steps_tree()
    return deepcopy(_locate_step(tree, path))


def check_task() -> dict[str, Any]:
    """Summarize completion state across all leaf steps."""
    tree = read_steps_tree()
    leaves = _dfs_paths(tree)
    if not leaves:
        return {"complete": False, "reason": "no_steps"}
    pending = 0
    undone = 0
    for path in leaves:
        node = _locate_step(tree, path)
        result = _normalize_result(node.get("result"))
        if result in _PENDING_RESULTS:
            pending += 1
        elif result not in _DONE_RESULTS:
            undone += 1
    return {
        "complete": pending == 0 and undone == 0,
        "pending_count": pending,
        "undone_count": undone,
        "leaf_count": len(leaves),
    }
