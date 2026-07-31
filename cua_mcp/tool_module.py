from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from cua_mcp import hand_tools
from cua_mcp.select_mouse_target import find_mouse_point, resolve_mouse_point
from cua_mcp.visual_mouse import resolve_visual_mouse_point
from cua_mcp.storage import store_clipboard_text, store_image, store_text, _current_run_paths
from src.common.runtime_context import is_smart_mode


def _with_unified_target_metadata(
    result: dict[str, Any],
    *,
    target_kind: str,
    target_text: str = "",
    target_icons: list[dict[str, Any]] | None = None,
    target_bbox: dict[str, int] | None = None,
) -> dict[str, Any]:
    merged = dict(result)
    merged["target_kind"] = target_kind
    merged["target_text"] = target_text
    merged["target_icons"] = list(target_icons or [])
    if target_bbox is not None:
        merged["target_bbox"] = dict(target_bbox)
    return merged


def _click(button: str = "left", modifiers: list[str] | None = None) -> dict[str, Any]:
    return hand_tools.click(button=button, modifiers=modifiers)


def _type_text(text: str) -> dict[str, Any]:
    return hand_tools.type_text(text=text)


def _press_key(key: str) -> dict[str, Any]:
    return hand_tools.hotkey(keys=key)


def _hotkey(keys: list[str] | str) -> dict[str, Any]:
    return hand_tools.hotkey(keys=keys)


async def _move_mouse(
    instruction: str,
    nearby_objects: list[str] | None = None,
    duration: float = 0.0,
) -> dict[str, Any]:
    gx, gy, meta = await resolve_mouse_point(
        instruction,
        nearby_objects=nearby_objects,
    )
    result = hand_tools.move(x=gx, y=gy, duration=duration)
    merged: dict[str, Any] = dict(result)
    merged.update(meta)
    merged["instruction"] = instruction
    if nearby_objects is not None:
        merged["nearby_objects_arg"] = list(nearby_objects)
    return _with_unified_target_metadata(
        merged,
        target_kind=str(meta.get("target_kind", "mouse_target")),
        target_text=str(meta.get("target_text", "")),
        target_icons=meta.get("target_icons", []),
        target_bbox=meta.get("target_bbox"),
    )


async def _move_mouse_visual(
    instruction: str,
    duration: float = 0.0,
) -> dict[str, Any]:
    """Move to the candidate selected by one multimodal screenshot+OCR LLM pass."""
    gx, gy, meta = await resolve_visual_mouse_point(instruction)
    result = hand_tools.move(x=gx, y=gy, duration=duration)
    merged: dict[str, Any] = dict(result)
    merged.update(meta)
    merged["instruction"] = instruction
    return _with_unified_target_metadata(
        merged,
        target_kind=str(meta.get("target_kind", "visual_mouse_target")),
        target_text=str(meta.get("target_text", "")),
        target_icons=meta.get("target_icons", []),
        target_bbox=meta.get("target_bbox"),
    )


async def _check_object_exists(
    instruction: str,
    nearby_objects: list[str] | None = None,
) -> dict[str, Any]:
    """Report whether a UI target is on screen; does not move or click."""
    found = await find_mouse_point(
        instruction,
        nearby_objects=nearby_objects,
    )
    if found is None:
        result: dict[str, Any] = {
            "exists": False,
            "instruction": instruction,
        }
        if nearby_objects is not None:
            result["nearby_objects_arg"] = list(nearby_objects)
        return result

    gx, gy, meta = found
    merged: dict[str, Any] = {
        "exists": True,
        "instruction": instruction,
        "x": gx,
        "y": gy,
    }
    merged.update(meta)
    if nearby_objects is not None:
        merged["nearby_objects_arg"] = list(nearby_objects)
    return _with_unified_target_metadata(
        merged,
        target_kind=str(meta.get("target_kind", "mouse_target")),
        target_text=str(meta.get("target_text", "")),
        target_icons=meta.get("target_icons", []),
        target_bbox=meta.get("target_bbox"),
    )


def _wait(seconds: float) -> dict[str, Any]:
    return hand_tools.wait(seconds=seconds)


def _key(key: str) -> dict[str, Any]:
    return hand_tools.key_press(key)


def _right_click() -> dict[str, Any]:
    return hand_tools.click(button="right")


def _middle_click() -> dict[str, Any]:
    return hand_tools.click(button="middle")


def _double_click(modifiers: list[str] | None = None) -> dict[str, Any]:
    return hand_tools.click(button="left", clicks=2, interval=0.1, modifiers=modifiers)


def _triple_click() -> dict[str, Any]:
    return hand_tools.click(button="left", clicks=3, interval=0.1)


def _drag_at_points(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float = 0.5,
    button: str = "left",
) -> dict[str, Any]:
    return hand_tools.drag(x1, y1, x2, y2, duration=duration, button=button)


async def _drag(
    start_instruction: str,
    destination_instruction: str,
    start_nearby_objects: list[str] | None = None,
    destination_nearby_objects: list[str] | None = None,
    duration: float = 0.5,
    button: str = "left",
) -> dict[str, Any]:
    if is_smart_mode():
        # Smart mode: one-pass multimodal selection (same path as move_mouse_visual).
        x1, y1, start_meta = await resolve_visual_mouse_point(start_instruction)
        x2, y2, end_meta = await resolve_visual_mouse_point(destination_instruction)
        default_kind = "visual_mouse_target"
    else:
        x1, y1, start_meta = await resolve_mouse_point(
            start_instruction,
            nearby_objects=start_nearby_objects,
        )
        x2, y2, end_meta = await resolve_mouse_point(
            destination_instruction,
            nearby_objects=destination_nearby_objects,
        )
        default_kind = "mouse_target"
    result = _drag_at_points(x1, y1, x2, y2, duration=duration, button=button)
    merged: dict[str, Any] = dict(result)
    merged["start_instruction"] = start_instruction
    merged["destination_instruction"] = destination_instruction
    if start_nearby_objects is not None:
        merged["start_nearby_objects_arg"] = list(start_nearby_objects)
    if destination_nearby_objects is not None:
        merged["destination_nearby_objects_arg"] = list(destination_nearby_objects)
    merged["start_target"] = dict(start_meta)
    merged["destination_target"] = dict(end_meta)
    return _with_unified_target_metadata(
        merged,
        target_kind=str(end_meta.get("target_kind", default_kind)),
        target_text=str(end_meta.get("target_text", "")),
        target_icons=end_meta.get("target_icons", []),
        target_bbox=end_meta.get("target_bbox"),
    )


def _screenshot(path: str = "", instruction: str = "") -> dict[str, Any]:
    p = path.strip() if path else ""
    storage_dir, _ = _current_run_paths()
    if not p:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        p = str(storage_dir / f"screenshot_{stamp}.png")
    else:
        candidate = Path(p)
        if not candidate.is_absolute():
            p = str(storage_dir / candidate.name)
    return hand_tools.screenshot_to_file(p)


def _cursor_position(instruction: str = "") -> dict[str, Any]:
    return hand_tools.cursor_position()


def _left_mouse_down() -> dict[str, Any]:
    return hand_tools.mouse_down(button="left")


def _left_mouse_up() -> dict[str, Any]:
    return hand_tools.mouse_up(button="left")


def _scroll(clicks: int) -> dict[str, Any]:
    return hand_tools.scroll_at(clicks)


def _hold_key(key: str, seconds: float) -> dict[str, Any]:
    return hand_tools.hold_key_down(key, seconds)


def _zoom(scroll_clicks: int) -> dict[str, Any]:
    return hand_tools.zoom_scroll(scroll_clicks)


async def _maximize_windows(window_title_contains: str, instruction: str = "") -> dict[str, Any]:
    return await hand_tools.maximize_windows(
        window_title_contains=window_title_contains,
        instruction=instruction,
    )


async def _close_windows(window_title_contains: str, instruction: str = "") -> dict[str, Any]:
    return await hand_tools.close_windows(
        window_title_contains=window_title_contains,
        instruction=instruction,
    )


async def _minimize_windows(window_title_contains: str, instruction: str = "") -> dict[str, Any]:
    return await hand_tools.minimize_windows(
        window_title_contains=window_title_contains,
        instruction=instruction,
    )


def _store_text(
    text: str,
    title: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return store_text(text=text, title=title, tags=tags)


def _store_clipboard_text(
    title: str = "",
    tags: list[str] | None = None,
    file_name: str = "",
) -> dict[str, Any]:
    return store_clipboard_text(title=title, tags=tags, file_name=file_name)


def _store_image(
    image_path: str,
    summary: str = "",
    alias: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return store_image(image_path=image_path, summary=summary, alias=alias, tags=tags)


def _list_storage_files(
    pattern: str = "*",
    max_results: int = 200,
) -> dict[str, Any]:
    storage_dir, storage_json = _current_run_paths()
    pat = (pattern or "*").strip() or "*"
    limit = int(max_results)
    if limit <= 0:
        raise ValueError("max_results must be a positive integer")

    rows: list[dict[str, Any]] = []
    for p in sorted(storage_dir.iterdir(), key=lambda x: x.name.casefold()):
        if not p.is_file():
            continue
        if not fnmatch(p.name, pat):
            continue
        st = p.stat()
        rows.append(
            {
                "file_name": p.name,
                "stored_path": str(p),
                "size_bytes": int(st.st_size),
                "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
        if len(rows) >= limit:
            break

    return {
        "storage_dir": str(storage_dir),
        "storage_index_path": str(storage_json),
        "pattern": pat,
        "max_results": limit,
        "count": len(rows),
        "files": rows,
    }


def _read_storage_text(
    file_name: str,
    max_chars: int = 20000,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    storage_dir, _ = _current_run_paths()
    base = Path((file_name or "").strip()).name
    if not base:
        raise ValueError("file_name must be a non-empty basename (e.g. 'notes.txt')")

    p = (storage_dir / base).resolve()
    # Ensure the resolved path is still within storage_dir to avoid traversal.
    if storage_dir.resolve() not in p.parents:
        raise ValueError("file_name must resolve under this run's storage directory")
    if not p.exists():
        raise FileNotFoundError(f"storage file not found: {base}")
    if not p.is_file():
        raise ValueError(f"storage path is not a file: {base}")
    if p.suffix.lower() != ".txt":
        raise ValueError("only .txt storage files can be opened with this tool")

    limit = int(max_chars)
    if limit <= 0:
        raise ValueError("max_chars must be a positive integer")

    text = p.read_text(encoding=encoding, errors="replace")
    truncated = len(text) > limit
    out = text[:limit]
    return {
        "file_name": base,
        "stored_path": str(p),
        "encoding": encoding,
        "max_chars": limit,
        "truncated": truncated,
        "content": out,
        "content_chars": len(out),
        "total_chars": len(text),
    }
