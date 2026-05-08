from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from cua_mcp import hand_tools
from cua_mcp.coordinate_selection import _resolve_point, _with_clicked_text
from cua_mcp.ui_element_selection import resolve_ui_element_point
from cua_mcp.storage import store_clipboard_text, store_image, store_text, _current_run_paths


def _click(button: str = "left") -> dict[str, Any]:
    return hand_tools.click(button=button)


def _type_text(text: str) -> dict[str, Any]:
    return hand_tools.type_text(text=text)


def _press_key(key: str) -> dict[str, Any]:
    return hand_tools.hotkey(keys=key)


def _hotkey(keys: list[str] | str) -> dict[str, Any]:
    return hand_tools.hotkey(keys=keys)


async def _move(target: str, instruction: str, duration: float = 0.0) -> dict[str, Any]:
    x, y, clicked = await _resolve_point(target=target, instruction=instruction)
    return _with_clicked_text(hand_tools.move(x=x, y=y, duration=duration), clicked)


async def _move_to_ui_element(instruction: str, duration: float = 0.0) -> dict[str, Any]:
    gx, gy, meta = await resolve_ui_element_point(instruction)
    result = hand_tools.move(x=gx, y=gy, duration=duration)
    merged: dict[str, Any] = dict(result)
    merged.update(meta)
    merged["instruction"] = instruction
    return merged


def _wait(seconds: float) -> dict[str, Any]:
    return hand_tools.wait(seconds=seconds)


def _key(key: str) -> dict[str, Any]:
    return hand_tools.key_press(key)


def _right_click() -> dict[str, Any]:
    return hand_tools.click(button="right")


def _middle_click() -> dict[str, Any]:
    return hand_tools.click(button="middle")


def _double_click() -> dict[str, Any]:
    return hand_tools.click(button="left", clicks=2, interval=0.1)


def _triple_click() -> dict[str, Any]:
    return hand_tools.click(button="left", clicks=3, interval=0.1)


def _left_click_drag(x2: int, y2: int, duration: float = 0.5) -> dict[str, Any]:
    pos = hand_tools.cursor_position()
    x1, y1 = pos["x"], pos["y"]
    return hand_tools.drag(x1, y1, x2, y2, duration=duration, button="left")


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
