from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POINTER_EVENT_KINDS = frozenset(
    {
        "click",
        "double_click",
        "triple_click",
        "right_click",
        "middle_click",
        "scroll",
        "drag",
        "hold",
    }
)


@dataclass
class RecordedEvent:
    index: int
    timestamp_utc: str
    kind: str
    cursor_xy: tuple[int, int] | None = None
    end_xy: tuple[int, int] | None = None
    button: str | None = None
    modifiers: list[str] | None = None
    key: str | None = None
    keys: list[str] | None = None
    text: str | None = None
    scroll_delta: int | None = None
    duration_seconds: float | None = None
    click_count: int | None = None
    screenshot_path: str = ""
    monitor_index: int | None = None
    monitor_offset: tuple[int, int] | None = None
    end_screenshot_path: str = ""
    end_monitor_index: int | None = None
    end_monitor_offset: tuple[int, int] | None = None
    anchor_click_xy: tuple[int, int] | None = None
    # Caret/UIA focus bounds in global screen coords (left, top, right, bottom).
    focus_rect: tuple[int, int, int, int] | None = None
    window_change: dict[str, Any] | None = None
    target_window_title: str | None = None
    window_snapshot_debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.cursor_xy is not None:
            data["cursor_xy"] = list(self.cursor_xy)
        if self.end_xy is not None:
            data["end_xy"] = list(self.end_xy)
        if self.monitor_offset is not None:
            data["monitor_offset"] = list(self.monitor_offset)
        if self.end_monitor_offset is not None:
            data["end_monitor_offset"] = list(self.end_monitor_offset)
        if self.anchor_click_xy is not None:
            data["anchor_click_xy"] = list(self.anchor_click_xy)
        if self.focus_rect is not None:
            data["focus_rect"] = list(self.focus_rect)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RecordedEvent:
        cursor = raw.get("cursor_xy")
        end = raw.get("end_xy")
        offset = raw.get("monitor_offset")
        end_offset = raw.get("end_monitor_offset")
        anchor = raw.get("anchor_click_xy")
        focus = raw.get("focus_rect")
        keys = raw.get("keys")
        modifiers = raw.get("modifiers")
        return cls(
            index=int(raw["index"]),
            timestamp_utc=str(raw["timestamp_utc"]),
            kind=str(raw["kind"]),
            cursor_xy=tuple(cursor) if isinstance(cursor, list) and len(cursor) == 2 else None,
            end_xy=tuple(end) if isinstance(end, list) and len(end) == 2 else None,
            button=raw.get("button"),
            modifiers=list(modifiers) if isinstance(modifiers, list) else None,
            key=raw.get("key"),
            keys=list(keys) if isinstance(keys, list) else None,
            text=raw.get("text"),
            scroll_delta=raw.get("scroll_delta"),
            duration_seconds=(
                float(raw["duration_seconds"])
                if raw.get("duration_seconds") is not None
                else None
            ),
            click_count=(
                int(raw["click_count"])
                if raw.get("click_count") is not None
                else None
            ),
            screenshot_path=str(raw.get("screenshot_path", "")),
            monitor_index=raw.get("monitor_index"),
            monitor_offset=tuple(offset) if isinstance(offset, list) and len(offset) == 2 else None,
            end_screenshot_path=str(raw.get("end_screenshot_path", "")),
            end_monitor_index=raw.get("end_monitor_index"),
            end_monitor_offset=(
                tuple(end_offset) if isinstance(end_offset, list) and len(end_offset) == 2 else None
            ),
            anchor_click_xy=tuple(anchor) if isinstance(anchor, list) and len(anchor) == 2 else None,
            focus_rect=(
                tuple(int(v) for v in focus)
                if isinstance(focus, list) and len(focus) == 4
                else None
            ),
            window_change=raw.get("window_change") if isinstance(raw.get("window_change"), dict) else None,
            target_window_title=raw.get("target_window_title"),
            window_snapshot_debug=(
                raw.get("window_snapshot_debug")
                if isinstance(raw.get("window_snapshot_debug"), dict)
                else None
            ),
        )


@dataclass
class SessionManifest:
    run_id: str
    started_at_utc: str
    stopped_at_utc: str | None = None
    event_count: int = 0
    events: list[str] = field(default_factory=list)
    # Settled UI after the last action, captured before the hub window is restored.
    final_after_screenshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_json_path(run_dir: Path, index: int) -> Path:
    return run_dir / "events" / f"event_{index:03d}.json"


def screenshot_path_for_event(run_dir: Path, index: int) -> Path:
    return run_dir / "screenshots" / f"event_{index:03d}.jpeg"


def screenshot_path_for_event_end(run_dir: Path, index: int) -> Path:
    return run_dir / "screenshots" / f"event_{index:03d}_end.jpeg"


def final_after_screenshot_path(run_dir: Path) -> Path:
    return Path(run_dir) / "screenshots" / "final_after.jpeg"
