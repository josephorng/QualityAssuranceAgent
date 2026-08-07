"""Directional nearby landmarks: 9-section grid, phrases, and hint parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class Side(str, Enum):
    """Script-side relation: where the click anchor sits relative to the landmark."""

    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    UPPER_LEFT = "upper_left"
    UPPER_RIGHT = "upper_right"
    LOWER_LEFT = "lower_left"
    LOWER_RIGHT = "lower_right"
    INSIDE = "inside"


class LandmarkCell(str, Enum):
    """Where the landmark center sits relative to the anchor bbox edges."""

    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    UPPER_LEFT = "upper_left"
    UPPER_RIGHT = "upper_right"
    LOWER_LEFT = "lower_left"
    LOWER_RIGHT = "lower_right"


# Landmark cell → script side (anchor relative to landmark).
_CELL_TO_SCRIPT_SIDE: dict[LandmarkCell, Side | None] = {
    LandmarkCell.CENTER: None,
    LandmarkCell.RIGHT: Side.LEFT,
    LandmarkCell.LEFT: Side.RIGHT,
    LandmarkCell.BELOW: Side.ABOVE,
    LandmarkCell.ABOVE: Side.BELOW,
    LandmarkCell.UPPER_RIGHT: Side.LOWER_LEFT,
    LandmarkCell.UPPER_LEFT: Side.LOWER_RIGHT,
    LandmarkCell.LOWER_RIGHT: Side.UPPER_LEFT,
    LandmarkCell.LOWER_LEFT: Side.UPPER_RIGHT,
}

_SIDE_TO_ZH: dict[Side, str] = {
    Side.LEFT: "左邊",
    Side.RIGHT: "右邊",
    Side.ABOVE: "上面",
    Side.BELOW: "下面",
    Side.UPPER_LEFT: "左上方",
    Side.UPPER_RIGHT: "右上方",
    Side.LOWER_LEFT: "左下方",
    Side.LOWER_RIGHT: "右下方",
    Side.INSIDE: "裡面",
}

_ZH_TO_SIDE: dict[str, Side] = {zh: side for side, zh in _SIDE_TO_ZH.items()}

_SIDE_TOKEN = "|".join(
    sorted(_ZH_TO_SIDE.keys(), key=len, reverse=True)
)  # longer first: 左上方 before 左邊

_DIRECTED_PHRASE_RE = re.compile(
    rf"^(?:(?P<loc>起點|終點))?在(?P<label>.+)的(?P<side>{_SIDE_TOKEN})$"
)
_DIRECTED_IN_TEXT_RE = re.compile(
    rf"(?:(?P<loc>起點|終點))?在(?P<label>.+?)的(?P<side>{_SIDE_TOKEN})"
)
_NEARBY_HAVE_RE = re.compile(
    r"(?:(?P<loc>起點|終點))?附近有(?P<body>.+)"
)
_PAREN_RE = re.compile(r"（([^）]*)）")


@dataclass(frozen=True)
class NearbyHint:
    """A nearby landmark label with optional script-side constraint."""

    label: str
    side: Side | None = None

    def as_phrase(self) -> str:
        """Serialize for tool args / metadata / picker hints."""
        if self.side is None:
            return self.label
        return format_directed_phrase(self.label, self.side)


def _bbox_xywh_to_edges(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convert ``(x, y, w, h)`` to ``(x1, y1, x2, y2)``."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def landmark_cell_from_anchor_bbox(
    anchor_bbox: tuple[int, int, int, int],
    landmark_cx: int,
    landmark_cy: int,
) -> LandmarkCell:
    """Return which of the nine cells contains ``(landmark_cx, landmark_cy)``.

    ``anchor_bbox`` is ``(x, y, w, h)``. Points on an edge belong to the center
    band for that axis (``x1 <= x <= x2``, ``y1 <= y <= y2``).
    """
    x1, y1, x2, y2 = _bbox_xywh_to_edges(anchor_bbox)
    if landmark_cx < x1:
        col = -1
    elif landmark_cx > x2:
        col = 1
    else:
        col = 0
    if landmark_cy < y1:
        row = -1
    elif landmark_cy > y2:
        row = 1
    else:
        row = 0

    return {
        (-1, -1): LandmarkCell.UPPER_LEFT,
        (0, -1): LandmarkCell.ABOVE,
        (1, -1): LandmarkCell.UPPER_RIGHT,
        (-1, 0): LandmarkCell.LEFT,
        (0, 0): LandmarkCell.CENTER,
        (1, 0): LandmarkCell.RIGHT,
        (-1, 1): LandmarkCell.LOWER_LEFT,
        (0, 1): LandmarkCell.BELOW,
        (1, 1): LandmarkCell.LOWER_RIGHT,
    }[(col, row)]


def side_from_anchor_bbox(
    anchor_bbox: tuple[int, int, int, int],
    landmark_cx: int,
    landmark_cy: int,
) -> Side | None:
    """Script side for an anchor bbox given a landmark center, or None if CENTER."""
    cell = landmark_cell_from_anchor_bbox(anchor_bbox, landmark_cx, landmark_cy)
    return _CELL_TO_SCRIPT_SIDE[cell]


def _point_inside_bbox_xywh(
    x: int,
    y: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    """True when ``(x, y)`` lies inside an axis-aligned ``(x, y, w, h)`` bbox."""
    bx, by, bw, bh = bbox
    return bx <= x < bx + bw and by <= y < by + bh


def anchor_center_xy(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return the integer center of an ``(x, y, w, h)`` bbox."""
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def anchor_satisfies_side(
    anchor_bbox: tuple[int, int, int, int],
    landmark_cx: int,
    landmark_cy: int,
    side: Side,
    *,
    landmark_bbox: tuple[int, int, int, int] | None = None,
) -> bool:
    """True when geometry satisfies ``side`` for this anchor/landmark pair.

    Directional sides use the 9-grid + inversion. ``Side.INSIDE`` requires the
    anchor center to fall inside ``landmark_bbox``.
    """
    if side == Side.INSIDE:
        if landmark_bbox is None:
            return False
        ax, ay = anchor_center_xy(anchor_bbox)
        return _point_inside_bbox_xywh(ax, ay, landmark_bbox)
    return side_from_anchor_bbox(anchor_bbox, landmark_cx, landmark_cy) == side


def side_to_zh(side: Side) -> str:
    return _SIDE_TO_ZH[side]


def side_from_zh(token: str) -> Side | None:
    return _ZH_TO_SIDE.get((token or "").strip())


def format_directed_phrase(label: str, side: Side) -> str:
    """Format ``在{label}的左邊`` (no location prefix)."""
    return f"在{label}的{side_to_zh(side)}"


def parse_nearby_hint_string(raw: str) -> NearbyHint | None:
    """Parse a tool/instruction nearby string into a hint, or None if empty."""
    text = (raw or "").strip()
    if not text:
        return None
    match = _DIRECTED_PHRASE_RE.match(text)
    if match:
        label = (match.group("label") or "").strip()
        side = side_from_zh(match.group("side") or "")
        if label and side is not None:
            return NearbyHint(label=label, side=side)
    return NearbyHint(label=text, side=None)


def _split_undirected_labels(body: str) -> list[str]:
    parts = [p.strip() for p in (body or "").split("、")]
    return [p for p in parts if p]


def _trim_directed_label_fragment(label: str) -> str:
    """Drop leading list fragments before a label, keeping 、 inside 「…」 quotes.

    Directed matches may accidentally absorb a prior undirected clause joined by
    ``、``. Split only on commas outside Chinese quotation marks so icon names
    like ``「搜尋、放大鏡」圖示`` stay intact.
    """
    text = (label or "").strip()
    if "、" not in text:
        return text
    segments: list[str] = []
    buf: list[str] = []
    in_quote = False
    for ch in text:
        if ch == "「":
            in_quote = True
            buf.append(ch)
        elif ch == "」":
            in_quote = False
            buf.append(ch)
        elif ch == "、" and not in_quote:
            segment = "".join(buf).strip()
            if segment:
                segments.append(segment)
            buf = []
        else:
            buf.append(ch)
    trailing = "".join(buf).strip()
    if trailing:
        segments.append(trailing)
    if not segments:
        return text
    return segments[-1]


def _is_nearby_comment_inner(inner: str) -> bool:
    """True when a Chinese parenthetical looks like a nearby-context comment."""
    text = (inner or "").strip()
    if not text:
        return False
    if _DIRECTED_IN_TEXT_RE.search(text):
        return True
    if _NEARBY_HAVE_RE.search(text):
        return True
    return False


def extract_nearby_hints_from_instruction(instruction: str) -> list[NearbyHint]:
    """Deterministically extract nearby hints from parenthetical context comments."""
    by_location = extract_nearby_hints_by_location(instruction)
    return merge_nearby_hints(
        by_location.get("附近"),
        by_location.get("起點"),
        by_location.get("終點"),
    )


def extract_nearby_hints_by_location(
    instruction: str,
) -> dict[str, list[NearbyHint]]:
    """Extract nearby hints keyed by ``附近`` / ``起點`` / ``終點``."""
    text = instruction or ""
    buckets: dict[str, list[NearbyHint]] = {
        "附近": [],
        "起點": [],
        "終點": [],
    }
    seen_by_loc: dict[str, set[str]] = {
        "附近": set(),
        "起點": set(),
        "終點": set(),
    }

    def _add(location: str, hint: NearbyHint) -> None:
        loc = location if location in buckets else "附近"
        if not hint.label or hint.label in seen_by_loc[loc]:
            return
        seen_by_loc[loc].add(hint.label)
        buckets[loc].append(hint)

    for paren in _PAREN_RE.findall(text):
        inner = paren.strip()
        if not inner:
            continue
        directed_found = False
        for match in _DIRECTED_IN_TEXT_RE.finditer(inner):
            label = _trim_directed_label_fragment(match.group("label") or "")
            side = side_from_zh(match.group("side") or "")
            loc = (match.group("loc") or "").strip() or "附近"
            if label and side is not None:
                _add(loc, NearbyHint(label=label, side=side))
                directed_found = True

        for match in _NEARBY_HAVE_RE.finditer(inner):
            loc = (match.group("loc") or "").strip() or "附近"
            for label in _split_undirected_labels(match.group("body") or ""):
                if label in seen_by_loc.get(loc, set()):
                    continue
                if _DIRECTED_IN_TEXT_RE.search(label):
                    continue
                _add(loc, NearbyHint(label=label, side=None))

        if not directed_found and "附近有" not in inner and "在" not in inner:
            continue

    return buckets


def strip_nearby_context_comments(instruction: str) -> str:
    """Remove nearby-context parentheticals; keep the rest of the instruction."""

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1) or ""
        if _is_nearby_comment_inner(inner):
            return ""
        return match.group(0)

    return _PAREN_RE.sub(_replace, instruction or "")


_CLICK_ACTION_SUFFIX_RE = re.compile(r"(，(?:並|用).+。)$")


def _insert_nearby_before_click_suffix(instruction: str, comment: str) -> str:
    """Insert ``comment`` before a trailing click/hold action suffix when present."""
    match = _CLICK_ACTION_SUFFIX_RE.search(instruction)
    if match:
        return instruction[: match.start()] + comment + instruction[match.start() :]
    return instruction + comment


def apply_nearby_landmarks(
    instruction: str,
    hints: list[NearbyHint] | list[str] | list[Any] | None = None,
    *,
    kind: str = "click",
    end_hints: list[NearbyHint] | list[str] | list[Any] | None = None,
) -> str:
    """Strip existing nearby comments and re-apply selected landmarks.

    For ``drag``, ``hints`` are 起點 landmarks and ``end_hints`` are 終點.
    For other pointer kinds, ``hints`` use location ``附近`` and are inserted
    before any trailing ``，並…`` / ``，用…`` click suffix.
    """
    stripped = strip_nearby_context_comments(instruction)
    if kind == "drag":
        result = stripped
        start_comment = format_nearby_context_comment(hints or [], location="起點")
        if start_comment and "拖到" in result:
            drag_at = result.index("拖到")
            result = result[:drag_at] + start_comment + result[drag_at:]
        elif start_comment:
            result = result + start_comment
        end_comment = format_nearby_context_comment(end_hints or [], location="終點")
        if end_comment:
            result = result + end_comment
        return result

    comment = format_nearby_context_comment(hints or [], location="附近")
    if comment is None:
        return stripped
    return _insert_nearby_before_click_suffix(stripped, comment)


def format_nearby_context_comment(
    hints: list[NearbyHint] | list[str],
    *,
    location: str = "附近",
) -> str | None:
    """Format nearby hints as a parenthetical comment.

    ``location`` is ``附近``, ``起點``, or ``終點``.
    """
    normalized = normalize_nearby_hints(hints)
    if not normalized:
        return None

    directed_parts: list[str] = []
    undirected_labels: list[str] = []
    for hint in normalized:
        if hint.side is not None:
            phrase = format_directed_phrase(hint.label, hint.side)
            if location in ("起點", "終點"):
                directed_parts.append(f"{location}{phrase}")
            else:
                directed_parts.append(phrase)
        else:
            undirected_labels.append(hint.label)

    parts = list(directed_parts)
    if undirected_labels:
        if location in ("起點", "終點"):
            prefix = f"{location}附近有"
        else:
            prefix = "附近有"
        parts.append(prefix + "、".join(undirected_labels))

    if not parts:
        return None
    return f"（{'、'.join(parts)}）"


def normalize_nearby_hints(
    items: list[NearbyHint] | list[str] | list[Any] | None,
) -> list[NearbyHint]:
    """Strip, drop empties, and dedupe by label while preserving order.

    When the same label appears twice, keep the earlier entry; if the later
    entry has a side and the earlier does not, upgrade the kept side.
    """
    if not items:
        return []
    out: list[NearbyHint] = []
    index_by_label: dict[str, int] = {}
    for item in items:
        hint: NearbyHint | None
        if isinstance(item, NearbyHint):
            label = (item.label or "").strip()
            if not label:
                continue
            hint = NearbyHint(label=label, side=item.side)
        elif isinstance(item, str):
            hint = parse_nearby_hint_string(item)
        elif isinstance(item, dict):
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            side_raw = item.get("side")
            side: Side | None = None
            if isinstance(side_raw, Side):
                side = side_raw
            elif isinstance(side_raw, str) and side_raw.strip():
                try:
                    side = Side(side_raw.strip())
                except ValueError:
                    side = side_from_zh(side_raw)
            hint = NearbyHint(label=label, side=side)
        else:
            continue
        if hint is None:
            continue
        if hint.label in index_by_label:
            idx = index_by_label[hint.label]
            if out[idx].side is None and hint.side is not None:
                out[idx] = NearbyHint(label=hint.label, side=hint.side)
            continue
        index_by_label[hint.label] = len(out)
        out.append(hint)
    return out


def merge_nearby_hints(*sources: list[NearbyHint] | list[str] | None) -> list[NearbyHint]:
    """Merge nearby hint lists; earlier sources win on label duplicates.

    Prefer retaining a non-None ``side`` when a later duplicate upgrades it only
    if the earlier entry had no side (handled inside ``normalize_nearby_hints``
    per source; across sources earlier wins entirely including side).
    """
    merged: list[NearbyHint] = []
    seen: set[str] = set()
    for source in sources:
        for hint in normalize_nearby_hints(source):
            if hint.label in seen:
                # Later source: upgrade side only if kept entry has none.
                for i, existing in enumerate(merged):
                    if existing.label == hint.label and existing.side is None and hint.side is not None:
                        merged[i] = NearbyHint(label=hint.label, side=hint.side)
                        break
                continue
            seen.add(hint.label)
            merged.append(hint)
    return merged


def nearby_hints_to_phrases(hints: list[NearbyHint] | None) -> list[str]:
    """Serialize hints for metadata / picker Nearby line."""
    return [h.as_phrase() for h in (hints or [])]


def nearby_hints_to_labels(hints: list[NearbyHint] | None) -> list[str]:
    """Label-only list for similarity matching."""
    return [h.label for h in (hints or [])]


def enrich_nearby_objects_from_goal(
    goal: str,
    nearby_objects: list[str] | None = None,
    *,
    only_upgrade_existing: bool = False,
) -> list[str] | None:
    """Restore directed sides stripped from tool args using the step goal.

    Parses ``（在「…」的下面）``-style landmarks from ``goal``. When the goal has
    directed sides, those win for matching labels; any extra LLM landmarks are
    kept. With ``only_upgrade_existing=True``, only labels already present in
    ``nearby_objects`` are upgraded (no injection of other goal landmarks).

    Returns ``None`` when there is nothing to pass through (no goal sides and
    ``nearby_objects`` is ``None``).
    """
    goal_directed = [
        hint
        for hint in extract_nearby_hints_from_instruction(goal or "")
        if hint.side is not None
    ]
    if not goal_directed:
        if nearby_objects is None:
            return None
        return list(nearby_objects)

    if nearby_objects is None:
        if only_upgrade_existing:
            return None
        return nearby_hints_to_phrases(goal_directed)

    if only_upgrade_existing:
        current_labels = {
            hint.label for hint in normalize_nearby_hints(nearby_objects)
        }
        relevant = [hint for hint in goal_directed if hint.label in current_labels]
        if not relevant:
            return list(nearby_objects)
        # Goal sides first so undirected duplicates upgrade.
        return nearby_hints_to_phrases(merge_nearby_hints(relevant, nearby_objects))

    # Goal sides first so undirected / wrong-sided LLM duplicates upgrade correctly.
    return nearby_hints_to_phrases(merge_nearby_hints(goal_directed, nearby_objects))


def enrich_tool_arguments_from_goal(
    tool_name: str,
    arguments: dict[str, Any],
    goal: str,
) -> dict[str, Any]:
    """Shallow-copy ``arguments`` with nearby_* lists upgraded from ``goal`` sides."""
    args = dict(arguments)
    name = (tool_name or "").strip()

    def _apply(key: str, *, inject_if_missing: bool, only_upgrade_existing: bool) -> None:
        raw = args.get(key)
        if raw is None and not inject_if_missing:
            return
        current = list(raw) if isinstance(raw, list) else None
        enriched = enrich_nearby_objects_from_goal(
            goal,
            current,
            only_upgrade_existing=only_upgrade_existing,
        )
        if enriched is None:
            return
        if current is None or enriched != current:
            args[key] = enriched

    if name in ("move_mouse", "check_object_exists"):
        # Inject when the model omitted nearby_objects entirely but the goal
        # still carries directed landmarks.
        _apply(
            "nearby_objects",
            inject_if_missing=True,
            only_upgrade_existing=False,
        )
    elif name == "drag":
        # Only upgrade labels already present on each endpoint list so
        # destination-only landmarks are not copied onto the drag source.
        _apply(
            "start_nearby_objects",
            inject_if_missing=False,
            only_upgrade_existing=True,
        )
        _apply(
            "destination_nearby_objects",
            inject_if_missing=False,
            only_upgrade_existing=True,
        )
    return args


def side_to_schema_value(side: Side | None) -> str | None:
    return None if side is None else side.value


def parse_side_schema_value(raw: Any) -> Side | None:
    if raw is None:
        return None
    if isinstance(raw, Side):
        return raw
    if not isinstance(raw, str):
        raise ValueError("side must be a string or null")
    text = raw.strip()
    if not text:
        return None
    try:
        return Side(text)
    except ValueError as exc:
        raise ValueError(f"unknown side: {raw!r}") from exc
