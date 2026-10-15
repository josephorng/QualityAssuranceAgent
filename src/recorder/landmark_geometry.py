"""Scale-normalized landmark constellation for cache-replay mouse fallback.

Recording stores a full graph of landmark↔landmark and landmark→target vectors
as ``(dx/s, dy/s)`` where ``s`` is the median pairwise landmark-center distance.
Replay filters OCR twin landmarks that break the constellation, then picks a
detection near the synthesized target or clicks the synthesized point.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from src.common.nearby_side import (
    NearbyHint,
    normalize_nearby_hints,
    side_to_schema_value,
)

GEOMETRY_VERSION = 1

# Direction / magnitude agreement for inter-landmark edges.
COSINE_OK = 0.85
NORMALIZED_L2_OK = 0.4
# Candidate center must lie within this fraction of replay scale from synth point.
TARGET_NEAR_SYNTH = 0.35
# Drop a landmark when more than this fraction of its edges to other matched
# landmarks fail the vector checks.
EDGE_FAIL_FRACTION = 0.5

MIN_LANDMARKS_FOR_MUTUAL_FILTER = 3
MIN_LANDMARKS_FOR_BUILD = 2
MIN_LANDMARKS_FOR_SYNTH = 1
MIN_SCALE = 1.0


def _as_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def _as_bbox_xywh(value: Any) -> tuple[int, int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return int(value[0]), int(value[1]), int(value[2]), int(value[3])
    return None


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w * 0.5, y + h * 0.5


def _median_pairwise_scale(centers: list[tuple[float, float]]) -> float:
    """Median pairwise Euclidean distance among centers; at least ``MIN_SCALE``."""
    if len(centers) < 2:
        return MIN_SCALE
    distances: list[float] = []
    for i, (x0, y0) in enumerate(centers):
        for x1, y1 in centers[i + 1 :]:
            distances.append(math.hypot(x1 - x0, y1 - y0))
    if not distances:
        return MIN_SCALE
    return max(float(statistics.median(distances)), MIN_SCALE)


def normalize_vector(
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    """Return ``(to - from) / scale`` with ``scale`` floored at ``MIN_SCALE``."""
    s = max(float(scale), MIN_SCALE)
    return (to_xy[0] - from_xy[0]) / s, (to_xy[1] - from_xy[1]) / s


def _vector_l2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vector_cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    na = math.hypot(a[0], a[1])
    nb = math.hypot(b[0], b[1])
    if na < 1e-9 or nb < 1e-9:
        # Near-zero vectors: treat as matching only when both are tiny.
        return 1.0 if na < 1e-9 and nb < 1e-9 else 0.0
    return (a[0] * b[0] + a[1] * b[1]) / (na * nb)


def vectors_agree(
    recorded: tuple[float, float],
    live: tuple[float, float],
    *,
    cosine_ok: float = COSINE_OK,
    l2_ok: float = NORMALIZED_L2_OK,
) -> bool:
    """True when direction and normalized displacement both stay within thresholds."""
    if _vector_cosine(recorded, live) < cosine_ok:
        return False
    return _vector_l2(recorded, live) <= l2_ok


def target_far_from_synthesized(
    candidate_xy: tuple[float, float],
    synth_xy: tuple[float, float],
    scale: float,
    *,
    near_frac: float = TARGET_NEAR_SYNTH,
) -> bool:
    """True when ``candidate_xy`` is farther than ``near_frac * scale`` from synth."""
    s = max(float(scale), MIN_SCALE)
    dist = math.hypot(candidate_xy[0] - synth_xy[0], candidate_xy[1] - synth_xy[1])
    return dist / s > near_frac


def _candidate_hub_label(candidate: dict[str, Any]) -> str | None:
    """Hub-style landmark label for a vision candidate dict."""
    from src.recorder.vision_context import format_drag_candidate_anchor

    return format_drag_candidate_anchor(candidate)


def _resolve_target_xy(
    vision: dict[str, Any],
    target_xy: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if target_xy is not None:
        return float(target_xy[0]), float(target_xy[1])
    local = vision.get("local_cursor")
    parsed = _as_xy(local)
    if parsed is not None:
        return parsed
    candidates = vision.get("candidates") or []
    if candidates and isinstance(candidates[0], dict):
        bbox = _as_bbox_xywh(candidates[0].get("bbox"))
        if bbox is not None:
            return _bbox_center(bbox)
    return None


def _resolve_landmark_centers(
    vision: dict[str, Any],
    hints: list[NearbyHint],
) -> list[tuple[str, str | None, tuple[float, float]]]:
    """Map each hint label to the first matching candidate center (unique labels)."""
    candidates = vision.get("candidates") or []
    if not isinstance(candidates, list):
        return []
    label_to_center: dict[str, tuple[float, float]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        label = _candidate_hub_label(candidate)
        if not label or label in label_to_center:
            continue
        bbox = _as_bbox_xywh(candidate.get("bbox"))
        if bbox is None:
            continue
        label_to_center[label] = _bbox_center(bbox)

    resolved: list[tuple[str, str | None, tuple[float, float]]] = []
    seen: set[str] = set()
    for hint in hints:
        label = (hint.label or "").strip()
        if not label or label in seen:
            continue
        center = label_to_center.get(label)
        if center is None:
            continue
        seen.add(label)
        resolved.append((label, side_to_schema_value(hint.side), center))
    return resolved


def build_landmark_geometry(
    vision: dict[str, Any] | None,
    nearby_hints: list[NearbyHint] | list[str] | list[Any] | None,
    *,
    target_xy: tuple[float, float] | None = None,
    target_label: str = "",
) -> dict[str, Any] | None:
    """Build a full scale-normalized constellation from recording vision.

    Returns ``None`` when fewer than ``MIN_LANDMARKS_FOR_BUILD`` landmarks resolve
    or the target center is unavailable.
    """
    if not isinstance(vision, dict):
        return None
    hints = normalize_nearby_hints(nearby_hints)
    if len(hints) < MIN_LANDMARKS_FOR_BUILD:
        return None
    resolved = _resolve_landmark_centers(vision, hints)
    if len(resolved) < MIN_LANDMARKS_FOR_BUILD:
        return None
    target = _resolve_target_xy(vision, target_xy)
    if target is None:
        return None

    centers = [center for _label, _side, center in resolved]
    scale = _median_pairwise_scale(centers)

    landmarks = [
        {"label": label, "side": side} for label, side, _center in resolved
    ]
    landmark_to_landmark: list[dict[str, Any]] = []
    for i, (from_label, _fs, from_xy) in enumerate(resolved):
        for j, (to_label, _ts, to_xy) in enumerate(resolved):
            if i == j:
                continue
            dx, dy = normalize_vector(from_xy, to_xy, scale)
            landmark_to_landmark.append(
                {"from": from_label, "to": to_label, "dx": dx, "dy": dy}
            )
    landmark_to_target: list[dict[str, Any]] = []
    for label, _side, from_xy in resolved:
        dx, dy = normalize_vector(from_xy, target, scale)
        landmark_to_target.append({"from": label, "dx": dx, "dy": dy})

    payload: dict[str, Any] = {
        "version": GEOMETRY_VERSION,
        "scale": scale,
        "landmarks": landmarks,
        "landmark_to_landmark": landmark_to_landmark,
        "landmark_to_target": landmark_to_target,
    }
    cleaned_target = (target_label or "").strip()
    if cleaned_target:
        payload["target_label"] = cleaned_target
    return payload


def parse_landmark_geometry(raw: Any) -> dict[str, Any] | None:
    """Validate a stored geometry payload; return a normalized copy or None."""
    if not isinstance(raw, dict):
        return None
    try:
        version = int(raw.get("version", 0))
    except (TypeError, ValueError):
        return None
    if version != GEOMETRY_VERSION:
        return None
    try:
        scale = float(raw.get("scale", 0.0))
    except (TypeError, ValueError):
        return None
    if scale < MIN_SCALE:
        scale = MIN_SCALE
    landmarks_raw = raw.get("landmarks")
    if not isinstance(landmarks_raw, list) or not landmarks_raw:
        return None
    landmarks: list[dict[str, Any]] = []
    for item in landmarks_raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        side = item.get("side")
        landmarks.append(
            {
                "label": label,
                "side": None if side in (None, "") else str(side),
            }
        )
    if len(landmarks) < MIN_LANDMARKS_FOR_SYNTH:
        return None

    def _edge_list(key: str, *, need_to: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in raw.get(key) or []:
            if not isinstance(item, dict):
                continue
            frm = str(item.get("from") or "").strip()
            if not frm:
                continue
            try:
                dx = float(item.get("dx"))
                dy = float(item.get("dy"))
            except (TypeError, ValueError):
                continue
            row: dict[str, Any] = {"from": frm, "dx": dx, "dy": dy}
            if need_to:
                to = str(item.get("to") or "").strip()
                if not to:
                    continue
                row["to"] = to
            rows.append(row)
        return rows

    l2l = _edge_list("landmark_to_landmark", need_to=True)
    l2t = _edge_list("landmark_to_target", need_to=False)
    if not l2t:
        return None
    out: dict[str, Any] = {
        "version": GEOMETRY_VERSION,
        "scale": scale,
        "landmarks": landmarks,
        "landmark_to_landmark": l2l,
        "landmark_to_target": l2t,
    }
    target_label = str(raw.get("target_label") or "").strip()
    if target_label:
        out["target_label"] = target_label
    return out


def _recorded_edge_map(
    recorded: dict[str, Any],
) -> dict[tuple[str, str], tuple[float, float]]:
    edges: dict[tuple[str, str], tuple[float, float]] = {}
    for item in recorded.get("landmark_to_landmark") or []:
        edges[(str(item["from"]), str(item["to"]))] = (
            float(item["dx"]),
            float(item["dy"]),
        )
    return edges


def filter_landmarks_by_constellation(
    recorded: dict[str, Any],
    matched_centers: dict[str, tuple[float, float]],
    *,
    cosine_ok: float = COSINE_OK,
    l2_ok: float = NORMALIZED_L2_OK,
    edge_fail_fraction: float = EDGE_FAIL_FRACTION,
) -> list[str]:
    """Drop landmarks whose inter-landmark vectors disagree with the recording.

    With fewer than ``MIN_LANDMARKS_FOR_MUTUAL_FILTER`` matched labels, returns
    all matched labels unchanged (mutual filter skipped).
    """
    kept = {
        str(label).strip()
        for label in matched_centers
        if str(label).strip() and label in matched_centers
    }
    if len(kept) < MIN_LANDMARKS_FOR_MUTUAL_FILTER:
        return sorted(kept)

    edge_map = _recorded_edge_map(recorded)
    if not edge_map:
        return sorted(kept)

    # Normalize live edges with the *recorded* scale so one OCR twin cannot
    # inflate the median and cascade-fail every edge via L2.
    try:
        recorded_scale = max(float(recorded.get("scale", MIN_SCALE)), MIN_SCALE)
    except (TypeError, ValueError):
        recorded_scale = MIN_SCALE

    changed = True
    while changed and len(kept) >= MIN_LANDMARKS_FOR_MUTUAL_FILTER:
        changed = False
        drop: list[str] = []
        for label in sorted(kept):
            others = [other for other in kept if other != label]
            if len(others) < 2:
                continue
            fail = 0
            total = 0
            for other in others:
                recorded_vec = edge_map.get((label, other))
                if recorded_vec is None:
                    continue
                live_vec = normalize_vector(
                    matched_centers[label],
                    matched_centers[other],
                    recorded_scale,
                )
                total += 1
                if not vectors_agree(
                    recorded_vec,
                    live_vec,
                    cosine_ok=cosine_ok,
                    l2_ok=l2_ok,
                ):
                    fail += 1
            if total > 0 and (fail / total) > edge_fail_fraction:
                drop.append(label)
        if drop:
            for label in drop:
                kept.discard(label)
            changed = True

    return sorted(kept)


def replay_scale(
    recorded: dict[str, Any],
    kept_centers: dict[str, tuple[float, float]],
) -> float:
    """Replay scale from remaining landmarks, else recorded scale."""
    if len(kept_centers) >= 2:
        return _median_pairwise_scale(list(kept_centers.values()))
    try:
        return max(float(recorded.get("scale", MIN_SCALE)), MIN_SCALE)
    except (TypeError, ValueError):
        return MIN_SCALE


def synthesize_target_point(
    recorded: dict[str, Any],
    kept_centers: dict[str, tuple[float, float]],
) -> tuple[int, int] | None:
    """Mean of landmark→target predictions using recorded normalized vectors."""
    if len(kept_centers) < MIN_LANDMARKS_FOR_SYNTH:
        return None
    scale = replay_scale(recorded, kept_centers)
    preds: list[tuple[float, float]] = []
    for item in recorded.get("landmark_to_target") or []:
        label = str(item.get("from") or "").strip()
        center = kept_centers.get(label)
        if center is None:
            continue
        try:
            dx = float(item["dx"])
            dy = float(item["dy"])
        except (KeyError, TypeError, ValueError):
            continue
        preds.append((center[0] + dx * scale, center[1] + dy * scale))
    if not preds:
        return None
    mx = sum(p[0] for p in preds) / len(preds)
    my = sum(p[1] for p in preds) / len(preds)
    return int(round(mx)), int(round(my))


def pick_detection_near_synthesized(
    candidates: list[Any],
    synth_xy: tuple[int, int],
    scale: float,
    *,
    near_frac: float = TARGET_NEAR_SYNTH,
    center_attr: str = "cx",
) -> Any | None:
    """Return the candidate whose center is closest to ``synth_xy`` and not far.

    ``candidates`` are objects with ``.cx`` / ``.cy`` (e.g. ``UiDetection``).
    """
    best = None
    best_dist = float("inf")
    for det in candidates:
        try:
            cx = float(getattr(det, "cx"))
            cy = float(getattr(det, "cy"))
        except (TypeError, ValueError, AttributeError):
            continue
        if target_far_from_synthesized((cx, cy), synth_xy, scale, near_frac=near_frac):
            continue
        dist = math.hypot(cx - synth_xy[0], cy - synth_xy[1])
        if dist < best_dist:
            best_dist = dist
            best = det
    return best
