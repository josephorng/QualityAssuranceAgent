from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    return ax < bx2 and bx < ax2 and ay < by2 and by < ay2


def merge_two_boxes(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the smallest axis-aligned box containing both ``(x, y, w, h)`` boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    return x1, y1, x2 - x1, y2 - y1


def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge all transitive overlaps into single bounding boxes."""
    if len(boxes) < 2:
        return boxes
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int]] = []
        while merged:
            current = merged.pop()
            merged_with_current = False
            for i, other in enumerate(merged):
                if boxes_overlap(current, other):
                    current = merge_two_boxes(current, other)
                    merged.pop(i)
                    merged.append(current)
                    changed = True
                    merged_with_current = True
                    break
            if not merged_with_current:
                next_boxes.append(current)
        merged = next_boxes
    return merged


def iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for ``(x, y, w, h)`` boxes; 0 when either has no area."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    union = float(aw * ah + bw * bh) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def sort_by_reading_order(
    items: list[T],
    *,
    center_fn,
    row_height_fn,
    x_fn,
) -> list[T]:
    if not items:
        return []
    sorted_items = sorted(items, key=lambda i: (center_fn(i)[1], x_fn(i)))
    mean_h = sum(float(row_height_fn(i)) for i in items) / len(items)
    tol = max(10.0, mean_h * 0.5)
    rows: list[list[T]] = []
    row: list[T] = []
    row_y0: float | None = None
    for item in sorted_items:
        cy = float(center_fn(item)[1])
        if row_y0 is None:
            row = [item]
            row_y0 = cy
            continue
        if abs(cy - row_y0) <= tol:
            row.append(item)
        else:
            rows.append(sorted(row, key=x_fn))
            row = [item]
            row_y0 = cy
    if row:
        rows.append(sorted(row, key=x_fn))
    ordered: list[T] = []
    for r in rows:
        ordered.extend(r)
    return ordered
