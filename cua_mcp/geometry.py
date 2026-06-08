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
