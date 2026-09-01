"""Spatial (SLIC) color segmentation for UI panels and landmark-scoped YOLO lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic

from cua_mcp.read_screen_text.get_coordinates import _yolo_classed_boxes
from cua_mcp.select_mouse_target import _detect_mouse_targets_from_bgr
from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    MOUSE_TARGET_CLASS_IDS,
    OCR_DETECTION_CLASS_IDS,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
)
from src.common.io_utils import imread_bgr, read_json
from src.common.settings import ROOT_DIR

T = TypeVar("T")

MASK_CLASS_IDS = OCR_DETECTION_CLASS_IDS | {YOLO_CLASS_INPUT, YOLO_CLASS_SCROLLBAR}
FILTER_CLASS_IDS = OCR_DETECTION_CLASS_IDS
_DEFAULT_COLOR_SEGMENT_PARAMS_PATH = ROOT_DIR / "color_segment_params.json"
_UNASSIGNED_SPATIAL_RANK = 10_000

# Slider bounds mirrored from the color-segments viewer (for JSON load clamping).
_PARAM_CLAMP_BOUNDS: dict[str, tuple[float, float]] = {
    "num_colors": (0.0, 300.0),
    "min_area_frac": (0.05, 20.0),
    "blur_ksize": (1.0, 31.0),
    "edge_canny_low": (5.0, 150.0),
    "edge_canny_high": (30.0, 255.0),
    "edge_dilate": (0.0, 8.0),
    "slic_compactness": (1.0, 30.0),
    "split_max_area_frac": (2.0, 40.0),
    "merge_color_dist": (2.0, 40.0),
}

_CLASS_NAME_TO_ID: dict[str, int] = {
    name: idx for idx, name in YOLO_CLASS_NAMES.items()
}
_SKIP_SIDEcar_CLASS_NAMES = frozenset({"scrollbar_original", "input_original"})


@dataclass(frozen=True)
class ColorSegmentParams:
    num_colors: int = 120
    slic_compactness: float = 10.0
    min_area_frac: float = 0.003
    blur_ksize: int = 5
    mask_text_icons: bool = True
    require_yolo_objects: bool = True
    merge_superpixels: bool = True
    merge_similar: bool = False
    merge_color_dist: float = 10.0
    split_large_regions: bool = True
    split_max_area_frac: float = 0.06
    edge_canny_low: int = 30
    edge_canny_high: int = 100
    edge_dilate: int = 2


@dataclass(frozen=True)
class ColorRegion:
    region_id: int
    bbox: tuple[int, int, int, int]
    mean_color: tuple[int, int, int]
    area: int


@dataclass(frozen=True)
class SegmentDetection:
    """One YOLO UI detection used for masking and region filtering."""

    box: tuple[int, int, int, int]
    class_id: int
    class_name: str = ""
    text: str = ""


@dataclass
class ColorSegmentResult:
    regions: list[ColorRegion]
    quantized: Image.Image
    label_map: np.ndarray
    masked_box_count: int = 0
    regions_before_yolo_filter: int = 0
    prepared: Image.Image | None = None
    mask_boxes: list[tuple[int, int, int, int]] | None = None
    detections: list[SegmentDetection] = field(default_factory=list)


def _clamp_param(key: str, value: float) -> float:
    bounds = _PARAM_CLAMP_BOUNDS.get(key)
    if bounds is None:
        return float(value)
    lo, hi = bounds
    return max(lo, min(hi, float(value)))


def load_color_segment_params(
    path: Path | None = None,
) -> ColorSegmentParams:
    """Load segmentation params from JSON; fall back to defaults on missing/invalid."""
    defaults = ColorSegmentParams()
    params_path = path or _DEFAULT_COLOR_SEGMENT_PARAMS_PATH
    raw = read_json(params_path, default={})
    if not isinstance(raw, dict):
        return defaults

    def _num(key: str, fallback: float) -> float:
        value = raw.get(key, fallback)
        try:
            return _clamp_param(key, float(value))
        except (TypeError, ValueError):
            return fallback

    blur = int(round(_num("blur_ksize", defaults.blur_ksize)))
    if blur % 2 == 0:
        blur = max(1, blur - 1)
    min_area_pct = _num("min_area_frac", defaults.min_area_frac * 100.0)
    split_pct = _num("split_max_area_frac", defaults.split_max_area_frac * 100.0)
    return ColorSegmentParams(
        num_colors=max(0, int(round(_num("num_colors", defaults.num_colors)))),
        slic_compactness=float(_num("slic_compactness", defaults.slic_compactness)),
        min_area_frac=max(0.0001, min_area_pct / 100.0),
        blur_ksize=blur,
        mask_text_icons=bool(raw.get("mask_text_icons", defaults.mask_text_icons)),
        require_yolo_objects=bool(
            raw.get("require_yolo_objects", defaults.require_yolo_objects)
        ),
        merge_superpixels=bool(raw.get("merge_superpixels", defaults.merge_superpixels)),
        merge_similar=bool(raw.get("merge_similar", defaults.merge_similar)),
        merge_color_dist=float(_num("merge_color_dist", defaults.merge_color_dist)),
        split_large_regions=bool(raw.get("split_large_regions", defaults.split_large_regions)),
        split_max_area_frac=max(0.01, split_pct / 100.0),
        edge_canny_low=int(round(_num("edge_canny_low", defaults.edge_canny_low))),
        edge_canny_high=int(round(_num("edge_canny_high", defaults.edge_canny_high))),
        edge_dilate=max(0, int(round(_num("edge_dilate", defaults.edge_dilate)))),
    )


def _sidecar_json_path(image_path: Path, run_dir: Path | None = None) -> Path:
    sidecar = image_path.with_suffix(".json")
    if sidecar.is_file():
        return sidecar
    if run_dir is not None:
        yolo_json = run_dir / "yolo_ocr" / f"{image_path.stem}.json"
        if yolo_json.is_file():
            return yolo_json
    return sidecar


def _class_id_from_name(class_name: str, class_id: int | None) -> int | None:
    if class_id is not None:
        return int(class_id)
    name = class_name.strip().lower()
    if name in _CLASS_NAME_TO_ID:
        return _CLASS_NAME_TO_ID[name]
    return None


def _detection_from_candidate(item: dict[str, Any]) -> SegmentDetection | None:
    bbox = item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    class_name = str(item.get("class_name", "")).strip()
    if class_name in _SKIP_SIDEcar_CLASS_NAMES:
        return None
    class_id = _class_id_from_name(class_name, item.get("class_id") if isinstance(item.get("class_id"), int) else None)
    if class_id is None:
        return None
    return SegmentDetection(
        box=(x, y, w, h),
        class_id=class_id,
        class_name=class_name or YOLO_CLASS_NAMES.get(class_id, ""),
        text=str(item.get("text") or "").strip(),
    )


def _detection_from_line_item(item: Any) -> SegmentDetection | None:
    if isinstance(item, dict):
        box = item.get("box") or item.get("bbox") or item.get("rect")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            x, y, w, h = (int(v) for v in box)
        except (TypeError, ValueError):
            return None
        class_name = str(item.get("class_name", "")).strip()
        if class_name in _SKIP_SIDEcar_CLASS_NAMES:
            return None
        raw_id = item.get("class_id")
        class_id = _class_id_from_name(
            class_name,
            int(raw_id) if isinstance(raw_id, int) else None,
        )
        if class_id is None:
            line_type = str(item.get("line_type", "")).strip()
            if line_type == "ocr" or class_name == "text":
                class_id = YOLO_CLASS_TEXT
            elif class_name in ("element", "input", "scrollbar"):
                class_id = _CLASS_NAME_TO_ID.get(class_name)
            else:
                class_id = YOLO_CLASS_TEXT
        if class_id is None:
            return None
        return SegmentDetection(
            box=(x, y, w, h),
            class_id=class_id,
            class_name=class_name or YOLO_CLASS_NAMES.get(class_id, ""),
            text=str(item.get("text", "")).strip(),
        )
    if (
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and isinstance(item[0], (list, tuple))
        and len(item[0]) == 4
    ):
        try:
            x, y, w, h = (int(v) for v in item[0])
        except (TypeError, ValueError):
            return None
        if len(item) == 3 and isinstance(item[2], list):
            text = "".join(str(p) for p in item[2]).strip()
        else:
            text = str(item[1]).strip()
        return SegmentDetection(
            box=(x, y, w, h),
            class_id=YOLO_CLASS_TEXT,
            class_name="text",
            text=text,
        )
    return None


def _parse_sidecar_detections(data: dict[str, Any]) -> list[SegmentDetection]:
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        out: list[SegmentDetection] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            det = _detection_from_candidate(item)
            if det is not None:
                out.append(det)
        return out
    lines = data.get("lines", [])
    if not isinstance(lines, list):
        return []
    out: list[SegmentDetection] = []
    for item in lines:
        det = _detection_from_line_item(item)
        if det is not None:
            out.append(det)
    return out


def _detections_for_classes(
    detections: list[SegmentDetection],
    class_ids: set[int] | frozenset[int],
) -> list[SegmentDetection]:
    classes = set(class_ids)
    return [det for det in detections if det.class_id in classes]


def _boxes_for_classes(
    detections: list[SegmentDetection],
    class_ids: set[int] | frozenset[int],
) -> list[tuple[int, int, int, int]]:
    return [det.box for det in _detections_for_classes(detections, class_ids)]


def resolve_segment_detections(
    rgb: np.ndarray,
    *,
    image_path: Path | None = None,
    run_dir: Path | None = None,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> list[SegmentDetection]:
    """Resolve UI detections from sidecar JSON when available, else live YOLO."""
    if image_path is not None:
        json_path = _sidecar_json_path(image_path, run_dir)
        if json_path.is_file():
            try:
                raw = read_json(json_path, default={})
                if isinstance(raw, dict):
                    parsed = _parse_sidecar_detections(raw)
                    if parsed:
                        return parsed
            except Exception:
                pass

        try:
            bgr = imread_bgr(image_path)
            if bgr is not None:
                candidates = _detect_mouse_targets_from_bgr(
                    bgr,
                    yolo_conf_threshold=yolo_conf_threshold,
                )
                if candidates:
                    return [
                        SegmentDetection(
                            box=det.bbox,
                            class_id=int(det.class_id),
                            class_name=str(det.class_name or ""),
                            text=str(det.text or "").strip(),
                        )
                        for det in candidates
                    ]
        except Exception:
            pass

    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        classed = _yolo_classed_boxes(
            bgr,
            class_ids=MOUSE_TARGET_CLASS_IDS,
            conf_threshold=yolo_conf_threshold,
        )
        return [
            SegmentDetection(
                box=bbox,
                class_id=cls_id,
                class_name=YOLO_CLASS_NAMES.get(cls_id, ""),
            )
            for bbox, cls_id in classed
        ]
    except Exception:
        return []


def resolve_segment_box_sets(
    rgb: np.ndarray,
    *,
    image_path: Path | None = None,
    run_dir: Path | None = None,
    mask_class_ids: frozenset[int] | set[int] = MASK_CLASS_IDS,
    filter_class_ids: frozenset[int] | set[int] = FILTER_CLASS_IDS,
    yolo_conf_threshold: float = DEFAULT_CONF_YOLOV26_END2END,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Return ``(mask_boxes, text_icon_boxes)`` from one detection pass."""
    detections = resolve_segment_detections(
        rgb,
        image_path=image_path,
        run_dir=run_dir,
        yolo_conf_threshold=yolo_conf_threshold,
    )
    return (
        _boxes_for_classes(detections, mask_class_ids),
        _boxes_for_classes(detections, filter_class_ids),
    )


def region_id_at_point(label_map: np.ndarray, x: int, y: int) -> int | None:
    if y < 0 or x < 0 or y >= label_map.shape[0] or x >= label_map.shape[1]:
        return None
    region_id = int(label_map[y, x])
    return region_id if region_id >= 0 else None


def _region_ids_overlapping_box(
    label_map: np.ndarray,
    box: tuple[int, int, int, int],
) -> set[int]:
    """Return region ids with any labeled pixel under the box (xywh)."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return set()
    h_img, w_img = label_map.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w_img, x0 + int(w))
    y1 = min(h_img, y0 + int(h))
    if x1 <= x0 or y1 <= y0:
        return set()
    patch = label_map[y0:y1, x0:x1]
    return {int(rid) for rid in np.unique(patch) if int(rid) >= 0}


def _smallest_region_id(label_map: np.ndarray, region_ids: set[int]) -> int | None:
    if not region_ids:
        return None
    if len(region_ids) == 1:
        return next(iter(region_ids))
    return min(region_ids, key=lambda rid: int(np.sum(label_map == rid)))


def _largest_region_id(label_map: np.ndarray, region_ids: set[int]) -> int | None:
    if not region_ids:
        return None
    if len(region_ids) == 1:
        return next(iter(region_ids))
    return max(region_ids, key=lambda rid: int(np.sum(label_map == rid)))


def _point_inside_box_xywh(
    px: int,
    py: int,
    box: tuple[int, int, int, int],
) -> bool:
    x, y, w, h = box
    return int(x) <= px < int(x) + int(w) and int(y) <= py < int(y) + int(h)


def _nearest_region_id_to_point(
    regions: list[ColorRegion],
    px: float,
    py: float,
) -> int | None:
    if not regions:
        return None
    return min(
        regions,
        key=lambda region: _region_center_distance_sq(
            region,
            px,
            py,
        ),
    ).region_id


def _point_in_region_bbox_xyxy(
    px: float,
    py: float,
    bbox: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = bbox
    return float(x0) <= px <= float(x1) and float(y0) <= py <= float(y1)


def _bbox_area_xyxy(bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    w = max(0, int(x1) - int(x0) + 1)
    h = max(0, int(y1) - int(y0) + 1)
    return w * h


def _smallest_region_id_containing_point(
    regions: list[ColorRegion],
    px: float,
    py: float,
) -> int | None:
    """Pick the smallest region bbox that contains ``(px, py)``."""
    candidates = [
        region
        for region in regions
        if _point_in_region_bbox_xyxy(px, py, region.bbox)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda region: _bbox_area_xyxy(region.bbox)).region_id


def region_id_for_box(
    label_map: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    regions: list[ColorRegion] | None = None,
) -> int | None:
    """Map a detection box to a region; when several overlap, pick the smallest."""
    overlapping = _region_ids_overlapping_box(label_map, box)
    if overlapping:
        smallest = _smallest_region_id(label_map, overlapping)
        if smallest is not None:
            return smallest
        return _largest_region_id(label_map, overlapping)
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    center_rid = region_id_at_point(label_map, x + w // 2, y + h // 2)
    if center_rid is not None:
        return center_rid
    if regions:
        bbox_rid = _smallest_region_id_containing_point(regions, cx, cy)
        if bbox_rid is not None:
            return bbox_rid
        return _nearest_region_id_to_point(regions, cx, cy)
    return None


def landmark_region_id_for_box(
    label_map: np.ndarray,
    landmark_box: tuple[int, int, int, int],
    *,
    cursor_xy: tuple[int, int] | None = None,
    regions: list[ColorRegion] | None = None,
) -> int | None:
    """Resolve the landmark color region for spatial ranking.

    Prefer the click/cursor point inside ``landmark_box``, then the box center,
    then smallest overlapping region, largest overlapping, then nearest region
    bbox center.
    """
    overlapping = _region_ids_overlapping_box(label_map, landmark_box)
    x, y, w, h = landmark_box
    box_cx = x + w // 2
    box_cy = y + h // 2
    ref_x, ref_y = box_cx, box_cy

    if cursor_xy is not None:
        cx, cy = int(cursor_xy[0]), int(cursor_xy[1])
        if _point_inside_box_xywh(cx, cy, landmark_box):
            ref_x, ref_y = cx, cy
            cursor_rid = region_id_at_point(label_map, cx, cy)
            if cursor_rid is not None:
                return cursor_rid

    center_rid = region_id_at_point(label_map, box_cx, box_cy)
    if center_rid is not None:
        return center_rid

    if overlapping:
        smallest = _smallest_region_id(label_map, overlapping)
        if smallest is not None:
            return smallest
        return _largest_region_id(label_map, overlapping)

    if regions:
        bbox_rid = _smallest_region_id_containing_point(regions, ref_x, ref_y)
        if bbox_rid is not None:
            return bbox_rid
        return _nearest_region_id_to_point(regions, ref_x, ref_y)
    return None


_REGION_INSIDE_MIN_FRAC = 0.8


def _bbox_intersection_area_xyxy(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(int(ax0), int(bx0))
    iy0 = max(int(ay0), int(by0))
    ix1 = min(int(ax1), int(bx1))
    iy1 = min(int(ay1), int(by1))
    if ix1 < ix0 or iy1 < iy0:
        return 0
    return (ix1 - ix0 + 1) * (iy1 - iy0 + 1)


def _region_mostly_inside(
    inner: ColorRegion,
    outer: ColorRegion,
    *,
    min_frac: float = _REGION_INSIDE_MIN_FRAC,
) -> bool:
    if inner.area >= outer.area:
        return False
    inner_area = _bbox_area_xyxy(inner.bbox)
    if inner_area <= 0:
        return False
    overlap = _bbox_intersection_area_xyxy(inner.bbox, outer.bbox)
    return overlap / inner_area >= min_frac


def _build_immediate_parent_map(regions: list[ColorRegion]) -> dict[int, int | None]:
    """Map each region id to its immediate parent (smallest container region)."""
    parent: dict[int, int | None] = {region.region_id: None for region in regions}
    for inner in sorted(regions, key=lambda region: region.area):
        containers = [
            outer
            for outer in regions
            if outer.region_id != inner.region_id
            and outer.area > inner.area
            and _region_mostly_inside(inner, outer)
        ]
        if containers:
            parent[inner.region_id] = min(containers, key=lambda region: region.area).region_id
    return parent


_VIRTUAL_ROOT_ID = -1


def _extended_parent_map(parent_map: dict[int, int | None]) -> dict[int, int | None]:
    """Attach a virtual root parent to every top-level region."""
    extended = dict(parent_map)
    for region_id, parent_id in parent_map.items():
        if parent_id is None:
            extended[region_id] = _VIRTUAL_ROOT_ID
    return extended


def _region_tree_distance(
    parent_map: dict[int, int | None],
    from_id: int,
    to_id: int,
) -> int:
    """Steps between regions on the containment tree (virtual root over top-level)."""
    if from_id == to_id:
        return 0
    extended = _extended_parent_map(parent_map)
    ancestor_depth: dict[int, int] = {}
    current = from_id
    depth = 0
    while True:
        ancestor_depth[current] = depth
        parent_id = extended.get(current)
        if parent_id is None:
            break
        current = parent_id
        depth += 1

    current = to_id
    steps_to = 0
    while current not in ancestor_depth:
        parent_id = extended.get(current)
        if parent_id is None:
            break
        current = parent_id
        steps_to += 1
    return ancestor_depth[current] + steps_to


def _landmark_ancestor_region_ids(
    landmark_region_id: int,
    parent_map: dict[int, int | None],
) -> list[int]:
    """Landmark region followed by each ancestor up to the top level."""
    ids: list[int] = []
    current: int | None = landmark_region_id
    while current is not None and current != _VIRTUAL_ROOT_ID:
        ids.append(current)
        current = parent_map.get(current)
    return ids


def _is_under_region(
    region_id: int,
    ancestor_id: int,
    parent_map: dict[int, int | None],
) -> bool:
    current: int | None = region_id
    while current is not None:
        if current == ancestor_id:
            return True
        current = parent_map.get(current)
    return False


def _point_in_bbox_xyxy(
    px: float,
    py: float,
    bbox: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = bbox
    return float(x0) <= px <= float(x1) and float(y0) <= py <= float(y1)


def _spatial_region_rank_for_detection(
    landmark_region_id: int | None,
    assigned_region_id: int | None,
    detection_box: tuple[int, int, int, int],
    parent_map: dict[int, int | None],
    region_by_id: dict[int, ColorRegion],
) -> int:
    """Rank one detection relative to the landmark region.

    When the assigned region sits on a sibling branch of the landmark's parent
    (e.g. white panel vs dark frame), rank by how many landmark ancestors
    contain the detection center instead of tree distance to the assigned leaf.
    """
    if landmark_region_id is None or assigned_region_id is None:
        return _UNASSIGNED_SPATIAL_RANK
    if _is_under_region(assigned_region_id, landmark_region_id, parent_map):
        return _region_tree_distance(
            parent_map,
            landmark_region_id,
            assigned_region_id,
        )
    if _is_under_region(landmark_region_id, assigned_region_id, parent_map):
        return _region_tree_distance(
            parent_map,
            landmark_region_id,
            assigned_region_id,
        )
    landmark_parent = parent_map.get(landmark_region_id)
    if landmark_parent is not None and _is_under_region(
        assigned_region_id,
        landmark_parent,
        parent_map,
    ):
        return _region_tree_distance(
            parent_map,
            landmark_region_id,
            assigned_region_id,
        )
    x, y, w, h = detection_box
    cx = x + w / 2.0
    cy = y + h / 2.0
    for depth, ancestor_id in enumerate(
        _landmark_ancestor_region_ids(landmark_region_id, parent_map)
    ):
        region = region_by_id.get(ancestor_id)
        if region is None:
            continue
        if _point_in_bbox_xyxy(cx, cy, region.bbox):
            return depth
    return _region_tree_distance(
        parent_map,
        landmark_region_id,
        assigned_region_id,
    )


def _region_center_distance_sq(
    region: ColorRegion,
    lcx: float,
    lcy: float,
) -> float:
    x0, y0, x1, y1 = region.bbox
    rcx = (x0 + x1) / 2.0
    rcy = (y0 + y1) / 2.0
    return (rcx - lcx) ** 2 + (rcy - lcy) ** 2


def _region_ranks_for_landmark(
    landmark_region_id: int | None,
    regions: list[ColorRegion],
    parent_map: dict[int, int | None],
    landmark_box: tuple[int, int, int, int],
    *,
    candidate_region_ids: set[int] | None = None,
) -> dict[int, int]:
    """Rank regions by tree-step distance from the landmark region."""
    if landmark_region_id is None:
        return {}
    all_ids = {region.region_id for region in regions}
    target_ids = all_ids | (candidate_region_ids or set())
    return {
        region_id: _region_tree_distance(parent_map, landmark_region_id, region_id)
        for region_id in target_ids
    }


def color_segment_to_json_dict(
    result: ColorSegmentResult,
    *,
    landmark_box: tuple[int, int, int, int] | None = None,
    cursor_xy: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Serialize color regions for persistence (e.g. ``yolo_ocr/event_NNN.json``)."""
    parent_map = _build_immediate_parent_map(result.regions)
    landmark_region_id: int | None = None
    region_ranks: dict[int, int] = {}
    if landmark_box is not None:
        landmark_region_id = landmark_region_id_for_box(
            result.label_map,
            landmark_box,
            cursor_xy=cursor_xy,
            regions=result.regions,
        )
        region_ranks = _region_ranks_for_landmark(
            landmark_region_id,
            result.regions,
            parent_map,
            landmark_box,
        )
    regions_out: list[dict[str, Any]] = []
    for region in result.regions:
        parent_id = parent_map.get(region.region_id)
        regions_out.append(
            {
                "region_id": region.region_id,
                "bbox": [int(v) for v in region.bbox],
                "mean_color": [int(v) for v in region.mean_color],
                "area": int(region.area),
                "parent_region_id": parent_id,
                "spatial_region_rank": region_ranks.get(
                    region.region_id,
                    _UNASSIGNED_SPATIAL_RANK,
                ),
            }
        )
    return {
        "region_count": len(result.regions),
        "regions_before_yolo_filter": result.regions_before_yolo_filter,
        "landmark_region_id": landmark_region_id,
        "regions": regions_out,
    }


def rank_regions_near_landmark(
    landmark_box: tuple[int, int, int, int],
    regions: list[ColorRegion],
) -> list[ColorRegion]:
    """Sort color regions by distance from the landmark center to each region bbox center."""
    lx, ly, lw, lh = landmark_box
    lcx = lx + lw / 2.0
    lcy = ly + lh / 2.0

    def distance(region: ColorRegion) -> float:
        x0, y0, x1, y1 = region.bbox
        rcx = (x0 + x1) / 2.0
        rcy = (y0 + y1) / 2.0
        return (rcx - lcx) ** 2 + (rcy - lcy) ** 2

    return sorted(regions, key=distance)


def select_detections_for_landmark(
    landmark_box: tuple[int, int, int, int],
    result: ColorSegmentResult,
    detections: list[SegmentDetection] | None = None,
    *,
    cursor_xy: tuple[int, int] | None = None,
    nearby_region_limit: int = 8,
) -> list[SegmentDetection]:
    """Prefer YOLO items in the landmark's color region, then items in nearby regions."""
    dets = list(detections if detections is not None else result.detections)
    if not dets:
        return []
    label_map = result.label_map
    regions = result.regions
    landmark_region_id = landmark_region_id_for_box(
        label_map,
        landmark_box,
        cursor_xy=cursor_xy,
        regions=regions,
    )

    def detection_region_id(det: SegmentDetection) -> int | None:
        bx, by, bw, bh = det.box
        return region_id_for_box(label_map, (bx, by, bw, bh), regions=regions)

    same_region = [
        det
        for det in dets
        if landmark_region_id is not None and detection_region_id(det) == landmark_region_id
    ]
    other = [det for det in dets if det not in same_region]
    if landmark_region_id is None:
        return dets

    parent_map = _build_immediate_parent_map(result.regions)
    region_by_id = {region.region_id: region for region in regions}

    def other_detection_rank(det: SegmentDetection) -> int:
        rid = detection_region_id(det)
        return _spatial_region_rank_for_detection(
            landmark_region_id,
            rid,
            det.box,
            parent_map,
            region_by_id,
        )

    other_sorted = sorted(other, key=other_detection_rank)
    ordered = list(same_region)
    ordered.extend(other_sorted[:nearby_region_limit])
    ordered.extend(other_sorted[nearby_region_limit:])
    return ordered


def spatial_region_rank_for_detections(
    landmark_box: tuple[int, int, int, int],
    result: ColorSegmentResult | None,
    detections: list[SegmentDetection],
    *,
    cursor_xy: tuple[int, int] | None = None,
) -> dict[tuple[int, int, int, int], int]:
    """Map each detection box to a spatial rank (0 = same region as landmark)."""
    if result is None or not detections:
        return {}
    label_map = result.label_map
    regions = result.regions
    landmark_region_id = landmark_region_id_for_box(
        label_map,
        landmark_box,
        cursor_xy=cursor_xy,
        regions=regions,
    )
    parent_map = _build_immediate_parent_map(regions)
    region_by_id = {region.region_id: region for region in regions}

    out: dict[tuple[int, int, int, int], int] = {}
    for det in detections:
        rid = region_id_for_box(label_map, det.box, regions=regions)
        out[tuple(det.box)] = _spatial_region_rank_for_detection(
            landmark_region_id,
            rid,
            det.box,
            parent_map,
            region_by_id,
        )
    return out


def reorder_detections_for_landmark(
    landmark_box: tuple[int, int, int, int],
    result: ColorSegmentResult | None,
    items: list[T],
    box_fn: Callable[[T], tuple[int, int, int, int]],
) -> list[T]:
    """Reorder items: same color region as landmark first, then nearby regions."""
    if not items or result is None:
        return list(items)
    seg_dets = [
        SegmentDetection(box=box_fn(item), class_id=0)
        for item in items
    ]
    ordered_seg = select_detections_for_landmark(
        landmark_box,
        result,
        seg_dets,
    )
    ordered_items: list[T] = []
    seen_ids: set[int] = set()
    for seg in ordered_seg:
        target_box = tuple(seg.box)
        for item in items:
            if tuple(box_fn(item)) != target_box or id(item) in seen_ids:
                continue
            ordered_items.append(item)
            seen_ids.add(id(item))
            break
    for item in items:
        if id(item) not in seen_ids:
            ordered_items.append(item)
    return ordered_items


def _region_contains_text_or_icon(
    region_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    if not boxes or not np.any(region_mask):
        return False
    h, w = region_mask.shape[:2]
    for bx, by, bw, bh in boxes:
        x0 = max(0, int(bx))
        y0 = max(0, int(by))
        x1 = min(w, x0 + max(1, int(bw)))
        y1 = min(h, y0 + max(1, int(bh)))
        if np.any(region_mask[y0:y1, x0:x1]):
            return True
    return False


def _fill_boxes_with_local_background(
    rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    *,
    ring_px: int = 5,
) -> np.ndarray:
    if not boxes:
        return rgb
    out = rgb.copy()
    h, w = out.shape[:2]
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            continue
        rx0 = max(0, x0 - ring_px)
        ry0 = max(0, y0 - ring_px)
        rx1 = min(w, x1 + ring_px)
        ry1 = min(h, y1 + ring_px)
        patch = out[ry0:ry1, rx0:rx1]
        ring_mask = np.ones(patch.shape[:2], dtype=bool)
        ring_mask[(y0 - ry0) : (y1 - ry0), (x0 - rx0) : (x1 - rx0)] = False
        ring_pixels = patch[ring_mask]
        if ring_pixels.size == 0:
            fill = out[max(0, y0 - 1) : y0, x0:x1].reshape(-1, 3)
            if fill.size == 0:
                fill = out[y0:y1, max(0, x0 - 1) : x0].reshape(-1, 3)
        else:
            fill = ring_pixels
        if fill.size == 0:
            continue
        out[y0:y1, x0:x1] = np.median(fill, axis=0).astype(np.uint8)
    return out


def _lab_color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    a = cv2.cvtColor(np.uint8([[c1]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    b = cv2.cvtColor(np.uint8([[c2]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return float(np.linalg.norm(a - b))


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0, max(ax0, bx0) - min(ax1, bx1) - 1)
    dy = max(0, max(ay0, by0) - min(ay1, by1) - 1)
    return max(dx, dy)


def _union_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _merge_similar_color_regions(
    regions: list[ColorRegion],
    label_map: np.ndarray,
    *,
    color_dist: float,
    max_gap: int,
    max_area_frac: float = 0.15,
) -> tuple[list[ColorRegion], np.ndarray]:
    if len(regions) < 2:
        return regions, label_map

    img_area = max(1, int(label_map.shape[0] * label_map.shape[1]))
    max_area = max(1, int(img_area * max_area_frac))
    parent = {region.region_id: region.region_id for region in regions}

    def find(region_id: int) -> int:
        root = region_id
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(left_id: int, right_id: int) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, left in enumerate(regions):
        for right in regions[i + 1 :]:
            if _lab_color_distance(left.mean_color, right.mean_color) > color_dist:
                continue
            if _bbox_gap(left.bbox, right.bbox) > max_gap:
                continue
            if left.area + right.area > max_area:
                continue
            union(left.region_id, right.region_id)

    groups: dict[int, list[ColorRegion]] = {}
    for region in regions:
        groups.setdefault(find(region.region_id), []).append(region)

    merged_regions: list[ColorRegion] = []
    merged_label_map = np.full_like(label_map, -1)
    for new_id, group in enumerate(groups.values()):
        merge_mask = np.zeros(label_map.shape, dtype=bool)
        merged_bbox = group[0].bbox
        total_area = 0
        colors: list[tuple[int, int, int]] = []
        for region in group:
            merge_mask |= label_map == region.region_id
            merged_bbox = _union_bbox(merged_bbox, region.bbox)
            total_area += region.area
            colors.append(region.mean_color)
        mean_color = tuple(int(v) for v in np.mean(np.array(colors, dtype=np.float32), axis=0))
        merged_label_map[merge_mask] = new_id
        merged_regions.append(
            ColorRegion(
                region_id=new_id,
                bbox=merged_bbox,
                mean_color=mean_color,
                area=total_area,
            )
        )

    merged_regions.sort(key=lambda region: region.area, reverse=True)
    final_label_map = np.full_like(merged_label_map, -1)
    final_regions: list[ColorRegion] = []
    for new_id, region in enumerate(merged_regions):
        old_id = region.region_id
        final_label_map[merged_label_map == old_id] = new_id
        final_regions.append(
            ColorRegion(
                region_id=new_id,
                bbox=region.bbox,
                mean_color=region.mean_color,
                area=region.area,
            )
        )
    return final_regions, final_label_map


def prepare_segmentation_image(
    rgb: np.ndarray,
    params: ColorSegmentParams,
    *,
    image_path: Path | None = None,
    run_dir: Path | None = None,
    detections: list[SegmentDetection] | None = None,
) -> tuple[
    np.ndarray,
    int,
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    np.ndarray,
    list[SegmentDetection],
]:
    work = rgb.copy()
    masked_boxes = 0
    resolved = (
        list(detections)
        if detections is not None
        else resolve_segment_detections(
            work,
            image_path=image_path,
            run_dir=run_dir,
        )
    )
    mask_boxes = _boxes_for_classes(resolved, MASK_CLASS_IDS)
    text_icon_boxes = _boxes_for_classes(resolved, FILTER_CLASS_IDS)
    if params.mask_text_icons and mask_boxes:
        masked_boxes = len(mask_boxes)
        work = _fill_boxes_with_local_background(work, mask_boxes)
    masked_before_blur = work.copy()
    blur_k = int(params.blur_ksize)
    if blur_k > 1:
        if blur_k % 2 == 0:
            blur_k += 1
        work = cv2.GaussianBlur(work, (blur_k, blur_k), 0)
    return work, masked_boxes, mask_boxes, text_icon_boxes, masked_before_blur, resolved


def _regions_from_label_map(
    rgb: np.ndarray,
    label_map: np.ndarray,
    *,
    min_area: int,
) -> tuple[list[ColorRegion], np.ndarray]:
    h, w = label_map.shape[:2]
    out_map = np.full((h, w), -1, dtype=np.int32)
    regions: list[ColorRegion] = []
    next_region_id = 0
    for label_id in np.unique(label_map):
        if label_id < 0:
            continue
        mask = (label_map == label_id).astype(np.uint8)
        n_comp, comp_map, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for comp_id in range(1, n_comp):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            bw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
            comp_mask = comp_map == comp_id
            mean_color = tuple(int(v) for v in rgb[comp_mask].mean(axis=0))
            out_map[comp_mask] = next_region_id
            regions.append(
                ColorRegion(
                    region_id=next_region_id,
                    bbox=(x, y, x + bw - 1, y + bh - 1),
                    mean_color=mean_color,
                    area=area,
                )
            )
            next_region_id += 1
    return regions, out_map


def _quantized_from_label_map(
    rgb: np.ndarray, label_map: np.ndarray, regions: list[ColorRegion]
) -> Image.Image:
    quantized = rgb.copy()
    for region in regions:
        mask = label_map == region.region_id
        if not np.any(mask):
            continue
        quantized[mask] = np.array(region.mean_color, dtype=np.uint8)
    return Image.fromarray(quantized, mode="RGB")


def _adjacent_label_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    left = labels[:, :-1]
    right = labels[:, 1:]
    mask = left != right
    for a, b in zip(left[mask], right[mask], strict=False):
        pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    top = labels[:-1, :]
    bottom = labels[1:, :]
    mask = top != bottom
    for a, b in zip(top[mask], bottom[mask], strict=False):
        pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    return pairs


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _split_mask_by_internal_edges(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    min_area: int,
    canny_low: int,
    canny_high: int,
    edge_dilate: int,
) -> list[np.ndarray]:
    if not np.any(mask):
        return []
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop_rgb = rgb[y0 : y1 + 1, x0 : x1 + 1]
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, int(canny_low), int(canny_high))
    edges = cv2.bitwise_and(edges, edges, mask=crop_mask.astype(np.uint8))
    dilate_k = max(1, int(edge_dilate) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
    edges = cv2.dilate(edges, kernel, iterations=1)
    interior = crop_mask & (edges == 0)
    n_comp, comp_map, stats, _centroids = cv2.connectedComponentsWithStats(
        interior.astype(np.uint8),
        connectivity=8,
    )
    sub_masks: list[np.ndarray] = []
    for comp_id in range(1, n_comp):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        full = np.zeros(mask.shape, dtype=bool)
        full[y0 : y1 + 1, x0 : x1 + 1] = comp_map == comp_id
        sub_masks.append(full)
    return sub_masks if len(sub_masks) > 1 else [mask]


def _split_oversized_regions(
    rgb: np.ndarray,
    regions: list[ColorRegion],
    label_map: np.ndarray,
    params: ColorSegmentParams,
) -> tuple[list[ColorRegion], np.ndarray]:
    if not params.split_large_regions or not regions:
        return regions, label_map
    h, w = label_map.shape[:2]
    img_area = max(1, h * w)
    min_area = max(1, int(img_area * params.min_area_frac))
    max_area = max(min_area + 1, int(img_area * params.split_max_area_frac))
    pending: list[tuple[np.ndarray, tuple[int, int, int] | None]] = []
    for region in regions:
        mask = label_map == region.region_id
        if int(mask.sum()) <= max_area:
            pending.append((mask, region.mean_color))
            continue
        sub_masks = _split_mask_by_internal_edges(
            rgb,
            mask,
            min_area=min_area,
            canny_low=params.edge_canny_low,
            canny_high=params.edge_canny_high,
            edge_dilate=params.edge_dilate,
        )
        for sub_mask in sub_masks:
            if int(sub_mask.sum()) < min_area:
                continue
            pixels = rgb[sub_mask]
            mean_color = tuple(int(v) for v in pixels.mean(axis=0))
            pending.append((sub_mask, mean_color))

    new_map = np.full((h, w), -1, dtype=np.int32)
    new_regions: list[ColorRegion] = []
    for region_id, (mask, mean_color) in enumerate(pending):
        area = int(mask.sum())
        if area < min_area:
            continue
        x0, y0, x1, y1 = _bbox_from_mask(mask)
        new_map[mask] = region_id
        new_regions.append(
            ColorRegion(
                region_id=region_id,
                bbox=(x0, y0, x1, y1),
                mean_color=mean_color or (0, 0, 0),
                area=area,
            )
        )
    return new_regions, new_map


def _run_slic_pipeline(
    rgb: np.ndarray, work: np.ndarray, params: ColorSegmentParams
) -> tuple[list[ColorRegion], np.ndarray, Image.Image]:
    h, w = rgb.shape[:2]
    min_area = max(1, int(h * w * params.min_area_frac))
    n_segments = max(1, int(params.num_colors))
    superpixels = slic(
        work,
        n_segments=n_segments,
        compactness=float(params.slic_compactness),
        start_label=0,
        channel_axis=-1,
    ).astype(np.int32)
    label_map = superpixels
    if params.merge_superpixels:
        label_map = _merge_labels_by_color(
            label_map,
            rgb,
            color_dist=float(params.merge_color_dist),
        )
    regions, label_map = _regions_from_label_map(rgb, label_map, min_area=min_area)
    if params.merge_similar and regions:
        regions, label_map = _merge_similar_color_regions(
            regions,
            label_map,
            color_dist=float(params.merge_color_dist),
            max_gap=8,
            max_area_frac=0.15,
        )
    quantized = _quantized_from_label_map(rgb, label_map, regions)
    return regions, label_map, quantized


def segment_image_by_spatial(
    rgb: np.ndarray, work: np.ndarray, params: ColorSegmentParams
) -> tuple[list[ColorRegion], np.ndarray, Image.Image]:
    regions, label_map, quantized = _run_slic_pipeline(rgb, work, params)
    regions, label_map = _split_oversized_regions(rgb, regions, label_map, params)
    if params.split_large_regions:
        regions, label_map = _split_oversized_regions(rgb, regions, label_map, params)
    quantized = _quantized_from_label_map(rgb, label_map, regions)
    return regions, label_map, quantized


def _merge_labels_by_color(
    labels: np.ndarray,
    rgb: np.ndarray,
    *,
    color_dist: float,
) -> np.ndarray:
    unique_ids = [int(v) for v in np.unique(labels) if int(v) >= 0]
    if len(unique_ids) < 2:
        return labels.astype(np.int32)
    means: dict[int, tuple[int, int, int]] = {}
    for label_id in unique_ids:
        pixels = rgb[labels == label_id]
        if pixels.size == 0:
            continue
        means[label_id] = tuple(int(v) for v in pixels.mean(axis=0))

    parent = {label_id: label_id for label_id in means}

    def find(label_id: int) -> int:
        root = label_id
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(left_id: int, right_id: int) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_id, right_id in _adjacent_label_pairs(labels):
        if left_id not in means or right_id not in means:
            continue
        if _lab_color_distance(means[left_id], means[right_id]) <= color_dist:
            union(left_id, right_id)

    remap = {label_id: find(label_id) for label_id in means}
    roots = sorted(set(remap.values()))
    root_to_new = {root: idx for idx, root in enumerate(roots)}
    merged = labels.astype(np.int32).copy()
    for label_id, root in remap.items():
        merged[labels == label_id] = root_to_new[root]
    return merged


def _detection_counts_by_region(
    label_map: np.ndarray,
    detections: list[SegmentDetection],
    *,
    regions: list[ColorRegion] | None = None,
) -> dict[int, int]:
    """Count YOLO detections assigned to each region (same rules as ``region_id_for_box``)."""
    counts: dict[int, int] = {}
    for det in detections:
        rid = region_id_for_box(label_map, det.box, regions=regions)
        if rid is None:
            continue
        counts[rid] = counts.get(rid, 0) + 1
    return counts


def _filter_regions_with_text_icons(
    regions: list[ColorRegion],
    label_map: np.ndarray,
    text_icon_boxes: list[tuple[int, int, int, int]],
    *,
    detections: list[SegmentDetection] | None = None,
    min_detections_per_region: int = 1,
) -> tuple[list[ColorRegion], np.ndarray]:
    if not regions or not text_icon_boxes:
        return regions, label_map
    det_counts = (
        _detection_counts_by_region(label_map, detections, regions=regions)
        if detections
        else {}
    )
    min_count = max(1, int(min_detections_per_region))
    kept: list[ColorRegion] = []
    filtered_map = np.full_like(label_map, -1)
    for new_id, region in enumerate(regions):
        region_mask = label_map == region.region_id
        if not _region_contains_text_or_icon(region_mask, text_icon_boxes):
            continue
        if detections and det_counts.get(region.region_id, 0) < min_count:
            continue
        kept.append(
            ColorRegion(
                region_id=new_id,
                bbox=region.bbox,
                mean_color=region.mean_color,
                area=region.area,
            )
        )
        filtered_map[region_mask] = new_id
    return kept, filtered_map


def segment_image_by_color(
    image: Image.Image,
    params: ColorSegmentParams | None = None,
    *,
    image_path: Path | None = None,
    run_dir: Path | None = None,
    detections: list[SegmentDetection] | None = None,
) -> ColorSegmentResult:
    """Segment an image into large color regions using the spatial SLIC method."""
    p = params or ColorSegmentParams()
    rgb = np.asarray(image.convert("RGB"))
    work, masked_boxes, mask_boxes, text_icon_boxes, masked_before_blur, resolved = (
        prepare_segmentation_image(
            rgb,
            p,
            image_path=image_path,
            run_dir=run_dir,
            detections=detections,
        )
    )
    regions, label_map, quantized = segment_image_by_spatial(rgb, work, p)

    regions_before_yolo_filter = len(regions)
    if p.require_yolo_objects and text_icon_boxes:
        regions, label_map = _filter_regions_with_text_icons(
            regions,
            label_map,
            text_icon_boxes,
            detections=resolved if resolved else None,
            min_detections_per_region=2 if resolved else 1,
        )
        quantized = _quantized_from_label_map(rgb, label_map, regions)

    regions.sort(key=lambda region: region.area, reverse=True)
    final_label_map = np.full_like(label_map, -1)
    final_regions: list[ColorRegion] = []
    for new_id, region in enumerate(regions):
        old_id = region.region_id
        final_label_map[label_map == old_id] = new_id
        final_regions.append(
            ColorRegion(
                region_id=new_id,
                bbox=region.bbox,
                mean_color=region.mean_color,
                area=region.area,
            )
        )
    return ColorSegmentResult(
        regions=final_regions,
        quantized=quantized,
        label_map=final_label_map,
        masked_box_count=masked_boxes,
        regions_before_yolo_filter=regions_before_yolo_filter,
        prepared=Image.fromarray(masked_before_blur.astype(np.uint8), mode="RGB"),
        mask_boxes=list(mask_boxes),
        detections=list(resolved),
    )
