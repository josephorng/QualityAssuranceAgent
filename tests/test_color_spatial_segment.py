from __future__ import annotations

import numpy as np
from PIL import Image

from cua_mcp.yolo_onnx import YOLO_CLASS_INPUT
from cua_mcp.color_spatial_segment import (
    ColorRegion,
    ColorSegmentParams,
    ColorSegmentResult,
    SegmentDetection,
    _build_immediate_parent_map,
    _filter_regions_with_text_icons,
    _opencv_ximgproc_slic,
    _region_mostly_inside,
    _region_ranks_for_landmark,
    _region_tree_distance,
    _spatial_region_rank_for_detection,
    color_segment_to_json_dict,
    landmark_region_id_for_box,
    load_color_segment_params,
    region_id_for_box,
    reorder_detections_for_landmark,
    segment_image_by_color,
    spatial_region_rank_for_detections,
)


def _split_region_result() -> ColorSegmentResult:
    label_map = np.zeros((100, 200), dtype=np.int32)
    label_map[:, :100] = 0
    label_map[:, 100:] = 1
    regions = [
        ColorRegion(region_id=0, bbox=(0, 0, 99, 99), mean_color=(40, 80, 120), area=10000),
        ColorRegion(region_id=1, bbox=(100, 0, 199, 99), mean_color=(40, 80, 120), area=10000),
    ]
    return ColorSegmentResult(
        regions=regions,
        quantized=Image.new("RGB", (200, 100)),
        label_map=label_map,
    )


def test_spatial_region_rank_same_and_nearby_regions() -> None:
    result = _split_region_result()
    landmark_box = (10, 10, 30, 30)
    detections = [
        SegmentDetection((50, 10, 20, 20), 0, "text", "left"),
        SegmentDetection((110, 10, 20, 20), 0, "text", "right"),
    ]
    ranks = spatial_region_rank_for_detections(landmark_box, result, detections)
    assert ranks[(50, 10, 20, 20)] == 0
    assert ranks[(110, 10, 20, 20)] == 2


def test_reorder_detections_for_landmark_prefers_same_region() -> None:
    result = _split_region_result()
    landmark_box = (10, 10, 30, 30)
    items = ["right", "left"]
    boxes = {
        "right": (110, 10, 20, 20),
        "left": (50, 10, 20, 20),
    }
    ordered = reorder_detections_for_landmark(
        landmark_box,
        result,
        items,
        lambda label: boxes[label],
    )
    assert ordered == ["left", "right"]


def test_region_id_for_box_prefers_smallest_overlapping_region() -> None:
    label_map = np.zeros((100, 100), dtype=np.int32)
    label_map[40:60, 40:60] = 1
    # Center (50,50) is in region 1, but a wide box also overlaps large region 0.
    assert region_id_for_box(label_map, (35, 48, 30, 4)) == 1
    # Nested box fully inside small region.
    assert region_id_for_box(label_map, (42, 42, 16, 16)) == 1


def test_filter_regions_drops_single_detection_regions() -> None:
    label_map = np.zeros((100, 200), dtype=np.int32)
    label_map[:, :100] = 0
    label_map[:, 100:] = 1
    regions = [
        ColorRegion(region_id=0, bbox=(0, 0, 99, 99), mean_color=(40, 80, 120), area=10000),
        ColorRegion(region_id=1, bbox=(100, 0, 199, 99), mean_color=(40, 80, 120), area=10000),
    ]
    text_icon_boxes = [(10, 10, 20, 20), (50, 10, 20, 20), (110, 10, 20, 20)]
    detections = [
        SegmentDetection((10, 10, 20, 20), 0, "text", "a"),
        SegmentDetection((50, 10, 20, 20), 0, "text", "b"),
        SegmentDetection((110, 10, 20, 20), 0, "text", "c"),
    ]
    kept, filtered_map = _filter_regions_with_text_icons(
        regions,
        label_map,
        text_icon_boxes,
        detections=detections,
        min_detections_per_region=2,
    )
    assert len(kept) == 1
    assert kept[0].bbox == (0, 0, 99, 99)
    assert np.all(filtered_map[:, :100] == 0)
    assert np.all(filtered_map[:, 100:] == -1)


def test_filter_regions_ignores_non_text_icon_detections_for_count() -> None:
    label_map = np.zeros((100, 200), dtype=np.int32)
    label_map[:, :100] = 0
    label_map[:, 100:] = 1
    regions = [
        ColorRegion(region_id=0, bbox=(0, 0, 99, 99), mean_color=(40, 80, 120), area=10000),
        ColorRegion(region_id=1, bbox=(100, 0, 199, 99), mean_color=(40, 80, 120), area=10000),
    ]
    text_icon_boxes = [(10, 10, 20, 20), (110, 10, 20, 20), (110, 40, 20, 20)]
    detections = [
        SegmentDetection((10, 10, 20, 20), 0, "text", "a"),
        SegmentDetection((50, 10, 20, 20), 0, "text", "b"),
        SegmentDetection((110, 10, 20, 20), 0, "text", "c"),
        SegmentDetection((110, 40, 20, 20), YOLO_CLASS_INPUT, "input"),
    ]
    kept, filtered_map = _filter_regions_with_text_icons(
        regions,
        label_map,
        text_icon_boxes,
        detections=detections,
        min_detections_per_region=2,
    )
    assert len(kept) == 1
    assert kept[0].bbox == (0, 0, 99, 99)
    assert np.all(filtered_map[:, :100] == 0)
    assert np.all(filtered_map[:, 100:] == -1)


def test_load_color_segment_params_defaults_when_missing(tmp_path) -> None:
    params = load_color_segment_params(tmp_path / "missing.json")
    assert isinstance(params, ColorSegmentParams)
    assert params.num_colors == 120


def _nested_bac_regions() -> list[ColorRegion]:
    """Outer B, middle A, inner C (B ⊃ A ⊃ C)."""
    return [
        ColorRegion(region_id=0, bbox=(0, 0, 99, 99), mean_color=(40, 80, 120), area=10000),
        ColorRegion(region_id=1, bbox=(10, 10, 89, 89), mean_color=(50, 90, 130), area=6400),
        ColorRegion(region_id=2, bbox=(20, 20, 79, 79), mean_color=(60, 100, 140), area=3600),
    ]


def _nested_eab_regions() -> list[ColorRegion]:
    """Outer E, middle A, inner B (E ⊃ A ⊃ B)."""
    return [
        ColorRegion(region_id=0, bbox=(0, 0, 99, 99), mean_color=(40, 80, 120), area=10000),
        ColorRegion(region_id=1, bbox=(10, 10, 89, 89), mean_color=(50, 90, 130), area=6400),
        ColorRegion(region_id=2, bbox=(20, 20, 79, 79), mean_color=(60, 100, 140), area=3600),
    ]


def test_region_mostly_inside_and_parent_map_nested_bac() -> None:
    regions = _nested_bac_regions()
    b, a, c = regions
    assert _region_mostly_inside(c, a)
    assert _region_mostly_inside(a, b)
    assert not _region_mostly_inside(b, a)

    parent_map = _build_immediate_parent_map(regions)
    assert parent_map[2] == 1
    assert parent_map[1] == 0
    assert parent_map[0] is None


def test_region_ranks_nested_bac_landmarks() -> None:
    regions = _nested_bac_regions()
    parent_map = _build_immediate_parent_map(regions)
    landmark_box = (50, 50, 4, 4)

    ranks_b = _region_ranks_for_landmark(0, regions, parent_map, landmark_box)
    assert ranks_b == {0: 0, 1: 1, 2: 2}

    ranks_a = _region_ranks_for_landmark(1, regions, parent_map, landmark_box)
    assert ranks_a == {1: 0, 0: 1, 2: 1}

    ranks_c = _region_ranks_for_landmark(2, regions, parent_map, landmark_box)
    assert ranks_c == {2: 0, 1: 1, 0: 2}


def test_region_ranks_nested_eab_landmark_in_inner_b() -> None:
    regions = _nested_eab_regions()
    parent_map = _build_immediate_parent_map(regions)
    # region ids: E=0, A=1, B=2
    ranks = _region_ranks_for_landmark(2, regions, parent_map, (50, 50, 4, 4))
    assert ranks == {2: 0, 1: 1, 0: 2}


def test_region_tree_distance_virtual_root_siblings() -> None:
    """Top-level siblings connect through the virtual root (distance 2)."""
    regions = _split_region_result().regions
    parent_map = _build_immediate_parent_map(regions)
    assert parent_map[0] is None
    assert parent_map[1] is None
    assert _region_tree_distance(parent_map, 0, 0) == 0
    assert _region_tree_distance(parent_map, 0, 1) == 2
    assert _region_tree_distance(parent_map, 1, 0) == 2


def test_spatial_region_rank_nested_bac() -> None:
    regions = _nested_bac_regions()
    label_map = np.zeros((100, 100), dtype=np.int32)
    label_map[10:90, 10:90] = 1
    label_map[20:80, 20:80] = 2
    result = ColorSegmentResult(
        regions=regions,
        quantized=Image.new("RGB", (100, 100)),
        label_map=label_map,
    )
    detections = [
        SegmentDetection((2, 2, 6, 6), 0, "text", "in_b"),
        SegmentDetection((15, 15, 10, 10), 0, "text", "in_a"),
        SegmentDetection((25, 25, 10, 10), 0, "text", "in_c"),
    ]
    ranks = spatial_region_rank_for_detections((25, 25, 10, 10), result, detections)
    assert ranks[(25, 25, 10, 10)] == 0
    assert ranks[(15, 15, 10, 10)] == 1
    assert ranks[(2, 2, 6, 6)] == 2


def test_color_segment_to_json_dict_includes_regions() -> None:
    regions = _nested_bac_regions()
    label_map = np.zeros((100, 100), dtype=np.int32)
    label_map[10:90, 10:90] = 1
    label_map[20:80, 20:80] = 2
    result = ColorSegmentResult(
        regions=regions,
        quantized=Image.new("RGB", (100, 100)),
        label_map=label_map,
        regions_before_yolo_filter=3,
    )
    payload = color_segment_to_json_dict(result, landmark_box=(25, 25, 10, 10))
    assert payload["region_count"] == 3
    assert payload["regions_before_yolo_filter"] == 3
    assert payload["landmark_region_id"] == 2
    assert len(payload["regions"]) == 3
    region0 = payload["regions"][0]
    assert region0["region_id"] == 0
    assert region0["bbox"] == [0, 0, 99, 99]
    assert region0["mean_color"] == [40, 80, 120]
    assert region0["area"] == 10000
    assert region0["parent_region_id"] is None
    assert region0["spatial_region_rank"] == 2
    region2 = payload["regions"][2]
    assert region2["region_id"] == 2
    assert region2["parent_region_id"] == 1
    assert region2["spatial_region_rank"] == 0


def test_landmark_region_id_prefers_cursor_over_wide_box_overlap() -> None:
    """Cursor on inner panel should beat smallest overlap from a huge landmark box."""
    label_map = np.zeros((100, 200), dtype=np.int32)
    label_map[:, :] = 1
    label_map[40:60, 40:60] = 0
    regions = [
        ColorRegion(region_id=0, bbox=(40, 40, 59, 59), mean_color=(60, 100, 140), area=400),
        ColorRegion(region_id=1, bbox=(0, 0, 199, 99), mean_color=(40, 80, 120), area=20000),
    ]
    # Wide box spans both regions; box center sits in outer region #1.
    wide_box = (0, 0, 200, 100)
    assert landmark_region_id_for_box(label_map, wide_box, regions=regions) == 1
    # Cursor in inner region — should resolve to #0 even with wide box.
    assert landmark_region_id_for_box(
        label_map,
        wide_box,
        cursor_xy=(50, 50),
        regions=regions,
    ) == 0


def test_region_id_for_box_nearest_fallback_when_unlabeled() -> None:
    label_map = np.full((50, 50), -1, dtype=np.int32)
    label_map[10:20, 10:20] = 0
    regions = [
        ColorRegion(region_id=0, bbox=(10, 10, 19, 19), mean_color=(40, 80, 120), area=100),
    ]
    # Box center is unlabeled but nearest region #0 should be returned.
    assert region_id_for_box(label_map, (30, 30, 4, 4), regions=regions) == 0


def test_region_id_for_box_prefers_bbox_containing_center_over_nearest() -> None:
    """Unlabeled taskbar text should map to taskbar region bbox, not nearer guest bar."""
    label_map = np.full((1080, 1920), -1, dtype=np.int32)
    label_map[901:944, 336:493] = 8
    regions = [
        ColorRegion(
            region_id=1,
            bbox=(0, 0, 1919, 1029),
            mean_color=(20, 20, 20),
            area=200000,
        ),
        ColorRegion(
            region_id=3,
            bbox=(0, 1035, 1919, 1079),
            mean_color=(219, 219, 231),
            area=60000,
        ),
        ColorRegion(
            region_id=8,
            bbox=(336, 901, 492, 943),
            mean_color=(232, 235, 236),
            area=3866,
        ),
    ]
    search_box = (580, 1049, 34, 17)
    assert region_id_for_box(label_map, search_box, regions=regions) == 3

    guest_box = (394, 913, 77, 19)
    assert region_id_for_box(label_map, guest_box, regions=regions) == 8


def test_spatial_region_rank_taskbar_search_not_same_as_guest_bar() -> None:
    label_map = np.full((1080, 1920), -1, dtype=np.int32)
    label_map[901:944, 336:493] = 8
    regions = [
        ColorRegion(region_id=1, bbox=(0, 0, 1919, 1029), mean_color=(20, 20, 20), area=200000),
        ColorRegion(region_id=3, bbox=(0, 1035, 1919, 1079), mean_color=(219, 219, 231), area=60000),
        ColorRegion(region_id=8, bbox=(336, 901, 492, 943), mean_color=(232, 235, 236), area=3866),
    ]
    result = ColorSegmentResult(
        regions=regions,
        quantized=Image.new("RGB", (1920, 1080)),
        label_map=label_map,
    )
    guest_det = SegmentDetection((394, 913, 77, 19), 0, "text", "guest")
    search_det = SegmentDetection((580, 1049, 34, 17), 0, "text", "search")
    ranks = spatial_region_rank_for_detections(
        (394, 913, 77, 19),
        result,
        [guest_det, search_det],
        cursor_xy=(444, 919),
    )
    assert ranks[tuple(guest_det.box)] == 0
    assert ranks[tuple(search_det.box)] == 3


def test_spatial_region_rank_uses_cursor_landmark_region() -> None:
    label_map = np.zeros((100, 200), dtype=np.int32)
    label_map[:, :] = 1
    label_map[40:60, 40:60] = 0
    regions = [
        ColorRegion(region_id=0, bbox=(40, 40, 59, 59), mean_color=(60, 100, 140), area=400),
        ColorRegion(region_id=1, bbox=(0, 0, 199, 99), mean_color=(40, 80, 120), area=20000),
    ]
    result = ColorSegmentResult(
        regions=regions,
        quantized=Image.new("RGB", (200, 100)),
        label_map=label_map,
    )
    wide_box = (0, 0, 200, 100)
    inner_det = SegmentDetection((42, 42, 16, 16), 0, "text", "inner")
    outer_det = SegmentDetection((5, 5, 10, 10), 0, "text", "outer")
    ranks_inner_landmark = spatial_region_rank_for_detections(
        wide_box,
        result,
        [inner_det, outer_det],
        cursor_xy=(50, 50),
    )
    assert ranks_inner_landmark[tuple(inner_det.box)] == 0
    assert ranks_inner_landmark[tuple(outer_det.box)] == 1
    ranks_outer_landmark = spatial_region_rank_for_detections(
        wide_box,
        result,
        [inner_det, outer_det],
        cursor_xy=(10, 10),
    )
    assert ranks_outer_landmark[tuple(outer_det.box)] == 0
    assert ranks_outer_landmark[tuple(inner_det.box)] == 1


def test_spatial_region_rank_sibling_branch_uses_landmark_ancestor_bbox() -> None:
    """Toggle on white panel (sibling branch) ranks by landmark ancestor, not leaf distance."""
    label_map = np.zeros((200, 200), dtype=np.int32)
    label_map[:, :] = 1
    label_map[50:150, 50:150] = 0
    label_map[160:180, 60:120] = 8
    regions = [
        ColorRegion(region_id=1, bbox=(0, 0, 199, 199), mean_color=(40, 80, 120), area=20000),
        ColorRegion(region_id=0, bbox=(50, 50, 149, 149), mean_color=(250, 250, 250), area=25000),
        ColorRegion(region_id=8, bbox=(60, 160, 119, 179), mean_color=(230, 230, 230), area=1200),
    ]
    result = ColorSegmentResult(
        regions=regions,
        quantized=Image.new("RGB", (200, 200)),
        label_map=label_map,
    )
    parent_map = _build_immediate_parent_map(regions)
    assert parent_map[8] == 1
    assert parent_map[0] is None

    guest_det = SegmentDetection((65, 165, 50, 10), 0, "text", "guest")
    toggle_det = SegmentDetection((90, 90, 40, 10), 0, "text", "toggle")
    ranks = spatial_region_rank_for_detections(
        (65, 165, 50, 10),
        result,
        [guest_det, toggle_det],
        cursor_xy=(90, 170),
    )
    assert ranks[tuple(guest_det.box)] == 0
    assert ranks[tuple(toggle_det.box)] == 1
    assert _spatial_region_rank_for_detection(
        8,
        0,
        toggle_det.box,
        parent_map,
        {r.region_id: r for r in regions},
    ) == 1


def test_segment_image_by_color_uses_opencv_slic() -> None:
    _opencv_ximgproc_slic()
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[:, :60] = (40, 80, 120)
    rgb[:, 60:] = (200, 40, 40)
    image = Image.fromarray(rgb, mode="RGB")
    params = ColorSegmentParams(
        num_colors=8,
        min_area_frac=0.01,
        mask_text_icons=False,
        require_yolo_objects=False,
        merge_superpixels=False,
        split_large_regions=False,
    )
    result = segment_image_by_color(image, params)
    assert result.regions
    assert result.label_map.shape == (80, 120)
    assert int(result.label_map.max()) >= 0
