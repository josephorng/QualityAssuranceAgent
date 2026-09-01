from __future__ import annotations

import os
import re
import shutil
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from cua_mcp.color_spatial_segment import (
    ColorRegion,
    ColorSegmentParams,
    ColorSegmentResult,
    SegmentDetection,
    prepare_segmentation_image,
    region_id_for_box,
    segment_image_by_color,
    spatial_region_rank_for_detections,
)
from cua_mcp.icon_map import is_pua_char, lookup_pua_icon, text_has_pua, unknown_icon_record
from cua_mcp.input_box_rectangles import (
    LineSegmentParams,
    detect_horizontal_rectangles,
)
from cua_mcp.read_screen_text.get_coordinates import ocr_regions_from_image_path
from cua_mcp.select_mouse_target import (
    _dedupe_overlapping_detections,
    _detect_mouse_targets_from_bgr,
    _detection_from_bbox,
)
from cua_mcp.select_ui_element import UiDetection, _format_ui_candidates_text
from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    MOUSE_TARGET_CLASS_IDS,
    OCR_DETECTION_CLASS_IDS,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_INPUT,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_SCROLLBAR,
    YOLO_CLASS_TEXT,
    run_yolo_onnx_end2end,
)
from src.common.io_utils import imread_bgr, read_json, write_json
from src.common.settings import ROOT_DIR, resolve_recordings_dir, resolve_runs_dir

YOLO_UNDONE_IMAGES = Path(
    r"C:\Users\Joseph Hung\Documents\Repos\Git\YOLO\real_screenshot\undone\images"
)
OCR_EXPORT_DEFAULT_DIR = Path(
    r"C:\Users\Joseph Hung\Documents\Repos\Git\OCR\data\train\cua_data"
)
OCR_EXPORT_ICONS_DIR = Path(
    r"C:\Users\Joseph Hung\Documents\Repos\Git\OCR\data\train\elements"
)
UI_EXPORT_DEFAULT_DIR = Path(
    r"C:\Users\Joseph Hung\Documents\Repos\Git\OCR\data\train\elements"
)
OCR_VALIDATE_DIR = Path(
    r"C:\Users\Joseph Hung\Documents\Repos\Git\OCR\data\validate\cua_data"
)
# Same weights as ONNX export used by OCR (`cua_mcp/yolo_onnx.DEFAULT_YOLO_ONNX_PATH`).
DEFAULT_ULTRALYTICS_PT_PATH = ROOT_DIR / "cua_mcp" / "best.pt"
DEFAULT_TEST_IMAGES_DIR = ROOT_DIR / "test_images"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Tk widgets and canvas overlay labels for the test-images tab (default Tk ~9pt on Windows).
UI_FONT_SIZE = 12
OVERLAY_FONT_SIZE = 15

BOX_EDIT_STEP = 1
BOX_EDIT_STEP_SHIFT = 8
MIN_BOX_SIZE = 2


def toggle_box_edit_mode(mode_var: tk.StringVar, status_var: tk.StringVar) -> None:
    mode_var.set("shrink" if mode_var.get() == "expand" else "expand")
    label = "Expand" if mode_var.get() == "expand" else "Shrink"
    status_var.set(f"Box edit mode: {label}")


@dataclass(frozen=True)
class OcrLine:
    box: tuple[int, int, int, int]
    text: str
    line_type: str = "ocr"  # "ocr" | "element" | "ocr_icon" | …
    class_name: str = ""
    class_id: int | None = None
    chinese_ids: tuple[str, ...] = ()


def _is_ui_object_line(line: OcrLine) -> bool:
    return line.line_type != "ocr"


def _unknown_icon_label() -> str:
    return str(unknown_icon_record().get("chinese_id", "未知圖示"))


def _icon_label_for_pua(char: str) -> str:
    mapped = lookup_pua_icon(char)
    if mapped is None:
        return _unknown_icon_label()
    chinese_id = str(mapped.get("chinese_id", "")).strip()
    if chinese_id:
        return chinese_id
    icon_id = str(mapped.get("id", "")).strip()
    if icon_id and icon_id != "unknown_icon":
        return icon_id.replace("_", " ")
    return _unknown_icon_label()


def _icon_label_for_record(record: dict[str, Any]) -> str:
    chinese_id = str(record.get("chinese_id", "")).strip()
    if chinese_id:
        return chinese_id
    pua = record.get("pua")
    if isinstance(pua, str) and pua:
        return _icon_label_for_pua(pua)
    return _unknown_icon_label()


def _icon_labels_for_text(text: str) -> list[str]:
    return [_icon_label_for_pua(ch) for ch in text or "" if is_pua_char(ch)]


def _display_text_with_icon_labels(text: str) -> str:
    if not text:
        return ""
    if not text_has_pua(text):
        return text.strip()
    labels = _icon_labels_for_text(text)
    non_icon = "".join(ch for ch in text if not is_pua_char(ch)).strip()
    if labels and non_icon:
        return f"{', '.join(labels)} + {non_icon}"
    if labels:
        return ", ".join(labels)
    return non_icon


def _resolved_ui_ids(line: OcrLine) -> list[str]:
    if line.chinese_ids:
        return list(line.chinese_ids)
    if line.text:
        return _icon_labels_for_text(line.text)
    return []


def _ui_object_label(line: OcrLine) -> str:
    ids = _resolved_ui_ids(line)
    if ids:
        name = ", ".join(ids)
    elif line.text.strip():
        name = _display_text_with_icon_labels(line.text) or line.text.strip()
    else:
        name = line.class_name or line.line_type or "ui"
        if line.class_id is not None:
            name = f"{name} (class_id={line.class_id})"
    x, y, w, h = line.box
    return f"{name} @ ({x},{y}) {w}×{h}"


def _display_label_for_line(line: OcrLine) -> str:
    if _is_ui_object_line(line):
        return _ui_object_label(line)
    mapped = _display_text_with_icon_labels(line.text)
    return mapped if mapped else "<empty>"


_CLASS_NAME_TO_ID: dict[str, int] = {name: cid for cid, name in YOLO_CLASS_NAMES.items()}


def _class_id_for_line(line: OcrLine) -> int:
    if isinstance(line.class_id, int):
        return line.class_id
    name = (line.class_name or "").strip().lower()
    if name == "scrollbar_original":
        return YOLO_CLASS_SCROLLBAR
    if name == "input_original":
        return YOLO_CLASS_INPUT
    if name in _CLASS_NAME_TO_ID:
        return _CLASS_NAME_TO_ID[name]
    if line.line_type == "ocr":
        return YOLO_CLASS_TEXT
    return YOLO_CLASS_ELEMENT


def _ocr_line_to_ui_detection(line: OcrLine) -> UiDetection:
    """Map a viewer line to ``UiDetection`` for the agent candidate formatter."""
    class_id = _class_id_for_line(line)
    icons = (
        [{"chinese_id": cid} for cid in line.chinese_ids if cid]
        if line.chinese_ids
        else None
    )
    text = (line.text or "").strip() or None
    det = _detection_from_bbox(line.box, class_id, text=text, icons=icons)
    name = (line.class_name or "").strip()
    if name and name != det.class_name:
        return UiDetection(
            bbox=det.bbox,
            cx=det.cx,
            cy=det.cy,
            class_id=det.class_id,
            class_name=name,
            text=det.text,
            icons=det.icons,
        )
    return det


def _agent_format_detection_rows(lines: list[OcrLine]) -> list[str]:
    """Same rows as the agent LLM candidate list (``_format_ui_candidates_text``)."""
    if not lines:
        return []
    detections = [_ocr_line_to_ui_detection(line) for line in lines]
    text = _format_ui_candidates_text(detections, include_geometry=True)
    return text.split("\n") if text else []


_SPATIAL_RANK_UNASSIGNED = 10_000


@dataclass(frozen=True)
class YoloSpatialSegmentState:
    result: ColorSegmentResult | None
    spatial_ranks: dict[tuple[int, int, int, int], int]


def _ocr_line_to_segment_detection(line: OcrLine) -> SegmentDetection:
    class_id = _class_id_for_line(line)
    name = (line.class_name or "").strip()
    if not name:
        name = YOLO_CLASS_NAMES.get(class_id, "")
    return SegmentDetection(
        box=tuple(int(v) for v in line.box),
        class_id=class_id,
        class_name=name,
        text=(line.text or "").strip(),
    )


def _lines_for_spatial_segmentation(lines: list[OcrLine]) -> list[OcrLine]:
    return [
        line
        for line in lines
        if not _is_scrollbar_original_line(line) and not _is_input_original_line(line)
    ]


def _format_color_region_row(region: ColorRegion) -> str:
    x0, y0, x1, y1 = region.bbox
    r, g, b = region.mean_color
    return (
        f"#{region.region_id + 1} ({x0},{y0})-({x1},{y1}) "
        f"area={region.area}px rgb=({r},{g},{b})"
    )


def _format_spatial_rank_label(rank: int) -> str:
    if rank == _SPATIAL_RANK_UNASSIGNED:
        return "unassigned"
    return str(rank)


def _build_yolo_spatial_segment_state(
    image: Image.Image | None,
    lines: list[OcrLine],
) -> YoloSpatialSegmentState:
    if image is None:
        return YoloSpatialSegmentState(None, {})
    try:
        params = _load_color_segment_params()
        seg_lines = _lines_for_spatial_segmentation(lines)
        seg_dets = [_ocr_line_to_segment_detection(line) for line in seg_lines]
        result = segment_image_by_color(image, params=params, detections=seg_dets)
        spatial_ranks: dict[tuple[int, int, int, int], int] = {}
        if seg_lines:
            landmark_box = tuple(int(v) for v in seg_lines[0].box)
            bx, by, bw, bh = landmark_box
            spatial_ranks = spatial_region_rank_for_detections(
                landmark_box,
                result,
                seg_dets,
                cursor_xy=(bx + bw // 2, by + bh // 2),
            )
        return YoloSpatialSegmentState(result, spatial_ranks)
    except Exception:
        return YoloSpatialSegmentState(None, {})


def _yolo_detection_list_rows(
    lines: list[OcrLine],
    *,
    segment_result: ColorSegmentResult | None = None,
    spatial_ranks: dict[tuple[int, int, int, int], int] | None = None,
) -> list[str]:
    base_rows = _agent_format_detection_rows(lines)
    if segment_result is None:
        return base_rows
    rows: list[str] = []
    for line, base in zip(lines, base_rows):
        box = tuple(int(v) for v in line.box)
        rid = region_id_for_box(
            segment_result.label_map,
            box,
            regions=segment_result.regions,
        )
        region_label = f"#{rid + 1}" if rid is not None else "unassigned"
        rank_label = "—"
        if spatial_ranks is not None:
            rank_label = _format_spatial_rank_label(
                spatial_ranks.get(box, _SPATIAL_RANK_UNASSIGNED)
            )
        index_close = base.find("]")
        if index_close > 0 and base.startswith("[index "):
            row = (
                f"{base[:index_close]}, region {region_label}, rank={rank_label}"
                f"{base[index_close:]}"
            )
        else:
            row = f"{base} region {region_label} rank={rank_label}"
        rows.append(row)
    return rows


def _is_scrollbar_original_line(line: OcrLine) -> bool:
    return (line.class_name or "").strip().lower() == "scrollbar_original"


def _is_input_original_line(line: OcrLine) -> bool:
    return (line.class_name or "").strip().lower() == "input_original"


def _split_debug_original_lines(
    lines: list[OcrLine],
) -> tuple[list[OcrLine], list[OcrLine], list[OcrLine]]:
    main: list[OcrLine] = []
    scrollbar_originals: list[OcrLine] = []
    input_originals: list[OcrLine] = []
    for line in lines:
        if _is_scrollbar_original_line(line):
            scrollbar_originals.append(line)
        elif _is_input_original_line(line):
            input_originals.append(line)
        else:
            main.append(line)
    return main, scrollbar_originals, input_originals


def _dedupe_ocr_lines(lines: list[OcrLine]) -> list[OcrLine]:
    """Drop heavily overlapping same-content boxes (agent ``_dedupe_overlapping_detections``).

    ``scrollbar_original`` / ``input_original`` debug boxes are kept aside and
    appended unchanged so they are not removed as duplicates of the fitted box.
    """
    main, scrollbar_originals, input_originals = _split_debug_original_lines(lines)
    if len(main) < 2:
        return [*main, *scrollbar_originals, *input_originals]
    detections = [_ocr_line_to_ui_detection(line) for line in main]
    kept = _dedupe_overlapping_detections(detections)
    by_id = {id(det): idx for idx, det in enumerate(detections)}
    deduped = [main[by_id[id(det)]] for det in kept]
    return [*deduped, *scrollbar_originals, *input_originals]


def _is_icon_ocr_line(line: OcrLine) -> bool:
    return line.line_type == "ocr" and text_has_pua(line.text)


def _is_pua_icon_identity_text(text: str) -> bool:
    """True when ``text`` is PUA-only (one or more icons, no other characters).

    Matches ``_detect_mouse_targets_from_bgr`` / ``_text_is_pua_only``: multi-icon
    crops stay icon identity, not mixed text.
    """
    if not text or not text_has_pua(text):
        return False
    non_pua = "".join(ch for ch in text if not is_pua_char(ch)).strip()
    return not non_pua


def _export_dest_for_icon_identity(is_icon: bool) -> Path:
    """Train export folder for explicit icon vs text identity."""
    return OCR_EXPORT_ICONS_DIR if is_icon else OCR_EXPORT_DEFAULT_DIR


def _export_dest_for_text(text: str) -> Path:
    """Train export folder: PUA-only icons → ``icons``; pure text or mixed PUA+text → ``cua_data``."""
    return _export_dest_for_icon_identity(_is_pua_icon_identity_text(text))


def _undo_export_files(paths: list[Path]) -> int:
    """Delete files from a prior export. Returns the number of files removed."""
    removed = 0
    for path in paths:
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


_STRING_LINE_RE = re.compile(r"^\[(\d+),(\d+),(\d+),(\d+)\]\s*(.*)$")
_STRING_CENTER_RE = re.compile(r"^\[(\d+),(\d+)\]\s*(.*)$")


def _parse_conf_0_to_1(raw: str) -> tuple[float | None, str | None]:
    """Parse confidence in ``[0, 1]``; returns ``(value, None)`` or ``(None, error_message)``."""
    s = raw.strip()
    if not s:
        return None, "confidence is empty"
    try:
        v = float(s)
    except ValueError:
        return None, "confidence must be a number (e.g. 0.25)"
    if not (0.0 <= v <= 1.0):
        return None, "confidence must be between 0 and 1"
    return v, None


def _parse_string_line(line: str) -> OcrLine | None:
    raw = line.strip()
    match = _STRING_LINE_RE.match(raw)
    if match:
        x, y, w, h = (int(match.group(i)) for i in range(1, 5))
        return OcrLine(box=(x, y, w, h), text=match.group(5).strip())
    center_match = _STRING_CENTER_RE.match(raw)
    if center_match:
        cx, cy = (int(center_match.group(i)) for i in range(1, 3))
        # format_coordinate_text_from_regions() uses center points; render with a tiny marker box.
        x = max(0, cx - 2)
        y = max(0, cy - 2)
        return OcrLine(box=(x, y, 4, 4), text=center_match.group(3).strip())
    return None


def _normalize_lines(raw_lines: Any) -> list[OcrLine]:
    if not isinstance(raw_lines, list):
        return []
    parsed: list[OcrLine] = []
    for item in raw_lines:
        if isinstance(item, str):
            line = _parse_string_line(item)
            if line is not None:
                parsed.append(line)
            continue
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and isinstance(item[0], (list, tuple))
            and len(item[0]) == 4
            and isinstance(item[1], (list, tuple))
            and len(item[1]) == 2
        ):
            try:
                x, y, w, h = (int(v) for v in item[0])
            except (TypeError, ValueError):
                continue
            preds = item[2]
            if isinstance(preds, list):
                text = "".join(str(p) for p in preds).strip()
            else:
                text = str(preds).strip()
            parsed.append(OcrLine(box=(x, y, w, h), text=text))
            continue
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (list, tuple))
            and len(item[0]) == 4
        ):
            try:
                x, y, w, h = (int(v) for v in item[0])
            except (TypeError, ValueError):
                continue
            text = str(item[1]).strip()
            parsed.append(OcrLine(box=(x, y, w, h), text=text))
            continue
        if isinstance(item, dict):
            box = item.get("box") or item.get("bbox") or item.get("rect")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    x, y, w, h = (int(v) for v in box)
                except (TypeError, ValueError):
                    continue
                class_name = str(item.get("class_name", "")).strip()
                class_id = item.get("class_id")
                line_type = (
                    str(item.get("line_type", "")).strip()
                    or ("ocr" if class_name == "text" else "element" if class_name else "ocr")
                )
                parsed.append(
                    OcrLine(
                        box=(x, y, w, h),
                        text=str(item.get("text", "")).strip(),
                        line_type=line_type,
                        class_name=class_name,
                        class_id=int(class_id) if isinstance(class_id, int) else None,
                    )
                )
    return parsed


def copy_image_and_ocr_json_to_dir(
    image_path: Path,
    lines: list[OcrLine],
    dest_dir: Path,
) -> tuple[Path, Path]:
    """Copy ``image_path`` and a sidecar OCR JSON (from ``lines``) into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_img = dest_dir / image_path.name
    shutil.copy2(image_path, dest_img)
    dest_json = dest_dir / image_path.with_suffix(".json").name

    existing: dict[str, Any] = {}
    json_src = image_path.with_suffix(".json")
    if json_src.exists():
        raw = read_json(json_src, default={})
        if isinstance(raw, dict):
            existing = raw

    line_pairs = [[list(line.box), line.text] for line in lines]
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": str(dest_img),
        "image_name": dest_img.name,
        "line_height": existing.get("line_height", 32),
        "lines": line_pairs,
        "text": "\n".join(line.text for line in lines),
    }
    for key in ("yolo_elapsed_ms", "ocr_elapsed_ms", "source"):
        if key in existing:
            payload[key] = existing[key]
    write_json(dest_json, payload)
    return dest_img, dest_json


def _resolve_run_image_path(run_dir: Path, raw_path: str) -> Path | None:
    """Resolve a screenshot path from a run manifest or yolo_ocr JSON field."""
    if not raw_path.strip():
        return None
    direct = Path(raw_path)
    if direct.is_file():
        return direct
    by_name = run_dir / "screenshots" / direct.name
    if by_name.is_file():
        return by_name
    from_root = ROOT_DIR / raw_path
    if from_root.is_file():
        return from_root
    return None


def _sidecar_json_path(image_path: Path, run_dir: Path | None = None) -> Path:
    sidecar = image_path.with_suffix(".json")
    if sidecar.is_file():
        return sidecar
    if run_dir is not None:
        yolo_json = run_dir / "yolo_ocr" / f"{image_path.stem}.json"
        if yolo_json.is_file():
            return yolo_json
    return sidecar


def _load_candidate_lines(candidates: Any) -> list[OcrLine]:
    if not isinstance(candidates, list):
        return []
    lines: list[OcrLine] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = (int(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        class_name = str(item.get("class_name", ""))
        text = str(item.get("text") or "").strip()
        icons = item.get("icons")
        icon_labels: tuple[str, ...] = ()
        if isinstance(icons, list):
            icon_labels = tuple(
                str(icon.get("chinese_id", "")).strip()
                for icon in icons
                if isinstance(icon, dict) and str(icon.get("chinese_id", "")).strip()
            )
        class_id = item.get("class_id")
        lines.append(
            OcrLine(
                box=(x, y, w, h),
                text=text,
                line_type="ocr" if class_name == "text" else "element",
                class_name=class_name,
                class_id=int(class_id) if isinstance(class_id, int) else None,
                chinese_ids=icon_labels,
            )
        )
    return lines


def load_ocr_lines(json_path: Path) -> tuple[list[OcrLine], str]:
    if not json_path.exists():
        return [], "Missing OCR JSON"
    try:
        data = read_json(json_path, default={})
    except Exception as exc:
        return [], f"JSON parse error: {exc}"
    if not isinstance(data, dict):
        return [], "Invalid JSON root"
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        lines = _load_candidate_lines(candidates)
        return lines, f"Loaded {len(lines)} vision candidates"
    lines = _normalize_lines(data.get("lines", []))
    return lines, f"Loaded {len(lines)} OCR lines"


def load_yolo_lines(image_path: Path, *, yolo_conf_threshold: float) -> tuple[list[OcrLine], str]:
    bgr = imread_bgr(image_path)
    if bgr is None:
        return [], "Could not read image for YOLO"
    try:
        original_scrollbars: list[tuple[int, int, int, int]] = []
        original_inputs: list[tuple[int, int, int, int]] = []
        candidates = _detect_mouse_targets_from_bgr(
            bgr,
            yolo_conf_threshold=yolo_conf_threshold,
            original_scrollbar_bboxes_out=original_scrollbars,
            original_input_bboxes_out=original_inputs,
        )
    except Exception as exc:
        return [], f"YOLO detect failed: {type(exc).__name__}: {exc}"
    lines: list[OcrLine] = []
    for det in candidates:
        text = det.text or ""
        if det.icons:
            icon_labels = tuple(_icon_label_for_record(i) for i in det.icons)
        else:
            icon_labels = tuple(_icon_labels_for_text(text))
        lines.append(
            OcrLine(
                box=det.bbox,
                text=text,
                line_type="ocr" if det.class_name == "text" else "element",
                class_name=det.class_name,
                class_id=det.class_id,
                chinese_ids=icon_labels,
            )
        )
    # Pre-fit YOLO scrollbar boxes (when arrow-fit changed them) for debug.
    for bbox in original_scrollbars:
        lines.append(
            OcrLine(
                box=bbox,
                text="",
                line_type="element",
                class_name="scrollbar_original",
                class_id=YOLO_CLASS_SCROLLBAR,
                chinese_ids=("scrollbar (original)",),
            )
        )
    for bbox in original_inputs:
        lines.append(
            OcrLine(
                box=bbox,
                text="",
                line_type="element",
                class_name="input_original",
                class_id=YOLO_CLASS_INPUT,
                chinese_ids=("input (original)",),
            )
        )
    return lines, f"Loaded {len(lines)} YOLO detections"


YoloLinesCache = dict[tuple[str, float], tuple[list[OcrLine], str]]


def resolve_image_lines(
    image_path: Path,
    *,
    yolo_conf_threshold: float,
    allow_yolo: bool,
    yolo_cache: YoloLinesCache | None = None,
    force_yolo: bool = False,
    run_dir: Path | None = None,
) -> tuple[list[OcrLine], str]:
    """Load detections from sidecar JSON, else cache or optional live YOLO."""
    if not force_yolo:
        json_path = _sidecar_json_path(image_path, run_dir)
        if json_path.is_file():
            return load_ocr_lines(json_path)

    cache_key = (str(image_path.resolve()), yolo_conf_threshold)
    if not force_yolo and yolo_cache is not None and cache_key in yolo_cache:
        return yolo_cache[cache_key]

    if not allow_yolo:
        return [], "No OCR JSON — click Reload YOLO detections"

    lines, status = load_yolo_lines(image_path, yolo_conf_threshold=yolo_conf_threshold)
    if yolo_cache is not None:
        yolo_cache[cache_key] = (lines, status)
    return lines, status


def _discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.iterdir() if p.is_dir()], reverse=True)


def _discover_run_images(run_dir: Path) -> list[Path]:
    """Images for a run: yolo_ocr/*.png|jpg, screenshots/, or paths from yolo_ocr JSON."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.is_file():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    yolo_dir = run_dir / "yolo_ocr"
    if yolo_dir.is_dir():
        for entry in sorted(yolo_dir.iterdir()):
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg"}:
                add(entry)
            elif suffix == ".json":
                raw = read_json(entry, default={})
                if isinstance(raw, dict):
                    image_raw = raw.get("image_path")
                    if isinstance(image_raw, str):
                        resolved = _resolve_run_image_path(run_dir, image_raw)
                        if resolved is not None:
                            add(resolved)

    shots_dir = run_dir / "screenshots"
    if shots_dir.is_dir():
        for entry in sorted(shots_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                add(entry)

    return sorted(found, key=lambda p: p.name)


def _yolo_ocr_paired_images(run_dir: Path) -> list[Path]:
    return _discover_run_images(run_dir)


# Border outline by YOLO / picker class (selection stays red).
_CLASS_OUTLINE_COLORS: dict[str, str] = {
    "text": "lime",
    "element": "dodgerblue",
    "input": "orange",
    "input_original": "gold",
    "scrollbar": "mediumpurple",
    "scrollbar_original": "violet",
    "unknown": "gray",
}


def _yolo_class_key(line: OcrLine) -> str:
    """Resolve a display class key from ``class_name``, ``class_id``, or ``line_type``."""
    name = (line.class_name or "").strip().lower()
    if name:
        return name
    if line.class_id is not None:
        mapped = YOLO_CLASS_NAMES.get(line.class_id)
        if mapped:
            return mapped
    if line.line_type == "ocr":
        return "text"
    if line.line_type == "element":
        return "element"
    return ""


def _box_outline_for_line(line: OcrLine, *, is_selected: bool) -> str:
    if is_selected:
        return "red"
    class_key = _yolo_class_key(line)
    if class_key in _CLASS_OUTLINE_COLORS:
        return _CLASS_OUTLINE_COLORS[class_key]
    if _is_icon_ocr_line(line):
        return "cyan"
    if _is_ui_object_line(line):
        return "orange"
    return "lime"


def _pil_overlay_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("segoeui.ttf", "arial.ttf", "Arial.ttf"):
        path = win_fonts / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _configure_ui_fonts(root: tk.Misc, size: int) -> tkfont.Font:
    ui_font = tkfont.Font(root=root, family="Segoe UI", size=size)
    root.option_add("*Font", ui_font)
    style = ttk.Style(root)
    for widget in ("TLabel", "TButton", "TCheckbutton", "TEntry", "TFrame", "TRadiobutton"):
        style.configure(widget, font=ui_font)
    return ui_font


def _discover_folder_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            out.append(p)
    return out


def _regions_to_ocr_lines(
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], list[str]]],
) -> list[OcrLine]:
    lines: list[OcrLine] = []
    for box, _center, preds in regions:
        text = "".join(str(p) for p in preds).strip()
        lines.append(OcrLine(box=tuple(int(v) for v in box), text=text))
    return lines


def _save_test_image_ocr_json(
    image_path: Path,
    lines: list[OcrLine],
    *,
    yolo_elapsed_ms: float | None = None,
    ocr_elapsed_ms: float | None = None,
    source: str,
) -> Path:
    out_path = image_path.with_suffix(".json")
    line_pairs = [[list(line.box), line.text] for line in lines]
    write_json(
        out_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_path": str(image_path),
            "image_name": image_path.name,
            "line_height": 32,
            "yolo_elapsed_ms": yolo_elapsed_ms,
            "ocr_elapsed_ms": ocr_elapsed_ms,
            "source": source,
            "lines": line_pairs,
            "text": "\n".join(line.text for line in lines),
        },
    )
    return out_path


# key, label, from, to, default, display decimals, step
_LINE_PARAM_SLIDERS: tuple[tuple[str, str, float, float, float, int, float], ...] = (
    ("blur_ksize", "Blur ksize", 1, 31, 5, 0, 2),
    ("canny_low", "Canny low", 0, 255, 50, 0, 1),
    ("canny_high", "Canny high", 0, 255, 150, 0, 1),
    ("rho", "Hough rho", 0.5, 10.0, 1.0, 1, 0.5),
    ("theta_deg", "Theta (deg)", 0.1, 10.0, 1.0, 1, 0.1),
    ("threshold", "Threshold", 1, 300, 100, 0, 1),
    ("min_line_length", "minLineLength", 0, 300, 20, 0, 1),
    ("max_line_gap", "maxLineGap", 0, 100, 0, 0, 1),
    ("min_height", "Min height", 1, 100, 10.0, 1, 0.5),
    ("min_overlap_frac", "Min overlap", 0.0, 1.0, 0.95, 2, 0.01),
    ("min_width_over_height", "Min W/H", 0.5, 50.0, 5.0, 1, 0.5),
    ("vertical_merge_gap", "V merge gap", 0, 200, 60.0, 0, 5),
)

_LINE_PARAM_TOOLTIPS: dict[str, str] = {
    "blur_ksize": (
        "高斯模糊核心大小（奇數）。數值越大越能抑制雜訊，"
        "但也可能把輸入框短邊這類細線糊掉。"
    ),
    "canny_low": (
        "Canny 下限滯後閾值。低於此值的弱邊緣會被忽略，"
        "除非與強邊緣相連。"
    ),
    "canny_high": (
        "Canny 上限滯後閾值。高於此值視為強邊緣。"
        "調高可只保留高對比邊界。"
    ),
    "rho": (
        "Hough 距離解析度（像素）。越小線段定位越精準；"
        "越大則較粗略、較快。"
    ),
    "theta_deg": (
        "Hough 角度解析度（度）。越小越能分辨接近水平／"
        "略微傾斜等細微方向差異。"
    ),
    "threshold": (
        "被認定為直線所需的最低票數。調低可找出更多／較弱線段；"
        "調高則只保留較明顯、較長的線。"
    ),
    "min_line_length": (
        "可接受的最短線段長度（像素）。調低較容易抓到輸入框"
        "較短的左右邊；調高可過濾雜訊小段。"
    ),
    "max_line_gap": (
        "合併共線碎段時允許的最大間隙（像素）。"
        "虛線或斷開邊框可調高此值。"
    ),
    "min_height": (
        "相鄰水平線配對時，兩線之間的最小垂直距離（像素）。"
        "低於此值視為同一條線，不會形成矩形。"
    ),
    "min_overlap_frac": (
        "兩條水平線在 X 軸上重疊長度，須各自達線段長度的此比例才配對。"
        "調低可接受較短的重疊；調高要求更完整的對齊。"
    ),
    "min_width_over_height": (
        "配對矩形寬度 ÷ 高度的最低比例。輸入框通常很扁，"
        "預設較高以過濾非輸入框的形狀。"
    ),
    "vertical_merge_gap": (
        "合併垂直線段時允許的最大 Y 間隙（像素）。表格直線常因"
        "水平格線而斷成多段；調高可串接同一欄的碎段。"
    ),
}


class _WidgetTooltip:
    """Simple delayed hover tooltip for Tk / ttk widgets."""

    def __init__(self, widget: tk.Misc, text: str, *, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event: object | None = None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _event: object | None = None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except tk.TclError:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        label = tk.Label(
            tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            foreground="#222222",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
            wraplength=280,
            font=("Segoe UI", 9),
        )
        label.pack()
        self._tip = tip

    def _hide(self) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def _attach_tooltip(widget: tk.Misc, text: str) -> _WidgetTooltip:
    return _WidgetTooltip(widget, text)

_LINE_SEGMENT_PARAMS_PATH = ROOT_DIR / "line_segment_params.json"


def _line_segment_params_path() -> Path:
    return _LINE_SEGMENT_PARAMS_PATH


def _clamp_line_segment_param(key: str, value: float) -> float:
    for slider_key, _label, lo, hi, _default, _decimals, _step in _LINE_PARAM_SLIDERS:
        if slider_key != key:
            continue
        return max(lo, min(hi, float(value)))
    return float(value)


def _load_line_segment_params() -> LineSegmentParams:
    """Load saved slider values, falling back to defaults for missing/invalid fields."""
    defaults = LineSegmentParams()
    raw = read_json(_line_segment_params_path(), default={})
    if not isinstance(raw, dict):
        return defaults

    def _num(key: str, fallback: float) -> float:
        value = raw.get(key, fallback)
        try:
            return _clamp_line_segment_param(key, float(value))
        except (TypeError, ValueError):
            return fallback

    blur = int(round(_num("blur_ksize", defaults.blur_ksize)))
    if blur % 2 == 0:
        blur = max(1, blur - 1)
    return LineSegmentParams(
        blur_ksize=blur,
        canny_low=int(round(_num("canny_low", defaults.canny_low))),
        canny_high=int(round(_num("canny_high", defaults.canny_high))),
        rho=float(_num("rho", defaults.rho)),
        theta_deg=float(_num("theta_deg", defaults.theta_deg)),
        threshold=int(round(_num("threshold", defaults.threshold))),
        min_line_length=int(round(_num("min_line_length", defaults.min_line_length))),
        max_line_gap=int(round(_num("max_line_gap", defaults.max_line_gap))),
        min_width_over_height=float(
            _num("min_width_over_height", defaults.min_width_over_height)
        ),
        min_overlap_frac=float(_num("min_overlap_frac", defaults.min_overlap_frac)),
        min_height=float(_num("min_height", defaults.min_height)),
        vertical_merge_gap=float(
            _num("vertical_merge_gap", defaults.vertical_merge_gap)
        ),
    )


_COLOR_PARAM_SLIDERS: tuple[tuple[str, str, float, float, float, int, float], ...] = (
    ("num_colors", "Segment count", 0, 300, 120, 0, 5),
    ("min_area_frac", "Min area %", 0.05, 20.0, 0.3, 2, 0.05),
    ("blur_ksize", "Blur ksize", 1, 31, 5, 0, 2),
    ("edge_canny_low", "Canny low", 5, 150, 30, 0, 5),
    ("edge_canny_high", "Canny high", 30, 255, 100, 0, 5),
    ("edge_dilate", "Edge dilate", 0, 8, 2, 0, 1),
    ("slic_compactness", "SLIC compact", 1.0, 30.0, 10.0, 0, 1),
    ("split_max_area_frac", "Split above %", 2.0, 40.0, 6.0, 1, 0.5),
    ("merge_color_dist", "Merge dist", 2.0, 40.0, 10.0, 0, 1),
)

_COLOR_PARAM_TOOLTIPS: dict[str, str] = {
    "num_colors": (
        "Superpixel count (0 uses 1). Higher values yield more candidate regions."
    ),
    "min_area_frac": (
        "Minimum region size as a percentage of total image area. Smaller blobs "
        "are discarded so only large color regions remain."
    ),
    "blur_ksize": (
        "Gaussian blur kernel (odd). Smooths fine texture before segmentation so "
        "clustering follows broad color blocks rather than pixel noise."
    ),
    "edge_canny_low": (
        "Spatial split. Canny lower threshold. Lower values detect more subtle "
        "panel borders and input-box lines."
    ),
    "edge_canny_high": (
        "Spatial split. Canny upper threshold. Raise to keep only strong "
        "structural borders."
    ),
    "edge_dilate": (
        "Spatial split. Dilate detected edge pixels so thin lines become solid "
        "walls when splitting oversized regions."
    ),
    "slic_compactness": (
        "Spatial only. Lower values follow color edges more tightly; higher "
        "values produce more grid-like superpixels."
    ),
    "split_max_area_frac": (
        "Regions larger than this percentage of the image are split again using "
        "internal edges, even when they share one flat color."
    ),
    "merge_color_dist": (
        "LAB color distance for merging only adjacent superpixels or regions. "
        "Non-adjacent same-color areas are never merged."
    ),
}

_COLOR_SEGMENT_PARAMS_PATH = ROOT_DIR / "color_segment_params.json"


def _color_segment_params_path() -> Path:
    return _COLOR_SEGMENT_PARAMS_PATH


def _clamp_color_segment_param(key: str, value: float) -> float:
    for slider_key, _label, lo, hi, _default, _decimals, _step in _COLOR_PARAM_SLIDERS:
        if slider_key != key:
            continue
        return max(lo, min(hi, float(value)))
    return float(value)


def _load_color_segment_params() -> ColorSegmentParams:
    defaults = ColorSegmentParams()
    raw = read_json(_color_segment_params_path(), default={})
    if not isinstance(raw, dict):
        return defaults

    def _num(key: str, fallback: float) -> float:
        value = raw.get(key, fallback)
        try:
            return _clamp_color_segment_param(key, float(value))
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


def _save_color_segment_params(params: ColorSegmentParams) -> None:
    write_json(
        _color_segment_params_path(),
        {
            "num_colors": int(params.num_colors),
            "slic_compactness": float(params.slic_compactness),
            "min_area_frac": round(params.min_area_frac * 100.0, 3),
            "blur_ksize": int(params.blur_ksize),
            "mask_text_icons": bool(params.mask_text_icons),
            "require_yolo_objects": bool(params.require_yolo_objects),
            "merge_superpixels": bool(params.merge_superpixels),
            "merge_similar": bool(params.merge_similar),
            "merge_color_dist": float(params.merge_color_dist),
            "split_large_regions": bool(params.split_large_regions),
            "split_max_area_frac": round(params.split_max_area_frac * 100.0, 2),
            "edge_canny_low": int(params.edge_canny_low),
            "edge_canny_high": int(params.edge_canny_high),
            "edge_dilate": int(params.edge_dilate),
        },
    )


_REGION_OUTLINE_COLORS: tuple[str, ...] = (
    "lime",
    "orange",
    "cyan",
    "magenta",
    "yellow",
    "deepskyblue",
    "violet",
    "springgreen",
    "coral",
    "gold",
)


def _draw_color_segment_overlays(
    image: Image.Image,
    result: ColorSegmentResult | None,
    *,
    show_quantized: bool,
    show_masked_input: bool = False,
    show_boxes: bool,
    show_labels: bool,
    selected_region_id: int | None = None,
    masked_input_override: Image.Image | None = None,
    mask_boxes_override: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    if result is None and not show_masked_input:
        return image.copy()
    if show_masked_input:
        prepared = masked_input_override
        if prepared is None and result is not None:
            prepared = result.prepared
        if prepared is not None:
            out = prepared.copy()
        else:
            out = image.copy()
        mask_boxes = mask_boxes_override
        if mask_boxes is None and result is not None:
            mask_boxes = result.mask_boxes
        if mask_boxes:
            draw = ImageDraw.Draw(out)
            for x, y, bw, bh in mask_boxes:
                draw.rectangle(
                    [(x, y), (x + bw, y + bh)],
                    outline="red",
                    width=2,
                )
        if not show_boxes and not show_labels:
            return out
        if result is None:
            return out
        draw = ImageDraw.Draw(out)
        for region in result.regions:
            x0, y0, x1, y1 = region.bbox
            is_sel = selected_region_id == region.region_id
            color = (
                "lime"
                if is_sel
                else _REGION_OUTLINE_COLORS[
                    region.region_id % len(_REGION_OUTLINE_COLORS)
                ]
            )
            if show_boxes:
                draw.rectangle(
                    [(x0, y0), (x1, y1)], outline=color, width=4 if is_sel else 2
                )
            if show_labels:
                label = f"#{region.region_id + 1}"
                draw.text((x0 + 2, y0 + 2), label, fill=color)
        return out
    if result is None:
        return image.copy()
    out = result.quantized.copy() if show_quantized else image.copy()
    if not show_boxes and not show_labels:
        return out
    draw = ImageDraw.Draw(out)
    for region in result.regions:
        x0, y0, x1, y1 = region.bbox
        is_sel = selected_region_id == region.region_id
        color = (
            "red"
            if is_sel
            else _REGION_OUTLINE_COLORS[region.region_id % len(_REGION_OUTLINE_COLORS)]
        )
        if show_boxes:
            draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=4 if is_sel else 2)
        if show_labels:
            label = f"#{region.region_id + 1}"
            draw.text((x0 + 2, y0 + 2), label, fill=color)
    return out


def _save_line_segment_params(params: LineSegmentParams) -> None:
    write_json(
        _line_segment_params_path(),
        {
            "blur_ksize": int(params.blur_ksize),
            "canny_low": int(params.canny_low),
            "canny_high": int(params.canny_high),
            "rho": float(params.rho),
            "theta_deg": float(params.theta_deg),
            "threshold": int(params.threshold),
            "min_line_length": int(params.min_line_length),
            "max_line_gap": int(params.max_line_gap),
            "min_width_over_height": float(params.min_width_over_height),
            "min_overlap_frac": float(params.min_overlap_frac),
            "min_height": float(params.min_height),
            "vertical_merge_gap": float(params.vertical_merge_gap),
        },
    )


def _draw_overlays(
    image: Image.Image,
    lines: list[OcrLine],
    show_boxes: bool,
    show_labels: bool,
    selected_idx: int | None = None,
    overlay_font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None,
    line_segments: list[tuple[int, int, int, int]] | None = None,
    line_segment_color: str = "lime",
    merged_segments: list[tuple[int, int, int, int]] | None = None,
    merged_segment_color: str = "orange",
    candidate_segments: list[tuple[int, int, int, int]] | None = None,
    candidate_segment_color: str = "magenta",
    vertical_segments: list[tuple[int, int, int, int]] | None = None,
    vertical_segment_color: str = "cyan",
    rectangle_boxes: list[tuple[int, int, int, int]] | None = None,
    rectangle_box_color: str = "blue",
    selected_rectangle_idx: int | None = None,
    selected_rectangle_color: str = "red",
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = overlay_font if overlay_font is not None else ImageFont.load_default()
    for idx, line in enumerate(lines):
        x, y, w, h = line.box
        x2, y2 = x + w, y + h
        is_selected = selected_idx is not None and idx == selected_idx
        class_key = _yolo_class_key(line)
        draw_box = show_boxes or class_key in ("scrollbar_original", "input_original")
        if draw_box:
            outline = _box_outline_for_line(line, is_selected=is_selected)
            width = 2 if is_selected or _is_ui_object_line(line) else 1
            draw.rectangle([(x, y), (x2, y2)], outline=outline, width=width)
        if show_labels:
            text = _display_label_for_line(line)
            if not text or text == "<empty>":
                continue
            text_bbox = draw.textbbox((x, y), text, font=font)
            tx1, ty1, tx2, ty2 = text_bbox
            pad = 3 if overlay_font is not None else 2
            draw.rectangle([(tx1 - pad, ty1 - pad), (tx2 + pad, ty2 + pad)], fill="black")
            if is_selected:
                text_color = "red"
            else:
                class_key = _yolo_class_key(line)
                if class_key in _CLASS_OUTLINE_COLORS:
                    text_color = _CLASS_OUTLINE_COLORS[class_key]
                elif _is_icon_ocr_line(line):
                    text_color = "cyan"
                elif _is_ui_object_line(line):
                    text_color = "orange"
                else:
                    text_color = "yellow"
            draw.text((x, y), text, font=font, fill=text_color)

    def _draw_segs(
        segs: list[tuple[int, int, int, int]] | None, color: str, width: int = 2
    ) -> None:
        if not segs:
            return
        for x1, y1, x2, y2 in segs:
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    _draw_segs(line_segments, line_segment_color)
    _draw_segs(merged_segments, merged_segment_color)
    _draw_segs(candidate_segments, candidate_segment_color)
    _draw_segs(vertical_segments, vertical_segment_color)
    if rectangle_boxes:
        for idx, (x0, y0, x1, y1) in enumerate(rectangle_boxes):
            is_sel = selected_rectangle_idx is not None and idx == selected_rectangle_idx
            color = selected_rectangle_color if is_sel else rectangle_box_color
            width = 4 if is_sel else 3
            draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=width)
    return out


def _clamp_box(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int, *, min_size: int = MIN_BOX_SIZE
) -> tuple[int, int, int, int]:
    x = max(0, min(x, max(0, img_w - 1)))
    y = max(0, min(y, max(0, img_h - 1)))
    w = max(min_size, min(w, img_w - x))
    h = max(min_size, min(h, img_h - y))
    return x, y, w, h


def _smallest_box_hit_index(lines: list[OcrLine], img_x: int, img_y: int) -> int | None:
    """Return the index of the smallest box containing ``(img_x, img_y)``."""
    best_idx: int | None = None
    best_area = 0
    for idx, line in enumerate(lines):
        x, y, w, h = line.box
        if x <= img_x <= x + w and y <= img_y <= y + h:
            area = w * h
            if best_idx is None or area < best_area:
                best_idx = idx
                best_area = area
    return best_idx


def _adjust_box_edge(
    box: tuple[int, int, int, int],
    direction: str,
    *,
    expand: bool,
    step: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Move one edge of ``(x, y, w, h)`` outward (expand) or inward (shrink)."""
    x, y, w, h = box
    s = max(1, step)
    if expand:
        if direction == "up":
            y -= s
            h += s
        elif direction == "down":
            h += s
        elif direction == "left":
            x -= s
            w += s
        elif direction == "right":
            w += s
    else:
        if direction == "up":
            y += s
            h -= s
        elif direction == "down":
            h -= s
        elif direction == "left":
            x += s
            w -= s
        elif direction == "right":
            w -= s
    return _clamp_box(x, y, w, h, img_w, img_h)


class OcrViewerApp:
    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4  # pixels²; drag beyond this pans instead of selecting
    # Right-drag: each horizontal pixel nudges zoom by this power (right = in, left = out).
    _RMB_ZOOM_PER_PIXEL = 1.0012

    def __init__(
        self,
        parent: tk.Misc,
        runs_root: Path,
        *,
        manage_window: bool = True,
        bind_global_hotkeys: bool = True,
        session_list_label: str = "Runs",
    ):
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.runs_root = runs_root
        self._session_list_label = session_list_label
        self._manage_window = manage_window
        self._bind_global_hotkeys = bind_global_hotkeys
        self._hotkeys_active = False
        self.run_dirs = _discover_runs(runs_root)
        self.current_run_images: list[Path] = []
        self.current_display: ImageTk.PhotoImage | None = None
        self.current_image: Image.Image | None = None
        self.current_lines: list[OcrLine] = []
        self.selected_line_idx: int | None = None
        self._ultralytics_model_holder: list[Any] = []

        self.show_boxes = tk.BooleanVar(value=True)
        self.show_original_scrollbar = tk.BooleanVar(value=False)
        self.show_original_input = tk.BooleanVar(value=False)
        self.show_rectangles = tk.BooleanVar(value=False)
        self.show_color_regions = tk.BooleanVar(value=True)
        self.show_region_labels = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        _dcf = DEFAULT_CONF_YOLOV26_END2END
        self.yolo_conf_var = tk.StringVar(value=f"{_dcf:g}")
        self.box_edit_mode = tk.StringVar(value="expand")

        self._view_zoom = 1.0
        self._rmb_last_x: int | None = None
        self._render_scale = 1.0
        self._lmb_press_xy: tuple[int, int] | None = None
        self._lmb_panning = False
        self._yolo_lines_cache: YoloLinesCache = {}
        self._original_scrollbar_lines: list[OcrLine] = []
        self._original_input_lines: list[OcrLine] = []
        self.input_box_rectangles: list[tuple[int, int, int, int]] = []
        self.segment_result: ColorSegmentResult | None = None
        self.selected_region_id: int | None = None
        self._spatial_rank_by_box: dict[tuple[int, int, int, int], int] = {}

        self._build_ui()
        self._populate_runs()
        if bind_global_hotkeys:
            self.activate_hotkeys()

    def _set_current_lines(self, lines: list[OcrLine]) -> None:
        deduped = _dedupe_ocr_lines(lines)
        self.current_lines, self._original_scrollbar_lines, self._original_input_lines = (
            _split_debug_original_lines(deduped)
        )

    def _all_display_lines(self) -> list[OcrLine]:
        lines = list(self.current_lines)
        if self.show_original_scrollbar.get() and self._original_scrollbar_lines:
            lines.extend(self._original_scrollbar_lines)
        if self.show_original_input.get() and self._original_input_lines:
            lines.extend(self._original_input_lines)
        return lines

    def _on_toggle_debug_originals(self) -> None:
        display_len = len(self._all_display_lines())
        if self.selected_line_idx is not None and self.selected_line_idx >= display_len:
            self.selected_line_idx = None
            self.item_list.select_clear(0, tk.END)
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        if self.selected_line_idx is not None:
            self.item_list.select_set(self.selected_line_idx)
            self.item_list.see(self.selected_line_idx)
        self._refresh_image()

    def _refresh_spatial_segmentation(self) -> None:
        state = _build_yolo_spatial_segment_state(self.current_image, self.current_lines)
        self.segment_result = state.result
        self._spatial_rank_by_box = dict(state.spatial_ranks)
        if self.selected_region_id is not None and self.segment_result is not None:
            region_ids = {region.region_id for region in self.segment_result.regions}
            if self.selected_region_id not in region_ids:
                self.selected_region_id = None
        self._populate_region_list()

    def _populate_region_list(self) -> None:
        self.region_list.delete(0, tk.END)
        if self.segment_result is None:
            return
        for region in self.segment_result.regions:
            self.region_list.insert(tk.END, _format_color_region_row(region))
        if self.selected_region_id is not None:
            for idx, region in enumerate(self.segment_result.regions):
                if region.region_id == self.selected_region_id:
                    self.region_list.select_set(idx)
                    self.region_list.see(idx)
                    break

    def _spatial_segment_status_suffix(self) -> str:
        if self.segment_result is None:
            return ""
        before = self.segment_result.regions_before_yolo_filter
        kept = len(self.segment_result.regions)
        return f" | color regions: {kept} kept ({before} before YOLO filter)"

    def _on_region_select(self, _event: object | None = None) -> None:
        selected = self.region_list.curselection()
        if not selected or self.segment_result is None:
            self.selected_region_id = None
            self._refresh_image()
            return
        idx = selected[0]
        if idx < 0 or idx >= len(self.segment_result.regions):
            self.selected_region_id = None
            self._refresh_image()
            return
        region = self.segment_result.regions[idx]
        self.selected_region_id = region.region_id
        self._refresh_image()
        x0, y0, x1, y1 = region.bbox
        r, g, b = region.mean_color
        self.status_var.set(
            f"Region #{region.region_id + 1}: ({x0},{y0})-({x1},{y1}), "
            f"area={region.area}px, rgb=({r},{g},{b})"
        )

    def _horizontal_scrollbar_boxes_from_lines(self) -> list[tuple[int, int, int, int]]:
        from cua_mcp.scrollbar_arrows import scrollbar_orientation

        boxes: list[tuple[int, int, int, int]] = []
        for line in self.current_lines:
            if _yolo_class_key(line) != "scrollbar":
                continue
            if scrollbar_orientation(line.box) != "horizontal":
                continue
            boxes.append(line.box)
        return boxes

    def _detect_input_box_rectangles(self) -> None:
        self.input_box_rectangles = []
        if self.current_image is None:
            return
        try:
            result = detect_horizontal_rectangles(
                self.current_image,
                _load_line_segment_params(),
                horizontal_scrollbar_boxes=self._horizontal_scrollbar_boxes_from_lines(),
            )
            self.input_box_rectangles = result.rectangles
        except Exception:
            self.input_box_rectangles = []

    def _input_box_rectangle_overlay(self) -> list[tuple[int, int, int, int]] | None:
        if not self.show_rectangles.get() or not self.input_box_rectangles:
            return None
        return self.input_box_rectangles

    def _line_at_display_index(self, idx: int) -> OcrLine | None:
        lines = self._all_display_lines()
        if idx < 0 or idx >= len(lines):
            return None
        return lines[idx]

    def _is_ocr_line_index(self, idx: int) -> bool:
        return 0 <= idx < len(self.current_lines)

    def _build_ui(self) -> None:
        if self._manage_window and isinstance(self.parent, tk.Tk):
            self.parent.title("OCR Overlay Viewer")
            self.parent.geometry("1280x840")
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)

        left = ttk.Frame(self.parent, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        ttk.Label(left, text=self._session_list_label).grid(row=0, column=0, sticky="w")
        ttk.Label(left, text="YOLO Images").grid(row=0, column=1, sticky="w", padx=(8, 0))
        left.rowconfigure(1, weight=1)
        run_wrap = ttk.Frame(left)
        run_wrap.grid(row=1, column=0, sticky="nsew")
        run_wrap.columnconfigure(0, weight=1)
        run_wrap.rowconfigure(0, weight=1)
        self.run_list = tk.Listbox(run_wrap, exportselection=False, height=10, width=24)
        self.run_list.grid(row=0, column=0, sticky="nsew")
        self.run_scroll = ttk.Scrollbar(run_wrap, orient="vertical", command=self.run_list.yview)
        self.run_scroll.grid(row=0, column=1, sticky="ns")
        self.run_list.configure(yscrollcommand=self.run_scroll.set)
        self.run_list.bind("<<ListboxSelect>>", self._on_run_select)

        image_wrap = ttk.Frame(left)
        image_wrap.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        image_wrap.columnconfigure(0, weight=1)
        image_wrap.rowconfigure(0, weight=1)
        self.image_list = tk.Listbox(image_wrap, exportselection=False, height=10, width=24)
        self.image_list.grid(row=0, column=0, sticky="nsew")
        self.image_scroll = ttk.Scrollbar(image_wrap, orient="vertical", command=self.image_list.yview)
        self.image_scroll.grid(row=0, column=1, sticky="ns")
        self.image_list.configure(yscrollcommand=self.image_scroll.set)
        self.image_list.bind("<<ListboxSelect>>", self._on_image_select)

        ttk.Label(left, text="Color regions").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        region_wrap = ttk.Frame(left)
        region_wrap.grid(row=3, column=0, columnspan=2, sticky="ew")
        region_wrap.columnconfigure(0, weight=1)
        self.region_list = tk.Listbox(region_wrap, exportselection=False, height=5, width=72)
        self.region_list.grid(row=0, column=0, sticky="ew")
        self.region_scroll = ttk.Scrollbar(
            region_wrap, orient="vertical", command=self.region_list.yview
        )
        self.region_scroll.grid(row=0, column=1, sticky="ns")
        self.region_list.configure(yscrollcommand=self.region_scroll.set)
        self.region_list.bind("<<ListboxSelect>>", self._on_region_select)

        ttk.Label(left, text="YOLO Detections (rank vs #1)").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        item_wrap = ttk.Frame(left)
        item_wrap.grid(row=5, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(5, weight=1)
        item_wrap.columnconfigure(0, weight=1)
        item_wrap.rowconfigure(0, weight=1)
        self.item_list = tk.Listbox(item_wrap, exportselection=False, height=10, width=72)
        self.item_list.grid(row=0, column=0, sticky="nsew")
        self.item_scroll = ttk.Scrollbar(item_wrap, orient="vertical", command=self.item_list.yview)
        self.item_scroll.grid(row=0, column=1, sticky="ns")
        self.item_list.configure(yscrollcommand=self.item_scroll.set)
        self.item_list.bind("<<ListboxSelect>>", self._on_item_select)
        self.item_list.bind("<Double-Button-1>", self._on_item_double_click)
        self.item_list.bind("<Delete>", self._delete_selected_detection)

        controls = ttk.Frame(left)
        controls.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for col in range(4):
            controls.columnconfigure(col, weight=1)
        ttk.Checkbutton(controls, text="Boxes", variable=self.show_boxes, command=self._refresh_image).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Color regions",
            variable=self.show_color_regions,
            command=self._refresh_image,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Region labels",
            variable=self.show_region_labels,
            command=self._refresh_image,
        ).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Original scrollbar",
            variable=self.show_original_scrollbar,
            command=self._on_toggle_debug_originals,
        ).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Rectangles",
            variable=self.show_rectangles,
            command=self._refresh_image,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Original input",
            variable=self.show_original_input,
            command=self._on_toggle_debug_originals,
        ).grid(row=1, column=1, sticky="w")
        ttk.Label(controls, text="Arrows").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Radiobutton(
            controls,
            text="Expand",
            variable=self.box_edit_mode,
            value="expand",
        ).grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Radiobutton(
            controls,
            text="Shrink",
            variable=self.box_edit_mode,
            value="shrink",
        ).grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Button(controls, text="Prev", command=self._prev_image).grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Next", command=self._next_image).grid(row=3, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Zoom +", command=self._zoom_in).grid(row=3, column=2, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Zoom -", command=self._zoom_out).grid(row=3, column=3, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Reload YOLO detections", command=self._run_select_text_current_image).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Copy to undone/images", command=self._copy_current_image_to_undone).grid(
            row=4, column=2, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(
            controls,
            text="Copy all run images to undone/images",
            command=self._copy_all_run_images_to_undone,
        ).grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(controls, text="YOLO confidence").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.yolo_conf_var, width=10).grid(
            row=6, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=(6, 0)
        )
        ttk.Button(controls, text="Delete selected", command=self._delete_selected_detection).grid(
            row=7, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Reset Zoom", command=self._reset_zoom).grid(
            row=8, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )

        canvas_wrap = ttk.Frame(self.parent, padding=8)
        canvas_wrap.grid(row=0, column=1, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, bg="#1e1e1e", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-3>", self._on_rmb_press)
        self.canvas.bind("<B3-Motion>", self._on_rmb_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self.canvas.bind("<ButtonPress-1>", self._on_lmb_press)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<ButtonPress-2>", self._on_mmb_press)
        self.canvas.bind("<B2-Motion>", self._on_mmb_drag)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Delete>", self._delete_selected_detection)

        status = ttk.Label(self.parent, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.canvas.bind(key, self._on_arrow_key)
            self.item_list.bind(key, self._on_arrow_key)
            self.image_list.bind(key, self._on_arrow_key)
            self.run_list.bind(key, self._on_arrow_key)
        self._bind_shift_box_edit_toggle(self.canvas, self.item_list, self.image_list, self.run_list)
        self.parent.bind("<Configure>", lambda _event: self._refresh_image())

    def activate_hotkeys(self) -> None:
        if self._hotkeys_active:
            return
        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.root.bind(key, self._on_arrow_key)
        self._bind_shift_box_edit_toggle(self.root)
        self.root.bind("<Control-plus>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-equal>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-minus>", self._on_zoom_out_hotkey)
        self.root.bind("<Control-0>", self._on_reset_zoom_hotkey)
        self._hotkeys_active = True

    def deactivate_hotkeys(self) -> None:
        if not self._hotkeys_active:
            return
        for key in (
            "<Up>",
            "<Down>",
            "<Left>",
            "<Right>",
            "<KeyPress-Shift_L>",
            "<KeyPress-Shift_R>",
            "<Control-plus>",
            "<Control-equal>",
            "<Control-minus>",
            "<Control-0>",
        ):
            self.root.unbind(key)
        self._hotkeys_active = False

    def _populate_runs(self) -> None:
        self.run_list.delete(0, tk.END)
        for run in self.run_dirs:
            self.run_list.insert(tk.END, run.name)
        if self.run_dirs:
            self.run_list.select_set(0)
            self._on_run_select()
        else:
            label = self._session_list_label.lower()
            self.status_var.set(f"No {label} found at {self.runs_root}")

    def _selected_run(self) -> Path | None:
        selected = self.run_list.curselection()
        if not selected:
            return None
        return self.run_dirs[selected[0]]

    def _on_run_select(self, _event: object | None = None) -> None:
        run = self._selected_run()
        self.current_run_images = _yolo_ocr_paired_images(run) if run is not None else []
        self.selected_line_idx = None
        self._yolo_lines_cache.clear()
        self.image_list.delete(0, tk.END)
        for img in self.current_run_images:
            self.image_list.insert(tk.END, img.name)
        if self.current_run_images:
            self._select_image_index(0)
        else:
            self.current_image = None
            self._set_current_lines([])
            self.segment_result = None
            self.selected_region_id = None
            self._spatial_rank_by_box = {}
            self.item_list.delete(0, tk.END)
            self.region_list.delete(0, tk.END)
            self.canvas.delete("all")
            self.status_var.set(
                f"No images found for {run.name if run else '-'}"
            )

    def _selected_image_index(self) -> int | None:
        selected = self.image_list.curselection()
        if not selected:
            return None
        return selected[0]

    def _current_image_path(self) -> Path | None:
        idx = self._selected_image_index()
        if idx is None or idx >= len(self.current_run_images):
            return None
        return self.current_run_images[idx]

    def _select_image_index(self, idx: int) -> None:
        self.image_list.select_clear(0, tk.END)
        self.image_list.select_set(idx)
        self.image_list.see(idx)

    def _on_image_select(self, _event: object | None = None) -> None:
        image_path = self._current_image_path()
        run = self._selected_run()
        if image_path is None or run is None:
            return
        self.current_image = Image.open(image_path).convert("RGB")
        self._view_zoom = 1.0
        conf, err = _parse_conf_0_to_1(self.yolo_conf_var.get())
        if conf is None:
            self._set_current_lines([])
            status = f"Invalid confidence: {err}"
        else:
            lines, status = resolve_image_lines(
                image_path,
                yolo_conf_threshold=conf,
                allow_yolo=False,
                yolo_cache=self._yolo_lines_cache,
                run_dir=run,
            )
            self._set_current_lines(lines)
        self.selected_line_idx = None
        self.selected_region_id = None
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        self._detect_input_box_rectangles()
        self.status_var.set(f"{image_path.name} - {status}{self._spatial_segment_status_suffix()}")
        self._refresh_image()

    def _on_rmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            self._rmb_last_x = None
            return
        self._rmb_last_x = int(event.x)

    def _on_rmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._rmb_last_x is None or self.current_image is None:
            return
        x = int(event.x)
        dx = x - self._rmb_last_x
        self._rmb_last_x = x
        # Horizontal only: drag right → zoom in, drag left → zoom out.
        if dx == 0:
            return
        z = self._view_zoom * (self._RMB_ZOOM_PER_PIXEL**dx)
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, z))
        self._refresh_image()

    def _on_rmb_release(self, _event: tk.Event[tk.Canvas]) -> None:
        self._rmb_last_x = None

    def _on_lmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            return
        self._lmb_press_xy = (int(event.x), int(event.y))
        self._lmb_panning = False

    def _on_lmb_motion(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        x0, y0 = self._lmb_press_xy
        x, y = int(event.x), int(event.y)
        if not self._lmb_panning:
            if (x - x0) ** 2 + (y - y0) ** 2 < self._PAN_CLICK_THRESHOLD_SQ:
                return
            self.canvas.scan_mark(x0, y0)
            self._lmb_panning = True
        self.canvas.scan_dragto(x, y, gain=1)

    def _on_lmb_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        try:
            if not self._lmb_panning:
                self._select_ocr_at_canvas_event(event)
        finally:
            self._lmb_press_xy = None
            self._lmb_panning = False

    def _ocr_hit_index_at_canvas(self, event: tk.Event[tk.Canvas]) -> int | None:
        lines = self._all_display_lines()
        if self.current_image is None or not lines:
            return None
        canvas_x = self.canvas.canvasx(int(event.x))
        canvas_y = self.canvas.canvasy(int(event.y))
        img_x = int(canvas_x / max(self._render_scale, 1e-6))
        img_y = int(canvas_y / max(self._render_scale, 1e-6))
        return _smallest_box_hit_index(lines, img_x, img_y)

    def _on_canvas_double_click(self, event: tk.Event[tk.Canvas]) -> None:
        idx = self._ocr_hit_index_at_canvas(event)
        if idx is None:
            return
        self.selected_line_idx = idx
        self.item_list.select_clear(0, tk.END)
        self.item_list.select_set(idx)
        self.item_list.see(idx)
        self._refresh_image()
        if self._is_ocr_line_index(idx):
            self._open_item_edit_popup(idx)
            return
        line = self._line_at_display_index(idx)
        if line is not None and _is_ui_object_line(line):
            self._open_ui_object_export_popup(idx)

    def _on_mmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_mark(int(event.x), int(event.y))

    def _on_mmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_dragto(int(event.x), int(event.y), gain=1)

    def _on_canvas_mousewheel(self, event: tk.Event[tk.Canvas]) -> None:
        if event.state & 0x0004:
            if event.delta > 0:
                self._apply_zoom_factor(self._ZOOM_STEP)
            elif event.delta < 0:
                self._apply_zoom_factor(1.0 / self._ZOOM_STEP)
            return
        if event.state & 0x0001:
            self.canvas.xview_scroll(int(-(event.delta / 120)), "units")
        else:
            self.canvas.yview_scroll(int(-(event.delta / 120)), "units")

    def _set_zoom(self, zoom: float) -> None:
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, zoom))
        self._refresh_image()

    def _apply_zoom_factor(self, factor: float) -> None:
        self._set_zoom(self._view_zoom * factor)

    def _zoom_in(self) -> None:
        self._apply_zoom_factor(self._ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._apply_zoom_factor(1.0 / self._ZOOM_STEP)

    def _reset_zoom(self) -> None:
        self._set_zoom(1.0)

    def _on_zoom_in_hotkey(self, _event: tk.Event[tk.Tk]) -> str:
        self._zoom_in()
        return "break"

    def _on_zoom_out_hotkey(self, _event: tk.Event[tk.Tk]) -> str:
        self._zoom_out()
        return "break"

    def _on_reset_zoom_hotkey(self, _event: tk.Event[tk.Tk]) -> str:
        self._reset_zoom()
        return "break"

    def _arrow_direction(self, keysym: str) -> str | None:
        return {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}.get(keysym)

    def _bind_shift_box_edit_toggle(self, *widgets: tk.Misc) -> None:
        for widget in widgets:
            for key in ("<KeyPress-Shift_L>", "<KeyPress-Shift_R>"):
                widget.bind(key, self._on_shift_toggle_box_mode)

    def _on_shift_toggle_box_mode(self, _event: tk.Event) -> str:
        toggle_box_edit_mode(self.box_edit_mode, self.status_var)
        return "break"

    def _adjust_selected_box(self, direction: str, *, step: int) -> bool:
        idx = self.selected_line_idx
        if idx is None or self.current_image is None or not self._is_ocr_line_index(idx):
            return False
        img_w, img_h = self.current_image.size
        line = self.current_lines[idx]
        expand = self.box_edit_mode.get() == "expand"
        new_box = _adjust_box_edge(
            line.box,
            direction,
            expand=expand,
            step=step,
            img_w=img_w,
            img_h=img_h,
        )
        if new_box == line.box:
            return False
        self.current_lines[idx] = OcrLine(
            box=new_box,
            text=line.text,
            line_type=line.line_type,
            class_name=line.class_name,
            class_id=line.class_id,
            chinese_ids=line.chinese_ids,
        )
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        self._refresh_image()
        mode = "Expand" if expand else "Shrink"
        x, y, w, h = new_box
        self.status_var.set(f"{mode} box #{idx + 1}: ({x},{y}) {w}×{h}")
        return True

    def _on_arrow_key(self, event: tk.Event) -> str | None:
        direction = self._arrow_direction(event.keysym)
        if direction is None:
            return None
        if self.selected_line_idx is not None:
            step = BOX_EDIT_STEP_SHIFT if (event.state & 0x0001) else BOX_EDIT_STEP
            if self._adjust_selected_box(direction, step=step):
                return "break"
            return "break"
        if event.keysym == "Left":
            self._prev_image()
            return "break"
        if event.keysym == "Right":
            self._next_image()
            return "break"
        return None

    def _populate_item_list(self) -> None:
        self.item_list.delete(0, tk.END)
        for row in _yolo_detection_list_rows(
            self._all_display_lines(),
            segment_result=self.segment_result,
            spatial_ranks=self._spatial_rank_by_box,
        ):
            self.item_list.insert(tk.END, row)

    def _on_item_select(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            self.selected_line_idx = None
            self.selected_region_id = None
            self._refresh_image()
            return
        self.selected_line_idx = selected[0]
        line = self._line_at_display_index(self.selected_line_idx)
        if line is not None and self.segment_result is not None:
            self.selected_region_id = region_id_for_box(
                self.segment_result.label_map,
                tuple(int(v) for v in line.box),
            )
            if self.selected_region_id is not None:
                for idx, region in enumerate(self.segment_result.regions):
                    if region.region_id == self.selected_region_id:
                        self.region_list.select_clear(0, tk.END)
                        self.region_list.select_set(idx)
                        self.region_list.see(idx)
                        break
        else:
            self.selected_region_id = None
        self._refresh_image()

    def _delete_selected_detection(self, _event: object | None = None) -> str:
        idx = self.selected_line_idx
        display_lines = self._all_display_lines()
        if idx is None or idx < 0 or idx >= len(display_lines):
            self.status_var.set("Select a detection to delete")
            return "break"

        if idx < len(self.current_lines):
            deleted = self.current_lines.pop(idx)
        elif self.show_original_scrollbar.get() and idx < len(self.current_lines) + len(
            self._original_scrollbar_lines
        ):
            deleted = self._original_scrollbar_lines.pop(idx - len(self.current_lines))
        else:
            scrollbar_count = (
                len(self._original_scrollbar_lines) if self.show_original_scrollbar.get() else 0
            )
            deleted = self._original_input_lines.pop(
                idx - len(self.current_lines) - scrollbar_count
            )

        self.selected_line_idx = None
        self.selected_region_id = None
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        remaining = self._all_display_lines()
        if remaining:
            next_idx = min(idx, len(remaining) - 1)
            self.item_list.select_set(next_idx)
            self.item_list.see(next_idx)
            self.selected_line_idx = next_idx
        self._refresh_image()
        deleted_row = _agent_format_detection_rows([deleted])
        detail = deleted_row[0] if deleted_row else _display_label_for_line(deleted)
        self.status_var.set(f"Deleted detection: {detail}")
        return "break"

    def _on_item_double_click(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            return
        idx = selected[0]
        if self._line_at_display_index(idx) is None:
            return
        self.selected_line_idx = idx
        self._refresh_image()
        line = self._line_at_display_index(idx)
        if line is not None and _is_ui_object_line(line):
            self._open_ui_object_export_popup(idx)
        elif self._is_ocr_line_index(idx):
            self._open_item_edit_popup(idx)

    def _open_item_edit_popup(self, display_idx: int) -> None:
        if not self._is_ocr_line_index(display_idx):
            return
        line = self.current_lines[display_idx]
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit OCR Item #{display_idx + 1}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        ttk.Label(dialog, text="Corrected text").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))
        text_var = tk.StringVar(value=line.text)
        text_entry = ttk.Entry(dialog, textvariable=text_var, width=72)
        text_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=10, pady=(10, 6))

        is_icon = _is_pua_icon_identity_text(line.text)
        icon_var = tk.BooleanVar(value=is_icon)

        def _on_icon_toggle() -> None:
            dest_var.set(str(_export_dest_for_icon_identity(icon_var.get())))

        ttk.Checkbutton(
            dialog,
            text="Icon identity (not text)",
            variable=icon_var,
            command=_on_icon_toggle,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))

        ttk.Label(dialog, text="Export folder").grid(row=2, column=0, sticky="w", padx=10, pady=(0, 6))
        dest_var = tk.StringVar(value=str(_export_dest_for_icon_identity(is_icon)))
        dest_entry = ttk.Entry(dialog, textvariable=dest_var, width=72)
        dest_entry.grid(row=2, column=1, sticky="ew", padx=10, pady=(0, 6))

        def _browse_folder() -> None:
            chosen = filedialog.askdirectory(initialdir=dest_var.get() or str(OCR_EXPORT_DEFAULT_DIR))
            if chosen:
                dest_var.set(chosen)

        ttk.Button(dialog, text="Browse...", command=_browse_folder).grid(
            row=2, column=2, sticky="ew", padx=(0, 10), pady=(0, 6)
        )

        button_bar = ttk.Frame(dialog)
        button_bar.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=(2, 10))
        button_bar.columnconfigure(0, weight=1)
        button_bar.columnconfigure(1, weight=1)
        button_bar.columnconfigure(2, weight=1)
        button_bar.columnconfigure(3, weight=1)
        button_bar.columnconfigure(4, weight=1)

        last_export_paths: list[Path] = []

        def _after_export(paths: list[Path], status: str) -> None:
            nonlocal last_export_paths
            last_export_paths = list(paths)
            undo_btn.config(state="normal")
            self.status_var.set(status)

        def _undo_last_export() -> None:
            nonlocal last_export_paths
            if not last_export_paths:
                return
            try:
                n = _undo_export_files(last_export_paths)
            except Exception as exc:
                self.status_var.set(f"Undo export failed: {type(exc).__name__}: {exc}")
                return
            last_export_paths = []
            undo_btn.config(state="disabled")
            self.status_var.set(f"Undid export ({n} file(s) removed) for item #{display_idx + 1}")

        def _save_text() -> None:
            new_text = text_var.get().strip()
            self.current_lines[display_idx] = OcrLine(
                box=line.box,
                text=new_text,
                line_type=line.line_type,
                class_name=line.class_name,
                class_id=line.class_id,
                chinese_ids=line.chinese_ids,
            )
            self._populate_item_list()
            self.item_list.select_clear(0, tk.END)
            self.item_list.select_set(display_idx)
            self.item_list.see(display_idx)
            self.selected_line_idx = display_idx
            self._refresh_image()
            self.status_var.set(f"Updated OCR text for item #{display_idx + 1}")

        def _export_current() -> None:
            _save_text()
            corrected = text_var.get().strip()
            export_as_icon = icon_var.get()
            dest_dir = (
                OCR_EXPORT_ICONS_DIR if export_as_icon else Path(dest_var.get().strip())
            )
            try:
                paths = self._export_display_item(
                    display_idx,
                    dest_dir,
                    label_text=corrected,
                    export_as_icon=export_as_icon,
                )
            except Exception as exc:
                self.status_var.set(f"Export failed: {type(exc).__name__}: {exc}")
                return
            kind = "image + label" if len(paths) > 1 else "image"
            _after_export(paths, f"Exported {kind} for item #{display_idx + 1} → {dest_dir}")

        def _export_to_validate() -> None:
            _save_text()
            try:
                paths = self._export_display_item(
                    display_idx,
                    OCR_VALIDATE_DIR,
                    label_text=text_var.get().strip(),
                    export_as_icon=icon_var.get(),
                )
            except Exception as exc:
                self.status_var.set(f"Export to validate failed: {type(exc).__name__}: {exc}")
                return
            kind = "image + label" if len(paths) > 1 else "image"
            _after_export(paths, f"Exported {kind} to validate for item #{display_idx + 1}")

        ttk.Button(button_bar, text="Save Text", command=_save_text).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_bar, text="Export", command=_export_current).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(button_bar, text="Export Validate", command=_export_to_validate).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        undo_btn = ttk.Button(button_bar, text="Undo Export", command=_undo_last_export, state="disabled")
        undo_btn.grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(button_bar, text="Close", command=dialog.destroy).grid(row=0, column=4, sticky="ew", padx=(4, 0))

        text_entry.focus_set()
        text_entry.selection_range(0, tk.END)
        dialog.bind("<Return>", lambda _event: _save_text())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())

    def _open_ui_object_export_popup(self, display_idx: int) -> None:
        line = self._line_at_display_index(display_idx)
        if line is None or not _is_ui_object_line(line):
            return
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Export UI Object #{display_idx + 1}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        ttk.Label(dialog, text="Object").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))
        ttk.Label(dialog, text=_display_label_for_line(line), wraplength=520).grid(
            row=0, column=1, columnspan=2, sticky="w", padx=10, pady=(10, 6)
        )

        ttk.Label(dialog, text="Export folder").grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))
        dest_var = tk.StringVar(value=str(UI_EXPORT_DEFAULT_DIR))
        dest_entry = ttk.Entry(dialog, textvariable=dest_var, width=72)
        dest_entry.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 6))

        def _browse_folder() -> None:
            chosen = filedialog.askdirectory(initialdir=dest_var.get() or str(UI_EXPORT_DEFAULT_DIR))
            if chosen:
                dest_var.set(chosen)

        ttk.Button(dialog, text="Browse...", command=_browse_folder).grid(
            row=1, column=2, sticky="ew", padx=(0, 10), pady=(0, 6)
        )

        button_bar = ttk.Frame(dialog)
        button_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(2, 10))
        button_bar.columnconfigure(0, weight=1)
        button_bar.columnconfigure(1, weight=1)
        button_bar.columnconfigure(2, weight=1)

        last_export_paths: list[Path] = []

        def _after_export(paths: list[Path], status: str) -> None:
            nonlocal last_export_paths
            last_export_paths = list(paths)
            undo_btn.config(state="normal")
            self.status_var.set(status)

        def _undo_last_export() -> None:
            nonlocal last_export_paths
            if not last_export_paths:
                return
            try:
                n = _undo_export_files(last_export_paths)
            except Exception as exc:
                self.status_var.set(f"Undo export failed: {type(exc).__name__}: {exc}")
                return
            last_export_paths = []
            undo_btn.config(state="disabled")
            self.status_var.set(f"Undid export ({n} file(s) removed) for UI object #{display_idx + 1}")

        def _export_current() -> None:
            try:
                dest_dir = Path(dest_var.get().strip())
                paths = self._export_display_item(display_idx, dest_dir, label_text=None)
            except Exception as exc:
                self.status_var.set(f"Export failed: {type(exc).__name__}: {exc}")
                return
            _after_export(
                paths,
                f"Exported image (no label) for UI object #{display_idx + 1} → {dest_dir}",
            )

        ttk.Button(button_bar, text="Export", command=_export_current).grid(row=0, column=0, sticky="ew", padx=4)
        undo_btn = ttk.Button(button_bar, text="Undo Export", command=_undo_last_export, state="disabled")
        undo_btn.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(button_bar, text="Close", command=dialog.destroy).grid(row=0, column=2, sticky="ew", padx=4)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())

    def _export_display_item(
        self,
        display_idx: int,
        dest_dir: Path,
        *,
        label_text: str | None,
        export_as_icon: bool = False,
    ) -> list[Path]:
        """Export one item crop. UI objects and icon identity write ``.png`` only; text OCR lines also write ``.txt``."""
        if self.current_image is None:
            raise ValueError("no image loaded")
        line = self._line_at_display_index(display_idx)
        if line is None:
            raise ValueError("invalid item index")
        write_txt = not _is_ui_object_line(line) and not export_as_icon
        if write_txt and not (label_text or "").strip():
            raise ValueError("corrected text is empty")
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = self._current_image_path()
        base_name = src.stem if src is not None else "image"
        img_w, img_h = self.current_image.size
        x, y, w, h = line.box
        if w <= 0 or h <= 0:
            raise ValueError("invalid item box dimensions")
        crop_l = max(0, x)
        crop_t = max(0, y)
        crop_r = min(img_w, x + w)
        crop_b = min(img_h, y + h)
        if crop_r <= crop_l or crop_b <= crop_t:
            raise ValueError("box is outside image bounds")

        crop = self.current_image.crop((crop_l, crop_t, crop_r, crop_b))
        kind = "obj" if _is_ui_object_line(line) else "ocr"
        stem = f"{base_name}_{kind}_item{display_idx + 1:03d}"
        out_img = dest_dir / f"{stem}.png"
        crop.save(out_img)
        written = [out_img]
        if not write_txt:
            return written
        out_txt = dest_dir / f"{stem}.txt"
        out_txt.write_text(label_text or "", encoding="utf-8")
        written.append(out_txt)
        return written

    def _export_line_variants(
        self,
        display_idx: int,
        corrected_text: str,
        dest_dir: Path,
        *,
        export_as_icon: bool | None = None,
    ) -> list[Path]:
        """Backward-compatible wrapper for OCR export callers."""
        if export_as_icon is None:
            export_as_icon = _is_pua_icon_identity_text(corrected_text)
        return self._export_display_item(
            display_idx,
            dest_dir,
            label_text=corrected_text,
            export_as_icon=export_as_icon,
        )

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        base_image = self.current_image
        if self.show_color_regions.get() and self.segment_result is not None:
            base_image = _draw_color_segment_overlays(
                self.current_image,
                self.segment_result,
                show_quantized=False,
                show_boxes=True,
                show_labels=self.show_region_labels.get(),
                selected_region_id=self.selected_region_id,
            )
        rendered = _draw_overlays(
            base_image,
            self._all_display_lines(),
            show_boxes=self.show_boxes.get(),
            show_labels=False,
            selected_idx=self.selected_line_idx,
            rectangle_boxes=self._input_box_rectangle_overlay(),
            rectangle_box_color="blue",
        )

        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        img_w, img_h = rendered.size
        fit = min(canvas_w / img_w, canvas_h / img_h)
        fit = min(1.0, fit)
        scale = max(1e-6, fit * self._view_zoom)
        self._render_scale = scale
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        if new_size != (img_w, img_h):
            rendered = rendered.resize(new_size, Image.Resampling.LANCZOS)

        self.current_display = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.current_display, anchor="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _select_ocr_at_canvas_event(self, event: tk.Event[tk.Canvas]) -> None:
        selected_idx = self._ocr_hit_index_at_canvas(event)
        self.item_list.select_clear(0, tk.END)
        if selected_idx is None:
            self.selected_line_idx = None
            self._refresh_image()
            return

        self.canvas.focus_set()
        self.selected_line_idx = selected_idx
        self.item_list.select_set(selected_idx)
        self.item_list.see(selected_idx)
        self._refresh_image()

    def _prev_image(self) -> None:
        idx = self._selected_image_index()
        if idx is None:
            return
        self._select_image_index(max(0, idx - 1))

    def _next_image(self) -> None:
        idx = self._selected_image_index()
        if idx is None:
            return
        self._select_image_index(min(len(self.current_run_images) - 1, idx + 1))

    def _run_select_text_current_image(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected for YOLO detections")
            return
        conf, err = _parse_conf_0_to_1(self.yolo_conf_var.get())
        if conf is None:
            self.status_var.set(f"Invalid confidence: {err}")
            return
        self.status_var.set(f"Running YOLO detections (conf={conf:g})...")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        try:
            lines, status = resolve_image_lines(
                src,
                yolo_conf_threshold=conf,
                allow_yolo=True,
                yolo_cache=self._yolo_lines_cache,
                force_yolo=True,
                run_dir=self._selected_run(),
            )
        except Exception as exc:
            self.status_var.set(f"YOLO detections failed: {type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._set_current_lines(lines)
        self.selected_line_idx = None
        self.selected_region_id = None
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        self._detect_input_box_rectangles()
        self._refresh_image()
        self.status_var.set(
            f"{status} in {elapsed_ms:.0f} ms{self._spatial_segment_status_suffix()}"
        )

    def _copy_current_image_to_undone(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected to copy")
            return
        dest_dir = YOLO_UNDONE_IMAGES
        run = self._selected_run()
        folder_name = run.name if run is not None else src.parent.name
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"cua_{folder_name}_{src.name}"
            shutil.copy2(src, dest)
            self.status_var.set(f"Copied to {dest}")
        except OSError as exc:
            self.status_var.set(f"Copy failed: {exc}")

    def _copy_all_run_images_to_undone(self) -> None:
        run = self._selected_run()
        if run is None:
            self.status_var.set("No run selected")
            return
        images = [p for p in self.current_run_images if p.is_file()]
        if not images:
            self.status_var.set("No images in run to copy")
            return
        dest_dir = YOLO_UNDONE_IMAGES
        folder_name = run.name
        copied = 0
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in images:
                dest = dest_dir / f"cua_{folder_name}_{src.name}"
                shutil.copy2(src, dest)
                copied += 1
            self.status_var.set(f"Copied {copied} images to {dest_dir}")
        except OSError as exc:
            self.status_var.set(f"Copied {copied}/{len(images)} then failed: {exc}")

def run_app(
    runs_root: Path | None = None,
    *,
    recordings_root: Path | None = None,
    images_dir: Path | None = None,
    initial_tab: str = "runs",
) -> None:
    root = tk.Tk()
    CombinedImageViewerApp(
        root,
        runs_root=runs_root,
        recordings_root=recordings_root,
        images_dir=images_dir,
        initial_tab=initial_tab,
    )
    root.mainloop()


class TestImagesViewerApp:
    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4
    _RMB_ZOOM_PER_PIXEL = 1.0012

    def __init__(
        self,
        parent: tk.Misc,
        images_dir: Path,
        *,
        manage_window: bool = True,
        bind_global_hotkeys: bool = True,
    ):
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.images_dir = images_dir
        self._manage_window = manage_window
        self._bind_global_hotkeys = bind_global_hotkeys
        self._hotkeys_active = False
        self.image_paths: list[Path] = _discover_folder_images(images_dir)
        self.current_display: ImageTk.PhotoImage | None = None
        self.current_image: Image.Image | None = None
        self.current_lines: list[OcrLine] = []
        self.selected_line_idx: int | None = None
        self._ultralytics_model_holder: list[Any] = []

        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_color_regions = tk.BooleanVar(value=True)
        self.show_region_labels = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.folder_var = tk.StringVar(value=str(images_dir))
        self.yolo_conf_var = tk.StringVar(value=f"{DEFAULT_CONF_YOLOV26_END2END:g}")
        self.box_edit_mode = tk.StringVar(value="expand")

        self._view_zoom = 1.0
        self._rmb_last_x: int | None = None
        self._render_scale = 1.0
        self._lmb_press_xy: tuple[int, int] | None = None
        self._lmb_panning = False

        self._ui_font = _configure_ui_fonts(self.root, UI_FONT_SIZE)
        self._overlay_font = _pil_overlay_font(OVERLAY_FONT_SIZE)
        self.segment_result: ColorSegmentResult | None = None
        self.selected_region_id: int | None = None
        self._spatial_rank_by_box: dict[tuple[int, int, int, int], int] = {}

        self._build_ui()
        self._reload_image_list()
        if bind_global_hotkeys:
            self.activate_hotkeys()

    def _build_ui(self) -> None:
        if self._manage_window and isinstance(self.parent, tk.Tk):
            self.parent.title("Test Images — YOLO & OCR")
            self.parent.geometry("1280x840")
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)

        left = ttk.Frame(self.parent, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        ttk.Label(left, text="Folder").grid(row=0, column=0, sticky="w")
        ttk.Label(left, text="Images").grid(row=0, column=1, sticky="w", padx=(8, 0))
        left.rowconfigure(1, weight=1)
        folder_wrap = ttk.Frame(left)
        folder_wrap.grid(row=1, column=0, sticky="nsew")
        folder_wrap.columnconfigure(0, weight=1)
        ttk.Entry(folder_wrap, textvariable=self.folder_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_wrap, text="Browse…", command=self._browse_folder).grid(
            row=1, column=0, sticky="ew", pady=(4, 0)
        )

        image_wrap = ttk.Frame(left)
        image_wrap.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        image_wrap.columnconfigure(0, weight=1)
        image_wrap.rowconfigure(0, weight=1)
        self.image_list = tk.Listbox(
            image_wrap, exportselection=False, height=10, width=24, font=self._ui_font
        )
        self.image_list.grid(row=0, column=0, sticky="nsew")
        self.image_scroll = ttk.Scrollbar(image_wrap, orient="vertical", command=self.image_list.yview)
        self.image_scroll.grid(row=0, column=1, sticky="ns")
        self.image_list.configure(yscrollcommand=self.image_scroll.set)
        self.image_list.bind("<<ListboxSelect>>", self._on_image_select)

        ttk.Label(left, text="Color regions").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        region_wrap = ttk.Frame(left)
        region_wrap.grid(row=3, column=0, columnspan=2, sticky="ew")
        region_wrap.columnconfigure(0, weight=1)
        self.region_list = tk.Listbox(
            region_wrap, exportselection=False, height=5, width=40, font=self._ui_font
        )
        self.region_list.grid(row=0, column=0, sticky="ew")
        self.region_scroll = ttk.Scrollbar(
            region_wrap, orient="vertical", command=self.region_list.yview
        )
        self.region_scroll.grid(row=0, column=1, sticky="ns")
        self.region_list.configure(yscrollcommand=self.region_scroll.set)
        self.region_list.bind("<<ListboxSelect>>", self._on_region_select)

        ttk.Label(left, text="OCR / detection items (rank vs #1)").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        item_wrap = ttk.Frame(left)
        item_wrap.grid(row=5, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(5, weight=1)
        item_wrap.columnconfigure(0, weight=1)
        item_wrap.rowconfigure(0, weight=1)
        self.item_list = tk.Listbox(
            item_wrap, exportselection=False, height=10, width=40, font=self._ui_font
        )
        self.item_list.grid(row=0, column=0, sticky="nsew")
        self.item_scroll = ttk.Scrollbar(item_wrap, orient="vertical", command=self.item_list.yview)
        self.item_scroll.grid(row=0, column=1, sticky="ns")
        self.item_list.configure(yscrollcommand=self.item_scroll.set)
        self.item_list.bind("<<ListboxSelect>>", self._on_item_select)
        self.item_list.bind("<Double-Button-1>", self._on_item_double_click)
        self.item_list.bind("<Delete>", self._delete_selected_detection)

        controls = ttk.Frame(left)
        controls.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for col in range(4):
            controls.columnconfigure(col, weight=1)
        ttk.Checkbutton(
            controls, text="Boxes", variable=self.show_boxes, command=self._refresh_image
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            controls, text="Labels", variable=self.show_labels, command=self._refresh_image
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Color regions",
            variable=self.show_color_regions,
            command=self._refresh_image,
        ).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Region labels",
            variable=self.show_region_labels,
            command=self._refresh_image,
        ).grid(row=0, column=3, sticky="w")
        ttk.Label(controls, text="Arrows").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Radiobutton(
            controls,
            text="Expand",
            variable=self.box_edit_mode,
            value="expand",
        ).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Radiobutton(
            controls,
            text="Shrink",
            variable=self.box_edit_mode,
            value="shrink",
        ).grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Button(controls, text="Prev", command=self._prev_image).grid(
            row=2, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Next", command=self._next_image).grid(
            row=2, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Zoom +", command=self._zoom_in).grid(
            row=2, column=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Zoom -", command=self._zoom_out).grid(
            row=2, column=3, sticky="ew", pady=(6, 0)
        )
        ttk.Button(
            controls,
            text="YOLO text regions (select_text)",
            command=self._run_select_text_regions,
        ).grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(controls, text="YOLO confidence").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.yolo_conf_var, width=10).grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=(6, 0)
        )
        ttk.Button(controls, text="Delete selected", command=self._delete_selected_detection).grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Reset Zoom", command=self._reset_zoom).grid(
            row=6, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )

        canvas_wrap = ttk.Frame(self.parent, padding=8)
        canvas_wrap.grid(row=0, column=1, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, bg="#1e1e1e", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-3>", self._on_rmb_press)
        self.canvas.bind("<B3-Motion>", self._on_rmb_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self.canvas.bind("<ButtonPress-1>", self._on_lmb_press)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<ButtonPress-2>", self._on_mmb_press)
        self.canvas.bind("<B2-Motion>", self._on_mmb_drag)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Delete>", self._delete_selected_detection)

        status = ttk.Label(self.parent, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.canvas.bind(key, self._on_arrow_key)
            self.item_list.bind(key, self._on_arrow_key)
            self.image_list.bind(key, self._on_arrow_key)
        self._bind_shift_box_edit_toggle(self.canvas, self.item_list, self.image_list)
        self.parent.bind("<Configure>", lambda _e: self._refresh_image())

    def activate_hotkeys(self) -> None:
        if self._hotkeys_active:
            return
        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.root.bind(key, self._on_arrow_key)
        self._bind_shift_box_edit_toggle(self.root)
        self.root.bind("<Control-plus>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-equal>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-minus>", self._on_zoom_out_hotkey)
        self.root.bind("<Control-0>", self._on_reset_zoom_hotkey)
        self._hotkeys_active = True

    def deactivate_hotkeys(self) -> None:
        if not self._hotkeys_active:
            return
        for key in (
            "<Up>",
            "<Down>",
            "<Left>",
            "<Right>",
            "<KeyPress-Shift_L>",
            "<KeyPress-Shift_R>",
            "<Control-plus>",
            "<Control-equal>",
            "<Control-minus>",
            "<Control-0>",
        ):
            self.root.unbind(key)
        self._hotkeys_active = False

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory(
            initialdir=str(self.images_dir),
            title="Select test images folder",
        )
        if not chosen:
            return
        self.images_dir = Path(chosen)
        self.folder_var.set(str(self.images_dir))
        self._reload_image_list()

    def _reload_image_list(self) -> None:
        self.image_paths = _discover_folder_images(self.images_dir)
        self.image_list.delete(0, tk.END)
        for img in self.image_paths:
            self.image_list.insert(tk.END, img.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self.current_image = None
            self.current_lines = []
            self.item_list.delete(0, tk.END)
            self.region_list.delete(0, tk.END)
            self.canvas.delete("all")
            self.status_var.set(f"No images in {self.images_dir}")

    def _refresh_spatial_segmentation(self) -> None:
        state = _build_yolo_spatial_segment_state(self.current_image, self.current_lines)
        self.segment_result = state.result
        self._spatial_rank_by_box = dict(state.spatial_ranks)
        if self.selected_region_id is not None and self.segment_result is not None:
            region_ids = {region.region_id for region in self.segment_result.regions}
            if self.selected_region_id not in region_ids:
                self.selected_region_id = None
        self._populate_region_list()

    def _populate_region_list(self) -> None:
        self.region_list.delete(0, tk.END)
        if self.segment_result is None:
            return
        for region in self.segment_result.regions:
            self.region_list.insert(tk.END, _format_color_region_row(region))
        if self.selected_region_id is not None:
            for idx, region in enumerate(self.segment_result.regions):
                if region.region_id == self.selected_region_id:
                    self.region_list.select_set(idx)
                    self.region_list.see(idx)
                    break

    def _spatial_segment_status_suffix(self) -> str:
        if self.segment_result is None:
            return ""
        before = self.segment_result.regions_before_yolo_filter
        kept = len(self.segment_result.regions)
        return f" | color regions: {kept} kept ({before} before YOLO filter)"

    def _on_region_select(self, _event: object | None = None) -> None:
        selected = self.region_list.curselection()
        if not selected or self.segment_result is None:
            self.selected_region_id = None
            self._refresh_image()
            return
        idx = selected[0]
        if idx < 0 or idx >= len(self.segment_result.regions):
            self.selected_region_id = None
            self._refresh_image()
            return
        region = self.segment_result.regions[idx]
        self.selected_region_id = region.region_id
        self._refresh_image()
        x0, y0, x1, y1 = region.bbox
        r, g, b = region.mean_color
        self.status_var.set(
            f"Region #{region.region_id + 1}: ({x0},{y0})-({x1},{y1}), "
            f"area={region.area}px, rgb=({r},{g},{b})"
        )

    def _selected_image_index(self) -> int | None:
        selected = self.image_list.curselection()
        if not selected:
            return None
        return selected[0]

    def _current_image_path(self) -> Path | None:
        idx = self._selected_image_index()
        if idx is None or idx >= len(self.image_paths):
            return None
        return self.image_paths[idx]

    def _on_image_select(self, _event: object | None = None) -> None:
        image_path = self._current_image_path()
        if image_path is None:
            return
        self.current_image = Image.open(image_path).convert("RGB")
        self._view_zoom = 1.0
        json_path = image_path.with_suffix(".json")
        lines, status = load_ocr_lines(json_path)
        self.current_lines = _dedupe_ocr_lines(lines)
        self.selected_line_idx = None
        self.selected_region_id = None
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        self.status_var.set(f"{image_path.name} — {status}{self._spatial_segment_status_suffix()}")
        self._refresh_image()

    def _populate_item_list(self) -> None:
        self.item_list.delete(0, tk.END)
        for row in _yolo_detection_list_rows(
            self.current_lines,
            segment_result=self.segment_result,
            spatial_ranks=self._spatial_rank_by_box,
        ):
            self.item_list.insert(tk.END, row)

    def _on_item_select(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            self.selected_line_idx = None
            self.selected_region_id = None
            self._refresh_image()
            return
        self.selected_line_idx = selected[0]
        if (
            self.selected_line_idx is not None
            and self.segment_result is not None
            and 0 <= self.selected_line_idx < len(self.current_lines)
        ):
            line = self.current_lines[self.selected_line_idx]
            self.selected_region_id = region_id_for_box(
                self.segment_result.label_map,
                tuple(int(v) for v in line.box),
            )
            if self.selected_region_id is not None:
                for idx, region in enumerate(self.segment_result.regions):
                    if region.region_id == self.selected_region_id:
                        self.region_list.select_clear(0, tk.END)
                        self.region_list.select_set(idx)
                        self.region_list.see(idx)
                        break
        else:
            self.selected_region_id = None
        self._refresh_image()

    def _delete_selected_detection(self, _event: object | None = None) -> str:
        idx = self.selected_line_idx
        if idx is None or idx < 0 or idx >= len(self.current_lines):
            self.status_var.set("Select a detection to delete")
            return "break"

        deleted = self.current_lines.pop(idx)
        self.selected_line_idx = None
        self.selected_region_id = None
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        if self.current_lines:
            next_idx = min(idx, len(self.current_lines) - 1)
            self.item_list.select_set(next_idx)
            self.item_list.see(next_idx)
            self.selected_line_idx = next_idx
        self._refresh_image()
        deleted_row = _agent_format_detection_rows([deleted])
        detail = deleted_row[0] if deleted_row else _display_label_for_line(deleted)
        self.status_var.set(f"Deleted detection: {detail}")
        return "break"

    def _on_item_double_click(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            return
        idx = selected[0]
        if idx < 0 or idx >= len(self.current_lines):
            return
        self.selected_line_idx = idx
        self._refresh_image()
        self._open_item_edit_popup(idx)

    def _open_item_edit_popup(self, idx: int) -> None:
        line = self.current_lines[idx]
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit OCR Item #{idx + 1}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)
        dialog_pad = 12

        ttk.Label(dialog, text="Corrected text").grid(
            row=0, column=0, sticky="w", padx=dialog_pad, pady=(dialog_pad, 8)
        )
        text_var = tk.StringVar(value=line.text)
        text_entry = ttk.Entry(dialog, textvariable=text_var, width=64)
        text_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=dialog_pad, pady=(dialog_pad, 8))

        ttk.Label(dialog, text="Export folder").grid(
            row=1, column=0, sticky="w", padx=dialog_pad, pady=(0, 8)
        )
        dest_var = tk.StringVar(value=str(_export_dest_for_text(line.text)))
        dest_entry = ttk.Entry(dialog, textvariable=dest_var, width=64)
        dest_entry.grid(row=1, column=1, sticky="ew", padx=dialog_pad, pady=(0, 8))

        def _browse_folder() -> None:
            chosen = filedialog.askdirectory(initialdir=dest_var.get() or str(OCR_EXPORT_DEFAULT_DIR))
            if chosen:
                dest_var.set(chosen)

        ttk.Button(dialog, text="Browse...", command=_browse_folder).grid(
            row=1, column=2, sticky="ew", padx=(0, dialog_pad), pady=(0, 8)
        )

        button_bar = ttk.Frame(dialog)
        button_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=dialog_pad, pady=(4, dialog_pad))
        button_bar.columnconfigure(0, weight=1)
        button_bar.columnconfigure(1, weight=1)
        button_bar.columnconfigure(2, weight=1)
        button_bar.columnconfigure(3, weight=1)

        def _save_text() -> None:
            new_text = text_var.get().strip()
            self.current_lines[idx] = OcrLine(box=line.box, text=new_text)
            self._populate_item_list()
            self.item_list.select_clear(0, tk.END)
            self.item_list.select_set(idx)
            self.item_list.see(idx)
            self.selected_line_idx = idx
            self._refresh_image()
            dest_var.set(str(_export_dest_for_text(new_text)))
            self.status_var.set(f"Updated OCR text for item #{idx + 1}")

        def _export_current() -> None:
            _save_text()
            corrected = text_var.get().strip()
            if _is_pua_icon_identity_text(corrected):
                dest_dir = OCR_EXPORT_ICONS_DIR
            else:
                dest_dir = Path(dest_var.get().strip())
            try:
                n = self._export_line_variants(idx, corrected, dest_dir)
            except Exception as exc:
                self.status_var.set(f"Export failed: {type(exc).__name__}: {exc}")
                return
            kind = "image + label" if n > 1 else "image"
            self.status_var.set(f"Exported {kind} for item #{idx + 1} → {dest_dir}")

        def _export_to_validate() -> None:
            _save_text()
            try:
                n = self._export_line_variants(idx, text_var.get().strip(), OCR_VALIDATE_DIR)
            except Exception as exc:
                self.status_var.set(f"Export to validate failed: {type(exc).__name__}: {exc}")
                return
            kind = "image + label" if n > 1 else "image"
            self.status_var.set(f"Exported {kind} to validate for item #{idx + 1}")

        ttk.Button(button_bar, text="Save Text", command=_save_text).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_bar, text="Export", command=_export_current).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(button_bar, text="Export Validate", command=_export_to_validate).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(button_bar, text="Close", command=dialog.destroy).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        text_entry.focus_set()
        text_entry.selection_range(0, tk.END)
        dialog.bind("<Return>", lambda _event: _save_text())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())

    def _export_line_variants(self, idx: int, corrected_text: str, dest_dir: Path) -> int:
        if self.current_image is None:
            raise ValueError("no image loaded")
        if idx < 0 or idx >= len(self.current_lines):
            raise ValueError("invalid OCR item index")
        if not corrected_text:
            raise ValueError("corrected text is empty")
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = self._current_image_path()
        base_name = src.stem if src is not None else "image"
        img_w, img_h = self.current_image.size
        x, y, w, h = self.current_lines[idx].box
        if w <= 0 or h <= 0:
            raise ValueError("invalid OCR item box dimensions")
        crop_l = max(0, x)
        crop_t = max(0, y)
        crop_r = min(img_w, x + w)
        crop_b = min(img_h, y + h)
        if crop_r <= crop_l or crop_b <= crop_t:
            raise ValueError("OCR box is outside image bounds")

        crop = self.current_image.crop((crop_l, crop_t, crop_r, crop_b))
        stem = f"{base_name}_item{idx + 1:03d}"
        out_img = dest_dir / f"{stem}.png"
        crop.save(out_img)
        if _is_pua_icon_identity_text(corrected_text):
            return 1
        out_txt = dest_dir / f"{stem}.txt"
        out_txt.write_text(corrected_text, encoding="utf-8")
        return 2

    def _set_lines(self, lines: list[OcrLine], status: str) -> None:
        self.current_lines = _dedupe_ocr_lines(lines)
        self.selected_line_idx = None
        self.selected_region_id = None
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        self._refresh_image()
        self.status_var.set(f"{status}{self._spatial_segment_status_suffix()}")

    def _run_select_text_regions(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected")
            return
        conf, err = _parse_conf_0_to_1(self.yolo_conf_var.get())
        if conf is None:
            self.status_var.set(f"Invalid confidence: {err}")
            return
        self.status_var.set(f"Running YOLO text regions (select_text, conf={conf:g})…")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        try:
            regions = ocr_regions_from_image_path(str(src), yolo_conf_threshold=conf)
        except Exception as exc:
            self.status_var.set(f"YOLO text regions failed: {type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        lines = _regions_to_ocr_lines(regions)
        try:
            out_path = _save_test_image_ocr_json(
                src,
                lines,
                yolo_elapsed_ms=elapsed_ms,
                ocr_elapsed_ms=None,
                source="select_text_regions",
            )
            save_note = f", saved {out_path.name}"
        except Exception as exc:
            save_note = f", save failed: {exc}"
        self._set_lines(
            lines,
            f"YOLO text regions: {len(lines)} regions in {elapsed_ms:.0f} ms{save_note}",
        )

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        base_image = self.current_image
        if self.show_color_regions.get() and self.segment_result is not None:
            base_image = _draw_color_segment_overlays(
                self.current_image,
                self.segment_result,
                show_quantized=False,
                show_boxes=True,
                show_labels=self.show_region_labels.get(),
                selected_region_id=self.selected_region_id,
            )
        rendered = _draw_overlays(
            base_image,
            self.current_lines,
            show_boxes=self.show_boxes.get(),
            show_labels=self.show_labels.get(),
            selected_idx=self.selected_line_idx,
            overlay_font=self._overlay_font,
        )
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        img_w, img_h = rendered.size
        fit = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        scale = max(1e-6, fit * self._view_zoom)
        self._render_scale = scale
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        if new_size != (img_w, img_h):
            rendered = rendered.resize(new_size, Image.Resampling.LANCZOS)
        self.current_display = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.current_display, anchor="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _prev_image(self) -> None:
        idx = self._selected_image_index()
        if idx is None:
            return
        nxt = max(0, idx - 1)
        self.image_list.select_clear(0, tk.END)
        self.image_list.select_set(nxt)
        self.image_list.see(nxt)
        self._on_image_select()

    def _next_image(self) -> None:
        idx = self._selected_image_index()
        if idx is None:
            return
        nxt = min(len(self.image_paths) - 1, idx + 1)
        self.image_list.select_clear(0, tk.END)
        self.image_list.select_set(nxt)
        self.image_list.see(nxt)
        self._on_image_select()

    def _set_zoom(self, zoom: float) -> None:
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, zoom))
        self._refresh_image()

    def _apply_zoom_factor(self, factor: float) -> None:
        self._set_zoom(self._view_zoom * factor)

    def _zoom_in(self) -> None:
        self._apply_zoom_factor(self._ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._apply_zoom_factor(1.0 / self._ZOOM_STEP)

    def _reset_zoom(self) -> None:
        self._set_zoom(1.0)

    def _on_zoom_in_hotkey(self, _event: tk.Event[tk.Tk]) -> str:
        self._zoom_in()
        return "break"

    def _on_zoom_out_hotkey(self, _event: tk.Event[tk.Tk]) -> str:
        self._zoom_out()
        return "break"

    def _on_reset_zoom_hotkey(self, _event: tk.Event[tk.Tk]) -> str:
        self._reset_zoom()
        return "break"

    def _arrow_direction(self, keysym: str) -> str | None:
        return {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}.get(keysym)

    def _bind_shift_box_edit_toggle(self, *widgets: tk.Misc) -> None:
        for widget in widgets:
            for key in ("<KeyPress-Shift_L>", "<KeyPress-Shift_R>"):
                widget.bind(key, self._on_shift_toggle_box_mode)

    def _on_shift_toggle_box_mode(self, _event: tk.Event) -> str:
        toggle_box_edit_mode(self.box_edit_mode, self.status_var)
        return "break"

    def _adjust_selected_box(self, direction: str, *, step: int) -> bool:
        idx = self.selected_line_idx
        if idx is None or self.current_image is None or idx < 0 or idx >= len(self.current_lines):
            return False
        img_w, img_h = self.current_image.size
        line = self.current_lines[idx]
        expand = self.box_edit_mode.get() == "expand"
        new_box = _adjust_box_edge(
            line.box,
            direction,
            expand=expand,
            step=step,
            img_w=img_w,
            img_h=img_h,
        )
        if new_box == line.box:
            return False
        self.current_lines[idx] = OcrLine(box=new_box, text=line.text)
        self._refresh_spatial_segmentation()
        self._populate_item_list()
        self._refresh_image()
        mode = "Expand" if expand else "Shrink"
        x, y, w, h = new_box
        self.status_var.set(f"{mode} box #{idx + 1}: ({x},{y}) {w}×{h}")
        return True

    def _on_arrow_key(self, event: tk.Event) -> str | None:
        direction = self._arrow_direction(event.keysym)
        if direction is None:
            return None
        if self.selected_line_idx is not None:
            step = BOX_EDIT_STEP_SHIFT if (event.state & 0x0001) else BOX_EDIT_STEP
            if self._adjust_selected_box(direction, step=step):
                return "break"
            return "break"
        if event.keysym == "Left":
            self._prev_image()
            return "break"
        if event.keysym == "Right":
            self._next_image()
            return "break"
        return None

    def _on_rmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        self._rmb_last_x = int(event.x) if self.current_image is not None else None

    def _on_rmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._rmb_last_x is None or self.current_image is None:
            return
        x = int(event.x)
        dx = x - self._rmb_last_x
        self._rmb_last_x = x
        if dx == 0:
            return
        z = self._view_zoom * (self._RMB_ZOOM_PER_PIXEL**dx)
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, z))
        self._refresh_image()

    def _on_rmb_release(self, _event: tk.Event[tk.Canvas]) -> None:
        self._rmb_last_x = None

    def _on_lmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            return
        self._lmb_press_xy = (int(event.x), int(event.y))
        self._lmb_panning = False

    def _on_lmb_motion(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        x0, y0 = self._lmb_press_xy
        x, y = int(event.x), int(event.y)
        if not self._lmb_panning:
            if (x - x0) ** 2 + (y - y0) ** 2 < self._PAN_CLICK_THRESHOLD_SQ:
                return
            self.canvas.scan_mark(x0, y0)
            self._lmb_panning = True
        self.canvas.scan_dragto(x, y, gain=1)

    def _on_lmb_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        try:
            if not self._lmb_panning:
                self._select_ocr_at_canvas_event(event)
        finally:
            self._lmb_press_xy = None
            self._lmb_panning = False

    def _ocr_hit_index_at_canvas(self, event: tk.Event[tk.Canvas]) -> int | None:
        if self.current_image is None or not self.current_lines:
            return None
        canvas_x = self.canvas.canvasx(int(event.x))
        canvas_y = self.canvas.canvasy(int(event.y))
        img_x = int(canvas_x / max(self._render_scale, 1e-6))
        img_y = int(canvas_y / max(self._render_scale, 1e-6))
        return _smallest_box_hit_index(self.current_lines, img_x, img_y)

    def _on_canvas_double_click(self, event: tk.Event[tk.Canvas]) -> None:
        idx = self._ocr_hit_index_at_canvas(event)
        if idx is None:
            return
        self.selected_line_idx = idx
        self.item_list.select_clear(0, tk.END)
        self.item_list.select_set(idx)
        self.item_list.see(idx)
        self._refresh_image()
        self._open_item_edit_popup(idx)

    def _select_ocr_at_canvas_event(self, event: tk.Event[tk.Canvas]) -> None:
        selected_idx = self._ocr_hit_index_at_canvas(event)
        self.item_list.select_clear(0, tk.END)
        if selected_idx is None:
            self.selected_line_idx = None
            self._refresh_image()
            return
        self.canvas.focus_set()
        self.selected_line_idx = selected_idx
        self.item_list.select_set(selected_idx)
        self.item_list.see(selected_idx)
        self._refresh_image()

    def _on_mmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_mark(int(event.x), int(event.y))

    def _on_mmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_dragto(int(event.x), int(event.y), gain=1)

    def _on_canvas_mousewheel(self, event: tk.Event[tk.Canvas]) -> None:
        if event.state & 0x0004:
            if event.delta > 0:
                self._apply_zoom_factor(self._ZOOM_STEP)
            elif event.delta < 0:
                self._apply_zoom_factor(1.0 / self._ZOOM_STEP)
            return
        if event.state & 0x0001:
            self.canvas.xview_scroll(int(-(event.delta / 120)), "units")
        else:
            self.canvas.yview_scroll(int(-(event.delta / 120)), "units")


class LineSegmentsViewerApp:
    """Browse images and tune Canny / Hough line-segment detection.

    ``mode="folder"`` browses a flat image directory (Test images).
    ``mode="sessions"`` browses run/recording session folders like the YOLO tab.
    """

    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4
    _RMB_ZOOM_PER_PIXEL = 1.0012
    _PARAM_DETECT_DEBOUNCE_MS = 2000

    def __init__(
        self,
        parent: tk.Misc,
        source_root: Path,
        *,
        mode: str = "folder",
        session_list_label: str = "Runs",
        manage_window: bool = True,
        bind_global_hotkeys: bool = True,
    ):
        if mode not in ("folder", "sessions"):
            raise ValueError(f"Unsupported LineSegmentsViewerApp mode: {mode!r}")
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.source_root = source_root
        self.mode = mode
        self._session_list_label = session_list_label
        self._manage_window = manage_window
        self._bind_global_hotkeys = bind_global_hotkeys
        self._hotkeys_active = False
        self.session_dirs: list[Path] = (
            _discover_runs(source_root) if mode == "sessions" else []
        )
        self.image_paths: list[Path] = []
        self.current_display: ImageTk.PhotoImage | None = None
        self.current_image: Image.Image | None = None
        self.raw_line_segments: list[tuple[int, int, int, int]] = []
        self.merged_line_segments: list[tuple[int, int, int, int]] = []
        self.candidate_line_segments: list[tuple[int, int, int, int]] = []
        self.vertical_line_segments: list[tuple[int, int, int, int]] = []
        self.rectangle_boxes: list[tuple[int, int, int, int]] = []
        self.selected_rectangle_idx: int | None = None
        self.show_raw_lines = tk.BooleanVar(value=False)
        self.show_merged_lines = tk.BooleanVar(value=False)
        self.show_candidate_lines = tk.BooleanVar(value=False)
        self.show_vertical_lines = tk.BooleanVar(value=False)
        self.show_rectangles = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Ready")
        self.folder_var = tk.StringVar(value=str(source_root))
        self.param_vars: dict[str, tk.DoubleVar] = {}
        self.param_value_vars: dict[str, tk.StringVar] = {}
        self._param_decimals: dict[str, int] = {}
        self._suppress_param_events = True
        self._detect_after_id: str | None = None
        saved = _load_line_segment_params()
        saved_values = {
            "blur_ksize": float(saved.blur_ksize),
            "canny_low": float(saved.canny_low),
            "canny_high": float(saved.canny_high),
            "rho": float(saved.rho),
            "theta_deg": float(saved.theta_deg),
            "threshold": float(saved.threshold),
            "min_line_length": float(saved.min_line_length),
            "max_line_gap": float(saved.max_line_gap),
            "min_width_over_height": float(saved.min_width_over_height),
            "min_overlap_frac": float(saved.min_overlap_frac),
            "min_height": float(saved.min_height),
            "vertical_merge_gap": float(saved.vertical_merge_gap),
        }
        for key, _label, _lo, _hi, default, decimals, _step in _LINE_PARAM_SLIDERS:
            self.param_vars[key] = tk.DoubleVar(value=saved_values.get(key, float(default)))
            self.param_value_vars[key] = tk.StringVar()
            self._param_decimals[key] = decimals
            self._update_param_value_label(key)

        self._view_zoom = 1.0
        self._rmb_last_x: int | None = None
        self._render_scale = 1.0
        self._lmb_press_xy: tuple[int, int] | None = None
        self._lmb_panning = False

        self._ui_font = _configure_ui_fonts(self.root, UI_FONT_SIZE)
        self._build_ui()
        self._suppress_param_events = False
        if self.mode == "folder":
            self._reload_folder_images()
        else:
            self._populate_session_list()
        if bind_global_hotkeys:
            self.activate_hotkeys()

    def _build_ui(self) -> None:
        if self._manage_window and isinstance(self.parent, tk.Tk):
            self.parent.title("Line Segments")
            self.parent.geometry("1280x840")
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)

        left = ttk.Frame(self.parent, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        row = 0
        if self.mode == "folder":
            ttk.Label(left, text="Folder").grid(row=row, column=0, sticky="w")
            ttk.Label(left, text="Images").grid(row=row, column=1, sticky="w", padx=(8, 0))
            row += 1
            left.rowconfigure(row, weight=1)
            folder_wrap = ttk.Frame(left)
            folder_wrap.grid(row=row, column=0, sticky="nsew")
            folder_wrap.columnconfigure(0, weight=1)
            ttk.Entry(folder_wrap, textvariable=self.folder_var).grid(row=0, column=0, sticky="ew")
            ttk.Button(folder_wrap, text="Browse…", command=self._browse_folder).grid(
                row=1, column=0, sticky="ew", pady=(4, 0)
            )
            image_wrap = ttk.Frame(left)
            image_wrap.grid(row=row, column=1, sticky="nsew", padx=(8, 0))
            image_wrap.columnconfigure(0, weight=1)
            image_wrap.rowconfigure(0, weight=1)
            self.image_list = tk.Listbox(
                image_wrap, exportselection=False, height=10, width=24, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            self.image_scroll = ttk.Scrollbar(
                image_wrap, orient="vertical", command=self.image_list.yview
            )
            self.image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=self.image_scroll.set)
            self.image_list.bind("<<ListboxSelect>>", self._on_image_select)
            row += 1
        else:
            ttk.Label(left, text=self._session_list_label).grid(row=row, column=0, sticky="w")
            ttk.Label(left, text="Images").grid(row=row, column=1, sticky="w", padx=(8, 0))
            row += 1
            left.rowconfigure(row, weight=1)
            session_wrap = ttk.Frame(left)
            session_wrap.grid(row=row, column=0, sticky="nsew")
            session_wrap.columnconfigure(0, weight=1)
            session_wrap.rowconfigure(0, weight=1)
            self.session_list = tk.Listbox(
                session_wrap, exportselection=False, height=10, width=24, font=self._ui_font
            )
            self.session_list.grid(row=0, column=0, sticky="nsew")
            self.session_scroll = ttk.Scrollbar(
                session_wrap, orient="vertical", command=self.session_list.yview
            )
            self.session_scroll.grid(row=0, column=1, sticky="ns")
            self.session_list.configure(yscrollcommand=self.session_scroll.set)
            self.session_list.bind("<<ListboxSelect>>", self._on_session_select)
            image_wrap = ttk.Frame(left)
            image_wrap.grid(row=row, column=1, sticky="nsew", padx=(8, 0))
            image_wrap.columnconfigure(0, weight=1)
            image_wrap.rowconfigure(0, weight=1)
            self.image_list = tk.Listbox(
                image_wrap, exportselection=False, height=10, width=24, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            self.image_scroll = ttk.Scrollbar(
                image_wrap, orient="vertical", command=self.image_list.yview
            )
            self.image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=self.image_scroll.set)
            self.image_list.bind("<<ListboxSelect>>", self._on_image_select)
            row += 1

        params = ttk.LabelFrame(left, text="Line segment parameters", padding=6)
        params.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        params.columnconfigure(1, weight=1)
        self._param_steps: dict[str, float] = {}
        self._param_bounds: dict[str, tuple[float, float]] = {}
        for prow, (key, label, lo, hi, _default, _decimals, step) in enumerate(
            _LINE_PARAM_SLIDERS
        ):
            self._param_steps[key] = float(step)
            self._param_bounds[key] = (float(lo), float(hi))
            name = ttk.Label(params, text=label)
            name.grid(row=prow, column=0, sticky="w", pady=1)
            scale = ttk.Scale(
                params,
                from_=lo,
                to=hi,
                orient="horizontal",
                variable=self.param_vars[key],
                command=lambda _v, k=key: self._on_param_scale(k),
            )
            scale.grid(row=prow, column=1, sticky="ew", padx=(6, 2), pady=1)
            dec_btn = ttk.Button(
                params,
                text="◀",
                width=2,
                command=lambda k=key: self._nudge_param(k, -1),
            )
            dec_btn.grid(row=prow, column=2, sticky="e", padx=(2, 0), pady=1)
            value_label = ttk.Label(params, textvariable=self.param_value_vars[key], width=5)
            value_label.grid(row=prow, column=3, sticky="e", pady=1)
            inc_btn = ttk.Button(
                params,
                text="▶",
                width=2,
                command=lambda k=key: self._nudge_param(k, 1),
            )
            inc_btn.grid(row=prow, column=4, sticky="e", padx=(0, 0), pady=1)
            tip = _LINE_PARAM_TOOLTIPS.get(key, "")
            if tip:
                _attach_tooltip(name, tip)
                _attach_tooltip(scale, tip)
                _attach_tooltip(value_label, tip)
                _attach_tooltip(dec_btn, tip)
                _attach_tooltip(inc_btn, tip)
        row += 1

        controls = ttk.Frame(left)
        controls.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for col in range(2):
            controls.columnconfigure(col, weight=1)
        ttk.Checkbutton(
            controls,
            text="Raw lines",
            variable=self.show_raw_lines,
            command=self._refresh_image,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Merged",
            variable=self.show_merged_lines,
            command=self._refresh_image,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            controls,
            text="H-pair candidates",
            variable=self.show_candidate_lines,
            command=self._refresh_image,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Vertical lines",
            variable=self.show_vertical_lines,
            command=self._refresh_image,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Rectangles",
            variable=self.show_rectangles,
            command=self._refresh_image,
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Button(controls, text="Detect", command=self._detect_line_segments_now).grid(
            row=3, column=0, sticky="ew", pady=(6, 0), padx=(0, 3)
        )
        ttk.Button(controls, text="Clear overlay", command=self._clear_line_segments).grid(
            row=3, column=1, sticky="ew", pady=(6, 0), padx=(3, 0)
        )

        canvas_wrap = ttk.Frame(self.parent, padding=8)
        canvas_wrap.grid(row=0, column=1, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, bg="#1e1e1e", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-3>", self._on_rmb_press)
        self.canvas.bind("<B3-Motion>", self._on_rmb_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self.canvas.bind("<ButtonPress-1>", self._on_lmb_press)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<ButtonPress-2>", self._on_mmb_press)
        self.canvas.bind("<B2-Motion>", self._on_mmb_drag)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)

        status = ttk.Label(self.parent, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self.parent.bind("<Configure>", lambda _e: self._refresh_image())

    def activate_hotkeys(self) -> None:
        if self._hotkeys_active:
            return
        self.root.bind("<Control-plus>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-equal>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-minus>", self._on_zoom_out_hotkey)
        self.root.bind("<Control-0>", self._on_reset_zoom_hotkey)
        self._hotkeys_active = True

    def deactivate_hotkeys(self) -> None:
        if not self._hotkeys_active:
            return
        for key in ("<Control-plus>", "<Control-equal>", "<Control-minus>", "<Control-0>"):
            self.root.unbind(key)
        self._hotkeys_active = False

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory(
            initialdir=self.folder_var.get() or str(self.source_root)
        )
        if not chosen:
            return
        self.source_root = Path(chosen)
        self.folder_var.set(str(self.source_root))
        self._reload_folder_images()

    def _populate_session_list(self) -> None:
        self.session_dirs = _discover_runs(self.source_root)
        self.session_list.delete(0, tk.END)
        for session in self.session_dirs:
            self.session_list.insert(tk.END, session.name)
        if self.session_dirs:
            self.session_list.select_set(0)
            self._on_session_select()
        else:
            self.image_paths = []
            self.image_list.delete(0, tk.END)
            self.current_image = None
            self._clear_segment_state()
            self.canvas.delete("all")
            label = self._session_list_label.lower()
            self.status_var.set(f"No {label} found in {self.source_root}")

    def _on_session_select(self, _event: object | None = None) -> None:
        selected = self.session_list.curselection()
        if not selected:
            return
        session = self.session_dirs[selected[0]]
        self.image_paths = _yolo_ocr_paired_images(session)
        self.image_list.delete(0, tk.END)
        for img in self.image_paths:
            self.image_list.insert(tk.END, img.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self._cancel_pending_detect()
            self.current_image = None
            self._clear_segment_state()
            self.canvas.delete("all")
            self.status_var.set(f"No images found for {session.name}")

    def _reload_folder_images(self) -> None:
        self.image_paths = _discover_folder_images(self.source_root)
        self.image_list.delete(0, tk.END)
        for img in self.image_paths:
            self.image_list.insert(tk.END, img.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self.current_image = None
            self._clear_segment_state()
            self.canvas.delete("all")
            self.status_var.set(f"No images in {self.source_root}")
    def _selected_image_index(self) -> int | None:
        selected = self.image_list.curselection()
        if not selected:
            return None
        return selected[0]

    def _current_image_path(self) -> Path | None:
        idx = self._selected_image_index()
        if idx is None or idx >= len(self.image_paths):
            return None
        return self.image_paths[idx]

    def _on_image_select(self, _event: object | None = None) -> None:
        image_path = self._current_image_path()
        if image_path is None:
            return
        self._cancel_pending_detect()
        self.current_image = Image.open(image_path).convert("RGB")
        self._clear_segment_state()
        self._view_zoom = 1.0
        self.status_var.set(image_path.name)
        self._refresh_image()

    def _update_param_value_label(self, key: str) -> None:
        value = float(self.param_vars[key].get())
        if key == "blur_ksize":
            snapped = int(round(value))
            if snapped % 2 == 0:
                snapped = max(1, snapped - 1)
            self.param_value_vars[key].set(str(snapped))
            return
        decimals = self._param_decimals[key]
        if decimals <= 0:
            self.param_value_vars[key].set(str(int(round(value))))
        else:
            self.param_value_vars[key].set(f"{value:.{decimals}f}")

    def _on_param_scale(self, key: str) -> None:
        if key == "blur_ksize":
            snapped = int(round(float(self.param_vars[key].get())))
            if snapped % 2 == 0:
                snapped = max(1, snapped - 1)
            current = float(self.param_vars[key].get())
            if abs(current - snapped) > 1e-6:
                was_suppressed = self._suppress_param_events
                self._suppress_param_events = True
                try:
                    self.param_vars[key].set(float(snapped))
                finally:
                    self._suppress_param_events = was_suppressed
        self._update_param_value_label(key)
        if self._suppress_param_events:
            return
        self._schedule_detect()

    def _nudge_param(self, key: str, direction: int) -> None:
        """Increment or decrement ``key`` by one configured unit."""
        lo, hi = self._param_bounds[key]
        step = self._param_steps[key]
        current = float(self.param_vars[key].get())
        if key == "blur_ksize":
            current = int(round(current))
            if current % 2 == 0:
                current = max(1, current - 1)
        decimals = self._param_decimals[key]
        nxt = current + (step * direction)
        nxt = max(lo, min(hi, nxt))
        if decimals <= 0:
            nxt = float(int(round(nxt)))
        else:
            nxt = round(nxt, decimals)
        if key == "blur_ksize":
            snapped = int(round(nxt))
            if snapped % 2 == 0:
                snapped = max(1, snapped + (1 if direction > 0 else -1))
            nxt = float(max(int(lo), min(int(hi), snapped)))
            if int(nxt) % 2 == 0:
                nxt = float(max(1, int(nxt) - 1))
        if abs(nxt - current) < 1e-9:
            return
        self.param_vars[key].set(nxt)
        self._update_param_value_label(key)
        if not self._suppress_param_events:
            self._schedule_detect()

    def _cancel_pending_detect(self) -> None:
        if self._detect_after_id is not None:
            try:
                self.root.after_cancel(self._detect_after_id)
            except tk.TclError:
                pass
            self._detect_after_id = None

    def _schedule_detect(self) -> None:
        self._cancel_pending_detect()
        try:
            _save_line_segment_params(self._read_params())
        except OSError:
            pass
        self.status_var.set("Parameters changed — detecting in 2s...")
        self._detect_after_id = self.root.after(
            self._PARAM_DETECT_DEBOUNCE_MS, self._debounced_detect
        )

    def _debounced_detect(self) -> None:
        self._detect_after_id = None
        self._detect_line_segments()

    def _detect_line_segments_now(self) -> None:
        self._cancel_pending_detect()
        self._detect_line_segments()

    def _read_params(self) -> LineSegmentParams:
        blur = int(round(float(self.param_vars["blur_ksize"].get())))
        if blur % 2 == 0:
            blur = max(1, blur - 1)
        return LineSegmentParams(
            blur_ksize=blur,
            canny_low=int(round(float(self.param_vars["canny_low"].get()))),
            canny_high=int(round(float(self.param_vars["canny_high"].get()))),
            rho=float(self.param_vars["rho"].get()),
            theta_deg=float(self.param_vars["theta_deg"].get()),
            threshold=int(round(float(self.param_vars["threshold"].get()))),
            min_line_length=int(round(float(self.param_vars["min_line_length"].get()))),
            max_line_gap=int(round(float(self.param_vars["max_line_gap"].get()))),
            min_width_over_height=float(self.param_vars["min_width_over_height"].get()),
            min_overlap_frac=float(self.param_vars["min_overlap_frac"].get()),
            min_height=float(self.param_vars["min_height"].get()),
            vertical_merge_gap=float(self.param_vars["vertical_merge_gap"].get()),
        )

    def _clear_segment_state(self) -> None:
        self.raw_line_segments = []
        self.merged_line_segments = []
        self.candidate_line_segments = []
        self.vertical_line_segments = []
        self.rectangle_boxes = []
        self.selected_rectangle_idx = None

    def _clear_line_segments(self) -> None:
        self._cancel_pending_detect()
        self._clear_segment_state()
        self._refresh_image()
        self.status_var.set("Cleared line segment overlay")

    def _detect_line_segments(self) -> None:
        params = self._read_params()
        try:
            _save_line_segment_params(params)
        except OSError:
            pass
        if self.current_image is None:
            self.status_var.set("No image selected for line segment detection")
            return
        self.status_var.set("Detecting line segments...")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        try:
            result = detect_horizontal_rectangles(self.current_image, params)
            self.raw_line_segments = result.raw_segments
            self.merged_line_segments = result.merged_segments
            self.candidate_line_segments = result.candidate_segments
            self.vertical_line_segments = result.vertical_merged_segments
            self.rectangle_boxes = result.rectangles
            self.selected_rectangle_idx = None
        except Exception as exc:
            self.status_var.set(f"Line segment detection failed: {type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._refresh_image()
        self.status_var.set(
            f"Found {len(self.raw_line_segments)} raw, "
            f"{len(self.merged_line_segments)} merged, "
            f"{len(self.vertical_line_segments)} vertical, "
            f"{len(self.candidate_line_segments)} candidate, "
            f"{len(self.rectangle_boxes)} rectangle(s) "
            f"in {elapsed_ms:.0f} ms"
        )

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        rendered = _draw_overlays(
            self.current_image,
            [],
            show_boxes=False,
            show_labels=False,
            line_segments=self.raw_line_segments if self.show_raw_lines.get() else None,
            line_segment_color="lime",
            merged_segments=(
                self.merged_line_segments if self.show_merged_lines.get() else None
            ),
            merged_segment_color="orange",
            candidate_segments=(
                self.candidate_line_segments if self.show_candidate_lines.get() else None
            ),
            candidate_segment_color="magenta",
            vertical_segments=(
                self.vertical_line_segments if self.show_vertical_lines.get() else None
            ),
            vertical_segment_color="cyan",
            rectangle_boxes=(
                self.rectangle_boxes if self.show_rectangles.get() else None
            ),
            rectangle_box_color="blue",
            selected_rectangle_idx=self.selected_rectangle_idx,
            selected_rectangle_color="red",
        )
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        img_w, img_h = rendered.size
        fit = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        scale = max(1e-6, fit * self._view_zoom)
        self._render_scale = scale
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        if new_size != (img_w, img_h):
            rendered = rendered.resize(new_size, Image.Resampling.LANCZOS)
        self.current_display = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.current_display, anchor="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _prev_image(self) -> None:
        idx = self._selected_image_index()
        if idx is None:
            return
        nxt = max(0, idx - 1)
        self.image_list.select_clear(0, tk.END)
        self.image_list.select_set(nxt)
        self.image_list.see(nxt)
        self._on_image_select()

    def _next_image(self) -> None:
        idx = self._selected_image_index()
        if idx is None:
            return
        nxt = min(len(self.image_paths) - 1, idx + 1)
        self.image_list.select_clear(0, tk.END)
        self.image_list.select_set(nxt)
        self.image_list.see(nxt)
        self._on_image_select()

    def _apply_zoom_factor(self, factor: float) -> None:
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._view_zoom * factor))
        self._refresh_image()

    def _zoom_in(self) -> None:
        self._apply_zoom_factor(self._ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._apply_zoom_factor(1.0 / self._ZOOM_STEP)

    def _reset_zoom(self) -> None:
        self._view_zoom = 1.0
        self._refresh_image()

    def _on_zoom_in_hotkey(self, _event: object | None = None) -> str:
        self._zoom_in()
        return "break"

    def _on_zoom_out_hotkey(self, _event: object | None = None) -> str:
        self._zoom_out()
        return "break"

    def _on_reset_zoom_hotkey(self, _event: object | None = None) -> str:
        self._reset_zoom()
        return "break"

    def _on_rmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            self._rmb_last_x = None
            return
        self._rmb_last_x = int(event.x)

    def _on_rmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._rmb_last_x is None or self.current_image is None:
            return
        x = int(event.x)
        dx = x - self._rmb_last_x
        self._rmb_last_x = x
        if dx == 0:
            return
        z = self._view_zoom * (self._RMB_ZOOM_PER_PIXEL**dx)
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, z))
        self._refresh_image()

    def _on_rmb_release(self, _event: tk.Event[tk.Canvas]) -> None:
        self._rmb_last_x = None

    def _on_lmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            return
        self._lmb_press_xy = (int(event.x), int(event.y))
        self._lmb_panning = False

    def _on_lmb_motion(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        x0, y0 = self._lmb_press_xy
        x, y = int(event.x), int(event.y)
        if not self._lmb_panning:
            if (x - x0) ** 2 + (y - y0) ** 2 < self._PAN_CLICK_THRESHOLD_SQ:
                return
            self.canvas.scan_mark(x0, y0)
            self._lmb_panning = True
        self.canvas.scan_dragto(x, y, gain=1)

    def _on_lmb_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        try:
            if not self._lmb_panning:
                self._select_rectangle_at_canvas_event(event)
        finally:
            self._lmb_press_xy = None
            self._lmb_panning = False

    def _rectangle_hit_index_at_canvas(self, event: tk.Event[tk.Canvas]) -> int | None:
        if self.current_image is None or not self.rectangle_boxes:
            return None
        if not self.show_rectangles.get():
            return None
        canvas_x = self.canvas.canvasx(int(event.x))
        canvas_y = self.canvas.canvasy(int(event.y))
        img_x = int(canvas_x / max(self._render_scale, 1e-6))
        img_y = int(canvas_y / max(self._render_scale, 1e-6))
        best_idx: int | None = None
        best_area: float | None = None
        for idx, (x0, y0, x1, y1) in enumerate(self.rectangle_boxes):
            if x0 <= img_x <= x1 and y0 <= img_y <= y1:
                area = float(max(1, x1 - x0) * max(1, y1 - y0))
                if best_area is None or area < best_area:
                    best_area = area
                    best_idx = idx
        return best_idx

    def _select_rectangle_at_canvas_event(self, event: tk.Event[tk.Canvas]) -> None:
        selected_idx = self._rectangle_hit_index_at_canvas(event)
        self.selected_rectangle_idx = selected_idx
        self._refresh_image()
        if selected_idx is None:
            return
        x0, y0, x1, y1 = self.rectangle_boxes[selected_idx]
        self.status_var.set(
            f"Selected rectangle {selected_idx + 1}/{len(self.rectangle_boxes)} "
            f"({x0},{y0})-({x1},{y1})"
        )
        self.canvas.focus_set()

    def _on_mmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_mark(int(event.x), int(event.y))

    def _on_mmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_dragto(int(event.x), int(event.y), gain=1)

    def _on_canvas_mousewheel(self, event: tk.Event[tk.Canvas]) -> None:
        if event.state & 0x0004:
            if event.delta > 0:
                self._apply_zoom_factor(self._ZOOM_STEP)
            elif event.delta < 0:
                self._apply_zoom_factor(1.0 / self._ZOOM_STEP)
            return
        if event.state & 0x0001:
            self.canvas.xview_scroll(int(-(event.delta / 120)), "units")
        else:
            self.canvas.yview_scroll(int(-(event.delta / 120)), "units")


class ColorSegmentViewerApp:
    """Browse images and segment them into large color-based regions."""

    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4
    _RMB_ZOOM_PER_PIXEL = 1.0012
    _PARAM_DETECT_DEBOUNCE_MS = 2000

    def __init__(
        self,
        parent: tk.Misc,
        source_root: Path,
        *,
        mode: str = "folder",
        session_list_label: str = "Runs",
        manage_window: bool = True,
        bind_global_hotkeys: bool = True,
    ):
        if mode not in ("folder", "sessions"):
            raise ValueError(f"Unsupported ColorSegmentViewerApp mode: {mode!r}")
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.source_root = source_root
        self.mode = mode
        self._session_list_label = session_list_label
        self._manage_window = manage_window
        self._bind_global_hotkeys = bind_global_hotkeys
        self._hotkeys_active = False
        self.session_dirs: list[Path] = (
            _discover_runs(source_root) if mode == "sessions" else []
        )
        self.image_paths: list[Path] = []
        self.current_display: ImageTk.PhotoImage | None = None
        self.current_image: Image.Image | None = None
        self.segment_result: ColorSegmentResult | None = None
        self.selected_region_id: int | None = None
        self.show_quantized = tk.BooleanVar(value=True)
        self.show_masked_input = tk.BooleanVar(value=False)
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.mask_text_icons = tk.BooleanVar(value=True)
        self.require_yolo_objects = tk.BooleanVar(value=True)
        self.merge_superpixels = tk.BooleanVar(value=True)
        self.merge_similar = tk.BooleanVar(value=False)
        self.split_large_regions = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Ready")
        self.folder_var = tk.StringVar(value=str(source_root))
        self.param_vars: dict[str, tk.DoubleVar] = {}
        self.param_value_vars: dict[str, tk.StringVar] = {}
        self._param_decimals: dict[str, int] = {}
        self._suppress_param_events = True
        self._detect_after_id: str | None = None
        self._mask_preview: Image.Image | None = None
        self._mask_preview_boxes: list[tuple[int, int, int, int]] = []
        saved = _load_color_segment_params()
        self.mask_text_icons.set(saved.mask_text_icons)
        self.require_yolo_objects.set(saved.require_yolo_objects)
        self.merge_superpixels.set(saved.merge_superpixels)
        self.merge_similar.set(saved.merge_similar)
        self.split_large_regions.set(saved.split_large_regions)
        saved_values = {
            "num_colors": float(saved.num_colors),
            "min_area_frac": float(saved.min_area_frac * 100.0),
            "blur_ksize": float(saved.blur_ksize),
            "edge_canny_low": float(saved.edge_canny_low),
            "edge_canny_high": float(saved.edge_canny_high),
            "edge_dilate": float(saved.edge_dilate),
            "slic_compactness": float(saved.slic_compactness),
            "split_max_area_frac": float(saved.split_max_area_frac * 100.0),
            "merge_color_dist": float(saved.merge_color_dist),
        }
        for key, _label, _lo, _hi, default, decimals, _step in _COLOR_PARAM_SLIDERS:
            self.param_vars[key] = tk.DoubleVar(value=saved_values.get(key, float(default)))
            self.param_value_vars[key] = tk.StringVar()
            self._param_decimals[key] = decimals
            self._update_param_value_label(key)

        self._view_zoom = 1.0
        self._rmb_last_x: int | None = None
        self._render_scale = 1.0
        self._lmb_press_xy: tuple[int, int] | None = None
        self._lmb_panning = False

        self._ui_font = _configure_ui_fonts(self.root, UI_FONT_SIZE)
        self._build_ui()
        self._suppress_param_events = False
        if self.mode == "folder":
            self._reload_folder_images()
        else:
            self._populate_session_list()
        if bind_global_hotkeys:
            self.activate_hotkeys()

    def _build_ui(self) -> None:
        if self._manage_window and isinstance(self.parent, tk.Tk):
            self.parent.title("Color Segments")
            self.parent.geometry("1280x840")
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)

        left = ttk.Frame(self.parent, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        row = 0
        if self.mode == "folder":
            ttk.Label(left, text="Folder").grid(row=row, column=0, sticky="w")
            ttk.Label(left, text="Images").grid(row=row, column=1, sticky="w", padx=(8, 0))
            row += 1
            left.rowconfigure(row, weight=1)
            folder_wrap = ttk.Frame(left)
            folder_wrap.grid(row=row, column=0, sticky="nsew")
            folder_wrap.columnconfigure(0, weight=1)
            ttk.Entry(folder_wrap, textvariable=self.folder_var).grid(row=0, column=0, sticky="ew")
            ttk.Button(folder_wrap, text="Browse…", command=self._browse_folder).grid(
                row=1, column=0, sticky="ew", pady=(4, 0)
            )
            image_wrap = ttk.Frame(left)
            image_wrap.grid(row=row, column=1, sticky="nsew", padx=(8, 0))
            image_wrap.columnconfigure(0, weight=1)
            image_wrap.rowconfigure(0, weight=1)
            self.image_list = tk.Listbox(
                image_wrap, exportselection=False, height=10, width=24, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            self.image_scroll = ttk.Scrollbar(
                image_wrap, orient="vertical", command=self.image_list.yview
            )
            self.image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=self.image_scroll.set)
            self.image_list.bind("<<ListboxSelect>>", self._on_image_select)
            row += 1
        else:
            ttk.Label(left, text=self._session_list_label).grid(row=row, column=0, sticky="w")
            ttk.Label(left, text="Images").grid(row=row, column=1, sticky="w", padx=(8, 0))
            row += 1
            left.rowconfigure(row, weight=1)
            session_wrap = ttk.Frame(left)
            session_wrap.grid(row=row, column=0, sticky="nsew")
            session_wrap.columnconfigure(0, weight=1)
            session_wrap.rowconfigure(0, weight=1)
            self.session_list = tk.Listbox(
                session_wrap, exportselection=False, height=10, width=24, font=self._ui_font
            )
            self.session_list.grid(row=0, column=0, sticky="nsew")
            self.session_scroll = ttk.Scrollbar(
                session_wrap, orient="vertical", command=self.session_list.yview
            )
            self.session_scroll.grid(row=0, column=1, sticky="ns")
            self.session_list.configure(yscrollcommand=self.session_scroll.set)
            self.session_list.bind("<<ListboxSelect>>", self._on_session_select)
            image_wrap = ttk.Frame(left)
            image_wrap.grid(row=row, column=1, sticky="nsew", padx=(8, 0))
            image_wrap.columnconfigure(0, weight=1)
            image_wrap.rowconfigure(0, weight=1)
            self.image_list = tk.Listbox(
                image_wrap, exportselection=False, height=10, width=24, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            self.image_scroll = ttk.Scrollbar(
                image_wrap, orient="vertical", command=self.image_list.yview
            )
            self.image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=self.image_scroll.set)
            self.image_list.bind("<<ListboxSelect>>", self._on_image_select)
            row += 1

        params = ttk.LabelFrame(left, text="Color segment parameters", padding=6)
        params.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        params.columnconfigure(1, weight=1)
        self._param_steps: dict[str, float] = {}
        self._param_bounds: dict[str, tuple[float, float]] = {}
        for prow, (key, label, lo, hi, _default, _decimals, step) in enumerate(
            _COLOR_PARAM_SLIDERS
        ):
            self._param_steps[key] = float(step)
            self._param_bounds[key] = (float(lo), float(hi))
            name = ttk.Label(params, text=label)
            name.grid(row=prow, column=0, sticky="w", pady=1)
            scale = ttk.Scale(
                params,
                from_=lo,
                to=hi,
                orient="horizontal",
                variable=self.param_vars[key],
                command=lambda _v, k=key: self._on_param_scale(k),
            )
            scale.grid(row=prow, column=1, sticky="ew", padx=(6, 2), pady=1)
            dec_btn = ttk.Button(
                params,
                text="◀",
                width=2,
                command=lambda k=key: self._nudge_param(k, -1),
            )
            dec_btn.grid(row=prow, column=2, sticky="e", padx=(2, 0), pady=1)
            value_label = ttk.Label(params, textvariable=self.param_value_vars[key], width=5)
            value_label.grid(row=prow, column=3, sticky="e", pady=1)
            inc_btn = ttk.Button(
                params,
                text="▶",
                width=2,
                command=lambda k=key: self._nudge_param(k, 1),
            )
            inc_btn.grid(row=prow, column=4, sticky="e", padx=(0, 0), pady=1)
            tip = _COLOR_PARAM_TOOLTIPS.get(key, "")
            if tip:
                _attach_tooltip(name, tip)
                _attach_tooltip(scale, tip)
                _attach_tooltip(value_label, tip)
                _attach_tooltip(dec_btn, tip)
                _attach_tooltip(inc_btn, tip)
        row += 1

        controls = ttk.Frame(left)
        controls.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for col in range(2):
            controls.columnconfigure(col, weight=1)
        ttk.Checkbutton(
            controls,
            text="Quantized colors",
            variable=self.show_quantized,
            command=self._refresh_image,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Masked input",
            variable=self.show_masked_input,
            command=self._on_masked_input_toggled,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            controls,
            text="Region boxes",
            variable=self.show_boxes,
            command=self._refresh_image,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Region labels",
            variable=self.show_labels,
            command=self._refresh_image,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Mask text/icons/inputs/scrollbars",
            variable=self.mask_text_icons,
            command=self._on_segment_option_changed,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Require text/icon in region",
            variable=self.require_yolo_objects,
            command=self._on_segment_option_changed,
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Merge superpixels",
            variable=self.merge_superpixels,
            command=self._on_segment_option_changed,
        ).grid(row=3, column=1, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Split large regions",
            variable=self.split_large_regions,
            command=self._on_segment_option_changed,
        ).grid(row=4, column=1, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Merge similar",
            variable=self.merge_similar,
            command=self._on_segment_option_changed,
        ).grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Button(controls, text="Segment", command=self._segment_now).grid(
            row=5, column=0, sticky="ew", pady=(6, 0), padx=(0, 3)
        )
        ttk.Button(controls, text="Clear overlay", command=self._clear_segmentation).grid(
            row=5, column=1, sticky="ew", pady=(6, 0), padx=(3, 0)
        )

        canvas_wrap = ttk.Frame(self.parent, padding=8)
        canvas_wrap.grid(row=0, column=1, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, bg="#1e1e1e", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-3>", self._on_rmb_press)
        self.canvas.bind("<B3-Motion>", self._on_rmb_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self.canvas.bind("<ButtonPress-1>", self._on_lmb_press)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<ButtonPress-2>", self._on_mmb_press)
        self.canvas.bind("<B2-Motion>", self._on_mmb_drag)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)

        status = ttk.Label(self.parent, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self.parent.bind("<Configure>", lambda _e: self._refresh_image())

    def activate_hotkeys(self) -> None:
        if self._hotkeys_active:
            return
        self.root.bind("<Control-plus>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-equal>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-minus>", self._on_zoom_out_hotkey)
        self.root.bind("<Control-0>", self._on_reset_zoom_hotkey)
        self._hotkeys_active = True

    def deactivate_hotkeys(self) -> None:
        if not self._hotkeys_active:
            return
        for key in ("<Control-plus>", "<Control-equal>", "<Control-minus>", "<Control-0>"):
            self.root.unbind(key)
        self._hotkeys_active = False

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory(
            initialdir=self.folder_var.get() or str(self.source_root)
        )
        if not chosen:
            return
        self.source_root = Path(chosen)
        self.folder_var.set(str(self.source_root))
        self._reload_folder_images()

    def _populate_session_list(self) -> None:
        self.session_dirs = _discover_runs(self.source_root)
        self.session_list.delete(0, tk.END)
        for session in self.session_dirs:
            self.session_list.insert(tk.END, session.name)
        if self.session_dirs:
            self.session_list.select_set(0)
            self._on_session_select()
        else:
            self.image_paths = []
            self.image_list.delete(0, tk.END)
            self.current_image = None
            self._clear_segment_state()
            self.canvas.delete("all")
            label = self._session_list_label.lower()
            self.status_var.set(f"No {label} found in {self.source_root}")

    def _on_session_select(self, _event: object | None = None) -> None:
        selected = self.session_list.curselection()
        if not selected:
            return
        session = self.session_dirs[selected[0]]
        self.image_paths = _yolo_ocr_paired_images(session)
        self.image_list.delete(0, tk.END)
        for img in self.image_paths:
            self.image_list.insert(tk.END, img.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self._cancel_pending_detect()
            self.current_image = None
            self._clear_segment_state()
            self.canvas.delete("all")
            self.status_var.set(f"No images found for {session.name}")

    def _reload_folder_images(self) -> None:
        self.image_paths = _discover_folder_images(self.source_root)
        self.image_list.delete(0, tk.END)
        for img in self.image_paths:
            self.image_list.insert(tk.END, img.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self.current_image = None
            self._clear_segment_state()
            self.canvas.delete("all")
            self.status_var.set(f"No images in {self.source_root}")

    def _selected_run(self) -> Path | None:
        if self.mode != "sessions":
            return None
        selected = self.session_list.curselection()
        if not selected:
            return None
        return self.session_dirs[selected[0]]

    def _selected_image_index(self) -> int | None:
        selected = self.image_list.curselection()
        if not selected:
            return None
        return selected[0]

    def _current_image_path(self) -> Path | None:
        idx = self._selected_image_index()
        if idx is None or idx >= len(self.image_paths):
            return None
        return self.image_paths[idx]

    def _on_image_select(self, _event: object | None = None) -> None:
        image_path = self._current_image_path()
        if image_path is None:
            return
        self._cancel_pending_detect()
        self.current_image = Image.open(image_path).convert("RGB")
        self._clear_segment_state()
        self._view_zoom = 1.0
        self.status_var.set(image_path.name)
        self._refresh_image()
        self._schedule_detect()

    def _on_segment_option_changed(self) -> None:
        if self._suppress_param_events:
            return
        self._invalidate_mask_preview()
        self._schedule_detect()

    def _invalidate_mask_preview(self) -> None:
        self._mask_preview = None
        self._mask_preview_boxes = []

    def _on_masked_input_toggled(self) -> None:
        self._invalidate_mask_preview()
        self._refresh_image()

    def _get_mask_preview(
        self,
    ) -> tuple[Image.Image | None, list[tuple[int, int, int, int]]]:
        if self.current_image is None:
            return None, []
        rgb = np.asarray(self.current_image.convert("RGB"))
        params = self._read_params()
        image_path = self._current_image_path()
        run_dir = self._selected_run()
        try:
            _work, _masked_count, _mask_boxes, _text_icon_boxes, masked_before_blur, _ = (
                prepare_segmentation_image(
                    rgb,
                    params,
                    image_path=image_path,
                    run_dir=run_dir,
                )
            )
        except Exception:
            return self.current_image.copy(), []
        return (
            Image.fromarray(masked_before_blur.astype(np.uint8), mode="RGB"),
            list(_mask_boxes),
        )

    def _update_param_value_label(self, key: str) -> None:
        value = float(self.param_vars[key].get())
        if key in ("blur_ksize",):
            snapped = int(round(value))
            if key == "blur_ksize" and snapped % 2 == 0:
                snapped = max(1, snapped - 1)
            self.param_value_vars[key].set(str(snapped))
            return
        decimals = self._param_decimals[key]
        if decimals <= 0:
            self.param_value_vars[key].set(str(int(round(value))))
        else:
            self.param_value_vars[key].set(f"{value:.{decimals}f}")

    def _on_param_scale(self, key: str) -> None:
        if key in ("blur_ksize",):
            snapped = int(round(float(self.param_vars[key].get())))
            if key == "blur_ksize" and snapped % 2 == 0:
                snapped = max(1, snapped - 1)
            current = float(self.param_vars[key].get())
            if abs(current - snapped) > 1e-6:
                was_suppressed = self._suppress_param_events
                self._suppress_param_events = True
                try:
                    self.param_vars[key].set(float(snapped))
                finally:
                    self._suppress_param_events = was_suppressed
        self._update_param_value_label(key)
        if self._suppress_param_events:
            return
        self._invalidate_mask_preview()
        self._schedule_detect()

    def _nudge_param(self, key: str, direction: int) -> None:
        lo, hi = self._param_bounds[key]
        step = self._param_steps[key]
        current = float(self.param_vars[key].get())
        if key == "blur_ksize":
            current = int(round(current))
            if current % 2 == 0:
                current = max(1, current - 1)
        decimals = self._param_decimals[key]
        nxt = current + (step * direction)
        nxt = max(lo, min(hi, nxt))
        if decimals <= 0:
            nxt = float(int(round(nxt)))
        else:
            nxt = round(nxt, decimals)
        if key == "blur_ksize":
            snapped = int(round(nxt))
            if snapped % 2 == 0:
                snapped = max(1, snapped + (1 if direction > 0 else -1))
            nxt = float(max(int(lo), min(int(hi), snapped)))
            if int(nxt) % 2 == 0:
                nxt = float(max(1, int(nxt) - 1))
        if abs(nxt - current) < 1e-9:
            return
        self.param_vars[key].set(nxt)
        self._update_param_value_label(key)
        if not self._suppress_param_events:
            self._schedule_detect()

    def _cancel_pending_detect(self) -> None:
        if self._detect_after_id is not None:
            try:
                self.root.after_cancel(self._detect_after_id)
            except tk.TclError:
                pass
            self._detect_after_id = None

    def _schedule_detect(self) -> None:
        self._cancel_pending_detect()
        try:
            _save_color_segment_params(self._read_params())
        except OSError:
            pass
        self.status_var.set("Parameters changed — segmenting in 2s...")
        self._detect_after_id = self.root.after(
            self._PARAM_DETECT_DEBOUNCE_MS, self._debounced_segment
        )

    def _debounced_segment(self) -> None:
        self._detect_after_id = None
        self._run_segmentation()

    def _segment_now(self) -> None:
        self._cancel_pending_detect()
        self._run_segmentation()

    def _read_params(self) -> ColorSegmentParams:
        blur = int(round(float(self.param_vars["blur_ksize"].get())))
        if blur % 2 == 0:
            blur = max(1, blur - 1)
        return ColorSegmentParams(
            num_colors=max(0, int(round(float(self.param_vars["num_colors"].get())))),
            slic_compactness=float(self.param_vars["slic_compactness"].get()),
            min_area_frac=max(
                0.0001, float(self.param_vars["min_area_frac"].get()) / 100.0
            ),
            blur_ksize=blur,
            mask_text_icons=bool(self.mask_text_icons.get()),
            require_yolo_objects=bool(self.require_yolo_objects.get()),
            merge_superpixels=bool(self.merge_superpixels.get()),
            merge_similar=bool(self.merge_similar.get()),
            merge_color_dist=float(self.param_vars["merge_color_dist"].get()),
            split_large_regions=bool(self.split_large_regions.get()),
            split_max_area_frac=max(
                0.01, float(self.param_vars["split_max_area_frac"].get()) / 100.0
            ),
            edge_canny_low=int(round(float(self.param_vars["edge_canny_low"].get()))),
            edge_canny_high=int(round(float(self.param_vars["edge_canny_high"].get()))),
            edge_dilate=max(0, int(round(float(self.param_vars["edge_dilate"].get())))),
        )

    def _clear_segment_state(self) -> None:
        self.segment_result = None
        self.selected_region_id = None
        self._invalidate_mask_preview()

    def _clear_segmentation(self) -> None:
        self._cancel_pending_detect()
        self._clear_segment_state()
        self._refresh_image()
        self.status_var.set("Cleared color segment overlay")

    def _run_segmentation(self) -> None:
        params = self._read_params()
        try:
            _save_color_segment_params(params)
        except OSError:
            pass
        if self.current_image is None:
            self.status_var.set("No image selected for color segmentation")
            return
        self.status_var.set("Segmenting by color...")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        try:
            self.segment_result = segment_image_by_color(
                self.current_image,
                params,
                image_path=self._current_image_path(),
                run_dir=self._selected_run(),
            )
            self.selected_region_id = None
            self._invalidate_mask_preview()
        except Exception as exc:
            self.status_var.set(f"Color segmentation failed: {type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        region_count = len(self.segment_result.regions)
        masked = self.segment_result.masked_box_count
        self._refresh_image()
        mask_note = ""
        if params.mask_text_icons:
            if masked > 0:
                mask_note = f", masked {masked} UI detail box(es)"
            else:
                mask_note = ", masked 0 UI detail boxes (no YOLO detections found)"
        yolo_filter_note = ""
        if params.require_yolo_objects:
            before = self.segment_result.regions_before_yolo_filter
            removed = max(0, before - region_count)
            if removed > 0:
                yolo_filter_note = f", {removed} region(s) without text/icon removed"
        self.status_var.set(
            f"Found {region_count} color region(s) "
            f"({params.num_colors} segments){mask_note}{yolo_filter_note} "
            f"in {elapsed_ms:.0f} ms"
        )

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        show_masked = bool(self.show_masked_input.get())
        masked_override: Image.Image | None = None
        mask_boxes_override: list[tuple[int, int, int, int]] | None = None
        if show_masked:
            masked_override, mask_boxes_override = self._get_mask_preview()
        rendered = _draw_color_segment_overlays(
            self.current_image,
            self.segment_result,
            show_quantized=self.show_quantized.get(),
            show_masked_input=show_masked,
            show_boxes=self.show_boxes.get(),
            show_labels=self.show_labels.get(),
            selected_region_id=self.selected_region_id,
            masked_input_override=masked_override,
            mask_boxes_override=mask_boxes_override,
        )
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        img_w, img_h = rendered.size
        fit = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        scale = max(1e-6, fit * self._view_zoom)
        self._render_scale = scale
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        if new_size != (img_w, img_h):
            rendered = rendered.resize(new_size, Image.Resampling.LANCZOS)
        self.current_display = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.current_display, anchor="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _apply_zoom_factor(self, factor: float) -> None:
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._view_zoom * factor))
        self._refresh_image()

    def _zoom_in(self) -> None:
        self._apply_zoom_factor(self._ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._apply_zoom_factor(1.0 / self._ZOOM_STEP)

    def _reset_zoom(self) -> None:
        self._view_zoom = 1.0
        self._refresh_image()

    def _on_zoom_in_hotkey(self, _event: object | None = None) -> str:
        self._zoom_in()
        return "break"

    def _on_zoom_out_hotkey(self, _event: object | None = None) -> str:
        self._zoom_out()
        return "break"

    def _on_reset_zoom_hotkey(self, _event: object | None = None) -> str:
        self._reset_zoom()
        return "break"

    def _on_rmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            self._rmb_last_x = None
            return
        self._rmb_last_x = int(event.x)

    def _on_rmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._rmb_last_x is None or self.current_image is None:
            return
        x = int(event.x)
        dx = x - self._rmb_last_x
        self._rmb_last_x = x
        if dx == 0:
            return
        z = self._view_zoom * (self._RMB_ZOOM_PER_PIXEL**dx)
        self._view_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, z))
        self._refresh_image()

    def _on_rmb_release(self, _event: tk.Event[tk.Canvas]) -> None:
        self._rmb_last_x = None

    def _on_lmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.current_image is None:
            return
        self._lmb_press_xy = (int(event.x), int(event.y))
        self._lmb_panning = False

    def _on_lmb_motion(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        x0, y0 = self._lmb_press_xy
        x, y = int(event.x), int(event.y)
        if not self._lmb_panning:
            if (x - x0) ** 2 + (y - y0) ** 2 < self._PAN_CLICK_THRESHOLD_SQ:
                return
            self.canvas.scan_mark(x0, y0)
            self._lmb_panning = True
        self.canvas.scan_dragto(x, y, gain=1)

    def _on_lmb_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        try:
            if not self._lmb_panning:
                self._select_region_at_canvas_event(event)
        finally:
            self._lmb_press_xy = None
            self._lmb_panning = False

    def _region_hit_at_canvas(self, event: tk.Event[tk.Canvas]) -> int | None:
        if self.current_image is None or self.segment_result is None:
            return None
        if not self.show_boxes.get():
            return None
        canvas_x = self.canvas.canvasx(int(event.x))
        canvas_y = self.canvas.canvasy(int(event.y))
        img_x = int(canvas_x / max(self._render_scale, 1e-6))
        img_y = int(canvas_y / max(self._render_scale, 1e-6))
        best_id: int | None = None
        best_area: float | None = None
        for region in self.segment_result.regions:
            x0, y0, x1, y1 = region.bbox
            if x0 <= img_x <= x1 and y0 <= img_y <= y1:
                area = float(region.area)
                if best_area is None or area < best_area:
                    best_area = area
                    best_id = region.region_id
        return best_id

    def _select_region_at_canvas_event(self, event: tk.Event[tk.Canvas]) -> None:
        selected_id = self._region_hit_at_canvas(event)
        self.selected_region_id = selected_id
        self._refresh_image()
        if selected_id is None or self.segment_result is None:
            return
        region = next(
            (item for item in self.segment_result.regions if item.region_id == selected_id),
            None,
        )
        if region is None:
            return
        x0, y0, x1, y1 = region.bbox
        r, g, b = region.mean_color
        self.status_var.set(
            f"Region #{selected_id + 1}: ({x0},{y0})-({x1},{y1}), "
            f"area={region.area}px, rgb=({r},{g},{b})"
        )
        self.canvas.focus_set()

    def _on_mmb_press(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_mark(int(event.x), int(event.y))

    def _on_mmb_drag(self, event: tk.Event[tk.Canvas]) -> None:
        self.canvas.scan_dragto(int(event.x), int(event.y), gain=1)

    def _on_canvas_mousewheel(self, event: tk.Event[tk.Canvas]) -> None:
        if event.state & 0x0004:
            if event.delta > 0:
                self._apply_zoom_factor(self._ZOOM_STEP)
            elif event.delta < 0:
                self._apply_zoom_factor(1.0 / self._ZOOM_STEP)
            return
        if event.state & 0x0001:
            self.canvas.xview_scroll(int(-(event.delta / 120)), "units")
        else:
            self.canvas.yview_scroll(int(-(event.delta / 120)), "units")


class SourceTabShell:
    """Nested notebook: YOLO detection + Line segments + Color segments for one image source."""

    @staticmethod
    def _resolve_image_index(paths: list[Path], target: Path) -> int | None:
        try:
            target_resolved = target.resolve()
        except OSError:
            target_resolved = target
        for idx, path in enumerate(paths):
            try:
                if path.resolve() == target_resolved:
                    return idx
            except OSError:
                continue
        for idx, path in enumerate(paths):
            if path.name == target.name:
                return idx
        return None

    @classmethod
    def _sync_viewer_image_from_yolo(cls, yolo_viewer: Any, target_viewer: Any) -> None:
        """Keep line/color tabs on the same session + image as the YOLO tab."""
        if not hasattr(yolo_viewer, "_current_image_path"):
            return
        image_path = yolo_viewer._current_image_path()
        if image_path is None:
            return

        session_path: Path | None = None
        if hasattr(yolo_viewer, "_selected_run"):
            session_path = yolo_viewer._selected_run()

        if getattr(target_viewer, "mode", None) == "sessions":
            if session_path is None or not hasattr(target_viewer, "session_list"):
                return
            session_idx: int | None = None
            for idx, session in enumerate(target_viewer.session_dirs):
                try:
                    if session.resolve() == session_path.resolve():
                        session_idx = idx
                        break
                except OSError:
                    if session == session_path:
                        session_idx = idx
                        break
            if session_idx is None:
                return
            selected_session = target_viewer.session_list.curselection()
            if not selected_session or selected_session[0] != session_idx:
                target_viewer.session_list.select_clear(0, tk.END)
                target_viewer.session_list.select_set(session_idx)
                target_viewer._on_session_select()
            image_idx = cls._resolve_image_index(target_viewer.image_paths, image_path)
            if image_idx is None:
                return
            selected_image = target_viewer.image_list.curselection()
            if selected_image and selected_image[0] == image_idx:
                return
            target_viewer.image_list.select_clear(0, tk.END)
            target_viewer.image_list.select_set(image_idx)
            target_viewer.image_list.see(image_idx)
            target_viewer._on_image_select()
            return

        if not hasattr(target_viewer, "image_paths"):
            return
        image_idx = cls._resolve_image_index(target_viewer.image_paths, image_path)
        if image_idx is None:
            return
        selected_image = target_viewer.image_list.curselection()
        if selected_image and selected_image[0] == image_idx:
            return
        target_viewer.image_list.select_clear(0, tk.END)
        target_viewer.image_list.select_set(image_idx)
        target_viewer.image_list.see(image_idx)
        target_viewer._on_image_select()

    def __init__(
        self,
        parent: tk.Misc,
        *,
        make_yolo: Any,
        make_lines: Any,
        make_color: Any,
    ):
        self.parent = parent
        self._hotkeys_active = False

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        yolo_frame = ttk.Frame(self.notebook)
        lines_frame = ttk.Frame(self.notebook)
        color_frame = ttk.Frame(self.notebook)
        self.notebook.add(yolo_frame, text="YOLO detection")
        self.notebook.add(lines_frame, text="Line segments")
        self.notebook.add(color_frame, text="Color segments")

        self.yolo_viewer = make_yolo(yolo_frame)
        self.lines_viewer = make_lines(lines_frame)
        self.color_viewer = make_color(color_frame)
        self._viewers = (self.yolo_viewer, self.lines_viewer, self.color_viewer)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_subtab_changed)

    def activate_hotkeys(self) -> None:
        self._hotkeys_active = True
        self._on_subtab_changed()

    def deactivate_hotkeys(self) -> None:
        self._hotkeys_active = False
        for viewer in self._viewers:
            viewer.deactivate_hotkeys()

    def _on_subtab_changed(self, _event: object | None = None) -> None:
        if not self._hotkeys_active:
            for viewer in self._viewers:
                viewer.deactivate_hotkeys()
            return
        selected = self.notebook.index(self.notebook.select())
        for idx, viewer in enumerate(self._viewers):
            if idx == selected:
                if idx != 0:
                    self._sync_viewer_image_from_yolo(self.yolo_viewer, viewer)
                viewer.activate_hotkeys()
            else:
                viewer.deactivate_hotkeys()


class CombinedImageViewerApp:
    """Notebook shell: Run / Recordings / Test, each with YOLO + Line + Color segments."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        runs_root: Path | None = None,
        recordings_root: Path | None = None,
        images_dir: Path | None = None,
        initial_tab: str = "runs",
    ):
        self.root = root
        self.root.title("Image Viewer — YOLO & OCR")
        self.root.geometry("1280x840")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.notebook = notebook

        runs_tab = ttk.Frame(notebook)
        recordings_tab = ttk.Frame(notebook)
        test_tab = ttk.Frame(notebook)
        notebook.add(runs_tab, text="Run images")
        notebook.add(recordings_tab, text="Recordings")
        notebook.add(test_tab, text="Test images")

        runs_base = runs_root if runs_root is not None else resolve_runs_dir()
        recordings_base = (
            recordings_root if recordings_root is not None else resolve_recordings_dir()
        )
        test_base = images_dir if images_dir is not None else DEFAULT_TEST_IMAGES_DIR

        self.runs_shell = SourceTabShell(
            runs_tab,
            make_yolo=lambda frame: OcrViewerApp(
                frame,
                runs_base,
                manage_window=False,
                bind_global_hotkeys=False,
            ),
            make_lines=lambda frame: LineSegmentsViewerApp(
                frame,
                runs_base,
                mode="sessions",
                session_list_label="Runs",
                manage_window=False,
                bind_global_hotkeys=False,
            ),
            make_color=lambda frame: ColorSegmentViewerApp(
                frame,
                runs_base,
                mode="sessions",
                session_list_label="Runs",
                manage_window=False,
                bind_global_hotkeys=False,
            ),
        )
        self.recordings_shell = SourceTabShell(
            recordings_tab,
            make_yolo=lambda frame: OcrViewerApp(
                frame,
                recordings_base,
                manage_window=False,
                bind_global_hotkeys=False,
                session_list_label="Recordings",
            ),
            make_lines=lambda frame: LineSegmentsViewerApp(
                frame,
                recordings_base,
                mode="sessions",
                session_list_label="Recordings",
                manage_window=False,
                bind_global_hotkeys=False,
            ),
            make_color=lambda frame: ColorSegmentViewerApp(
                frame,
                recordings_base,
                mode="sessions",
                session_list_label="Recordings",
                manage_window=False,
                bind_global_hotkeys=False,
            ),
        )
        self.test_shell = SourceTabShell(
            test_tab,
            make_yolo=lambda frame: TestImagesViewerApp(
                frame,
                test_base,
                manage_window=False,
                bind_global_hotkeys=False,
            ),
            make_lines=lambda frame: LineSegmentsViewerApp(
                frame,
                test_base,
                mode="folder",
                manage_window=False,
                bind_global_hotkeys=False,
            ),
            make_color=lambda frame: ColorSegmentViewerApp(
                frame,
                test_base,
                mode="folder",
                manage_window=False,
                bind_global_hotkeys=False,
            ),
        )
        self._viewers = (self.runs_shell, self.recordings_shell, self.test_shell)
        self._tab_frames = {
            "runs": runs_tab,
            "recordings": recordings_tab,
            "test": test_tab,
        }
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        if initial_tab in self._tab_frames:
            notebook.select(self._tab_frames[initial_tab])
        self._on_tab_changed()

    def _on_tab_changed(self, _event: object | None = None) -> None:
        selected = self.notebook.index(self.notebook.select())
        for idx, viewer in enumerate(self._viewers):
            if idx == selected:
                viewer.activate_hotkeys()
            else:
                viewer.deactivate_hotkeys()


if __name__ == "__main__":
    run_app()
