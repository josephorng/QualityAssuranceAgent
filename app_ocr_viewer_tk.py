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

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

from cua_mcp.icon_map import is_pua_char, lookup_pua_icon, text_has_pua, unknown_icon_record
from cua_mcp.read_screen_text.get_coordinates import get_coordinates_from_image_path
from cua_mcp.select_mouse_target import _build_candidates_from_bgr
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END, YOLO_CLASS_NAMES
from src.common.io_utils import read_json, write_json
from src.common.settings import ROOT_DIR, resolve_runs_dir

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
    r"C:\Users\Joseph Hung\Documents\Repos\Git\OCR\data\validate\cua_validate"
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


def _is_icon_ocr_line(line: OcrLine) -> bool:
    return line.line_type == "ocr" and text_has_pua(line.text)


def _is_single_pua_icon_text(text: str) -> bool:
    """True when ``text`` is a lone PUA glyph (exactly one icon, no other characters)."""
    pua_chars = [ch for ch in text if is_pua_char(ch)]
    non_pua = "".join(ch for ch in text if not is_pua_char(ch)).strip()
    return len(pua_chars) == 1 and not non_pua


def _export_dest_for_icon_identity(is_icon: bool) -> Path:
    """Train export folder for explicit icon vs text identity."""
    return OCR_EXPORT_ICONS_DIR if is_icon else OCR_EXPORT_DEFAULT_DIR


def _export_dest_for_text(text: str) -> Path:
    """Train export folder: single-icon PUA → ``icons``; pure text or mixed PUA+text → ``cua_data``."""
    return _export_dest_for_icon_identity(_is_single_pua_icon_text(text))


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
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return [], "Could not read image for YOLO"
    try:
        candidates = _build_candidates_from_bgr(bgr, yolo_conf_threshold=yolo_conf_threshold)
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
    "scrollbar": "mediumpurple",
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


def _draw_overlays(
    image: Image.Image,
    lines: list[OcrLine],
    show_boxes: bool,
    show_labels: bool,
    selected_idx: int | None = None,
    overlay_font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = overlay_font if overlay_font is not None else ImageFont.load_default()
    for idx, line in enumerate(lines):
        x, y, w, h = line.box
        x2, y2 = x + w, y + h
        is_selected = selected_idx is not None and idx == selected_idx
        if show_boxes:
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
    ):
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.runs_root = runs_root
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
        self.show_labels = tk.BooleanVar(value=True)
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

        self._build_ui()
        self._populate_runs()
        if bind_global_hotkeys:
            self.activate_hotkeys()

    def _all_display_lines(self) -> list[OcrLine]:
        return self.current_lines

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

        ttk.Label(left, text="Runs").grid(row=0, column=0, sticky="w")
        run_wrap = ttk.Frame(left)
        run_wrap.grid(row=1, column=0, sticky="nsew")
        run_wrap.columnconfigure(0, weight=1)
        run_wrap.rowconfigure(0, weight=1)
        self.run_list = tk.Listbox(run_wrap, exportselection=False, height=8, width=48)
        self.run_list.grid(row=0, column=0, sticky="nsew")
        self.run_scroll = ttk.Scrollbar(run_wrap, orient="vertical", command=self.run_list.yview)
        self.run_scroll.grid(row=0, column=1, sticky="ns")
        self.run_list.configure(yscrollcommand=self.run_scroll.set)
        self.run_list.bind("<<ListboxSelect>>", self._on_run_select)

        ttk.Label(left, text="YOLO Images").grid(row=2, column=0, sticky="w", pady=(8, 0))
        image_wrap = ttk.Frame(left)
        image_wrap.grid(row=3, column=0, sticky="nsew")
        image_wrap.columnconfigure(0, weight=1)
        image_wrap.rowconfigure(0, weight=1)
        self.image_list = tk.Listbox(image_wrap, exportselection=False, height=10, width=48)
        self.image_list.grid(row=0, column=0, sticky="nsew")
        self.image_scroll = ttk.Scrollbar(image_wrap, orient="vertical", command=self.image_list.yview)
        self.image_scroll.grid(row=0, column=1, sticky="ns")
        self.image_list.configure(yscrollcommand=self.image_scroll.set)
        self.image_list.bind("<<ListboxSelect>>", self._on_image_select)

        ttk.Label(left, text="YOLO Detections").grid(row=4, column=0, sticky="w", pady=(8, 0))
        item_wrap = ttk.Frame(left)
        item_wrap.grid(row=5, column=0, sticky="nsew")
        item_wrap.columnconfigure(0, weight=1)
        item_wrap.rowconfigure(0, weight=1)
        self.item_list = tk.Listbox(item_wrap, exportselection=False, height=10, width=48)
        self.item_list.grid(row=0, column=0, sticky="nsew")
        self.item_scroll = ttk.Scrollbar(item_wrap, orient="vertical", command=self.item_list.yview)
        self.item_scroll.grid(row=0, column=1, sticky="ns")
        self.item_list.configure(yscrollcommand=self.item_scroll.set)
        self.item_list.bind("<<ListboxSelect>>", self._on_item_select)
        self.item_list.bind("<Double-Button-1>", self._on_item_double_click)
        self.item_list.bind("<Delete>", self._delete_selected_detection)

        controls = ttk.Frame(left)
        controls.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        for col in range(4):
            controls.columnconfigure(col, weight=1)
        ttk.Checkbutton(controls, text="Boxes", variable=self.show_boxes, command=self._refresh_image).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(controls, text="Labels", variable=self.show_labels, command=self._refresh_image).grid(row=0, column=1, sticky="w")
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
        ttk.Button(controls, text="Prev", command=self._prev_image).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Next", command=self._next_image).grid(row=2, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Zoom +", command=self._zoom_in).grid(row=2, column=2, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Zoom -", command=self._zoom_out).grid(row=2, column=3, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Reload YOLO detections", command=self._run_select_text_current_image).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Copy to undone/images", command=self._copy_current_image_to_undone).grid(
            row=3, column=2, columnspan=2, sticky="ew", pady=(6, 0)
        )
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
            self.status_var.set(f"No runs found at {self.runs_root}")

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
            self.current_lines = []
            self.item_list.delete(0, tk.END)
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
            self.current_lines = []
            status = f"Invalid confidence: {err}"
        else:
            self.current_lines, status = resolve_image_lines(
                image_path,
                yolo_conf_threshold=conf,
                allow_yolo=False,
                yolo_cache=self._yolo_lines_cache,
                run_dir=run,
            )
        self.selected_line_idx = None
        self._populate_item_list()
        self.status_var.set(f"{image_path.name} - {status}")
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
        for idx, line in enumerate(self._all_display_lines()):
            prefix = "OCR" if line.line_type == "ocr" else "OBJ"
            label = _display_label_for_line(line)
            self.item_list.insert(tk.END, f"{prefix} {idx + 1:03d}: {label}")

    def _on_item_select(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            self.selected_line_idx = None
            self._refresh_image()
            return
        self.selected_line_idx = selected[0]
        self._refresh_image()

    def _delete_selected_detection(self, _event: object | None = None) -> str:
        idx = self.selected_line_idx
        if idx is None or idx < 0 or idx >= len(self.current_lines):
            self.status_var.set("Select a detection to delete")
            return "break"

        deleted = self.current_lines.pop(idx)
        self.selected_line_idx = None
        self._populate_item_list()
        if self.current_lines:
            next_idx = min(idx, len(self.current_lines) - 1)
            self.item_list.select_set(next_idx)
            self.item_list.see(next_idx)
            self.selected_line_idx = next_idx
        self._refresh_image()
        self.status_var.set(f"Deleted detection #{idx + 1}: {_display_label_for_line(deleted)}")
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

        is_icon = _is_single_pua_icon_text(line.text)
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
            export_as_icon = _is_single_pua_icon_text(corrected_text)
        return self._export_display_item(
            display_idx,
            dest_dir,
            label_text=corrected_text,
            export_as_icon=export_as_icon,
        )

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        rendered = _draw_overlays(
            self.current_image,
            self._all_display_lines(),
            show_boxes=self.show_boxes.get(),
            show_labels=self.show_labels.get(),
            selected_idx=self.selected_line_idx,
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
        self.current_lines = lines
        self.selected_line_idx = None
        self._populate_item_list()
        self._refresh_image()
        self.status_var.set(f"{status} in {elapsed_ms:.0f} ms")

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

def run_app(
    runs_root: Path | None = None,
    *,
    images_dir: Path | None = None,
    initial_tab: str = "runs",
) -> None:
    root = tk.Tk()
    CombinedImageViewerApp(
        root,
        runs_root=runs_root,
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

        folder_row = ttk.Frame(left)
        folder_row.grid(row=0, column=0, sticky="ew")
        folder_row.columnconfigure(0, weight=1)
        ttk.Entry(folder_row, textvariable=self.folder_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_row, text="Browse…", command=self._browse_folder).grid(
            row=0, column=1, padx=(6, 0)
        )

        ttk.Label(left, text="Images").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.image_list = tk.Listbox(
            left, exportselection=False, height=14, width=40, font=self._ui_font
        )
        self.image_list.grid(row=2, column=0, sticky="nsew")
        self.image_list.bind("<<ListboxSelect>>", self._on_image_select)

        ttk.Label(left, text="OCR / detection items").grid(row=3, column=0, sticky="w", pady=(8, 0))
        item_wrap = ttk.Frame(left)
        item_wrap.grid(row=4, column=0, sticky="nsew")
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
        controls.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        for col in range(4):
            controls.columnconfigure(col, weight=1)
        ttk.Checkbutton(
            controls, text="Boxes", variable=self.show_boxes, command=self._refresh_image
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            controls, text="Labels", variable=self.show_labels, command=self._refresh_image
        ).grid(row=0, column=1, sticky="w")
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
            self.canvas.delete("all")
            self.status_var.set(f"No images in {self.images_dir}")

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
        self.current_lines, status = load_ocr_lines(json_path)
        self.selected_line_idx = None
        self._populate_item_list()
        self.status_var.set(f"{image_path.name} — {status}")
        self._refresh_image()

    def _populate_item_list(self) -> None:
        self.item_list.delete(0, tk.END)
        for idx, line in enumerate(self.current_lines):
            label = _display_label_for_line(line)
            self.item_list.insert(tk.END, f"{idx + 1:03d}: {label}")

    def _on_item_select(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            self.selected_line_idx = None
            self._refresh_image()
            return
        self.selected_line_idx = selected[0]
        self._refresh_image()

    def _delete_selected_detection(self, _event: object | None = None) -> str:
        idx = self.selected_line_idx
        if idx is None or idx < 0 or idx >= len(self.current_lines):
            self.status_var.set("Select a detection to delete")
            return "break"

        deleted = self.current_lines.pop(idx)
        self.selected_line_idx = None
        self._populate_item_list()
        if self.current_lines:
            next_idx = min(idx, len(self.current_lines) - 1)
            self.item_list.select_set(next_idx)
            self.item_list.see(next_idx)
            self.selected_line_idx = next_idx
        self._refresh_image()
        self.status_var.set(f"Deleted detection #{idx + 1}: {_display_label_for_line(deleted)}")
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
            if _is_single_pua_icon_text(corrected):
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
        if _is_single_pua_icon_text(corrected_text):
            return 1
        out_txt = dest_dir / f"{stem}.txt"
        out_txt.write_text(corrected_text, encoding="utf-8")
        return 2

    def _set_lines(self, lines: list[OcrLine], status: str) -> None:
        self.current_lines = lines
        self.selected_line_idx = None
        self._populate_item_list()
        self._refresh_image()
        self.status_var.set(status)

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
            regions = get_coordinates_from_image_path(str(src), yolo_conf_threshold=conf)
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
        rendered = _draw_overlays(
            self.current_image,
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


class CombinedImageViewerApp:
    """Notebook shell with Run images and Test images tabs."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        runs_root: Path | None = None,
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
        test_tab = ttk.Frame(notebook)
        notebook.add(runs_tab, text="Run images")
        notebook.add(test_tab, text="Test images")

        runs_base = runs_root if runs_root is not None else resolve_runs_dir()
        test_base = images_dir if images_dir is not None else DEFAULT_TEST_IMAGES_DIR

        self.runs_viewer = OcrViewerApp(
            runs_tab,
            runs_base,
            manage_window=False,
            bind_global_hotkeys=False,
        )
        self.test_viewer = TestImagesViewerApp(
            test_tab,
            test_base,
            manage_window=False,
            bind_global_hotkeys=False,
        )
        self._viewers = (self.runs_viewer, self.test_viewer)
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        if initial_tab == "test":
            notebook.select(test_tab)
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
