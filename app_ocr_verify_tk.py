"""Tk viewer: YOLO OCR text detections verified by Gemma 4 (vLLM multimodal)."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageTk

from app_ocr_viewer_tk import (
    DEFAULT_TEST_IMAGES_DIR,
    OCR_EXPORT_DEFAULT_DIR,
    OCR_EXPORT_ICONS_DIR,
    OcrLine,
    _configure_ui_fonts,
    _discover_folder_images,
    _discover_run_images,
    _discover_runs,
    _display_label_for_line,
    _icon_labels_for_text,
    _is_pua_icon_identity_text,
    _parse_conf_0_to_1,
    _unknown_icon_label,
    _undo_export_files,
    load_yolo_lines,
    YOLO_UNDONE_IMAGES,
)
from cua_mcp.read_screen_text.ocr_image import _expand_box
from cua_mcp.selection_engine import request_json_with_retry
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END, YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT
from src.common.io_utils import imread_bgr
from src.common.settings import load_settings, resolve_recordings_dir, resolve_runs_dir

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
UI_FONT_SIZE = 12

OCR_READ_SHEET_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "readings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["readings", "summary"],
    "additionalProperties": False,
}

# Rows per verification sheet sent to Gemma (small crops, not the full screenshot).
OCR_VERIFY_SHEET_BATCH_SIZE = 50
OCR_VERIFY_SHEET_ROW_HEIGHT = 40
OCR_VERIFY_SHEET_LABEL_WIDTH = 56
OCR_VERIFY_SHEET_PADDING = 6
OCR_VERIFY_BOX_EXPAND = 3


@dataclass(frozen=True)
class TextVerifyResult:
    index: int
    recognized_text: str
    correct: bool
    expected_text: str = ""
    notes: str = ""


@dataclass(frozen=True)
class OcrVerifyOutcome:
    results: list[TextVerifyResult]
    summary: str = ""
    gemma_elapsed_s: float = 0.0
    sheet_elapsed_s: float = 0.0


def _format_inference_timing(
    *,
    ocr_elapsed_s: float | None = None,
    gemma_elapsed_s: float | None = None,
) -> str:
    parts: list[str] = []
    if ocr_elapsed_s is not None:
        parts.append(f"OCR {ocr_elapsed_s:.1f}s")
    if gemma_elapsed_s is not None:
        parts.append(f"Gemma {gemma_elapsed_s:.1f}s")
    return " | ".join(parts)


def visible_line_indices_for_filter(
    line_count: int,
    verify_results: list[TextVerifyResult] | None,
    *,
    mismatch_only: bool,
) -> list[int]:
    """Listbox rows map to these indices in ``text_lines``."""
    if line_count <= 0:
        return []
    if not mismatch_only or verify_results is None:
        return list(range(line_count))
    return [
        idx
        for idx in range(line_count)
        if idx < len(verify_results) and not verify_results[idx].correct
    ]


def copy_image_to_undone(src: Path, folder_name: str) -> Path:
    """Copy a screenshot to ``YOLO_UNDONE_IMAGES`` (same naming as the OCR viewer)."""
    dest_dir = YOLO_UNDONE_IMAGES
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"cua_{folder_name}_{src.name}"
    shutil.copy2(src, dest)
    return dest


def text_lines_from_yolo(lines: list[OcrLine]) -> list[OcrLine]:
    """Keep only YOLO ``text`` class detections."""
    out: list[OcrLine] = []
    for line in lines:
        if line.class_id == YOLO_CLASS_TEXT:
            out.append(line)
            continue
        name = (line.class_name or "").strip().lower()
        if name == "text":
            out.append(line)
    return out


def is_unknown_element_line(line: OcrLine) -> bool:
    """True for YOLO UI element/icon boxes whose icon label is unknown."""
    if line.class_name in ("scrollbar_original", "input_original"):
        return False
    if line.class_id == YOLO_CLASS_TEXT or (line.class_name or "").strip().lower() == "text":
        return False
    if (line.class_name or "").strip().lower() == "unknown":
        return True
    unknown_label = _unknown_icon_label()
    if line.chinese_ids:
        return unknown_label in line.chinese_ids
    if line.text:
        labels = _icon_labels_for_text(line.text)
        if labels:
            return unknown_label in labels
    if line.class_id == YOLO_CLASS_ELEMENT or (line.class_name or "").strip().lower() == "element":
        return not (line.text or "").strip() and not line.chinese_ids
    return False


def unknown_element_lines_from_yolo(lines: list[OcrLine]) -> list[OcrLine]:
    """Keep YOLO detections that look like unmapped UI icons/elements."""
    return [line for line in lines if is_unknown_element_line(line)]


def _crop_line_image(
    source: Image.Image,
    box: tuple[int, int, int, int],
    *,
    expand: int = OCR_VERIFY_BOX_EXPAND,
) -> Image.Image:
    img_w, img_h = source.size
    x, y, w, h = _expand_box(*box, img_w, img_h, margin=expand)
    crop_l = max(0, x)
    crop_t = max(0, y)
    crop_r = min(img_w, x + w)
    crop_b = min(img_h, y + h)
    if crop_r <= crop_l or crop_b <= crop_t:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))
    return source.crop((crop_l, crop_t, crop_r, crop_b))


def _scale_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height <= 0 or image.width <= 0:
        return Image.new("RGB", (1, max(1, height)), color=(255, 255, 255))
    scale = height / image.height
    new_w = max(1, int(image.width * scale))
    return image.resize((new_w, height), Image.Resampling.LANCZOS)


def build_verify_sheet_image(
    source: Image.Image,
    lines: list[OcrLine],
    *,
    start_index: int = 0,
    row_height: int = OCR_VERIFY_SHEET_ROW_HEIGHT,
    label_width: int = OCR_VERIFY_SHEET_LABEL_WIDTH,
    padding: int = OCR_VERIFY_SHEET_PADDING,
) -> Image.Image:
    """Stack numbered YOLO text crops top-to-bottom for multimodal OCR reading."""
    if not lines:
        return Image.new("RGB", (label_width + 32, row_height + padding * 2), color=(255, 255, 255))

    rows: list[tuple[str, Image.Image]] = []
    max_crop_width = 1
    for local_idx, line in enumerate(lines):
        idx = start_index + local_idx
        crop = _scale_to_height(_crop_line_image(source, line.box), row_height)
        max_crop_width = max(max_crop_width, crop.width)
        rows.append((f"{idx}.", crop))

    sheet_w = label_width + max_crop_width + padding * 2
    sheet_h = padding + len(rows) * (row_height + padding)
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    y = padding
    for label, crop in rows:
        draw.text((padding, y + max(0, (row_height - 12) // 2)), label, fill="black")
        sheet.paste(crop, (label_width, y))
        y += row_height + padding
    return sheet


def build_sheet_read_prompt(*, start_index: int, count: int) -> str:
    end_index = start_index + count - 1
    return (
        "The attached image is an OCR verification sheet.\n"
        "Each row shows an index number on the left and a cropped text region from a UI screenshot.\n"
        f"Read the visible text in every crop labeled {start_index} through {end_index}.\n"
        "Return JSON only: "
        '{"readings":[{"index":0,"text":"visible text"}],"summary":"brief note"}.\n'
        "Include one readings entry per index shown in the image.\n"
    )


def parse_sheet_read_response(content: str | None) -> tuple[dict[int, str], str]:
    if not content or not str(content).strip():
        raise ValueError("empty sheet read response")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("sheet read response root must be an object")
    raw_readings = data.get("readings")
    if not isinstance(raw_readings, list):
        raise ValueError("sheet read response missing readings array")
    summary = str(data.get("summary") or "").strip()
    readings: dict[int, str] = {}
    for item in raw_readings:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        readings[index] = str(item.get("text") or "")
    return readings, summary


def compare_readings_to_ocr(
    lines: list[OcrLine],
    readings: dict[int, str],
    *,
    start_index: int = 0,
) -> list[TextVerifyResult]:
    results: list[TextVerifyResult] = []
    for local_idx, line in enumerate(lines):
        global_idx = start_index + local_idx
        recognized = (line.text or "").strip()
        expected = readings.get(global_idx, "").strip()
        if global_idx not in readings:
            results.append(
                TextVerifyResult(
                    index=global_idx,
                    recognized_text=recognized,
                    correct=False,
                    expected_text="",
                    notes="missing from Gemma readings",
                )
            )
            continue
        correct = recognized == expected
        results.append(
            TextVerifyResult(
                index=global_idx,
                recognized_text=recognized,
                correct=correct,
                expected_text="" if correct else expected,
                notes="" if correct else "Gemma read different text",
            )
        )
    return results


def _verify_sheet_output_dir() -> Path:
    from src.common.run_state import get_run_state_manager

    try:
        paths = get_run_state_manager().require_paths()
        out_dir = paths.root / "verify_sheets"
    except RuntimeError:
        out_dir = resolve_runs_dir() / "ocr_verify_tool" / "verify_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


async def _read_verify_sheet_with_gemma(
    sheet_path: Path,
    lines: list[OcrLine],
    *,
    start_index: int = 0,
    log_info: Callable[[str], None] | None = None,
) -> OcrVerifyOutcome:
    prompt = build_sheet_read_prompt(start_index=start_index, count=len(lines))
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": [str(sheet_path)],
        }
    ]

    def _parse(content: str | None) -> OcrVerifyOutcome:
        readings, summary = parse_sheet_read_response(content)
        results = compare_readings_to_ocr(lines, readings, start_index=start_index)
        return OcrVerifyOutcome(results=results, summary=summary)

    return await request_json_with_retry(
        messages=messages,
        response_schema=OCR_READ_SHEET_RESPONSE_SCHEMA,
        parse_reply=_parse,
        retry_instruction=(
            'Return strict JSON only: {"readings":[{"index":0,"text":"..."}],'
            '"summary":"..."}.'
        ),
        log_info=log_info,
        think=False,
        append_image_sizes=False,
    )


async def verify_text_ocr_with_gemma(
    source_image: Image.Image,
    lines: list[OcrLine],
    *,
    log_info: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> OcrVerifyOutcome:
    if not lines:
        return OcrVerifyOutcome(results=[], summary="No text detections to verify")

    batch_size = OCR_VERIFY_SHEET_BATCH_SIZE
    merged_results: list[TextVerifyResult | None] = [None] * len(lines)
    summaries: list[str] = []
    total_batches = (len(lines) + batch_size - 1) // batch_size
    sheet_dir = _verify_sheet_output_dir()
    stamp = int(time.time() * 1000)
    gemma_elapsed_s = 0.0
    sheet_elapsed_s = 0.0

    for batch_num, start in enumerate(range(0, len(lines), batch_size), start=1):
        if on_progress is not None:
            on_progress(batch_num, total_batches)
        batch_lines = lines[start : start + batch_size]
        sheet_t0 = time.perf_counter()
        sheet = build_verify_sheet_image(source_image, batch_lines, start_index=start)
        sheet_path = sheet_dir / f"verify_sheet_{stamp}_{batch_num:02d}.png"
        sheet.save(sheet_path)
        sheet_elapsed_s += time.perf_counter() - sheet_t0
        gemma_t0 = time.perf_counter()
        outcome = await _read_verify_sheet_with_gemma(
            sheet_path,
            batch_lines,
            start_index=start,
            log_info=log_info,
        )
        gemma_elapsed_s += time.perf_counter() - gemma_t0
        if outcome.summary:
            summaries.append(outcome.summary)
        for item in outcome.results:
            if 0 <= item.index < len(merged_results):
                merged_results[item.index] = item

    final_results: list[TextVerifyResult] = []
    for idx, line in enumerate(lines):
        item = merged_results[idx]
        if item is not None:
            final_results.append(item)
        else:
            final_results.append(
                TextVerifyResult(
                    index=idx,
                    recognized_text=line.text or "",
                    correct=False,
                    expected_text="",
                    notes="missing from Gemma readings",
                )
            )
    summary = " | ".join(s for s in summaries if s) or (
        f"Read {len(lines)} crops from {total_batches} sheet(s)"
    )
    return OcrVerifyOutcome(
        results=final_results,
        summary=summary,
        gemma_elapsed_s=gemma_elapsed_s,
        sheet_elapsed_s=sheet_elapsed_s,
    )


def _draw_yolo_boxes(
    image: Image.Image,
    lines: list[OcrLine],
    *,
    selected_idx: int | None = None,
    default_outline: str = "lime",
    selected_outline: str = "red",
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for idx, line in enumerate(lines):
        x, y, w, h = line.box
        is_selected = selected_idx is not None and idx == selected_idx
        outline = selected_outline if is_selected else default_outline
        width = 3 if is_selected else 2
        draw.rectangle([(x, y), (x + w, y + h)], outline=outline, width=width)
    return out


def export_element_line_to_dir(
    image: Image.Image,
    line: OcrLine,
    dest_dir: Path,
    *,
    base_name: str,
    item_index: int,
) -> list[Path]:
    """Export one UI element crop as ``.png`` only (no label), matching the OCR viewer."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    img_w, img_h = image.size
    x, y, w, h = line.box
    if w <= 0 or h <= 0:
        raise ValueError("invalid item box dimensions")
    crop_l = max(0, x)
    crop_t = max(0, y)
    crop_r = min(img_w, x + w)
    crop_b = min(img_h, y + h)
    if crop_r <= crop_l or crop_b <= crop_t:
        raise ValueError("box is outside image bounds")

    crop = image.crop((crop_l, crop_t, crop_r, crop_b))
    stem = f"{base_name}_obj_item{item_index + 1:03d}"
    out_img = dest_dir / f"{stem}.png"
    crop.save(out_img)
    return [out_img]


def gemma_label_for_verified_line(
    line: OcrLine,
    result: TextVerifyResult | None,
) -> str:
    """Text label from Gemma verification (empty when not verified)."""
    if result is None:
        return ""
    if result.correct:
        return (result.recognized_text or line.text or "").strip()
    return (result.expected_text or "").strip()


def export_label_for_verified_line(
    line: OcrLine,
    result: TextVerifyResult | None,
    *,
    override: str | None = None,
) -> str:
    """Label to write for training export: Gemma 4 reading (optionally user-edited)."""
    if override is not None:
        return override.strip()
    return gemma_label_for_verified_line(line, result)


def export_text_line_to_dir(
    image: Image.Image,
    line: OcrLine,
    label_text: str,
    dest_dir: Path,
    *,
    base_name: str,
    item_index: int,
) -> list[Path]:
    """Export one text crop to ``cua_data`` (``.png`` + ``.txt``), matching the OCR viewer."""
    if not _is_pua_icon_identity_text(label_text) and not label_text.strip():
        raise ValueError("label text is empty")
    dest_dir.mkdir(parents=True, exist_ok=True)
    img_w, img_h = image.size
    x, y, w, h = line.box
    if w <= 0 or h <= 0:
        raise ValueError("invalid item box dimensions")
    crop_l = max(0, x)
    crop_t = max(0, y)
    crop_r = min(img_w, x + w)
    crop_b = min(img_h, y + h)
    if crop_r <= crop_l or crop_b <= crop_t:
        raise ValueError("box is outside image bounds")

    crop = image.crop((crop_l, crop_t, crop_r, crop_b))
    stem = f"{base_name}_item{item_index + 1:03d}"
    out_img = dest_dir / f"{stem}.png"
    crop.save(out_img)
    written = [out_img]
    if _is_pua_icon_identity_text(label_text):
        return written
    out_txt = dest_dir / f"{stem}.txt"
    out_txt.write_text(label_text, encoding="utf-8")
    written.append(out_txt)
    return written


class _CanvasZoomMixin:
    """Shared zoom/pan canvas behavior for OCR verify panels."""

    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4
    _RMB_ZOOM_PER_PIXEL = 1.0012

    current_image: Image.Image | None
    canvas: tk.Canvas
    current_display: ImageTk.PhotoImage | None
    _view_zoom: float
    _render_scale: float
    _rmb_last_x: int | None
    _lmb_press_xy: tuple[int, int] | None
    _lmb_panning: bool
    _hotkeys_active: bool
    root: tk.Misc

    def _render_overlay(self, base: Image.Image) -> Image.Image:
        raise NotImplementedError

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

    def _on_lmb_release(self, _event: tk.Event[tk.Canvas]) -> None:
        self._lmb_press_xy = None
        self._lmb_panning = False

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

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        rendered = self._render_overlay(self.current_image)
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


class OcrVerifyPanel(_CanvasZoomMixin):
    """Browse images from runs/recordings sessions or a flat folder; detect and verify text OCR."""

    def _render_overlay(self, base: Image.Image) -> Image.Image:
        return _draw_yolo_boxes(
            base,
            self.text_lines,
            selected_idx=self.selected_line_idx,
        )

    def __init__(
        self,
        parent: tk.Misc,
        source_root: Path,
        *,
        mode: str = "sessions",
        session_list_label: str = "Runs",
        manage_window: bool = False,
    ) -> None:
        if mode not in ("sessions", "folder"):
            raise ValueError(f"Unsupported OcrVerifyPanel mode: {mode!r}")
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.source_root = source_root
        self.mode = mode
        self._session_list_label = session_list_label
        self._manage_window = manage_window

        self.session_dirs: list[Path] = (
            _discover_runs(source_root) if mode == "sessions" else []
        )
        self.image_paths: list[Path] = []
        self.current_image: Image.Image | None = None
        self.text_lines: list[OcrLine] = []
        self.verify_results: list[TextVerifyResult] | None = None
        self.verify_summary: str = ""
        self.last_ocr_elapsed_s: float | None = None
        self.last_gemma_elapsed_s: float | None = None
        self._visible_line_indices: list[int] = []
        self.selected_line_idx: int | None = None
        self.current_display: ImageTk.PhotoImage | None = None
        self._busy = False
        self._hotkeys_active = False
        self._view_zoom = 1.0
        self._rmb_last_x: int | None = None
        self._render_scale = 1.0
        self._lmb_press_xy: tuple[int, int] | None = None
        self._lmb_panning = False

        self.status_var = tk.StringVar(value="Ready")
        self.folder_var = tk.StringVar(value=str(source_root))
        self.yolo_conf_var = tk.StringVar(value=f"{DEFAULT_CONF_YOLOV26_END2END:g}")
        self.summary_var = tk.StringVar(value="")
        self.filter_mismatch_var = tk.BooleanVar(value=True)
        self.detections_label_var = tk.StringVar(value="Text detections")

        self._ui_font = _configure_ui_fonts(self.root, UI_FONT_SIZE)
        self._build_ui()
        if self.mode == "folder":
            self._reload_folder_images()
        else:
            self._populate_sessions()

    def _build_ui(self) -> None:
        if self._manage_window and isinstance(self.parent, tk.Tk):
            self.parent.title("OCR Text Verifier")
            self.parent.geometry("1280x840")

        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)

        left = ttk.Frame(self.parent, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        left.rowconfigure(1, weight=1)

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
                image_wrap, exportselection=False, height=12, width=28, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            image_scroll = ttk.Scrollbar(image_wrap, orient="vertical", command=self.image_list.yview)
            image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=image_scroll.set)
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
                session_wrap, exportselection=False, height=12, width=28, font=self._ui_font
            )
            self.session_list.grid(row=0, column=0, sticky="nsew")
            session_scroll = ttk.Scrollbar(
                session_wrap, orient="vertical", command=self.session_list.yview
            )
            session_scroll.grid(row=0, column=1, sticky="ns")
            self.session_list.configure(yscrollcommand=session_scroll.set)
            self.session_list.bind("<<ListboxSelect>>", self._on_session_select)

            image_wrap = ttk.Frame(left)
            image_wrap.grid(row=row, column=1, sticky="nsew", padx=(8, 0))
            image_wrap.columnconfigure(0, weight=1)
            image_wrap.rowconfigure(0, weight=1)
            self.image_list = tk.Listbox(
                image_wrap, exportselection=False, height=12, width=28, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            image_scroll = ttk.Scrollbar(image_wrap, orient="vertical", command=self.image_list.yview)
            image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=image_scroll.set)
            self.image_list.bind("<<ListboxSelect>>", self._on_image_select)
            row += 1

        ttk.Label(left, textvariable=self.detections_label_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        row += 1
        filter_row = ttk.Frame(left)
        filter_row.grid(row=row, column=0, columnspan=2, sticky="ew")
        ttk.Checkbutton(
            filter_row,
            text="Show OCR ≠ Gemma only",
            variable=self.filter_mismatch_var,
            command=self._on_filter_toggle,
        ).grid(row=0, column=0, sticky="w")
        row += 1
        left.rowconfigure(row, weight=1)
        result_wrap = ttk.Frame(left)
        result_wrap.grid(row=row, column=0, columnspan=2, sticky="nsew")
        result_wrap.columnconfigure(0, weight=1)
        result_wrap.rowconfigure(0, weight=1)
        self.result_list = tk.Listbox(
            result_wrap,
            exportselection=False,
            selectmode=tk.EXTENDED,
            height=14,
            width=72,
            font=self._ui_font,
        )
        self.result_list.grid(row=0, column=0, sticky="nsew")
        result_scroll = ttk.Scrollbar(result_wrap, orient="vertical", command=self.result_list.yview)
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.result_list.configure(yscrollcommand=result_scroll.set)
        self.result_list.bind("<<ListboxSelect>>", self._on_result_select)
        self.result_list.bind("<Double-Button-1>", self._on_result_double_click)
        row += 1

        controls = ttk.Frame(left)
        controls.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        ttk.Label(controls, text="YOLO confidence").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.yolo_conf_var, width=10).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(4, 0)
        )
        ttk.Button(controls, text="Detect text (YOLO OCR)", command=self._run_detect).grid(
            row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 2)
        )
        ttk.Button(controls, text="Verify (Gemma 4)", command=self._run_verify).grid(
            row=1, column=1, sticky="ew", pady=(6, 0), padx=2
        )
        ttk.Button(controls, text="Detect + Verify", command=self._run_detect_and_verify).grid(
            row=1, column=2, sticky="ew", pady=(6, 0), padx=(2, 0)
        )
        ttk.Button(controls, text="Select non-matching", command=self._select_non_matching).grid(
            row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 2)
        )
        ttk.Button(controls, text="Select all", command=self._select_all_results).grid(
            row=2, column=1, sticky="ew", pady=(6, 0), padx=2
        )
        ttk.Button(controls, text="Export to cua_data", command=self._export_selected_to_cua_data).grid(
            row=2, column=2, sticky="ew", pady=(6, 0), padx=(2, 0)
        )
        ttk.Button(controls, text="Copy to undone/images", command=self._copy_current_image_to_undone).grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        row += 1

        ttk.Label(left, textvariable=self.summary_var, wraplength=360).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        canvas_wrap = ttk.Frame(self.parent, padding=8)
        canvas_wrap.grid(row=0, column=1, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, bg="#1e1e1e", highlightthickness=0)
        v_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-3>", self._on_rmb_press)
        self.canvas.bind("<B3-Motion>", self._on_rmb_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self.canvas.bind("<ButtonPress-1>", self._on_lmb_press)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<ButtonPress-2>", self._on_mmb_press)
        self.canvas.bind("<B2-Motion>", self._on_mmb_drag)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Configure>", lambda _e: self._refresh_image())
        self.parent.bind("<Configure>", lambda _e: self._refresh_image())

        status = ttk.Label(self.parent, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        if self._manage_window:
            self.activate_hotkeys()

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
        chosen = filedialog.askdirectory(initialdir=str(self.source_root))
        if not chosen:
            return
        self.source_root = Path(chosen)
        self.folder_var.set(chosen)
        self._reload_folder_images()

    def _populate_sessions(self) -> None:
        self.session_dirs = _discover_runs(self.source_root)
        self.session_list.delete(0, tk.END)
        for session in self.session_dirs:
            self.session_list.insert(tk.END, session.name)
        self.image_paths = []
        self.image_list.delete(0, tk.END)
        if self.session_dirs:
            self.session_list.select_set(0)
            self._on_session_select()

    def _reload_folder_images(self) -> None:
        self.image_paths = _discover_folder_images(self.source_root)
        self.image_list.delete(0, tk.END)
        for path in self.image_paths:
            self.image_list.insert(tk.END, path.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self.status_var.set(f"No images in {self.source_root}")

    def _on_session_select(self, _event: object | None = None) -> None:
        selected = self.session_list.curselection()
        if not selected:
            return
        session = self.session_dirs[selected[0]]
        self.image_paths = _discover_run_images(session)
        self.image_list.delete(0, tk.END)
        for path in self.image_paths:
            self.image_list.insert(tk.END, path.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self._clear_detection_state()
            self.status_var.set(f"No images in {session.name}")

    def _on_image_select(self, _event: object | None = None) -> None:
        path = self._current_image_path()
        if path is None:
            self._clear_detection_state()
            return
        bgr = imread_bgr(path)
        if bgr is None:
            self._clear_detection_state()
            self.status_var.set(f"Could not read image: {path.name}")
            return
        self.current_image = Image.fromarray(bgr[:, :, ::-1])
        self._view_zoom = 1.0
        self._clear_detection_state(keep_image=True)
        self.status_var.set(f"Loaded {path.name}")
        self._refresh_image()

    def _clear_detection_state(self, *, keep_image: bool = False) -> None:
        if not keep_image:
            self.current_image = None
        self.text_lines = []
        self.verify_results = None
        self.verify_summary = ""
        self.last_ocr_elapsed_s = None
        self.last_gemma_elapsed_s = None
        self._visible_line_indices = []
        self.selected_line_idx = None
        self.summary_var.set("")
        self.result_list.select_clear(0, tk.END)
        self._populate_result_list()
        if not keep_image:
            self.canvas.delete("all")

    def _current_image_path(self) -> Path | None:
        selected = self.image_list.curselection()
        if not selected:
            return None
        idx = selected[0]
        if idx < 0 or idx >= len(self.image_paths):
            return None
        return self.image_paths[idx]

    def _undone_folder_name(self) -> str:
        if self.mode == "sessions":
            selected = self.session_list.curselection()
            if selected and 0 <= selected[0] < len(self.session_dirs):
                return self.session_dirs[selected[0]].name
        src = self._current_image_path()
        return src.parent.name if src is not None else "unknown"

    def _copy_current_image_to_undone(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected to copy")
            return
        try:
            dest = copy_image_to_undone(src, self._undone_folder_name())
        except OSError as exc:
            self.status_var.set(f"Copy failed: {exc}")
            return
        self.status_var.set(f"Copied to {dest}")

    def _parse_conf(self) -> tuple[float | None, str | None]:
        return _parse_conf_0_to_1(self.yolo_conf_var.get())

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        if message:
            self.status_var.set(message)

    def _run_in_thread(
        self,
        work: Callable[[], Any],
        on_done: Callable[[Any, BaseException | None], None],
        *,
        status: str = "Working...",
    ) -> None:
        if self._busy:
            return

        def runner() -> None:
            try:
                result = work()
                self.root.after(0, lambda r=result: on_done(r, None))
            except BaseException as exc:
                self.root.after(0, lambda e=exc: on_done(None, e))

        self._set_busy(True, status)
        threading.Thread(target=runner, daemon=True).start()

    def _run_detect(self) -> None:
        path = self._current_image_path()
        if path is None:
            self.status_var.set("Select an image first")
            return
        conf, err = self._parse_conf()
        if conf is None:
            self.status_var.set(f"Invalid confidence: {err}")
            return

        def work() -> tuple[list[OcrLine], str, float]:
            t0 = time.perf_counter()
            lines, status = load_yolo_lines(path, yolo_conf_threshold=conf)
            text_lines = text_lines_from_yolo(lines)
            elapsed = time.perf_counter() - t0
            return text_lines, status, elapsed

        def on_done(result: Any, error: BaseException | None) -> None:
            self._set_busy(False)
            if error is not None:
                self.status_var.set(f"YOLO OCR failed: {type(error).__name__}: {error}")
                return
            text_lines, status, elapsed = result
            self.text_lines = text_lines
            self.verify_results = None
            self.verify_summary = ""
            self.last_ocr_elapsed_s = elapsed
            self.last_gemma_elapsed_s = None
            self.selected_line_idx = None
            self.summary_var.set("")
            self.result_list.select_clear(0, tk.END)
            self._populate_result_list()
            self._refresh_image()
            timing = _format_inference_timing(ocr_elapsed_s=elapsed)
            self.status_var.set(
                f"{status} | {len(text_lines)} text boxes | {timing} | "
                f"model={load_settings().brain_lm}"
            )

        self._run_in_thread(work, on_done, status=f"Running YOLO OCR (conf={conf:g})...")

    def _run_verify(self) -> None:
        path = self._current_image_path()
        if path is None:
            self.status_var.set("Select an image first")
            return
        if not self.text_lines:
            self.status_var.set("Run YOLO OCR detect first")
            return
        if self.current_image is None:
            self.status_var.set("No image loaded")
            return

        lines = list(self.text_lines)
        image = self.current_image
        total_batches = max(1, (len(lines) + OCR_VERIFY_SHEET_BATCH_SIZE - 1) // OCR_VERIFY_SHEET_BATCH_SIZE)

        def _on_progress(batch_num: int, batch_total: int) -> None:
            self.root.after(
                0,
                lambda b=batch_num, t=batch_total: self.status_var.set(
                    f"Gemma reading sheet {b}/{t} ({len(lines)} crops)…"
                ),
            )

        def work() -> OcrVerifyOutcome:
            return asyncio.run(
                verify_text_ocr_with_gemma(image, lines, on_progress=_on_progress)
            )

        def on_done(result: Any, error: BaseException | None) -> None:
            self._set_busy(False)
            if error is not None:
                self.status_var.set(f"Gemma verify failed: {type(error).__name__}: {error}")
                return
            outcome: OcrVerifyOutcome = result
            self.verify_results = outcome.results
            self.verify_summary = outcome.summary
            self.last_gemma_elapsed_s = outcome.gemma_elapsed_s
            correct = sum(1 for item in outcome.results if item.correct)
            total = len(outcome.results)
            timing = _format_inference_timing(
                ocr_elapsed_s=self.last_ocr_elapsed_s,
                gemma_elapsed_s=outcome.gemma_elapsed_s,
            )
            self.summary_var.set(
                outcome.summary
                or f"Verified {correct}/{total} correct"
            )
            self._populate_result_list()
            self._refresh_image()
            self.status_var.set(
                f"Verified {correct}/{total} text boxes | {timing} | "
                f"model={load_settings().brain_lm}"
            )

        self._run_in_thread(
            work,
            on_done,
            status=(
                f"Building crop sheet(s) and reading {len(lines)} texts with "
                f"{load_settings().brain_lm} ({total_batches} sheet(s))…"
            ),
        )

    def _run_detect_and_verify(self) -> None:
        path = self._current_image_path()
        if path is None:
            self.status_var.set("Select an image first")
            return
        conf, err = self._parse_conf()
        if conf is None:
            self.status_var.set(f"Invalid confidence: {err}")
            return

        def _on_progress(batch_num: int, batch_total: int) -> None:
            self.root.after(
                0,
                lambda b=batch_num, t=batch_total: self.status_var.set(
                    f"Gemma verify batch {b}/{t}…"
                ),
            )

        def work() -> tuple[list[OcrLine], OcrVerifyOutcome, float, float]:
            ocr_t0 = time.perf_counter()
            lines, _status = load_yolo_lines(path, yolo_conf_threshold=conf)
            text_lines = text_lines_from_yolo(lines)
            ocr_elapsed = time.perf_counter() - ocr_t0
            bgr = imread_bgr(path)
            if bgr is None:
                raise RuntimeError(f"could not read image: {path}")
            image = Image.fromarray(bgr[:, :, ::-1])
            outcome = asyncio.run(
                verify_text_ocr_with_gemma(image, text_lines, on_progress=_on_progress)
            )
            return text_lines, outcome, ocr_elapsed, outcome.gemma_elapsed_s

        def on_done(result: Any, error: BaseException | None) -> None:
            self._set_busy(False)
            if error is not None:
                self.status_var.set(f"Detect+verify failed: {type(error).__name__}: {error}")
                return
            text_lines, outcome, ocr_elapsed, gemma_elapsed = result
            self.text_lines = text_lines
            self.verify_results = outcome.results
            self.verify_summary = outcome.summary
            self.last_ocr_elapsed_s = ocr_elapsed
            self.last_gemma_elapsed_s = gemma_elapsed
            correct = sum(1 for item in outcome.results if item.correct)
            total = len(outcome.results)
            timing = _format_inference_timing(
                ocr_elapsed_s=ocr_elapsed,
                gemma_elapsed_s=gemma_elapsed,
            )
            self.summary_var.set(
                outcome.summary
                or f"Verified {correct}/{total} correct"
            )
            self._populate_result_list()
            self._refresh_image()
            self.status_var.set(
                f"Detected {total} text boxes, {correct} correct | {timing} | "
                f"model={load_settings().brain_lm}"
            )

        self._run_in_thread(work, on_done, status="Running YOLO OCR + Gemma crop-sheet verify...")

    def _format_result_row(self, idx: int) -> str:
        line = self.text_lines[idx]
        text = line.text.strip() or "<empty>"
        x, y, w, h = line.box
        prefix = "?"
        suffix = ""
        if self.verify_results is not None and idx < len(self.verify_results):
            item = self.verify_results[idx]
            prefix = "OK" if item.correct else "BAD"
            if not item.correct and item.expected_text.strip():
                suffix = f' -> "{item.expected_text.strip()}"'
            if item.notes:
                suffix += f" ({item.notes})"
        return f"[{prefix}] #{idx + 1} ({x},{y}) {w}x{h}: {text}{suffix}"

    def _on_filter_toggle(self) -> None:
        self.selected_line_idx = None
        self._populate_result_list()
        self._refresh_image()

    def _line_idx_for_list_row(self, row: int) -> int | None:
        if row < 0 or row >= len(self._visible_line_indices):
            return None
        return self._visible_line_indices[row]

    def _update_detections_label(self) -> None:
        total = len(self.text_lines)
        visible = len(self._visible_line_indices)
        if (
            self.filter_mismatch_var.get()
            and self.verify_results is not None
            and total
        ):
            self.detections_label_var.set(
                f"Text detections ({visible}/{total} OCR ≠ Gemma)"
            )
            return
        if total:
            self.detections_label_var.set(f"Text detections ({total})")
            return
        self.detections_label_var.set("Text detections")

    def _populate_result_list(self) -> None:
        self._visible_line_indices = visible_line_indices_for_filter(
            len(self.text_lines),
            self.verify_results,
            mismatch_only=self.filter_mismatch_var.get(),
        )
        self.result_list.delete(0, tk.END)
        for idx in self._visible_line_indices:
            self.result_list.insert(tk.END, self._format_result_row(idx))
        self._update_detections_label()

    def _on_result_select(self, _event: object | None = None) -> None:
        selected = self.result_list.curselection()
        if not selected:
            self.selected_line_idx = None
        else:
            self.selected_line_idx = self._line_idx_for_list_row(int(selected[-1]))
        self._refresh_image()

    def _result_index_at_event(self, event: tk.Event[tk.Misc]) -> int | None:
        try:
            row = int(self.result_list.nearest(event.y))
        except (tk.TclError, TypeError, ValueError):
            return None
        return self._line_idx_for_list_row(row)

    def _sync_result_list_selection(self, line_idx: int) -> None:
        self.selected_line_idx = line_idx
        if line_idx in self._visible_line_indices:
            row = self._visible_line_indices.index(line_idx)
            self.result_list.select_clear(0, tk.END)
            self.result_list.select_set(row)
            self.result_list.see(row)
        self._refresh_image()

    def _open_detection_detail_dialog(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.text_lines):
            return
        navigation_indices = list(self._visible_line_indices)
        if not navigation_indices:
            navigation_indices = list(range(len(self.text_lines)))
        if idx not in navigation_indices:
            return

        dialog = tk.Toplevel(self.root)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)

        state = {"idx": idx}
        preview_label = ttk.Label(dialog)
        preview_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 8))

        details_frame = ttk.Frame(dialog)
        details_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        details_frame.columnconfigure(1, weight=1)

        ttk.Label(dialog, text="OCR result").grid(row=2, column=0, sticky="nw", padx=10, pady=(8, 4))
        ocr_widget = tk.Text(dialog, height=1, width=64, wrap="word", font=self._ui_font)
        ocr_widget.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(8, 4))

        ttk.Label(dialog, text="Gemma 4 result").grid(row=3, column=0, sticky="w", padx=10, pady=4)
        gemma_var = tk.StringVar()
        gemma_entry = ttk.Entry(dialog, textvariable=gemma_var, width=72)
        gemma_entry.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=4)

        button_bar = ttk.Frame(dialog)
        button_bar.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 10))
        for col in range(7):
            button_bar.columnconfigure(col, weight=1 if col == 3 else 0)

        position_var = tk.StringVar()
        ttk.Label(button_bar, textvariable=position_var).grid(row=0, column=3, sticky="e", padx=4)
        prev_btn = ttk.Button(button_bar, text="Previous")
        prev_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        next_btn = ttk.Button(button_bar, text="Next")
        next_btn.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        last_export_paths: list[Path] = []
        undo_btn = ttk.Button(button_bar, text="Undo Export", state="disabled")

        def _clear_undo_state() -> None:
            nonlocal last_export_paths
            last_export_paths = []
            undo_btn.configure(state="disabled")

        def _after_export(paths: list[Path], line_idx: int) -> None:
            nonlocal last_export_paths
            last_export_paths = list(paths)
            undo_btn.configure(state="normal")
            self.status_var.set(
                f"Exported item #{line_idx + 1} ({len(paths)} file(s)) to {OCR_EXPORT_DEFAULT_DIR}"
            )

        def _undo_last_export() -> None:
            nonlocal last_export_paths
            if not last_export_paths:
                return
            line_idx = state["idx"]
            try:
                removed = _undo_export_files(last_export_paths)
            except OSError as exc:
                self.status_var.set(f"Undo export failed: {type(exc).__name__}: {exc}")
                return
            last_export_paths = []
            undo_btn.configure(state="disabled")
            self.status_var.set(
                f"Undid export ({removed} file(s) removed) for item #{line_idx + 1}"
            )

        def _set_ocr_text(text: str) -> None:
            display_ocr = text if text else "<empty>"
            ocr_widget.configure(state="normal")
            ocr_widget.delete("1.0", tk.END)
            ocr_widget.insert("1.0", display_ocr)
            ocr_widget.configure(state="disabled")
            ocr_lines = ocr_widget.count("1.0", "end", "displaylines")[0]
            ocr_widget.configure(height=max(1, min(6, ocr_lines)))

        def _render_details(line_idx: int) -> None:
            line = self.text_lines[line_idx]
            result = (
                self.verify_results[line_idx]
                if self.verify_results is not None and line_idx < len(self.verify_results)
                else None
            )
            x, y, w, h = line.box
            ocr_text = line.text or ""
            if result is None:
                status = "Not verified"
                notes = ""
            else:
                status = "Correct" if result.correct else "Incorrect"
                notes = result.notes
            gemma_text = gemma_label_for_verified_line(line, result)

            dialog.title(f"Text Detection #{line_idx + 1}")
            if self.current_image is not None:
                crop = _scale_to_height(
                    _crop_line_image(self.current_image, line.box),
                    min(120, max(40, h + 2 * OCR_VERIFY_BOX_EXPAND)),
                )
                preview = ImageTk.PhotoImage(crop)
                preview_label.configure(image=preview)
                preview_label.image = preview
            else:
                preview_label.configure(image="")
                preview_label.image = None

            for child in details_frame.winfo_children():
                child.destroy()
            readonly_fields: list[tuple[str, str]] = [
                ("Index", f"{line_idx} (#{line_idx + 1})"),
                ("BBox (x, y, w, h)", f"({x}, {y}, {w}, {h})"),
                ("YOLO class", line.class_name or "text"),
                ("Verify status", status),
            ]
            if notes:
                readonly_fields.append(("Notes", notes))
            for row, (label, value) in enumerate(readonly_fields):
                ttk.Label(details_frame, text=label).grid(
                    row=row, column=0, sticky="w", padx=10, pady=4
                )
                ttk.Label(details_frame, text=value, wraplength=420).grid(
                    row=row, column=1, sticky="w", padx=(0, 10), pady=4
                )

            _set_ocr_text(ocr_text)
            gemma_var.set(gemma_text)

            pos = navigation_indices.index(line_idx)
            position_var.set(f"{pos + 1} / {len(navigation_indices)}")
            prev_btn.configure(state="normal" if pos > 0 else "disabled")
            next_btn.configure(state="normal" if pos < len(navigation_indices) - 1 else "disabled")
            _clear_undo_state()
            self._sync_result_list_selection(line_idx)

        def _export_current() -> None:
            line_idx = state["idx"]
            label_text = gemma_var.get().strip()
            try:
                paths = self._export_line_to_cua_data(line_idx, label_text)
            except ValueError as exc:
                self.status_var.set(f"Export failed: {exc}")
                return
            except OSError as exc:
                self.status_var.set(f"Export failed: {type(exc).__name__}: {exc}")
                return
            _after_export(paths, line_idx)

        def _go_relative(delta: int) -> None:
            line_idx = state["idx"]
            pos = navigation_indices.index(line_idx) + delta
            if pos < 0 or pos >= len(navigation_indices):
                return
            state["idx"] = navigation_indices[pos]
            _render_details(state["idx"])
            gemma_entry.focus_set()
            gemma_entry.selection_range(0, tk.END)

        prev_btn.configure(command=lambda: _go_relative(-1))
        next_btn.configure(command=lambda: _go_relative(1))
        undo_btn.configure(command=_undo_last_export)
        ttk.Button(button_bar, text="Export to cua_data", command=_export_current).grid(
            row=0, column=4, sticky="ew", padx=(0, 4)
        )
        undo_btn.grid(row=0, column=5, sticky="ew", padx=(0, 4))
        ttk.Button(button_bar, text="Close", command=dialog.destroy).grid(
            row=0, column=6, sticky="ew"
        )

        def _on_escape(_event: object | None = None) -> None:
            dialog.destroy()

        def _on_left(_event: object | None = None) -> str:
            _go_relative(-1)
            return "break"

        def _on_right(_event: object | None = None) -> str:
            _go_relative(1)
            return "break"

        dialog.bind("<Escape>", _on_escape)
        dialog.bind("<Left>", _on_left)
        dialog.bind("<Right>", _on_right)
        gemma_entry.bind("<Control-Left>", _on_left)
        gemma_entry.bind("<Control-Right>", _on_right)

        _render_details(idx)
        gemma_entry.focus_set()
        gemma_entry.selection_range(0, tk.END)

    def _on_result_double_click(self, event: tk.Event[tk.Misc]) -> None:
        idx = self._result_index_at_event(event)
        if idx is None:
            return
        self._open_detection_detail_dialog(idx)

    def _select_non_matching(self) -> None:
        if not self.text_lines:
            self.status_var.set("No detections to select")
            return
        if self.verify_results is None:
            self.status_var.set("Run verify first to select non-matching items")
            return
        self.result_list.select_clear(0, tk.END)
        count = 0
        for row, line_idx in enumerate(self._visible_line_indices):
            if line_idx < len(self.verify_results) and not self.verify_results[line_idx].correct:
                self.result_list.select_set(row)
                count += 1
        if count:
            self.result_list.see(next(row for row, line_idx in enumerate(self._visible_line_indices)
                                    if line_idx < len(self.verify_results) and not self.verify_results[line_idx].correct))
        self.status_var.set(f"Selected {count} non-matching item(s)")

    def _select_all_results(self) -> None:
        if not self._visible_line_indices:
            self.status_var.set("No detections to select")
            return
        self.result_list.select_set(0, tk.END)
        self.status_var.set(f"Selected all {len(self._visible_line_indices)} item(s)")

    def _export_line_to_cua_data(self, idx: int, label_text: str) -> list[Path]:
        if self.current_image is None:
            raise ValueError("no image loaded")
        if idx < 0 or idx >= len(self.text_lines):
            raise ValueError("invalid item index")
        line = self.text_lines[idx]
        src = self._current_image_path()
        base_name = src.stem if src is not None else "image"
        return export_text_line_to_dir(
            self.current_image,
            line,
            label_text,
            OCR_EXPORT_DEFAULT_DIR,
            base_name=base_name,
            item_index=idx,
        )

    def _export_selected_to_cua_data(self) -> None:
        if self.current_image is None:
            self.status_var.set("No image loaded")
            return
        selected = self.result_list.curselection()
        if not selected:
            self.status_var.set("Select one or more items to export")
            return
        dest_dir = OCR_EXPORT_DEFAULT_DIR
        exported = 0
        files_written = 0
        skipped = 0
        errors: list[str] = []
        for row in selected:
            idx = self._line_idx_for_list_row(int(row))
            if idx is None:
                continue
            line = self.text_lines[idx]
            result = (
                self.verify_results[idx]
                if self.verify_results is not None and idx < len(self.verify_results)
                else None
            )
            label_text = export_label_for_verified_line(line, result)
            if not label_text.strip() and not _is_pua_icon_identity_text(label_text):
                skipped += 1
                errors.append(f"#{idx + 1}: no Gemma result (run verify first)")
                continue
            try:
                paths = self._export_line_to_cua_data(idx, label_text)
            except ValueError as exc:
                skipped += 1
                errors.append(f"#{idx + 1}: {exc}")
                continue
            except OSError as exc:
                errors.append(f"#{idx + 1}: {type(exc).__name__}: {exc}")
                continue
            exported += 1
            files_written += len(paths)
        if exported:
            status = f"Exported {exported} item(s) ({files_written} file(s)) to {dest_dir}"
            if skipped:
                status += f"; skipped {skipped}"
            self.status_var.set(status)
            return
        if errors:
            self.status_var.set(f"Export failed: {errors[0]}")
            return
        self.status_var.set("Nothing exported")


class UnknownElementsPanel(_CanvasZoomMixin):
    """Browse images and export unmapped YOLO element/icon crops (PNG only)."""

    def _render_overlay(self, base: Image.Image) -> Image.Image:
        return _draw_yolo_boxes(
            base,
            self.element_lines,
            selected_idx=self.selected_line_idx,
            default_outline="lime",
            selected_outline="red",
        )

    def __init__(
        self,
        parent: tk.Misc,
        source_root: Path,
        *,
        mode: str = "sessions",
        session_list_label: str = "Runs",
    ) -> None:
        if mode not in ("sessions", "folder"):
            raise ValueError(f"Unsupported UnknownElementsPanel mode: {mode!r}")
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.source_root = source_root
        self.mode = mode
        self._session_list_label = session_list_label

        self.session_dirs: list[Path] = (
            _discover_runs(source_root) if mode == "sessions" else []
        )
        self.image_paths: list[Path] = []
        self.current_image: Image.Image | None = None
        self.element_lines: list[OcrLine] = []
        self.selected_line_idx: int | None = None
        self.current_display: ImageTk.PhotoImage | None = None
        self._busy = False
        self._hotkeys_active = False
        self._view_zoom = 1.0
        self._rmb_last_x: int | None = None
        self._render_scale = 1.0
        self._lmb_press_xy: tuple[int, int] | None = None
        self._lmb_panning = False

        self.status_var = tk.StringVar(value="Ready")
        self.folder_var = tk.StringVar(value=str(source_root))
        self.yolo_conf_var = tk.StringVar(value=f"{DEFAULT_CONF_YOLOV26_END2END:g}")
        self.summary_var = tk.StringVar(value="")
        self.elements_label_var = tk.StringVar(value="Unknown elements")

        self._ui_font = _configure_ui_fonts(self.root, UI_FONT_SIZE)
        self._build_ui()
        if self.mode == "folder":
            self._reload_folder_images()
        else:
            self._populate_sessions()

    def _build_ui(self) -> None:
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)

        left = ttk.Frame(self.parent, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        left.rowconfigure(1, weight=1)

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
                image_wrap, exportselection=False, height=12, width=28, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            image_scroll = ttk.Scrollbar(image_wrap, orient="vertical", command=self.image_list.yview)
            image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=image_scroll.set)
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
                session_wrap, exportselection=False, height=12, width=28, font=self._ui_font
            )
            self.session_list.grid(row=0, column=0, sticky="nsew")
            session_scroll = ttk.Scrollbar(
                session_wrap, orient="vertical", command=self.session_list.yview
            )
            session_scroll.grid(row=0, column=1, sticky="ns")
            self.session_list.configure(yscrollcommand=session_scroll.set)
            self.session_list.bind("<<ListboxSelect>>", self._on_session_select)

            image_wrap = ttk.Frame(left)
            image_wrap.grid(row=row, column=1, sticky="nsew", padx=(8, 0))
            image_wrap.columnconfigure(0, weight=1)
            image_wrap.rowconfigure(0, weight=1)
            self.image_list = tk.Listbox(
                image_wrap, exportselection=False, height=12, width=28, font=self._ui_font
            )
            self.image_list.grid(row=0, column=0, sticky="nsew")
            image_scroll = ttk.Scrollbar(image_wrap, orient="vertical", command=self.image_list.yview)
            image_scroll.grid(row=0, column=1, sticky="ns")
            self.image_list.configure(yscrollcommand=image_scroll.set)
            self.image_list.bind("<<ListboxSelect>>", self._on_image_select)
            row += 1

        ttk.Label(left, textvariable=self.elements_label_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        row += 1
        left.rowconfigure(row, weight=1)
        result_wrap = ttk.Frame(left)
        result_wrap.grid(row=row, column=0, columnspan=2, sticky="nsew")
        result_wrap.columnconfigure(0, weight=1)
        result_wrap.rowconfigure(0, weight=1)
        self.result_list = tk.Listbox(
            result_wrap,
            exportselection=False,
            selectmode=tk.EXTENDED,
            height=14,
            width=72,
            font=self._ui_font,
        )
        self.result_list.grid(row=0, column=0, sticky="nsew")
        result_scroll = ttk.Scrollbar(result_wrap, orient="vertical", command=self.result_list.yview)
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.result_list.configure(yscrollcommand=result_scroll.set)
        self.result_list.bind("<<ListboxSelect>>", self._on_result_select)
        row += 1

        controls = ttk.Frame(left)
        controls.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        ttk.Label(controls, text="YOLO confidence").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.yolo_conf_var, width=10).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(4, 0)
        )
        ttk.Button(controls, text="Detect unknown elements", command=self._run_detect).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Select all", command=self._select_all_results).grid(
            row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 2)
        )
        ttk.Button(controls, text="Export to elements", command=self._export_selected).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(6, 0), padx=(2, 0)
        )
        ttk.Button(controls, text="Copy to undone/images", command=self._copy_current_image_to_undone).grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        row += 1

        ttk.Label(left, textvariable=self.summary_var, wraplength=360).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        canvas_wrap = ttk.Frame(self.parent, padding=8)
        canvas_wrap.grid(row=0, column=1, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, bg="#1e1e1e", highlightthickness=0)
        v_scroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-3>", self._on_rmb_press)
        self.canvas.bind("<B3-Motion>", self._on_rmb_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self.canvas.bind("<ButtonPress-1>", self._on_lmb_press)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<ButtonPress-2>", self._on_mmb_press)
        self.canvas.bind("<B2-Motion>", self._on_mmb_drag)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Configure>", lambda _e: self._refresh_image())
        self.parent.bind("<Configure>", lambda _e: self._refresh_image())

        status = ttk.Label(self.parent, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

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
        chosen = filedialog.askdirectory(initialdir=str(self.source_root))
        if not chosen:
            return
        self.source_root = Path(chosen)
        self.folder_var.set(chosen)
        self._reload_folder_images()

    def _populate_sessions(self) -> None:
        self.session_dirs = _discover_runs(self.source_root)
        self.session_list.delete(0, tk.END)
        for session in self.session_dirs:
            self.session_list.insert(tk.END, session.name)
        self.image_paths = []
        self.image_list.delete(0, tk.END)
        if self.session_dirs:
            self.session_list.select_set(0)
            self._on_session_select()

    def _reload_folder_images(self) -> None:
        self.image_paths = _discover_folder_images(self.source_root)
        self.image_list.delete(0, tk.END)
        for path in self.image_paths:
            self.image_list.insert(tk.END, path.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self.status_var.set(f"No images in {self.source_root}")

    def _on_session_select(self, _event: object | None = None) -> None:
        selected = self.session_list.curselection()
        if not selected:
            return
        session = self.session_dirs[selected[0]]
        self.image_paths = _discover_run_images(session)
        self.image_list.delete(0, tk.END)
        for path in self.image_paths:
            self.image_list.insert(tk.END, path.name)
        if self.image_paths:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self._clear_detection_state()
            self.status_var.set(f"No images in {session.name}")

    def _on_image_select(self, _event: object | None = None) -> None:
        path = self._current_image_path()
        if path is None:
            self._clear_detection_state()
            return
        bgr = imread_bgr(path)
        if bgr is None:
            self._clear_detection_state()
            self.status_var.set(f"Could not read {path.name}")
            return
        self.current_image = Image.fromarray(bgr[:, :, ::-1])
        self._view_zoom = 1.0
        self._clear_detection_state(keep_image=True)
        self.status_var.set(f"Loaded {path.name}")
        self._refresh_image()

    def _clear_detection_state(self, *, keep_image: bool = False) -> None:
        if not keep_image:
            self.current_image = None
        self.element_lines = []
        self.selected_line_idx = None
        self.summary_var.set("")
        self.result_list.select_clear(0, tk.END)
        self._populate_result_list()
        if not keep_image:
            self.canvas.delete("all")

    def _current_image_path(self) -> Path | None:
        selected = self.image_list.curselection()
        if not selected:
            return None
        idx = selected[0]
        if idx < 0 or idx >= len(self.image_paths):
            return None
        return self.image_paths[idx]

    def _undone_folder_name(self) -> str:
        if self.mode == "sessions":
            selected = self.session_list.curselection()
            if selected and 0 <= selected[0] < len(self.session_dirs):
                return self.session_dirs[selected[0]].name
        src = self._current_image_path()
        return src.parent.name if src is not None else "unknown"

    def _copy_current_image_to_undone(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected to copy")
            return
        try:
            dest = copy_image_to_undone(src, self._undone_folder_name())
        except OSError as exc:
            self.status_var.set(f"Copy failed: {exc}")
            return
        self.status_var.set(f"Copied to {dest}")

    def _parse_conf(self) -> tuple[float | None, str | None]:
        return _parse_conf_0_to_1(self.yolo_conf_var.get())

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        if message:
            self.status_var.set(message)

    def _run_in_thread(
        self,
        work: Callable[[], Any],
        on_done: Callable[[Any, BaseException | None], None],
        *,
        status: str = "Working...",
    ) -> None:
        if self._busy:
            return

        def runner() -> None:
            try:
                result = work()
                self.root.after(0, lambda r=result: on_done(r, None))
            except BaseException as exc:
                self.root.after(0, lambda e=exc: on_done(None, e))

        self._set_busy(True, status)
        threading.Thread(target=runner, daemon=True).start()

    def _run_detect(self) -> None:
        path = self._current_image_path()
        if path is None:
            self.status_var.set("Select an image first")
            return
        conf, err = self._parse_conf()
        if conf is None:
            self.status_var.set(f"Invalid confidence: {err}")
            return

        def work() -> tuple[list[OcrLine], str, float]:
            t0 = time.perf_counter()
            lines, status = load_yolo_lines(path, yolo_conf_threshold=conf)
            element_lines = unknown_element_lines_from_yolo(lines)
            elapsed = time.perf_counter() - t0
            return element_lines, status, elapsed

        def on_done(result: Any, error: BaseException | None) -> None:
            self._set_busy(False)
            if error is not None:
                self.status_var.set(f"YOLO detect failed: {type(error).__name__}: {error}")
                return
            element_lines, status, elapsed = result
            self.element_lines = element_lines
            self.selected_line_idx = None
            self.summary_var.set("")
            self.result_list.select_clear(0, tk.END)
            self._populate_result_list()
            self._refresh_image()
            self.status_var.set(
                f"{status} | {len(element_lines)} unknown element(s) | OCR {elapsed:.1f}s"
            )

        self._run_in_thread(work, on_done, status=f"Running YOLO detect (conf={conf:g})...")

    def _format_result_row(self, idx: int) -> str:
        line = self.element_lines[idx]
        label = _display_label_for_line(line)
        x, y, w, h = line.box
        return f"#{idx + 1} ({x},{y}) {w}x{h}: {label}"

    def _populate_result_list(self) -> None:
        self.result_list.delete(0, tk.END)
        for idx in range(len(self.element_lines)):
            self.result_list.insert(tk.END, self._format_result_row(idx))
        count = len(self.element_lines)
        if count:
            self.elements_label_var.set(f"Unknown elements ({count})")
        else:
            self.elements_label_var.set("Unknown elements")

    def _on_result_select(self, _event: object | None = None) -> None:
        selected = self.result_list.curselection()
        if not selected:
            self.selected_line_idx = None
        else:
            self.selected_line_idx = int(selected[-1])
        self._refresh_image()

    def _select_all_results(self) -> None:
        if not self.element_lines:
            self.status_var.set("No unknown elements to select")
            return
        self.result_list.select_set(0, tk.END)
        self.status_var.set(f"Selected all {len(self.element_lines)} item(s)")

    def _export_line(self, idx: int) -> list[Path]:
        if self.current_image is None:
            raise ValueError("no image loaded")
        if idx < 0 or idx >= len(self.element_lines):
            raise ValueError("invalid item index")
        line = self.element_lines[idx]
        src = self._current_image_path()
        base_name = src.stem if src is not None else "image"
        return export_element_line_to_dir(
            self.current_image,
            line,
            OCR_EXPORT_ICONS_DIR,
            base_name=base_name,
            item_index=idx,
        )

    def _export_selected(self) -> None:
        if self.current_image is None:
            self.status_var.set("No image loaded")
            return
        selected = self.result_list.curselection()
        if not selected:
            self.status_var.set("Select one or more items to export")
            return
        dest_dir = OCR_EXPORT_ICONS_DIR
        exported = 0
        files_written = 0
        errors: list[str] = []
        for row in selected:
            idx = int(row)
            if idx < 0 or idx >= len(self.element_lines):
                continue
            try:
                paths = self._export_line(idx)
            except (ValueError, OSError) as exc:
                errors.append(f"#{idx + 1}: {exc}")
                continue
            exported += 1
            files_written += len(paths)
        if exported:
            self.status_var.set(
                f"Exported {exported} item(s) ({files_written} PNG) to {dest_dir}"
            )
            return
        if errors:
            self.status_var.set(f"Export failed: {errors[0]}")
            return
        self.status_var.set("Nothing exported")


class OcrVerifyApp:
    """Notebook shell with the same image sources as the OCR viewer."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        runs_root: Path | None = None,
        recordings_root: Path | None = None,
        images_dir: Path | None = None,
        initial_tab: str = "runs",
    ) -> None:
        self.root = root
        self.root.title("OCR Text Verifier — YOLO + Gemma 4")
        self.root.geometry("1280x840")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        runs_tab = ttk.Frame(self.notebook)
        recordings_tab = ttk.Frame(self.notebook)
        test_tab = ttk.Frame(self.notebook)
        unknown_tab = ttk.Frame(self.notebook)
        self.notebook.add(runs_tab, text="Run images")
        self.notebook.add(recordings_tab, text="Recordings")
        self.notebook.add(test_tab, text="Test images")
        self.notebook.add(unknown_tab, text="Unknown elements")

        runs_base = runs_root if runs_root is not None else resolve_runs_dir()
        recordings_base = (
            recordings_root if recordings_root is not None else resolve_recordings_dir()
        )
        test_base = images_dir if images_dir is not None else DEFAULT_TEST_IMAGES_DIR

        self.runs_panel = OcrVerifyPanel(
            runs_tab,
            runs_base,
            mode="sessions",
            session_list_label="Runs",
        )
        self.recordings_panel = OcrVerifyPanel(
            recordings_tab,
            recordings_base,
            mode="sessions",
            session_list_label="Recordings",
        )
        self.test_panel = OcrVerifyPanel(
            test_tab,
            test_base,
            mode="folder",
        )

        unknown_tab.rowconfigure(0, weight=1)
        unknown_tab.columnconfigure(0, weight=1)
        self.unknown_notebook = ttk.Notebook(unknown_tab)
        self.unknown_notebook.grid(row=0, column=0, sticky="nsew")
        unknown_runs_tab = ttk.Frame(self.unknown_notebook)
        unknown_recordings_tab = ttk.Frame(self.unknown_notebook)
        unknown_test_tab = ttk.Frame(self.unknown_notebook)
        self.unknown_notebook.add(unknown_runs_tab, text="Run images")
        self.unknown_notebook.add(unknown_recordings_tab, text="Recordings")
        self.unknown_notebook.add(unknown_test_tab, text="Test images")

        self.unknown_runs_panel = UnknownElementsPanel(
            unknown_runs_tab,
            runs_base,
            mode="sessions",
            session_list_label="Runs",
        )
        self.unknown_recordings_panel = UnknownElementsPanel(
            unknown_recordings_tab,
            recordings_base,
            mode="sessions",
            session_list_label="Recordings",
        )
        self.unknown_test_panel = UnknownElementsPanel(
            unknown_test_tab,
            test_base,
            mode="folder",
        )

        self._tab_frames = {
            "runs": runs_tab,
            "recordings": recordings_tab,
            "test": test_tab,
            "unknown": unknown_tab,
        }
        self._text_panels = (self.runs_panel, self.recordings_panel, self.test_panel)
        self._unknown_panels = (
            self.unknown_runs_panel,
            self.unknown_recordings_panel,
            self.unknown_test_panel,
        )
        self._all_panels = self._text_panels + self._unknown_panels
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.unknown_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        if initial_tab in self._tab_frames:
            self.notebook.select(self._tab_frames[initial_tab])
        self._on_tab_changed()

    def _active_panel(self) -> _CanvasZoomMixin:
        top_idx = self.notebook.index(self.notebook.select())
        if top_idx < len(self._text_panels):
            return self._text_panels[top_idx]
        inner_idx = self.unknown_notebook.index(self.unknown_notebook.select())
        return self._unknown_panels[inner_idx]

    def _on_tab_changed(self, _event: object | None = None) -> None:
        active = self._active_panel()
        for panel in self._all_panels:
            if panel is active:
                panel.activate_hotkeys()
            else:
                panel.deactivate_hotkeys()


def _ensure_run_state_for_tooling() -> None:
    """Initialize run state so vLLM/OCR helpers can log outside the main agent."""
    from src.common.run_state import get_run_state_manager
    from src.common.runtime_context import set_runtime_env

    manager = get_run_state_manager()
    if manager.paths is not None:
        return
    paths = manager.init_run("ocr_verify_tool", "ocr_verify_tool")
    set_runtime_env(paths.root, paths.root.name)


def run_app(
    runs_root: Path | None = None,
    *,
    recordings_root: Path | None = None,
    images_dir: Path | None = None,
    initial_tab: str = "runs",
) -> None:
    _ensure_run_state_for_tooling()
    root = tk.Tk()
    OcrVerifyApp(
        root,
        runs_root=runs_root,
        recordings_root=recordings_root,
        images_dir=images_dir,
        initial_tab=initial_tab,
    )
    root.mainloop()


if __name__ == "__main__":
    run_app()
