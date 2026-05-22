"""Tkinter viewer for YOLO + OCR on ``test_images/`` using ``cua_mcp.read_screen_text.ocr_image``."""

from __future__ import annotations

import os
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageTk

from app_ocr_viewer_tk import (
    BOX_EDIT_STEP,
    BOX_EDIT_STEP_SHIFT,
    DEFAULT_ULTRALYTICS_PT_PATH,
    OCR_EXPORT_DEFAULT_DIR,
    OcrLine,
    _adjust_box_edge,
    _parse_conf_0_to_1,
    _run_ultralytics_yolo_pt,
    _run_yolo_onnx_best_detections,
    load_ocr_lines,
)
from cua_mcp.read_screen_text.ocr_image import (
    format_coordinate_text_from_regions,
    get_coordinates_from_path,
    get_text_boxes_from_path,
)
from cua_mcp.yolo_onnx import DEFAULT_CONF_YOLOV26_END2END
from src.common.io_utils import write_json
from src.common.settings import ROOT_DIR

DEFAULT_TEST_IMAGES_DIR = ROOT_DIR / "test_images"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Tk widgets and canvas overlay labels (default Tk ~9pt on Windows).
UI_FONT_SIZE = 12
OVERLAY_FONT_SIZE = 15

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


def _draw_overlays(
    image: Image.Image,
    lines: list[OcrLine],
    *,
    show_boxes: bool,
    show_labels: bool,
    selected_idx: int | None,
    overlay_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for idx, line in enumerate(lines):
        x, y, w, h = line.box
        x2, y2 = x + w, y + h
        is_selected = selected_idx is not None and idx == selected_idx
        if show_boxes:
            outline = "red" if is_selected else "lime"
            draw.rectangle([(x, y), (x2, y2)], outline=outline, width=2 if is_selected else 1)
        if show_labels and line.text:
            text = line.text
            text_bbox = draw.textbbox((x, y), text, font=overlay_font)
            tx1, ty1, tx2, ty2 = text_bbox
            pad = 3
            draw.rectangle([(tx1 - pad, ty1 - pad), (tx2 + pad, ty2 + pad)], fill="black")
            text_color = "red" if is_selected else "yellow"
            draw.text((x, y), text, font=overlay_font, fill=text_color)
    return out


def _discover_images(folder: Path) -> list[Path]:
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


def _boxes_to_ocr_lines(boxes: list[tuple[int, int, int, int]]) -> list[OcrLine]:
    return [OcrLine(box=tuple(int(v) for v in b), text="(text)") for b in boxes]


def _save_ocr_json(
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


class TestImagesViewerApp:
    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4
    _RMB_ZOOM_PER_PIXEL = 1.0012

    def __init__(self, root: tk.Tk, images_dir: Path):
        self.root = root
        self.images_dir = images_dir
        self.image_paths: list[Path] = _discover_images(images_dir)
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

        self._ui_font = _configure_ui_fonts(root, UI_FONT_SIZE)
        self._overlay_font = _pil_overlay_font(OVERLAY_FONT_SIZE)

        self._build_ui()
        self._reload_image_list()

    def _build_ui(self) -> None:
        self.root.title("Test Images — YOLO & OCR")
        self.root.geometry("1280x840")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
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
            text="YOLO text boxes (ocr_image)",
            command=self._run_yolo_text_boxes,
        ).grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(
            controls,
            text="YOLO+OCR (get_coordinates_from_path)",
            command=self._run_yolo_ocr,
        ).grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(controls, text="YOLO confidence").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.yolo_conf_var, width=10).grid(
            row=5, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=(6, 0)
        )
        ttk.Button(
            controls,
            text="YOLO best.onnx (text+element)",
            command=self._run_yolo_onnx,
        ).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(
            controls,
            text="YOLO .pt (Ultralytics)",
            command=self._run_ultralytics_yolo,
        ).grid(row=7, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Reset Zoom", command=self._reset_zoom).grid(
            row=8, column=0, columnspan=4, sticky="ew", pady=(6, 0)
        )

        canvas_wrap = ttk.Frame(self.root, padding=8)
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

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.root.bind(key, self._on_arrow_key)
            self.canvas.bind(key, self._on_arrow_key)
            self.item_list.bind(key, self._on_arrow_key)
            self.image_list.bind(key, self._on_arrow_key)
        self.root.bind("<Control-plus>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-equal>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-minus>", self._on_zoom_out_hotkey)
        self.root.bind("<Control-0>", self._on_reset_zoom_hotkey)
        self.root.bind("<Configure>", lambda _e: self._refresh_image())

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
        self.image_paths = _discover_images(self.images_dir)
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
            text = line.text if line.text else "<empty>"
            self.item_list.insert(tk.END, f"{idx + 1:03d}: {text}")

    def _on_item_select(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            self.selected_line_idx = None
            self._refresh_image()
            return
        self.selected_line_idx = selected[0]
        self._refresh_image()

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
        dest_var = tk.StringVar(value=str(OCR_EXPORT_DEFAULT_DIR))
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

        def _save_text() -> None:
            new_text = text_var.get().strip()
            self.current_lines[idx] = OcrLine(box=line.box, text=new_text)
            self._populate_item_list()
            self.item_list.select_clear(0, tk.END)
            self.item_list.select_set(idx)
            self.item_list.see(idx)
            self.selected_line_idx = idx
            self._refresh_image()
            self.status_var.set(f"Updated OCR text for item #{idx + 1}")

        def _export_current() -> None:
            _save_text()
            try:
                saved = self._export_line_variants(idx, text_var.get().strip(), Path(dest_var.get().strip()))
            except Exception as exc:
                self.status_var.set(f"Export failed: {type(exc).__name__}: {exc}")
                return
            self.status_var.set(f"Exported {saved} files for item #{idx + 1}")

        ttk.Button(button_bar, text="Save Text", command=_save_text).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_bar, text="Export", command=_export_current).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(button_bar, text="Close", command=dialog.destroy).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        text_entry.focus_set()
        text_entry.selection_range(0, tk.END)
        dialog.bind("<Return>", lambda _event: _save_text())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())

    def _variant_crop_boxes(
        self, box: tuple[int, int, int, int], image_w: int, image_h: int
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            raise ValueError("invalid OCR item box dimensions")
        base_l = max(0, x)
        base_t = max(0, y)
        base_r = min(image_w, x + w)
        base_b = min(image_h, y + h)
        if base_r <= base_l or base_b <= base_t:
            raise ValueError("OCR box is outside image bounds")

        dx = max(1, int(round((base_r - base_l) * 0.05)))
        dy = max(1, int(round((base_b - base_t) * 0.05)))
        sx = max(1, int(round((base_r - base_l) * 0.02)))
        sy = max(1, int(round((base_b - base_t) * 0.02)))

        expanded_l = max(0, base_l - dx)
        expanded_t = max(0, base_t - dy)
        expanded_r = min(image_w, base_r + dx)
        expanded_b = min(image_h, base_b + dy)

        shrunk_l = min(base_r - 1, base_l + sx)
        shrunk_t = min(base_b - 1, base_t + sy)
        shrunk_r = max(shrunk_l + 1, base_r - sx)
        shrunk_b = max(shrunk_t + 1, base_b - sy)
        if shrunk_r <= shrunk_l or shrunk_b <= shrunk_t:
            shrunk_l, shrunk_t, shrunk_r, shrunk_b = base_l, base_t, base_r, base_b

        return [
            ("orig", (base_l, base_t, base_r, base_b)),
            ("expand5", (expanded_l, expanded_t, expanded_r, expanded_b)),
            ("shrink2", (shrunk_l, shrunk_t, shrunk_r, shrunk_b)),
        ]

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
        variants = self._variant_crop_boxes(self.current_lines[idx].box, img_w, img_h)
        saved_files = 0
        for suffix, crop_box in variants:
            crop = self.current_image.crop(crop_box)
            stem = f"{base_name}_item{idx + 1:03d}_{suffix}"
            out_img = dest_dir / f"{stem}.png"
            out_json = dest_dir / f"{stem}.json"
            crop.save(out_img)
            write_json(
                out_json,
                {
                    "rec_text": corrected_text,
                    "revised": True,
                    "char": "",
                },
            )
            saved_files += 2
        return saved_files

    def _set_lines(self, lines: list[OcrLine], status: str) -> None:
        self.current_lines = lines
        self.selected_line_idx = None
        self._populate_item_list()
        self._refresh_image()
        self.status_var.set(status)

    def _run_yolo_text_boxes(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected")
            return
        self.status_var.set("Running YOLO text boxes (get_text_boxes_from_path)…")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        try:
            boxes = get_text_boxes_from_path(str(src))
        except Exception as exc:
            self.status_var.set(f"YOLO text boxes failed: {type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        lines = _boxes_to_ocr_lines(boxes)
        try:
            out_path = _save_ocr_json(
                src, lines, yolo_elapsed_ms=elapsed_ms, ocr_elapsed_ms=None, source="yolo_text_boxes"
            )
            save_note = f", saved {out_path.name}"
        except Exception as exc:
            save_note = f", save failed: {exc}"
        self._set_lines(
            lines,
            f"YOLO text: {len(lines)} boxes in {elapsed_ms:.0f} ms{save_note}",
        )

    def _run_yolo_ocr(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected")
            return
        self.status_var.set("Running YOLO+OCR (get_coordinates_from_path)…")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        try:
            _offset, regions = get_coordinates_from_path(str(src))
            hint = format_coordinate_text_from_regions(regions)
        except Exception as exc:
            self.status_var.set(f"YOLO+OCR failed: {type(exc).__name__}: {exc}")
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        lines = _regions_to_ocr_lines(regions)
        try:
            out_path = _save_ocr_json(
                src,
                lines,
                yolo_elapsed_ms=None,
                ocr_elapsed_ms=elapsed_ms,
                source="get_coordinates_from_path",
            )
            save_note = f", saved {out_path.name}"
        except Exception as exc:
            save_note = f", save failed: {exc}"
        preview = hint.replace("\n", " | ")[:120]
        if len(hint) > 120:
            preview += "…"
        self._set_lines(
            lines,
            f"YOLO+OCR: {len(lines)} regions in {elapsed_ms:.0f} ms{save_note} — {preview}",
        )

    def _run_yolo_onnx(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected")
            return
        conf, err = _parse_conf_0_to_1(self.yolo_conf_var.get())
        if conf is None:
            self.status_var.set(f"Invalid confidence: {err}")
            return
        self.status_var.set(f"Running YOLO ONNX (conf={conf:g})…")
        self.root.update_idletasks()
        try:
            lines, elapsed_ms = _run_yolo_onnx_best_detections(src, conf_threshold=conf)
        except Exception as exc:
            self.status_var.set(f"YOLO ONNX failed: {type(exc).__name__}: {exc}")
            return
        self._set_lines(lines, f"YOLO ONNX: {len(lines)} boxes in {elapsed_ms:.0f} ms")

    def _run_ultralytics_yolo(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected")
            return
        conf, err = _parse_conf_0_to_1(self.yolo_conf_var.get())
        if conf is None:
            self.status_var.set(f"Invalid confidence: {err}")
            return
        self.status_var.set(
            f"Running Ultralytics YOLO ({DEFAULT_ULTRALYTICS_PT_PATH.name}, conf={conf:g})…"
        )
        self.root.update_idletasks()
        try:
            lines, elapsed_ms = _run_ultralytics_yolo_pt(
                src,
                DEFAULT_ULTRALYTICS_PT_PATH,
                conf=conf,
                model_holder=self._ultralytics_model_holder,
            )
        except ImportError as exc:
            self.status_var.set(str(exc))
            return
        except Exception as exc:
            self.status_var.set(f"Ultralytics YOLO failed: {type(exc).__name__}: {exc}")
            return
        self._set_lines(
            lines,
            f"Ultralytics YOLO: {len(lines)} boxes in {elapsed_ms:.0f} ms",
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
        for idx, line in enumerate(self.current_lines):
            x, y, w, h = line.box
            if x <= img_x <= x + w and y <= img_y <= y + h:
                return idx
        return None

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


def run_app(images_dir: Path | None = None) -> None:
    base = images_dir if images_dir is not None else DEFAULT_TEST_IMAGES_DIR
    root = tk.Tk()
    TestImagesViewerApp(root, base)
    root.mainloop()


if __name__ == "__main__":
    run_app()
