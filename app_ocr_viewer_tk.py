from __future__ import annotations

import re
import shutil
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

from cua_mcp.read_screen_text import ocr_image as ocr_image_mod
from cua_mcp.read_screen_text.ocr_image import (
    format_coordinate_text_from_regions,
    get_coordinates_from_path,
)
from cua_mcp.yolo_onnx import (
    DEFAULT_CONF_YOLOV26_END2END,
    DEFAULT_YOLO_ONNX_PATH,
    YOLO_CLASS_ELEMENT,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_TEXT,
    run_best_onnx_end2end,
)
from src.common.io_utils import read_json, write_json
from src.common.settings import ROOT_DIR

SCREENSHOT_CREATOR_UNDONE_IMAGES = Path(
    r"C:\Users\Joseph Hung\Documents\Repos\Git\ScreenshotCreator\real_screenshot\undone\images"
)
OCR_EXPORT_DEFAULT_DIR = Path(r"C:\Users\Joseph Hung\Documents\Repos\Git\crnn.pytorch\finetune\data")
# Same weights as ONNX export used by OCR (`cua_mcp/yolo_onnx.DEFAULT_YOLO_ONNX_PATH`).
DEFAULT_ULTRALYTICS_PT_PATH = ROOT_DIR / "cua_mcp" / "best.pt"

BOX_EDIT_STEP = 1
BOX_EDIT_STEP_SHIFT = 8
MIN_BOX_SIZE = 2


@dataclass(frozen=True)
class OcrLine:
    box: tuple[int, int, int, int]
    text: str


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
                parsed.append(OcrLine(box=(x, y, w, h), text=str(item.get("text", "")).strip()))
    return parsed


def load_ocr_lines(json_path: Path) -> tuple[list[OcrLine], str]:
    if not json_path.exists():
        return [], "Missing OCR JSON"
    try:
        data = read_json(json_path, default={})
    except Exception as exc:
        return [], f"JSON parse error: {exc}"
    if not isinstance(data, dict):
        return [], "Invalid JSON root"
    lines = _normalize_lines(data.get("lines", []))
    return lines, f"Loaded {len(lines)} OCR lines"


def _discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.iterdir() if p.is_dir()], reverse=True)


def _yolo_ocr_paired_images(run_dir: Path) -> list[Path]:
    """PNG/JPEG files in yolo_ocr/ that have a sibling JSON with the same stem."""
    yolo_dir = run_dir / "yolo_ocr"
    if not yolo_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(yolo_dir.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if p.with_suffix(".json").is_file():
            out.append(p)
    return out


def _draw_overlays(
    image: Image.Image,
    lines: list[OcrLine],
    show_boxes: bool,
    show_labels: bool,
    selected_idx: int | None = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for idx, line in enumerate(lines):
        x, y, w, h = line.box
        x2, y2 = x + w, y + h
        is_selected = selected_idx is not None and idx == selected_idx
        if show_boxes:
            outline = "red" if is_selected else "lime"
            draw.rectangle([(x, y), (x2, y2)], outline=outline, width=1)
        if show_labels and line.text:
            text = line.text
            text_bbox = draw.textbbox((x, y), text, font=font)
            tx1, ty1, tx2, ty2 = text_bbox
            pad = 2
            draw.rectangle([(tx1 - pad, ty1 - pad), (tx2 + pad, ty2 + pad)], fill="black")
            text_color = "red" if is_selected else "yellow"
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


def _run_ocr_with_boxes(image_path: Path) -> tuple[list[OcrLine], float | None, float]:
    """Run YOLO+OCR and return full rectangle boxes with timings."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"could not read image: {image_path}")
    img_h, img_w = bgr.shape[:2]
    predictor = ocr_image_mod._get_crnn_predictor()  # noqa: SLF001

    yolo_start = time.perf_counter()
    boxes = ocr_image_mod._yolo_text_boxes(bgr)  # noqa: SLF001
    yolo_elapsed_ms = (time.perf_counter() - yolo_start) * 1000.0
    if not boxes:
        boxes = [(0, 0, img_w, img_h)]
    boxes = ocr_image_mod._merge_overlapping_boxes(boxes)  # noqa: SLF001
    boxes = ocr_image_mod._sort_boxes_reading_order(boxes)  # noqa: SLF001

    out_lines: list[OcrLine] = []
    ocr_elapsed_ms = 0.0
    for x, y, w, h in boxes:
        crop = bgr[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        ocr_start = time.perf_counter()
        preds = ocr_image_mod._ocr_crop_predicted_texts(crop, predictor, 32)  # noqa: SLF001
        text = "".join(preds).strip()
        ocr_elapsed_ms += (time.perf_counter() - ocr_start) * 1000.0
        out_lines.append(OcrLine(box=(x, y, w, h), text=text))
    return out_lines, yolo_elapsed_ms, ocr_elapsed_ms


def _run_ultralytics_yolo_pt(
    image_path: Path,
    weights_path: Path,
    *,
    conf: float,
    model_holder: list[Any],
) -> tuple[list[OcrLine], float]:
    """
    Run Ultralytics ``YOLO`` on ``weights_path``; overlay labels use ``class conf``.
    ``model_holder`` is a single-element list caching the loaded model across clicks.

    ``conf`` is passed to ``model.predict(conf=...)`` (0–1 inclusive).
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc

    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    t0 = time.perf_counter()
    if not model_holder:
        model_holder.append(YOLO(str(weights_path)))
    model = model_holder[0]
    results = model.predict(
        str(image_path),
        conf=float(conf),
        imgsz=640,
        iou=0.7,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not results:
        return [], elapsed_ms
    r0 = results[0]
    names: dict[int, str] = getattr(model, "names", {}) or {}
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return [], elapsed_ms

    xyxy = boxes.xyxy.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    lines: list[OcrLine] = []
    for i in range(len(xyxy)):
        x1f, y1f, x2f, y2f = (float(v) for v in xyxy[i])
        x1 = int(round(x1f))
        y1 = int(round(y1f))
        x2 = int(round(x2f))
        y2 = int(round(y2f))
        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)
        cid = int(cls_arr[i])
        label = str(names.get(cid, str(cid)))
        conf_val = float(conf_arr[i])
        lines.append(OcrLine(box=(x1, y1, w_box, h_box), text=f"{label} {conf_val:.2f}"))
    lines.sort(key=lambda ln: (ln.box[1] + ln.box[3] // 2, ln.box[0] + ln.box[2] // 2))
    return lines, elapsed_ms


def _run_yolo_onnx_best_detections(
    image_path: Path, *, conf_threshold: float
) -> tuple[list[OcrLine], float]:
    """
    Run ``cua_mcp/best.onnx`` via :func:`run_best_onnx_end2end` (ONNX Runtime, CPU).

    Keeps ``text`` and ``element`` detections above ``conf_threshold``.
    """
    if not DEFAULT_YOLO_ONNX_PATH.is_file():
        raise FileNotFoundError(f"YOLO ONNX model not found: {DEFAULT_YOLO_ONNX_PATH}")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"could not read image: {image_path}")
    t0 = time.perf_counter()
    xyxy, scores, cls_ids = run_best_onnx_end2end(
        bgr,
        class_ids={YOLO_CLASS_TEXT, YOLO_CLASS_ELEMENT},
        conf_threshold=float(conf_threshold),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    lines: list[OcrLine] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)
        cid = int(cls_ids[i])
        label = YOLO_CLASS_NAMES.get(cid, str(cid))
        conf_val = float(scores[i])
        lines.append(OcrLine(box=(x1, y1, w_box, h_box), text=f"{label} {conf_val:.2f}"))
    lines.sort(key=lambda ln: (ln.box[1] + ln.box[3] // 2, ln.box[0] + ln.box[2] // 2))
    return lines, elapsed_ms


class OcrViewerApp:
    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4  # pixels²; drag beyond this pans instead of selecting
    # Right-drag: each horizontal pixel nudges zoom by this power (right = in, left = out).
    _RMB_ZOOM_PER_PIXEL = 1.0012

    def __init__(self, root: tk.Tk, runs_root: Path):
        self.root = root
        self.runs_root = runs_root
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

        self._build_ui()
        self._populate_runs()

    def _build_ui(self) -> None:
        self.root.title("OCR Overlay Viewer")
        self.root.geometry("1280x840")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Runs").grid(row=0, column=0, sticky="w")
        self.run_list = tk.Listbox(left, exportselection=False, height=8, width=48)
        self.run_list.grid(row=1, column=0, sticky="nsew")
        self.run_list.bind("<<ListboxSelect>>", self._on_run_select)

        ttk.Label(left, text="YOLO OCR (image + JSON)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.image_list = tk.Listbox(left, exportselection=False, height=10, width=48)
        self.image_list.grid(row=3, column=0, sticky="nsew")
        self.image_list.bind("<<ListboxSelect>>", self._on_image_select)

        ttk.Label(left, text="OCR Items").grid(row=4, column=0, sticky="w", pady=(8, 0))
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
        ttk.Button(controls, text="Run YOLO+OCR", command=self._run_ocr_current_image).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(controls, text="Copy to undone/images", command=self._copy_current_image_to_undone).grid(
            row=3, column=2, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Label(controls, text="YOLO confidence").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.yolo_conf_var, width=10).grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=(6, 0)
        )
        ttk.Button(
            controls,
            text="YOLO .pt (Ultralytics)",
            command=self._run_ultralytics_yolo_current_image,
        ).grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(
            controls,
            text="YOLO best.onnx (ORT)",
            command=self._run_yolo_onnx_current_image,
        ).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Reset Zoom", command=self._reset_zoom).grid(
            row=7, column=0, columnspan=4, sticky="ew", pady=(6, 0)
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
            self.run_list.bind(key, self._on_arrow_key)
        self.root.bind("<Control-plus>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-equal>", self._on_zoom_in_hotkey)
        self.root.bind("<Control-minus>", self._on_zoom_out_hotkey)
        self.root.bind("<Control-0>", self._on_reset_zoom_hotkey)
        self.root.bind("<Configure>", lambda _event: self._refresh_image())

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
        self.image_list.delete(0, tk.END)
        for img in self.current_run_images:
            self.image_list.insert(tk.END, img.name)
        if self.current_run_images:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self.current_image = None
            self.current_lines = []
            self.item_list.delete(0, tk.END)
            self.canvas.delete("all")
            self.status_var.set(
                f"No paired image+JSON in yolo_ocr for {run.name if run else '-'}"
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

    def _on_image_select(self, _event: object | None = None) -> None:
        image_path = self._current_image_path()
        run = self._selected_run()
        if image_path is None or run is None:
            return
        self.current_image = Image.open(image_path).convert("RGB")
        self._view_zoom = 1.0
        json_path = image_path.with_suffix(".json")
        self.current_lines, status = load_ocr_lines(json_path)
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

        ttk.Label(dialog, text="Corrected text").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))
        text_var = tk.StringVar(value=line.text)
        text_entry = ttk.Entry(dialog, textvariable=text_var, width=72)
        text_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=10, pady=(10, 6))

        ttk.Label(dialog, text="Export folder").grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))
        dest_var = tk.StringVar(value=str(OCR_EXPORT_DEFAULT_DIR))
        dest_entry = ttk.Entry(dialog, textvariable=dest_var, width=72)
        dest_entry.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 6))

        def _browse_folder() -> None:
            chosen = filedialog.askdirectory(initialdir=dest_var.get() or str(OCR_EXPORT_DEFAULT_DIR))
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

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        rendered = _draw_overlays(
            self.current_image,
            self.current_lines,
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

        self.selected_line_idx = selected_idx
        self.item_list.select_set(selected_idx)
        self.item_list.see(selected_idx)
        self._refresh_image()

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
        nxt = min(len(self.current_run_images) - 1, idx + 1)
        self.image_list.select_clear(0, tk.END)
        self.image_list.select_set(nxt)
        self.image_list.see(nxt)
        self._on_image_select()

    def _run_ultralytics_yolo_current_image(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected for Ultralytics YOLO")
            return
        conf_pt, err = _parse_conf_0_to_1(self.yolo_conf_var.get())
        if conf_pt is None:
            self.status_var.set(f"Ultralytics .pt — invalid confidence: {err}")
            return
        self.status_var.set(
            f"Running Ultralytics YOLO ({DEFAULT_ULTRALYTICS_PT_PATH.name}, conf={conf_pt:g})..."
        )
        self.root.update_idletasks()
        try:
            lines, elapsed_ms = _run_ultralytics_yolo_pt(
                src,
                DEFAULT_ULTRALYTICS_PT_PATH,
                conf=conf_pt,
                model_holder=self._ultralytics_model_holder,
            )
        except ImportError as exc:
            self.status_var.set(str(exc))
            return
        except Exception as exc:
            self.status_var.set(f"Ultralytics YOLO failed: {type(exc).__name__}: {exc}")
            return
        self.current_lines = lines
        self.selected_line_idx = None
        self._populate_item_list()
        self._refresh_image()
        self.status_var.set(
            f"Ultralytics YOLO: {len(lines)} boxes in {elapsed_ms:.0f} ms "
            f"(conf={conf_pt:g}, {DEFAULT_ULTRALYTICS_PT_PATH.name})"
        )

    def _run_yolo_onnx_current_image(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected for YOLO ONNX")
            return
        conf_onnx, err = _parse_conf_0_to_1(self.yolo_conf_var.get())
        if conf_onnx is None:
            self.status_var.set(f"YOLO ONNX — invalid confidence: {err}")
            return
        self.status_var.set(
            f"Running YOLO ONNX ({DEFAULT_YOLO_ONNX_PATH.name}, conf={conf_onnx:g})..."
        )
        self.root.update_idletasks()
        try:
            lines, elapsed_ms = _run_yolo_onnx_best_detections(src, conf_threshold=conf_onnx)
        except Exception as exc:
            self.status_var.set(f"YOLO ONNX failed: {type(exc).__name__}: {exc}")
            return
        self.current_lines = lines
        self.selected_line_idx = None
        self._populate_item_list()
        self._refresh_image()
        self.status_var.set(
            f"YOLO ONNX: {len(lines)} boxes in {elapsed_ms:.0f} ms "
            f"(conf={conf_onnx:g}, {DEFAULT_YOLO_ONNX_PATH.name})"
        )

    def _copy_current_image_to_undone(self) -> None:
        src = self._current_image_path()
        if src is None or not src.is_file():
            self.status_var.set("No image selected to copy")
            return
        dest_dir = SCREENSHOT_CREATOR_UNDONE_IMAGES
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            self.status_var.set(f"Copied to {dest}")
        except OSError as exc:
            self.status_var.set(f"Copy failed: {exc}")

    def _run_ocr_current_image(self) -> None:
        src = self._current_image_path()
        run = self._selected_run()
        if src is None or not src.is_file():
            self.status_var.set("No image selected for OCR")
            return
        if run is None:
            self.status_var.set("No run selected for OCR output")
            return
        self.status_var.set("Running YOLO+OCR...")
        self.root.update_idletasks()
        try:
            _offset, regions = get_coordinates_from_path(str(src))
            output = format_coordinate_text_from_regions(regions)
        except Exception as exc:
            self.status_var.set(f"YOLO+OCR failed: {type(exc).__name__}: {exc}")
            return
        if not regions:
            self.status_var.set("YOLO+OCR returned no regions (empty or failed)")
            return
        try:
            out_dir = src.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = src.with_suffix(".json")
            existing = read_json(out_path, default={}) if out_path.exists() else {}
            existing_data = existing if isinstance(existing, dict) else {}
            loaded_lines = _normalize_lines(existing_data.get("lines", []))
            has_real_boxes = any(line.box[2] > 4 or line.box[3] > 4 for line in loaded_lines)
            if loaded_lines and has_real_boxes:
                self.current_lines = loaded_lines
                yolo_elapsed_ms = existing_data.get("yolo_elapsed_ms")
                ocr_elapsed_ms = existing_data.get("ocr_elapsed_ms")
            else:
                self.current_lines, yolo_elapsed_ms, ocr_elapsed_ms = _run_ocr_with_boxes(src)
            line_pairs = [[list(line.box), line.text] for line in self.current_lines]
            write_json(
                out_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "image_path": str(src),
                    "image_name": src.name,
                    "line_height": 32,
                    "yolo_elapsed_ms": yolo_elapsed_ms,
                    "ocr_elapsed_ms": ocr_elapsed_ms,
                    "lines": line_pairs,
                    "text": "\n".join(line.text for line in self.current_lines),
                },
            )
            self.selected_line_idx = None
            self._populate_item_list()
            self._refresh_image()
            self.status_var.set(f"YOLO+OCR complete: {len(self.current_lines)} lines (saved: {out_path.name})")
        except Exception as exc:
            self.status_var.set(
                f"YOLO+OCR complete: {len(self.current_lines)} lines, but save failed: {type(exc).__name__}: {exc}"
            )


def run_app(runs_root: Path | None = None) -> None:
    base = runs_root if runs_root is not None else ROOT_DIR / "runs"
    root = tk.Tk()
    OcrViewerApp(root, base)
    root.mainloop()


if __name__ == "__main__":
    run_app()
