from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from src.common.io_utils import read_json
from src.common.settings import ROOT_DIR


def _discover_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.iterdir() if p.is_dir()], reverse=True)


@dataclass(frozen=True)
class UiElement:
    bbox: tuple[int, int, int, int]  # x,y,w,h
    center: tuple[int, int]  # cx,cy
    class_name: str
    value: str


def _yolo_ui_paired_images(run_dir: Path) -> list[Path]:
    """PNG/JPEG files in yolo_ui/ that have a sibling JSON with the same stem."""
    yolo_dir = run_dir / "yolo_ui"
    if not yolo_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(yolo_dir.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if p.name.endswith("_result.png") or p.name.endswith("_result.jpg") or p.name.endswith("_result.jpeg"):
            continue
        if p.with_suffix(".json").is_file():
            out.append(p)
    return out


def load_yolo_ui_elements(json_path: Path) -> tuple[list[UiElement], str]:
    if not json_path.exists():
        return [], "Missing yolo_ui JSON"
    try:
        data = read_json(json_path, default={})
    except Exception as exc:
        return [], f"JSON parse error: {exc}"
    if not isinstance(data, dict):
        return [], "Invalid JSON root"
    detections = data.get("detections", [])
    if not isinstance(detections, list):
        return [], "Invalid detections list"
    out: list[UiElement] = []
    for d in detections:
        if not isinstance(d, dict):
            continue
        bbox = d.get("bbox")
        center = d.get("center")
        if not isinstance(bbox, dict) or not isinstance(center, dict):
            continue
        try:
            x = int(bbox.get("x"))
            y = int(bbox.get("y"))
            w = int(bbox.get("w"))
            h = int(bbox.get("h"))
            cx = int(center.get("x"))
            cy = int(center.get("y"))
        except (TypeError, ValueError):
            continue
        class_name = str(d.get("class_name", "") or "")
        raw_value = d.get("value")
        if raw_value is None:
            # Fall back to class name when no explicit value is present.
            value = class_name or str(d.get("class_id", "") or "")
        else:
            value = str(raw_value)
        out.append(UiElement(bbox=(x, y, w, h), center=(cx, cy), class_name=class_name, value=value))
    return out, f"Loaded {len(out)} UI elements"


def _draw_ui_overlays(
    image: Image.Image,
    elements: list[UiElement],
    *,
    selected_idx: int | None,
    show_boxes: bool,
    show_centers: bool,
    show_labels: bool,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for idx, el in enumerate(elements):
        x, y, w, h = el.bbox
        x2, y2 = x + w, y + h
        cx, cy = el.center
        is_selected = selected_idx is not None and idx == selected_idx
        if show_boxes:
            outline = "red" if is_selected else "lime"
            width = 3 if is_selected else 2
            draw.rectangle([(x, y), (x2, y2)], outline=outline, width=width)
        if show_centers:
            r = 5 if is_selected else 3
            fill = "red" if is_selected else "cyan"
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=fill, fill=fill)
            draw.line([(cx - (r + 4), cy), (cx + (r + 4), cy)], fill=fill, width=2)
            draw.line([(cx, cy - (r + 4)), (cx, cy + (r + 4))], fill=fill, width=2)
        if show_labels:
            label = el.value or el.class_name or ""
            if label:
                tx = x
                ty = max(0, y - 12)
                text_bbox = draw.textbbox((tx, ty), label, font=font)
                (tx1, ty1, tx2, ty2) = text_bbox
                pad = 2
                draw.rectangle([(tx1 - pad, ty1 - pad), (tx2 + pad, ty2 + pad)], fill="black")
                draw.text((tx, ty), label, font=font, fill=("red" if is_selected else "yellow"))
    return out


class YoloUiViewerApp:
    _MIN_ZOOM = 0.125
    _MAX_ZOOM = 32.0
    _ZOOM_STEP = 1.15
    _PAN_CLICK_THRESHOLD_SQ = 4 * 4
    _RMB_ZOOM_PER_PIXEL = 1.0012

    def __init__(self, root: tk.Tk, runs_root: Path):
        self.root = root
        self.runs_root = runs_root
        self.run_dirs = _discover_runs(runs_root)
        self.current_run_images: list[Path] = []
        self.current_image: Image.Image | None = None
        self.current_display: ImageTk.PhotoImage | None = None

        self.elements_all: list[UiElement] = []
        self.elements_filtered: list[UiElement] = []
        self.filtered_to_all_index: list[int] = []
        self.selected_all_idx: int | None = None

        self.filter_var = tk.StringVar(value="")
        self.sort_var = tk.StringVar(value="A→Z")
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_centers = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self._view_zoom = 1.0
        self._rmb_last_x: int | None = None
        self._render_scale = 1.0
        self._lmb_press_xy: tuple[int, int] | None = None
        self._lmb_panning = False

        self._build_ui()
        self._populate_runs()
        self.filter_var.trace_add("write", lambda *_args: self._apply_filter())
        self.sort_var.trace_add("write", lambda *_args: self._apply_filter())

    def _build_ui(self) -> None:
        self.root.title("YOLO UI Viewer")
        self.root.geometry("1400x900")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Runs").grid(row=0, column=0, sticky="w")
        self.run_list = tk.Listbox(left, exportselection=False, height=8, width=52)
        self.run_list.grid(row=1, column=0, sticky="nsew")
        self.run_list.bind("<<ListboxSelect>>", self._on_run_select)

        ttk.Label(left, text="YOLO UI (image + JSON)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.image_list = tk.Listbox(left, exportselection=False, height=10, width=52)
        self.image_list.grid(row=3, column=0, sticky="nsew")
        self.image_list.bind("<<ListboxSelect>>", self._on_image_select)

        ttk.Label(left, text="Filter (value contains)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        filter_entry = ttk.Entry(left, textvariable=self.filter_var, width=52)
        filter_entry.grid(row=5, column=0, sticky="ew")

        sort_row = ttk.Frame(left)
        sort_row.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        sort_row.columnconfigure(1, weight=1)
        ttk.Label(sort_row, text="Sort").grid(row=0, column=0, sticky="w")
        self.sort_combo = ttk.Combobox(
            sort_row,
            textvariable=self.sort_var,
            values=["A→Z", "Z→A"],
            state="readonly",
            width=10,
        )
        self.sort_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(left, text="UI Elements").grid(row=7, column=0, sticky="w", pady=(8, 0))
        item_wrap = ttk.Frame(left)
        item_wrap.grid(row=8, column=0, sticky="nsew")
        item_wrap.columnconfigure(0, weight=1)
        item_wrap.rowconfigure(0, weight=1)
        self.item_list = tk.Listbox(item_wrap, exportselection=False, height=16, width=52)
        self.item_list.grid(row=0, column=0, sticky="nsew")
        self.item_scroll = ttk.Scrollbar(item_wrap, orient="vertical", command=self.item_list.yview)
        self.item_scroll.grid(row=0, column=1, sticky="ns")
        self.item_list.configure(yscrollcommand=self.item_scroll.set)
        self.item_list.bind("<<ListboxSelect>>", self._on_item_select)

        preview = ttk.Labelframe(left, text="Selected icon crop", padding=6)
        preview.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        preview.columnconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview)
        self.preview_label.grid(row=0, column=0, sticky="ew")
        self._preview_img: ImageTk.PhotoImage | None = None

        controls = ttk.Frame(left)
        controls.grid(row=10, column=0, sticky="ew", pady=(8, 0))
        for col in range(4):
            controls.columnconfigure(col, weight=1)
        ttk.Checkbutton(controls, text="Boxes", variable=self.show_boxes, command=self._refresh_image).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(controls, text="Centers", variable=self.show_centers, command=self._refresh_image).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Checkbutton(controls, text="Labels", variable=self.show_labels, command=self._refresh_image).grid(
            row=0, column=2, sticky="w"
        )
        ttk.Button(controls, text="Prev", command=self._prev_image).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Next", command=self._next_image).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Zoom +", command=self._zoom_in).grid(row=1, column=2, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Zoom -", command=self._zoom_out).grid(row=1, column=3, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Reset Zoom", command=self._reset_zoom).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0)
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
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        self.root.bind("<Left>", lambda _event: self._prev_image())
        self.root.bind("<Right>", lambda _event: self._next_image())
        self.root.bind("<Control-plus>", lambda _event: self._zoom_in() or "break")
        self.root.bind("<Control-equal>", lambda _event: self._zoom_in() or "break")
        self.root.bind("<Control-minus>", lambda _event: self._zoom_out() or "break")
        self.root.bind("<Control-0>", lambda _event: self._reset_zoom() or "break")
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
        self.current_run_images = _yolo_ui_paired_images(run) if run is not None else []
        self.image_list.delete(0, tk.END)
        for img in self.current_run_images:
            self.image_list.insert(tk.END, img.name)
        if self.current_run_images:
            self.image_list.select_set(0)
            self._on_image_select()
        else:
            self.current_image = None
            self.elements_all = []
            self._apply_filter()
            self.canvas.delete("all")
            self.status_var.set(f"No paired image+JSON in yolo_ui for {run.name if run else '-'}")

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
        if image_path is None:
            return
        self.current_image = Image.open(image_path).convert("RGB")
        self._view_zoom = 1.0
        self.selected_all_idx = None
        self.filter_var.set(self.filter_var.get())  # keep current filter
        self.elements_all, status = load_yolo_ui_elements(image_path.with_suffix(".json"))
        self._apply_filter()
        self.status_var.set(f"{image_path.name} - {status}")
        self._refresh_image()

    def _apply_filter(self) -> None:
        text = (self.filter_var.get() or "").strip().lower()
        indices: list[int] = []
        if not text:
            indices = list(range(len(self.elements_all)))
        else:
            for i, el in enumerate(self.elements_all):
                hay = f"{el.value} {el.class_name}".lower()
                if text in hay:
                    indices.append(i)

        reverse = (self.sort_var.get() or "A→Z").strip() == "Z→A"
        indices.sort(key=lambda i: (self.elements_all[i].value or "").lower(), reverse=reverse)

        self.elements_filtered = [self.elements_all[i] for i in indices]
        self.filtered_to_all_index = indices
        self._populate_item_list()
        # If current selection is filtered out, clear it.
        if self.selected_all_idx is not None and self.selected_all_idx not in set(self.filtered_to_all_index):
            self.selected_all_idx = None
            self._update_preview()
            self._refresh_image()

    def _populate_item_list(self) -> None:
        self.item_list.delete(0, tk.END)
        for display_idx, el in enumerate(self.elements_filtered):
            x, y, w, h = el.bbox
            cx, cy = el.center
            value = el.value or el.class_name or "<empty>"
            self.item_list.insert(
                tk.END,
                f"{display_idx + 1:03d}  value={value}  bbox=[{x},{y},{w},{h}]  center=[{cx},{cy}]",
            )

    def _on_item_select(self, _event: object | None = None) -> None:
        selected = self.item_list.curselection()
        if not selected:
            self.selected_all_idx = None
            self._update_preview()
            self._refresh_image()
            return
        disp_idx = selected[0]
        if disp_idx < 0 or disp_idx >= len(self.filtered_to_all_index):
            return
        self.selected_all_idx = self.filtered_to_all_index[disp_idx]
        self._update_preview()
        self._refresh_image()

    def _update_preview(self) -> None:
        self._preview_img = None
        self.preview_label.configure(image="")
        if self.current_image is None or self.selected_all_idx is None:
            return
        if self.selected_all_idx < 0 or self.selected_all_idx >= len(self.elements_all):
            return
        x, y, w, h = self.elements_all[self.selected_all_idx].bbox
        img_w, img_h = self.current_image.size
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(1, min(x + max(1, w), img_w))
        y2 = max(1, min(y + max(1, h), img_h))
        crop = self.current_image.crop((x1, y1, x2, y2))
        max_side = 220
        cw, ch = crop.size
        scale = min(1.0, max_side / max(1, max(cw, ch)))
        if scale < 1.0:
            crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.Resampling.LANCZOS)
        self._preview_img = ImageTk.PhotoImage(crop)
        self.preview_label.configure(image=self._preview_img)

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

    def _ui_hit_index_at_canvas(self, event: tk.Event[tk.Canvas]) -> int | None:
        if self.current_image is None or not self.elements_all:
            return None
        canvas_x = self.canvas.canvasx(int(event.x))
        canvas_y = self.canvas.canvasy(int(event.y))
        img_x = int(canvas_x / max(self._render_scale, 1e-6))
        img_y = int(canvas_y / max(self._render_scale, 1e-6))
        for idx, el in enumerate(self.elements_all):
            x, y, w, h = el.bbox
            if x <= img_x <= x + w and y <= img_y <= y + h:
                return idx
        return None

    def _on_lmb_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._lmb_press_xy is None:
            return
        try:
            if not self._lmb_panning:
                idx = self._ui_hit_index_at_canvas(event)
                self.selected_all_idx = idx
                self._update_preview()
                self._sync_list_selection()
                self._refresh_image()
        finally:
            self._lmb_press_xy = None
            self._lmb_panning = False

    def _sync_list_selection(self) -> None:
        self.item_list.select_clear(0, tk.END)
        if self.selected_all_idx is None:
            return
        try:
            disp_idx = self.filtered_to_all_index.index(self.selected_all_idx)
        except ValueError:
            return
        self.item_list.select_set(disp_idx)
        self.item_list.see(disp_idx)

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

    def _refresh_image(self) -> None:
        if self.current_image is None:
            return
        rendered = _draw_ui_overlays(
            self.current_image,
            self.elements_all,
            selected_idx=self.selected_all_idx,
            show_boxes=self.show_boxes.get(),
            show_centers=self.show_centers.get(),
            show_labels=self.show_labels.get(),
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


def run_app(runs_root: Path | None = None) -> None:
    base = runs_root if runs_root is not None else ROOT_DIR / "runs"
    root = tk.Tk()
    YoloUiViewerApp(root, base)
    root.mainloop()


if __name__ == "__main__":
    run_app()
