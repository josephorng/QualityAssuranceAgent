# ROI-gated YOLO enhance / OCR (implementation notes)

Notes from the 2026-09-22 dialog so this can be re-implemented later.
Related context: outside-taskbar landmarks, wrong `click_window`, missing Excel click after「篩選」.

Status check (2026-09-22): `click_window` / `resolve_ocr_roi_local` / ROI gating APIs
are **absent** from the tree — Phase 0 is required, not optional. Deferred-click
flush in `capture.py` (`_flush_superseded_pending_left_gesture`) **already exists**;
only verify + add a unit test if missing.

---

## Goal

Limit **YOLO Stage 3/4 enhance**, **OCR**, and **input line-rectangle merge** to the
target window (recorded `click_window` / taskbar), not the full desktop.

- **Keep full-frame first YOLO** (Stages 1–2 / recursive detect).
- **Gate only follow-up work**: overlap refine, line cut, OCR, input refine,
  candidate admission.
- **Skip the gate** (treat as full-frame) when the window is maximized or covers
  ≥ ~80% of the image.
- **Cluster rule**: if *any* box center in a refine cluster falls in the padded
  ROI, refine the whole cluster.
- **Post-enhance filter still applies**: cluster members refined outside the ROI
  must not become OCR'd mouse candidates — admission is always center-in-padded-ROI
  after Stage 3/4.

Motivation: nearby landmarks and OCR noise from other windows (taskbar, other
apps) pollute instructions and replay targeting.

---

## Design rules

| Concern | Behavior |
|--------|----------|
| First YOLO pass | Always full-frame |
| Stage 3 `overlap_refine` | Only clusters with ≥1 member center in padded ROI |
| Stage 4 `line_cut` | Only tall text boxes whose center is in padded ROI |
| OCR | Only boxes whose center is in padded ROI |
| Text / element candidates | Drop if center outside ROI when ROI is set (same gate as OCR admission) |
| Input line refine | Only in-ROI inputs + in-ROI horizontal scrollbars passed to merge; drop out-ROI inputs before refine (unless debug overlay needs pre-merge list) |
| Non-OCR candidates (input/scrollbar/…) | Drop if center outside ROI when ROI is set |
| Maximized / ≥~80% coverage | `ocr_roi` / `enhance_roi` = `None` → no gating |
| Missing `click_window` (legacy recordings) | `None` ROI → ungated full-frame (compat) |
| Recording | Resolve ROI from press-time `click_window` |
| Replay | Resolve ROI from `move_mouse(click_window=…)` via live window match |

**Naming:** `ocr_roi` (select / vision layer) and `enhance_roi` (YOLO Stage 3/4)
are the **same image-local xywh** from `resolve_ocr_roi_local`. Keep both names at
API boundaries; do not invent a second rect.

ROI geometry: image-local **xywh**, with pad (default **16px**).

Coverage:

```text
coverage = area(roi clipped to image) / (image_w * image_h)
return None if maximized or coverage >= max_coverage (~0.8)
```

Replay resolves per monitor with that monitor's `monitor_offset` so a window on
monitor B does not gate monitor A's full-frame detections incorrectly.

Center-in-rect helper (same idea as `point_in_rect_xywh`):

```text
cx, cy = bbox center
x - pad <= cx < x + w + pad  and  y - pad <= cy < y + h + pad
```

### Pipeline order (when ROI is set)

1. Full-frame YOLO Stages 1–2.
2. Stage 3: refine only clusters intersecting padded ROI (whole cluster).
3. Stage 4: line-cut only tall boxes with center in padded ROI.
4. Input refine: only in-ROI inputs (and in-ROI horizontal scrollbars for merge geometry).
5. **Admit** text / element / input / scrollbar candidates only if center in padded ROI
   (OCR runs only on admitted OCR-class boxes).

Step 5 is the hard safety net: enhanced-but-out-of-ROI cluster siblings stay out
of instructions and `move_mouse` candidates.

### Observability

When `ocr_roi` / `enhance_roi` is set, log one line with: roi xywh, pad, clusters
skipped, OCR boxes skipped, candidates dropped. Useful when live `click_window`
matching is wrong.

---

## Phase 0 — `click_window` + ROI resolver (required)

ROI gating assumes press-time `click_window` on recorded events and replay tool
calls. As of this note, those pieces are missing — restore / implement first.

### Payload schema (sketch)

`RecordedEvent.click_window: dict | None`, roughly:

```text
hwnd, title, process_name (or exe),
rect: local xywh (or screen + convert),
is_maximized: bool,
is_taskbar: bool (or class/role),
optional: is_flyout / owner hints
```

Serialize in `to_dict` / `from_dict`. Compile into `move_mouse(click_window=…)`.

### Files

1. **`src/recorder/models.py`** — `RecordedEvent.click_window: dict | None`
2. **`src/recorder/window_snapshot.py`**
   - `ClickWindowInfo`, `resolve_click_window(x, y)`
   - **`resolve_ocr_roi_local(click_window, image_w, image_h, monitor_offset=(0,0), max_coverage≈0.8) -> RectXywh | None`**
     - Prefer live match via `find_matching_click_window`, else recorded local `rect`
     - Return `None` if maximized or coverage ≥ `max_coverage`
3. **`src/recorder/capture.py`** — capture `click_window` at **mouse-down**
   (`_click_window_payload_at`), store on queued event (not post-settle)
4. **`src/recorder/compile_tool_calls.py` / `cua_mcp/tools.py`** — pass
   `click_window` into `move_mouse`

### Flyout / overlay rule (do not skip)

Wrong ROI recreates the original pollution (taskbar landmarks, Explorer instead of
Start /「快顯主機」).

| Click target | `resolve_click_window` must return | ROI for that event |
|-------------|-------------------------------------|--------------------|
| Taskbar / tray | Taskbar (or tray) window | Thin taskbar rect |
| Start / search / shell flyout | The **flyout / popup** HWND under the point | Flyout rect — not taskbar, not Explorer |
| Click that *opens* a flyout | Press-time window under the press (e.g. taskbar button) | That press-time rect |
| Normal app window | Top-level (or tight) window under point | Window rect |

Rules:

- Prefer **press-time** window; do not replace with a window that appears under the
  same point after open/settle (Start → Explorer).
- For a click *inside* an already-open flyout, ROI is the **flyout**, not the owner.
- Prefer stored `event.click_window` over live re-resolve during analysis when the
  click opened/closed another window under the same point.

### Landmark hard gate — **explicitly deferred**

Nearby landmarks restricted to the same click window was discussed in the same
investigation but is **out of scope for this ROI-enhance pass**. Track separately;
do not half-wire it into Phase 0 without its own file list + tests.

---

## File-by-file changes (Phase 1+)

### 1. `cua_mcp/yolo_onnx.py` — `run_yolo_onnx_end2end`

Add:

```python
enhance_roi: tuple[int, int, int, int] | None = None,
enhance_roi_pad: int = 16,
```

Forward both into:

- `refine_overlapping_text_with_crops(..., enhance_roi=..., enhance_roi_pad=...)`
- `cut_multiline_text_boxes(..., enhance_roi=..., enhance_roi_pad=...)`

Docstring: first YOLO stays full-frame; Stage 3/4 limited to ROI.

### 2. `cua_mcp/yolo_divide_conquer.py`

**`find_text_overlap_refine_clusters`**

- Add `enhance_roi`, `enhance_roi_pad`.
- After building clusters, if `enhance_roi is not None`, keep only clusters where
  `_cluster_intersects_enhance_roi` is true (any member center in padded ROI).

Helpers:

```python
def _xyxy_center_in_xywh(box, rect, *, pad=0) -> bool: ...
def _cluster_intersects_enhance_roi(xyxy, cluster, roi, *, pad=0) -> bool: ...
```

**`refine_overlapping_text_with_crops`**

- Add same params; pass through to `find_text_overlap_refine_clusters`.

(Optional consistency: Ultralytics helpers can take the same params later;
recording/replay production path uses ONNX.)

### 3. `cua_mcp/text_line_cut.py` — `cut_multiline_text_boxes`

Add `enhance_roi`, `enhance_roi_pad`.

Before attempting a cut on a tall text box, require center in padded ROI; otherwise
pass the box through unchanged.

### 4. `cua_mcp/select_mouse_target.py`

**`_bbox_center_in_ocr_roi(bbox, ocr_roi, pad=...)`** — thin wrapper over
`point_in_rect_xywh` (or equivalent).

**`_detect_mouse_targets_from_bgr`**

- Params: `ocr_roi=None`, `ocr_roi_pad=16`.
- Call `run_yolo_onnx_end2end(..., enhance_roi=ocr_roi, enhance_roi_pad=ocr_roi_pad)`.
- `refine_inputs`:
  - If `ocr_roi is None`: refine all inputs (current behavior).
  - Else: pass only in-ROI inputs and in-ROI `horizontal_scrollbar_boxes` into
    `merge_yolo_inputs_with_line_rectangles`; do not keep out-ROI inputs as
    passthrough candidates (drop them at admission).
- When `ocr_roi` set: admit text / element / input / scrollbar only if center in
  padded ROI; OCR only those admitted OCR-class boxes.
- Log skip counts when ROI is set (see Observability).

**Replay path** — `_capture_and_detect_mouse_candidates` / `_collect_monitor_detections`:

- Accept `click_window`.
- Per monitor: `ocr_roi = resolve_ocr_roi_local(click_window, image_w, image_h, monitor_offset=active_monitor_offset(i))`.
- Pass `ocr_roi_by_monitor` into collect → `_detect_mouse_targets_from_bgr(..., ocr_roi=...)`.
- Missing `click_window` → all monitors ungated.

**`move_mouse` tool** — accept `click_window` and forward it.

### 5. `src/recorder/vision_context.py` — `build_vision_context_at_point`

After resolving press-time / live `click_window_payload`:

```python
enhance_roi = (
    resolve_ocr_roi_local(
        click_window_payload,
        image_w=img_w,
        image_h=img_h,
        monitor_offset=(0, 0),  # payload already local via to_local_payload(offset)
    )
    if click_window_payload is not None
    else None
)
all_detections = _detect_mouse_targets_from_bgr(
    bgr, coord_offset=offset, ocr_roi=enhance_roi,
)
```

Prefer **stored press-time** `event.click_window` over live `resolve_click_window`
after the click opened/closed another window under the same point.

**Do not** gate `build_vision_context_full_image` (intentional full-frame scan).

### 6. `src/recorder/window_snapshot.py` — `resolve_ocr_roi_local`

Implement as described under Phase 0 (including flyout / taskbar rules).

---

## Tests to restore / add

| Test | Assert |
|------|--------|
| `find_text_overlap_refine_clusters` + `enhance_roi` | Outside-ROI clusters dropped; cluster kept if any member in ROI |
| Cluster sibling outside ROI after refine | Sibling not admitted as OCR/candidate |
| `run_yolo_onnx_end2end` | Forwards `enhance_roi` / pad to Stage 3 and Stage 4 spies |
| `cut_multiline_text_boxes` + ROI | Outside ROI → `cut==0`; in ROI → still cuts |
| `_detect_mouse_targets_from_bgr` + `ocr_roi` | Passes `enhance_roi` into YOLO; out-of-ROI text never becomes a candidate |
| Input refine + `ocr_roi` | `merge_yolo_inputs_with_line_rectangles` only called with in-ROI inputs (+ in-ROI horizontal scrollbars) |
| `build_vision_context` + click_window | Passes `ocr_roi=(…)` from small window |
| Maximized click_window | `ocr_roi is None` |
| Missing `click_window` | `ocr_roi is None` (legacy ungated) |
| `resolve_ocr_roi_local` coverage | Maximized / large coverage → `None` |
| Taskbar / flyout ROI | Taskbar click → thin ROI; flyout click → flyout rect (not owner/Explorer) |
| Multi-monitor `monitor_offset` | ROI only on the monitor that contains the window |

Suggested pytest filter when re-running:

```text
tests/test_yolo_overlap_refine.py
tests/test_yolo_text_line_cut.py
tests/test_select_mouse_target.py::test_detect_ocr_roi*
tests/test_select_mouse_target.py::test_detect_refine_inputs_only_inside_ocr_roi
tests/test_recorder_vision_context.py::test_build_vision_passes_click_window_as_ocr_roi
tests/test_recorder_vision_context.py::test_build_vision_skips_ocr_roi_when_click_window_maximized
tests/test_recorder_window_snapshot.py::test_resolve_ocr_roi*
```

---

## Effective in recording and replay?

| Path | Wired? |
|------|--------|
| Recording `build_vision_context_at_point` | Yes → `ocr_roi` |
| Replay `move_mouse` + `click_window` | Yes → per-monitor `resolve_ocr_roi_local` |
| Full-image recording helper | No (by design) |
| Maximized / huge window / missing `click_window` | No-op (`None` ROI) |

---

## Separate bug — deferred click drop (mostly done)

### Missing Excel click after「篩選」(`recording_20260922_071138_595474`)

- Event 3: click「篩選」→ opens「快顯主機」.
- Event 4: close Explorer; toolbar already shows Excel filter + `.xlsx` list.
- **No event** for choosing Excel in the flyout (`purged=[]`, only 4 events).

**Cause:** left clicks deferred ~0.35s for double-click detection; a second press at
a different location cancelled the timer and overwrote `_pending_click_coords`
without emitting the first click.

**Status:** `capture.py` already has `_flush_superseded_pending_left_gesture`
(flush pending before superseding with a far press). **Do not re-implement.**

**Follow-up:** add/confirm unit test — pending click A → press B far away within
0.35s → both events recorded.

---

## Implementation order (recommended)

1. **Phase 0:** `click_window` capture + payload + `resolve_ocr_roi_local`
   (taskbar / flyout rules). Landmark hard gate stays deferred.
2. Gate Stage 3/4 APIs (`yolo_divide_conquer`, `text_line_cut`, `yolo_onnx`).
3. Wire `select_mouse_target` (`ocr_roi` → enhance + OCR + input refine + admission
   filter + logging).
4. Wire recording `vision_context` + replay `click_window` path.
5. Tests above (including flyout, multi-monitor, legacy missing `click_window`).
6. Verify deferred-click flush + add unit test only if missing.
