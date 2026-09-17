# YOLO Divide-and-Conquer, Overlap Refine & Line Cut

Logic specification for `yolo_divide_conquer.py` (Stage 3 + optional flat 2×2 helpers) and
`text_line_cut.py` (Stage 4).

Used by:

- **Production ONNX** — `yolo_onnx.run_yolo_onnx_end2end`
  - Stages 1–2: recursive overlapping 2×2 **quadtree** inside `yolo_onnx.py`
    (overlap **0.125**, cross-tile NMS IoU **0.5**, depth/size limits)
  - Stage 3: `refine_overlapping_text_with_crops` from this module
    (`overlap_refine=True` by default; crop predict uses the recursive path
    with refine disabled)
  - Stage 4: `cut_multiline_text_boxes` from `text_line_cut.py`
    (`line_cut=True` by default; model-free, runs even when Stage 3 changed nothing)
- **Labeler / Ultralytics helpers** (flat non-recursive 2×2):
  - `predict_ultralytics_with_2x2_fallback`
  - `predict_onnx_with_2x2_fallback`  
    Do **not** wrap `run_yolo_onnx_end2end` with these helpers (nests tiling).

---

## Pipeline overview

### Production (`run_yolo_onnx_end2end`)

```
Input image (BGR)
        │
        ▼
┌─────────────────────────────┐
│ 1–2. Full-frame YOLO        │
│      + recursive quadtree   │
│      when hit max_det       │
│      + merge / expand /     │
│        small-text→element   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. Overlap refine           │
│    (text class only;        │
│     on by default)          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. Line cut                 │
│    (text class only;        │
│     on by default;          │
│     no model call)          │
│    + re-apply small-text→   │
│      element if changed     │
└─────────────┬───────────────┘
              │
              ▼
     (xyxy, scores, cls)
```

### Flat helpers (`predict_*_with_2x2_fallback`)

```
Input image (BGR)
        │
        ▼
┌───────────────────────┐
│ 1. Full-frame YOLO    │
└───────────┬───────────┘
            │
            ├── count < max_det ──────────────────────────┐
            │                                             │
            └── count ≥ max_det                           │
                    │                                     │
                    ▼                                     │
            ┌───────────────────┐                         │
            │ 2. Flat 2×2 tiles │                         │
            │    + class NMS    │                         │
            └─────────┬─────────┘                         │
                      │                                   │
                      └────────────────┬──────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │ 3. Overlap refine       │
                         │    (text class only;    │
                         │     on by default)      │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │ 4. Line cut             │
                         │    (text class only;    │
                         │     on by default)      │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                              DetectArrays result
```

Result type for flat helpers: `DetectArrays`

| Field | Meaning |
|-------|---------|
| `xyxy` | Boxes in source-image pixels |
| `scores` | Confidences |
| `cls` | Class ids |
| `used_tiles` | Whether 2×2 fallback ran |
| `full_count` | Detection count from the full-frame pass |
| `used_overlap_refine` | Whether any cluster was replaced |
| `refine_crop_count` | Number of clusters accepted for replacement |
| `line_cut_count` | Number of text boxes split into lines by Stage 4 |

Production `run_yolo_onnx_end2end` returns `(xyxy, scores, cls)` only; refine
activity is logged as `overlap-refine×N` and line cuts as `line-cut×N`.

---

## Stage 1 — Full-frame YOLO

1. Load image as BGR (`load_bgr`).
2. Run the model once on the full image with the caller’s `conf` / kwargs.
3. Record `full_count = N` detections.

If `full_count < max_det` (default **300**), skip Stage 2.

---

## Stage 2 — 2×2 tile fallback

Triggered only when Stage 1 hits the detection cap (`N ≥ max_det`).

### Production (quadtree in `yolo_onnx.py`)

- Overlapping 2×2 ROIs with overlap fraction **0.125**
- Recurse per tile while still at cap (max depth **3**, half-side ≥ **320**)
- Cross-tile class-aware NMS IoU **0.5**, then same-class merge / input expand /
  small-text→element postprocess

### Flat helpers (`split_bgr_2x2`)

- Cut the image into four overlapping quadrants: TL, TR, BL, BR.
- Overlap fraction default: **0.15** of half-width / half-height
  (each tile extends 15% past the center into the neighboring half).
- Each tile returns `(tile_bgr, origin_x, origin_y)`.

### Predict & merge (flat helpers)

1. Run YOLO on each tile.
2. Offset tile boxes by `(origin_x, origin_y)` into full-image coordinates.
3. Concatenate all tile detections.
4. Class-aware NMS with `merge_iou` default **0.20**:
   - Same class only
   - Lower-score box suppressed when IoU exceeds threshold

Set `used_tiles = True`.

---

## Stage 3 — Overlap refine (text)

Enabled by default (`overlap_refine=True`).  
Operates only on **text** class id **0** (`OVERLAP_REFINE_TEXT_CLASS_ID`).

Goal: when tight vertical gaps cause mixed / duplicate / tall merged text boxes, crop the hard region, re-run YOLO at higher effective resolution, and keep the result only if it is clearly better than keeping or simply deduping the originals.

### 3.1 Cluster discovery (`find_text_overlap_refine_clusters`)

Consider text boxes only.

#### Link rules (`_text_boxes_should_link`)

Two text boxes are linked if **any** of:

1. **IoU link**: pairwise IoU `> 0.08`
2. **Crossing link**: boxes intersect, and neither fully contains the other
3. **Dense vertical stack** (optional; **off by default**):
   - `max_v_gap_frac < 0` disables this path
   - When enabled: x-overlap fraction ≥ 0.45 **and** vertical gap ≤ `max_v_gap_frac × min(h1, h2)`

Why dense-stack linking is off: it pulled clean neighbors into huge crops; re-detect then tended to merge lines again.

#### Cluster formation

1. Build pairwise links among text boxes.
2. Union-find → connected components.
3. Keep components with **≥ 2** boxes.
4. Also keep **singleton** text boxes that look multi-line:

   ```
   height ≥ 1.85 × median_text_height
   ```

5. Sort clusters largest-first (cap applied later: max **24**).

### 3.2 Per-cluster processing (`refine_overlapping_text_with_crops`)

For each cluster (skip indices already replaced by an earlier cluster):

#### Skip conditions

- Empty after prior replacements
- Singleton that is **not** tall (`height < 1.85 × median`)
- Padded crop smaller than `24×24`

#### Crop geometry

1. **Union** = axis-aligned bbox of all live boxes in the cluster.
2. **Pad** the union to form the crop:

   | Pad | Rule |
   |-----|------|
   | X | `max(min_pad_x_px=36, 0.75 × median_text_width, union_w × 0.15)` |
   | Y | `max(min_pad_y_px=8, 0.5 × median_text_height, union_h × 0.15)` |

   Extra X pad matters: fragment unions are often too narrow; narrow crops miss dropdown lines.

3. Clamp crop to image bounds.

#### Re-predict (`_predict_roi_with_optional_strips`)

- If crop height ≤ `~2.6 × median_line_height × 1.35`: **one-shot** predict on the crop.
- Else: split into overlapping **horizontal strips**:
  - strip height ≈ `max(48, 2.6 × median_line_height)`
  - strip overlap ≈ 45% of strip height
  - predict each strip, offset to full-image coords, class-aware NMS merge

#### Filter crop outputs

1. Keep **text** boxes whose **center** lies inside a **soft union**
   (original union expanded by ~15% of its w/h, still clipped to the crop).
   - Avoids neighboring labels pulled in by generous X-pad from replacing the cluster.
2. Drop text boxes taller than `1.85 × median_line_height`
   (reject multi-line merges from the crop pass).

### 3.3 Choose replacement (`_pick_best_cluster_boxes`)

Build up to three candidates:

| Name | Source |
|------|--------|
| `old` | Original cluster boxes |
| `dedup` | Class-aware NMS on `old` with IoU ≥ `max(0.20, 0.35)` |
| `crop` | Strip/crop re-detect — **only admitted** if unique-line count ≥ `old`’s unique-line count |

#### Unique-line estimate (`_approx_unique_box_count`)

NMS with equal scores at IoU **0.35** → approximate number of distinct lines.

#### Overlap-pair count (`_overlap_pair_count`)

Number of pairs with IoU `> 0.08`.

#### Ranking key (`_cluster_candidate_key`)

Minimize, in order:

1. Overlap-pair count (fewer is better)
2. `-unique_line_count` (more unique lines is better)
3. `len(boxes) - unique_lines` (fewer raw duplicates is better)

Empty crop scores as worst.

#### Decision

- Pick the best candidate by ranking key.
- If best is `old`, or not strictly better than `old` → **no change** for this cluster.
- Otherwise replace the cluster with the winner.

#### Tall-singleton special case

If pick returns `None` but the cluster is a single tall box, still accept `crop` when:

- crop has ≥ 2 boxes
- unique-line count ≥ 2
- crop overlap-pair count == 0

### 3.4 Merge back into full detection set

1. Drop all replaced cluster indices.
2. Keep every non-replaced detection (all classes).
3. Append accepted replacements.
4. Final class-aware NMS (`merge_iou`, default 0.20).
5. Set `used_overlap_refine = (accepted > 0)`, `refine_crop_count = accepted`.

---

## Stage 4 — Line cut (text)

Enabled by default (`line_cut=True`), implemented in `text_line_cut.py`.
Operates only on **text** class id **0**. No model call — pure NumPy / OpenCV.

Goal: Stage 3 can only *swap in* boxes the detector re-found, so a tall box whose lines the model keeps merging survives untouched (crop output that is still multi-line is dropped by `_drop_tall_text_boxes`, and `_pick_best_cluster_boxes` then returns `None`). Stage 4 adds the missing **split**: cut a tall box at the valleys of its row ink profile.

It runs **whether or not Stage 3 accepted anything** — a cluster Stage 3 declined is exactly the case that still needs splitting.

### 4.1 Ink measurement (polarity-free)

Per box crop, ink is deviation from a **per-row** background estimate, not an absolute intensity:

1. Grayscale the crop.
2. `row_bg = median(gray, axis=1)` — one background value per row.
3. `dev = |gray - row_bg|`.
4. `thr = max(LINE_CUT_MIN_DEV=12, 0.22 × (p98(gray) - p2(gray)))`.
5. `mask = dev > thr`; rows where the mask ends up in the majority are inverted so ink is always the minority class in its row.

Consequences:

- **White-on-black and black-on-white behave identically** — only the magnitude of the deviation matters, never its sign.
- Colored / selected rows, alternating table striping, and soft gradients stay neutral, because each row is compared against its own background.
- A full-width separator rule is uniform within its own row, so its row median equals the rule color and the rule reads as a **valley**, not a peak. No special-case peak logic is needed.

### 4.2 Profile & valleys

- `profile[row] = fraction of ink pixels in that row` — text rows are peaks, blank rows are valleys.
- A row is a **gap** when `profile ≤ max(LINE_CUT_ABS_GAP=0.02, 0.18 × p95(smoothed profile))`.
  The peak reference comes from a 3-row smoothed copy (one spiky row cannot raise the bar), but the test itself runs on the **raw** profile: smoothing the decision would widen every band by a row and swallow the 2–3px valleys of tightly-leaded lines.
  The absolute floor (2% of box width) is what absorbs single stray pixels in a blank row.
- **Bands** are the runs of non-gap rows. A valley splits two bands when it spans at least `max(2, 0.10 × median_line_h)` rows **or** contains a fully blank row (`≤ LINE_CUT_ABS_GAP`).
  The blank-row rule is what makes compact lists work: a 12px-pitch dropdown separates its items by exactly **one** empty row, and the row-count floor alone merges all of them back into a single band. Rows that merely dip below the soft gap threshold still need the full span, so a thin spot inside a glyph row cannot split a line.
  Leading / trailing gaps are therefore dropped, each band is already trimmed to its own ink extent, and splitting between bands is equivalent to cutting at the center of the separating valley.
- Bands thinner than `LINE_CUT_SLIVER_H_FRAC=0.5` × the box's own median band height are **dropped** as slivers (a highlighted / selected row leaves a 1px run at the edge of its background block). Dropping one sliver is better than letting it fail an otherwise clean split.
- Each band's X extent is re-trimmed by a column projection inside that band — this only ever shrinks a box, and also tightens over-wide boxes inherited from a Stage 3 cluster union.

### 4.3 Candidate gate

A detection is attempted only when **all** hold:

- class is **text**
- `height ≥ 1.85 × median_text_height` (frame median; same factor as `OVERLAP_REFINE_TALL_HEIGHT_FACTOR`, so Stage 3 and Stage 4 agree on "suspiciously multi-line")
- `height ≥ 16 px` and `width ≥ 8 px`
- fewer than `LINE_CUT_MAX_BOXES=64` boxes attempted so far

Everything else passes through untouched and in original order.

### 4.4 Validation (all-or-nothing)

Bands are judged **against each other**, not against the frame median. Stacked lines of one control share a pitch, whatever that pitch is; a compact list whose font is smaller than the frame's median text box is still perfectly valid. Anchoring the band-height window and an expected band count to the frame median rejected exactly the dense dropdowns this stage exists for.

A split is accepted only when **every** check passes; otherwise the original box is kept, so the worst case equals no Stage 4 at all:

- at least **2** bands (after slivers are dropped)
- median band height ≥ `LINE_CUT_MIN_BAND_H_PX=5` — no real text line is thinner, and this is what stops blank rows *inside* glyphs (二, 三) from reading as line separators
- tallest band ≤ `LINE_CUT_BAND_UNIFORM_MAX_FRAC=1.8` × the box's median band height (uniform pitch)
- thinnest band ≥ `0.5` × the box's median band height
- tallest band ≤ `LINE_CUT_BAND_MAX_H_FRAC=1.65` × `median_line_h` — the only frame-median tie, so an emitted band can never itself be multi-line
- bands retain at least **60%** of the box's total ink

### 4.5 Output

1. The parent box is dropped; its bands are appended as **text** boxes inheriting the parent score.
2. Bands are **disjoint in Y with no vertical padding**. This is a hard invariant: the OCR path unions any overlapping boxes transitively (`geometry.merge_overlapping_boxes`), which would re-merge padded bands into the tall box that was just split. Vertical margin for the OCR crop is added later by `_expand_box`, after that merge.
3. Production logs `line-cut×N` and re-applies the small-text → element dual-stream; flat helpers report `line_cut_count`.

Stage 4 must stay **after** `merge_touching_same_class_xyxy` — that merge would fuse the bands straight back together.

### 4.6 Known limitations

- Two-column content inside one box with offset baselines has no clean valley, as does a box mixing an icon with text. The uniformity checks reject both and keep the original box.
- Genuinely touching glyph rows (zero blank rows between lines) cannot be split by projection at all.
- The frame median still gates *candidacy*, so a two-line box in a font much smaller than the frame median can fall under `1.85 × median_text_height` and never be attempted.

---

## Key defaults

| Constant | Default | Role |
|----------|---------|------|
| `YOLO_MAX_DET_DEFAULT` | 300 | Cap that triggers 2×2 tiles |
| `TILE_OVERLAP_FRAC_DEFAULT` | 0.15 | Tile overlap |
| `TILE_MERGE_IOU_DEFAULT` | 0.20 | Tile / final NMS IoU |
| `OVERLAP_REFINE_DEFAULT` | `True` | Enable Stage 3 |
| `OVERLAP_REFINE_TEXT_CLASS_ID` | 0 | Text class |
| `OVERLAP_REFINE_LINK_IOU` | 0.08 | Cluster link IoU |
| `OVERLAP_REFINE_LINK_CROSSING` | `True` | Link crossing boxes |
| `OVERLAP_REFINE_MAX_V_GAP_FRAC` | -1.0 | Dense-stack link off |
| `OVERLAP_REFINE_TALL_HEIGHT_FACTOR` | 1.85 | Tall / multi-line threshold |
| `OVERLAP_REFINE_MIN_PAD_X_PX` | 36 | Minimum crop X pad |
| `OVERLAP_REFINE_STRIP_HEIGHT_FACTOR` | 2.6 | When to use strips |
| `OVERLAP_REFINE_STRIP_OVERLAP_FRAC` | 0.45 | Strip overlap |
| `OVERLAP_REFINE_UNIQUE_LINE_IOU` | 0.35 | Unique-line / dedupe IoU |
| `OVERLAP_REFINE_MAX_CROPS` | 24 | Max clusters processed |
| `LINE_CUT_DEFAULT` | `True` | Enable Stage 4 |
| `LINE_CUT_TALL_HEIGHT_FACTOR` | 1.85 | Candidate gate vs median text height |
| `LINE_CUT_MIN_DEV` | 12.0 | Minimum ink deviation from row background |
| `LINE_CUT_DEV_RANGE_FRAC` | 0.22 | Ink threshold as fraction of crop range |
| `LINE_CUT_SMOOTH_ROWS` | 3 | Rows smoothed for the peak reference only |
| `LINE_CUT_ABS_GAP` | 0.02 | Absolute gap-row ink floor |
| `LINE_CUT_GAP_FRAC` | 0.18 | Gap threshold as fraction of profile p95 |
| `LINE_CUT_MIN_GAP_FRAC` | 0.10 | Soft-valley rows needed to split (× line h) |
| `LINE_CUT_SLIVER_H_FRAC` | 0.5 | Sliver-band drop / min band vs box median |
| `LINE_CUT_BAND_UNIFORM_MAX_FRAC` | 1.8 | Max band height vs box median band |
| `LINE_CUT_MIN_BAND_H_PX` | 5 | Absolute minimum median band height |
| `LINE_CUT_BAND_MAX_H_FRAC` | 1.65 | Max band height vs **frame** median line h |
| `LINE_CUT_MIN_COVERAGE_FRAC` | 0.6 | Minimum retained ink fraction |
| `LINE_CUT_MAX_BOXES` | 64 | Max boxes attempted per frame |

---

## App-side follow-up (labeler)

Where a labeler UI consumes `DetectArrays` directly, it still:

1. Maps boxes into the image (and optional selection ROI offset).
2. Drops same-class predictions with IoU `> 0.90`, keeping higher confidence.
3. Skips geometry duplicates already on the canvas.
4. Status may include `2×2 tiles (...)`, `overlap-refine×N`, and/or `line-cut×N`.

That 90% dedupe is **separate** from the refine-stage NMS.

---

## Design intent

1. Full frame first — cheapest path when the image is not detection-capped.
2. Tiles only when the model hits `max_det` — recover missed objects in dense UIs
   (production: recursive quadtree; flat helpers: one-level 2×2).
3. Overlap refine only for conflicted / tall **text**:
   - Crop + optional strips to increase effective resolution on hard regions.
   - Always compete against simple cluster NMS dedupe.
   - Never keep a crop that loses unique line coverage vs the original cluster.
4. Prefer not changing clean detections; only replace when the ranking key improves.
5. Line cut last, for what the detector cannot fix:
   - Deterministic and model-free, so it can split a box the model insists on merging.
   - Polarity-free per-row ink, so themes / selected rows / striping do not matter.
   - All-or-nothing per box, so a bad profile degrades to today's behavior.
6. Production wires Stages 3–4 once at the end of `run_yolo_onnx_end2end` so OCR /
   mouse-target callers get both without nesting flat tile wrappers.
