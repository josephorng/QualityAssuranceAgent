# Recording pipeline

How a live session becomes durable `events/`, `yolo_ocr/`, and `analysis/` artifacts.

For low-level capture threads (before-shot, settle UIA, idle hover), see [`RECORDING_LOOPS.md`](../RECORDING_LOOPS.md).

Interactive overview (open in a browser; switch Replay / Recording): [`uia-replay-path.html`](uia-replay-path.html).

## Overview

```mermaid
flowchart TD
  A[Hub: start RecordingSession] --> B[Live capture]
  B --> C[events/event_NNN.json<br/>screenshot + uia_snapshot + EFP]
  C --> D[Vision prefetch<br/>yolo_ocr/event_NNN.json]
  A2[Hub: begin_stop / finalize_stop] --> E[session.json + final_after.jpeg]
  E --> F[analyze_recording_session]
  F --> G{recording_uia_skip_reason?}
  G -->|None → UIA sufficient| H["instruction （UIA）"]
  G -->|reason → YOLO| I["instruction （YOLO+OCR）"]
  H --> J[analysis/event_NNN.json<br/>tool_calls + optional yolo_ocr_fallback twin]
  I --> K[analysis/event_NNN.json<br/>tool_calls]
  J --> L[Ready for cache replay]
  K --> L
```

## Stages

| # | Stage | Entry | Main outputs |
|---|--------|--------|--------------|
| 1 | Start session | `app_main_hub` → `RecordingSession.start` | `events/`, `screenshots/`, `yolo_ocr/` |
| 2 | Live capture | mouse/keyboard → worker queue | `events/event_NNN.json` |
| 3 | Vision prefetch | `VisionPrefetchWorker` | `yolo_ocr/event_NNN.json` (often) |
| 4 | Stop / finalize | `begin_stop` → `finalize_stop` | `session.json`, `final_after.jpeg` |
| 5 | Analyze | `analyze_recording_session` | `analysis/event_NNN.json`, `report.json` |

---

## 1. Session start / stop

```mermaid
flowchart LR
  Start[start] --> Folders[Create run folders]
  Folders --> Worker[screen-recorder-worker]
  Worker --> Seed[seed_settle_windows]
  Seed --> Loops[Before-shot + pre-click loops]
  Loops --> Listeners[pynput mouse + keyboard]
  Listeners --> Accept[_accepting_input = True]
  Accept --> Prefetch[VisionPrefetchWorker]

  Stop1[begin_stop] --> Flush[Stop listeners / flush pending]
  Flush --> Stop2[finalize_stop]
  Stop2 --> Join[Join worker + window-step]
  Join --> Manifest[session.json]
  Manifest --> Drain[vision_prefetch.drain_and_stop]
  Drain --> Analyze{analyze=True?}
  Analyze -->|yes| RunAnalyze[analyze_screen_recording]
```

| Role | Path |
|------|------|
| Hub | `app_main_hub.py` → `_start_recording` / `_stop_recording` |
| Capture | `src/recorder/capture.py` → `RecordingSession` |
| Analyze entry | `main.py` → `analyze_screen_recording` → `orchestrator.analyze_recording_session` |

---

## 2. Event capture

```mermaid
flowchart TD
  Mouse[Mouse / keyboard hooks] --> Q[Worker queue]
  Q --> Persist[_persist_queued_event]
  Persist --> JSON[events/event_NNN.json]
  Persist --> Shot[JPEG screenshot]
  Persist --> Notify[Notify hub → vision prefetch]
  Persist --> WinStep[window-step-N thread]
  WinStep --> Patch[Late-patch window_change / settle UIA / signals]
```

A pointer event typically stores:

- `index`, `kind`, `cursor_xy`, button / modifiers / click count
- `screenshot_path`, `monitor_index`, `monitor_offset`, `click_window`
- `uia_snapshot` — press-time FindAll when settle cache hwnd + generation match
- `element_from_point` — Name / Value / bbox / control_type (worker probe, not the mouse hook)
- later patches: `window_change`, `focused_element`, drag `end_*`

Key APIs: `RecordingSession._persist_queued_event`, `models.write_event_json`.

---

## 3. Analyze session

```mermaid
flowchart TD
  A[analyze_recording_session] --> C[coalesce events]
  C --> V[vision: prepare_event_vision parallel]
  V --> T[text_choice LLM refine]
  T --> I[instruction: analyze_event_to_cache]
  I --> W[assemble: _write_event_analysis]
  W --> R[yolo_ocr_retry failed pointers]
  R --> Cache[ensure_recording_tool_cache]
  Cache --> Report[report.json + HTML]
```

### Per-event vision (`prepare_event_vision`)

1. Caption-button shortcut → skip vision  
2. Else cache hit (`try_rebuild_vision_from_cache`)  
3. Else `build_vision_context` → YOLO+OCR → `build_vision_context_at_point`

### Per-event instruction (`analyze_event_to_cache`)

Deterministic first (text / keys / window / caption / click|drag|scroll from vision).  
LLM only if those fail and pointer vision did not soft-fail.  
Prefix via `_prefix_instruction_vision_mode`: `（UIA）` or `（YOLO+OCR）`.

---

## 4. UIA vs YOLO at analyze time

Core decision in `_detect_recording_mouse_targets` (`vision_context.py`):

```mermaid
flowchart TD
  Load[Load UIA from uia_snapshot] --> EFP[Inject EFP peer if usable]
  EFP --> YOLO[Always run YOLO+OCR on screenshot]
  YOLO --> Skip{recording_uia_skip_reason is None?}
  Skip -->|yes UIA sufficient| UIA["vision_path = uia<br/>primary = UIA (+ sparse YOLO nearby peers)<br/>store YOLO as yolo_ocr_fallback_detections"]
  Skip -->|no| YO["vision_path = yolo_ocr<br/>primary = YOLO (+ EFP peer)"]
  UIA --> Weak{_maybe_demote_uia_path_for_weak_efp_ocr_disagree?}
  Weak -->|disagree| YO
  Weak -->|ok| KeepUIA[Keep UIA path]
```

UIA is sufficient when there is a usable UIA text / textless input / scrollbar under the click, or a uniquely landmark-selected UIA hub, or a primary EFP-labeled leaf. Otherwise the path is YOLO (`vision_path_reason` = skip reason).

Chrome icons such as `展開節點` are in `_CONTROL_CHROME_ICON_IDS` and can rank as the primary click target over overlapping OCR text.

Disk: `yolo_ocr/event_NNN.json` (candidates, `vision_path`, optional nested fallback, debug layers).

---

## 5. Analysis JSON + YOLO twin

`_write_event_analysis` writes `analysis/event_NNN.json`:

| Field | Meaning |
|-------|---------|
| `instruction` | Hub script line (`（UIA）` or `（YOLO+OCR）`) |
| `tool_calls` | From `compile_tool_calls` (`move_mouse` + `click`, etc.) |
| `yolo_ocr_fallback` | Twin `{instruction, tool_calls}` when UIA primary kept YOLO detections |
| `vision` | Thin: `used_vision`, `candidate_text` (full candidates stay in `yolo_ocr/`) |
| `window_verify` | Foreground / appear-disappear checks |
| `landmark_geometry` | Inside `move_mouse` args when nearby / offset / scroll-hover |

`move_mouse` args also carry `hub_source` (`findall` \| `uia_efp`), recorded EFP `name`/`value`, and `click_window`.

Twin build: `build_yolo_ocr_fallback_from_vision` when `vision["yolo_ocr_fallback_vision"]` exists.

> **Note:** rewriting a YOLO icon step into a UIA hub (`write_uia_replay_repair`) happens at **cache replay**, not inside `analyze_recording_session`. Analyze prepares the twin; replay may promote the hub. See [`replay-flow.md`](replay-flow.md).

---

## 6. Example: tree expand glyph

```mermaid
sequenceDiagram
  participant User
  participant Capture
  participant Vision
  participant Analyze
  participant Analysis as analysis/event_NNN.json

  User->>Capture: Click 展開節點 glyph
  Capture->>Capture: events/event_NNN.json + screenshot + EFP
  Capture->>Vision: prefetch / analyze YOLO+OCR
  Note over Vision: Icon chinese_id=展開節點<br/>UIA often insufficient at glyph
  Vision->>Analyze: vision_path=yolo_ocr
  Analyze->>Analysis: （YOLO+OCR）「展開節點」圖示 tool_calls
  Note over Analysis: Later replay may repair to<br/>（UIA）「龜速的網域 (3)|1」文字<br/>+ yolo_ocr_fallback twin for the icon
```

---

## Key files

| Concern | File | Functions |
|---------|------|-----------|
| Start/stop | `src/recorder/capture.py` | `RecordingSession.start`, `begin_stop`, `finalize_stop` |
| Hub | `app_main_hub.py` | `_start_recording`, `_stop_recording` |
| Analyze | `src/recorder/orchestrator.py` | `analyze_recording_session`, `prepare_event_vision`, `_write_event_analysis` |
| Instructions | `src/recorder/analyze.py` | `analyze_event_to_cache`, `instruction_for_click` |
| UIA vs YOLO | `src/recorder/vision_context.py` | `_detect_recording_mouse_targets`, `recording_uia_skip_reason` |
| Tool compile | `src/recorder/compile_tool_calls.py` | `compile_tool_calls`, `_move_mouse_call` |
| Twin | `src/recorder/uia_replay_repair.py` | `build_yolo_ocr_fallback_from_vision` |
| Geometry | `src/recorder/landmark_geometry.py` | `build_landmark_geometry` |

## Disk artifacts

| Path | When |
|------|------|
| `events/event_NNN.json` | Capture persist (+ window-step patches) |
| `signals/event_NNN_*.json` | Peeled from event debug |
| `screenshots/*`, `final_after.jpeg` | Capture / stop |
| `yolo_ocr/event_NNN.json` | Prefetch and/or analyze vision |
| `analysis/event_NNN.json` | Analyze assemble |
| `session.json`, `report.json`, `record.log` | Stop / analyze |
