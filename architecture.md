# Architecture

Computer-use agent for scripted and smart desktop automation. Runtime is a single in-process coordinator (no multi-server Eye/Brain/Hand topology).

## Stack

| Layer | Implementation |
| --- | --- |
| Entry | `main.py` → `app_main_hub.py` (CustomTkinter hub) |
| Orchestration | `src/runtime/coordinator.py` (`RuntimeCoordinator`) and smart variants |
| Modules | `src/eye`, `src/brain`, `src/hand` |
| Tools / vision | `cua_mcp/` (MCP tool façade, YOLO/OCR via Triton, mouse targeting) |
| LLM | `src/common/vllm_client.py` → OpenAI-compatible vLLM (`google/gemma-4-26B-A4B-it`) |
| Settings | Defaults in `src/common/settings.py`; overrides in `runs/agent_settings.json` |
| Prompts | `src/common/prompts.py` (keyed variants; first variant used) |

Note: the settings field `ollama_host` is the **vLLM base URL** (legacy name kept for persisted settings / `.env`).

## Run layout

Each task creates `runs/<task>_<utc_timestamp>/` with:

- `eye/` — screenshots used for decisions and verify
- `thinking/` — decision records
- `storage/` + `storage.json` — user-stored artifacts
- `hand.csv` — executed actions
- `long_term_memory.txt` — capped long-term memory
- `run.log` — debug log

## Modules

### Eye

Captures the selected monitor(s) when the coordinator needs a fresh screen state (after actions, for verify, etc.). Not a fixed-interval poller. Monitor selection comes from hub UI / `EYE_MONITOR_INDEX(S)`.

### Brain

Drives script steps (and smart mode): decide next tool call, verify baselines/outcomes against screenshots, and recover when steps fail. Uses vLLM plus tools exposed through Hand/`cua_mcp`.

### Hand

Executes MCP tools (mouse, keyboard, windows, screen read, etc.) defined under `cua_mcp/`. Returns results to the coordinator for the next Brain cycle.

## Coordinator loop

`RuntimeCoordinator` initializes Eye/Brain/Hand against shared run state, then repeatedly:

1. Optionally accept a runtime/smart command from the hub
2. `brain.process_step()` — plan/verify and request tools
3. Hand executes tools in-process
4. Stop when the script completes, the user ends the run, or a step fails

Pause/undo hooks exist for interactive runtime-command mode.

## Vision

YOLO UI detection and CRNN OCR run through Triton (`cua_mcp/vision_triton.py`), with preprocess/decode helpers in `cua_mcp/yolo_onnx.py` and `cua_mcp/read_screen_text/`. Spatial color segmentation (`cua_mcp/color_spatial_segment.py`) supports landmark-scoped targeting.
