# Computer Use Agent

Desktop automation agent with an in-process coordinator and three modules:

- **Eye** — captures screenshots from the selected monitor(s)
- **Brain** — plans script / smart steps and checks recorded window and signal predicates
- **Hand** — executes desktop actions through MCP tools in `cua_mcp/`

## Requirements

- Python 3.11+
- A reachable **vLLM** OpenAI-compatible server (default model: `google/gemma-4-26B-A4B-it`)
- A reachable **Triton Inference Server** for YOLO + OCR vision
- Desktop environment that allows screenshot and UI automation

```bash
pip install -r requirements.txt
```

## Configure

1. Copy `.env.example` to `.env` and edit hosts as needed.
2. Prefer the hub gear dialog for LLM/vision settings (saved to `runs/agent_settings.json`).

Key settings (defaults in code; overrides in `runs/agent_settings.json` / `.env`):

- `llm_backend`, `brain_lm`, `ollama_host` (vLLM base URL; legacy key name)
- `triton_http_url`, `vision_backend`

## Run

GUI hub (primary):

```bash
python main.py
```

This opens `app_main_hub.py`. `RuntimeCoordinator` in `src/runtime/coordinator.py` runs scripted, queue, smart, and recording flows in-process: Brain plans a step, Hand runs the MCP tools, then the loop stops on completion, a failed step, or a user stop. Each run lands under `runs/<task_slug>_<timestamp>/` with `eye/`, `thinking/`, `storage/`, `hand.csv`, `long_term_memory.txt`, and `run.log`.

YOLO and CRNN OCR go through Triton (`cua_mcp/vision_triton.py`). Prompts live in `src/common/prompts.py`. The settings field `ollama_host` is the vLLM base URL.

Offline OCR tooling (not part of the packaged hub build):

```bash
python app_ocr_viewer_tk.py
python app_ocr_verify_tk.py
```

## Tests

```bash
pytest
```

Or `run_all_tests.bat` on Windows.

## Troubleshooting

- If model calls fail, verify the vLLM host in settings / `OLLAMA_HOST` (OpenAI-compatible base URL, e.g. `http://host:8000`).
- If vision fails, verify Triton at `TRITON_HTTP_URL` and that YOLO/OCR models are loaded.
- If desktop actions fail on Windows, run with UI permissions and avoid elevated target apps unless this process is elevated too.
- If screenshots are black/empty, verify capture permissions and the selected display.
