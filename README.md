# Computer Use Agent MVP

This project implements a runnable MVP of a computer-use agent with a single in-process coordinator and three modules:

- `Eye` module: captures screenshots from the selected monitor.
- `Brain` module: reasons on events and decides actions.
- `Hand` module: executes desktop actions and records results.

`main.py` acts as the runtime entrypoint and runs the coordinator loop directly (no subprocess server topology).

## Requirements

- Python 3.11+
- Ollama installed and running
- Gemma model available in Ollama (default: `gemma4:e2b`)
- Desktop environment that allows screenshot and automation access

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure

1. Copy `.env.example` to `.env` and edit as needed.
2. Optionally adjust defaults in `constants.json`.

Key values:

- `eye_vlm`, `brain_lm`
- model/runtime settings

## Run

```bash
python main.py
```

This creates a run session under `runs/<task_slug>_<timestamp>/` with:

- `eye/` screenshots
- `thinking/` decision records
- `storage/` user-storage files
- `hand.csv` action history
- `long_term_memory.txt` long-term memory (capped)
- `storage.json` storage index
- `run.log` debug log

## Tests

```bash
pytest
```

## Troubleshooting

- If model calls fail, verify Ollama is running at `OLLAMA_HOST`.
- If desktop actions fail on Windows, run with proper UI permissions and avoid elevated target apps unless this process is elevated too.
- If screenshots are black/empty, verify capture permissions and active display availability.
