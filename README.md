# Computer Use Agent MVP

This project implements a runnable MVP of a computer-use agent with three asynchronous services:

- `Eye` (`8001`): captures screenshots, detects meaningful changes, and emits events.
- `Brain` (`8002`): reasons on events, handles interruptions, and decides next actions.
- `Hand` (`8003`): executes desktop actions and reports results back to Brain.

`main.py` acts as the subprocess manager that starts and monitors all services.

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
- `screenshot_interval_seconds`
- `screenshot_similarity_threshold`
- ports for each service

## Run

```bash
python main.py --task "Your task description here"
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

## Integration Demo

```bash
python scripts/run_demo.py
```

## Troubleshooting

- If model calls fail, verify Ollama is running at `OLLAMA_HOST`.
- If desktop actions fail on Windows, run with proper UI permissions and avoid elevated target apps unless this process is elevated too.
- If screenshots are black/empty, verify capture permissions and active display availability.
