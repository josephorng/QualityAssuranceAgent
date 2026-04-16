# Computer Use Agent (Scaffold)

Local-first Python scaffold of an Eye/Brain/Hand computer-use agent based on `architecture.md`.

## What is implemented

- Task-oriented run session initialization under `runs/<timestamp-task>/`.
- Required runtime artifacts:
  - `eye/` screenshots
  - `thinking/` decision logs per screenshot
  - `storage/` saved artifacts
  - `hand.csv`, `brain.txt`, `storage.json`, and run `.log`
- Eye pipeline:
  - periodic screenshots
  - similarity filtering against previous image
  - first image always processed
- Brain pipeline:
  - screenshot description adapter (mock)
  - decision engine with interaction vs retrieval tool branching
  - long-term memory file with max-length compaction
- Hand pipeline:
  - executes Brain interaction commands through MCP tool registry
  - logs every action to `hand.csv`
- MCP tool scaffold:
  - interaction tools (`click`, `type_text`, `key_press`)
  - retrieval tools (`get_running_programs`, `mock_ocr`)

## Install

```bash
pip install -e .
```

## Run

Dry run (default, no real mouse/keyboard actions):

```bash
computer-use-agent "open a browser and search weather"
```

Live input mode (real keyboard/mouse actions):

```bash
computer-use-agent "perform a task" --live-input
```

## Smoke test

```bash
python scripts/smoke_run.py
```

## Extend next

- Replace `eye/describe.py` mock with real Gemma call.
- Replace `brain/reasoner.py` mock policy with model-driven tool planning.
- Add real retrieval implementations (OCR/YOLO/icon captioner/system process list).
- Expand MCP tools in `src/computer_use_agent/mcp/tools/`.
