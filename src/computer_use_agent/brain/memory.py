from __future__ import annotations

from pathlib import Path


class BrainMemory:
    def __init__(self, memory_path: Path, max_chars: int) -> None:
        self.memory_path = memory_path
        self.max_chars = max_chars

    def append(self, entry: str) -> None:
        existing = self.read()
        updated = (existing + "\n" + entry).strip()
        if len(updated) > self.max_chars:
            updated = self._summarize(updated)
        self.memory_path.write_text(updated + "\n", encoding="utf-8")

    def read(self) -> str:
        if not self.memory_path.exists():
            return ""
        return self.memory_path.read_text(encoding="utf-8").strip()

    def _summarize(self, text: str) -> str:
        # Local-first placeholder summarizer. Replace with model adapter.
        keep = int(self.max_chars * 0.7)
        return "Summary (mock):\n" + text[-keep:]
