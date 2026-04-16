from __future__ import annotations

import json
from pathlib import Path

from computer_use_agent.core.session import utc_stamp


class StorageManager:
    def __init__(self, storage_dir: Path, storage_index_path: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_index_path = storage_index_path

    def save_text(self, name: str, content: str, summary: str) -> Path:
        artifact = self.storage_dir / name
        artifact.write_text(content, encoding="utf-8")
        self._append_index(name=name, summary=summary)
        return artifact

    def _append_index(self, name: str, summary: str) -> None:
        if self.storage_index_path.exists():
            entries = json.loads(self.storage_index_path.read_text(encoding="utf-8"))
        else:
            entries = []
        entries.append({"name": name, "summary": summary, "timestamp": utc_stamp()})
        self.storage_index_path.write_text(
            json.dumps(entries, ensure_ascii=True, indent=2), encoding="utf-8"
        )
