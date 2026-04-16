from __future__ import annotations

from pathlib import Path


class ScreenshotDescriber:
    def describe(self, image_path: Path) -> str:
        return (
            "Mock description: screenshot captured from desktop; "
            f"file={image_path.name}. Integrate real Gemma call here."
        )
