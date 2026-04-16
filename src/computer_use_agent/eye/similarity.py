from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def similarity_score(current_image: Path, previous_image: Path) -> float:
    with Image.open(current_image).convert("RGB") as a, Image.open(previous_image).convert(
        "RGB"
    ) as b:
        if a.size != b.size:
            b = b.resize(a.size)
        diff = ImageChops.difference(a, b)
        stat = ImageStat.Stat(diff)
        mean_diff = sum(stat.mean) / (len(stat.mean) * 255.0)
        return max(0.0, 1.0 - mean_diff)
