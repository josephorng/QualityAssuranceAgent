"""Lightweight consecutive-frame similarity for recording settle probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Downscale target keeps compares cheap and ignores fine cursor/noise.
_COMPARE_SIZE = (160, 90)
# Normalized mean absolute difference in [0, 1]; below this → frames are similar.
DEFAULT_SIMILARITY_THRESHOLD = 0.02


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def mean_abs_diff(
    path_a: str | Path,
    path_b: str | Path,
    *,
    size: tuple[int, int] = _COMPARE_SIZE,
) -> float:
    """Return normalized mean absolute pixel difference in ``[0, 1]``.

    Loads both images as grayscale, resizes to ``size``, and divides MAD by 255.
    Missing or unreadable files raise ``OSError`` / ``ValueError``.
    """
    from PIL import Image
    import numpy as np

    a = _as_path(path_a)
    b = _as_path(path_b)
    with Image.open(a) as img_a, Image.open(b) as img_b:
        arr_a = np.asarray(img_a.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
        arr_b = np.asarray(img_b.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"resized shapes differ: {arr_a.shape} vs {arr_b.shape}")
    return float(np.mean(np.abs(arr_a - arr_b)) / 255.0)


def compare_frames(
    path_a: str | Path,
    path_b: str | Path,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    size: tuple[int, int] = _COMPARE_SIZE,
) -> tuple[bool, float | None]:
    """Return ``(similar, mad)``; ``mad`` is ``None`` when compare fails."""
    try:
        diff = mean_abs_diff(path_a, path_b, size=size)
    except (OSError, ValueError, Exception):
        return False, None
    return diff <= float(threshold), float(diff)


def frames_similar(
    path_a: str | Path,
    path_b: str | Path,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    size: tuple[int, int] = _COMPARE_SIZE,
) -> bool:
    """True when downscaled grayscale MAD is at or below ``threshold``."""
    similar, _diff = compare_frames(path_a, path_b, threshold=threshold, size=size)
    return similar


def settle_probe_staging_path(run_dir: Path, event_index: int) -> Path:
    return Path(run_dir) / "screenshots" / f"_settle_probe_{event_index:03d}.jpeg"


def settle_probe_kept_path(run_dir: Path, event_index: int) -> Path:
    return Path(run_dir) / "screenshots" / f"_settle_stable_{event_index:03d}.jpeg"


def last_settle_frame_path(run_dir: Path) -> Path:
    """Durable frame reused as the next action's before-shot when available."""
    return Path(run_dir) / "screenshots" / "_last_settle_frame.jpeg"


def settle_probe_debug_dir(run_dir: Path, event_index: int) -> Path:
    """Per-event folder for retained settle-probe samples (threshold tuning)."""
    return Path(run_dir) / "screenshots" / "settle_debug" / f"event_{event_index:03d}"


def settle_probe_debug_sample_path(
    run_dir: Path,
    event_index: int,
    sample_index: int,
    *,
    age_s: float,
    tag: str = "sample",
) -> Path:
    """Path for one archived settle sample (``tag`` e.g. ``sample`` / ``final``)."""
    age_token = f"{float(age_s):.3f}".replace(".", "p")
    return (
        settle_probe_debug_dir(run_dir, event_index)
        / f"{tag}_{int(sample_index):03d}_age{age_token}s.jpeg"
    )


def settle_probe_debug_log_path(run_dir: Path, event_index: int) -> Path:
    return settle_probe_debug_dir(run_dir, event_index) / "comparisons.jsonl"


def archive_settle_probe_sample(
    *,
    run_dir: Path,
    event_index: int,
    sample_index: int,
    source: Path,
    age_s: float,
    tag: str = "sample",
) -> Path | None:
    """Copy ``source`` into the settle-debug folder; return the archive path."""
    import shutil

    src = Path(source)
    if not src.is_file():
        return None
    dest = settle_probe_debug_sample_path(
        run_dir, event_index, sample_index, age_s=age_s, tag=tag
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dest)
    except OSError:
        return None
    return dest


def append_settle_probe_debug_record(
    run_dir: Path,
    event_index: int,
    record: dict[str, Any],
) -> None:
    """Append one JSON object to the per-event settle comparisons log."""
    import json

    path = settle_probe_debug_log_path(run_dir, event_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def apply_settle_sample(
    *,
    kept_path: Path | None,
    kept_age_s: float | None,
    staging_path: Path,
    sample_age_s: float,
    similar: bool | None = None,
    frames_similar_fn: Any = frames_similar,
) -> tuple[Path | None, float | None, float | None]:
    """Advance settle-probe retention given a new staging sample.

    Returns ``(new_kept_path, new_kept_age_s, observed_settle_seconds)``.
    When ``observed_settle_seconds`` is not ``None``, a similar plateau was
    found (age of the earlier frame). Callers record that once for timing but
    keep sampling until the next event replaces the probe.

    On a similar pair, ``new_kept_path`` is the later frame (freshest UI).

    ``similar`` may be supplied by tests; otherwise compares ``kept`` vs ``staging``.
    """
    staging = Path(staging_path)
    if not staging.is_file():
        return kept_path, kept_age_s, None

    if kept_path is None or not Path(kept_path).is_file():
        dest = staging  # caller promotes to kept path
        return staging, float(sample_age_s), None

    kept = Path(kept_path)
    if similar is None:
        similar = bool(frames_similar_fn(kept, staging))

    if similar:
        # Stable: settle is age of the earlier frame; keep the newer image
        # as this probe's comparison baseline. Before-shots come from the
        # before-shot loop, not from this file.
        observed = (
            float(kept_age_s) if kept_age_s is not None else float(sample_age_s)
        )
        try:
            kept.unlink(missing_ok=True)
        except OSError:
            pass
        return staging, float(sample_age_s), observed

    # Still changing: drop old kept, promote staging.
    try:
        kept.unlink(missing_ok=True)
    except OSError:
        pass
    return staging, float(sample_age_s), None
