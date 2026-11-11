"""Lightweight frame similarity helpers (replay UI-change checks, debug JPEG paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Downscale target keeps compares cheap and ignores fine cursor/noise.
_COMPARE_SIZE = (160, 90)
# Normalized mean absolute difference in [0, 1]; below this → frames are similar.
DEFAULT_SIMILARITY_THRESHOLD = 0.02


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _gray_resized_mad(arr_a: Any, arr_b: Any) -> float:
    """Normalized MAD in ``[0, 1]`` for two already-loaded grayscale float arrays."""
    import numpy as np

    if arr_a.shape != arr_b.shape:
        raise ValueError(f"resized shapes differ: {arr_a.shape} vs {arr_b.shape}")
    return float(np.mean(np.abs(arr_a - arr_b)) / 255.0)


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
    return _gray_resized_mad(arr_a, arr_b)


def bgr_mean_abs_diff(
    bgr_a: Any,
    bgr_b: Any,
    *,
    size: tuple[int, int] = _COMPARE_SIZE,
) -> float:
    """Return normalized MAD in ``[0, 1]`` for two in-memory BGR ``ndarray`` frames.

    Converts each frame to grayscale, resizes to ``size``, and divides MAD by 255.
    Empty or mismatched inputs raise ``ValueError``.
    """
    from PIL import Image
    import numpy as np

    def _as_gray(bgr: Any) -> Any:
        arr = np.asarray(bgr)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"expected HxWx3 BGR array, got shape={getattr(arr, 'shape', None)}")
        if arr.size == 0:
            raise ValueError("empty BGR frame")
        # OpenCV BGR → RGB for PIL.
        rgb = arr[:, :, ::-1]
        image = Image.fromarray(rgb.astype(np.uint8, copy=False), mode="RGB").convert("L")
        return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)

    return _gray_resized_mad(_as_gray(bgr_a), _as_gray(bgr_b))


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


def last_settle_frame_path(run_dir: Path) -> Path:
    """Durable debug JPEG written when a before-shot is adopted (encoded)."""
    return Path(run_dir) / "screenshots" / "_last_settle_frame.jpeg"
