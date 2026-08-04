"""Constrained greedy CTC decode for shared text+icon CRNN vocabularies."""

from __future__ import annotations

from typing import Literal

import numpy as np

DecodeMode = Literal["any", "text", "icon"]

PUA_MIN = 0xE000
PUA_MAX = 0xF8FF


def is_pua_char(char: str) -> bool:
    if not char or len(char) != 1:
        return False
    code = ord(char)
    return PUA_MIN <= code <= PUA_MAX


def build_allowed_class_mask(
    char_decode_dict: dict,
    nclass: int,
    mode: DecodeMode = "any",
    *,
    blank_idx: int | None = None,
) -> np.ndarray:
    """Return a boolean mask of shape [nclass] for classes allowed under ``mode``.

    Modes:
      - any: all known dictionary classes + blank
      - text: non-PUA dictionary classes + blank
      - icon: PUA dictionary classes + blank
    """
    if blank_idx is None:
        blank_idx = nclass - 1
    if blank_idx < 0 or blank_idx >= nclass:
        raise ValueError(f"blank_idx {blank_idx} out of range for nclass={nclass}")

    allowed = np.zeros(nclass, dtype=bool)
    allowed[blank_idx] = True

    for idx_str, char in char_decode_dict.items():
        try:
            idx = int(idx_str)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= nclass or idx == blank_idx:
            continue
        if mode == "any":
            allowed[idx] = True
        elif mode == "text":
            if not is_pua_char(char):
                allowed[idx] = True
        elif mode == "icon":
            if is_pua_char(char):
                allowed[idx] = True
        else:
            raise ValueError(f"Unknown decode mode: {mode!r}")

    if not allowed.any():
        raise ValueError(f"No allowed classes for mode={mode!r}")
    return allowed


def apply_crnn_class_masks(model, char_decode_dict: dict, nclass: int | None = None) -> None:
    """Configure a models_ready CRNN with text/icon class masks for dual top-1 export."""
    if nclass is None:
        nclass = int(getattr(model, "nclass", 0)) or int(model.text_class_mask.numel())
    blank_idx = nclass - 1
    text_mask = build_allowed_class_mask(
        char_decode_dict, nclass, "text", blank_idx=blank_idx
    )
    icon_mask = build_allowed_class_mask(
        char_decode_dict, nclass, "icon", blank_idx=blank_idx
    )
    import torch

    model.set_class_masks(
        torch.from_numpy(text_mask),
        torch.from_numpy(icon_mask),
    )


def _as_numpy_ids(pred_indices) -> np.ndarray:
    if hasattr(pred_indices, "detach"):
        pred_indices = pred_indices.detach().cpu().numpy()
    pred_indices = np.asarray(pred_indices)
    if pred_indices.ndim == 1:
        pred_indices = pred_indices[np.newaxis, ...]
    if pred_indices.ndim != 2:
        raise ValueError(
            f"Expected class ids shaped [batch, seq], got {pred_indices.shape}"
        )
    return pred_indices


def greedy_ctc_decode_ids(
    pred_indices,
    char_decode_dict: dict,
    *,
    blank_idx: int,
) -> list[str]:
    """Greedy CTC collapse over a [batch, seq] integer class-id sequence."""
    pred_indices = _as_numpy_ids(pred_indices)
    pred_chars: list[str] = []
    for seq_indices in pred_indices:
        chars: list[str] = []
        prev_idx = None
        for idx in seq_indices.tolist():
            idx = int(idx)
            if idx == blank_idx or idx == prev_idx:
                prev_idx = idx
                continue
            prev_idx = idx
            char = char_decode_dict.get(str(idx), "")
            if char:
                chars.append(char)
        pred_chars.append("".join(chars))
    return pred_chars


def select_mode_ids(text_ids, icon_ids, mode: DecodeMode = "text"):
    """Pick the ONNX top-1 id stream for the requested mode."""
    if mode == "icon":
        return icon_ids
    # "text" and "any" both use the text-constrained top-1 stream.
    return text_ids


def greedy_ctc_decode_logits(
    logits,
    char_decode_dict: dict,
    *,
    nclass: int | None = None,
    blank_idx: int | None = None,
    mode: DecodeMode = "any",
    allowed_mask: np.ndarray | None = None,
    return_confidence: bool = False,
) -> list[str] | tuple[list[str], list[float]]:
    """Greedy CTC decode from full logits (e.g. models_develop CRNN)."""
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits)
    if logits.ndim == 2:
        logits = logits[np.newaxis, ...]
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits shaped [batch, seq, nclass], got {logits.shape}"
        )
    if nclass is None:
        nclass = int(logits.shape[-1])
    if blank_idx is None:
        blank_idx = nclass - 1
    if allowed_mask is None:
        allowed_mask = build_allowed_class_mask(
            char_decode_dict, nclass, mode, blank_idx=blank_idx
        )
    masked = np.where(allowed_mask, logits, np.float32("-inf"))
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    pred_indices = np.argmax(masked, axis=-1)
    pred_probs = np.take_along_axis(
        probs, pred_indices[..., np.newaxis], axis=-1
    ).squeeze(-1)

    pred_chars: list[str] = []
    avg_probs: list[float] = []
    for seq_indices, seq_probs in zip(pred_indices, pred_probs):
        chars: list[str] = []
        char_probs: list[float] = []
        prev_idx = None
        for idx, prob in zip(seq_indices.tolist(), seq_probs.tolist()):
            if idx == blank_idx or idx == prev_idx:
                prev_idx = idx
                continue
            prev_idx = idx
            char = char_decode_dict.get(str(idx), "")
            if not char:
                continue
            chars.append(char)
            char_probs.append(float(prob))
        pred_chars.append("".join(chars))
        avg_probs.append(
            float(sum(char_probs) / len(char_probs)) if char_probs else 0.0
        )
    if return_confidence:
        return pred_chars, avg_probs
    return pred_chars


# Back-compat alias
greedy_ctc_decode = greedy_ctc_decode_logits


def pua_class_indices(char_decode_dict: dict) -> list[int]:
    """Return sorted class indices whose characters are in the Unicode PUA range."""
    indices: list[int] = []
    for idx_str, char in char_decode_dict.items():
        if not is_pua_char(char):
            continue
        try:
            indices.append(int(idx_str))
        except (TypeError, ValueError):
            continue
    return sorted(indices)
