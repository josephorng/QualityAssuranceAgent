"""Pure-Python CTC decode for dual top-1 ONNX outputs (text_ids / icon_ids).

No NumPy / Torch dependency — safe to copy next to deployed ONNX assets.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

DecodeMode = Literal["any", "text", "icon"]

PUA_MIN = 0xE000
PUA_MAX = 0xF8FF


def is_pua_char(char: str) -> bool:
    if not char or len(char) != 1:
        return False
    code = ord(char)
    return PUA_MIN <= code <= PUA_MAX


def select_mode_ids(text_ids: Any, icon_ids: Any, mode: DecodeMode = "text") -> Any:
    """Pick the ONNX top-1 id stream for the requested mode."""
    if mode == "icon":
        return icon_ids
    # "text" and "any" both use the text-constrained top-1 stream.
    return text_ids


def _as_batch_sequences(pred_indices: Any) -> list[Sequence[Any]]:
    """Normalize torch / numpy / nested lists into list[sequence] of shape [batch][seq]."""
    if hasattr(pred_indices, "detach"):
        pred_indices = pred_indices.detach().cpu().tolist()
    elif hasattr(pred_indices, "tolist") and not isinstance(pred_indices, (list, tuple)):
        pred_indices = pred_indices.tolist()

    if not isinstance(pred_indices, (list, tuple)):
        raise TypeError(
            f"Expected class ids as nested sequences, got {type(pred_indices)!r}"
        )
    if not pred_indices:
        return []
    first = pred_indices[0]
    # Single sequence [seq] -> batch of one
    if not isinstance(first, (list, tuple)):
        return [pred_indices]
    return list(pred_indices)


def greedy_ctc_decode_ids(
    pred_indices: Any,
    char_decode_dict: dict,
    *,
    blank_idx: int,
) -> list[str]:
    """Greedy CTC collapse over a [batch, seq] integer class-id sequence."""
    batch = _as_batch_sequences(pred_indices)
    pred_chars: list[str] = []
    for seq_indices in batch:
        chars: list[str] = []
        prev_idx = None
        for raw in seq_indices:
            idx = int(raw)
            if idx == blank_idx or idx == prev_idx:
                prev_idx = idx
                continue
            prev_idx = idx
            char = char_decode_dict.get(str(idx), "")
            if char:
                chars.append(char)
        pred_chars.append("".join(chars))
    return pred_chars


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
