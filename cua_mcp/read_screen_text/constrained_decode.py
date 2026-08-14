"""Pure-Python CTC decode for dual top-1 ONNX outputs (text_ids / icon_ids).

No NumPy / Torch dependency — safe to copy next to deployed ONNX assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

DecodeMode = Literal["any", "text", "icon"]


@dataclass(frozen=True)
class CharSpan:
    """One collapsed CTC character run with timestep and pixel x bounds."""

    char: str
    t_start: int
    t_end: int
    x_start: float
    x_end: float

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


def fill_blank_ids(seq_indices: Sequence[Any], blank_idx: int) -> list[int]:
    """Replace blank class ids by expanding neighboring character ids into gaps."""
    seq = [int(raw) for raw in seq_indices]
    if not seq:
        return []

    anchors = [(t, idx) for t, idx in enumerate(seq) if idx != blank_idx]
    if not anchors:
        return seq

    filled = list(seq)
    first_t, first_id = anchors[0]
    last_t, last_id = anchors[-1]

    for t in range(first_t):
        filled[t] = first_id
    for t in range(last_t + 1, len(seq)):
        filled[t] = last_id

    for (left_t, left_id), (right_t, right_id) in zip(anchors, anchors[1:]):
        if right_t - left_t <= 1:
            continue
        midpoint = (left_t + right_t + 1) // 2
        for t in range(left_t + 1, right_t):
            if seq[t] != blank_idx:
                continue
            filled[t] = left_id if t <= midpoint else right_id

    return filled


def _fill_blank_batch(pred_indices: Any, blank_idx: int) -> list[list[int]]:
    batch = _as_batch_sequences(pred_indices)
    return [fill_blank_ids(seq, blank_idx) for seq in batch]


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


def _span_from_run(
    char: str,
    t_start: int,
    t_end: int,
    *,
    content_width: int,
    max_timesteps: int,
) -> CharSpan:
    if max_timesteps <= 0:
        max_timesteps = 1
    x_start = t_start / max_timesteps * content_width
    x_end = (t_end + 1) / max_timesteps * content_width
    return CharSpan(
        char=char,
        t_start=t_start,
        t_end=t_end,
        x_start=x_start,
        x_end=x_end,
    )


def greedy_ctc_decode_spans(
    pred_indices: Any,
    char_decode_dict: dict,
    *,
    blank_idx: int,
    content_width: int,
    max_timesteps: int,
) -> list[list[CharSpan]]:
    """Greedy CTC collapse returning character spans with pixel x ranges."""
    batch = _as_batch_sequences(pred_indices)
    results: list[list[CharSpan]] = []
    for seq_indices in batch:
        spans: list[CharSpan] = []
        prev_idx: int | None = None
        run_start: int | None = None
        run_char: str | None = None
        for t, raw in enumerate(seq_indices):
            idx = int(raw)
            if idx == blank_idx:
                if run_char is not None and run_start is not None:
                    spans.append(
                        _span_from_run(
                            run_char,
                            run_start,
                            t - 1,
                            content_width=content_width,
                            max_timesteps=max_timesteps,
                        )
                    )
                prev_idx = idx
                run_start = None
                run_char = None
                continue
            if idx == prev_idx:
                continue
            if run_char is not None and run_start is not None:
                spans.append(
                    _span_from_run(
                        run_char,
                        run_start,
                        t - 1,
                        content_width=content_width,
                        max_timesteps=max_timesteps,
                    )
                )
            prev_idx = idx
            run_start = t
            run_char = char_decode_dict.get(str(idx), "")
            if not run_char:
                run_start = None
                run_char = None
        if run_char is not None and run_start is not None:
            spans.append(
                _span_from_run(
                    run_char,
                    run_start,
                    len(seq_indices) - 1,
                    content_width=content_width,
                    max_timesteps=max_timesteps,
                )
            )
        results.append(spans)
    return results


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
