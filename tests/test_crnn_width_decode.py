from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from cua_mcp.read_screen_text.constrained_decode import (
    CharSpan,
    fill_blank_ids,
    greedy_ctc_decode_spans,
)
from cua_mcp.read_screen_text.inference_onnx import TextPredictor


def test_fill_blank_ids_expands_gaps_with_midpoint_split() -> None:
    blank = 9999
    seq = [
        blank,
        blank,
        blank,
        123,
        blank,
        blank,
        blank,
        456,
        blank,
        blank,
        blank,
        blank,
        789,
        blank,
        blank,
        blank,
    ]
    filled = fill_blank_ids(seq, blank)
    assert filled == [
        123,
        123,
        123,
        123,
        123,
        123,
        456,
        456,
        456,
        456,
        456,
        789,
        789,
        789,
        789,
        789,
    ]


def test_greedy_ctc_decode_spans_collapses_and_maps_x() -> None:
    blank = 9999
    char_dict = {"33": "A", "34": "B"}
    ids = [[33, 33, blank, 34, 34, 34]]
    spans = greedy_ctc_decode_spans(
        ids,
        char_dict,
        blank_idx=blank,
        content_width=100,
        max_timesteps=6,
    )
    assert len(spans) == 1
    row = spans[0]
    assert len(row) == 2
    assert row[0].char == "A"
    assert row[0].t_start == 0
    assert row[0].t_end == 1
    assert row[0].x_start == pytest.approx(0.0)
    assert row[0].x_end == pytest.approx(2 / 6 * 100)
    assert row[1].char == "B"
    assert row[1].t_start == 3
    assert row[1].t_end == 5
    assert row[1].x_start == pytest.approx(3 / 6 * 100)
    assert row[1].x_end == pytest.approx(6 / 6 * 100)


def test_decode_outputs_truncates_padding_timesteps() -> None:
    predictor = TextPredictor(quiet=True)
    blank = predictor.blank_idx
    # First half decodes to "A"; second half is padding that would decode to "B".
    text_ids = np.array(
        [[33, blank, blank, blank, blank, 34, 34, 34, 34, 34]],
        dtype=np.int64,
    )
    icon_ids = text_ids.copy()

    texts, spans = predictor.decode_outputs(
        text_ids,
        icon_ids,
        mode="text",
        widths=[50],
        padded_w=100,
    )

    assert texts == ["A"]
    assert len(spans) == 1
    assert len(spans[0]) == 1
    assert spans[0][0].char == "A"
    assert spans[0][0].t_end < 5


def test_decode_outputs_width_mismatch_raises() -> None:
    predictor = TextPredictor(quiet=True)
    text_ids = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="widths length"):
        predictor.decode_outputs(
            text_ids,
            text_ids,
            widths=[50],
            padded_w=100,
        )


def test_decode_outputs_requires_padded_w() -> None:
    predictor = TextPredictor(quiet=True)
    text_ids = np.zeros((1, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="padded_w"):
        predictor.decode_outputs(
            text_ids,
            text_ids,
            widths=[50],
        )


def test_predict_images_without_widths_returns_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    predictor = TextPredictor(quiet=True)
    blank = predictor.blank_idx
    text_ids = np.array([[33, 33, blank]], dtype=np.int64)
    icon_ids = text_ids.copy()
    batch = np.zeros((1, 32, 8), dtype=np.float32)

    with patch(
        "cua_mcp.vision_triton.infer_crnn",
        return_value=(text_ids, icon_ids),
    ):
        result = predictor.predict_images(batch, mode="text")

    assert isinstance(result, list)
    assert result[0] == "A"


def test_predict_images_with_widths_returns_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISION_BACKEND", "triton")
    predictor = TextPredictor(quiet=True)
    blank = predictor.blank_idx
    text_ids = np.array(
        [[33, blank, blank, blank, blank, 34, 34, 34, 34, 34]],
        dtype=np.int64,
    )
    icon_ids = text_ids.copy()
    batch = np.zeros((1, 32, 100), dtype=np.float32)

    with patch(
        "cua_mcp.vision_triton.infer_crnn",
        return_value=(text_ids, icon_ids),
    ):
        texts, spans = predictor.predict_images(batch, widths=[50], mode="text")

    assert texts == ["A"]
    assert len(spans) == 1
    assert spans[0][0].char == "A"
    assert 0 <= spans[0][0].x_start < spans[0][0].x_end <= 50
