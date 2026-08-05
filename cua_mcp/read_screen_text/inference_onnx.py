import os
import time
from typing import Sequence

import numpy as np
import json

from .constrained_decode import (
    DecodeMode,
    greedy_ctc_decode_ids,
    select_mode_ids,
)


def _log_crnn_profile(message: str) -> None:
    try:
        from src.common.run_state import get_run_state_manager

        get_run_state_manager().log_info(f"[vision/crnn] {message}")
    except RuntimeError:
        pass


def _normalize_modes(mode: DecodeMode | Sequence[DecodeMode], batch_size: int) -> list[DecodeMode]:
    if isinstance(mode, str):
        return [mode] * batch_size
    modes = list(mode)
    if len(modes) != batch_size:
        raise ValueError(
            f"mode length {len(modes)} does not match batch size {batch_size}"
        )
    return modes


class TextPredictor:
    def __init__(self, model_path=None, *, quiet: bool = False):
        self.device = "cpu"
        self.quiet = quiet
        if not quiet:
            print("Using device:", self.device)

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "crnn_model.onnx")
        self.model_path = model_path

        # Load configuration and dictionaries
        with open(os.path.join(os.path.dirname(__file__), 'char_dict.json'), "r", encoding="utf-8") as f:
            self.char_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'char_decode_dict.json'), "r", encoding="utf-8") as f:
            self.char_decode_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'model_config.json'), "r", encoding="utf-8") as f:
            self.config_dict = json.load(f)

        self.session = None
        self.input_name = None
        self.blank_idx = int(self.config_dict["nclass"]) - 1

    def decode_outputs(
        self,
        text_ids,
        icon_ids=None,
        *,
        mode: DecodeMode | Sequence[DecodeMode] = "text",
    ):
        """
        Decode dual top-1 ONNX/Triton id streams into text predictions.

        Args:
            text_ids: [batch, seq] text-constrained top-1 class ids
            icon_ids: [batch, seq] icon/PUA-constrained top-1 class ids
            mode: ``"text"`` / ``"icon"`` / ``"any"``, or one mode per batch row

        Returns:
            pred_chars: List of predicted character strings
        """
        if icon_ids is None:
            # Back-compat for callers that still pass a single id stream.
            icon_ids = text_ids

        text_ids = np.asarray(text_ids)
        icon_ids = np.asarray(icon_ids)
        if text_ids.ndim == 1:
            text_ids = text_ids[np.newaxis, ...]
        if icon_ids.ndim == 1:
            icon_ids = icon_ids[np.newaxis, ...]
        if text_ids.shape != icon_ids.shape:
            raise ValueError(
                f"text_ids shape {text_ids.shape} != icon_ids shape {icon_ids.shape}"
            )

        modes = _normalize_modes(mode, text_ids.shape[0])
        if len(set(modes)) == 1:
            ids = select_mode_ids(text_ids, icon_ids, mode=modes[0])
            return greedy_ctc_decode_ids(
                ids, self.char_decode_dict, blank_idx=self.blank_idx
            )

        pred_chars: list[str] = []
        for i, row_mode in enumerate(modes):
            ids = select_mode_ids(
                text_ids[i : i + 1], icon_ids[i : i + 1], mode=row_mode
            )
            pred_chars.extend(
                greedy_ctc_decode_ids(
                    ids, self.char_decode_dict, blank_idx=self.blank_idx
                )
            )
        return pred_chars

    def predict_images(
        self,
        images,
        hxs=None,
        *,
        mode: DecodeMode | Sequence[DecodeMode] = "text",
    ):
        # Input: [batch, line_height, width] float32 (line_height is typically 32).
        if isinstance(images, list):
            images = np.array(images)
        if len(images.shape) == 2:  # [H, W]
            images = np.expand_dims(images, axis=0)  # [1, H, W]
        images = images.astype(np.float32)
        if hxs is None:
            hxs = None
        else:
            hxs = hxs.to(self.device)

        from cua_mcp.vision_triton import infer_crnn

        started = time.perf_counter()
        text_ids, icon_ids = infer_crnn(images)
        elapsed = time.perf_counter() - started
        _log_crnn_profile(
            f"infer backend=triton shape={list(images.shape)} "
            f"elapsed_s={elapsed:.3f}"
        )

        return self.decode_outputs(text_ids, icon_ids, mode=mode)

class TextExtractor:
    """
    Legacy placeholder kept for backward compatibility.

    The active OCR pipeline uses `TextPredictor` via `ocr_image.py`.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TextExtractor is deprecated and no longer supported. Use TextPredictor-based OCR via cua_mcp.read_screen_text.ocr_image."
        )
