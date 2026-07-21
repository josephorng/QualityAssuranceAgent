import os
import time
import numpy as np
import json


def _log_crnn_profile(message: str) -> None:
    try:
        from src.common.run_state import get_run_state_manager

        get_run_state_manager().log_info(f"[vision/crnn] {message}")
    except RuntimeError:
        pass


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

    def decode_outputs(self, outputs):
        """
        Decode the ONNX model outputs into text predictions.
        
        Args:
            outputs: numpy array of shape [batch, seq_len, num_classes]
            
        Returns:
            pred_chars: List of predicted character strings
            avg_probs: List of confidence scores for each prediction
        """
        pred_chars = []
        
        for seq_indices in outputs:
            chars = []
            prev_char = None
            
            # Convert indices to characters, skipping repeated and blank tokens
            for idx in seq_indices:
                idx = idx.item()
                
                # Skip if same as previous char or blank token (last class)
                if idx == self.config_dict['nclass'] - 1 or idx == prev_char:
                    prev_char = idx
                    continue
                    
                prev_char = idx
                # Get character from char_dict by finding key with matching value
                char = self.char_decode_dict.get(str(idx), '')
                chars.append(char)
            
            pred_chars.append(''.join(chars))
        
        return pred_chars

    def predict_images(self, images, hxs=None):
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
        outputs = infer_crnn(images)
        elapsed = time.perf_counter() - started
        _log_crnn_profile(
            f"infer backend=triton shape={list(images.shape)} "
            f"elapsed_s={elapsed:.3f}"
        )

        pred_chars = self.decode_outputs(outputs)

        return pred_chars

class TextExtractor:
    """
    Legacy placeholder kept for backward compatibility.

    The active OCR pipeline uses `TextPredictor` via `ocr_image.py`.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TextExtractor is deprecated and no longer supported. Use TextPredictor-based OCR via cua_mcp.read_screen_text.ocr_image."
        )
