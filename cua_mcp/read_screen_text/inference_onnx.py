import os
import time
import numpy as np
import json
import onnxruntime

onnxruntime.set_default_logger_severity(3)

class TextPredictor:
    def __init__(self, model_path=None, *, quiet: bool = False):
        self.device = "cpu"
        if not quiet:
            print("Using device:", self.device)

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "crnn_model.onnx")

        # Load configuration and dictionaries
        with open(os.path.join(os.path.dirname(__file__), 'char_dict.json'), "r", encoding="utf-8") as f:
            self.char_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'char_decode_dict.json'), "r", encoding="utf-8") as f:
            self.char_decode_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'model_config.json'), "r", encoding="utf-8") as f:
            self.config_dict = json.load(f)

        # Initialize ONNX Runtime session with dynamic axes configuration
        if not quiet:
            print("Loading ONNX model...")
        start_time = time.time()
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        if not quiet:
            print("ONNX model loaded from", model_path)
            end_time = time.time()
            print("Time taken: ", end_time - start_time)

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
        # ✅ Convert image to NumPy array of shape [1, 1, H, W]
        if isinstance(images, list):
            images = np.array(images)
        if len(images.shape) == 2:  # [H, W]
            images = np.expand_dims(images, axis=0)  # [1, H, W]
        images = images.astype(np.float32)
        if hxs is None:
            hxs = None
        else:
            hxs = hxs.to(self.device)
        outputs = self.session.run(None, {self.input_name: images})[0]
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
   