import argparse
import json
import os

import numpy as np
import torch


class TextPredictor:
    def __init__(self, model_path):
        """Load the TorchScript CRNN model and decode metadata."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        if model_path is None:
            raise ValueError("You must provide a path to the TorchScript model.")

        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        print("TorchScript model loaded from", model_path)

        with open(os.path.join(os.path.dirname(__file__), "char_dict.json"), "r", encoding="utf-8") as f:
            self.char_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), "char_decode_dict.json"), "r", encoding="utf-8") as f:
            self.char_decode_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), "model_config.json"), "r", encoding="utf-8") as f:
            self.config_dict = json.load(f)

    def decode_outputs(self, outputs):
        """Greedy-decode model logits into text and average confidence."""
        probs = torch.softmax(outputs, dim=-1)
        pred_probs, pred_indices = torch.max(probs, dim=-1)

        pred_chars = []
        avg_probs = []
        for seq_indices, seq_probs in zip(pred_indices, pred_probs):
            chars = []
            probs_list = []
            prev_char = None
            for idx, prob in zip(seq_indices, seq_probs):
                idx = idx.item()
                prob = prob.item()

                if idx == self.config_dict["nclass"] - 1 or idx == prev_char:
                    prev_char = idx
                    continue

                prev_char = idx
                char = self.char_decode_dict.get(str(idx), "")
                chars.append(char)
                probs_list.append(prob)

            avg_prob = sum(probs_list) / len(probs_list) if probs_list else 0.0
            pred_chars.append("".join(chars))
            avg_probs.append(avg_prob)

        return pred_chars, avg_probs

    def beam_search_decode_outputs(self, outputs):
        """Decode beam-search token indices into text strings."""
        pred_chars = []
        for seq in outputs:
            chars = []
            prev_char = None
            for idx in seq:
                idx = idx.item()
                if idx == self.config_dict["num_classes"] - 1 or idx == prev_char:
                    prev_char = idx
                    continue
                char = self.char_decode_dict.get(str(idx), "")
                chars.append(char)
                prev_char = idx
            pred_chars.append("".join(chars))
        return pred_chars

    def predict_images(self, images, hxs=None, beam_search=True, beam_width=2):
        """Run CRNN inference on image tensors and return decoded text."""
        self.model.eval()
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self.device)

        if len(images.shape) == 2:
            images = images.unsqueeze(0)

        if hxs is not None:
            hxs = hxs.to(self.device)

        if beam_search:
            pred_chars = self.model.beam_search(images, beam_width=beam_width)
            pred_chars = self.beam_search_decode_outputs(pred_chars)
            pred_prob_avg = None
        else:
            outputs = self.model(images)
            pred_chars, pred_prob_avg = self.decode_outputs(outputs)

        return pred_chars, pred_prob_avg


os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    """CLI entrypoint for running CRNN on a saved numpy tensor."""
    parser = argparse.ArgumentParser(description="Run CRNN inference on a .npy image tensor.")
    parser.add_argument("input_npy_path", help="Path to a .npy array used as CRNN input.")
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.path.dirname(__file__), "crnn_cfc_model.pth"),
        help="Path to the TorchScript CRNN model.",
    )
    args = parser.parse_args()

    predictor = TextPredictor(args.model_path)
    images = np.load(args.input_npy_path)
    predicted_texts, pred_prob_avg = predictor.predict_images(images, beam_search=False)
    print("predicted_texts:", predicted_texts)
    print("pred_prob_avg:", pred_prob_avg)


if __name__ == "__main__":
    main()
import os
import argparse
import time
import cv2
import numpy as np
import os
import json
import torch

class TextPredictor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        if model_path is None:
            raise ValueError("You must provide a path to the TorchScript model.")
        
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        print("TorchScript model loaded from", model_path)
        
        
        with open(os.path.join(os.path.dirname(__file__), 'char_dict.json'), "r", encoding="utf-8") as f:
            self.char_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'char_decode_dict.json'), "r", encoding="utf-8") as f:
            self.char_decode_dict = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'model_config.json'), "r", encoding="utf-8") as f:
            self.config_dict = json.load(f)

    def decode_outputs(self, outputs):
        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=-1)  # [batch, seq_len, nclass]
        
        # Get predicted indices and their probabilities
        pred_probs, pred_indices = torch.max(probs, dim=-1)  # [batch, seq_len], [batch, seq_len]
        
        # Convert indices to characters and collect probabilities
        pred_chars = []
        avg_probs = []
        
        for seq_indices, seq_probs in zip(pred_indices, pred_probs):
            chars = []
            probs_list = []
            prev_char = None
            
            # Convert indices to characters, skipping repeated and blank tokens
            for idx, prob in zip(seq_indices, seq_probs):
                idx = idx.item()
                prob = prob.item()
                
                # Skip if same as previous char or blank token (last class)
                if idx == self.config_dict['nclass'] - 1 or idx == prev_char:
                    prev_char = idx
                    continue
                    
                prev_char = idx
                # Get character from char_dict by finding key with matching value
                char = self.char_decode_dict.get(str(idx), '')
                chars.append(char)
                probs_list.append(prob)
                
            # Compute average probability for the sequence
            avg_prob = sum(probs_list) / len(probs_list) if probs_list else 0.0
            
            pred_chars.append(''.join(chars))
            avg_probs.append(avg_prob)
        
        return pred_chars, avg_probs
    
    def beam_search_decode_outputs(self, outputs):
        # Get predicted indices by taking argmax along class dimension
        pred_indices = outputs  # [batch, seq_len, probabilities]
        # Convert indices to characters
        pred_chars = []
        for seq in pred_indices:
            chars = []
            # Convert indices to characters, skipping repeated and blank tokens
            prev_char = None
            for idx in seq:
                idx = idx.item()
                # Skip if same as previous char or blank token (last class)
                if idx == self.config_dict['num_classes'] - 1 or idx == prev_char:
                    prev_char = idx
                    continue
                # Get character from char_dict by finding key with matching value
                char = self.char_decode_dict.get(str(idx), '')
                chars.append(char)
                prev_char = idx
            pred_chars.append(''.join(chars))
        return pred_chars
    
    def predict_images(self, images, hxs=None, beam_search=True, beam_width=2):    
        self.model.eval()
        # Convert numpy array to torch tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self.device)
        # Check if images has 4 dimensions [batch, seq, height, width, channel]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)  # Add batch dimension if missing
        batch_size = images.size(0)
        if hxs is None:
            hxs = None
        else:
            hxs = hxs.to(self.device)
        # print("images.shape=", images.shape)
        # outputs, hx = self.model(images, hxs)  # [batch, seq_len, num_classes + 1]
        if beam_search:
            pred_chars = self.model.beam_search(images, beam_width=beam_width)
            pred_chars = self.beam_search_decode_outputs(pred_chars)
            pred_prob_avg = None # need implement beam search prob
        else:
            outputs = self.model(images)  # [batch, seq_len, num_classes + 1]
            pred_chars, pred_prob_avg = self.decode_outputs(outputs)

        return pred_chars, pred_prob_avg

os.environ['PYTHONIOENCODING'] = 'utf-8'


NEXT = 0
PREVIOUS = 1
FIRST_CHILD = 2
PARENT = 3

BINARY_THRESHOLD = 0.9

class TextExtractor:
    """
    A class for extracting text from images by segmenting into lines, words, and characters.
    """
    def __init__(self, image_path=None, save_dir=None, rgb=True, **kwargs):
        """
        Initialize the TextExtractor.
        
        Args:
            image_path (str, optional): Path to input image file
            save_dir (str, optional): Directory to save output files
            rgb (bool): Whether to output RGB images (True) or grayscale (False)
            **kwargs: Additional keyword arguments including:
                - image: Direct image array input instead of file
                - debug (bool): Enable debug output
                - verbose (bool): Enable verbose logging
                - plot (bool): Enable plotting of intermediate results
                - height (int): Target height for resizing (default 128)
        """
        self.image_path = image_path
        self.save_dir = save_dir
        if self.save_dir:
            if os.path.exists(self.save_dir):
                try:
                    import shutil
                    shutil.rmtree(self.save_dir)
                except Exception as e:
                    print(f"Error deleting directory {self.save_dir}: {e}")
            os.makedirs(self.save_dir, exist_ok=True)
        self.rgb = rgb
        if image_path:
            self.image = cv2.imread(image_path)
        elif kwargs.get('image') is not None:
            self.image = kwargs.get('image')
        else:
            raise ValueError("Either image_path or image must be provided")

        self.debug = kwargs.get('debug', False)
        self.verbose = kwargs.get('verbose', True)
        self.plot = kwargs.get('plot', False)
        self.height = kwargs.get('height', 128)
        
        # Preprocess the image once
        if self.image is not None:
            self._preprocess_image()
        else:
            raise ValueError("Image is None", image_path)

    def _preprocess_image(self):
        """Preprocess the image once and store results."""
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.edges = cv2.Canny(self.blurred, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.dilated = cv2.dilate(self.edges, kernel, iterations=3)
        self.contours, self.hierarchy = cv2.findContours(self.dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._show_images(self.dilated, 'Dilated', "Preprocessed Dilated Image", force=self.plot)
    
    def _show_images(self, images, title=None, description=None, debug=False, force=False):
        """
        Display images for debugging/visualization.
        
        Args:
            images: Single image or list of images to display
            title (str): Window title
            description (str): Description to print if verbose
            debug (bool): Force display if debug mode
            force (bool): Force display regardless of settings
        """
        if self.plot or (debug and self.debug) or force:
            if self.verbose and description:
                print(description)
            if isinstance(images, list):
                for image in images:
                    cv2.imshow(title, image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                cv2.imshow(title, images)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def _print(self, *args, **kwargs):
        """
        Print if verbose mode is enabled.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments for print
        """
        if self.verbose:
            print(*args, **kwargs)
    
    
    def _replace_consecutive_with_average(self, numbers):
        """
        Replace consecutive numbers with their average.
        
        Args:
            numbers: List of numbers
            
        Returns:
            List with consecutive numbers replaced by their average
        """
        if numbers is None or len(numbers) == 0:
            return []
        new_numbers = []
        count = 1
        total = numbers[0]
        for i in range(1, len(numbers)):
            if abs(numbers[i] - numbers[i - 1]) <= 1:
                count += 1
                total += numbers[i]
            else:
                new_numbers.append(round(total / count) if count > 1 else numbers[i - 1])
                count = 1
                total = numbers[i]
        new_numbers.append(round(total / count) if count > 1 else numbers[-1])
        return new_numbers

    def _find_words_indices(self, numbers, space_width=5):
        """
        Find gaps between words based on indices.
        
        Args:
            numbers: List of gap indices
            space_width (int): Minimum width to consider as word gap
            
        Returns:
            List of (start,end) tuples for word boundaries
        """
        new_numbers = sorted(set(numbers))
        grouped_numbers = []
        current_group = []

        for number in new_numbers:
            if not current_group or number - current_group[-1] == 1:
                current_group.append(number)
            else:
                grouped_numbers.append(current_group)
                current_group = [number]

        if current_group:
            grouped_numbers.append(current_group)
            
        new_numbers = [[group[0], group[-1]] for group in grouped_numbers if len(group) >= space_width]
        new_numbers = [num for group in new_numbers for num in group]
        if len(new_numbers) == 0:
            return []
        if new_numbers[0]!=0:
            new_numbers.insert(0, 0)
        else:
            new_numbers.pop(0)
        if new_numbers[-1]!=numbers[-1]:
            new_numbers.append(numbers[-1])
        else:
            new_numbers.pop(-1)
        new_numbers = [(new_numbers[i], new_numbers[i + 1]) for i in range(0, len(new_numbers) - 1, 2)]
        return new_numbers

    def _find_text_area_rectangles(self):
        """
        Find rectangular regions containing text in the image.
        
        Uses edge detection and contour finding to identify text regions.
        Filters overlapping rectangles.
        
        Returns:
            List of (x,y,w,h) tuples defining text rectangles
        """
        self.txt_rectangles = []
        image_copy = self.image.copy()
        contours = sorted(self.contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3], reverse=True)
        
        filtered_rectangles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            current_rectangle = (x, y, w, h)
            overlap_found = False
            for rect in filtered_rectangles:
                x2, y2, w2, h2 = rect
                area1, area2 = w * h, w2 * h2
                x_overlap = max(0, min(x + w, x2 + w2) - max(x, x2))
                y_overlap = max(0, min(y + h, y2 + h2) - max(y, y2))
                overlap_area = x_overlap * y_overlap
                if overlap_area > 0 and overlap_area / min(area1, area2) >= 0.5:
                    overlap_found = True
                    break
            if not overlap_found:
                filtered_rectangles.append(current_rectangle)
                self.txt_rectangles.append((x, y, w, h))
                if self.plot:
                    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self._show_images(image_copy, 'Text Area Rectangles', "find_text_area_rectangles => Text Area Rectangles")
        return self.txt_rectangles

    def _detect_and_remove_border(self, image, image_rect):
        """
        Detect if there is a border around the image and remove it if present.
        
        Args:
            image (numpy.ndarray): Input image in BGR format (as a NumPy array).
        
        Returns:
            tuple: (bool, numpy.ndarray)
                - bool: True if a border was detected and removed, False otherwise.
                - numpy.ndarray: The image with the border removed if detected, otherwise the original image.
        """
        x0, y0, w0, h0 = image_rect
        relevant_indices = [i for i, c in enumerate(self.contours) 
                        if cv2.pointPolygonTest(c, (x0 + w0/2, y0 + h0/2), False) >= 0]
        if len(relevant_indices) == 0 or self.hierarchy is None:
            return False, image, image_rect
        
        img_area_threshold = 0.8 * w0 * h0
        hierarchy = self.hierarchy[0]
        
        def is_border_like(contour):
            x, y, w, h = cv2.boundingRect(contour)
            return (x - x0 <= 5 and y - y0 <= 5 and w >= w0 - 10 and h >= h0 - 10 and w * h >= img_area_threshold)
        
        root_indices = [i for i in relevant_indices if hierarchy[i][3] == -1]
        for idx in root_indices:
            if is_border_like(self.contours[idx]):
                child_indices = [i for i, h in enumerate(hierarchy) if h[PARENT] == idx and i in relevant_indices]
                if child_indices:
                    sub_area = sum(cv2.contourArea(self.contours[i]) for i in child_indices)
                    if sub_area >= 0.25 * cv2.contourArea(self.contours[idx]):
                        all_points = np.vstack([self.contours[i] for i in child_indices])
                        x, y, w, h = cv2.boundingRect(all_points)
                        cropped_image = image[y:y+h, x:x+w]
                        self._show_images(cropped_image, 'Cropped Image', "detect_and_remove_border => Cropped Image")
                        return True, cropped_image, (x0 + x, y0 + y, w, h)
        return False, image, image_rect
    
    def _remove_surrounded_black_space(self, line_contour, line_rect):
        """
        Remove trailing black space from the text area by analyzing column histogram.
        Also removes top and bottom black space.
        
        Args:
            line_contour: Image array of text line
            line_rect: (x,y,w,h) tuple defining rectangle region
                        
        Returns:
            Tuple containing:
            - Image array with trailing black space removed
            - Updated (x,y,w,h) rectangle coordinates
        """
        col_histogram = np.sum(line_contour, axis=0)
        threshold = 0.1 * np.mean(col_histogram)
        left, right = 0, line_contour.shape[1] - 1
        while left < right and col_histogram[left] < threshold:
            left += 1
        while right > left and col_histogram[right] < threshold:
            right -= 1
        
        row_histogram = np.sum(line_contour, axis=1)
        top, bottom = 0, line_contour.shape[0] - 1
        while top < bottom and row_histogram[top] < threshold:
            top += 1
        while bottom > top and row_histogram[bottom] < threshold:
            bottom -= 1
        
        cropped_contour = line_contour[top:bottom+1, left:right+1]
        x, y, w, h = line_rect
        new_rect = (x + left, y + top, right - left + 1, bottom - top + 1)
        self._show_images(cropped_contour, 'Remove_trailing_black_space', "remove_trailing_black_space => Remove_trailing_black_space")
        return cropped_contour, new_rect
    
    def _get_normalized_np_arr(self, text_area):
        """
        Normalize the image array to [0, 255].
        
        Args:
            text_area: Image array of text region (grayscale, typically 0-255)
            
        Returns:
            normalized_area: Image array of normalized text area
        """
        text_area_float = text_area.astype(np.float32)
        min_val, max_val = np.min(text_area_float), np.max(text_area_float)
        if max_val > min_val:  # Avoid division by zero or uniform image
            normalized_area = 255 * (text_area_float - min_val) / (max_val - min_val)
        else:
            normalized_area = text_area_float
        cv_normalized_area = cv2.convertScaleAbs(normalized_area)
        self._show_images(cv_normalized_area, 'Text Area', "get_normalized_np_arr => Text Area")
        return normalized_area, cv_normalized_area

    def _is_perimeter_black(self, normalized_area_np_arr):
        """
        Check if the perimeter of a text area is black after normalizing to [0, 1].
        
        Args:
            text_area: Image array of text region (grayscale, typically 0-255)
            
        Returns:
            bool: True if perimeter is black (mean < 0.5 after normalization), False otherwise
            normalized_area: Image array of normalized text area
        """
        if isinstance(normalized_area_np_arr, cv2.Mat):
            normalized_area_np_arr = np.array(normalized_area_np_arr)
        perimeter_pixels = np.concatenate((
            normalized_area_np_arr[0, :], normalized_area_np_arr[-1, :],
            normalized_area_np_arr[:, 0], normalized_area_np_arr[:, -1]
        ))
        mean_val = np.mean(perimeter_pixels)
        if self.verbose:
            self._print(f"is_perimeter_black => Text Area, mean_val={mean_val:.3f}")
        return mean_val < 128
    
    
    def find_lines(self, contour, contour_rect):
        """
        Find text lines in a contour region.
        
        Args:
            contour: Image array of contour region
            rect: (x,y,w,h) tuple defining rectangle region
            
        Returns:
            List of image arrays containing individual text lines
        """
        _, result_image, result_rect = self._detect_and_remove_border(contour, contour_rect)
        if result_image.shape[0] == 0 or result_image.shape[1] == 0:
            return []
        lines = self._find_lines_in_contour(result_image, result_rect)
        lines = [self._remove_surrounded_black_space(line, line_rect) for line, line_rect in lines]
        for i, (line, line_rect) in enumerate(lines):
            normalized_area, _ = self._get_normalized_np_arr(line)
            if not self._is_perimeter_black(normalized_area):
                lines[i] = (cv2.bitwise_not(line), line_rect)
        lines = [(line, line_rect) for line, line_rect in lines if line_rect[2] * line_rect[3] > 100 and line_rect[3] > 10]
        return lines
    
    def _find_lines_in_contour(self, contour, contour_rect):
        """
        Find text lines in a contour using horizontal projection.
        
        Args:
            contour: Image array of contour region
            contour_rect: (x,y,w,h) tuple defining rectangle region
            
        Returns:
            List of tuples containing:
            - Image array of text line
            - (x,y,w,h) rectangle coordinates for that line
        """
        lines_with_rects = []
        contour = cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY)
        normalized_area, contour = self._get_normalized_np_arr(contour)
        if not self._is_perimeter_black(normalized_area):
            contour = cv2.bitwise_not(contour)
        
        T, _ = cv2.threshold(contour, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, contour = cv2.threshold(contour, T * BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        horizontal_histogram = np.sum(contour, axis=1)
        min_line_height, max_gap_to_merge = 5, 3
        gap_indices = np.where(horizontal_histogram <= 0)[0]
        line_indices = np.insert(gap_indices, 0, 0)
        line_indices = np.append(line_indices, contour.shape[0])
        line_indices = self._replace_consecutive_with_average(line_indices)
        
        merged_indices = [line_indices[0]]
        for i in range(1, len(line_indices)):
            gap_size = line_indices[i] - merged_indices[-1]
            if gap_size <= max_gap_to_merge and i < len(line_indices) - 1:
                continue
            elif gap_size < min_line_height and i < len(line_indices) - 1:
                continue
            else:
                merged_indices.append(line_indices[i])
        
        x, y, w, h = contour_rect
        for i in range(len(merged_indices) - 1):
            line_start, line_end = merged_indices[i], merged_indices[i + 1]
            if line_end - line_start >= min_line_height:
                line_image = contour[line_start:line_end, :]
                lines_with_rects.append((line_image, (x, y + line_start, w, line_end - line_start)))
        
        self._show_images([line for line, _ in lines_with_rects], 'Lines', "find_lines => Lines")
        return lines_with_rects

    def _find_words(self, line):
        """
        Find words in a line of text using vertical projection.
        
        Args:
            line: Image array containing a line of text
            
        Returns:
            List of image arrays containing individual words
        """
        
        self._show_images(line, 'Line', "find_words => Line")
        
        # Ensure the line image is binary (1 channel)
        if line.ndim != 2:  # Check if the line is not a 2D array (grayscale)
            line = cv2.cvtColor(np.array(line), cv2.COLOR_RGB2GRAY)
        ret, line_binary = cv2.threshold(line, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        self._show_images(line_binary, 'Line Binary', "find_words => Line Binary")
        self._print("threshold=", ret)


        segmented_words = []
        # Calculate the histogram of the binary line image
        vertical_histogram = np.sum(line_binary, axis=0)

        # Threshold for identifying words based on the histogram
        threshold = np.mean(vertical_histogram) * 0.1
        space_indices = np.where(vertical_histogram <= threshold)[0]
        space_indices = np.insert(space_indices, 0, 0)  # Add 0 at the beginning
        space_indices = np.append(space_indices, line.shape[1])  # Add width at the end
        space_width = int(line.shape[0] * 0.25)
        gap_pairs = self._find_words_indices(space_indices, space_width=space_width)

        # Find start and end indices of words
        if len(gap_pairs) > 0:
            for gap_pair in gap_pairs:
                w_start = gap_pair[0]
                w_end = gap_pair[1]
                segmented_words.append(line[:, w_start:w_end])
        else:
            segmented_words.append(line)

        if self.plot:
            import matplotlib.pyplot as plt
            for word in segmented_words:
                self._print("Show the output word image for find_words")
                fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(10, 5))
                ax2.set_title('Word Image')
                ax2.imshow(word, cmap='gray')
                ax2.axis('off')  # Hide axis

                # Plot the word histogram
                ax1.set_title('Word Histogram')
                ax1.set_xlabel('Column Index')
                ax1.set_ylabel('Sum of Pixel Values')
                ax1.plot(vertical_histogram, color='blue')
                ax1.axhline(y=threshold, color='red', linestyle='--', label='Mean Histogram')
                ax1.legend() 
                plt.tight_layout() 
                plt.show(block=True) 

        self._print(f'Found {len(segmented_words)} words in a line.')
        
        if len(segmented_words) == 0 and self.debug:
            self._show_images(line, 'Line', "find_words => Line", debug=True)
        return segmented_words

    def _find_characters(self, word):
        """
        Find individual characters in a word using vertical projection.
        
        Args:
            word: Image array containing a word
            
        Returns:
            List of image arrays containing individual characters
        """

        # Ensure the line image is binary (1 channel)
        if word.ndim != 2:  # Check if the line is not a 2D array (grayscale)
            word = cv2.cvtColor(np.array(word), cv2.COLOR_RGB2GRAY)
        ret, word_binary = cv2.threshold(word, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        self._print("threshold=", ret)

        self._print("Show the input word image for find_characters")
        self._show_images(word, 'Line', "find_characters => Line")

        # Calculate the histogram of the binary line image
        vertical_histogram = np.sum(np.array(word_binary), axis=0)

        # Threshold for identifying words based on the histogram
        threshold = 10 # np.mean(vertical_histogram) * 0.3
        word_indices = np.where(vertical_histogram <= threshold)[0]
        word_indices = np.insert(word_indices, 0, 0)  # Add 0 at the beginning
        word_indices = np.append(word_indices, word_binary.shape[1])  # Add width at the end
        word_indices = self._replace_consecutive_with_average(word_indices)

        # Find start and end indices of characters
        segmented_characters = []
        if len(word_indices) > 1:
            for i in range(1, len(word_indices)):
                w_start = word_indices[i - 1]
                w_end = word_indices[i]
                character = word[:, w_start:w_end]
                character_padded = cv2.copyMakeBorder(character, 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=0)
                segmented_characters.append(character_padded)
        else:
            segmented_characters.append(word)

        if self.rgb:
            segmented_characters = [cv2.cvtColor(character, cv2.COLOR_GRAY2RGB) for character in segmented_characters]

        if self.plot:
            import matplotlib.pyplot as plt
            for character in segmented_characters:
                self._print("Show the output character image for find_characters")
                fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(10, 5))
                ax2.set_title('Character Image')
                ax2.imshow(character, cmap='gray')
                ax2.axis('off')  # Hide axis

                # Plot the word histogram
                ax1.set_title('Character Histogram')
                ax1.set_xlabel('Column Index')
                ax1.set_ylabel('Sum of Pixel Values')
                ax1.plot(vertical_histogram, color='blue')
                ax1.axhline(y=threshold, color='red', linestyle='--', label='Mean Histogram')
                ax1.legend() 
                plt.tight_layout() 
                plt.show(block=True) 

        self._print(f'Found {len(segmented_characters)} characters in a word.')
        
        if len(segmented_characters) == 0 and self.debug:
            self._show_images(word, 'Line', "find_characters => Show the input word image because no characters is found", debug=True)
            
        return segmented_characters

    def line_iterator(self, save_character=False):
        """
        Iterator that yields characters from text lines with word/line boundary flags.
        
        Args:
            save_character (bool): Whether to save individual character images
            
        Yields:
            Tuple containing:
            - character: Image array of character
            - is_last_character: Bool indicating if last char in word
            - is_last_word: Bool indicating if last word in line 
            - save_dir: Directory path if saving character
        """    
        segmented_words = self._find_words(self.image)
        for i_word, word in enumerate(segmented_words):
            if self.save_dir:
                os.makedirs(f'{self.save_dir}/word_{i_word}', exist_ok=True)
                cv2.imwrite(f'{self.save_dir}/word_{i_word}.png', word)
            segmented_characters = self._find_characters(word)
            for i_character, character in enumerate(segmented_characters):
                return_save_dir = None
                if save_character:
                    return_save_dir = f'{self.save_dir}/word_{i_word}/{i_character}'
                    cv2.imwrite(f'{return_save_dir}.png', character)
                is_last_character = i_character == len(segmented_characters) - 1
                is_last_word = i_word == len(segmented_words) - 1
                yield character, is_last_character, is_last_word, return_save_dir
    
    def line_images(self, batched_np):
        """
        Get list of line images, optionally as batched numpy array.
        
        Args:
            batched_np (bool): Whether to return as batched numpy array
            
        Returns:
            List of line images or batched numpy array
        """
        line_images = []
        line_rects = []
        max_line_length = 0
        for line, line_rect in self.image_iterator_on_lines():
            height, width = line.shape[:2]
            aspect_ratio = width / height
            new_width = int(aspect_ratio * self.height)
            resized_line = cv2.resize(line, (new_width, self.height), interpolation=cv2.INTER_LINEAR)
            line_images.append(resized_line)
            line_rects.append(line_rect)
            max_line_length = max(max_line_length, new_width)
        
        if batched_np:
            for i_line_image in range(len(line_images)):
                line_images[i_line_image] = np.concatenate([line_images[i_line_image], np.zeros((self.height, max_line_length - line_images[i_line_image].shape[1]))], axis=1)
            line_images = np.array(line_images)
            print("line_images.shape=", line_images.shape)
        else:
            for i_line_image, line_image in enumerate(line_images):
                line_images[i_line_image] = np.expand_dims(np.array(line_image), axis=0)
        return line_images, line_rects
    
    
    
    def _contour_line_generator(self):
        """Yield tuples of (contour, lines) for each valid text area in the image."""
        for rect in self._find_text_area_rectangles():
            x, y, w, h = rect
            if h < 10 or np.all(self.image[y:y+h, x:x+w] == 0) or np.all(self.image[y:y+h, x:x+w] == 255):
                continue
            contour = self.image[y:y+h, x:x+w]
            lines = self.find_lines(contour, rect)
            yield contour, lines
    
    def image_iterator_on_lines(self):
        """
        Iterator that yields lines of text from the image.
        Yields:
            Image array containing a line of text
            (x,y,w,h) tuple defining rectangle region
        """
        for i_contour, (contour, lines) in enumerate(self._contour_line_generator()):
            if self.save_dir and len(lines) > 0:
                os.makedirs(f'{self.save_dir}/contour_{i_contour}', exist_ok=True)
                cv2.imwrite(f'{self.save_dir}/contour_{i_contour}.png', contour)
            for i_line, (line, line_rect) in enumerate(lines):
                if self.save_dir:
                    cv2.imwrite(f'{self.save_dir}/contour_{i_contour}/line_{i_line}.png', line)
                yield line, line_rect

    def image_iterator_on_words(self):
        """
        Iterator that yields words from lines of text.
        
        Yields:
            Tuple containing:
            - List of character images from a word
            - Bool indicating if last word in line
        """
        for contour, lines in self._contour_line_generator():
            for i_line, (line, line_rect) in enumerate(lines):
                words = self._find_words(line)
                for i_word, word in enumerate(words):
                    characters = self._find_characters(word)
                    is_last_word = i_word == len(words) - 1
                    yield characters, is_last_word

    def draw_rectangles(self, rectangles, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles on a copy of the image.
        
        Args:
            rectangles: List of (x,y,w,h) tuples defining rectangles to draw
            color: BGR color tuple for rectangles (default: green)
            thickness: Line thickness for rectangles (default: 2)
            
        Returns:
            Image array with rectangles drawn
        """
        image_copy = self.image.copy()
        for rect in rectangles:
            x, y, w, h = rect
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
        return image_copy


def main():
    print("Starting inference...")
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("image_path", help="Path to the input image.")
    args = parser.parse_args()
    image_path = args.image_path
    print("Image path: ", image_path)

    # src = 'C:/Users/Joseph Hung/Pictures/data/debug/'
    # image_path = src + 'wiki_asia.png'

    text_extractor = TextExtractor(image_path, save_dir=None, debug=False, verbose=False, plot=False, rgb=True, height=32)

    text_predictor = TextPredictor(os.path.join(os.path.dirname(__file__), 'crnn_cfc_model.pth'))

    images, rects = text_extractor.line_images(batched_np=False)
    results = zip(images, rects)
    start_time = time.time()
    print("Predicting...")
    for image, rect in results:
        predicted_texts, pred_prob_avg = text_predictor.predict_images(image, beam_search=False)
        text = ''.join(predicted_texts)
        print("rect: ", rect, "text: ", text)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    return results

if __name__ == "__main__":
    main()
