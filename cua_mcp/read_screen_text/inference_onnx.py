import os
import time
import cv2
import numpy as np
import json
import onnxruntime

onnxruntime.set_default_logger_severity(3)

class TextPredictor:
    def __init__(self, model_path=None):
        self.device = "cpu"
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

# os.environ['PYTHONIOENCODING'] = 'utf-8'


NEXT = 0
PREVIOUS = 1
FIRST_CHILD = 2
PARENT = 3

MIN_LINE_HEIGHT = 10 # Image below this is too blurred to recognize
MIN_LINE_WIDTH = 6

VERTICAL_GAP_THRESHOLD = 6

HEIGHT = 32
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
        
        # Preprocess the image once
        if self.image is not None:
            self._preprocess_image()
        else:
            raise ValueError("Image is None", image_path)
        
        self.copied_image = self.image.copy() if self.debug else None

    def _preprocess_image(self):
        """Preprocess the image once and store results."""
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self._show_images(self.gray, 'Gray', "Preprocessed Gray Image", force=self.plot)
        self.blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self._show_images(self.blurred, 'Blurred', "Preprocessed Blurred Image", force=self.plot)
        self.edges = cv2.Canny(self.blurred, 80, 145)
        self._show_images(self.edges, 'Edges', "Preprocessed Edges Image", force=self.plot)
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
        if self.plot or self.debug or force:
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
    
    
    def _replace_consecutive_with_min(self, numbers: list, values: list):
        """
        Replace consecutive numbers with their minimum value.
        
        Args:
            numbers: List of numbers
            values: List of corresponding values
            
        Returns:
            List with consecutive numbers replaced by the number with minimum value
        """
        debug_zip = [f"{n}:{v}" for n, v in zip(numbers, values)]
        if numbers is None or len(numbers) == 0:
            return []
        new_numbers = []
        current_group = [numbers[0]]
        current_values = [values[0]]
        
        def get_min_idx(values: list):
            first_min_idx = values.index(min(values))
            last_min_idx = len(values) - list(reversed(values)).index(min(values)) - 1
            return (first_min_idx + last_min_idx)//2
        
        for i in range(1, len(numbers)):
            if abs(numbers[i] - numbers[i - 1]) <= 1:
                current_group.append(numbers[i])
                current_values.append(values[i])
            else:
                # Find index of minimum value in current group
                min_idx = get_min_idx(current_values)
                new_numbers.append(current_group[min_idx])
                current_group = [numbers[i]]
                current_values = [values[i]]
                
        # Handle last group
        min_idx = get_min_idx(current_values)
        new_numbers.append(current_group[min_idx])
        
        return new_numbers
    
    def _moving_average_smoothing(self, hist, window_size=5):
        smoothed = []
        half = window_size // 2
        for i in range(len(hist)):
            start = max(0, i - half)
            end = min(len(hist), i + half + 1)
            avg = sum(hist[start:end]) / (end - start)
            smoothed.append(avg)
        return smoothed
    
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

    def _find_consecutive_indices(self, indices, consecutive_count=1):
        """
        Find consecutive indices with minimum consecutive count.
        
        Args:
            indices: List of indices to check
            consecutive_count: Minimum number of consecutive indices required (default: 1)
            
        Returns:
            List of tuples containing (start_index, end_index) pairs where the number of
            consecutive indices between start and end is at least consecutive_count
        """
        if not indices:
            return []
            
        consecutive_pairs = []
        start_idx = indices[0]
        count = 1
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                count += 1
            else:
                if count >= consecutive_count:
                    consecutive_pairs.append((start_idx, indices[i-1]))
                start_idx = indices[i]
                count = 1
                
        # Handle the last sequence
        if count >= consecutive_count:
            consecutive_pairs.append((start_idx, indices[-1]))
            
        return consecutive_pairs
    
    
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
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < MIN_LINE_WIDTH or h < MIN_LINE_HEIGHT:
                # self._show_images(self.image[y:y+h, x:x+w], 'Skipped', f"find_text_area_rectangles => Contour skipped (w={w}, h={h})")
                continue
            self.txt_rectangles.append((x, y, w, h))
            self._show_images(self.image[y:y+h, x:x+w], 'Added', f"find_text_area_rectangles => Contour added (w={w}, h={h})")
            if self.plot or self.debug:
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self._show_images(image_copy, 'Text Area Rectangles', "find_text_area_rectangles => Text Area Rectangles")
        self.txt_rectangles.sort(key=lambda rect: (rect[1], rect[0]))
        return self.txt_rectangles

    # def _detect_and_remove_border(self, image, image_rect):
    #     """
    #     Detect if there is a border around the image and remove it if present.
        
    #     Args:
    #         image (numpy.ndarray): Input image in BGR format (as a NumPy array).
        
    #     Returns:
    #         tuple: (bool, numpy.ndarray)
    #             - bool: True if a border was detected and removed, False otherwise.
    #             - numpy.ndarray: The image with the border removed if detected, otherwise the original image.
    #     """
    #     # Get the coordinates and dimensions of the input image rectangle
    #     x0, y0, w0, h0 = image_rect
        
    #     # Find contours that contain the center point of the image
    #     # This helps identify relevant contours that might form a border
    #     relevant_indices = [i for i, c in enumerate(self.contours) 
    #                     if cv2.pointPolygonTest(c, (x0 + w0/2, y0 + h0/2), False) >= 0]
        
    #     # If no relevant contours found or no hierarchy data, return original image
    #     if len(relevant_indices) == 0 or self.hierarchy is None:
    #         return image, image_rect
        
    #     # Calculate threshold for border area (80% of image area)
    #     img_area_threshold = 0.8 * w0 * h0
        
    #     # Helper function to check if a contour looks like a border
    #     # A border-like contour should:
    #     # - Be very close to the image edges (within 5 pixels)
    #     # - Cover most of the image area (width and height within 10 pixels of image)
    #     # - Have area at least 80% of the image area
    #     def is_border_like(contour):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         return (x - x0 <= 5 and y - y0 <= 5 and w >= w0 - 10 and h >= h0 - 10 and w * h >= img_area_threshold)
        
    #     # Find root contours (those without parents) among relevant contours
    #     hierarchy = self.hierarchy[0]
    #     root_indices = [i for i in relevant_indices if hierarchy[i][PARENT] == -1]
        
    #     # Check each root contour for border-like properties
    #     for idx in root_indices:
    #         if not is_border_like(self.contours[idx]):
    #             continue
    #         # Find child contours (those inside the potential border)
    #         child_indices = [i for i, h in enumerate(hierarchy) if h[PARENT] == idx and i in relevant_indices]
            
    #         if len(child_indices) == 0:
    #             continue
    #         # Calculate total area of child contours
    #         sub_area = sum(cv2.contourArea(self.contours[i]) for i in child_indices)
            
    #         # If child contours occupy at least 25% of the border area,
    #         # consider this a valid border with content inside
    #         if sub_area < 0.25 * cv2.contourArea(self.contours[idx]):
    #             continue
            
    #         # Combine all child contour points to find the bounding box
    #         all_points = np.vstack([self.contours[i] for i in child_indices])
    #         x, y, w, h = cv2.boundingRect(all_points)
            
    #         # Crop the image to remove the border
    #         cropped_image = image[y:y+h, x:x+w]
            
    #         # Skip if cropped image is too small
    #         if cropped_image.shape[0] <= MIN_LINE_HEIGHT or cropped_image.shape[1] <= MIN_LINE_WIDTH:
    #             continue
                
    #         # Show the cropped image if in debug/plot mode
    #         self._show_images(cropped_image, 'Cropped Image', "detect_and_remove_border => Cropped Image")
            
    #         # Return cropped image and updated rectangle coordinates
    #         return cropped_image, (x0 + x, y0 + y, w, h)
        
    #     # If no valid border found, return original image
    #     return image, image_rect
    
    def _remove_surrounded_black_space(self, line_contour, line_rect):
        """
        Remove trailing black space from the text area by analyzing column histogram.
        (Removes black space from the top, bottom, left, right of the line image.)
        
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
    
    
    def _find_lines_in_contour(self, contour_image, contour_rect):
        """
        Find text lines in a contour region.
        
        Args:
            contour: Image array of contour region
            rect: (x,y,w,h) tuple defining rectangle region
            
        Returns:
            List of tuples containing:
            - Image array of text line
            - (x,y,w,h) rectangle coordinates for that line
        """
        # result_image, result_rect = self._detect_and_remove_border(contour, contour_rect)
        result_image, result_rect = contour_image, contour_rect
        if result_image.shape[0] < MIN_LINE_HEIGHT or result_image.shape[1] < MIN_LINE_WIDTH:
            return []
        contours_list = self._find_vertically_sliced_contour(result_image, result_rect)
        output_lines = []
        for sub_contour, sub_contour_rect in contours_list:
            lines = self._find_horizontal_lines_in_contour(sub_contour, sub_contour_rect)
            lines = [self._remove_surrounded_black_space(line, line_rect) for line, line_rect in lines]
            # for i, (line, line_rect) in enumerate(lines):
            #     normalized_area, _ = self._get_normalized_np_arr(line)
            #     if not self._is_perimeter_black(normalized_area):
            #         lines[i] = (cv2.bitwise_not(line), line_rect)
            if self.debug:
                lines_skipped = [(line, line_rect) for line, line_rect in lines if line_rect[2] < MIN_LINE_WIDTH or line_rect[3] < MIN_LINE_HEIGHT]
                for line, line_rect in lines_skipped:
                    self._show_images(line, f'Lines skipped', f"find_lines_in_contour => Lines skipped (w={line_rect[2]}, h={line_rect[3]})")
            lines = [(line, line_rect) for line, line_rect in lines if line_rect[2] >= MIN_LINE_WIDTH and line_rect[3] >= MIN_LINE_HEIGHT]
            output_lines.extend(lines)
        return output_lines
    
    def _find_vertically_sliced_contour(self, contour_image, contour_rect):
        """
        Find text lines in a contour using vertical projection.
        
        Args:
            contour: Image array of contour region
            contour_rect: (x,y,w,h) tuple defining rectangle region
            
        Returns:
            List of tuples containing:
            - Image array of text line
            - (x,y,w,h) rectangle coordinates for that line
        """
        self._show_images(contour_image, 'Contour Image', "find_vertically_sliced_contour => Contour Image before COLOR_BGR2GRAY")
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
        self._show_images(contour_image, 'Contour Image', "find_vertically_sliced_contour => Contour Image COLOR_BGR2GRAY")
        normalized_area, contour_image = self._get_normalized_np_arr(contour_image)
        if not self._is_perimeter_black(normalized_area):
            contour_image = cv2.bitwise_not(contour_image)
        
        T, _ = cv2.threshold(contour_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, contour_image = cv2.threshold(contour_image, T * BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        vertical_histogram = np.sum(contour_image, axis=0)
        vertical_gap_indices = np.where(vertical_histogram <= 0)[0]
        consecutive_indices = self._find_consecutive_indices(list(vertical_gap_indices), consecutive_count=VERTICAL_GAP_THRESHOLD)
        lines_with_rects = []
        x, y, w, h = contour_rect
        
        # If no gaps found, return the entire contour as one line
        if len(consecutive_indices) == 0:
            lines_with_rects.append((contour_image, contour_rect))
            return lines_with_rects
        
        # Add first line from start to first gap
        first_start, first_end = consecutive_indices[0]
        if first_start > 0:
            line_image = contour_image[:, :first_start]
            lines_with_rects.append((line_image, (x, y, first_start, h)))
        
        # Add lines between gaps
        for i in range(len(consecutive_indices)):
            start, end = consecutive_indices[i]
            if i < len(consecutive_indices) - 1:
                next_start, _ = consecutive_indices[i + 1]
                line_image = contour_image[:, end:next_start]
                lines_with_rects.append((line_image, (x + end, y, next_start - end, h)))
        
        # Add last line from last gap to end
        last_start, last_end = consecutive_indices[-1]
        if last_end < contour_image.shape[1] - 1:
            line_image = contour_image[:, last_end:]
            lines_with_rects.append((line_image, (x + last_end, y, contour_image.shape[1] - last_end, h)))
        
        self._show_images([line for line, _ in lines_with_rects], 'Vertical Lines', "find_vertical_lines => Vertical Lines")
        return lines_with_rects
        
    
    def _find_horizontal_lines_in_contour(self, contour_image, contour_rect):
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
        
        horizontal_histogram = np.sum(contour_image, axis=1)
        min_line_height, max_gap_to_merge = 5, 3
        
        gap_indices = np.where(horizontal_histogram <= 0)[0]
        if len(gap_indices) == 0:
            line_indices = [0, contour_image.shape[0]-1]
        else:
            line_indices = np.insert(gap_indices, 0, 0) if gap_indices[0] != 0 else gap_indices
            line_indices = np.append(line_indices, contour_image.shape[0]-1) if line_indices[-1] != contour_image.shape[0]-1 else line_indices
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
        
        lines_with_rects = []
        x, y, w, h = contour_rect
        for i in range(len(merged_indices) - 1):
            line_start, line_end = merged_indices[i], merged_indices[i + 1]
            if line_end - line_start >= min_line_height:
                line_image = contour_image[line_start:line_end, :]
                if line_end - line_start < MIN_LINE_HEIGHT:
                    self._show_images(line_image, 'Line Image', f"find_horizontal_lines_in_contour => Line Image skipped (w={w}, h={h})")
                    continue
                lines_with_rects.append((line_image, (x, y + line_start, w, line_end - line_start)))
        
        self._show_images([line for line, _ in lines_with_rects], 'Lines', "find_lines => Lines")
        return lines_with_rects

    @staticmethod
    def resize_to_height_32(image):
        height, width = image.shape[:2]
        if height==HEIGHT:
            return image
        aspect_ratio = width / height
        new_width = int(aspect_ratio * HEIGHT)
        return cv2.resize(image, (new_width, HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    def line_images(self, batched_np=False):
        """Line crops resized to height 32. Optionally batch-pad to equal width."""
        line_images = []
        line_rects = []
        max_line_length = 0
        for line, line_rect in self.image_iterator_on_lines():

            # Resize and normalize for AI input
            # => Move normalization outside of the model for onnx inference
            resized_line = self.resize_to_height_32(line)
            normalized = (resized_line - np.min(resized_line)) / (
                np.max(resized_line) - np.min(resized_line) + 1e-7
            )
            max_line_length = max(max_line_length, normalized.shape[1])
            line_images.append(normalized)
            line_rects.append(line_rect)

        if batched_np:
            for i in range(len(line_images)):
                pad_w = max_line_length - line_images[i].shape[1]
                if pad_w > 0:
                    line_images[i] = np.concatenate(
                        [
                            line_images[i],
                            np.zeros(
                                (line_images[i].shape[0], pad_w),
                                dtype=line_images[i].dtype,
                            ),
                        ],
                        axis=1,
                    )
            line_images = np.array(line_images)
            print("line_images.shape=", line_images.shape)
        else:
            line_images = [np.expand_dims(li, axis=0) for li in line_images]

        return line_images, line_rects
    
    
    def _contour_line_generator(self):
        """Yield tuples of (contour, lines) for each valid text area in the image."""
        for rect in self._find_text_area_rectangles():
            x, y, w, h = rect
            # if h < MIN_LINE_HEIGHT:# or np.all(self.image[y:y+h, x:x+w] == 0) or np.all(self.image[y:y+h, x:x+w] == 255):
            #     self._show_images(self.image[y:y+h, x:x+w], 'Contour skipped', "contour_line_generator => Contour skipped")
            #     continue
            self._show_images(self.image[y:y+h, x:x+w], 'Contour', f"contour_line_generator => Contour (w={w}, h={h})")
            contour_image = self.image[y:y+h, x:x+w]
            lines = self._find_lines_in_contour(contour_image, rect)
            yield contour_image, lines
    
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
                final_line = self.image[line_rect[1]:line_rect[1]+line_rect[3], line_rect[0]:line_rect[0]+line_rect[2]]
                final_line = cv2.cvtColor(final_line, cv2.COLOR_BGR2GRAY)
                final_line, _ = self._get_normalized_np_arr(final_line)
                if self.save_dir:
                    cv2.imwrite(f'{self.save_dir}/contour_{i_contour}/line_{i_line}.png', final_line)
                yield final_line, line_rect


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
   