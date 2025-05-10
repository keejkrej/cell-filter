"""
Core cell generator functionality for cell-counter.
"""

import cv2
import numpy as np
from skimage import img_as_ubyte
from nd2reader import ND2Reader
from typing import List, Tuple
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class CellGenerator:
    """
    A class for generating and processing cell data from images.
    
    This class handles the loading and processing of pattern and cell images from ND2 files,
    providing methods to extract and analyze cell regions.
    
    Attributes:
        patterns_path (str): Path to the patterns ND2 file
        cells_path (str): Path to the cell ND2 file
        patterns_reader (ND2Reader): Reader for patterns ND2 file
        cells_reader (ND2Reader): Reader for cells ND2 file
        patterns_metadata (Dict[str, int]): Metadata for patterns file
        cells_metadata (Dict[str, int]): Metadata for cells file
        n_views (int): Number of views in the files
        n_frames (int): Number of frames in the cells file
        current_view (int): Current view index
        current_frame (int): Current frame index
        patterns (Optional[np.ndarray]): Current patterns image
        n_patterns (int): Number of detected patterns
        contours (Optional[List[np.ndarray]]): Pattern contours
        bounding_boxes (Optional[List[Tuple[int, int, int, int]]]): Pattern bounding boxes
        centers (Optional[List[Tuple[int, int]]]): Pattern centers
        frame_nuclei (Optional[np.ndarray]): Current nuclei frame
        frame_cyto (Optional[np.ndarray]): Current cytoplasm frame
        gaussian_blur_size (Tuple[int, int]): Size of Gaussian blur kernel
        bimodal_threshold (float): Threshold for coefficient of variation to determine if areas are bimodal
        min_area_ratio (float): Minimum area ratio to mean for filtering small areas
        max_iterations (int): Maximum number of iterations for iterative area filtering
        std_deviations_for_outliers (int): Number of standard deviations for outlier filtering
        threshold_value (int): Threshold value for binary thresholding
        edge_tolerance (int): Number of pixels to exclude from image edges
        morph_open_size (Tuple[int, int]): Size of kernel for morphological opening
        morph_close_size (Tuple[int, int]): Size of kernel for morphological closing
        nuclei_channel (int): Channel index for nuclei (default: 0)
        cyto_channel (int): Channel index for cytoplasm (default: 1)
    """

    # =====================================================================
    # Constructor and Initialization
    # =====================================================================

    def __init__(
        self, 
        patterns_path: str, 
        cells_path: str,
        gaussian_blur_size: Tuple[int, int] = (11, 11),
        bimodal_threshold: float = 0.1,
        min_area_ratio: float = 0.1,
        max_iterations: int = 10,
        std_deviations_for_outliers: int = 5,
        threshold_value: int = 1,
        edge_tolerance: int = 10,
        morph_open_size: Tuple[int, int] = (3, 3),
        morph_close_size: Tuple[int, int] = (5, 5),
        nuclei_channel: int = 1,
        cyto_channel: int = 0
    ) -> None:
        """
        Initialize the CellGenerator with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns ND2 file
            cells_path (str): Path to the cell ND2 file containing nuclei and cytoplasm channels
            gaussian_blur_size (Tuple[int, int]): Size of Gaussian blur kernel
            bimodal_threshold (float): Threshold for coefficient of variation to determine if areas are bimodal
            min_area_ratio (float): Minimum area ratio to mean for filtering small areas
            max_iterations (int): Maximum number of iterations for iterative area filtering
            std_deviations_for_outliers (int): Number of standard deviations for outlier filtering
            threshold_value (int): Threshold value for binary thresholding
            edge_tolerance (int): Number of pixels to exclude from image edges
            morph_open_size (Tuple[int, int]): Size of kernel for morphological opening (removes small noise)
            morph_close_size (Tuple[int, int]): Size of kernel for morphological closing (fills small holes)
            nuclei_channel (int): Channel index for nuclei (default: 0)
            cyto_channel (int): Channel index for cytoplasm (default: 1)
            
        Raises:
            ValueError: If initialization fails or files are invalid
        """
        self.patterns_path = str(Path(patterns_path).resolve())
        self.cells_path = str(Path(cells_path).resolve())
        self.gaussian_blur_size = gaussian_blur_size
        self.bimodal_threshold = bimodal_threshold
        self.min_area_ratio = min_area_ratio
        self.max_iterations = max_iterations
        self.std_deviations_for_outliers = std_deviations_for_outliers
        self.threshold_value = threshold_value
        self.edge_tolerance = edge_tolerance
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        self.nuclei_channel = nuclei_channel
        self.cyto_channel = cyto_channel
        
        try:
            self._init_patterns()
            self._init_cells()
            self._validate_files()
            logger.debug(f"Successfully initialized CellGenerator with patterns: {self.patterns_path} and cells: {self.cells_path}")
        except Exception as e:
            self.close_files()
            logger.error(f"Error initializing ND2 readers: {e}")
            raise ValueError(f"Error initializing ND2 readers: {e}")
        
        self._init_memory()

    def _init_patterns(self) -> None:
        """Initialize the patterns reader and metadata."""
        try:
            self.patterns_reader = ND2Reader(self.patterns_path)
            self.patterns_metadata = {
                'channels': self.patterns_reader.sizes.get('c', 0),
                'frames': self.patterns_reader.sizes.get('t', 0),
                'views': self.patterns_reader.sizes.get('v', 0),
            }
            logger.debug(f"Patterns metadata: {self.patterns_metadata}")
        except Exception as e:
            logger.error(f"Error initializing patterns reader: {e}")
            raise

    def _init_cells(self) -> None:
        """Initialize the cells reader and metadata."""
        try:
            self.cells_reader = ND2Reader(self.cells_path)
            self.cells_metadata = {
                'channels': self.cells_reader.sizes.get('c', 0),
                'frames': self.cells_reader.sizes.get('t', 0),
                'views': self.cells_reader.sizes.get('v', 0),
            }
            logger.debug(f"Cells metadata: {self.cells_metadata}")
        except Exception as e:
            logger.error(f"Error initializing cells reader: {e}")
            raise

    def _validate_files(self) -> None:
        """
        Validate the ND2 files meet the required specifications.
        
        Raises:
            ValueError: If files don't meet the required specifications
        """
        if self.patterns_metadata['channels'] != 0:
            raise ValueError("Patterns ND2 file shouldn't have any channels")
        # if self.cells_metadata['channels'] != 2:
        #     raise ValueError("Cells ND2 file must contain exactly 2 channels (nuclei and cytoplasm)")
        if self.patterns_metadata['frames'] != 1:
            raise ValueError("Patterns ND2 file must contain exactly 1 frame")
        if self.patterns_metadata['views'] != self.cells_metadata['views']:
            raise ValueError("Patterns and cells ND2 files must contain the same number of views")
        
        self.n_views = self.cells_metadata['views']
        self.n_frames = self.cells_metadata['frames']
        logger.debug(f"Validated files: {self.n_views} views, {self.n_frames} frames")

    def _init_memory(self) -> None:
        """Initialize memory variables to default values."""
        self.current_view = 0
        self.current_frame = 0
        self.patterns = None
        self.n_patterns = 0
        self.thresh = None
        self.contours = None
        self.bounding_boxes = None
        self.centers = None
        self.frame_nuclei = None
        self.frame_cyto = None
        logger.debug("Initialized memory variables")

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in an image using thresholding and contour detection.
        
        Args:
            image (np.ndarray): Input image to find contours in
            
        Returns:
            List[np.ndarray]: List of detected contours
            
        Raises:
            ValueError: If image is None or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")
            
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(image, self.gaussian_blur_size, 0)
        
        # Apply binary thresholding
        _, thresh = cv2.threshold(blur, self.threshold_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones(self.morph_open_size, np.uint8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones(self.morph_close_size, np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"Found {len(contours)} contours in image")
        
        return contours, thresh

    def _refine_contours(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Tuple[int, int, np.ndarray, Tuple[int, int, int, int]]]:
        """
        Refine contours by filtering based on area and calculating centers.
        
        Args:
            contours (List[np.ndarray]): List of contours to refine
            image_shape (Tuple[int, int]): Shape of the image (height, width)
            
        Returns:
            List[Tuple[int, int, np.ndarray, Tuple[int, int, int, int]]]: List of (center_y, center_x, contour, bounding_box)
            
        Raises:
            ValueError: If contours list is empty
        """
        if not contours:
            raise ValueError("No contours provided")
            
        # Calculate areas
        areas = np.array([cv2.contourArea(contour) for contour in contours])
        
        # Iteratively remove small areas until CV falls below threshold
        current_contours = list(contours)
        current_areas = areas.copy()
        
        for iteration in range(self.max_iterations):
            cv = np.std(current_areas) / np.mean(current_areas)
            if cv < self.bimodal_threshold:
                break
                
            # Remove areas smaller than min_area_ratio of mean
            mean_area = np.mean(current_areas)
            min_area = self.min_area_ratio * mean_area
            
            # Create new lists of contours and areas
            new_contours = []
            new_areas = []
            for i, (contour, area) in enumerate(zip(current_contours, current_areas)):
                if area >= min_area:
                    new_contours.append(contour)
                    new_areas.append(area)
            
            # Update for next iteration
            current_contours = new_contours
            current_areas = np.array(new_areas)
            
            if len(current_contours) == 0:
                logger.warning("All contours were removed during iterative filtering")
                break
                
        logger.debug(f"After {iteration + 1} iterations, CV reduced to {cv:.3f}")
        
        # Now filter outliers using standard deviations
        mean_area = np.mean(current_areas)
        std_area = np.std(current_areas)
        lower_bound = mean_area - self.std_deviations_for_outliers * std_area
        upper_bound = mean_area + self.std_deviations_for_outliers * std_area
        
        contour_data = []
        for i, contour in enumerate(current_contours):
            area = current_areas[i]
            if area < lower_bound or area > upper_bound:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too close to edges
            if (x < self.edge_tolerance or 
                y < self.edge_tolerance or 
                x + w > image_shape[1] - self.edge_tolerance or 
                y + h > image_shape[0] - self.edge_tolerance):
                continue
                
            center_x = x + w // 2
            center_y = y + h // 2
            contour_data.append((center_y, center_x, contour, (x, y, w, h)))

        # Sort contours by center
        contour_data.sort(key=lambda x: (x[0], x[1]))
            
        logger.debug(f"Filtered {len(contours)} contours to {len(contour_data)} using iterative area analysis")
        return contour_data
    
    def _extract_region(self, frame: np.ndarray, pattern_idx: int) -> np.ndarray:
        """
        Extract a region from a frame based on pattern index.
        
        Args:
            frame (np.ndarray): Frame to extract region from
            pattern_idx (int): Index of the pattern to extract
            
        Returns:
            np.ndarray: Extracted region
            
        Raises:
            ValueError: If frame is None, pattern index is invalid, or extraction fails
        """
        if frame is None:
            raise ValueError("Frame not provided")
        if pattern_idx >= self.n_patterns or pattern_idx < 0:
            raise ValueError(f"Pattern index {pattern_idx} out of range (0-{self.n_patterns-1})")
        if self.bounding_boxes is None:
            raise ValueError("No bounding boxes provided")
        
        try:
            x, y, w, h = self.bounding_boxes[pattern_idx]
            region = frame[y:y+h, x:x+w]
            return region
        except Exception as e:
            logger.error(f"Error extracting region: {e}")
            raise ValueError(f"Error extracting region: {e}")

    # =====================================================================
    # Public Methods
    # =====================================================================

    def close_files(self) -> None:
        """Safely close all ND2 readers."""
        if hasattr(self, 'patterns_reader'):
            self.patterns_reader.close()
        if hasattr(self, 'cells_reader'):
            self.cells_reader.close()
        logger.debug("Closed all ND2 readers")

    def load_view(self, view_idx: int) -> None:
        """
        Load a specific view from the ND2 files.
        
        Args:
            view_idx (int): Index of the view to load
            
        Raises:
            ValueError: If view index is invalid
        """
        if view_idx >= self.n_views or view_idx < 0:
            raise ValueError(f"View index {view_idx} out of range (0-{self.n_views-1})")
        self.current_view = view_idx
        logger.debug(f"Loaded view {view_idx}")

    def load_patterns(self) -> None:
        """
        Load the patterns from ND2 file.
        
        Raises:
            ValueError: If loading fails
        """
        try:
            self.patterns = img_as_ubyte(self.patterns_reader.get_frame_2D(c=0, t=0, v=self.current_view))
            logger.debug(f"Loaded patterns for view {self.current_view}")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            raise ValueError(f"Error loading patterns: {e}")
    
    def load_nuclei(self, frame_idx: int) -> None:
        """
        Load nuclei frame from ND2 file.
        
        Args:
            frame_idx (int): Index of the frame to load
            
        Raises:
            ValueError: If frame index is invalid or loading fails
        """
        if frame_idx >= self.n_frames:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.n_frames-1})")
        try:
            self.frame_nuclei = img_as_ubyte(self.cells_reader.get_frame_2D(c=self.nuclei_channel, t=frame_idx, v=self.current_view))
            logger.debug(f"Loaded nuclei frame {frame_idx} for view {self.current_view} from channel {self.nuclei_channel}")
        except Exception as e:
            logger.error(f"Error loading nuclei: {e}")
            raise ValueError(f"Error loading nuclei: {e}")
    
    def load_cyto(self, frame_idx: int) -> None:
        """
        Load cytoplasm frame from ND2 file.
        
        Args:
            frame_idx (int): Index of the frame to load
            
        Raises:
            ValueError: If frame index is invalid or loading fails
        """
        if frame_idx >= self.n_frames:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.n_frames-1})")
        try:
            self.frame_cyto = img_as_ubyte(self.cells_reader.get_frame_2D(c=self.cyto_channel, t=frame_idx, v=self.current_view))
            logger.debug(f"Loaded cytoplasm frame {frame_idx} for view {self.current_view} from channel {self.cyto_channel}")
        except Exception as e:
            logger.error(f"Error loading cyto: {e}")
            raise ValueError(f"Error loading cyto: {e}")
        
    def process_patterns(self) -> None:
        """
        Process pattern image to extract contours and their bounding boxes.
        
        Raises:
            ValueError: If patterns haven't been loaded
        """
        if self.patterns is None:
            raise ValueError("Patterns must be loaded before processing")
            
        contours, self.thresh = self._find_contours(self.patterns)
        contour_data = self._refine_contours(contours, self.patterns.shape)
        self.contours = [x[2] for x in contour_data]
        self.bounding_boxes = [x[3] for x in contour_data]
        self.centers = [x[0:2] for x in contour_data]
        self.n_patterns = len(self.contours)
        logger.debug(f"Processed {self.n_patterns} patterns")

    def extract_nuclei(self, pattern_idx: int) -> np.ndarray:
        """
        Extract nuclei region for a specific pattern.
        
        Args:
            pattern_idx (int): Index of the pattern to extract
            
        Returns:
            np.ndarray: Extracted nuclei region
            
        Raises:
            ValueError: If nuclei frame hasn't been loaded
        """
        if self.frame_nuclei is None:
            raise ValueError("Nuclei frame must be loaded before extraction")
        return self._extract_region(self.frame_nuclei, pattern_idx)
    
    def extract_cyto(self, pattern_idx: int) -> np.ndarray:
        """
        Extract cytoplasm region for a specific pattern.
        
        Args:
            pattern_idx (int): Index of the pattern to extract
            
        Returns:
            np.ndarray: Extracted cytoplasm region
            
        Raises:
            ValueError: If cytoplasm frame hasn't been loaded
        """
        if self.frame_cyto is None:
            raise ValueError("Cytoplasm frame must be loaded before extraction")
        return self._extract_region(self.frame_cyto, pattern_idx)

    def extract_pattern(self, pattern_idx: int) -> np.ndarray:
        """
        Extract pattern region.
        
        Args:
            pattern_idx (int): Index of the pattern to extract
            
        Returns:
            np.ndarray: Extracted pattern region
            
        Raises:
            ValueError: If patterns haven't been loaded
        """
        if self.patterns is None:
            raise ValueError("Patterns must be loaded before extraction")
        return self._extract_region(self.patterns, pattern_idx)
