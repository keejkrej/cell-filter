"""
Core cell generator functionality for cell-counter.
"""

import cv2
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte

class CellGenerator:
    """
    A class for generating and processing cell data from images.
    """
    
    def __init__(self, patterns_path, nuclei_path=None, cyto_path=None, grid_size=20):
        """
        Initialize the CellGenerator with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns image file
            nuclei_path (str, optional): Path to the nuclei image file
            cyto_path (str, optional): Path to the cytoplasm image file
            grid_size (int, optional): Size of the grid for snapping pattern centers (default: 20)
        """
        self.patterns_path = patterns_path
        self.nuclei_path = nuclei_path
        self.cyto_path = cyto_path
        self.grid_size = grid_size
        
        self._load_patterns()
        self._process_patterns()
        
        # Initialize frame storage
        self.frame_nuclei = None
        self.frame_cyto = None
        
        # Load and cache stacks
        if self.nuclei_path:
            self.nuclei_stack = imread(self.nuclei_path)
            self.n_frames_nuclei = len(self.nuclei_stack)
        else:
            self.nuclei_stack = None
            self.n_frames_nuclei = 0
            
        if self.cyto_path:
            self.cyto_stack = imread(self.cyto_path)
            self.n_frames_cyto = len(self.cyto_stack)
        else:
            self.cyto_stack = None
            self.n_frames_cyto = 0
    
    def _load_patterns(self):
        """Load the patterns image."""
        self.patterns = img_as_ubyte(imread(self.patterns_path))
    
    def _process_patterns(self):
        """Process pattern image to extract contours and their bounding boxes."""
        blur = cv2.GaussianBlur(self.patterns, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Add tolerance by eroding the thresholded image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # First pass: collect all areas to calculate statistics
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)
        
        # Calculate mean and standard deviation of areas
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        # Calculate grid cell size based on grid_size parameter
        height, width = self.patterns.shape
        grid_cell_h = height / self.grid_size
        grid_cell_w = width / self.grid_size
        
        # Calculate centers and store with contours and bounding boxes
        contour_data = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if area deviates too much from mean (e.g., more than 2 standard deviations)
            if abs(area - mean_area) > 2 * std_area:
                continue
                
            # Calculate center and snap to nearest grid point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Snap to grid by rounding to nearest grid cell
            grid_x = round(center_x / grid_cell_w)
            grid_y = round(center_y / grid_cell_h)
            
            # Convert back to image coordinates
            snapped_center_x = int(grid_x * grid_cell_w)
            snapped_center_y = int(grid_y * grid_cell_h)
            
            contour_data.append((snapped_center_y, snapped_center_x, contour, (x, y, w, h)))
        
        # Sort by y (row) first, then x within each row
        contour_data.sort(key=lambda x: (x[0], x[1])) 
        
        # Store sorted contours and bounding boxes
        self.contours = []
        self.bounding_boxes = []
        for _, _, contour, bbox in contour_data:
            self.contours.append(contour)
            self.bounding_boxes.append(bbox)
        
        self.n_contours = len(self.contours)
    
    def load_frame_nuclei(self, frame_idx):
        """
        Load a specific frame from the nuclei stack into memory.
        
        Args:
            frame_idx (int): Frame index to load
        """
        if self.nuclei_stack is None:
            raise ValueError("No nuclei stack loaded")

        if frame_idx >= self.n_frames_nuclei:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.n_frames_nuclei-1})")
        self.frame_nuclei = img_as_ubyte(self.nuclei_stack[frame_idx])
    
    def load_frame_cyto(self, frame_idx):
        """
        Load a specific frame from the cytoplasm stack into memory.
        
        Args:
            frame_idx (int): Frame index to load
        """
        if self.cyto_stack is None:
            raise ValueError("No cytoplasm stack loaded")
            
        if frame_idx >= self.n_frames_cyto:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.n_frames_cyto-1})")
            
        self.frame_cyto = img_as_ubyte(self.cyto_stack[frame_idx])
    
    def _extract_region(self, frame, contour_idx):
        """
        Extract a region from the given frame using the specified contour.
        
        Args:
            frame (numpy.ndarray): Frame to extract from
            contour_idx (int): Index of the contour to use
            
        Returns:
            numpy.ndarray: The extracted region
        """
        if frame is None:
            raise ValueError("No frame provided")
            
        contour = self.contours[contour_idx]
        x, y, w, h = self.bounding_boxes[contour_idx]
        
        return frame[y:y+h, x:x+w]
    
    def extract_nuclei(self, contour_idx, threshold=None):
        """
        Extract nuclei from the current nuclei frame.
        
        Args:
            contour_idx (int): Index of the contour to use
            threshold (float, optional): Threshold value for binarization. If None, uses Otsu's method.
            
        Returns:
            numpy.ndarray: The extracted nuclei
        """
        if self.frame_nuclei is None:
            raise ValueError("No nuclei frame loaded")
            
        region = self._extract_region(self.frame_nuclei, contour_idx)
        if threshold is not None:
            _, region = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY)
        return region
    
    def extract_cyto(self, contour_idx, threshold=None):
        """
        Extract cytoplasm from the current cytoplasm frame.
        
        Args:
            contour_idx (int): Index of the contour to use
            threshold (float, optional): Threshold value for binarization. If None, uses Otsu's method.
            
        Returns:
            numpy.ndarray: The extracted cytoplasm
        """
        if self.frame_cyto is None:
            raise ValueError("No cytoplasm frame loaded")
            
        region = self._extract_region(self.frame_cyto, contour_idx)
        if threshold is not None:
            _, region = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY)
        return region

    def extract_pattern(self, contour_idx, threshold=None):
        """
        Extract pattern region for the specified contour.
        
        Args:
            contour_idx (int): Index of the contour to use
            threshold (float, optional): Threshold value for binarization. If None, uses Otsu's method.
            
        Returns:
            numpy.ndarray: The extracted pattern region
        """
        region = self._extract_region(self.patterns, contour_idx)
        if threshold is not None:
            _, region = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY)
        return region 