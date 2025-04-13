"""
Core cell generator functionality for cell-counter.
"""

import cv2
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte
from nd2reader import ND2Reader

class CellGenerator:
    """
    A class for generating and processing cell data from images.
    """
    
    def __init__(self, patterns_path, cells_path, grid_size=20):
        """
        Initialize the CellGenerator with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns ND2 file
            cells_path (str): Path to the cell ND2 file containing nuclei and cytoplasm channels
            grid_size (int, optional): Size of the grid for snapping pattern centers (default: 20)
        """
        self.patterns_path = patterns_path
        self.cells_path = cells_path
        self.grid_size = grid_size
        
        # Initialize frame storage
        self.frame_nuclei = None
        self.frame_cyto = None
        
        # Initialize ND2 reader and get metadata
        self.cell_reader = ND2Reader(self.cells_path)
        self.cell_metadata = {
            'channels': self.cell_reader.sizes.get('c', 1),
            'frames': self.cell_reader.sizes.get('t', 1),
            'views': self.cell_reader.sizes.get('v', 1),
            'z_slices': self.cell_reader.sizes.get('z', 1)
        }
            
        if self.cell_metadata['channels'] != 2:
            self.cell_reader.close()
            raise ValueError("Cell ND2 file must contain exactly 2 channels (nuclei and cytoplasm)")
        
        self.current_view = 0
        self.current_z = 0
    
    def __del__(self):
        """Clean up by closing the ND2 reader when the object is destroyed."""
        if hasattr(self, 'cell_reader'):
            self.cell_reader.close()
    
    def load_patterns(self, view_idx):
        """Load the patterns from ND2 file."""
        with ND2Reader(self.patterns_path) as images:
            # Take the first frame/view/channel if multiple exist
            self.patterns = img_as_ubyte(images[0])
    
    def process_patterns(self):
        """Process pattern image to extract contours and their bounding boxes."""
        blur = cv2.GaussianBlur(self.patterns, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
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
    
    def load_frame_nuclei(self, frame_idx, view_idx=None, z_idx=None):
        """
        Load a specific frame from the nuclei channel into memory.
        
        Args:
            frame_idx (int): Frame index to load
            view_idx (int, optional): View index to load (defaults to current view)
            z_idx (int, optional): Z-slice index to load (defaults to current z)
        """
        if frame_idx >= self.cell_metadata['frames']:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.cell_metadata['frames']-1})")
            
        view = self.current_view if view_idx is None else view_idx
        z = self.current_z if z_idx is None else z_idx
        
        if view >= self.cell_metadata['views']:
            raise ValueError(f"View index {view} out of range (0-{self.cell_metadata['views']-1})")
        if z >= self.cell_metadata['z_slices']:
            raise ValueError(f"Z index {z} out of range (0-{self.cell_metadata['z_slices']-1})")
            
        self.frame_nuclei = img_as_ubyte(self.cell_reader.get_frame_2D(c=0, t=frame_idx, v=view, z=z))
        
        if view_idx is not None:
            self.current_view = view_idx
        if z_idx is not None:
            self.current_z = z_idx
    
    def load_frame_cyto(self, frame_idx, view_idx=None, z_idx=None):
        """
        Load a specific frame from the cytoplasm channel into memory.
        
        Args:
            frame_idx (int): Frame index to load
            view_idx (int, optional): View index to load (defaults to current view)
            z_idx (int, optional): Z-slice index to load (defaults to current z)
        """
        if frame_idx >= self.cell_metadata['frames']:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.cell_metadata['frames']-1})")
            
        view = self.current_view if view_idx is None else view_idx
        z = self.current_z if z_idx is None else z_idx
        
        if view >= self.cell_metadata['views']:
            raise ValueError(f"View index {view} out of range (0-{self.cell_metadata['views']-1})")
        if z >= self.cell_metadata['z_slices']:
            raise ValueError(f"Z index {z} out of range (0-{self.cell_metadata['z_slices']-1})")
            
        self.frame_cyto = img_as_ubyte(self.cell_reader.get_frame_2D(c=1, t=frame_idx, v=view, z=z))
        
        if view_idx is not None:
            self.current_view = view_idx
        if z_idx is not None:
            self.current_z = z_idx
    
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