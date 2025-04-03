import cv2
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte

class CellGenerator:
    """
    A class for generating and processing cell data from images.
    """
    
    def __init__(self, patterns_path, cells_path):
        """
        Initialize the CellGenerator with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns image file
            cells_path (str): Path to the cells image file
        """
        self.patterns_path = patterns_path
        self.cells_path = cells_path
        
        # Load and process images
        self._load_and_process_images()
    
    def _load_and_process_images(self):
        """Load and process all images and extract contours."""
        # Load images
        self.img_patterns = img_as_ubyte(imread(self.patterns_path))
        self.cells_stack = imread(self.cells_path)
        self.num_frames = len(self.cells_stack)
        
        # Process pattern image to extract contours
        blur = cv2.GaussianBlur(self.img_patterns, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.num_contours = len(self.contours)
    
    def extract_cell(self, contour_idx, frame_idx, use_mask=False):
        """
        Extract a cell from the cells stack.
        
        Args:
            contour_idx (int): Index of the contour to use
            frame_idx (int): Frame index in the cells stack
            use_mask (bool): Whether to apply the mask to the cell
            
        Returns:
            numpy.ndarray: The extracted cell
        """
        # Get contour and its bounding box
        contour = self.contours[contour_idx]
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract cell from frame
        img_cells = img_as_ubyte(self.cells_stack[frame_idx])
        cell_crop = img_cells[y:y+h, x:x+w]
        
        if use_mask:
            # Create and apply mask
            mask = np.zeros_like(self.img_patterns, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            mask_crop = mask[y:y+h, x:x+w]
            return cv2.bitwise_and(mask_crop, cell_crop)
        
        return cell_crop