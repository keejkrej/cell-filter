import cv2
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte

class CellGenerator:
    """
    A class for generating and processing cell data from images.
    """
    
    def __init__(self, patterns_path, nuclei_path=None, cyto_path=None):
        """
        Initialize the CellGenerator with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns image file
            nuclei_path (str, optional): Path to the nuclei image file
            cyto_path (str, optional): Path to the cytoplasm image file
        """
        self.patterns_path = patterns_path
        self.nuclei_path = nuclei_path
        self.cyto_path = cyto_path
        
        self._load_patterns()
        self._process_patterns()
        
        # Initialize frame storage
        self.frame_nuclei = None
        self.frame_cyto = None
        
        # Load frame counts
        if self.nuclei_path:
            self.n_frames_nuclei = len(imread(self.nuclei_path))
        else:
            self.n_frames_nuclei = 0
            
        if self.cyto_path:
            self.n_frames_cyto = len(imread(self.cyto_path))
        else:
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
        
        # Calculate centers and store with contours and bounding boxes
        contour_data = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            contour_data.append((center_y, center_x, contour, (x, y, w, h)))
        
        # Sort by y (row) first, then by x within each row
        contour_data.sort()
        
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
        if not self.nuclei_path:
            raise ValueError("No nuclei stack loaded")
            
        if frame_idx >= self.n_frames_nuclei:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.n_frames_nuclei-1})")
            
        stack = imread(self.nuclei_path)
        self.frame_nuclei = img_as_ubyte(stack[frame_idx])
    
    def load_frame_cyto(self, frame_idx):
        """
        Load a specific frame from the cytoplasm stack into memory.
        
        Args:
            frame_idx (int): Frame index to load
        """
        if not self.cyto_path:
            raise ValueError("No cytoplasm stack loaded")
            
        if frame_idx >= self.n_frames_cyto:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.n_frames_cyto-1})")
            
        stack = imread(self.cyto_path)
        self.frame_cyto = img_as_ubyte(stack[frame_idx])
    
    def _extract_region(self, frame, contour_idx, use_mask=False):
        """
        Extract a region from the given frame using the specified contour.
        
        Args:
            frame (numpy.ndarray): Frame to extract from
            contour_idx (int): Index of the contour to use
            use_mask (bool): Whether to apply the mask to the region
            
        Returns:
            numpy.ndarray: The extracted region
        """
        if frame is None:
            raise ValueError("No frame provided")
            
        contour = self.contours[contour_idx]
        x, y, w, h = self.bounding_boxes[contour_idx]
        
        crop = frame[y:y+h, x:x+w]
        
        if use_mask:
            mask = np.zeros_like(self.patterns, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            mask_crop = mask[y:y+h, x:x+w]
            return cv2.bitwise_and(mask_crop, crop)
        
        return crop
    
    def extract_nuclei(self, contour_idx, use_mask=False):
        """
        Extract nuclei from the current nuclei frame.
        
        Args:
            contour_idx (int): Index of the contour to use
            use_mask (bool): Whether to apply the mask to the nuclei
            
        Returns:
            numpy.ndarray: The extracted nuclei
        """
        if self.frame_nuclei is None:
            raise ValueError("No nuclei frame loaded")
        return self._extract_region(self.frame_nuclei, contour_idx, use_mask)
    
    def extract_cyto(self, contour_idx, use_mask=False):
        """
        Extract cytoplasm from the current cytoplasm frame.
        
        Args:
            contour_idx (int): Index of the contour to use
            use_mask (bool): Whether to apply the mask to the cytoplasm
            
        Returns:
            numpy.ndarray: The extracted cytoplasm
        """
        if self.frame_cyto is None:
            raise ValueError("No cytoplasm frame loaded")
        return self._extract_region(self.frame_cyto, contour_idx, use_mask) 