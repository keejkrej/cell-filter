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
        
        self._load_images()
        self._process_patterns()
    
    def _load_images(self):
        """Load all image stacks."""
        self.patterns = img_as_ubyte(imread(self.patterns_path))
        
        if self.nuclei_path:
            self.nuclei_stack = imread(self.nuclei_path)
            self.n_frames = len(self.nuclei_stack)
        else:
            self.nuclei_stack = None
        
        if self.cyto_path:
            self.cyto_stack = imread(self.cyto_path)
            if self.nuclei_stack is None:
                self.n_frames = len(self.cyto_stack)
        else:
            self.cyto_stack = None
    
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
    
    def _extract_region(self, stack, contour_idx, frame_idx, use_mask=False):
        """
        Extract a region from the given stack using the specified contour.
        
        Args:
            stack (numpy.ndarray): Image stack to extract from
            contour_idx (int): Index of the contour to use
            frame_idx (int): Frame index in the stack
            use_mask (bool): Whether to apply the mask to the region
            
        Returns:
            numpy.ndarray: The extracted region
        """
        if stack is None:
            raise ValueError("No stack loaded")
            
        contour = self.contours[contour_idx]
        x, y, w, h = self.bounding_boxes[contour_idx]
        
        frame = img_as_ubyte(stack[frame_idx])
        crop = frame[y:y+h, x:x+w]
        
        if use_mask:
            mask = np.zeros_like(self.patterns, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            mask_crop = mask[y:y+h, x:x+w]
            return cv2.bitwise_and(mask_crop, crop)
        
        return crop
    
    def extract_nuclei(self, contour_idx, frame_idx, use_mask=False):
        """
        Extract nuclei from the nuclei stack.
        
        Args:
            contour_idx (int): Index of the contour to use
            frame_idx (int): Frame index in the nuclei stack
            use_mask (bool): Whether to apply the mask to the nuclei
            
        Returns:
            numpy.ndarray: The extracted nuclei
        """
        return self._extract_region(self.nuclei_stack, contour_idx, frame_idx, use_mask)
    
    def extract_cyto(self, contour_idx, frame_idx, use_mask=False):
        """
        Extract cytoplasm from the cytoplasm stack.
        
        Args:
            contour_idx (int): Index of the contour to use
            frame_idx (int): Frame index in the cytoplasm stack
            use_mask (bool): Whether to apply the mask to the cytoplasm
            
        Returns:
            numpy.ndarray: The extracted cytoplasm
        """
        return self._extract_region(self.cyto_stack, contour_idx, frame_idx, use_mask)