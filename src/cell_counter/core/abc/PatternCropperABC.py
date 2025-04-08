"""
Abstract base class for pattern processing and region extraction functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np
from .ImageHandlerABC import ImageHandlerABC

class PatternCropperABC(ABC):
    """
    Abstract base class defining the interface for pattern processing and region extraction.
    """
    
    @abstractmethod
    def __init__(
        self,
        image_handler: ImageHandlerABC,
        grid_size: int = 20
    ) -> None:
        """
        Initialize the pattern cropper with an image handler.
        
        Args:
            image_handler (ImageHandlerABC): Instance of ImageHandlerABC for accessing images
            grid_size (int): Size of the grid for snapping pattern centers (default: 20)
        """
        pass
    
    @abstractmethod
    def process_patterns(self) -> None:
        """
        Process pattern image to extract contours and their bounding boxes.
        """
        pass
    
    @abstractmethod
    def extract_pattern(
        self,
        contour_idx: int,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract pattern region for the specified contour.
        
        Args:
            contour_idx (int): Index of the contour to use
            threshold (Optional[float]): Threshold value for binarization. If None, uses Otsu's method.
            
        Returns:
            np.ndarray: The extracted pattern region
        """
        pass
    
    @abstractmethod
    def extract_nuclei(
        self,
        frame_idx: int,
        contour_idx: int,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract nuclei from the specified frame.
        
        Args:
            frame_idx (int): Frame index to extract from
            contour_idx (int): Index of the contour to use
            threshold (Optional[float]): Threshold value for binarization. If None, uses Otsu's method.
            
        Returns:
            np.ndarray: The extracted nuclei
        """
        pass
    
    @abstractmethod
    def extract_cyto(
        self,
        frame_idx: int,
        contour_idx: int,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract cytoplasm from the specified frame.
        
        Args:
            frame_idx (int): Frame index to extract from
            contour_idx (int): Index of the contour to use
            threshold (Optional[float]): Threshold value for binarization. If None, uses Otsu's method.
            
        Returns:
            np.ndarray: The extracted cytoplasm
        """
        pass
    
    @abstractmethod
    def get_contour(self, contour_idx: int) -> np.ndarray:
        """
        Get the contour at the specified index.
        
        Args:
            contour_idx (int): Index of the contour to get
            
        Returns:
            np.ndarray: The contour
        """
        pass
    
    @abstractmethod
    def get_bounding_box(self, contour_idx: int) -> Tuple[int, int, int, int]:
        """
        Get the bounding box for the contour at the specified index.
        
        Args:
            contour_idx (int): Index of the contour to get bounding box for
            
        Returns:
            Tuple[int, int, int, int]: (x, y, width, height) of the bounding box
        """
        pass
    
    @abstractmethod
    def get_contour_center(self, contour_idx: int) -> Tuple[int, int]:
        """
        Get the center coordinates for the contour at the specified index.
        
        Args:
            contour_idx (int): Index of the contour to get center for
            
        Returns:
            Tuple[int, int]: (x, y) coordinates of the center
        """
        pass
    
    @abstractmethod
    def get_contour_area(self, contour_idx: int) -> float:
        """
        Get the area of the contour at the specified index.
        
        Args:
            contour_idx (int): Index of the contour to get area for
            
        Returns:
            float: Area of the contour
        """
        pass
    
    @abstractmethod
    def get_contour_mask(self, contour_idx: int) -> np.ndarray:
        """
        Get a binary mask for the contour at the specified index.
        
        Args:
            contour_idx (int): Index of the contour to get mask for
            
        Returns:
            np.ndarray: Binary mask of the contour
        """
        pass 