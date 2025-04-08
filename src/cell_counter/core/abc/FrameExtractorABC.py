"""
Abstract base class for frame extraction and data processing functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union
from pathlib import Path
import numpy as np
from .ImageHandlerABC import ImageHandlerABC
from .PatternCropperABC import PatternCropperABC

class FrameExtractorABC(ABC):
    """
    Abstract base class defining the interface for frame extraction and data processing.
    """
    
    @abstractmethod
    def __init__(
        self,
        image_handler: ImageHandlerABC,
        pattern_cropper: PatternCropperABC
    ) -> None:
        """
        Initialize the frame extractor with image handler and pattern cropper.
        
        Args:
            image_handler (ImageHandlerABC): Instance of ImageHandlerABC for accessing images
            pattern_cropper (PatternCropperABC): Instance of PatternCropperABC for pattern processing
        """
        pass
    
    @abstractmethod
    def extract_frame(
        self,
        frame_idx: int,
        save_images: bool = False,
        save_data: bool = False
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Extract data from a specific frame.
        
        Args:
            frame_idx (int): Frame index to extract from
            save_images (bool): Whether to save extracted images
            save_data (bool): Whether to save extracted data
            
        Returns:
            List[Dict[str, Union[str, int, float]]]: List of dictionaries containing extracted data
        """
        pass
    
    @abstractmethod
    def extract_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        save_images: bool = False,
        save_data: bool = False
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Extract data from a range of frames.
        
        Args:
            start_frame (int): Starting frame index
            end_frame (Optional[int]): Ending frame index (inclusive)
            save_images (bool): Whether to save extracted images
            save_data (bool): Whether to save extracted data
            
        Returns:
            List[Dict[str, Union[str, int, float]]]: List of dictionaries containing extracted data
        """
        pass
    
    @abstractmethod
    def save_extracted_data(
        self,
        data: List[Dict[str, Union[str, int, float]]],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save extracted data to a file.
        
        Args:
            data (List[Dict[str, Union[str, int, float]]]): Data to save
            output_path (Union[str, Path]): Path to save the data
        """
        pass 