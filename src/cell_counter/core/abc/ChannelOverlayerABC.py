"""
Abstract base class for channel overlaying functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np
from .ImageHandlerABC import ImageHandlerABC

class ChannelOverlayerABC(ABC):
    """
    Abstract base class defining the interface for channel overlaying.
    """
    
    @abstractmethod
    def __init__(
        self,
        image_handler: ImageHandlerABC,
        nuclei_color: Tuple[int, int, int] = (255, 0, 0),
        cyto_color: Tuple[int, int, int] = (0, 255, 0),
        pattern_color: Tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        """
        Initialize the channel overlayer with an image handler and color settings.
        
        Args:
            image_handler (ImageHandlerABC): Instance of ImageHandlerABC for accessing images
            nuclei_color (Tuple[int, int, int]): RGB color for nuclei channel (default: red)
            cyto_color (Tuple[int, int, int]): RGB color for cytoplasm channel (default: green)
            pattern_color (Tuple[int, int, int]): RGB color for pattern channel (default: blue)
        """
        pass
    
    @abstractmethod
    def create_overlay(
        self,
        frame_idx: int,
        save: bool = False,
        filename: Optional[str] = None
    ) -> np.ndarray:
        """
        Create an overlay of channels for a specific frame.
        
        Args:
            frame_idx (int): Frame index to create overlay for
            save (bool): Whether to save the overlay
            filename (Optional[str]): Name of the output file if saving
            
        Returns:
            np.ndarray: The overlay image
        """
        pass
    
    @abstractmethod
    def create_overlays(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        save: bool = False,
        filename_pattern: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Create overlays for a range of frames.
        
        Args:
            start_frame (int): Starting frame index
            end_frame (Optional[int]): Ending frame index (inclusive)
            save (bool): Whether to save the overlays
            filename_pattern (Optional[str]): Pattern for output filenames if saving
            
        Returns:
            List[np.ndarray]: List of overlay images
        """
        pass 