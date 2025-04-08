"""
Abstract base class for time series analysis functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union
from pathlib import Path
import numpy as np
from .ImageHandlerABC import ImageHandlerABC
from .PatternCropperABC import PatternCropperABC

class SeriesAnalyzerABC(ABC):
    """
    Abstract base class defining the interface for time series analysis.
    """
    
    @abstractmethod
    def __init__(
        self,
        image_handler: ImageHandlerABC,
        pattern_cropper: PatternCropperABC
    ) -> None:
        """
        Initialize the series analyzer with image handler and pattern cropper.
        
        Args:
            image_handler (ImageHandlerABC): Instance of ImageHandlerABC for accessing images
            pattern_cropper (PatternCropperABC): Instance of PatternCropperABC for pattern processing
        """
        pass
    
    @abstractmethod
    def analyze_frame(
        self,
        frame_idx: int,
        save_results: bool = False
    ) -> Dict[str, Union[str, int, float]]:
        """
        Analyze a specific frame.
        
        Args:
            frame_idx (int): Frame index to analyze
            save_results (bool): Whether to save analysis results
            
        Returns:
            Dict[str, Union[str, int, float]]: Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def analyze_series(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        save_results: bool = False
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Analyze a range of frames.
        
        Args:
            start_frame (int): Starting frame index
            end_frame (Optional[int]): Ending frame index (inclusive)
            save_results (bool): Whether to save analysis results
            
        Returns:
            List[Dict[str, Union[str, int, float]]]: List of dictionaries containing analysis results
        """
        pass
    
    @abstractmethod
    def save_time_series(
        self,
        results: List[Dict[str, Union[str, int, float]]],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save time series analysis results to a file.
        
        Args:
            results (List[Dict[str, Union[str, int, float]]]): Analysis results to save
            output_path (Union[str, Path]): Path to save the results
        """
        pass 