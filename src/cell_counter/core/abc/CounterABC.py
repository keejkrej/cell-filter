"""
Abstract base class for counting nuclei in images.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class CounterABC(ABC):
    """
    Abstract base class for counting nuclei in images.
    """
    
    @abstractmethod
    def count_nuclei(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[int]:
        """
        Count nuclei in one or more images.
        
        Args:
            images: Single image or list of images to count nuclei in
            wanted: Expected number of nuclei
            
        Returns:
            List of nuclei counts for each image
        """
        pass 