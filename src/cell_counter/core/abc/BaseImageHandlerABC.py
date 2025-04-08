"""
Abstract base class for image handling functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Literal
from pathlib import Path
import numpy as np

class BaseImageHandlerABC(ABC):
    """
    Abstract base class defining the interface for image handling.
    """
    
    @abstractmethod
    def get_n_views(self) -> int:
        pass
    
    @abstractmethod
    def get_n_frames(self) -> int:
        pass
        
    @abstractmethod
    def load_view(self, view_idx: int) -> None:
        pass
    
    @abstractmethod
    def get_patterns(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_nuclei_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    def get_cyto_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    def save(
        self,
        image: np.ndarray,
        filename: str,
    ) -> Path:
        pass