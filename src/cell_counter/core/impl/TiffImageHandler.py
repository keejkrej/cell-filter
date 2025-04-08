from pathlib import Path
from typing import Optional
import numpy as np
from ..abc.BaseImageHandlerABC import BaseImageHandlerABC

class TiffImageHandler(BaseImageHandlerABC):
    def __init__(self, pattern_tiff_path: Path, nuclei_tiff_path: Optional[Path]=None, cyto_tiff_path: Optional[Path]=None):
        pass

    def get_n_views(self) -> int:
        pass
    
    def get_n_frames(self) -> int:
        pass
        
    def load_view(self, view_idx: int) -> None:
        pass
    
    def get_patterns(self) -> np.ndarray:
        pass
    
    def get_nuclei_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        pass
    
    def get_cyto_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        pass
    
    def save(
        self,
        image: np.ndarray,
        filename: str,
    ) -> Path:
        pass

    def _load_tiff(self) -> None:
        pass