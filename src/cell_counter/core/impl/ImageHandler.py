"""
Implementation of the ImageHandler interface.
"""

from typing import Optional, Union, Tuple, Literal
from pathlib import Path
import numpy as np
from nd2 import ND2Reader
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity

from cell_counter.core.abc.ImageHandlerABC import ImageHandlerABC

class ImageHandler(ImageHandlerABC):
    """
    Implementation of the ImageHandler interface for handling both TIFF and ND2 files.
    """
    
    def __init__(
        self,
        patterns_path: Union[str, Path],
        nuclei_path: Optional[Union[str, Path]] = None,
        cyto_path: Optional[Union[str, Path]] = None,
        nd2_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the image handler with paths to images and output directory.
        
        Args:
            patterns_path (Union[str, Path]): Path to patterns image
            nuclei_path (Optional[Union[str, Path]]): Path to nuclei stack
            cyto_path (Optional[Union[str, Path]]): Path to cytoplasm stack
            nd2_path (Optional[Union[str, Path]]): Path to ND2 file containing both nuclei and cytoplasm
            output_dir (Optional[Union[str, Path]]): Directory to save output images
        """
        # Convert paths to Path objects
        self.patterns_path = Path(patterns_path)
        self.nuclei_path = Path(nuclei_path) if nuclei_path else None
        self.cyto_path = Path(cyto_path) if cyto_path else None
        self.nd2_path = Path(nd2_path) if nd2_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Initialize attributes
        self.patterns: np.ndarray
        self.nuclei: Optional[np.ndarray] = None
        self.cyto: Optional[np.ndarray] = None
        self.nd2_reader: Optional[ND2Reader] = None
        self.n_frames_nuclei: int = 0
        self.n_frames_cyto: int = 0
        self.image_shape: Tuple[int, int] = (0, 0)
        
        # Determine mode and load appropriate files
        if nd2_path and (nuclei_path or cyto_path):
            raise ValueError("Cannot provide both ND2 path and separate file paths")
        
        # Always load patterns first
        self._load_patterns()
        
        if nd2_path:
            self.mode: Literal['tiff', 'nd2'] = 'nd2'
            self._load_nd2()
        else:
            self.mode = 'tiff'
            if nuclei_path and cyto_path:
                self._load_nuclei_stack()
                self._load_cyto_stack()
            else:
                raise ValueError("Must provide either ND2 path or both nuclei and cyto paths")
    
    # ============= Shared Methods =============
    
    def get_patterns(self) -> np.ndarray:
        """Get the patterns image."""
        if self.patterns is None:
            raise ValueError("Patterns not loaded - initialization error")
        return self.patterns
    
    def get_n_views(self) -> int:
        """Get the number of views available."""
        if self.mode == 'tiff':
            return 1
        else:
            if self.nd2_reader is None:
                raise ValueError("ND2 reader not initialized")
            return self.nd2_reader.sizes.get('V', 1)
    
    def get_n_frames(self) -> int:
        """Get the number of frames available."""
        if self.n_frames_nuclei == 0 and self.n_frames_cyto == 0:
            raise ValueError("No frames loaded - neither nuclei nor cytoplasm stacks are initialized")
            
        # If only one stack is loaded, use that count
        if self.n_frames_nuclei == 0:
            return self.n_frames_cyto
        if self.n_frames_cyto == 0:
            return self.n_frames_nuclei
            
        # If both are loaded, they must match
        if self.n_frames_nuclei != self.n_frames_cyto:
            raise ValueError(
                f"Nuclei and cytoplasm stacks have different numbers of frames: "
                f"nuclei={self.n_frames_nuclei}, cyto={self.n_frames_cyto}"
            )
        return self.n_frames_nuclei
    
    def get_nuclei_frame(self, frame_idx: int, view_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame from the nuclei stack."""
        if view_idx >= self.get_n_views():
            raise ValueError(f"View index {view_idx} out of range (max: {self.get_n_views() - 1})")
        
        if self.mode == 'tiff':
            return self._get_tiff_frame(frame_idx, 'nuclei')
        else:
            return self._get_nd2_frame(frame_idx, channel=0, view_idx=view_idx)
    
    def get_cyto_frame(self, frame_idx: int, view_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame from the cytoplasm stack."""
        if view_idx >= self.get_n_views():
            raise ValueError(f"View index {view_idx} out of range (max: {self.get_n_views() - 1})")
        
        if self.mode == 'tiff':
            return self._get_tiff_frame(frame_idx, 'cyto')
        else:
            return self._get_nd2_frame(frame_idx, channel=1, view_idx=view_idx)
    
    def save(
        self,
        image: np.ndarray,
        filename: str,
        normalize: bool = True,
        overwrite: bool = False
    ) -> Path:
        """Save an image or image stack."""
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image must be a numpy array, got {type(image)}")
            
        if self.output_dir is None:
            raise ValueError("Output directory not set")
        
        output_path = self.output_dir / filename
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"File {output_path} already exists")
        
        if normalize:
            image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        
        imsave(output_path, image)
        return output_path
    
    # ============= TIFF-specific Methods =============
    
    def _load_patterns(self) -> None:
        """Load the patterns image."""
        if self.patterns_path is None:
            raise ValueError("Patterns path not set")
        
        self.patterns = imread(self.patterns_path)
        if self.patterns.ndim != 2:
            raise ValueError("Patterns image must be 2D")
        
        if self.image_shape == (0, 0):
            self.image_shape = self.patterns.shape
        elif self.patterns.shape != self.image_shape:
            raise ValueError(
                f"Patterns image shape {self.patterns.shape} does not match "
                f"expected shape {self.image_shape}"
            )
    
    def _load_nuclei_stack(self) -> None:
        """Load the nuclei image stack."""
        if self.nuclei_path is None:
            raise ValueError("Nuclei path not set")
        
        self.nuclei = imread(self.nuclei_path)
        if self.nuclei.ndim != 3:
            raise ValueError("Nuclei stack must be 3D")
        
        self.n_frames_nuclei = self.nuclei.shape[0]
        if self.image_shape == (0, 0):
            self.image_shape = self.nuclei.shape[1:]
        elif self.nuclei.shape[1:] != self.image_shape:
            raise ValueError(
                f"Nuclei stack shape {self.nuclei.shape[1:]} does not match "
                f"expected shape {self.image_shape}"
            )
    
    def _load_cyto_stack(self) -> None:
        """Load the cytoplasm image stack."""
        if self.cyto_path is None:
            raise ValueError("Cyto path not set")
        
        self.cyto = imread(self.cyto_path)
        if self.cyto.ndim != 3:
            raise ValueError("Cyto stack must be 3D")
        
        self.n_frames_cyto = self.cyto.shape[0]
        if self.image_shape == (0, 0):
            self.image_shape = self.cyto.shape[1:]
        elif self.cyto.shape[1:] != self.image_shape:
            raise ValueError(
                f"Cyto stack shape {self.cyto.shape[1:]} does not match "
                f"expected shape {self.image_shape}"
            )
    
    def _get_tiff_frame(self, frame_idx: int, channel: Literal['nuclei', 'cyto']) -> Optional[np.ndarray]:
        """Get a specific frame from a TIFF stack."""
        if channel not in ['nuclei', 'cyto']:
            raise ValueError(f"Invalid channel: {channel}")
        
        stack = self.nuclei if channel == 'nuclei' else self.cyto
        if stack is None:
            return None
        
        if frame_idx < 0 or frame_idx >= stack.shape[0]:
            raise ValueError(f"Frame index {frame_idx} out of range (max: {stack.shape[0] - 1})")
        
        return stack[frame_idx]
    
    # ============= ND2-specific Methods =============
    
    def _load_nd2(self) -> None:
        """Load the ND2 file."""
        if self.nd2_path is None:
            raise ValueError("ND2 path not set")
        
        self.nd2_reader = ND2Reader(self.nd2_path)
        
        # Get frame counts from metadata
        self.n_frames_nuclei = self.nd2_reader.sizes.get('T', 0)
        self.n_frames_cyto = self.n_frames_nuclei  # ND2 files have same number of frames for both channels
        
        # Get image shape from first frame
        if self.n_frames_nuclei > 0:
            frame = self.nd2_reader.get_frame_2D(c=0, t=0, v=0)
            self.image_shape = frame.shape
    
    def _get_nd2_frame(self, frame_idx: int, channel: int = 0, view_idx: int = 0) -> Optional[np.ndarray]:
        """Get a specific frame from the ND2 file."""
        if self.nd2_reader is None:
            raise ValueError("ND2 reader not initialized")
        
        if frame_idx < 0 or frame_idx >= self.n_frames_nuclei:
            raise ValueError(f"Frame index {frame_idx} out of range (max: {self.n_frames_nuclei - 1})")
        
        if view_idx < 0 or view_idx >= self.get_n_views():
            raise ValueError(f"View index {view_idx} out of range (max: {self.get_n_views() - 1})")
        
        return self.nd2_reader.get_frame_2D(c=channel, t=frame_idx, v=view_idx) 