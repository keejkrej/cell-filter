"""
Abstract base class for image handling functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Literal
from pathlib import Path
import numpy as np

class ImageHandlerABC(ABC):
    """
    Abstract base class defining the interface for image handling.
    """
    
    @abstractmethod
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
            
        Note:
            Either (nuclei_path and cyto_path) or nd2_path should be provided, but not both.
            If nd2_path is provided, the implementation will handle extracting the appropriate channels.
            
        Required Attributes:
            patterns (np.ndarray): Loaded patterns image
            nuclei (Optional[np.ndarray]): Loaded nuclei stack
            cyto (Optional[np.ndarray]): Loaded cytoplasm stack
            nd2_reader (Optional[ND2Reader]): ND2 file reader if using ND2 format
            n_frames_nuclei (int): Number of frames in nuclei stack
            n_frames_cyto (int): Number of frames in cytoplasm stack
            image_shape (Tuple[int, int]): Expected shape of images (height, width)
            mode (Literal['tiff', 'nd2']): Operating mode of the handler, either 'tiff' for separate files or 'nd2' for ND2 file
        """
        pass
    
    # ============= Shared Methods =============
    
    @abstractmethod
    def get_patterns(self) -> np.ndarray:
        """
        Get the patterns image.
        
        Returns:
            np.ndarray: Patterns image
        """
        pass
    
    @abstractmethod
    def get_n_views(self) -> int:
        """
        Get the number of views available.
        
        Returns:
            int: Number of views (1 for TIFF mode, may be more for ND2 mode)
        """
        pass
    
    @abstractmethod
    def get_n_frames(self) -> int:
        """
        Get the number of frames available.
        
        Returns:
            int: Number of frames in both nuclei and cytoplasm stacks
            
        Raises:
            ValueError: If nuclei and cytoplasm stacks have different numbers of frames
        """
        pass
    
    @abstractmethod
    def get_nuclei_frame(self, frame_idx: int, view_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the nuclei stack.
        
        Args:
            frame_idx (int): Frame index
            view_idx (int): View index
            
        Returns:
            Optional[np.ndarray]: Nuclei frame if loaded, None otherwise
            
        Note:
            If nd2_path was provided during initialization, this will return the appropriate channel
            from the ND2 file.
            
        Raises:
            ValueError: If view_idx is out of range (>= get_n_views())
        """
        pass
    
    @abstractmethod
    def get_cyto_frame(self, frame_idx: int, view_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the cytoplasm stack.
        
        Args:
            frame_idx (int): Frame index
            view_idx (int): View index
            
        Returns:
            Optional[np.ndarray]: Cytoplasm frame if loaded, None otherwise
            
        Note:
            If nd2_path was provided during initialization, this will return the appropriate channel
            from the ND2 file.
            
        Raises:
            ValueError: If view_idx is out of range (>= get_n_views())
        """
        pass
    
    @abstractmethod
    def save(
        self,
        image: np.ndarray,
        filename: str,
        normalize: bool = True,
        overwrite: bool = False
    ) -> Path:
        """
        Save an image or image stack.
        
        Args:
            image (np.ndarray): Image or image stack to save
            filename (str): Name of the output file
            normalize (bool): Whether to normalize intensity
            overwrite (bool): Whether to overwrite existing file
            
        Returns:
            Path: Path to the saved image/stack
        """
        pass
    
    # ============= TIFF-specific Methods =============
    
    @abstractmethod
    def _load_patterns(self) -> None:
        """
        Load the patterns image.
        This is an implementation detail and should not be called directly.
        """
        pass
    
    @abstractmethod
    def _load_nuclei_stack(self) -> None:
        """
        Load the nuclei image stack.
        This is an implementation detail and should not be called directly.
        """
        pass
    
    @abstractmethod
    def _load_cyto_stack(self) -> None:
        """
        Load the cytoplasm image stack.
        This is an implementation detail and should not be called directly.
        """
        pass
    
    @abstractmethod
    def _get_tiff_frame(self, frame_idx: int, channel: Literal['nuclei', 'cyto']) -> Optional[np.ndarray]:
        """
        Protected method to get a specific frame from a TIFF stack.
        This is an implementation detail and should not be called directly.
        
        Args:
            frame_idx (int): Frame index
            channel (Literal['nuclei', 'cyto']): Channel to get frame from
            
        Returns:
            Optional[np.ndarray]: Frame from TIFF stack if loaded, None otherwise
            
        Raises:
            ValueError: If channel is not 'nuclei' or 'cyto'
        """
        pass
    
    # ============= ND2-specific Methods =============
    
    @abstractmethod
    def _load_nd2(self) -> None:
        """
        Load the ND2 file.
        This is an implementation detail and should not be called directly.
        """
        pass
    
    @abstractmethod
    def _get_nd2_frame(self, frame_idx: int, channel: int = 0, view_idx: int = 0) -> Optional[np.ndarray]:
        """
        Protected method to get a specific frame from the ND2 file.
        This is an implementation detail and should not be called directly.
        
        Args:
            frame_idx (int): Frame index
            channel (int): Channel index (0 for nuclei, 1 for cytoplasm)
            view_idx (int): View index
            
        Returns:
            Optional[np.ndarray]: Frame from ND2 file if loaded, None otherwise
        """
        pass