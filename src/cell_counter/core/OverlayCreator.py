"""
Create RGB overlays from two grayscale images.
"""

import numpy as np
from skimage.io import imread, imsave
from pathlib import Path
import warnings

class OverlayCreator:
    """
    Create RGB overlays from two grayscale images.
    
    The images can be placed in any of the RGB channels (0=red, 1=green, 2=blue).
    By default, nuclei go in the red channel and cytoplasm in the green channel.
    
    Args:
        nuclei_path (str): Path to the nuclei image
        cyto_path (str): Path to the cytoplasm image
        ch_nuclei (int): Channel index (0=red, 1=green, 2=blue) for the nuclei image (default: 0)
        ch_cyto (int): Channel index (0=red, 1=green, 2=blue) for the cytoplasm image (default: 1)
    """
    
    def __init__(self, nuclei_path, cyto_path, ch_nuclei=0, ch_cyto=1):
        # Validate channel indices
        if ch_nuclei not in [0, 1, 2] or ch_cyto not in [0, 1, 2]:
            raise ValueError("Channel indices must be 0, 1, or 2")
        if ch_nuclei == ch_cyto:
            raise ValueError("Channel indices must be different")
            
        self.ch_nuclei = ch_nuclei
        self.ch_cyto = ch_cyto
        
        # Load images
        self.nuclei = imread(nuclei_path)
        self.cyto = imread(cyto_path)
        
        # Ensure both images are stacks
        if self.nuclei.ndim == 2:
            self.nuclei = self.nuclei[np.newaxis, ...]
        if self.cyto.ndim == 2:
            self.cyto = self.cyto[np.newaxis, ...]
            
        # Validate dimensions
        if self.nuclei.shape != self.cyto.shape:
            raise ValueError(f"Image shapes must match. Got {self.nuclei.shape} and {self.cyto.shape}")
            
    def _normalize_intensity(self, img):
        """
        Normalize image intensity to [0, 255] range.
        
        Args:
            img (ndarray): Image stack to normalize
            
        Returns:
            ndarray: Normalized image stack
        """
        # Normalize each frame independently
        normalized = np.zeros_like(img, dtype=np.uint8)
        for i in range(img.shape[0]):
            frame = img[i]
            min_val = frame.min()
            max_val = frame.max()
            if max_val > min_val:  # Avoid division by zero
                normalized[i] = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized[i] = frame.astype(np.uint8)
        return normalized
            
    def create_overlay(self, output_path):
        """
        Create RGB overlay and save to file.
        
        Args:
            output_path (str): Path to save the output TIFF file
        """
        # Normalize intensities
        nuclei_norm = self._normalize_intensity(self.nuclei)
        cyto_norm = self._normalize_intensity(self.cyto)
        
        # Create RGB stack
        rgb_stack = np.zeros((self.nuclei.shape[0], self.nuclei.shape[1], self.nuclei.shape[2], 3), dtype=np.uint8)
        rgb_stack[..., self.ch_nuclei] = nuclei_norm
        rgb_stack[..., self.ch_cyto] = cyto_norm
        
        # Save with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            imsave(output_path, rgb_stack)
