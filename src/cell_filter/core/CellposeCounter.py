"""
Cellpose-based counter for cell-counter.
"""

from typing import List, Union
import numpy as np
from cellpose import models

class CellposeCounter():
    """
    Counter that uses Cellpose to detect and count nuclei.
    """
    
    def __init__(
        self,
        use_gpu: bool
    ):
        """
        Initialize the Cellpose counter.
        
        Args:
            use_gpu: Whether to use GPU for Cellpose
        """
        self.model = models.Cellpose(
            gpu=use_gpu,
            model_type="cyto3"
        )
    
    def count_nuclei(self, images: Union[np.ndarray, List[np.ndarray]], diameter: int) -> List[int]:
        """
        Count nuclei in one or more images using Cellpose.
        
        Args:
            images: Single image or list of images to count nuclei in
            wanted: Expected number of nuclei
            
        Returns:
            List of nuclei counts for each image
        """
        # Convert single image to list
        if isinstance(images, np.ndarray):
            images = [images]
        
        # Run Cellpose on all images
        masks_list = self.model.eval(
            images,
            diameter=diameter,
            channels=[0, 0]
        )[0]
        
        # Count nuclei in each image
        counts = []
        for masks in masks_list:
            count = np.unique(masks).size - 1 if masks.size > 0 else 0
            counts.append(count)

        return counts