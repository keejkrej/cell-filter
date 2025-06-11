"""
Cellpose-based counter for cell-filter.
"""

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
        self.model = models.CellposeModel(
            gpu=use_gpu,
        )
    
    def count_nuclei(self, images: np.ndarray | list[np.ndarray]) -> list[int]:
        """
        Count nuclei in one or more images using Cellpose.
        
        Args:
            images: Single image or list of images to count nuclei in
            wanted: Expected number of nuclei
            
        Returns:
            list of nuclei counts for each image
        """
        # Convert single image to list
        if isinstance(images, np.ndarray):
            images = [images]
        
        # Run Cellpose on all images
        masks_list, _, _, _ = self.model.eval(
            images,
        )
        
        # Count nuclei in each image
        counts = []
        for masks in masks_list:
            count = np.unique(masks).size - 1 if masks.size > 0 else 0
            counts.append(count)

        return counts