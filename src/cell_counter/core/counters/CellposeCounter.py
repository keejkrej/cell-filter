"""
Cellpose-based counter for cell-counter.
"""

from typing import List, Union
from matplotlib import pyplot as plt
import numpy as np
from cellpose import models
from .Counter import Counter

class CellposeCounter(Counter):
    """
    Counter that uses Cellpose to detect and count nuclei.
    """
    
    def __init__(
        self,
        diameter: int = 15,
        channels: str = "0,0",
        model_type: str = "cyto3",
        use_gpu: bool = True
    ):
        """
        Initialize the Cellpose counter.
        
        Args:
            diameter: Expected diameter of cells in pixels
            channels: Channel indices for Cellpose
            model_type: Type of Cellpose model to use
            use_gpu: Whether to use GPU for Cellpose
        """
        self.model = models.Cellpose(
            gpu=use_gpu,
            model_type=model_type
        )
        self.diameter = diameter
        self.channels = [int(x) for x in channels.split(",")]
    
    def count_nuclei(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[int]:
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
        masks_list, _, _, _ = self.model.eval(
            images,
            diameter=self.diameter,
            channels=self.channels
        )
        
        # Count nuclei in each image
        counts = []
        for masks in masks_list:
            count = np.unique(masks).size - 1 if masks.size > 0 else 0
            counts.append(count)

        return counts