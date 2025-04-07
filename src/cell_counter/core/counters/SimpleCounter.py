"""
Simple thresholding-based counter for cell-counter.
"""

from typing import List, Union
import numpy as np
from skimage import measure
from .Counter import Counter

class SimpleCounter(Counter):
    """
    Counter that uses simple thresholding to detect and count nuclei.
    """
    
    def count_nuclei(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[int]:
        """
        Count nuclei in one or more images using simple thresholding.
        
        Args:
            images: Single image or list of images to count nuclei in
            wanted: Expected number of nuclei
            
        Returns:
            List of nuclei counts for each image
        """
        # Convert single image to list
        if isinstance(images, np.ndarray):
            images = [images]
        
        counts = []
        for image in images:
            # Apply threshold
            thresh = np.mean(image) + 2 * np.std(image)
            binary = image > thresh
            
            # Find connected components
            labels = measure.label(binary)
            
            # Count unique labels (excluding background)
            count = int(np.max(labels)) if labels.size > 0 else 0
            counts.append(count)
        
        return counts 