"""
Core info functionality for cell-counter.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from .CellGenerator import CellGenerator

class InfoDisplayer:
    """
    A class for displaying information about image stacks and visualizing patterns.
    """
    
    def __init__(
        self,
        patterns_path: str,
        nuclei_path: str = None,
        cyto_path: str = None,
        grid_size: int = 20
    ):
        """
        Initialize the InfoDisplayer with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns image file
            nuclei_path (str, optional): Path to the nuclei image file
            cyto_path (str, optional): Path to the cytoplasm image file
            grid_size (int, optional): Size of the grid for snapping pattern centers (default: 20)
        """
        self.generator = CellGenerator(
            patterns_path=patterns_path,
            nuclei_path=nuclei_path,
            cyto_path=cyto_path,
            grid_size=grid_size
        )
        self.patterns_path = patterns_path
        self.nuclei_path = nuclei_path
        self.cyto_path = cyto_path
        self.grid_size = grid_size

    def show_patterns(self) -> None:
        """Show visualization of patterns image and contours/bounding boxes."""
        plt.figure(figsize=(12, 8))
        
        # Plot original patterns image
        plt.subplot(1, 2, 1)
        plt.imshow(self.generator.patterns, cmap='gray')
        plt.title('Original Patterns Image')
        plt.axis('off')

        # Plot contours and bounding boxes on black background
        plt.subplot(1, 2, 2)
        vis_img = np.zeros_like(self.generator.patterns, dtype=np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        
        # Draw contours in green
        cv2.drawContours(vis_img, self.generator.contours, -1, (0, 255, 0), 2)
        
        # Draw bounding boxes in red and add index numbers
        for idx, bbox in enumerate(self.generator.bounding_boxes):
            x, y, w, h = bbox
            # Draw bounding box
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Add index number with larger font size and thickness
            cv2.putText(vis_img, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255), 2)
        
        plt.imshow(vis_img)
        plt.title('Contours (green) and Bounding Boxes (red) with Indices')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def get_info(self) -> dict:
        """
        Get information about the image stacks.
        
        Returns:
            dict: Dictionary containing information about the images
        """
        info = {
            'patterns': {
                'path': self.patterns_path,
                'dimensions': self.generator.patterns.shape,
                'grid_size': self.grid_size
            }
        }

        # Add nuclei stack info if provided
        if self.nuclei_path:
            info['nuclei'] = {
                'path': self.nuclei_path,
                'num_frames': self.generator.n_frames_nuclei,
                'dimensions_per_frame': self.generator.patterns.shape  # Use patterns shape as reference
            }

        # Add cytoplasm stack info if provided
        if self.cyto_path:
            info['cyto'] = {
                'path': self.cyto_path,
                'num_frames': self.generator.n_frames_cyto,
                'dimensions_per_frame': self.generator.patterns.shape  # Use patterns shape as reference
            }

        # Add contours info
        info['contours'] = {
            'total_contours': len(self.generator.contours),
            'contours_after_filtering': len(self.generator.contours),
            'contours_filtered_out': 0  # This will be updated if any contours are filtered
        }

        return info