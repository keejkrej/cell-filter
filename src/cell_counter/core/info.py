"""
Core info functionality for cell-counter.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from ..core.cell_generator import CellGenerator


def show_patterns(generator):
    """Show visualization of patterns image and contours/bounding boxes.
    
    Args:
        generator: CellGenerator instance containing patterns image and contours
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original patterns image
    plt.subplot(1, 2, 1)
    plt.imshow(generator.patterns, cmap='gray')
    plt.title('Original Patterns Image')
    plt.axis('off')

    # Plot contours and bounding boxes on black background
    plt.subplot(1, 2, 2)
    vis_img = np.zeros_like(generator.patterns, dtype=np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    
    # Draw contours in green
    cv2.drawContours(vis_img, generator.contours, -1, (0, 255, 0), 2)
    
    # Draw bounding boxes in red and add index numbers
    for idx, bbox in enumerate(generator.bounding_boxes):
        x, y, w, h = bbox
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Add index number
        cv2.putText(vis_img, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
    
    plt.imshow(vis_img)
    plt.title('Contours (green) and Bounding Boxes (red) with Indices')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_image_info(patterns_path, nuclei_path=None, cyto_path=None):
    """Get information about the image stacks.
    
    Args:
        patterns_path: Path to the patterns image file
        nuclei_path: Path to the nuclei image file (optional)
        cyto_path: Path to the cytoplasm image file (optional)
        
    Returns:
        dict: Dictionary containing information about the images
    """
    # Initialize generator
    generator = CellGenerator(
        patterns_path=patterns_path,
        nuclei_path=nuclei_path,
        cyto_path=cyto_path,
    )

    info = {
        'patterns': {
            'path': patterns_path,
            'dimensions': generator.patterns.shape
        }
    }

    # Add nuclei stack info if provided
    if nuclei_path:
        info['nuclei'] = {
            'path': nuclei_path,
            'num_frames': generator.n_frames_nuclei,
            'dimensions_per_frame': generator.patterns.shape  # Use patterns shape as reference
        }

    # Add cytoplasm stack info if provided
    if cyto_path:
        info['cyto'] = {
            'path': cyto_path,
            'num_frames': generator.n_frames_cyto,
            'dimensions_per_frame': generator.patterns.shape  # Use patterns shape as reference
        }

    # Add contours info
    info['contours'] = {
        'total_contours': len(generator.contours),
        'contours_after_filtering': len(generator.contours),
        'contours_filtered_out': 0  # Since we're not currently filtering contours
    }

    return info, generator 