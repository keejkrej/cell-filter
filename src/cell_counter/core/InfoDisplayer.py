"""
Core info displayer functionality for cell-counter.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Optional, Tuple
import logging
from .CellGenerator import CellGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfoDisplayer:
    """
    A class for displaying information about patterns.
    
    This class provides functionality to visualize the patterns image for a specific view,
    with bounding boxes and pattern indices overlaid on top.
    
    Attributes:
        generator (CellGenerator): Cell generator instance
        patterns_path (str): Path to the patterns ND2 file
        cells_path (str): Path to the cells ND2 file
    """

    # =====================================================================
    # Constructor and Initialization
    # =====================================================================

    def __init__(
        self,
        patterns_path: str,
        cells_path: str
    ) -> None:
        """
        Initialize the InfoDisplayer with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns ND2 file
            cells_path (str): Path to the cells ND2 file containing nuclei and cytoplasm channels
            
        Raises:
            ValueError: If initialization fails
        """
        try:
            self.generator = CellGenerator(patterns_path, cells_path)
            self.patterns_path = patterns_path
            self.cells_path = cells_path
            logger.info(f"Successfully initialized InfoDisplayer with patterns: {patterns_path} and cells: {cells_path}")
        except Exception as e:
            logger.error(f"Error initializing InfoDisplayer: {e}")
            raise ValueError(f"Error initializing InfoDisplayer: {e}")

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _draw_boxes(self, image: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes and indices for all patterns.
        
        Args:
            image (np.ndarray): Original patterns image
            
        Returns:
            np.ndarray: Image with bounding boxes and indices drawn
        """
        # Convert to RGB for colored annotations
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        for pattern_idx in range(self.generator.n_patterns):
            # Get contour
            contour = self.generator.get_contour(pattern_idx)
            if contour is not None:
                # Get bounding box coordinates
                y_min, x_min = np.min(contour, axis=0)
                y_max, x_max = np.max(contour, axis=0)
                
                # Draw bounding box
                cv2.rectangle(image, 
                            (int(x_min), int(y_min)), 
                            (int(x_max), int(y_max)), 
                            (0, 255, 0),  # Green color
                            2)  # Line thickness
                
                # Add pattern index
                cv2.putText(image, 
                          f"{pattern_idx}", 
                          (int(x_min), int(y_min) - 10),  # Position above the box
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5,  # Font scale
                          (0, 255, 0),  # Green color
                          2)  # Line thickness
        
        return image

    # =====================================================================
    # Public Methods
    # =====================================================================

    def plot_view(self, view_idx: int, output_path: Optional[str] = None) -> None:
        """
        Plot the patterns image for a specific view with bounding boxes and indices.
        
        Args:
            view_idx (int): Index of the view to plot
            output_path (Optional[str]): Path to save the plot (if None, display plot)
            
        Raises:
            ValueError: If view index is invalid or plotting fails
        """
        try:
            # Load view and patterns
            self.generator.load_view(view_idx)
            self.generator.load_patterns()
            self.generator.process_patterns()
            
            # Create figure
            fig = plt.figure(figsize=(15, 8))
            ax = plt.gca()
            
            # Get patterns image and draw boxes
            patterns_image = self.generator.patterns.copy()
            annotated_image = self._draw_boxes(patterns_image)
            
            # Plot annotated image
            ax.imshow(annotated_image)
            
            # Set title
            ax.set_title(f"View {view_idx} - Patterns with Bounding Boxes")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show plot
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved plot to {output_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting view {view_idx}: {e}")
            raise ValueError(f"Error plotting view {view_idx}: {e}")
        finally:
            plt.close()

    def close(self) -> None:
        """Close all open files."""
        self.generator.close_files()
        logger.info("Closed all files")