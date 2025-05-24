"""
Core pattern displayer functionality for cell-filter.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional
import logging
from .generate import CellGenerator, CellGeneratorParameters

# Configure logging
logger = logging.getLogger(__name__)

class PatternDisplayer:
    """
    A class for displaying patterns.
    
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
        cells_path: str,
        nuclei_channel: int,
        cyto_channel: int
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
            self.generator = CellGenerator(
                patterns_path,
                cells_path,
                CellGeneratorParameters(
                    nuclei_channel=nuclei_channel,
                    cyto_channel=cyto_channel
                )
            )
            self.n_views = self.generator.pattern_views
            logger.info(f"Successfully initialized PatternDisplayer with patterns: {patterns_path} and cells: {cells_path}")
        except Exception as e:
            logger.error(f"Error initializing PatternDisplayer: {e}")
            raise ValueError(f"Error initializing PatternDisplayer: {e}")

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
        
        if self.generator.bounding_boxes is None:
            return image
            
        for pattern_idx in range(self.generator.n_patterns):
            # Get bounding box coordinates
            bbox = self.generator.bounding_boxes[pattern_idx]
            if bbox is not None:
                x, y, w, h = bbox
                
                # Draw bounding box
                cv2.rectangle(image, 
                            (int(x), int(y)), 
                            (int(x + w), int(y + h)), 
                            (0, 255, 0),  # Green color
                            2)  # Line thickness
                
                # Add pattern index
                cv2.putText(image, 
                          f"{pattern_idx}", 
                          (int(x), int(y) - 10),  # Position above the box
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
            if self.generator.thresh is None:
                raise ValueError("Threshold image not available")
            patterns_image = np.copy(self.generator.thresh)
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