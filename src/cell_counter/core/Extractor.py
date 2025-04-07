import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
from .CellGenerator import CellGenerator

class Extractor:
    """
    A class for extracting valid frames from time series analysis results.
    """
    
    def __init__(
        self,
        patterns_path: str,
        nuclei_path: str,
        grid_size: int = 20
    ):
        """
        Initialize the Extractor with paths to pattern and nuclei images.
        
        Args:
            patterns_path (str): Path to the patterns image file
            nuclei_path (str): Path to the nuclei image file
            grid_size (int): Size of the grid for snapping pattern centers (default: 20)
        """
        self.generator = CellGenerator(patterns_path, nuclei_path, grid_size=grid_size)
        self.patterns_path = patterns_path
        self.nuclei_path = nuclei_path
        self.grid_size = grid_size

    def extract_valid_frames(
        self,
        time_series_path: str,
        output_dir: str,
        min_frames: int = 10
    ) -> None:
        """
        Extract valid frames for each contour based on time series analysis results.
        
        Args:
            time_series_path (str): Path to the time series analysis JSON file
            output_dir (str): Directory to save extracted frames
            min_frames (int): Minimum number of valid frames required for extraction
        """
        # Load time series analysis results
        with open(time_series_path, 'r') as f:
            data = json.load(f)
        
        time_lapse = {
            int(contour_idx): frames for contour_idx, frames in data['time_lapse'].items()
        }
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each contour
        print(f"\nExtracting frames for contours with at least {min_frames} valid frames...")
        for contour_idx, frames in tqdm(time_lapse.items()):
            if len(frames) < min_frames:
                continue
                
            # Sort frames to ensure chronological order
            frames.sort()
            
            # Initialize stack for this contour
            stack = []
            
            # Extract frames
            for frame_idx in frames:
                try:
                    # Load and extract frame
                    self.generator.load_frame_nuclei(frame_idx)
                    nuclei = self.generator.extract_nuclei(contour_idx)
                    stack.append(nuclei)
                except Exception as e:
                    print(f"\nError extracting frame {frame_idx} for contour {contour_idx}: {str(e)}")
                    continue
            
            if not stack:
                continue
                
            # Convert stack to numpy array
            stack = np.array(stack)
            
            # Save stack
            output_path = output_dir / f"contour_{contour_idx:03d}.tif"
            imsave(output_path, stack)
            
            print(f"\nSaved {len(stack)} frames for contour {contour_idx} to {output_path}") 