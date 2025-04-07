import json
from pathlib import Path
import numpy as np
from skimage.io import imsave
import warnings
from .CellGenerator import CellGenerator

class Extractor:
    """
    A class for extracting valid frames from time series analysis results.
    """
    
    def __init__(
        self,
        patterns_path: str,
        nuclei_path: str = None,
        cyto_path: str = None,
        grid_size: int = 20
    ):
        """
        Initialize the Extractor with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns image file
            nuclei_path (str, optional): Path to the nuclei image file
            cyto_path (str, optional): Path to the cytoplasm image file
            grid_size (int): Size of the grid for snapping pattern centers (default: 20)
        """
        self.generator = CellGenerator(patterns_path, nuclei_path, cyto_path, grid_size=grid_size)
        self.patterns_path = patterns_path
        self.nuclei_path = nuclei_path
        self.cyto_path = cyto_path
        self.grid_size = grid_size

    def extract_valid_frames(
        self,
        time_series_path: str,
        output_dir: str,
        min_frames: int = 20,
        image_type: str = "nuclei"
    ) -> None:
        """
        Extract valid frames for each contour based on time series analysis results.
        
        Args:
            time_series_path (str): Path to the time series analysis JSON file
            output_dir (str): Directory to save extracted frames
            min_frames (int): Minimum number of valid frames required for extraction (default: 20)
            image_type (str): Type of image to extract ("nuclei" or "cyto")
        """
        if image_type not in ["nuclei", "cyto"]:
            raise ValueError("image_type must be either 'nuclei' or 'cyto'")
            
        if image_type == "nuclei" and not self.nuclei_path:
            raise ValueError("Nuclei path not provided")
        elif image_type == "cyto" and not self.cyto_path:
            raise ValueError("Cytoplasm path not provided")
        
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
        print(f"\nExtracting {image_type} frames for contours with at least {min_frames} valid frames...")
        for contour_idx, frames in time_lapse.items():
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
                    if image_type == "nuclei":
                        self.generator.load_frame_nuclei(frame_idx)
                        image = self.generator.extract_nuclei(contour_idx)
                    else:
                        self.generator.load_frame_cyto(frame_idx)
                        image = self.generator.extract_cyto(contour_idx)
                    stack.append(image)
                except Exception as e:
                    print(f"\nError extracting frame {frame_idx} for contour {contour_idx}: {str(e)}")
                    continue
            
            if not stack:
                continue
                
            # Convert stack to numpy array
            stack = np.array(stack)
            
            # Save stack with warning filter
            output_path = output_dir / f"{image_type}_{contour_idx:03d}.tif"
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*is a low contrast image.*')
                imsave(output_path, stack)
            
            print(f"\nSaved {len(stack)} {image_type} frames for contour {contour_idx} to {output_path}")

    def extract_patterns(
        self,
        time_series_path: str,
        output_dir: str,
        min_frames: int = 20
    ) -> None:
        """
        Extract pattern regions for each contour based on time series analysis results.
        
        Args:
            time_series_path (str): Path to the time series analysis JSON file
            output_dir (str): Directory to save extracted patterns
            min_frames (int): Minimum number of valid frames required for extraction (default: 20)
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
        print(f"\nExtracting patterns for contours with at least {min_frames} valid frames...")
        for contour_idx, frames in time_lapse.items():
            if len(frames) < min_frames:
                continue
                
            # Extract pattern
            try:
                pattern = self.generator.extract_pattern(contour_idx)
                
                # Create a stack with the same pattern repeated for each frame
                stack = np.array([pattern] * len(frames))
                
                # Save stack with warning filter
                output_path = output_dir / f"pattern_{contour_idx:03d}.tif"
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*is a low contrast image.*')
                    imsave(output_path, stack)
                
                print(f"\nSaved pattern for contour {contour_idx} to {output_path}")
            except Exception as e:
                print(f"\nError extracting pattern for contour {contour_idx}: {str(e)}")
                continue 