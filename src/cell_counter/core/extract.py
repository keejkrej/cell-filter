import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
from .cell_generator import CellGenerator

def extract_valid_frames(
    patterns_path: str,
    nuclei_path: str,
    time_series_path: str,
    output_dir: str,
    min_frames: int = 10
):
    """
    Extract valid frames for each contour based on time series analysis results.
    
    Args:
        patterns_path (str): Path to the patterns image file
        nuclei_path (str): Path to the nuclei image file
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
    
    # Initialize cell generator
    generator = CellGenerator(patterns_path, nuclei_path)
    
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
                generator.load_frame_nuclei(frame_idx)
                nuclei = generator.extract_nuclei(contour_idx)
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

def main():
    """Main entry point for extract command."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract valid frames for each contour based on time series analysis')
    
    # Required arguments
    parser.add_argument('--patterns', type=str, required=True, help='Path to the patterns image file')
    parser.add_argument('--nuclei', type=str, required=True, help='Path to the nuclei image file')
    parser.add_argument('--time-series', type=str, required=True, help='Path to the time series analysis JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for extracted frames')
    
    # Optional arguments
    parser.add_argument('--min-frames', type=int, default=10, help='Minimum number of valid frames required (default: 10)')
    
    args = parser.parse_args()
    
    extract_valid_frames(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        time_series_path=args.time_series,
        output_dir=args.output,
        min_frames=args.min_frames
    )

if __name__ == '__main__':
    main() 