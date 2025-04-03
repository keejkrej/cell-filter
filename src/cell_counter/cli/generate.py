"""
Generate command for cell-counter.
"""

import argparse
from pathlib import Path
import numpy as np
from skimage.io import imsave
from cell_counter.data.cell_generator import CellGenerator

def normalize_intensity(image, min_val=0, max_val=15):
    """Normalize image intensity to 0-255 range."""
    if not min_val and not max_val:
        min_val = np.min(image)
        max_val = np.max(image)
    if max_val == min_val:
        return image
    normalized = ((image - min_val) / (max_val - min_val) * 255)
    normalized = np.clip(normalized, 0, 255)
    normalized = normalized.astype(np.uint8)
    return normalized

def main():
    """Main entry point for generate command."""
    parser = argparse.ArgumentParser(description='Generate cell data from images')
    parser.add_argument('--patterns', type=str, required=True, help='Path to the patterns image file')
    parser.add_argument('--cells', type=str, required=True, help='Path to the cells image file')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save extracted cells')
    parser.add_argument('--test', action='store_true', help='Run in test mode (first frame, first 10 contours)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cell generator
    generator = CellGenerator(args.patterns, args.cells)
    
    # Determine range of frames and contours to process
    if args.test:
        frame_range = range(1)  # First frame only
        contour_range = range(generator.num_contours)
    else:
        frame_range = range(generator.num_frames)
        contour_range = range(generator.num_contours)
    
    # Process each frame and contour
    for frame_idx in frame_range:
        # Create frame directory
        frame_dir = output_dir / f"frame_{frame_idx:03d}"
        frame_dir.mkdir(exist_ok=True)
        
        for contour_idx in contour_range:
            try:
                # Extract cell
                cell = generator.extract_cell(contour_idx, frame_idx)
                
                # Normalize intensity
                cell_normalized = normalize_intensity(cell, min_val=4, max_val=15)
                
                # Save cell as PNG file
                cell_path = frame_dir / f"cell_{contour_idx:03d}.png"
                imsave(cell_path, cell_normalized)
                
                print(f"Saved cell: {cell_path}")
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}, contour {contour_idx}: {str(e)}")

if __name__ == '__main__':
    main() 