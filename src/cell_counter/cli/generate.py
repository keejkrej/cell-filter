"""
Generate command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Generate nuclei data
    python -m cell_counter.cli.generate --patterns <patterns_path> --nuclei <nuclei_path> --output <output_dir>
    
    # Generate cytoplasm data
    python -m cell_counter.cli.generate --patterns <patterns_path> --cyto <cyto_path> --output <output_dir>
    
    # Generate both
    python -m cell_counter.cli.generate --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path> --output <output_dir>
    
    Optional arguments:
    --frames: Range of frames to process (e.g., "0-5" for frames 0 to 5, "0,2,4" for specific frames)
    --contours: Range of contours to process (e.g., "0-5" for contours 0 to 5, "0,2,4" for specific contours).
                 Contours are sorted in row-major order (left to right, top to bottom).
    
    Example:
    python -m cell_counter.cli.generate --patterns /path/to/patterns.tif --nuclei /path/to/nuclei.tif --cyto /path/to/cyto.tif --output /path/to/output_dir --frames 0-5 --contours 0-10
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

import argparse
from pathlib import Path
import numpy as np
from skimage.io import imsave
from tqdm import tqdm
from cell_counter.data.cell_generator import CellGenerator

def parse_range(range_str):
    """Parse range string into a list of indices."""
    if not range_str:
        return None
    
    try:
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            return range(start, end + 1)
        elif ',' in range_str:
            return [int(x) for x in range_str.split(',')]
        else:
            return [int(range_str)]
    except ValueError:
        raise ValueError("Invalid range format. Use 'start-end' or comma-separated values")

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
    parser = argparse.ArgumentParser(description='Generate nuclei and/or cytoplasm data from images')
    parser.add_argument('--patterns', type=str, required=True, help='Path to the patterns image file')
    parser.add_argument('--nuclei', type=str, help='Path to the nuclei image file')
    parser.add_argument('--cyto', type=str, help='Path to the cytoplasm image file')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save extracted data')
    parser.add_argument('--frames', type=str, help='Range of frames to process (e.g., "0-5" or "0,2,4")')
    parser.add_argument(
        "--contours",
        type=str,
        help="Range of contours to process (e.g., '0-5' for contours 0 to 5, '0,2,4' for specific contours). "
             "Contours are sorted in row-major order (left to right, top to bottom).",
    )
    
    args = parser.parse_args()
    
    if not args.nuclei and not args.cyto:
        parser.error("At least one of --nuclei or --cyto must be provided")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cell generator
    generator = CellGenerator(args.patterns, args.nuclei, args.cyto)
    
    # Determine range of frames and contours to process
    try:
        frame_range = parse_range(args.frames) or range(generator.n_frames)
        contour_range = parse_range(args.contours) or range(generator.n_contours)
    except ValueError as e:
        parser.error(f"Invalid range: {str(e)}")
    
    # If either range is not specified, ask for confirmation
    if not args.frames or not args.contours:
        print(f"\nWarning: {'Frame' if not args.frames else 'Contour'} range not specified.")
        if not args.frames:
            print(f"This will process all {generator.n_frames} frames.")
        if not args.contours:
            print(f"This will process all {generator.n_contours} contours.")
        print(f"Total number of images to generate: {len(frame_range) * len(contour_range)}")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Process each frame and contour
    total_images = len(frame_range) * len(contour_range)
    print(f"\nProcessing {total_images} images across {len(frame_range)} frames...")
    
    for frame_idx in frame_range:
        if frame_idx >= generator.n_frames:
            print(f"Warning: Frame {frame_idx} is out of range. Skipping...")
            continue
            
        # Create frame directory
        frame_dir = output_dir / f"frame_{frame_idx:03d}"
        frame_dir.mkdir(exist_ok=True)
        
        # Create progress bar for this frame
        with tqdm(total=len(contour_range), desc=f"Frame {frame_idx:03d}") as pbar:
            for contour_idx in contour_range:
                if contour_idx >= generator.n_contours:
                    print(f"Warning: Contour {contour_idx} is out of range. Skipping...")
                    continue
                    
                try:
                    # Extract and save nuclei if requested
                    if args.nuclei:
                        nuclei = generator.extract_nuclei(contour_idx, frame_idx)
                        nuclei_normalized = normalize_intensity(nuclei, min_val=4, max_val=15)
                        nuclei_path = frame_dir / f"nuclei_{frame_idx:03d}_{contour_idx:03d}.png"
                        imsave(nuclei_path, nuclei_normalized)
                    
                    # Extract and save cytoplasm if requested
                    if args.cyto:
                        cyto = generator.extract_cyto(contour_idx, frame_idx)
                        cyto_normalized = normalize_intensity(cyto, min_val=4, max_val=15)
                        cyto_path = frame_dir / f"cyto_{frame_idx:03d}_{contour_idx:03d}.png"
                        imsave(cyto_path, cyto_normalized)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing frame {frame_idx}, contour {contour_idx}: {str(e)}")
                    pbar.update(1)

if __name__ == '__main__':
    main() 