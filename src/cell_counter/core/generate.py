import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

from pathlib import Path
import numpy as np
from skimage.io import imsave
from tqdm import tqdm
from .cell_generator import CellGenerator

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

def generate_data(
    patterns_path: str,
    nuclei_path: str = None,
    cyto_path: str = None,
    output_dir: str = None,
    frames: str = None,
    contours: str = None
):
    """
    Generate nuclei and/or cytoplasm data from images.
    
    Args:
        patterns_path (str): Path to the patterns image file
        nuclei_path (str, optional): Path to the nuclei image file
        cyto_path (str, optional): Path to the cytoplasm image file
        output_dir (str): Output directory to save extracted data
        frames (str, optional): Range of frames to process (e.g., "0-5" or "0,2,4")
        contours (str, optional): Range of contours to process (e.g., "0-5" or "0,2,4")
    """
    if not nuclei_path and not cyto_path:
        raise ValueError("At least one of nuclei_path or cyto_path must be provided")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cell generator
    generator = CellGenerator(patterns_path, nuclei_path, cyto_path)
    
    # Determine range of frames and contours to process
    frame_range = parse_range(frames)
    contour_range = parse_range(contours) or range(generator.n_contours)
    
    # If no frame range specified, use the appropriate range based on available data
    if frame_range is None:
        if nuclei_path and cyto_path:
            # Use the smaller range if both are available
            frame_range = range(min(generator.n_frames_nuclei, generator.n_frames_cyto))
        elif nuclei_path:
            frame_range = range(generator.n_frames_nuclei)
        else:
            frame_range = range(generator.n_frames_cyto)
    
    # Process each frame and contour
    total_images = len(frame_range) * len(contour_range)
    print(f"\nProcessing {total_images} images across {len(frame_range)} frames...")
    
    for frame_idx in frame_range:
        # Create frame directory
        frame_dir = output_dir / f"frame_{frame_idx:03d}"
        frame_dir.mkdir(exist_ok=True)
        
        # Create progress bar for this frame
        with tqdm(total=len(contour_range), desc=f"Frame {frame_idx:03d}") as pbar:
            # Load frames if available
            if nuclei_path and frame_idx < generator.n_frames_nuclei:
                generator.load_frame_nuclei(frame_idx)
            if cyto_path and frame_idx < generator.n_frames_cyto:
                generator.load_frame_cyto(frame_idx)
            
            for contour_idx in contour_range:
                if contour_idx >= generator.n_contours:
                    print(f"Warning: Contour {contour_idx} is out of range. Skipping...")
                    continue
                    
                try:
                    # Extract and save nuclei if requested
                    if nuclei_path and frame_idx < generator.n_frames_nuclei:
                        nuclei = generator.extract_nuclei(contour_idx)
                        nuclei_normalized = normalize_intensity(nuclei, min_val=0, max_val=15)
                        nuclei_path = frame_dir / f"nuclei_{frame_idx:03d}_{contour_idx:03d}.png"
                        imsave(nuclei_path, nuclei_normalized)
                    
                    # Extract and save cytoplasm if requested
                    if cyto_path and frame_idx < generator.n_frames_cyto:
                        cyto = generator.extract_cyto(contour_idx)
                        cyto_normalized = normalize_intensity(cyto, min_val=5, max_val=15)
                        cyto_path = frame_dir / f"cyto_{frame_idx:03d}_{contour_idx:03d}.png"
                        imsave(cyto_path, cyto_normalized)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing frame {frame_idx}, contour {contour_idx}: {str(e)}")
                    pbar.update(1) 