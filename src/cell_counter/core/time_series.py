import json
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
from .cell_generator import CellGenerator
from .count import count_nuclei
import numpy as np
import cv2

def analyze_time_series(
    patterns_path: str,
    nuclei_path: str,
    wanted_nuclei: int = 3,
    use_cellpose: bool = True,
    use_gpu: bool = True,
    diameter: int = 15,
    channels: List[int] = [0, 0],
    model_type: str = "cyto3",
    min_intensity: int = 10,
    debug_dir: Path = None,
    debug: bool = False
) -> Dict[int, List[int]]:
    """
    Analyze time series data to find valid frames for each contour based on nuclei counts.
    
    Args:
        patterns_path (str): Path to the patterns image file
        nuclei_path (str): Path to the nuclei image file
        wanted_nuclei (int): Desired number of nuclei per contour
        use_cellpose (bool): Whether to use Cellpose for counting
        use_gpu (bool): Whether to use GPU for Cellpose
        diameter (int): Expected diameter of cells in pixels
        channels (List[int]): Channel indices for Cellpose
        model_type (str): Type of Cellpose model to use
        min_intensity (int): Minimum average intensity for valid regions
        debug_dir (Path): Directory to save problematic frames
        debug (bool): Whether to enable debug output
        
    Returns:
        Dict[int, List[int]]: Dictionary mapping contour indices to lists of valid frame indices
    """
    # Initialize cell generator
    generator = CellGenerator(patterns_path, nuclei_path)
    
    # Initialize time lapse dictionary
    time_lapse: Dict[int, List[int]] = {
        contour_idx: [] for contour_idx in range(generator.n_contours)
    }
    
    # Process each frame
    print(f"\nAnalyzing {generator.n_frames_nuclei} frames...")
    for frame_idx in tqdm(range(generator.n_frames_nuclei)):
        # Load current frame
        generator.load_frame_nuclei(frame_idx)
        
        # Collect all nuclei for this frame
        nuclei_list = []
        contour_indices = list(time_lapse.keys())  # Get current valid contours
        for contour_idx in contour_indices:
            try:
                nuclei = generator.extract_nuclei(contour_idx)
                # Check if the extracted region is empty or too dark
                mean_intensity = np.mean(nuclei)
                if mean_intensity < min_intensity:
                    if debug:
                        print(f"\nWarning: Frame {frame_idx}, contour {contour_idx} appears too dark (mean intensity: {mean_intensity:.2f})")
                        if debug_dir:
                            save_path = debug_dir / f"dark_frame_{frame_idx:03d}_contour_{contour_idx:03d}.png"
                            cv2.imwrite(str(save_path), nuclei)
                    continue
                nuclei_list.append(nuclei)
            except Exception as e:
                if debug:
                    print(f"\nError extracting nuclei for frame {frame_idx}, contour {contour_idx}: {str(e)}")
                continue
        
        if not nuclei_list:
            if debug:
                print(f"\nWarning: No valid nuclei regions found in frame {frame_idx}")
            continue
            
        # Count nuclei for all contours in this frame
        try:
            counts = count_nuclei(
                nuclei_list,
                use_cellpose=use_cellpose,
                use_gpu=use_gpu,
                diameter=diameter,
                channels=channels,
                model_type=model_type
            )
        except Exception as e:
            if debug:
                print(f"\nError counting nuclei in frame {frame_idx}: {str(e)}")
                if debug_dir:
                    for i, nuclei in enumerate(nuclei_list):
                        save_path = debug_dir / f"error_frame_{frame_idx:03d}_contour_{i:03d}.png"
                        cv2.imwrite(str(save_path), nuclei)
            continue
        
        # Update time lapse based on counts
        for contour_idx, n_count in zip(contour_indices, counts):
            if n_count == wanted_nuclei:
                time_lapse[contour_idx].append(frame_idx)
            elif n_count > wanted_nuclei:
                # Remove contour if too many nuclei
                del time_lapse[contour_idx]
                if debug:
                    print(f"\nRemoved contour {contour_idx} at frame {frame_idx} due to too many nuclei ({n_count})")
            elif n_count == 0:
                if debug:
                    print(f"\nWarning: No nuclei found in frame {frame_idx}, contour {contour_idx}")
    
    return time_lapse

def save_time_series(
    time_lapse: Dict[int, List[int]],
    output_path: str,
    patterns_path: str,
    nuclei_path: str,
    wanted_nuclei: int = 3,
    use_cellpose: bool = True,
    use_gpu: bool = True,
    diameter: int = 15,
    channels: List[int] = [0, 0],
    model_type: str = "cyto3"
):
    """
    Save time series analysis results to a JSON file.
    
    Args:
        time_lapse (Dict[int, List[int]]): Time series analysis results
        output_path (str): Path to save the results
        patterns_path (str): Path to the patterns image file
        nuclei_path (str): Path to the nuclei image file
        wanted_nuclei (int): Desired number of nuclei per contour
        use_cellpose (bool): Whether to use Cellpose for counting
        use_gpu (bool): Whether to use GPU for Cellpose
        diameter (int): Expected diameter of cells in pixels
        channels (List[int]): Channel indices for Cellpose
        model_type (str): Type of Cellpose model to use
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON
    data = {
        "metadata": {
            "patterns_path": str(patterns_path),
            "nuclei_path": str(nuclei_path),
            "wanted_nuclei": wanted_nuclei,
            "use_cellpose": use_cellpose,
            "use_gpu": use_gpu,
            "diameter": diameter,
            "channels": channels,
            "model_type": model_type,
            "total_contours": len(time_lapse),
            "total_frames": sum(len(frames) for frames in time_lapse.values())
        },
        "time_lapse": {
            str(contour_idx): frames for contour_idx, frames in time_lapse.items()
        }
    }
    
    # Save to JSON with pretty printing
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
