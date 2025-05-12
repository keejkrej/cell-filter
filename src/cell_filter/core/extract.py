"""
Core extractor functionality for cell-filter.
"""

import json
from pathlib import Path
import numpy as np
from imageio.v3 import imwrite
import warnings
from .generate import CellGenerator, CellGeneratorParameters
import logging
from typing import Dict, List

# Configure logging
logger = logging.getLogger(__name__)

class Extractor:
    """
    A class for extracting valid frames and patterns from time series analysis results.
    
    This class handles the extraction of time lapse sequences from analysis results,
    including refining time lapses, adding head/tail frames, and saving the results
    in a structured directory format.
    
    Attributes:
        generator (CellGenerator): Cell generator instance
        patterns_path (str): Path to the patterns ND2 file
        cells_path (str): Path to the cells ND2 file
        output_folder (str): Path to save extracted frames
    """

    # =====================================================================
    # Constructor and Initialization
    # =====================================================================

    def __init__(
        self,
        patterns_path: str,
        cells_path: str,
        output_folder: str,
        nuclei_channel: int,
        cyto_channel: int
    ) -> None:
        """
        Initialize the Extractor with paths to pattern and cell images.
        
        Args:
            patterns_path (str): Path to the patterns ND2 file
            cells_path (str): Path to the cells ND2 file containing nuclei and cytoplasm channels
            output_folder (str): Path to save extracted frames
            
        Raises:
            ValueError: If initialization fails
        """
        self.patterns_path = str(Path(patterns_path).resolve())
        self.cells_path = str(Path(cells_path).resolve())
        self.output_folder = str(Path(output_folder).resolve())
        try:
            self.generator = CellGenerator(
                patterns_path,
                cells_path,
                parameters=CellGeneratorParameters(
                    nuclei_channel=nuclei_channel,
                    cyto_channel=cyto_channel
                )
            )
            logger.info(f"Successfully initialized Extractor with patterns: {patterns_path} and cells: {cells_path}")
        except Exception as e:
            logger.error(f"Error initializing Extractor: {e}")
            raise ValueError(f"Error initializing Extractor: {e}")

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _refine_time_series(self, time_series: Dict[int, List[int]], min_frames:int ) -> Dict[int, List[int]]:
        """
        Refine the time lapse dictionary by applying modifications to the frame indices.
        Splits time lapses when gaps between consecutive frames are larger than 6.
        Fills in missing frames when gaps are small (≤6) to create continuous sequences.
        Only stores the start and end frames of each sequence.
        Pattern indices are formatted with leading zeros (e.g., 000, 001).
        
        Args:
            time_series (Dict[int, List[int]]): Dictionary mapping pattern indices to frame indices
            
        Returns:
            Dict[int, List[int]]: Refined time lapse dictionary with split sequences,
                                 each sequence represented by [start_frame, end_frame]
        """
        MAX_GAP = 6
        refined_time_series = {}
        
        for pattern_idx, frames in time_series.items():
            if not frames:
                continue
                
            # Sort frames to ensure chronological order
            frames.sort()
            
            # Initialize variables for current sequence
            sequence_start = frames[0]
            current_end = frames[0]
            sequences = []
            
            # Process frames to find gaps
            for i in range(1, len(frames)):
                gap = frames[i] - frames[i-1]
                if gap > MAX_GAP:
                    # Large gap found, save current sequence and start new one
                    sequences.append([sequence_start, current_end])
                    sequence_start = frames[i]
                    current_end = frames[i]
                else:
                    current_end = frames[i]
            
            # Add the last sequence
            sequences.append([sequence_start, current_end])
            
            # Add sequences to refined time lapse with new pattern indices
            for i, (start, end) in enumerate(sequences):
                # Format pattern index with leading zeros (3 digits)
                if end - start <= min_frames:
                    continue
                new_pattern_idx = int(f"{pattern_idx:03d}{i:03d}")
                refined_time_series[new_pattern_idx] = [start, end]
                
                logger.debug(f"Split pattern {pattern_idx} into {len(sequences)} sequences")
                logger.debug(f"Sequence {i}: frames {start} to {end}")
                
        return refined_time_series

    def _add_head_tail(self, time_series: Dict[int, List[int]], n_frames: int = 3) -> Dict[int, List[int]]:
        """
        Add extra frames at the beginning and end of each sequence for better inspection.
        
        Args:
            time_series (Dict[int, List[int]]): Dictionary mapping pattern indices to [start_frame, end_frame]
            n_frames (int): Number of extra frames to add at each end (default: 3)
            
        Returns:
            Dict[int, List[int]]: Time lapse dictionary with added head and tail frames
        """
        extended_time_series = {}
        
        # Get total number of frames from the generator
        total_frames = self.generator.n_frames
        
        for pattern_idx, (start, end) in time_series.items():
            # Add head frames (before start)
            new_start = max(0, start - n_frames)
            
            # Add tail frames (after end), but don't exceed total_frames
            new_end = min(total_frames - 1, end + n_frames)
            
            extended_time_series[pattern_idx] = [new_start, new_end]
            
            logger.debug(f"Extended pattern {pattern_idx} with {start-new_start} head frames and {new_end-end} tail frames")
            
        return extended_time_series
    
    def _extract_frame_stack(self,
                             pattern_idx: int,
                             start_frame: int,
                             end_frame: int,
                             frame_output_path: Path,
                             json_output_path: Path) -> None:
        """
        Extract and save frame stack for a given pattern.
        """
        # Initialize stacks for nuclei and cytoplasm
        nuclei_stack = []
        cyto_stack = []
        # Extract frames
        for frame_idx in range(start_frame, end_frame + 1):
            # Load and extract both channels
            self.generator.load_nuclei(frame_idx)
            self.generator.load_cyto(frame_idx)
            
            nuclei = self.generator.extract_nuclei(pattern_idx)
            cyto = self.generator.extract_cyto(pattern_idx)
            
            nuclei_stack.append(nuclei)
            cyto_stack.append(cyto)
        
        if not nuclei_stack or not cyto_stack:
            return
            
        # Convert stacks to numpy arrays
        nuclei_stack = np.array(nuclei_stack, dtype=self.generator.dtype)
        cyto_stack = np.array(cyto_stack, dtype=self.generator.dtype)
        
        # Calculate number of frames in this sequence
        n_frames = end_frame - start_frame + 1

        # Create RGB stack (nuclei in red, cytoplasm in green)
        rgb_stack = np.zeros((n_frames, nuclei_stack.shape[1], nuclei_stack.shape[2], 3), dtype=self.generator.dtype)
        rgb_stack[..., 0] = nuclei_stack  # Red channel for nuclei
        rgb_stack[..., 1] = cyto_stack   # Green channel for cytoplasm
        
        # Save frame stack
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*is a low contrast image.*')
            imwrite(frame_output_path, rgb_stack)

        # Save frame indices
        with open(json_output_path, 'w') as f:
            json.dump({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_frames": n_frames
            }, f, indent=2)

        logger.info(f"Saved pattern {pattern_idx} frames from {start_frame} to {end_frame} to {frame_output_path}")

    def _save_pattern(self,
                      pattern_idx: int,
                      pattern_output_path: Path,
                      ) -> None:
        """
        Extract and save pattern
        """
        # Extract pattern
        pattern = self.generator.extract_pattern(pattern_idx)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*is a low contrast image.*')
            imwrite(pattern_output_path, pattern)
        
        logger.info(f"Saved pattern {pattern_idx} image to {pattern_output_path}")

    def _process_time_series(self, time_series: Dict, view_idx: int, output_dir: Path, min_frames: int) -> None:
        """
        Process a single time series JSON file.
        
        Args:
            data (Dict): Time series data
            view_idx (int): View index
            output_dir (Path): Directory to save extracted frames
            min_frames (int): Minimum number of frames required for extraction
            
        Raises:
            ValueError: If processing fails
        """     
        # Refine time lapse
        time_series = self._refine_time_series(time_series, min_frames)
        
        # Add head and tail frames
        time_series = self._add_head_tail(time_series)
        
        # Load view and patterns
        self.generator.load_view(view_idx)
        self.generator.load_patterns()
        self.generator.process_patterns()
        
        # Process each pattern
        for pattern_sequence_idx, (start_frame, end_frame) in time_series.items():
            # Extract original pattern index and sequence number
            pattern_idx = pattern_sequence_idx // 1000
            sequence_idx = pattern_sequence_idx % 1000

            # Filenames and directories
            filename_prefix = f"view_{view_idx:03d}_pattern_{pattern_idx:03d}_{sequence_idx:03d}"
            frames_dir = output_dir / 'frames'
            pattern_dir = output_dir / 'pattern'
            json_dir = output_dir / 'json'
            frames_dir.mkdir(parents=True, exist_ok=True)
            pattern_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)
            frame_output_path = frames_dir / f"{filename_prefix}_frames.tif"
            pattern_output_path = pattern_dir / f"{filename_prefix}_pattern.tif"
            json_output_path = json_dir / f"{filename_prefix}_frames.json"
            
            try:
                self._extract_frame_stack(
                    pattern_idx=pattern_idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_output_path=frame_output_path,
                    json_output_path=json_output_path,
                )

                self._save_pattern(
                    pattern_idx=pattern_idx,
                    pattern_output_path=pattern_output_path,
                )
            except Exception as e:
                logger.warning(f"Error processing pattern {pattern_idx}: {e}")
                continue

    # =====================================================================
    # Public Methods
    # =====================================================================

    def extract(
        self,
        time_series_dir: str,
        min_frames: int = 20
    ) -> None:
        """
        Extract valid frames and patterns for each pattern based on time series analysis results.
        Creates dual-channel outputs with nuclei in red and cytoplasm in green.
        Each time lapse gets its own directory named with view and pattern index.
        
        Args:
            time_series_dir (str): Directory containing time series JSON files
            min_frames (int): Minimum number of valid frames required for extraction (default: 20)
            
        Raises:
            ValueError: If extraction fails
        """
        try:
            # Create base output directory
            output_dir = Path(self.output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all time series JSON files
            time_series_dir = Path(time_series_dir)
            json_files = sorted(list(time_series_dir.glob("time_series_*.json")))
            
            if not json_files:
                raise ValueError(f"No time series JSON files found in {time_series_dir}")
                
            logger.info(f"Found {len(json_files)} time series files")
            
            # Process each time series file
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    time_series = {
                        int(pattern_idx): frames for pattern_idx, frames in data['time_series'].items()
                    }
                    view_idx = int(json_file.stem.split('_')[-1])
                    logger.info(f"Processing time series for view {view_idx}")
                    self._process_time_series(time_series, view_idx, output_dir, min_frames)
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {e}")
                    raise ValueError(f"Error processing {json_file}: {e}")

        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise ValueError(f"Error during extraction: {e}") 