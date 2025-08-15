"""
Core extractor functionality for cell-filter.
"""

import json
from pathlib import Path
import numpy as np
from .generate import Generator, GeneratorParameters
import logging

# Configure logging
logger = logging.getLogger(__name__)


class Extractor:
    """Extract frames and patterns from time series analysis results."""

    # Constructor

    def __init__(
        self,
        patterns_path: str,
        cells_path: str,
        output_folder: str,
        nuclei_channel: int,
    ) -> None:
        """Initialize Extractor with paths and configuration."""
        self.patterns_path = str(Path(patterns_path).resolve())
        self.cells_path = str(Path(cells_path).resolve())
        self.output_folder = str(Path(output_folder).resolve())
        try:
            self.generator = Generator(
                patterns_path,
                cells_path,
                parameters=GeneratorParameters(nuclei_channel=nuclei_channel),
            )
            logger.info(
                f"Successfully initialized Extractor with patterns: {patterns_path} and cells: {cells_path}"
            )
        except Exception as e:
            logger.error(f"Error initializing Extractor: {e}")
            raise ValueError(f"Error initializing Extractor: {e}")

    # Private Methods

    def _refine_time_series(
        self, time_series: dict[int, list[int]], min_frames: int
    ) -> dict[int, list[int]]:
        """Refine time series by splitting gaps and filtering by minimum frames."""
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
                gap = frames[i] - frames[i - 1]
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

                logger.debug(
                    f"Split pattern {pattern_idx} into {len(sequences)} sequences"
                )
                logger.debug(f"Sequence {i}: frames {start} to {end}")

        return refined_time_series

    def _add_head_tail(
        self, time_series: dict[int, list[int]], n_frames: int = 3
    ) -> dict[int, list[int]]:
        """Add head and tail frames to time series sequences."""
        extended_time_series = {}

        # Get total number of frames from the generator
        total_frames = self.generator.n_frames

        for pattern_idx, (start, end) in time_series.items():
            # Add head frames (before start)
            new_start = max(0, start - n_frames)

            # Add tail frames (after end), but don't exceed total_frames
            new_end = min(total_frames - 1, end + n_frames)

            extended_time_series[pattern_idx] = [new_start, new_end]

            logger.debug(
                f"Extended pattern {pattern_idx} with {start - new_start} head frames and {new_end - end} tail frames"
            )

        return extended_time_series

    def _extract_frame_stack(
        self,
        pattern_idx: int,
        start_frame: int,
        end_frame: int,
        frame_output_path: Path,
        json_output_path: Path,
    ) -> None:
        """Extract and save frame stack for a pattern sequence."""

        pattern = self.generator.extract_pattern(pattern_idx)  # (h, w)

        # Initialize stack for all cell regions
        cell_stack = []

        # Extract frames
        for frame_idx in range(start_frame, end_frame + 1):
            # Load all cell channels at once
            self.generator.load_cell(frame_idx)

            # Extract regions from all cell channels at once
            cell = self.generator.extract_cell(pattern_idx)

            # Stack all cell regions together
            cell_stack.append(cell)

        # Calculate number of frames in this sequence
        n_frames = end_frame - start_frame + 1

        # Convert cell regions stack to numpy array
        # cell_stack is a list of arrays, each with shape (n_channels, h, w)
        cell_array = np.array(cell_stack)  # Shape: (n_frames, n_channels, h, w)

        # Expand pattern to match frame dimensions
        # pattern has shape (h, w), so we add new axes to make it (1, h, w)
        # then broadcast to (n_frames, h, w)
        pattern_expanded = np.broadcast_to(
            pattern[np.newaxis, :, :], (n_frames, pattern.shape[0], pattern.shape[1])
        )  # Shape: (n_frames, h, w)

        # Add channel dimension to pattern to make it (n_frames, 1, h, w)
        pattern_with_channel = pattern_expanded[
            :, np.newaxis, :, :
        ]  # Shape: (n_frames, 1, h, w)

        # Stack pattern and cell data along channel axis (pattern first)
        # Resulting shape: (n_frames, n_channels+1, h, w)
        final_stack = np.concatenate([pattern_with_channel, cell_array], axis=1)

        # Save frame stack as NPY
        np.save(frame_output_path, final_stack)

        # Save frame indices and pattern bounding box
        pattern_bbox = (
            self.generator.bounding_boxes[pattern_idx]
            if self.generator.bounding_boxes
            else None
        )
        with open(json_output_path, "w") as f:
            json.dump(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "n_frames": n_frames,
                    "pattern_bbox": pattern_bbox,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Saved pattern {pattern_idx} frames from {start_frame} to {end_frame} to {frame_output_path}"
        )

    def _process_time_series(
        self, time_series: dict, view_idx: int, output_dir: Path, min_frames: int
    ) -> None:
        """Process time series data for a single view."""
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
            filename_prefix = (
                f"view_{view_idx:03d}_pattern_{pattern_idx:03d}_{sequence_idx:03d}"
            )
            extraction_dir = output_dir / "extraction"
            extraction_dir.mkdir(parents=True, exist_ok=True)
            frame_output_path = extraction_dir / f"{filename_prefix}.npy"
            json_output_path = extraction_dir / f"{filename_prefix}.json"

            try:
                self._extract_frame_stack(
                    pattern_idx=pattern_idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_output_path=frame_output_path,
                    json_output_path=json_output_path,
                )
            except Exception as e:
                logger.warning(f"Error processing pattern {pattern_idx}: {e}")
                continue

    # Public Methods

    def extract(self, time_series_dir: str, min_frames: int = 20) -> None:
        """Extract valid frames and patterns from time series analysis results."""
        try:
            # Create base output directory
            output_dir = Path(self.output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find all time series JSON files
            json_files = sorted(list(Path(time_series_dir).glob("time_series_*.json")))

            if not json_files:
                raise ValueError(
                    f"No time series JSON files found in {time_series_dir}"
                )

            logger.info(f"Found {len(json_files)} time series files")

            # Process each time series file
            for json_file in json_files:
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                    time_series = {
                        int(pattern_idx): frames
                        for pattern_idx, frames in data["time_series"].items()
                    }
                    view_idx = int(json_file.stem.split("_")[-1])
                    logger.info(f"Processing time series for view {view_idx}")
                    self._process_time_series(
                        time_series, view_idx, output_dir, min_frames
                    )
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {e}")
                    raise ValueError(f"Error processing {json_file}: {e}")

        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise ValueError(f"Error during extraction: {e}")
