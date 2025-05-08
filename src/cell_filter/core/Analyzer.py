"""
Core analyzer functionality for cell-filter.
"""

import json
import time
from typing import Dict, List

from . import CellGenerator, CellGeneratorParameters
from . import CellposeCounter
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class Patterns:
    """
    Class to manage the state of all patterns.
    
    This class tracks the state of patterns throughout the analysis process,
    including which patterns are being tracked, which have been dropped,
    and which frames have been saved for each pattern.
    
    Attributes:
        tracked (List[int]): List of pattern indices currently being tracked
        dropped_zero (List[int]): List of pattern indices dropped due to zero nuclei
        dropped_many (List[int]): List of pattern indices dropped due to too many nuclei
        saved (Dict[int, List[int]]): Dictionary mapping pattern indices to saved frame indices
    """
    
    def __init__(self, n_patterns: int) -> None:
        """
        Initialize the Patterns class.
        
        Args:
            n_patterns (int): Total number of patterns to track
        """
        self.tracked: List[int] = list(range(n_patterns))
        self.dropped_zero: List[int] = []
        self.dropped_many: List[int] = []
        self.saved: Dict[int, List[int]] = {i: [] for i in range(n_patterns)}
    
    def drop_zero(self, idx: int) -> None:
        """
        Mark pattern as dropped due to zero nuclei.
        
        Args:
            idx (int): Index of the pattern to drop
        """
        if idx in self.tracked:
            self.tracked.remove(idx)
            self.dropped_zero.append(idx)
            logger.debug(f"Dropped pattern {idx} due to zero nuclei")
    
    def drop_many(self, idx: int) -> None:
        """
        Mark pattern as dropped due to too many nuclei.
        
        Args:
            idx (int): Index of the pattern to drop
        """
        if idx in self.tracked:
            self.tracked.remove(idx)
            self.dropped_many.append(idx)
            logger.debug(f"Dropped pattern {idx} due to too many nuclei")
    
    def save_frame(self, idx: int, frame_idx: int) -> None:
        """
        Save a valid frame index for a pattern.
        
        Args:
            idx (int): Index of the pattern
            frame_idx (int): Index of the frame to save
        """
        self.saved[idx].append(frame_idx)
        logger.debug(f"Saved frame {frame_idx} for pattern {idx}")
    
    def get_tracked_indices(self) -> List[int]:
        """
        Get list of indices being tracked.
        
        Returns:
            List[int]: Copy of the list of tracked indices
        """
        return list(self.tracked)
    
    def get_valid_patterns(self) -> Dict[int, List[int]]:
        """
        Get dictionary of patterns with valid frames.
        
        Returns:
            Dict[int, List[int]]: Dictionary mapping pattern indices to their valid frame indices
        """
        return {idx: frames for idx, frames in self.saved.items() if frames}

class Analyzer:
    """
    A class for analyzing time series data and tracking nuclei counts.
    
    This class coordinates the analysis of cell patterns over time, tracking
    nuclei counts and maintaining state of valid patterns.
    
    Attributes:
        generator (CellGenerator): Cell generator instance
        counter (CellposeCounter): Cell counter instance
        wanted (int): Desired number of nuclei per pattern
        patterns (Patterns): Pattern tracking state
    """

    # =====================================================================
    # Constructor and Initialization
    # =====================================================================

    def __init__(
        self,
        patterns_path: str,
        cells_path: str,
        output_folder: str,
        wanted: int,
        use_gpu: bool,
        diameter: int,
        nuclei_channel: int,
        cyto_channel: int
    ) -> None:
        """
        Initialize the Analyzer with configuration parameters.
        
        Args:
            patterns_path (str): Path to the patterns ND2 file
            cells_path (str): Path to the cell ND2 file
            output_folder (str): Path to save analysis results
            wanted (int): Desired number of nuclei per pattern
            use_gpu (bool): Whether to use GPU for cell counting
            diameter (int): Expected diameter of nuclei
            
        Raises:
            ValueError: If initialization fails
        """
        self.output_folder = str(Path(output_folder).resolve())
        self.diameter = diameter

        try:
            self._init_generator(patterns_path, cells_path, nuclei_channel, cyto_channel)
            self._init_counter(wanted, use_gpu)
            logger.debug(f"Successfully initialized Analyzer with patterns: {patterns_path} and cells: {cells_path}")
        except Exception as e:
            logger.error(f"Error initializing Analyzer: {e}")
            raise ValueError(f"Error initializing Analyzer: {e}")

    def _init_generator(self, patterns_path: str, cells_path: str, nuclei_channel: int, cyto_channel: int) -> None:
        """
        Initialize the cell generator.
        
        Args:
            patterns_path (str): Path to the patterns ND2 file
            cells_path (str): Path to the cell ND2 file
            
        Raises:
            ValueError: If initialization fails
        """
        try:
            self.generator = CellGenerator(
                patterns_path=patterns_path,
                cells_path=cells_path,
                parameters=CellGeneratorParameters(
                    nuclei_channel=nuclei_channel,
                    cyto_channel=cyto_channel
                )
            )
            logger.debug(f"Initialized generator with {self.generator.n_views} views and {self.generator.n_frames} frames")
        except Exception as e:
            logger.error(f"Error initializing generator: {e}")
            raise
    
    def _init_counter(self, wanted: int, use_gpu: bool) -> None:
        """
        Initialize the cell counter.
        
        Args:
            wanted (int): Desired number of nuclei per pattern
            use_gpu (bool): Whether to use GPU for cell counting
            
        Raises:
            ValueError: If initialization fails
        """
        try:
            self.counter = CellposeCounter(
                use_gpu=use_gpu
            )
            self.wanted = wanted
            logger.debug(f"Initialized counter with wanted={wanted}")
        except Exception as e:
            logger.error(f"Error initializing counter: {e}")
            raise

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _process_frame(self, frame_idx: int) -> None:
        """
        Process a single frame and update pattern tracking.
        
        Args:
            frame_idx (int): Index of the frame to process
            
        Raises:
            ValueError: If processing fails
        """
        try:
            # Load current frame
            self.generator.load_nuclei(frame_idx)
            
            # Collect all nuclei for this frame
            nuclei_list = []
            tracked_indices = self.patterns.get_tracked_indices()
            for pattern_idx in tracked_indices:
                try:
                    nuclei = self.generator.extract_nuclei(pattern_idx)
                    nuclei_list.append(nuclei)
                except Exception as e:
                    logger.warning(f"Error extracting nuclei for frame {frame_idx}, pattern {pattern_idx}: {e}")
                    self.patterns.tracked.remove(pattern_idx)
                    continue
            
            if not nuclei_list:
                logger.warning(f"No valid nuclei regions found in frame {frame_idx}")
                return
            
            # Count nuclei for all patterns in this frame
            try:
                counts = self.counter.count_nuclei(nuclei_list, self.diameter)
            except Exception as e:
                logger.error(f"Error counting nuclei in frame {frame_idx}: {e}")
                return
            
            # Track changes for this frame
            saved_patterns = []
            dropped_many = []
            dropped_zero = []
            
            # Update pattern tracking based on counts
            for pattern_idx, n_count in zip(tracked_indices, counts):
                if n_count == self.wanted:
                    self.patterns.save_frame(pattern_idx, frame_idx)
                    saved_patterns.append(pattern_idx)
                    logger.debug(f"Pattern {pattern_idx} has {self.wanted} nuclei in frame {frame_idx}")
                elif n_count > self.wanted:
                    self.patterns.drop_many(pattern_idx)
                    dropped_many.append(pattern_idx)
                elif n_count == 0:
                    self.patterns.drop_zero(pattern_idx)
                    dropped_zero.append(pattern_idx)
            
            # Log detailed results for this frame
            if saved_patterns:
                logger.debug(f"Frame {frame_idx}: Patterns with {self.wanted} nuclei: {saved_patterns}")
            if dropped_many:
                logger.debug(f"Frame {frame_idx}: Dropped patterns (too many nuclei): {dropped_many}")
            if dropped_zero:
                logger.debug(f"Frame {frame_idx}: Dropped patterns (no nuclei): {dropped_zero}")
            logger.debug(f"Frame {frame_idx}: Remaining tracked patterns: {self.patterns.get_tracked_indices()}")
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            raise ValueError(f"Error processing frame {frame_idx}: {e}")

    # =====================================================================
    # Public Methods
    # =====================================================================

    def analyze_time_series(self, view_idx: int) -> Dict:
        """
        Analyze time series data and track nuclei counts for a single view.
        
        Args:
            view_idx (int): Index of the view to analyze
            
        Returns:
            Dict: Dictionary containing analysis results
            
        Raises:
            ValueError: If analysis fails
        """
        logger.info(f"Starting time series analysis for view {view_idx}")
        
        try:
            # Initialize analysis
            self.generator.load_view(view_idx)
            self.generator.load_patterns()
            self.generator.process_patterns()
            self.patterns = Patterns(self.generator.n_patterns)
            
            # Process each frame
            for frame_idx in range(self.generator.n_frames):
                logger.info(f"Processing frame {frame_idx}/{self.generator.n_frames}")
                self._process_frame(frame_idx)
            
            # Build results
            results = {
                "time_series": self.patterns.get_valid_patterns()
            }
            
            # Log final summary
            logger.debug(f"\nAnalysis complete:")
            logger.debug(f"  Final valid patterns: {list(results['time_series'].keys())}")
            
            # Log final dropped patterns summary
            logger.debug("\nFinal dropped patterns summary:")
            if self.patterns.dropped_many:
                logger.debug(f"  Too many nuclei: {self.patterns.dropped_many}")
            if self.patterns.dropped_zero:
                logger.debug(f"  No nuclei found: {self.patterns.dropped_zero}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            raise ValueError(f"Error in time series analysis: {e}")

    def process_views(self, start_view: int, end_view: int) -> None:
        """
        Process a range of views sequentially.
        
        Args:
            start_view (int): Starting view index (inclusive)
            end_view (int): Ending view index (exclusive)
            
        Raises:
            ValueError: If view range is invalid
        """
        if start_view < 0 or end_view > self.generator.n_views or start_view >= end_view:
            raise ValueError(f"Invalid view range: {start_view} to {end_view} (total views: {self.generator.n_views})")
            
        # Create output folder if it doesn't exist
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Path for tracking file
        tracking_file = output_path / "processed_views.json"
        
        # Load previously processed views if tracking file exists
        processed_views = set()
        if tracking_file.exists():
            try:
                with open(tracking_file, 'r') as f:
                    processed_views = set(json.load(f))
                logger.debug(f"Found {len(processed_views)} previously processed views")
            except Exception as e:
                logger.warning(f"Error reading tracking file: {e}")
        
        logger.debug(f"Starting sequential processing for views {start_view} to {end_view-1}")
        
        for view_idx in range(start_view, end_view):
            # Skip if already processed
            if view_idx in processed_views:
                logger.info(f"Skipping already processed view {view_idx}")
                continue
                
            try:
                # Process the view
                time_start = time.time()
                results = self.analyze_time_series(view_idx)
                time_end = time.time()
                logger.info(f"Time taken to process view {view_idx}: {time_end - time_start} seconds")
                
                # Save results immediately
                view_output_path = output_path / f"time_series_{view_idx:03d}.json"
                with open(view_output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Update tracking file
                processed_views.add(view_idx)
                with open(tracking_file, 'w') as f:
                    json.dump(sorted(list(processed_views)), f, indent=2)
                    
                logger.info(f"Saved results for view {view_idx} to {view_output_path}")
            except Exception as e:
                logger.error(f"Error processing view {view_idx}: {e}")
                # Continue with next view even if this one fails
                continue
                
        logger.debug(f"Sequential processing complete for views {start_view} to {end_view-1}")
        self.generator.close_files()
        
