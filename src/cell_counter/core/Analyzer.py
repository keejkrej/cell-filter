"""
Core analyzer functionality for cell-counter.
"""

import json
from typing import Dict, List, Optional
from .CellGenerator import CellGenerator
from .counters import CellposeCounter, SimpleCounter

class Contours:
    """Class to manage the state of all contours."""
    
    def __init__(self, n_contours: int):
        self.tracked: List[int] = list(range(n_contours))
        self.dropped_zero: List[int] = []
        self.dropped_many: List[int] = []
        self.saved: Dict[int, List[int]] = {i: [] for i in range(n_contours)}
    
    def drop_zero(self, idx: int):
        """Mark contour as dropped due to zero nuclei."""
        if idx in self.tracked:
            self.tracked.remove(idx)
            self.dropped_zero.append(idx)
    
    def drop_many(self, idx: int):
        """Mark contour as dropped due to too many nuclei."""
        if idx in self.tracked:
            self.tracked.remove(idx)
            self.dropped_many.append(idx)
    
    def save_frame(self, idx: int, frame_idx: int):
        """Save a valid frame index for a contour."""
        self.saved[idx].append(frame_idx)
    
    def get_tracked_indices(self) -> List[int]:
        """Get list of indices being tracked."""
        return list(self.tracked)  # Return a copy of the list
    
    def get_valid_contours(self) -> Dict[int, List[int]]:
        """Get dictionary of contours with valid frames."""
        return {idx: frames for idx, frames in self.saved.items() if frames}

class Analyzer:
    """
    A class for analyzing time series data and tracking nuclei counts.
    """
    
    def __init__(
        self,
        patterns_path: str,
        nuclei_path: str,
        wanted: int = 3,
        use_cellpose: bool = True,
        use_gpu: bool = True,
        diameter: int = 15,
        channels: str = "0,0",
        model_type: str = "cyto3",
        grid_size: int = 20,
        threshold: Optional[int] = None
    ):
        """
        Initialize the Analyzer with paths and parameters.
        
        Args:
            patterns_path: Path to the patterns image file
            nuclei_path: Path to the nuclei image file
            wanted: Number of nuclei to look for
            use_cellpose: Whether to use Cellpose for counting
            use_gpu: Whether to use GPU for Cellpose
            diameter: Expected diameter of cells in pixels
            channels: Channel indices for Cellpose
            model_type: Type of Cellpose model to use
            grid_size: Size of the grid for snapping pattern centers
            threshold: Threshold value for nuclei extraction
        """
        print(f"\nInitializing analyzer with:")
        print(f"  Patterns file: {patterns_path}")
        print(f"  Nuclei file: {nuclei_path}")
        print(f"  Expected nuclei count: {wanted}")
        print(f"  Using Cellpose: {use_cellpose}")
        if use_cellpose:
            print(f"  Using GPU: {use_gpu}")
            print(f"  Cell diameter: {diameter}")
            print(f"  Channels: {channels}")
            print(f"  Model type: {model_type}")
        print(f"  Grid size: {grid_size}")
        if threshold is not None:
            print(f"  Threshold: {threshold}")
        
        self.generator = CellGenerator(
            patterns_path=patterns_path,
            nuclei_path=nuclei_path,
            grid_size=grid_size
        )
        
        # Initialize counter based on method
        if use_cellpose:
            print("\nInitializing Cellpose counter...")
            self.counter = CellposeCounter(
                diameter=diameter,
                channels=channels,
                model_type=model_type,
                use_gpu=use_gpu
            )
        else:
            print("\nInitializing simple thresholding counter...")
            self.counter = SimpleCounter()
        
        self.wanted = wanted
        self.threshold = threshold
        
        # Initialize contours
        self.contours = Contours(len(self.generator.contours))
        
        # Store metadata
        self.metadata = {
            "patterns_path": str(patterns_path),
            "nuclei_path": str(nuclei_path),
            "wanted_nuclei": wanted,
            "use_cellpose": use_cellpose,
            "use_gpu": use_gpu,
            "diameter": diameter,
            "channels": channels,
            "model_type": model_type,
            "total_contours": len(self.generator.contours),
            "total_frames": self.generator.n_frames_nuclei
        }
        
        print(f"\nFound {self.metadata['total_contours']} contours and {self.metadata['total_frames']} frames")
    
    def analyze_time_series(self) -> Dict:
        """
        Analyze time series data and track nuclei counts.
        
        Returns:
            Dictionary containing analysis results
        """
        print("\nStarting time series analysis...")
        results = {
            "metadata": self.metadata,
            "time_lapse": {}
        }
        
        print(f"\nAnalyzing {self.generator.n_frames_nuclei} frames...")
        for frame_idx in range(self.generator.n_frames_nuclei):
            print(f"Processing frame {frame_idx}/{self.generator.n_frames_nuclei}")
            # Load current frame
            self.generator.load_frame_nuclei(frame_idx)
            
            # Collect all nuclei for this frame
            nuclei_list = []
            tracked_indices = self.contours.get_tracked_indices()  # Already returns a copy
            for contour_idx in tracked_indices:
                try:
                    nuclei = self.generator.extract_nuclei(contour_idx, threshold=self.threshold)
                    nuclei_list.append(nuclei)
                except Exception as e:
                    print(f"\nError extracting nuclei for frame {frame_idx}, contour {contour_idx}: {str(e)}")
                    self.contours.tracked.remove(contour_idx)
                    continue
            
            if not nuclei_list:
                print(f"\nWarning: No valid nuclei regions found in frame {frame_idx}")
                continue
            
            # Count nuclei for all contours in this frame
            try:
                counts = self.counter.count_nuclei(nuclei_list)
            except Exception as e:
                print(f"\nError counting nuclei in frame {frame_idx}: {str(e)}")
                continue
            
            # Track which contours had exactly wanted nuclei
            saved_contours = []
            dropped_many = []
            dropped_zero = []

            # Update contours based on counts - ensure indices and counts match
            for contour_idx, n_count in zip(tracked_indices, counts):
                if n_count == self.wanted:
                    self.contours.save_frame(contour_idx, frame_idx)
                    saved_contours.append(contour_idx)
                elif n_count > self.wanted:
                    # Stop tracking contour if too many nuclei
                    self.contours.drop_many(contour_idx)
                    dropped_many.append(contour_idx)
                elif n_count == 0:
                    # Stop tracking contour if no nuclei
                    self.contours.drop_zero(contour_idx)
                    dropped_zero.append(contour_idx)
            
            # Print results for this frame
            if saved_contours:
                print(f"  Contours with {self.wanted} nuclei: {saved_contours}")
            if dropped_many:
                print(f"  Dropped (too many nuclei): {dropped_many}")
            if dropped_zero:
                print(f"  Dropped (no nuclei): {dropped_zero}")
            print(f"  Remaining tracked contours: {self.contours.get_tracked_indices()}")
        
        # Build results from contours
        results["time_lapse"] = self.contours.get_valid_contours()
        
        print(f"\nAnalysis complete:")
        print(f"  Final valid contours: {list(results['time_lapse'].keys())}")
        
        # Print final dropped contours summary
        print("\nFinal dropped contours summary:")
        if self.contours.dropped_many:
            print(f"  Too many nuclei: {self.contours.dropped_many}")
        if self.contours.dropped_zero:
            print(f"  No nuclei found: {self.contours.dropped_zero}")
        
        return results
    
    def save_time_series(self, results: Dict, output_path: str) -> None:
        """
        Save time series analysis results to a JSON file.
        
        Args:
            results: Analysis results to save
            output_path: Path to save results to
        """
        print(f"\nSaving results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully.") 