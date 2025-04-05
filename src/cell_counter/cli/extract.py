"""
Extract valid frames command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage
    python -m cell_counter.cli.extract --patterns <patterns_path> --nuclei <nuclei_path> --time-series <time_series_path> --output <output_dir>
    
    # With custom parameters
    python -m cell_counter.cli.extract --patterns <patterns_path> --nuclei <nuclei_path> --time-series <time_series_path> --output <output_dir> --min-frames 20
    
    Optional arguments:
    --min-frames: Minimum number of valid frames required for extraction (default: 10)
"""

import argparse
from pathlib import Path
from ..core.extract import extract_valid_frames

def main():
    """Main entry point for extract command."""
    parser = argparse.ArgumentParser(description='Extract valid frames for each contour based on time series analysis')
    
    # Required arguments
    parser.add_argument('--patterns', type=str, required=True, help='Path to the patterns image file')
    parser.add_argument('--nuclei', type=str, required=True, help='Path to the nuclei image file')
    parser.add_argument('--time-series', type=str, required=True, help='Path to the time series analysis JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for extracted frames')
    
    # Optional arguments
    parser.add_argument('--min-frames', type=int, default=10, help='Minimum number of valid frames required (default: 10)')
    
    args = parser.parse_args()
    
    # Validate paths
    patterns_path = Path(args.patterns)
    nuclei_path = Path(args.nuclei)
    time_series_path = Path(args.time_series)
    
    if not patterns_path.exists():
        raise FileNotFoundError(f"Patterns file not found: {patterns_path}")
    if not nuclei_path.exists():
        raise FileNotFoundError(f"Nuclei file not found: {nuclei_path}")
    if not time_series_path.exists():
        raise FileNotFoundError(f"Time series file not found: {time_series_path}")
    
    # Run extraction
    extract_valid_frames(
        patterns_path=str(patterns_path),
        nuclei_path=str(nuclei_path),
        time_series_path=str(time_series_path),
        output_dir=args.output,
        min_frames=args.min_frames
    )

if __name__ == '__main__':
    main() 