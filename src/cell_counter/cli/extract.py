"""
Extract valid frames command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage (extracts patterns by default)
    python -m cell_counter.cli.extract --patterns <patterns_path> --time-series <time_series_path> --output <output_dir>
    
    # Extract patterns with nuclei
    python -m cell_counter.cli.extract --patterns <patterns_path> --nuclei <nuclei_path> --time-series <time_series_path> --output <output_dir>
    
    # Extract patterns with cytoplasm
    python -m cell_counter.cli.extract --patterns <patterns_path> --cyto <cyto_path> --time-series <time_series_path> --output <output_dir>
    
    # Extract patterns with both nuclei and cytoplasm
    python -m cell_counter.cli.extract --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path> --time-series <time_series_path> --output <output_dir>
    
    # With custom parameters
    python -m cell_counter.cli.extract --patterns <patterns_path> --time-series <time_series_path> --output <output_dir> --min-frames 20
    
    Optional arguments:
    --min-frames: Minimum number of valid frames required for extraction (default: 20)
    --grid-size: Size of the grid for snapping pattern centers (default: 20)
    --nuclei: Path to the nuclei image file (optional)
    --cyto: Path to the cytoplasm image file (optional)
"""

import argparse
import os
import sys
from pathlib import Path
from ..core.Extractor import Extractor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract valid frames for each contour based on time series analysis."
    )
    parser.add_argument(
        "--patterns",
        type=str,
        required=True,
        help="Path to the patterns image file",
    )
    parser.add_argument(
        "--nuclei",
        type=str,
        help="Path to the nuclei image file",
    )
    parser.add_argument(
        "--cyto",
        type=str,
        help="Path to the cytoplasm image file",
    )
    parser.add_argument(
        "--time-series",
        type=str,
        required=True,
        help="Path to the time series JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum number of valid frames required (default: 20)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Size of the grid for snapping pattern centers (default: 20)",
    )
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Check if patterns file exists
    if not os.path.exists(args.patterns):
        print(f"Error: Patterns file not found: {args.patterns}")
        sys.exit(1)

    # Check if nuclei file exists if provided
    if args.nuclei and not os.path.exists(args.nuclei):
        print(f"Error: Nuclei file not found: {args.nuclei}")
        sys.exit(1)

    # Check if cytoplasm file exists if provided
    if args.cyto and not os.path.exists(args.cyto):
        print(f"Error: Cytoplasm file not found: {args.cyto}")
        sys.exit(1)

    # Check if time series file exists
    if not os.path.exists(args.time_series):
        print(f"Error: Time series file not found: {args.time_series}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Extractor
    extractor = Extractor(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        cyto_path=args.cyto,
        grid_size=args.grid_size
    )

    # Always extract patterns
    extractor.extract_patterns(
        time_series_path=args.time_series,
        output_dir=output_dir / "patterns",
        min_frames=args.min_frames
    )

    # Extract nuclei if provided
    if args.nuclei:
        extractor.extract_valid_frames(
            time_series_path=args.time_series,
            output_dir=output_dir / "nuclei",
            min_frames=args.min_frames,
            image_type="nuclei"
        )
    
    # Extract cytoplasm if provided
    if args.cyto:
        extractor.extract_valid_frames(
            time_series_path=args.time_series,
            output_dir=output_dir / "cyto",
            min_frames=args.min_frames,
            image_type="cyto"
        )

if __name__ == "__main__":
    main() 