"""
Analyze command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage
    python -m cell_counter.cli.analyze --patterns <patterns_path> --nuclei <nuclei_path> --output <output_path>
    
    # With custom parameters
    python -m cell_counter.cli.analyze --patterns <patterns_path> --nuclei <nuclei_path> --output <output_path> --wanted 3 --no-cellpose --diameter 20
    
    Optional arguments:
    --wanted: Number of nuclei to look for (default: 3)
    --no-cellpose: Use simple thresholding instead of Cellpose
    --no-gpu: Don't use GPU for Cellpose
    --diameter: Expected diameter of cells in pixels (default: 15)
    --channels: Channel indices for Cellpose (default: "0,0")
    --model: Type of Cellpose model to use (default: "cyto3")
"""

import argparse
import sys
import os
from ..core.Analyzer import Analyzer

def parse_channels(channels_str: str) -> list:
    """Parse channel string into a list of integers."""
    try:
        return [int(x) for x in channels_str.split(',')]
    except ValueError:
        raise ValueError("Channels must be comma-separated integers")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze time series data and track nuclei counts."
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
        required=True,
        help="Path to the nuclei image file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--wanted",
        type=int,
        default=3,
        help="Number of nuclei to look for (default: 3)",
    )
    parser.add_argument(
        "--no-cellpose",
        action="store_true",
        help="Use simple thresholding instead of Cellpose",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Don't use GPU for Cellpose",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=15,
        help="Expected diameter of cells in pixels (default: 15)",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="0,0",
        help='Channel indices for Cellpose (default: "0,0")',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cyto3",
        help='Type of Cellpose model to use (default: "cyto3")',
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

    # Check if nuclei file exists
    if not os.path.exists(args.nuclei):
        print(f"Error: Nuclei file not found: {args.nuclei}")
        sys.exit(1)

    # Initialize analyzer
    analyzer = Analyzer(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        wanted=args.wanted,
        use_cellpose=not args.no_cellpose,
        use_gpu=not args.no_gpu,
        diameter=args.diameter,
        channels=args.channels,
        model_type=args.model,
        grid_size=args.grid_size
    )

    # Analyze time series
    results = analyzer.analyze_time_series()

    # Save results
    analyzer.save_time_series(results, args.output)

if __name__ == '__main__':
    main() 