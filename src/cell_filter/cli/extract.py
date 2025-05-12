"""
Extract command for cell-filter.

This script extracts valid frames and patterns from time series analysis results.
It processes the output from the analyze command to identify and save frames that meet
the specified criteria for further analysis.

Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage
    python -m cell_filter.cli.extract --patterns <patterns_path> --cells <cells_path> --time-series <time_series_dir> --output <output_dir>
    
    # With custom parameters
    python -m cell_filter.cli.extract --patterns <patterns_path> --cells <cells_path> --time-series <time_series_dir> --output <output_dir> --min-frames 20

Arguments:
    Required:
        --patterns: Path to the patterns ND2 file
        --cells: Path to the cells ND2 file containing nuclei and cytoplasm channels
        --time-series: Directory containing time series JSON files
        --output: Directory to save extracted frames
    
    Optional:
        --min-frames: Minimum number of valid frames required for extraction (default: 20)
        --debug: Enable debug logging
"""

import argparse
import sys
import os
from ..core.extract import Extractor
import logging
from pathlib import Path

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract valid frames and patterns from time series analysis results."
    )
    parser.add_argument(
        "--patterns",
        type=str,
        required=True,
        help="Path to the patterns ND2 file",
    )
    parser.add_argument(
        "--cells",
        type=str,
        required=True,
        help="Path to the cells ND2 file containing nuclei and cytoplasm channels",
    )
    parser.add_argument(
        "--nuclei-channel",
        type=int,
        required=True,
        help="Channel index for nuclei",
    )
    parser.add_argument(
        "--cyto-channel",
        type=int,
        required=True,
        help="Channel index for cytoplasm",
    )
    parser.add_argument(
        "--time-series",
        type=str,
        required=True,
        help="Directory containing time series JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum number of valid frames required for extraction (default: 20)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()

def main():
    """Main function to run the extraction pipeline.
    
    This function:
    1. Parses command line arguments
    2. Configures logging
    3. Validates input files and directories
    4. Creates output directory if needed
    5. Initializes the extractor
    6. Performs the extraction process
    """
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    
    # Set package logger level before getting logger instances
    logging.getLogger("cell_filter").setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Get the logger instance with explicit package path
    logger = logging.getLogger("cell_filter.cli.extract")

    # Check if patterns file exists
    if not os.path.exists(args.patterns):
        logger.error(f"Patterns file not found: {args.patterns}")
        sys.exit(1)

    # Check if cells file exists
    if not os.path.exists(args.cells):
        logger.error(f"Cells file not found: {args.cells}")
        sys.exit(1)

    # Check if time series directory exists
    if not os.path.exists(args.time_series):
        logger.error(f"Time series directory not found: {args.time_series}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize extractor
        extractor = Extractor(
            patterns_path=args.patterns,
            cells_path=args.cells,
            output_folder=args.output,
            nuclei_channel=args.nuclei_channel,
            cyto_channel=args.cyto_channel
        )

        # Extract frames
        logger.info("Starting extraction process")
        extractor.extract(
            time_series_dir=args.time_series,
            min_frames=args.min_frames
        )
        logger.info("Extraction completed successfully")

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise

if __name__ == "__main__":
    main() 