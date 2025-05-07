"""
Analyze command for cell-counter.

This script processes time series data to track and analyze nuclei counts in microscopy images.
It uses Cellpose for cell detection and segmentation.

Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage
    python -m cell_counter.cli.analyze --patterns <patterns_path> --cells <cells_path> --output <output_path>
    
    # With custom parameters
    python -m cell_counter.cli.analyze --patterns <patterns_path> --cells <cells_path> --output <output_path> --wanted 3 --no-gpu --diameter 20

Arguments:
    Required:
        --patterns: Path to the patterns image file
        --cells: Path to the cells image file containing nuclei and cytoplasm channels
        --output: Path to save the output JSON file
    
    Optional:
        --wanted: Number of nuclei to look for (default: 3)
        --no-gpu: Disable GPU acceleration for Cellpose
        --diameter: Expected diameter of cells in pixels (default: 15)
        --channels: Channel indices for Cellpose (default: "0,0")
        --model: Type of Cellpose model to use (default: "cyto3")
        --start-view: Starting view index (inclusive)
        --end-view: Ending view index (exclusive)
        --all: Process all views
        --debug: Enable debug logging
"""

import argparse
import sys
import os
from ..core import Analyzer
import logging

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
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
        "--cells",
        type=str,
        required=True,
        help="Path to the cells image file containing nuclei and cytoplasm channels",
    )
    parser.add_argument(
        "--nuclei-channel",
        type=int,
        default=1,
        help="Channel index for nuclei (default: 1)",
    )
    parser.add_argument(
        "--cyto-channel",
        type=int,
        default=0,
        help="Channel index for cytoplasm (default: 0)",
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
        "--diameter",
        type=int,
        default=15,
        help="Expected diameter of nuclei in pixels (default: 15)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Don't use GPU for Cellpose",
    )
    parser.add_argument(
        "--start-view",
        type=int,
        help="Starting view index (inclusive)"
    )
    parser.add_argument(
        "--end-view",
        type=int,
        help="Ending view index (exclusive). Required if --start-view is specified."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all views"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()

def main():
    """Main function to run the analysis pipeline.
    
    This function:
    1. Parses command line arguments
    2. Configures logging
    3. Validates input files
    4. Initializes the analyzer
    5. Processes the specified views
    """
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    
    # Set package logger level before getting logger instances
    logging.getLogger("cell_counter").setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Get the logger instance with explicit package path
    logger = logging.getLogger("cell_counter.cli.analyze")

    # Check if patterns file exists
    if not os.path.exists(args.patterns):
        logger.error(f"Error: Patterns file not found: {args.patterns}")
        sys.exit(1)

    # Check if cells file exists
    if not os.path.exists(args.cells):
        logger.error(f"Error: Cells file not found: {args.cells}")
        sys.exit(1)

    # Validate arguments
    if args.start_view is not None and args.end_view is None:
        logger.error("--end-view is required when --start-view is specified")
        sys.exit(1)

    try:
        # Initialize analyzer
        analyzer = Analyzer(
            patterns_path=args.patterns,
            cells_path=args.cells,
            output_folder=args.output,
            wanted=args.wanted,
            use_gpu=not args.no_gpu,
            diameter=args.diameter,
            nuclei_channel=args.nuclei_channel,
            cyto_channel=args.cyto_channel,
        )

        # Process views
        if args.all:
            logger.info("Processing all views")
            analyzer.process_views(0, analyzer.generator.n_views)
        else:
            logger.info(f"Processing views {args.start_view} to {args.end_view}")
            analyzer.process_views(args.start_view, args.end_view)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == '__main__':
    main() 