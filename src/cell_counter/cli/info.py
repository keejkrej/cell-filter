"""
Info command for cell-counter.

This script displays patterns images with bounding boxes and indices for visualization
and inspection purposes. It can show individual views or cycle through all views
in the dataset.

Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage - display single view
    python -m cell_counter.cli.info --patterns <patterns_path> --cells <cells_path> --view <view_idx>
    
    # Display all views sequentially
    python -m cell_counter.cli.info --patterns <patterns_path> --cells <cells_path> --view-all
    
    # Save plot to file (only works with single view)
    python -m cell_counter.cli.info --patterns <patterns_path> --cells <cells_path> --view <view_idx> --output <output_path>

Arguments:
    Required:
        --patterns: Path to the patterns ND2 file
        --cells: Path to the cells ND2 file containing nuclei and cytoplasm channels
    
    View Selection (one required):
        --view: Index of the view to display
        --view-all: Display all views sequentially
    
    Optional:
        --output: Path to save the plot (only works with --view, not --view-all)
        --debug: Enable debug logging
"""

import argparse
import sys
import os
from ..core import InfoDisplayer
import logging

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Display patterns image with bounding boxes and indices."
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
    view_group = parser.add_mutually_exclusive_group(required=True)
    view_group.add_argument(
        "--view",
        type=int,
        help="Index of the view to display",
    )
    view_group.add_argument(
        "--view-all",
        action="store_true",
        help="Display all views sequentially",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the plot (only works with --view, not --view-all)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()

def main():
    """Main function to run the visualization pipeline.
    
    This function:
    1. Parses command line arguments
    2. Configures logging
    3. Validates input files
    4. Initializes the info displayer
    5. Displays the requested view(s)
    6. Saves plot if output path is specified
    """
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(message)s')  # Root logger at WARNING to suppress third-party messages
    
    # Set package logger level before getting logger instances
    logging.getLogger("cell_counter").setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Get the logger instance with explicit package path
    logger = logging.getLogger("cell_counter.cli.info")

    # Check if patterns file exists
    if not os.path.exists(args.patterns):
        logger.error(f"Patterns file not found: {args.patterns}")
        sys.exit(1)

    # Check if cells file exists
    if not os.path.exists(args.cells):
        logger.error(f"Cells file not found: {args.cells}")
        sys.exit(1)

    try:
        # Initialize info displayer
        displayer = InfoDisplayer(
            patterns_path=args.patterns,
            cells_path=args.cells
        )

        if args.view_all:
            # Display each view
            logger.info("Displaying all views")
            view_idx = 0
            while True:
                try:
                    logger.info(f"Displaying view {view_idx}")
                    displayer.plot_view(view_idx)
                    view_idx += 1
                except ValueError:
                    # When view_idx is out of range, we're done
                    break
        else:
            # Plot single view
            logger.info(f"Plotting view {args.view}")
            displayer.plot_view(args.view, args.output)
            logger.info("Plotting completed successfully")

    except Exception as e:
        logger.error(f"Error during plotting: {e}")
        raise
    finally:
        displayer.close()

if __name__ == "__main__":
    main() 