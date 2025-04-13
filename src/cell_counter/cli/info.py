"""
Info command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage
    python -m cell_counter.cli.info --patterns <patterns_path> --cells <cells_path> --view <view_idx>
    
    # Save plot to file
    python -m cell_counter.cli.info --patterns <patterns_path> --cells <cells_path> --view <view_idx> --output <output_path>
"""

import argparse
import sys
import os
from ..core.InfoDisplayer import InfoDisplayer
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
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
    parser.add_argument(
        "--view",
        type=int,
        required=True,
        help="Index of the view to display",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the plot (if not provided, plot will be displayed)",
    )
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

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

        # Plot view
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