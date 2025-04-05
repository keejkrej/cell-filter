"""
Info command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Show info for patterns and nuclei
    python -m cell_counter.cli.info --patterns <patterns_path> --nuclei <nuclei_path>
    
    # Show info for patterns and cytoplasm
    python -m cell_counter.cli.info --patterns <patterns_path> --cyto <cyto_path>
    
    # Show info for all
    python -m cell_counter.cli.info --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path>
"""

import warnings
warnings.filterwarnings('ignore', message='.*Failed to create wl_display.*')
warnings.filterwarnings('ignore', message='.*Could not load the Qt platform plugin.*')

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import argparse
import sys
from pathlib import Path

from ..core.info import get_image_info, show_patterns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Display information about the image stacks."
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

    # Get image info
    info, generator = get_image_info(args.patterns, args.nuclei, args.cyto)

    # Print patterns image info
    print("\nPatterns Image:")
    print(f"  Path: {info['patterns']['path']}")
    print(f"  Dimensions: {info['patterns']['dimensions']}")

    # Print nuclei stack info if provided
    if 'nuclei' in info:
        print("\nNuclei Stack:")
        print(f"  Path: {info['nuclei']['path']}")
        print(f"  Number of frames: {info['nuclei']['num_frames']}")
        print(f"  Dimensions per frame: {info['nuclei']['dimensions_per_frame']}")

    # Print cytoplasm stack info if provided
    if 'cyto' in info:
        print("\nCytoplasm Stack:")
        print(f"  Path: {info['cyto']['path']}")
        if info['cyto']['num_frames'] is not None:  # Only print frames if nuclei not provided
            print(f"  Number of frames: {info['cyto']['num_frames']}")
        print(f"  Dimensions per frame: {info['cyto']['dimensions_per_frame']}")

    # Print contours info
    print("\nContours:")
    print(f"  Total contours found: {info['contours']['total_contours']}")
    print(f"  Contours after filtering: {info['contours']['contours_after_filtering']}")
    if info['contours']['contours_filtered_out'] > 0:
        print(f"  Contours filtered out: {info['contours']['contours_filtered_out']}")

    # Show visualization
    show_patterns(generator)


if __name__ == "__main__":
    main() 