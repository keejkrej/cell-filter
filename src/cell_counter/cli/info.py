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
import platform
import os
import argparse
import sys
from pathlib import Path
from ..core.InfoDisplayer import InfoDisplayer

# Only filter specific Qt warnings on Linux
if platform.system() == 'Linux':
    warnings.filterwarnings('ignore', message='Failed to create wl_display')
    warnings.filterwarnings('ignore', message='Could not load the Qt platform plugin')
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

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

    # Initialize InfoDisplayer
    info_displayer = InfoDisplayer(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        cyto_path=args.cyto,
        grid_size=args.grid_size
    )

    # Get and display info
    data = info_displayer.get_info()

    # Print patterns image info
    print("\nPatterns Image:")
    print(f"  Path: {data['patterns']['path']}")
    print(f"  Dimensions: {data['patterns']['dimensions']}")
    print(f"  Grid size: {args.grid_size}x{args.grid_size}")

    # Print nuclei stack info if provided
    if 'nuclei' in data:
        print("\nNuclei Stack:")
        print(f"  Path: {data['nuclei']['path']}")
        print(f"  Number of frames: {data['nuclei']['num_frames']}")
        print(f"  Dimensions per frame: {data['nuclei']['dimensions_per_frame']}")

    # Print cytoplasm stack info if provided
    if 'cyto' in data:
        print("\nCytoplasm Stack:")
        print(f"  Path: {data['cyto']['path']}")
        print(f"  Number of frames: {data['cyto']['num_frames']}")
        print(f"  Dimensions per frame: {data['cyto']['dimensions_per_frame']}")

    # Print contours info
    print("\nContours:")
    print(f"  Total contours found: {data['contours']['total_contours']}")
    print(f"  Contours after filtering: {data['contours']['contours_after_filtering']}")
    if data['contours']['contours_filtered_out'] > 0:
        print(f"  Contours filtered out: {data['contours']['contours_filtered_out']}")

    # Show patterns visualization
    info_displayer.show_patterns()

if __name__ == '__main__':
    main() 