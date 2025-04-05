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

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from cell_counter.data.cell_generator import CellGenerator


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


def show_patterns(generator):
    """Show visualization of patterns image and contours/bounding boxes.
    
    Args:
        generator: CellGenerator instance containing patterns image and contours
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original patterns image
    plt.subplot(1, 2, 1)
    plt.imshow(generator.patterns, cmap='gray')
    plt.title('Original Patterns Image')
    plt.axis('off')

    # Plot contours and bounding boxes on black background
    plt.subplot(1, 2, 2)
    vis_img = np.zeros_like(generator.patterns, dtype=np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    
    # Draw contours in green
    cv2.drawContours(vis_img, generator.contours, -1, (0, 255, 0), 2)
    
    # Draw bounding boxes in red and add index numbers
    for idx, bbox in enumerate(generator.bounding_boxes):
        x, y, w, h = bbox
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Add index number
        cv2.putText(vis_img, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
    
    plt.imshow(vis_img)
    plt.title('Contours (green) and Bounding Boxes (red) with Indices')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


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

    # Initialize generator
    generator = CellGenerator(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        cyto_path=args.cyto,
    )

    # Print patterns image info
    print("\nPatterns Image:")
    print(f"  Path: {args.patterns}")
    print(f"  Dimensions: {generator.patterns.shape}")

    # Print nuclei stack info if provided
    if args.nuclei:
        print("\nNuclei Stack:")
        print(f"  Path: {args.nuclei}")
        print(f"  Number of frames: {generator.n_frames}")
        print(f"  Dimensions per frame: {generator.nuclei_stack[0].shape}")

    # Print cytoplasm stack info if provided
    if args.cyto:
        print("\nCytoplasm Stack:")
        print(f"  Path: {args.cyto}")
        if not args.nuclei:  # Only print frames if nuclei not provided
            print(f"  Number of frames: {generator.n_frames}")
        print(f"  Dimensions per frame: {generator.cyto_stack[0].shape}")

    # Print contours info
    print("\nContours:")
    print(f"  Total contours found: {len(generator.contours)}")
    print(f"  Contours after filtering: {len(generator.contours)}")
    if len(generator.contours) < len(generator.contours):
        print(f"  Contours filtered out: {len(generator.contours) - len(generator.contours)}")

    # Show visualization
    show_patterns(generator)


if __name__ == "__main__":
    main() 