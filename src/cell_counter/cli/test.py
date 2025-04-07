"""
Test script for cellpose functionality.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from skimage import io
from cellpose import models, plot
import matplotlib.pyplot as plt
from ..core.CellGenerator import CellGenerator

def main():
    parser = argparse.ArgumentParser(description="Test cellpose functionality on a single contour from a frame")
    parser.add_argument("--nuclei", type=str, required=True, help="Path to nuclei TIFF file")
    parser.add_argument("--patterns", type=str, required=True, help="Path to patterns TIFF file")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to analyze")
    parser.add_argument("--contour", type=int, required=True, help="Contour index to analyze")
    parser.add_argument("--diameter", type=int, default=15, help="Expected cell diameter in pixels")
    parser.add_argument("--channels", type=str, default="0,0", help="Channel indices for cellpose")
    parser.add_argument("--model-type", type=str, default="cyto3", help="Type of cellpose model to use")
    parser.add_argument("--use-gpu", action="store_true", help="Whether to use GPU for cellpose")
    parser.add_argument("--show-plot", action="store_true", help="Whether to show segmentation plot")
    parser.add_argument("--grid-size", type=int, default=20, help="Size of the grid for snapping pattern centers")
    parser.add_argument("--threshold", type=int, help="Threshold value for nuclei extraction")
    parser.add_argument("--output", type=str, help="Path to save output image")
    
    args = parser.parse_args()
    
    # Convert channels string to list of ints
    channels = [int(c) for c in args.channels.split(",")]
    
    # Initialize CellGenerator
    print(f"\nInitializing CellGenerator...")
    generator = CellGenerator(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        grid_size=args.grid_size
    )
    
    # Load frame
    print(f"\nLoading frame {args.frame}...")
    generator.load_frame_nuclei(args.frame)
    
    # Check if contour index is valid
    if args.contour >= len(generator.contours):
        print(f"Error: Contour index {args.contour} is out of range. There are {len(generator.contours)} contours.")
        return
    
    # Initialize cellpose model
    print("\nInitializing cellpose model...")
    model = models.Cellpose(gpu=args.use_gpu, model_type=args.model_type)
    
    # Extract nuclei region
    print(f"\nProcessing contour {args.contour}...")
    try:
        nuclei = generator.extract_nuclei(args.contour, threshold=args.threshold)
    except Exception as e:
        print(f"Error extracting nuclei for contour {args.contour}: {str(e)}")
        return
    
    # Run cellpose
    print("Running cellpose...")
    masks_pred, flows, styles, diams = model.eval(
        [nuclei],  # Pass single image as list
        diameter=args.diameter,
        channels=channels
    )
    
    # Count nuclei in mask
    n_nuclei = len(np.unique(masks_pred[0])) - 1  # -1 to exclude background
    
    print(f"\nContour {args.contour} analysis:")
    print(f"  Number of nuclei found: {n_nuclei}")
    
    # Show segmentation plot if requested
    if args.show_plot:
        print("\nShowing segmentation plot...")
        fig = plt.figure(figsize=(10, 2))
        plot.show_segmentation(fig, nuclei, masks_pred[0], flows[0][0])
        plt.show()

    # Save output if requested
    if args.output:
        io.imsave(args.output, nuclei)
        print(f"Saved nuclei image to {args.output}")

if __name__ == "__main__":
    main()
