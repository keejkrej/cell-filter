"""
Count cells in PNG images and output results to CSV.
"""

import argparse
import csv
import os
from pathlib import Path
from glob import glob

from skimage.io import imread
from ..core.count import count_nuclei_cellpose, count_nuclei_simple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Count cells in PNG images and output results to CSV."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing PNG images to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--method",
        choices=["cellpose", "simple"],
        default="simple",
        help="Method to use for cell counting (default: simple)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results from both methods and output differences",
    )
    # Cellpose specific arguments
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for cell counting (if available, only for Cellpose)",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=15,
        help="Expected diameter of cells in pixels (only for Cellpose)",
    )
    # Simple method specific arguments
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum area in pixels to consider as a nucleus (only for simple method)",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Get list of PNG files and sort them
    png_files = sorted(glob(os.path.join(args.input_dir, "*.png")))
    if not png_files:
        print(f"Error: No PNG files found in {args.input_dir}")
        return 1

    # Load images
    print(f"Loading {len(png_files)} images...")
    images = [imread(img_path) for img_path in png_files]

    # Count cells using the selected method
    print("Counting cells...")
    if args.compare:
        # Count using both methods
        cellpose_counts = count_nuclei_cellpose(
            images,
            use_gpu=args.use_gpu,
            diameter=args.diameter,
        )
        simple_counts = count_nuclei_simple(
            images,
            min_area=args.min_area,
        )
        
        # Calculate differences
        differences = [cp - sc for cp, sc in zip(cellpose_counts, simple_counts)]
        
        # Write comparison results to CSV
        comparison_file = os.path.join(args.output, "comparison.csv")
        print(f"Writing comparison results to {comparison_file}...")
        with open(comparison_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "cellpose_count", "simple_count", "difference"])
            for img_path, cp_count, sc_count, diff in zip(png_files, cellpose_counts, simple_counts, differences):
                writer.writerow([
                    os.path.basename(img_path),
                    cp_count,
                    sc_count,
                    diff
                ])
        
        # Write individual method results
        cellpose_file = os.path.join(args.output, "cellpose_counts.csv")
        with open(cellpose_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "cell_count"])
            for img_path, count in zip(png_files, cellpose_counts):
                writer.writerow([os.path.basename(img_path), count])
        
        simple_file = os.path.join(args.output, "simple_counts.csv")
        with open(simple_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "cell_count"])
            for img_path, count in zip(png_files, simple_counts):
                writer.writerow([os.path.basename(img_path), count])
        
        # Print summary statistics
        print("\nComparison Summary:")
        print(f"Total images processed: {len(png_files)}")
        print(f"Average Cellpose count: {sum(cellpose_counts) / len(cellpose_counts):.2f}")
        print(f"Average Simple count: {sum(simple_counts) / len(simple_counts):.2f}")
        print(f"Average difference: {sum(differences) / len(differences):.2f}")
        print(f"Maximum difference: {max(abs(d) for d in differences)}")
        print(f"Images with differences: {sum(1 for d in differences if d != 0)}")
        print(f"\nResults written to:")
        print(f"- Comparison: {comparison_file}")
        print(f"- Cellpose counts: {cellpose_file}")
        print(f"- Simple counts: {simple_file}")
        
    else:
        # Use single method as before
        if args.method == "cellpose":
            cell_counts = count_nuclei_cellpose(
                images,
                use_gpu=args.use_gpu,
                diameter=args.diameter,
            )
            output_file = os.path.join(args.output, "cellpose_counts.csv")
        else:  # simple method
            cell_counts = count_nuclei_simple(
                images,
                min_area=args.min_area,
            )
            output_file = os.path.join(args.output, "simple_counts.csv")

        # Write results to CSV
        print(f"Writing results to {output_file}...")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "cell_count"])
            for img_path, count in zip(png_files, cell_counts):
                writer.writerow([os.path.basename(img_path), count])

    print("Done!")
    return 0


if __name__ == "__main__":
    main() 