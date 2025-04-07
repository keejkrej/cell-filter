"""
Overlay creation command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Basic usage (nuclei in red, cytoplasm in green)
    python -m cell_counter.cli.overlay --nuclei <nuclei_path> --cyto <cyto_path> --output <output_path>
    
    # Process folders
    python -m cell_counter.cli.overlay --nucleifolder <nuclei_folder> --cytofolder <cyto_folder> --output <output_folder>
    
    # Custom channels
    python -m cell_counter.cli.overlay --nuclei <nuclei_path> --cyto <cyto_path> --output <output_path> --ch-nuclei 2 --ch-cyto 0
    
Optional arguments:
- `--ch-nuclei`: Channel index (0=red, 1=green, 2=blue) for the nuclei image (default: 0)
- `--ch-cyto`: Channel index (0=red, 1=green, 2=blue) for the cytoplasm image (default: 1)
"""

import argparse
import os
import sys
from pathlib import Path
import re
from ..core.OverlayCreator import OverlayCreator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create RGB overlays from nuclei and cytoplasm images."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--nuclei",
        type=str,
        help="Path to the nuclei image",
    )
    group.add_argument(
        "--nucleifolder",
        type=str,
        help="Folder containing nuclei images",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cyto",
        type=str,
        help="Path to the cytoplasm image",
    )
    group.add_argument(
        "--cytofolder",
        type=str,
        help="Folder containing cytoplasm images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output TIFF file or folder",
    )
    parser.add_argument(
        "--ch-nuclei",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Channel index (0=red, 1=green, 2=blue) for the nuclei image (default: 0)",
    )
    parser.add_argument(
        "--ch-cyto",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Channel index (0=red, 1=green, 2=blue) for the cytoplasm image (default: 1)",
    )
    return parser.parse_args()

def get_matching_files(folder1, folder2):
    """
    Find matching files between two folders based on the numeric part of their names.
    For example, matches 'xxx_123.tif' with 'yyy_123.tif'.
    
    Args:
        folder1 (str): Path to first folder
        folder2 (str): Path to second folder
        
    Returns:
        list: List of tuples (file1, file2, name) where name is the numeric part
    """
    # Get all TIFF files from both folders
    files1 = list(Path(folder1).glob("*.tif"))
    files2 = list(Path(folder2).glob("*.tif"))
    
    # Find matching files
    matches = []
    for f1 in files1:
        num1 = f1.stem.split('_')[-1]
        for f2 in files2:
            num2 = f2.stem.split('_')[-1]
            if num1 == num2:
                matches.append((f1, f2, num1))
                break
    
    return matches

def main():
    """Main function."""
    args = parse_args()

    # Handle single file case
    if args.nuclei and args.cyto:
        # Check if input files exist
        if not os.path.exists(args.nuclei):
            print(f"Error: Nuclei image file not found: {args.nuclei}")
            sys.exit(1)
        if not os.path.exists(args.cyto):
            print(f"Error: Cytoplasm image file not found: {args.cyto}")
            sys.exit(1)

        # Create output directory if it doesn't exist
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create overlay
        creator = OverlayCreator(
            nuclei_path=args.nuclei,
            cyto_path=args.cyto,
            ch_nuclei=args.ch_nuclei,
            ch_cyto=args.ch_cyto
        )
        creator.create_overlay(args.output)
    
    # Handle folder case
    else:
        # Check if folders exist
        if not os.path.exists(args.nucleifolder):
            print(f"Error: Nuclei folder not found: {args.nucleifolder}")
            sys.exit(1)
        if not os.path.exists(args.cytofolder):
            print(f"Error: Cytoplasm folder not found: {args.cytofolder}")
            sys.exit(1)
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get matching files
        matches = get_matching_files(args.nucleifolder, args.cytofolder)
        if not matches:
            print("Error: No matching files found between the folders")
            sys.exit(1)
        
        # Process each pair
        for file1, file2, name in matches:
            output_path = output_dir / f"{name}.tif"
            print(f"Processing {name}...")
            
            creator = OverlayCreator(
                nuclei_path=str(file1),
                cyto_path=str(file2),
                ch_nuclei=args.ch_nuclei,
                ch_cyto=args.ch_cyto
            )
            creator.create_overlay(str(output_path))

if __name__ == "__main__":
    main() 