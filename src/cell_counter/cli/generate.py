"""
Generate command for cell-counter.
"""

"""
Usage:
    After installing the package with `pip install -e .`, run:
    
    # Generate nuclei data
    python -m cell_counter.cli.generate --patterns <patterns_path> --nuclei <nuclei_path> --output <output_dir>
    
    # Generate cytoplasm data
    python -m cell_counter.cli.generate --patterns <patterns_path> --cyto <cyto_path> --output <output_dir>
    
    # Generate both
    python -m cell_counter.cli.generate --patterns <patterns_path> --nuclei <nuclei_path> --cyto <cyto_path> --output <output_dir>
    
    Optional arguments:
    --frames: Range of frames to process (e.g., "0-5" for frames 0 to 5, "0,2,4" for specific frames)
    --contours: Range of contours to process (e.g., "0-5" for contours 0 to 5, "0,2,4" for specific contours).
                 Contours are sorted in row-major order (left to right, top to bottom).
    
    Example:
    python -m cell_counter.cli.generate --patterns /path/to/patterns.tif --nuclei /path/to/nuclei.tif --cyto /path/to/cyto.tif --output /path/to/output_dir --frames 0-5 --contours 0-10
"""

import argparse
from ..core.generate import generate_data

def main():
    """Main entry point for generate command."""
    parser = argparse.ArgumentParser(description='Generate nuclei and/or cytoplasm data from images')
    parser.add_argument('--patterns', type=str, required=True, help='Path to the patterns image file')
    parser.add_argument('--nuclei', type=str, help='Path to the nuclei image file')
    parser.add_argument('--cyto', type=str, help='Path to the cytoplasm image file')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save extracted data')
    parser.add_argument('--frames', type=str, help='Range of frames to process (e.g., "0-5" or "0,2,4")')
    parser.add_argument(
        "--contours",
        type=str,
        help="Range of contours to process (e.g., '0-5' for contours 0 to 5, '0,2,4' for specific contours). "
             "Contours are sorted in row-major order (left to right, top to bottom).",
    )
    
    args = parser.parse_args()
    
    if not args.nuclei and not args.cyto:
        parser.error("At least one of --nuclei or --cyto must be provided")
    
    # If either range is not specified, ask for confirmation
    if not args.frames or not args.contours:
        print(f"\nWarning: {'Frame' if not args.frames else 'Contour'} range not specified.")
        print(f"This will process all frames and contours.")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    generate_data(
        patterns_path=args.patterns,
        nuclei_path=args.nuclei,
        cyto_path=args.cyto,
        output_dir=args.output,
        frames=args.frames,
        contours=args.contours
    )

if __name__ == '__main__':
    main() 