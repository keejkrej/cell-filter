"""
Test script for ImageHandler functionality.

Example commands:
# Test ND2 mode
python tests/test_imagehandler.py --patterns data/patterns.tif --nd2 data/test.nd2 --output output/

# Test nuclei only mode
python tests/test_imagehandler.py --patterns data/patterns.tif --nuclei data/nuclei.tif --output output/

# Test cyto only mode
python tests/test_imagehandler.py --patterns data/patterns.tif --cyto data/cyto.tif --output output/

# Test both nuclei and cyto mode
python tests/test_imagehandler.py --patterns data/patterns.tif --nuclei data/nuclei.tif --cyto data/cyto.tif --output output/
"""

import argparse
import numpy as np
from pathlib import Path
from cell_counter.core.impl.ImageHandler import ImageHandler

def main():
    parser = argparse.ArgumentParser(description='Test ImageHandler functionality')
    parser.add_argument('--patterns', type=str, required=True,
                      help='Path to patterns file')
    parser.add_argument('--nuclei', type=str,
                      help='Path to nuclei stack')
    parser.add_argument('--cyto', type=str,
                      help='Path to cyto stack')
    parser.add_argument('--nd2', type=str,
                      help='Path to ND2 file')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for saving images')
    
    args = parser.parse_args()
    
    # Determine mode based on provided files
    if args.nd2 is not None:
        mode = 'nd2'
        if args.nuclei is not None or args.cyto is not None:
            print("Warning: ND2 mode selected, ignoring nuclei and cyto paths")
    elif args.nuclei is not None and args.cyto is not None:
        mode = 'both'
    elif args.nuclei is not None:
        mode = 'nuclei'
    elif args.cyto is not None:
        mode = 'cyto'
    else:
        parser.error("At least one of --nuclei, --cyto, or --nd2 must be provided")
    
    # Initialize ImageHandler based on mode
    if mode == 'nuclei':
        handler = ImageHandler(
            patterns_path=Path(args.patterns),
            nuclei_path=Path(args.nuclei),
            output_dir=Path(args.output)
        )
    elif mode == 'cyto':
        handler = ImageHandler(
            patterns_path=Path(args.patterns),
            cyto_path=Path(args.cyto),
            output_dir=Path(args.output)
        )
    elif mode == 'both':
        handler = ImageHandler(
            patterns_path=Path(args.patterns),
            nuclei_path=Path(args.nuclei),
            cyto_path=Path(args.cyto),
            output_dir=Path(args.output)
        )
    else:  # nd2 mode
        handler = ImageHandler(
            patterns_path=Path(args.patterns),
            nd2_path=Path(args.nd2),
            output_dir=Path(args.output)
        )
    
    # Test basic functionality
    print("\nTesting ImageHandler functionality:")
    print(f"Mode: {mode} (automatically determined)")
    print(f"Patterns loaded: {handler.get_patterns() is not None}")
    print(f"Number of frames: {handler.get_n_frames()}")
    print(f"Number of views: {handler.get_n_views()}")
    
    # Test frame access
    if mode in ['nuclei', 'both']:
        frame = handler.get_nuclei_frame(0, 0)
        print(f"Nuclei frame shape: {frame.shape if frame is not None else 'None'}")
    if mode in ['cyto', 'both']:
        frame = handler.get_cyto_frame(0, 0)
        print(f"Cyto frame shape: {frame.shape if frame is not None else 'None'}")
    
    # Test saving
    if mode in ['nuclei', 'both']:
        frame = handler.get_nuclei_frame(0, 0)
        if frame is not None:
            saved_path = handler.save(frame, "test_nuclei.tif")
            print(f"Saved nuclei frame to: {saved_path}")
    
    if mode in ['cyto', 'both']:
        frame = handler.get_cyto_frame(0, 0)
        if frame is not None:
            saved_path = handler.save(frame, "test_cyto.tif")
            print(f"Saved cyto frame to: {saved_path}")

if __name__ == '__main__':
    main() 