"""
Load and display labeled cell counting images.
"""

import os
import argparse
from pathlib import Path

from ..core.load_data import load_and_visualize_data, show_batch


def main():
    parser = argparse.ArgumentParser(description="Load and display labeled cell counting images")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing annotations")
    parser.add_argument("image_dir", type=str, help="Directory containing the images")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of images to display (default: 16)")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle the data")
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64], 
                       help="Size to resize images to (height width) (default: 64 64)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.json_file).exists():
        print(f"Error: JSON file '{args.json_file}' does not exist")
        return
        
    if not Path(args.image_dir).exists():
        print(f"Error: Image directory '{args.image_dir}' does not exist")
        return
    
    # Load and visualize data
    dataloader, label_to_idx, idx_to_label, images, labels = load_and_visualize_data(
        json_file=args.json_file,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        image_size=tuple(args.image_size)
    )
    
    # Print label mappings
    print("Label to index mapping:")
    for label, idx in label_to_idx.items():
        print(f"{label}: {idx}")
    
    # Print batch information
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Display the batch
    show_batch(images, labels, idx_to_label)


if __name__ == "__main__":
    main() 