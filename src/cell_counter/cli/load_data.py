import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from cell_counter.utils.dataloader import create_dataloader

def show_batch(images: torch.Tensor, labels: torch.Tensor, idx_to_label: dict, nrow: int = 4):
    """
    Display a batch of images with their labels.
    
    Args:
        images (torch.Tensor): Batch of images
        labels (torch.Tensor): Batch of labels
        idx_to_label (dict): Mapping from indices to labels
        nrow (int): Number of images per row in the grid
    """
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    
    # Create grid of images
    grid = make_grid(images, nrow=nrow, padding=2)
    
    # Convert to numpy and transpose for matplotlib
    grid = grid.permute(1, 2, 0).numpy()
    
    # Create figure and display images
    plt.figure(figsize=(6, 6))
    plt.imshow(grid)
    
    # Calculate image dimensions in the grid
    img_height = images.shape[2]
    img_width = images.shape[3]
    
    # Add labels
    for i, label_idx in enumerate(labels):
        row = i // nrow
        col = i % nrow
        
        # Calculate position for the label
        x = col * (img_width + 4) + img_width // 2  # Center of the image
        y = row * (img_height + 4) + 20  # 20 pixels from the top of each image
        
        label = idx_to_label[label_idx.item()]
        plt.text(x, y, 
                f"Label: {label}", 
                color='white', 
                bbox=dict(facecolor='black', alpha=0.5),
                ha='center',  # Horizontal alignment
                va='top',     # Vertical alignment
                fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()

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
    
    # Create dataloader
    dataloader, label_to_idx, idx_to_label = create_dataloader(
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
    
    # Get a batch of data
    images, labels = next(iter(dataloader))
    
    # Print batch information
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Display the batch
    show_batch(images, labels, idx_to_label)

if __name__ == "__main__":
    main() 