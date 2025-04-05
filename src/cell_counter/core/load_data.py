"""
Core load_data functionality for cell-counter.
"""

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from ..utils.dataloader import create_dataloader


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


def load_and_visualize_data(json_file, image_dir, batch_size=16, shuffle=True, image_size=(64, 64)):
    """
    Load and visualize labeled cell counting images.
    
    Args:
        json_file (str): Path to the JSON file containing annotations
        image_dir (str): Directory containing the images
        batch_size (int): Number of images to display (default: 16)
        shuffle (bool): Whether to shuffle the data (default: True)
        image_size (tuple): Size to resize images to (height, width) (default: (64, 64))
        
    Returns:
        tuple: (dataloader, label_to_idx, idx_to_label)
    """
    # Create dataloader
    dataloader, label_to_idx, idx_to_label = create_dataloader(
        json_file=json_file,
        image_dir=image_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        image_size=image_size
    )
    
    # Get a batch of data
    images, labels = next(iter(dataloader))
    
    return dataloader, label_to_idx, idx_to_label, images, labels 