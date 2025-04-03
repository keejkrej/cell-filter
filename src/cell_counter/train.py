import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
from .data_generator import generate_training_dataset
from .model import CellCounter

class CellDataset(Dataset):
    def __init__(self, images: List[np.ndarray], counts: List[int]):
        self.images = images
        self.counts = counts
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]) / 255.0
        count = torch.LongTensor([self.counts[idx]])[0]
        return image, count

def train_model(
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_samples: int = 1000,
    max_cells: int = 10,
    images: Optional[List[np.ndarray]] = None,
    counts: Optional[List[int]] = None
) -> CellCounter:
    """
    Train the cell counter model.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_samples: Number of training samples to generate (ignored if images and counts are provided)
        max_cells: Maximum number of cells per image
        images: Optional pre-generated images (if None, will generate new ones)
        counts: Optional pre-generated counts (if None, will generate new ones)
    
    Returns:
        Trained CellCounter model
    """
    # Generate training data if not provided
    if images is None or counts is None:
        print(f"Generating {num_samples} training samples...")
        images, counts = generate_training_dataset(
            num_samples=num_samples,
            max_cells=max_cells
        )
    else:
        print(f"Using {len(images)} pre-generated training samples")
    
    # Create dataset and dataloader
    dataset = CellDataset(images, counts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter(max_cells=max_cells).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, counts in dataloader:
            images = images.to(device)
            counts = counts.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, counts)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += counts.size(0)
            correct += (predicted == counts).sum().item()
        
        # Print epoch statistics
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
    
    return model 