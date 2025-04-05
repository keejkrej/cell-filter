import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import transforms

from cell_counter.utils.dataloader import create_dataloader

def train_model(
    annotations_path: str,
    images_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    image_size: Tuple[int, int] = (64, 64),
    model_size: Tuple[int, int] = (224, 224),
    weight_decay: float = 0.01,
    dropout_rate: float = 0.3,
    max_grad_norm: float = 1.0,
    confidence_threshold: float = 0.9
):
    """
    Train a cell counting model.
    
    Args:
        annotations_path (str): Path to the annotations JSON file
        images_dir (str): Directory containing the images
        output_dir (str): Directory to save the model and results
        batch_size (int): Batch size for training
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Learning rate
        validation_split (float): Fraction of data to use for validation
        image_size (Tuple[int, int]): Size to resize images to (height, width)
        model_size (Tuple[int, int]): Size to resize images to for model input
        weight_decay (float): L2 regularization strength
        dropout_rate (float): Dropout rate for regularization
        max_grad_norm (float): Maximum gradient norm for clipping
        confidence_threshold (float): Target confidence threshold for predictions
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training requires GPU.")
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataloader, label_to_idx, idx_to_label = create_dataloader(
        annotations_path,
        images_dir,
        batch_size=batch_size,
        image_size=image_size,
        model_size=model_size
    )
    
    # Split into train and validation sets
    dataset_size = len(dataloader.dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataloader.dataset, [train_size, val_size]
    )
    
    # Define data augmentation transforms for training
    train_transform = transforms.Compose([
        transforms.Resize(model_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(model_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(model_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms to datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Save label mappings
    mappings = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label
    }
    with open(os.path.join(output_dir, 'label_mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Initialize model
    print("Initializing model...")
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, len(label_to_idx))
    )
    
    # Move model to GPU
    model = model.to('cuda')
    
    # Define loss function with confidence weighting
    class ConfidenceWeightedCrossEntropy(nn.Module):
        def __init__(self, confidence_threshold: float = 0.9):
            super().__init__()
            self.confidence_threshold = confidence_threshold
            self.base_loss = nn.CrossEntropyLoss()
        
        def forward(self, outputs, targets):
            # Calculate base cross entropy loss
            base_loss = self.base_loss(outputs, targets)
            
            # Calculate confidence
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            
            # Calculate confidence penalty
            confidence_penalty = torch.mean(torch.relu(self.confidence_threshold - max_probs))
            
            # Combine losses
            total_loss = base_loss + 0.5 * confidence_penalty
            
            return total_loss
    
    criterion = ConfidenceWeightedCrossEntropy(confidence_threshold=confidence_threshold)
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    print("Starting training...")
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # Move data to GPU
            images = images.to('cuda', non_blocking=True)
            labels = labels.to('cuda', non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Move data to GPU
                images = images.to('cuda', non_blocking=True)
                labels = labels.to('cuda', non_blocking=True)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_loss += loss.item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"  New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'train_accuracy': epoch_accuracy,
                'val_accuracy': val_accuracy
            }
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Save final results
    results = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'best_val_accuracy': best_val_accuracy
    }
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Train a cell counting model')
    parser.add_argument('annotations_path', type=str, help='Path to the annotations JSON file')
    parser.add_argument('images_dir', type=str, help='Directory containing the images')
    parser.add_argument('output_dir', type=str, help='Directory to save the model and results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--image-size', type=int, nargs=2, default=[64, 64], help='Size to resize images to (height width)')
    parser.add_argument('--model-size', type=int, nargs=2, default=[224, 224], help='Size to resize images to for model input')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization strength')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate for regularization')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--confidence-threshold', type=float, default=0.9, help='Target confidence threshold for predictions')
    
    args = parser.parse_args()
    
    train_model(
        annotations_path=args.annotations_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        image_size=tuple(args.image_size),
        model_size=tuple(args.model_size),
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        max_grad_norm=args.max_grad_norm,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == '__main__':
    main() 