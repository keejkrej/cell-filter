#!/usr/bin/env python
"""
Script to train the cell counter model.
"""
import os
import pickle
import torch
import argparse
from cell_counter.train import train_model
from cell_counter.model import CellCounter

def train_from_saved_data(
    data_path="data/simulated/cell_data.pkl",
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    output_model="cell_counter_model.pth"
):
    """
    Train the model using saved simulated data.
    
    Args:
        data_path: Path to the saved data pickle file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        output_model: Path to save the trained model
    """
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run generate_data.py first to create the simulated data.")
        return None
    
    # Load the data
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    images = data["images"]
    counts = data["counts"]
    max_cells = data["max_cells"]
    
    print(f"Loaded {len(images)} images with cell counts ranging from 1 to {max_cells}")
    
    # Train the model
    print("Training cell counter model...")
    model = train_model(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_samples=len(images),  # This will be ignored since we're using saved data
        max_cells=max_cells
    )
    
    # Save the model
    torch.save(model.state_dict(), output_model)
    print(f"Model saved to {output_model}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cell counter model")
    parser.add_argument("--data-path", default="data/simulated/cell_data.pkl", 
                        help="Path to the saved data pickle file")
    parser.add_argument("--num-epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate for optimizer")
    parser.add_argument("--output-model", default="cell_counter_model.pth", 
                        help="Path to save the trained model")
    
    args = parser.parse_args()
    
    train_from_saved_data(
        data_path=args.data_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_model=args.output_model
    ) 