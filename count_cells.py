#!/usr/bin/env python
"""
Script to count cells in an image using the trained model.
"""
import argparse
import cv2
import numpy as np
import torch
from cell_counter.model import CellCounter

def count_cells_in_image(image_path, model_path="cell_counter_model.pth", max_cells=10, visualize=True):
    """
    Count cells in an image using the trained model.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        max_cells: Maximum number of cells the model can detect
        visualize: Whether to visualize the result
    
    Returns:
        Number of cells detected
    """
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCounter(max_cells=max_cells).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Create a copy for visualization
    vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Resize to match training size (64x64)
    resized_image = cv2.resize(gray_image, (64, 64))
    
    # Convert to tensor and normalize
    image_tensor = torch.FloatTensor(resized_image).unsqueeze(0).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        count = model.predict_count(image_tensor)
    
    # Visualize the result
    if visualize:
        # Add text with the count
        cv2.putText(vis_image, f"Cells: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the image
        cv2.imshow("Cell Count Result", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the result
        output_path = image_path.rsplit('.', 1)[0] + '_result.png'
        cv2.imwrite(output_path, vis_image)
        print(f"Result saved to {output_path}")
    
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count cells in an image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model", default="cell_counter_model.pth", help="Path to the trained model")
    parser.add_argument("--max-cells", type=int, default=10, help="Maximum number of cells the model can detect")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    count = count_cells_in_image(args.image_path, args.model, args.max_cells, not args.no_visualize)
    print(f"Detected {count} cells in the image.") 