import argparse
import os
from typing import List
from PIL import Image
from ..core.inference import (
    load_model,
    load_label_mappings,
    preprocess_image,
    predict,
    plot_predictions
)

def main():
    parser = argparse.ArgumentParser(description='Run inference on images using a trained model')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('mappings_path', type=str, help='Path to the label mappings JSON file')
    parser.add_argument('image_paths', type=str, nargs='+', help='Paths to the images to process')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes in the model')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate used during training')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224], help='Size to resize images to (height width)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--confidence-threshold', type=float, default=0.9, help='Minimum confidence required for a valid prediction')
    parser.add_argument('--filter-label', type=str, help='Only show predictions for this specific label')
    parser.add_argument('--ground-truths', type=str, nargs='+', help='Ground truth labels for each image')
    
    args = parser.parse_args()
    
    # Load model and label mappings
    model = load_model(args.model_path, args.num_classes, args.dropout_rate)
    label_to_idx, idx_to_label = load_label_mappings(args.mappings_path)
    
    # Process each image
    images = []
    predictions = []
    
    for image_path in args.image_paths:
        # Preprocess image
        image_tensor, original_image = preprocess_image(
            image_path,
            tuple(args.image_size)
        )
        
        # Make prediction
        preds = predict(
            model,
            image_tensor,
            idx_to_label,
            args.device,
            args.confidence_threshold
        )
        
        images.append(original_image)
        predictions.append(preds)
    
    # Plot predictions
    plot_predictions(
        images,
        predictions,
        args.ground_truths,
        args.confidence_threshold,
        args.filter_label
    )

if __name__ == '__main__':
    main() 