import argparse
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
# Set QT platform to XCB
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_model(model_path: str, num_classes: int, dropout_rate: float = 0.5) -> nn.Module:
    """
    Load a trained ResNet model.
    
    Args:
        model_path (str): Path to the saved model weights
        num_classes (int): Number of classes in the model
        dropout_rate (float): Dropout rate used during training
    
    Returns:
        nn.Module: Loaded model
    """
    # Initialize model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    # Load weights with weights_only=True for security
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    return model

def load_label_mappings(mappings_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load label mappings from JSON file.
    
    Args:
        mappings_path (str): Path to the label mappings JSON file
    
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: Label to index and index to label mappings
    """
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    
    # Print debug information about the mappings
    print("\nLabel mapping information:")
    print(f"Number of classes: {len(mappings['idx_to_label'])}")
    print("Available indices and labels:")
    for idx, label in sorted(mappings['idx_to_label'].items()):
        print(f"  Index {idx}: {label}")
    
    return mappings['label_to_idx'], mappings['idx_to_label']

def preprocess_image(
    image_path: str,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocess an image for inference.
    
    Args:
        image_path (str): Path to the image
        image_size (Tuple[int, int]): Size to resize images to (height, width)
    
    Returns:
        Tuple[torch.Tensor, Image.Image]: Preprocessed image tensor and original image
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be opened or processed
        RuntimeError: If the image dimensions are invalid
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Try to open the image
        try:
            print(f"Attempting to open image: {image_path}")
            image = Image.open(image_path)
            print(f"Image opened successfully. Mode: {image.mode}, Size: {image.size}")
            image = image.convert('RGB')
            print(f"Image converted to RGB. New size: {image.size}")
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            file_name, line_num, func_name, text = tb[-1]
            print(f"Error details while opening image: {type(e).__name__}: {str(e)}")
            print(f"Error occurred in {file_name}, line {line_num}: {text}")
            raise ValueError(f"Failed to open image {image_path}: {str(e)}")
        
        # Check image dimensions
        if image.size[0] == 0 or image.size[1] == 0:
            raise RuntimeError(f"Invalid image dimensions in {image_path}: {image.size}")
        
        original_image = image.copy()
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformations
        try:
            print(f"Applying transformations to image of size {image.size}")
            image_tensor = transform(image)
            print(f"Transformations applied successfully. Tensor shape: {image_tensor.shape}")
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            file_name, line_num, func_name, text = tb[-1]
            print(f"Error details during transformation: {type(e).__name__}: {str(e)}")
            print(f"Error occurred in {file_name}, line {line_num}: {text}")
            raise RuntimeError(f"Failed to transform image {image_path}: {str(e)}")
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_image
        
    except Exception as e:
        # Add more context to the error message
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        file_name, line_num, func_name, text = tb[-1]
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else repr(e)
        full_error = f"Error processing {image_path}:\nType: {error_type}\nMessage: {error_msg}\nFile: {file_name}\nLine: {line_num}\nCode: {text}"
        if isinstance(e, (FileNotFoundError, ValueError, RuntimeError)):
            raise type(e)(full_error)
        else:
            raise RuntimeError(full_error)

def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    idx_to_label: Dict[int, str],
    device: str = 'cuda',
    confidence_threshold: float = 0.9
) -> List[Dict[str, float]]:
    """
    Make predictions on an image.
    
    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        idx_to_label (Dict[int, str]): Index to label mapping
        device (str): Device to run inference on
        confidence_threshold (float): Minimum confidence required for a valid prediction
    
    Returns:
        List[Dict[str, float]]: List of predictions with probabilities
    """
    try:
        # Move model and input to device
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=len(idx_to_label))
        
        # Print debug information about the predictions
        print("\nModel output information:")
        print(f"Output shape: {outputs.shape}")
        print(f"Number of classes in output: {outputs.shape[1]}")
        print(f"Top indices predicted: {top_indices[0].tolist()}")
        print(f"Top probabilities: {top_probs[0].tolist()}")
        
        # Convert to list of dictionaries
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            try:
                # Convert integer index to string for lookup
                idx_str = str(idx.item())
                label = idx_to_label[idx_str]
                predictions.append({
                    'label': label,
                    'probability': prob.item()
                })
            except KeyError as e:
                print(f"Warning: Index {idx_str} not found in idx_to_label mapping")
                print(f"Available indices: {list(idx_to_label.keys())}")
                continue
            except Exception as e:
                import traceback
                tb = traceback.extract_tb(e.__traceback__)
                file_name, line_num, func_name, text = tb[-1]
                print(f"Error in prediction processing: {type(e).__name__}: {str(e)}")
                print(f"Error occurred in {file_name}, line {line_num}: {text}")
                continue
        
        # Check if highest confidence meets threshold
        if predictions and predictions[0]['probability'] < confidence_threshold:
            print(f"Warning: Low confidence prediction ({predictions[0]['probability']:.2f}) below threshold ({confidence_threshold})")
            return [{'label': 'Failed Prediction', 'probability': predictions[0]['probability']}]
        
        return predictions
        
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        file_name, line_num, func_name, text = tb[-1]
        error_msg = f"Error during prediction: {type(e).__name__}: {str(e)}\nFile: {file_name}\nLine: {line_num}\nCode: {text}"
        print(error_msg)
        raise RuntimeError(error_msg)

def plot_predictions(
    images: List[Image.Image],
    predictions: List[List[Dict[str, float]]],
    ground_truths: List[str] = None,
    confidence_threshold: float = 0.9,
    filter_label: str = None
):
    """
    Save predictions to files with correct/wrong indication.
    
    Args:
        images (List[Image.Image]): List of original images
        predictions (List[List[Dict[str, float]]]): List of predictions for each image
        ground_truths (List[str]): List of ground truth labels for each image
        confidence_threshold (float): Minimum confidence required for a valid prediction
        filter_label (str): Only show predictions for this specific label
    """
    print(f"\nDebug - Saving predictions:")
    print(f"Number of input images: {len(images)}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Filter label: {filter_label}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Create output directory
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter images and predictions based on the filter_label
    filtered_data = []
    for i, (img, preds) in enumerate(zip(images, predictions)):
        # Only include if there's an exact match for the filter label
        if not filter_label or any(p['label'] == filter_label and p['probability'] >= confidence_threshold for p in preds):
            filtered_data.append((img, preds, ground_truths[i] if ground_truths else None))
    
    print(f"Number of images after filtering: {len(filtered_data)}")
    
    if not filtered_data:
        print("\nNo images match the filter criteria. No files saved.")
        return
    
    # Unzip the filtered data
    filtered_images, filtered_preds, filtered_truths = zip(*filtered_data)
    
    # Save each image with its prediction
    for i in range(len(filtered_images)):
        try:
            if filtered_preds[i] and len(filtered_preds[i]) > 0:
                # If filter_label is set, find the matching prediction
                if filter_label:
                    matching_preds = [p for p in filtered_preds[i] if p['label'] == filter_label]
                    if matching_preds:
                        pred = matching_preds[0]
                    else:
                        continue
                else:
                    pred = filtered_preds[i][0]
                
                if pred['probability'] < confidence_threshold:
                    continue
                
                # Determine if prediction is correct
                is_correct = False
                if filtered_truths[i]:
                    is_correct = pred['label'] == filtered_truths[i]
                
                # Create filename with prediction info
                label = pred['label'].replace(' ', '_')
                confidence = f"{pred['probability']:.2f}"
                status = "correct" if is_correct else "wrong"
                filename = f"pred_{i}_{label}_{confidence}_{status}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Create figure and save
                fig = plt.figure(figsize=(10, 8))
                plt.imshow(filtered_images[i])
                plt.axis('off')
                
                # Create title with correct/wrong indication
                title = f"{pred['label']}\nConfidence: {pred['probability']:.2f}"
                if filtered_truths[i]:
                    title += f"\nStatus: {status.upper()}"
                    if not is_correct:
                        title += f" (Ground Truth: {filtered_truths[i]})"
                
                plt.title(title, fontsize=12, color='green' if is_correct else 'red')
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()
                
                print(f"Saved prediction to {filepath}")
                
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            file_name, line_num, func_name, text = tb[-1]
            print(f"Error saving prediction for image {i}: {type(e).__name__}: {str(e)}")
            print(f"Error occurred in {file_name}, line {line_num}: {text}")
            continue
    
    print(f"\nAll matching predictions saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Run inference on images using a trained cell counting model')
    parser.add_argument('model_dir', type=str, help='Directory containing the trained model and label mappings')
    parser.add_argument('image_path', type=str, help='Path to the image or directory of images to predict')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224], help='Size to resize images to (height width)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run inference on')
    parser.add_argument('--output', type=str, help='Path to save predictions (JSON file)')
    parser.add_argument('--confidence-threshold', type=float, default=0.9, help='Minimum confidence required for a valid prediction')
    parser.add_argument('--filter-label', type=str, help='Only show predictions for this specific label')
    parser.add_argument('--ground-truth', type=str, help='Path to JSON file containing ground truth labels')
    
    args = parser.parse_args()
    
    # Load ground truth labels if provided
    ground_truths = None
    if args.ground_truth:
        try:
            with open(args.ground_truth, 'r') as f:
                json_data_list = json.load(f)
            
            # Convert the list of annotations to a dictionary of image_path -> label
            ground_truths = {}
            for json_data in json_data_list:
                # Extract the filename from data.image
                filename = json_data["data"]["image"]
                
                # Get the annotations
                annotations = json_data["annotations"]
                
                # For each annotation, extract the label
                for annotation in annotations:
                    for result in annotation["result"]:
                        if result["type"] == "choices":
                            label = result["value"]["choices"][0]  # Get the first choice
                            ground_truths[filename] = label
            
            print(f"Loaded ground truth labels from {args.ground_truth}")
            
            # Validate that all labels exist in the label mappings
            model_dir = Path(args.model_dir)
            mappings_path = model_dir / 'label_mappings.json'
            if not mappings_path.exists():
                raise FileNotFoundError(f"Label mappings not found at {mappings_path}")
            
            with open(mappings_path, 'r') as f:
                label_mappings = json.load(f)
                available_labels = set(label_mappings['idx_to_label'].values())
                
                # Check each ground truth label
                for img_path, label in ground_truths.items():
                    if label not in available_labels:
                        print(f"Warning: Ground truth label '{label}' for {img_path} not found in model's label mappings")
                        print(f"Available labels: {sorted(available_labels)}")
            
        except Exception as e:
            print(f"Error loading ground truth labels: {e}")
            ground_truths = None
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
    
    # Load model and label mappings
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    mappings_path = os.path.join(args.model_dir, 'label_mappings.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Label mappings not found at {mappings_path}")
    
    label_to_idx, idx_to_label = load_label_mappings(mappings_path)
    
    # Validate filter label if provided
    if args.filter_label and args.filter_label not in label_to_idx:
        available_labels = list(label_to_idx.keys())
        raise ValueError(f"Invalid filter label: {args.filter_label}. Available labels: {available_labels}")
    
    model = load_model(model_path, len(label_to_idx))
    
    # Process single image or directory of images
    image_path = Path(args.image_path)
    predictions = {}
    images = []
    preds_list = []
    error_files = []
    filtered_images = []  # Keep track of images that match the filter
    truth_list = []  # Keep track of ground truth labels
    
    if image_path.is_file():
        # Single image
        try:
            print(f"\nProcessing single image: {image_path}")
            image_tensor, original_image = preprocess_image(
                str(image_path),
                tuple(args.image_size)
            )
            preds = predict(model, image_tensor, idx_to_label, args.device, args.confidence_threshold)
            
            # Only include if no filter or matches filter
            if not args.filter_label or any(p['label'] == args.filter_label for p in preds):
                predictions[str(image_path)] = preds
                images.append(original_image)
                preds_list.append(preds)
                filtered_images.append(str(image_path))
                if ground_truths:
                    truth_list.append(ground_truths.get(str(image_path)))
            
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            file_name, line_num, func_name, text = tb[-1]
            error_msg = f"{str(e)}\nFile: {file_name}\nLine: {line_num}\nCode: {text}"
            error_files.append((str(image_path), error_msg))
            print(f"Error processing {image_path}:\n{error_msg}")
        
    elif image_path.is_dir():
        # Directory of images
        print(f"\nProcessing directory: {image_path}")
        for img_file in image_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    print(f"\nProcessing image: {img_file}")
                    image_tensor, original_image = preprocess_image(
                        str(img_file),
                        tuple(args.image_size)
                    )
                    preds = predict(model, image_tensor, idx_to_label, args.device, args.confidence_threshold)
                    
                    # Only include if no filter or matches filter
                    if not args.filter_label or any(p['label'] == args.filter_label for p in preds):
                        predictions[str(img_file)] = preds
                        images.append(original_image)
                        preds_list.append(preds)
                        filtered_images.append(str(img_file))
                        if ground_truths:
                            truth_list.append(ground_truths.get(str(img_file)))
                    
                except Exception as e:
                    import traceback
                    tb = traceback.extract_tb(e.__traceback__)
                    file_name, line_num, func_name, text = tb[-1]
                    error_msg = f"{str(e)}\nFile: {file_name}\nLine: {line_num}\nCode: {text}"
                    error_files.append((str(img_file), error_msg))
                    print(f"Error processing {img_file}:\n{error_msg}")
    
    else:
        raise ValueError(f"Invalid image path: {args.image_path}")
    
    # Print summary of processed files
    if args.filter_label:
        print(f"\nFound {len(filtered_images)} images matching label '{args.filter_label}'")
    else:
        print(f"\nProcessed {len(predictions)} images successfully")
    
    if error_files:
        print(f"Failed to process {len(error_files)} images:")
        for file_path, error in error_files:
            print(f"  {file_path}: {error}")
    
    # Print predictions
    for img_path, preds in predictions.items():
        print(f"\nPredictions for {img_path}:")
        for pred in preds:
            if not args.filter_label or pred['label'] == args.filter_label:
                if pred['label'] == 'Failed Prediction':
                    print(f"  {pred['label']} (confidence: {pred['probability']:.4f})")
                else:
                    print(f"  {pred['label']}: {pred['probability']:.4f}")
    
    # Save predictions if output path is specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\nPredictions saved to {args.output}")
    
    # Plot predictions if we have any successful images
    if images:
        plot_predictions(
            images,
            preds_list,
            ground_truths=truth_list,
            confidence_threshold=args.confidence_threshold,
            filter_label=args.filter_label
        )
    else:
        if args.filter_label:
            print(f"\nNo images found with label '{args.filter_label}'. Cannot create plot.")
        else:
            print("\nNo images were successfully processed. Cannot create plot.")

if __name__ == '__main__':
    main() 