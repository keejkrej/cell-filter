import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
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
    
    # Plot each image with its predictions
    for i, (image, preds) in enumerate(zip(images, predictions)):
        # Skip if no predictions
        if not preds:
            continue
            
        # Filter by label if specified
        if filter_label and preds[0]['label'] != filter_label:
            continue
            
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, width_ratios=[1, 1])
        
        # Plot image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(image)
        ax1.axis('off')
        
        # Plot predictions
        ax2 = fig.add_subplot(gs[1])
        labels = [p['label'] for p in preds]
        probs = [p['probability'] for p in preds]
        
        # Create bar plot
        bars = ax2.barh(range(len(labels)), probs)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Probability')
        
        # Add probability text
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}',
                    ha='left', va='center')
        
        # Add ground truth if available
        if ground_truths and i < len(ground_truths):
            gt = ground_truths[i]
            ax1.set_title(f'Ground Truth: {gt}')
            
            # Check if prediction is correct
            if preds[0]['label'] == gt:
                fig.suptitle('Correct Prediction', color='green')
            else:
                fig.suptitle('Incorrect Prediction', color='red')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_{i}.png'))
        plt.close() 