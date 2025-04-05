import os
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .parse_labels import parse_labels_from_file

class CellCounterDataset(Dataset):
    """
    Dataset for cell counting images and their labels.
    """
    def __init__(
        self,
        json_file: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        model_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the dataset.
        
        Args:
            json_file (str): Path to the JSON file containing annotations
            image_dir (str): Directory containing the images
            transform (Optional[transforms.Compose]): Transformations to apply to images
            label_to_idx (Optional[Dict[str, int]]): Mapping from label strings to indices
            image_size (Tuple[int, int]): Size to resize images to (height, width)
            model_size (Tuple[int, int]): Size expected by the model (height, width)
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.model_size = model_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Parse the labels
        self.image_label_pairs = parse_labels_from_file(json_file)
        
        # Create label to index mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(set(label for _, label in self.image_label_pairs))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        # Create index to label mapping
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self) -> int:
        return len(self.image_label_pairs)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            Tuple[torch.Tensor, int]: Image tensor and label index
        """
        filename, label = self.image_label_pairs[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        return image, label_idx
        
    def get_label_mapping(self) -> Dict[str, int]:
        """Get the label to index mapping."""
        return self.label_to_idx
        
    def get_index_mapping(self) -> Dict[int, str]:
        """Get the index to label mapping."""
        return self.idx_to_label

def create_dataloader(
    json_file: str,
    image_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
    label_to_idx: Optional[Dict[str, int]] = None,
    image_size: Tuple[int, int] = (64, 64),
    model_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Create a DataLoader for the cell counting dataset.
    
    Args:
        json_file (str): Path to the JSON file containing annotations
        image_dir (str): Directory containing the images
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        transform (Optional[transforms.Compose]): Transformations to apply to images
        label_to_idx (Optional[Dict[str, int]]): Mapping from label strings to indices
        image_size (Tuple[int, int]): Size to resize images to (height, width)
        model_size (Tuple[int, int]): Size expected by the model (height, width)
        
    Returns:
        Tuple[DataLoader, Dict[str, int], Dict[int, str]]: DataLoader and label mappings
    """
    dataset = CellCounterDataset(
        json_file=json_file,
        image_dir=image_dir,
        transform=transform,
        label_to_idx=label_to_idx,
        image_size=image_size,
        model_size=model_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset.get_label_mapping(), dataset.get_index_mapping() 