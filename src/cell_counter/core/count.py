import numpy as np
from typing import List, Union
from cellpose import models
import cv2

def count_nuclei_cellpose(
    imgs: Union[np.ndarray, List[np.ndarray]],
    use_gpu: bool = True,
    diameter: int = 15,
    channels: List[int] = [0, 0],
    model_type: str = "cyto3"
) -> List[int]:
    """
    Count nuclei in images using Cellpose model.
    
    Args:
        imgs: Single image or list of images as numpy arrays
        use_gpu: Whether to use GPU for inference
        diameter: Expected diameter of cells in pixels
        channels: List of channel indices for the model
        model_type: Type of Cellpose model to use
        
    Returns:
        List of integers representing the number of nuclei in each image
        
    Raises:
        ValueError: If input images are invalid
        RuntimeError: If model fails to process images
    """
    try:
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
            
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise ValueError("All images must be numpy arrays")
            
        model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        masks_pred, flows, styles, diams = model.eval(
            imgs,
            diameter=diameter,
            channels=channels
        )
        num_nuclei = [int(np.max(mask)) for mask in masks_pred]
        return num_nuclei
        
    except Exception as e:
        raise RuntimeError(f"Failed to process images: {str(e)}")

def count_nuclei_simple(
    imgs: Union[np.ndarray, List[np.ndarray]],
    min_area: int = 50
) -> List[int]:
    """
    Count nuclei in images using simple thresholding and contour detection.
    
    Args:
        imgs: Single image or list of images as numpy arrays
        min_area: Minimum area in pixels to consider as a nucleus
        
    Returns:
        List of integers representing the number of nuclei in each image
    """
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    
    counts = []
    for img in imgs:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        counts.append(len(valid_contours))
    
    return counts

def count_nuclei(
    imgs: Union[np.ndarray, List[np.ndarray]],
    use_cellpose: bool = True,
    use_gpu: bool = True,
    diameter: int = 15,
    channels: List[int] = [0, 0],
    model_type: str = "cyto3"
) -> List[int]:
    """
    Count nuclei in images using either Cellpose or simple thresholding.
    
    Args:
        imgs: Single image or list of images as numpy arrays
        use_cellpose: Whether to use Cellpose (True) or simple thresholding (False)
        use_gpu: Whether to use GPU for Cellpose inference
        diameter: Expected diameter of cells in pixels (for Cellpose)
        channels: List of channel indices for the model (for Cellpose)
        model_type: Type of Cellpose model to use
        
    Returns:
        List of integers representing the number of nuclei in each image
    """
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    
    if use_cellpose:
        return count_nuclei_cellpose(imgs, use_gpu, diameter, channels, model_type)
    else:
        return count_nuclei_simple(imgs)