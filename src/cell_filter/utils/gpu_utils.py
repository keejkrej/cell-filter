"""
GPU validation utilities for cell-filter segmentation capabilities.

This module provides centralized GPU/CUDA validation that should be called
at application entrypoints (CLI commands, scripts) before any segmentation
operations are attempted.
"""

import logging

logger = logging.getLogger(__name__)


class GPUValidationError(Exception):
    """Raised when GPU validation fails."""
    pass


def validate_gpu_availability(require_gpu: bool = True) -> bool:
    """
    Validate GPU/CUDA availability for segmentation operations.

    This function should be called at application entrypoints before
    initializing any segmentation components.

    Parameters
    ----------
    require_gpu : bool, default True
        Whether GPU is required for the operation. If True and no GPU
        is available, raises GPUValidationError.

    Returns
    -------
    bool
        True if GPU is available, False otherwise

    Raises
    ------
    GPUValidationError
        If require_gpu=True and GPU is not available
    """
    gpu_available = _check_cuda_availability()

    if require_gpu and not gpu_available:
        error_msg = (
            "GPU/CUDA is required for segmentation operations but not available. "
            "Please ensure CUDA is properly installed and a compatible GPU is present."
        )
        logger.error(error_msg)
        raise GPUValidationError(error_msg)

    if gpu_available:
        logger.info("GPU/CUDA validation successful")
    else:
        logger.warning("GPU/CUDA not available - falling back to CPU mode")

    return gpu_available


def _check_cuda_availability() -> bool:
    """
    Check if CUDA is available for PyTorch operations.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not available - cannot check CUDA status")
        return False
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}")
        return False


def get_gpu_info() -> dict | None:
    """
    Get information about available GPUs.

    Returns
    -------
        dict | None
        Dictionary containing GPU information, or None if no GPU available
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        gpu_info = {
            'cuda_available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_total': torch.cuda.get_device_properties(0).total_memory,
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_cached': torch.cuda.memory_reserved(),
        }

        return gpu_info

    except ImportError:
        logger.debug("PyTorch not available for GPU info")
        return None
    except Exception as e:
        logger.debug(f"Error getting GPU info: {e}")
        return None


def validate_segmentation_requirements(enable_segmentation: bool = True) -> dict:
    """
    Validate all requirements for segmentation operations.

    This is the main validation function that should be called at
    application entrypoints when segmentation is enabled.

    Parameters
    ----------
    enable_segmentation : bool, default True
        Whether segmentation features are enabled

    Returns
    -------
    dict
        Validation results containing:
        - 'gpu_available': bool
        - 'segmentation_enabled': bool
        - 'gpu_info': dict or None

    Raises
    ------
    GPUValidationError
        If segmentation is enabled but GPU requirements are not met
    """
    if not enable_segmentation:
        logger.info("Segmentation disabled - skipping GPU validation")
        return {
            'gpu_available': False,
            'segmentation_enabled': False,
            'gpu_info': None
        }

    # Validate GPU availability
    gpu_available = validate_gpu_availability(require_gpu=True)
    gpu_info = get_gpu_info()

    if gpu_info:
        logger.info(f"Using GPU: {gpu_info['device_name']} "
                   f"({gpu_info['memory_total'] / 1e9:.1f}GB total memory)")

    return {
        'gpu_available': gpu_available,
        'segmentation_enabled': enable_segmentation,
        'gpu_info': gpu_info
    }


def log_system_info():
    """Log system information relevant to GPU operations."""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            # Check CUDA version using dynamic attribute access
            torch_version = getattr(torch, 'version', None)
            if torch_version:
                cuda_version = getattr(torch_version, 'cuda', 'Unknown')
                logger.info(f"CUDA version: {cuda_version}")
            else:
                logger.info("CUDA version: Unknown")

            # Check cuDNN version safely
            if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'version'):
                try:
                    cudnn_version = torch.backends.cudnn.version()
                    logger.info(f"cuDNN version: {cudnn_version}")
                except Exception:
                    logger.info("cuDNN version: Unknown")
            else:
                logger.info("cuDNN version: Unknown")

    except ImportError:
        logger.info("PyTorch not available")
    except Exception as e:
        logger.debug(f"Error logging system info: {e}")
