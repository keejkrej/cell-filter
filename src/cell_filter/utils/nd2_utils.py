"""
Utility functions for handling ND2 files using the nd2 package with xarray and dask.
"""

import numpy as np
import nd2
from pathlib import Path
from typing import TypedDict


class ND2Metadata(TypedDict):
    """Type definition for essential ND2 metadata"""

    filepath: str  # Full path to ND2 file
    filename: str  # Just the filename
    n_frames: int  # Number of time points
    height: int  # Image height in pixels
    width: int  # Image width in pixels
    n_fov: int  # Number of fields of view
    channels: list[str]  # Channel names
    n_channels: int  # Number of channels
    pixel_microns: float  # Pixel size in microns
    native_dtype: str  # Original data type from ND2


def get_nd2_frame(fov: int, channel: int, frame: int, xarr) -> np.ndarray:
    """
    Get a single 2D frame from the ND2 file using lazy loading.

    Args:
        fov: Field of view index
        channel: Channel index (-1 to return all channels)
        frame: Time frame index
        xarr: xarray DataArray instance

    Returns:
        2D numpy array of the requested frame (or 3D array if channel=-1)
    """
    # Build selection dict based on available dimensions
    selection = {}

    if "T" in xarr.dims:
        selection["T"] = frame
    if "P" in xarr.dims:
        selection["P"] = fov
    if "Z" in xarr.dims:
        selection["Z"] = 0  # Default to first Z plane

    # If channel is -1, return all channels
    if channel == -1:
        if "C" in xarr.dims:
            selection["C"] = slice(None)  # Select all channels
        # Select the specific frame and compute to numpy
        # This returns a 3D numpy array (C, H, W) when channel=-1
        return xarr.isel(**selection).compute().values

    # Otherwise, return specific channel
    if "C" in xarr.dims:
        selection["C"] = channel

    # Select the specific frame and compute to numpy
    # This returns a 2D numpy array directly
    return xarr.isel(**selection).compute().values


def load_nd2_metadata(nd2_path: str | Path) -> ND2Metadata:
    """
    Load essential ND2 file metadata for processing and visualization.

    Args:
        nd2_path: Path to ND2 file

    Returns:
        Dictionary containing essential metadata only
    """
    try:
        nd2_path = Path(nd2_path)

        with nd2.ND2File(str(nd2_path)) as f:
            # Get essential metadata only
            metadata: ND2Metadata = {
                "filepath": str(nd2_path),
                "filename": nd2_path.name,
                # Core dimensions
                "n_frames": f.sizes.get("T", 1),
                "height": f.sizes.get("Y", 0),
                "width": f.sizes.get("X", 0),
                "n_fov": f.sizes.get("P", 1),
                # Channel info - extract channel names from Channel objects
                "channels": [ch.channel.name for ch in (f.metadata.channels or [])],
                "n_channels": f.sizes.get("C", 1),
                # Physical units
                "pixel_microns": f.voxel_size().x if f.voxel_size() else 1.0,
                # Data type
                "native_dtype": str(f.dtype),
            }

            return metadata

    except Exception as e:
        raise RuntimeError(f"Failed to load ND2 metadata: {str(e)}")
