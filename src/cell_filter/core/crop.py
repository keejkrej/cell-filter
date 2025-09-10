"""
Core cell cropper functionality for cell-filter.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging
from cell_filter.utils import load_nd2, get_nd2_frame, get_nd2_channel_stack

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CropperParameters:
    """Parameters for the Cropper class."""

    gaussian_blur_size: tuple[int, int] = (11, 11)
    bimodal_threshold: float = 0.1
    min_area_ratio: float = 0.5
    max_area_ratio: float = 1.5
    max_iterations: int = 10
    edge_tolerance: int = 5
    morph_dilate_size: tuple[int, int] = (5, 5)
    nuclei_channel: int = 1


class Cropper:
    """Crop and process cell data from images."""

    # Constructor

    def __init__(
        self, patterns_path: str, cells_path: str, parameters: CropperParameters
    ) -> None:
        """Initialize Cropper and set paths."""
        self.patterns_path = Path(patterns_path).resolve()
        self.cells_path = Path(cells_path).resolve()
        self.parameters = parameters

        try:
            self._init_patterns()
            self._init_cells()
            self._validate_files()
            logger.debug(
                f"Successfully initialized Cropper with patterns: {self.patterns_path} and cells: {self.cells_path}"
            )
        except Exception as e:
            self.close_files()
            logger.error(f"Error initializing ND2 readers: {e}")
            raise ValueError(f"Error initializing ND2 readers: {e}")

        self._init_memory()

    def _init_patterns(self) -> None:
        """Initialize the patterns ND2 file reader and extract metadata."""
        try:
            # Load xarray and metadata using new API
            self.patterns_xarr, metadata = load_nd2(self.patterns_path)
            self.pattern_n_channels = metadata.n_channels
            self.pattern_n_frames = metadata.n_frames
            self.pattern_n_fovs = metadata.n_fovs
            logger.debug(
                f"Channels: {self.pattern_n_channels}, Frames: {self.pattern_n_frames}, Views: {self.pattern_n_fovs}"
            )
        except Exception as e:
            logger.error(f"Error initializing patterns reader: {e}")
            raise

    def _init_cells(self) -> None:
        """Initialize the cells reader and metadata."""
        try:
            # Load xarray and metadata using new API
            self.cells_xarr, metadata = load_nd2(self.cells_path)
            self.dtype = self.cells_xarr.dtype
            self.cells_n_channels = metadata.n_channels
            self.cells_n_frames = metadata.n_frames
            self.cells_n_fovs = metadata.n_fovs
            # Store channel names from metadata
            self.cells_channel_names = metadata.channels

            # Validate channel count consistency
            if len(self.cells_channel_names) != self.cells_n_channels:
                raise ValueError(
                    f"Channel count mismatch in cells ND2 file: "
                    f"metadata reports {self.cells_n_channels} channels but "
                    f"found {len(self.cells_channel_names)} channel names"
                )

            logger.debug(
                f"Channels: {self.cells_n_channels}, Frames: {self.cells_n_frames}, Views: {self.cells_n_fovs}"
            )
            logger.debug(f"Channel names: {self.cells_channel_names}")
            logger.info(f"Data type: {self.dtype}")
        except Exception as e:
            logger.error(f"Error initializing cells reader: {e}")
            raise

    def _validate_files(self) -> None:
        """Validate the ND2 files meet the required specifications."""
        if self.pattern_n_channels != 1:
            raise ValueError("Patterns ND2 file should have exactly 1 channel")
        if self.pattern_n_frames != 1:
            raise ValueError("Patterns ND2 file must contain exactly 1 frame")
        if self.pattern_n_fovs != self.cells_n_fovs:
            raise ValueError(
                "Patterns and cells ND2 files must contain the same number of views"
            )
        if self.cells_n_channels < 2:
            raise ValueError("Cells ND2 file must contain at least 2 channels")

        self.n_fovs = self.cells_n_fovs
        self.n_frames = self.cells_n_frames
        logger.debug(f"Validated files: {self.n_fovs} views, {self.n_frames} frames")

    def _init_memory(self) -> None:
        """Initialize memory variables to default values."""
        self.current_fov = 0
        self.current_frame = 0
        self.patterns = None
        self.n_patterns = 0
        self.thresh = None
        self.contours = None
        self.bounding_boxes = None
        self.centers = None
        self.frame_nuclei = None
        logger.debug("Initialized memory variables")

    # Private Methods

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image to a range of 0-255."""
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
        image = image.astype(np.uint8)
        return image

    def _normalize_pct(self, image: np.ndarray, low: int, high: int) -> np.ndarray:
        """
        Normalize an image to a range of 0-255.

        Args:
            image (np.ndarray): Input image to normalize
            low (int): Lower percentile
            high (int): Higher percentile

        Returns:
            np.ndarray: Normalized image

        Raises:
            ValueError: If image is None or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")

        percentile_high = np.percentile(image[image > 0], high)
        percentile_low = np.percentile(image[image > 0], low)
        image = np.clip(image, percentile_low, percentile_high)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
        image = image.astype(np.uint8)

        return image

    def _find_contours(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Find contours in an image using thresholding and contour detection."""
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(image, self.parameters.gaussian_blur_size, 0)

        # Apply thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations
        kernel = np.ones(self.parameters.morph_dilate_size, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        logger.debug(f"Found {len(contours)} contours in image")

        return list(contours), thresh

    def _refine_contours(
        self, contours: list[np.ndarray], image_shape: tuple[int, int]
    ) -> list[tuple[int, int, np.ndarray, tuple[int, int, int, int]]]:
        """
        Refine contours by filtering based on area and calculating centers.

        Args:
            contours (list[np.ndarray]): list of contours to refine
            image_shape (tuple[int, int]): Shape of the image (height, width)

        Returns:
            list[tuple[int, int, np.ndarray, tuple[int, int, int, int]]]: list of (center_y, center_x, contour, bounding_box)

        Raises:
            ValueError: If contours list is empty
        """
        if not contours:
            raise ValueError("No contours provided")

        # Calculate areas
        areas = np.array([cv2.contourArea(contour) for contour in contours])

        # Iteratively remove small and large areas until CV falls below threshold
        current_contours = list(contours)
        current_areas = areas.copy()
        iteration = 0
        cv = float("inf")

        for iteration in range(self.parameters.max_iterations):
            cv = np.std(current_areas) / np.mean(current_areas)
            if cv < self.parameters.bimodal_threshold:
                break

            # Remove areas smaller than min_area_ratio of mean
            mean_area = np.mean(current_areas)
            min_area = self.parameters.min_area_ratio * mean_area
            max_area = self.parameters.max_area_ratio * mean_area

            # Create new lists of contours and areas
            new_contours = []
            new_areas = []
            for i, (contour, area) in enumerate(zip(current_contours, current_areas)):
                if area >= min_area and area <= max_area:
                    new_contours.append(contour)
                    new_areas.append(area)

            # Update for next iteration
            current_contours = new_contours
            current_areas = np.array(new_areas)

            if len(current_contours) == 0:
                logger.warning("All contours were removed during iterative filtering")
                break

        logger.debug(f"After {iteration + 1} iterations, CV reduced to {cv:.3f}")

        contour_data = []
        for contour in current_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Skip if too close to edges
            if (
                x < self.parameters.edge_tolerance
                or y < self.parameters.edge_tolerance
                or x + w > image_shape[1] - self.parameters.edge_tolerance
                or y + h > image_shape[0] - self.parameters.edge_tolerance
            ):
                continue

            center_x = x + w // 2
            center_y = y + h // 2
            contour_data.append((center_y, center_x, contour, (x, y, w, h)))

        # Sort contours by center
        contour_data.sort(key=lambda x: (x[0], x[1]))

        logger.debug(
            f"Filtered {len(contours)} contours to {len(contour_data)} using iterative area analysis"
        )
        return contour_data

    def _extract_region(
        self, frame: np.ndarray, pattern_idx: int, normalize: bool
    ) -> np.ndarray:
        """Extract a region from a frame based on pattern index."""
        if frame is None:
            raise ValueError("Frame not provided")
        if pattern_idx >= self.n_patterns or pattern_idx < 0:
            raise ValueError(
                f"Pattern index {pattern_idx} out of range (0-{self.n_patterns - 1})"
            )
        if self.bounding_boxes is None:
            raise ValueError("No bounding boxes provided")

        try:
            x, y, w, h = self.bounding_boxes[pattern_idx]
            region = frame[y : y + h, x : x + w]
            if normalize:
                region = self._normalize(region)
            return region
        except Exception as e:
            logger.error(f"Error extracting region: {e}")
            raise ValueError(f"Error extracting region: {e}")

    # Public Methods

    def close_files(self) -> None:
        """Safely close all ND2 readers."""
        # When using nd2.imread with xarray/dask, we don't need to explicitly close files
        # The xarray DataArray objects don't have a close method
        # Context managers handle file closure automatically
        logger.debug(
            "ND2 files are managed by xarray/dask - no explicit closure needed"
        )

    def load_fov(self, fov_idx: int) -> None:
        """Load a specific view from the ND2 files."""
        if fov_idx >= self.n_fovs or fov_idx < 0:
            raise ValueError(f"View index {fov_idx} out of range (0-{self.n_fovs - 1})")
        self.current_fov = fov_idx
        logger.debug(f"Loaded view {fov_idx}")

    def load_patterns(self) -> None:
        """Load the patterns from ND2 file."""
        try:
            # Extract the pattern frame using utility function
            # For patterns file, we select frame 0, channel 0, and the current view (P)
            self.patterns = get_nd2_frame(self.patterns_xarr, self.current_fov, 0, 0)
            logger.debug(f"Loaded patterns for view {self.current_fov}")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            raise ValueError(f"Error loading patterns: {e}")

    def load_nuclei(self, frame_idx: int) -> None:
        """Load nuclei frame from ND2 file."""
        if frame_idx >= self.n_frames:
            raise ValueError(
                f"Frame index {frame_idx} out of range (0-{self.n_frames - 1})"
            )
        try:
            # Extract the nuclei frame using utility function
            # Select the specific frame, nuclei channel, and current view
            self.frame_nuclei = get_nd2_frame(
                self.cells_xarr,
                self.current_fov,
                self.parameters.nuclei_channel,
                frame_idx,
            )
            logger.debug(
                f"Loaded nuclei frame {frame_idx} for view {self.current_fov} from channel {self.parameters.nuclei_channel}"
            )
        except Exception as e:
            logger.error(f"Error loading nuclei: {e}")
            raise ValueError(f"Error loading nuclei: {e}")

    def load_cell(self, frame_idx: int) -> None:
        """Load frames from all channels in the cells ND2 file as a 3D array."""
        if frame_idx >= self.n_frames:
            raise ValueError(
                f"Frame index {frame_idx} out of range (0-{self.n_frames - 1})"
            )
        try:
            # Extract frames from all channels at once
            # Select the specific frame and current view
            self.frame_cell = get_nd2_channel_stack(
                self.cells_xarr, self.current_fov, frame_idx
            )
            logger.debug(
                f"Loaded cell frames {frame_idx} for view {self.current_fov} from all channels at once"
            )
        except Exception as e:
            logger.error(f"Error loading cell: {e}")
            raise ValueError(f"Error loading cell: {e}")

    def process_patterns(self) -> None:
        """Process pattern image to extract contours and their bounding boxes."""
        if self.patterns is None:
            raise ValueError("Patterns must be loaded before processing")

        self.patterns_norm = self._normalize_pct(self.patterns, 10, 90)
        contours, self.thresh = self._find_contours(self.patterns_norm)
        contour_data = self._refine_contours(contours, self.patterns_norm.shape)
        self.contours = [x[2] for x in contour_data]
        self.bounding_boxes = [x[3] for x in contour_data]
        self.centers = [x[0:2] for x in contour_data]
        self.n_patterns = len(self.contours)
        logger.debug(f"Processed {self.n_patterns} patterns")

    def extract_nuclei(self, pattern_idx: int, normalize: bool = False) -> np.ndarray:
        """Extract nuclei region for a specific pattern."""
        if self.frame_nuclei is None:
            raise ValueError("Nuclei frame must be loaded before extraction")
        return self._extract_region(self.frame_nuclei, pattern_idx, normalize)

    def extract_cell(
        self, pattern_idx: int, normalize: bool = False
    ) -> list[np.ndarray]:
        """Extract regions from all loaded cell channels at once."""
        if self.frame_cell is None:
            raise ValueError("Cell frames must be loaded before extraction")

        # Handle both 3D array (new format) and list of 2D arrays (old format)
        cell_regions = []
        if isinstance(self.frame_cell, np.ndarray) and self.frame_cell.ndim == 3:
            # New format: 3D array with shape (C, H, W)
            for c in range(self.frame_cell.shape[0]):
                region = self._extract_region(
                    self.frame_cell[c], pattern_idx, normalize
                )
                cell_regions.append(region)
        else:
            # Old format: list of 2D arrays
            for channel_frame in self.frame_cell:
                region = self._extract_region(channel_frame, pattern_idx, normalize)
                cell_regions.append(region)

        return cell_regions

    def extract_pattern(self, pattern_idx: int, normalize: bool = False) -> np.ndarray:
        """Extract pattern region."""
        if self.patterns is None:
            raise ValueError("Patterns must be loaded before extraction")
        return self._extract_region(self.patterns, pattern_idx, normalize)
