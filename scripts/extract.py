"""
Concise extraction script for cell-filter.
"""

import os
from cell_filter.extract import Extractor
from cell_filter.utils.gpu_utils import validate_segmentation_requirements
import logging
from pathlib import Path

# Define parameters here
PATTERNS = "data/20250806_patterns_after.nd2"
CELLS = "data/20250806_MDCK_timelapse_crop_fov0004.nd2"
NUCLEI_CHANNEL = 1
FILTER_RESULTS = "data/analysis/"
OUTPUT = "data/analysis/"
MIN_FRAMES = 10
MAX_GAP = 6  # Maximum frame gap before splitting pattern sequences
DEBUG = False

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("cell_filter").setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("cell_filter.scripts.extract")

# Check if patterns file exists
if not os.path.exists(PATTERNS):
    logger.error(f"Patterns file not found: {PATTERNS}")
    exit(1)

# Check if cells file exists
if not os.path.exists(CELLS):
    logger.error(f"Cells file not found: {CELLS}")
    exit(1)

# Check if filter results directory exists
if not os.path.exists(FILTER_RESULTS):
    logger.error(f"Filter results directory not found: {FILTER_RESULTS}")
    exit(1)

# Create output directory if it doesn't exist
output_dir = Path(OUTPUT)
output_dir.mkdir(parents=True, exist_ok=True)

# GPU validation (always required for segmentation)
try:
    validate_segmentation_requirements(enable_segmentation=True)
    logger.info("GPU validation successful for segmentation")
except Exception as e:
    logger.error(f"GPU validation failed: {e}")
    exit(1)

# Initialize extractor
extractor = Extractor(
    patterns_path=PATTERNS,
    cells_path=CELLS,
    output_folder=OUTPUT,
    nuclei_channel=NUCLEI_CHANNEL,
)

# Extract frames
logger.info("Starting extraction process with comprehensive segmentation")
extractor.extract(filter_results_dir=FILTER_RESULTS, min_frames=MIN_FRAMES, max_gap=MAX_GAP)
logger.info("Extraction completed successfully")
