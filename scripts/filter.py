"""
Concise filtering script for cell-filter.
"""

import os
from cell_filter.filter import Filterer
import logging

# Define parameters here
PATTERNS = "data/20250806_patterns_after.nd2"
CELLS = "data/20250806_MDCK_timelapse_crop_fov0004.nd2"
NUCLEI_CHANNEL = 1
OUTPUT = "data/analysis/"
N_CELLS = 4
ALL = False  # Set to False to use RANGE
RANGE = "0:1"  # Only used if ALL is False, end is exclusive
DEBUG = False

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("cell_filter").setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("cell_filter.scripts.filter")

# Check if patterns file exists
if not os.path.exists(PATTERNS):
    logger.error(f"Error: Patterns file not found: {PATTERNS}")
    exit(1)

# Check if cells file exists
if not os.path.exists(CELLS):
    logger.error(f"Error: Cells file not found: {CELLS}")
    exit(1)

# Initialize filterer
filter_processor = Filterer(
    patterns_path=PATTERNS,
    cells_path=CELLS,
    output_folder=OUTPUT,
    n_cells=N_CELLS,
    nuclei_channel=NUCLEI_CHANNEL,
)

# Process fovs
if ALL:
    logger.info("Processing all fovs")
    filter_processor.process_fovs(0, filter_processor.cropper.n_fovs)
else:
    logger.info(f"Processing fovs {RANGE}")
    fov_range = list(map(int, RANGE.split(":")))
    filter_processor.process_fovs(fov_range[0], fov_range[1])
