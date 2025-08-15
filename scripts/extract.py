"""
Concise extraction script for cell-filter.
"""

import os
from cell_filter.core.extract import Extractor
import logging
from pathlib import Path

# Define parameters here
PATTERNS = "path/to/patterns.nd2"
CELLS = "path/to/cells.nd2"
NUCLEI_CHANNEL = 0
TIME_SERIES = "path/to/time_series_json_dir"
OUTPUT = "path/to/output_dir"
MIN_FRAMES = 20
DEBUG = False

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("cell_filter").setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("cell_filter.cli.extract")

# Check if patterns file exists
if not os.path.exists(PATTERNS):
    logger.error(f"Patterns file not found: {PATTERNS}")
    exit(1)

# Check if cells file exists
if not os.path.exists(CELLS):
    logger.error(f"Cells file not found: {CELLS}")
    exit(1)

# Check if time series directory exists
if not os.path.exists(TIME_SERIES):
    logger.error(f"Time series directory not found: {TIME_SERIES}")
    exit(1)

# Create output directory if it doesn't exist
output_dir = Path(OUTPUT)
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize extractor
extractor = Extractor(
    patterns_path=PATTERNS,
    cells_path=CELLS,
    output_folder=OUTPUT,
    nuclei_channel=NUCLEI_CHANNEL,
)

# Extract frames
logger.info("Starting extraction process")
extractor.extract(time_series_dir=TIME_SERIES, min_frames=MIN_FRAMES)
logger.info("Extraction completed successfully")
