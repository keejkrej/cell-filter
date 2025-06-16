"""
Concise analysis script for cell-filter.
"""

import os
from cell_filter.core.analyze import Analyzer
import logging

# Define parameters here
PATTERNS = "path/to/patterns.tif"
CELLS = "path/to/cells.tif"
NUCLEI_CHANNEL = 0
CYTO_CHANNEL = 1
OUTPUT = "path/to/output.json"
WANTED = 3
DIAMETER = 15
NO_GPU = False  # Set to True to disable GPU
ALL = True      # Set to False to use RANGE
RANGE = "0:10"  # Only used if ALL is False
DEBUG = False

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logging.getLogger("cell_filter").setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("cell_filter.cli.analyze")

# Check if patterns file exists
if not os.path.exists(PATTERNS):
    logger.error(f"Error: Patterns file not found: {PATTERNS}")
    exit(1)

# Check if cells file exists
if not os.path.exists(CELLS):
    logger.error(f"Error: Cells file not found: {CELLS}")
    exit(1)

# Initialize analyzer
analyzer = Analyzer(
    patterns_path=PATTERNS,
    cells_path=CELLS,
    output_folder=OUTPUT,
    wanted=WANTED,
    use_gpu=not NO_GPU,
    nuclei_channel=NUCLEI_CHANNEL,
    cyto_channel=CYTO_CHANNEL,
)

# Process views
if ALL:
    logger.info("Processing all views")
    analyzer.process_views(0, analyzer.generator.n_views)
else:
    logger.info(f"Processing views {RANGE}")
    view_range = list(map(int, RANGE.split(":")))
    analyzer.process_views(view_range[0], view_range[1]) 