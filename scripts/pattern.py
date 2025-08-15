"""
Concise pattern display script for cell-filter.
"""

import os
from cell_filter.core.pattern import Patterner
import logging

# Define parameters here
PATTERNS = "path/to/patterns.nd2"
CELLS = "path/to/cells.nd2"
NUCLEI_CHANNEL = 1  # default in CLI
VIEW = 0  # Only used if VIEW_ALL is False
VIEW_ALL = True  # Set to False to use VIEW
OUTPUT = None  # Only used if VIEW_ALL is False
DEBUG = False

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("cell_filter").setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("cell_filter.scripts.pattern")

# Check if patterns file exists
if not os.path.exists(PATTERNS):
    logger.error(f"Patterns file not found: {PATTERNS}")
    exit(1)

# Check if cells file exists
if not os.path.exists(CELLS):
    logger.error(f"Cells file not found: {CELLS}")
    exit(1)

# Initialize info displayer
displayer = Patterner(
    patterns_path=PATTERNS,
    cells_path=CELLS,
    nuclei_channel=NUCLEI_CHANNEL,
)

if VIEW_ALL:
    logger.info("Displaying all views")
    for view_idx in range(displayer.n_views):
        logger.info(f"Displaying view {view_idx}")
        displayer.plot_view(view_idx)
else:
    logger.info(f"Plotting view {VIEW}")
    displayer.plot_view(VIEW, OUTPUT)
    logger.info("Plotting completed successfully")
displayer.close()
