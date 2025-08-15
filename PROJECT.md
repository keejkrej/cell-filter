# Cell Filter — Project Summary

## Overview

- **Goal**: Filter micropatterned timelapse microscopy image datasets by the number of cells per pattern.
- **Approach**: Segment nuclei/cytoplasm, count cells per micropattern, and extract time-series data for patterns that match the desired cell count.
- **Typical inputs**: ND2 or TIFF stacks for patterns and cells, with configurable channel mapping.
- **Typical outputs**: JSON summaries and extracted image sequences for qualifying patterns.

## Core Components

The core functionality is organized into several modules:

- **`cell_filter.core.generate`**: Core functionality for loading and processing microscopy image data. Handles both pattern and cell data loading, segmentation, and pattern processing.
  - **`Generator`**: Main class for loading ND2 files and extracting regions of interest
  - **`GeneratorParameters`**: Configuration parameters for image processing

- **`cell_filter.core.pattern`**: Display patterns with bounding boxes and indices for inspection.
  - **`Patterner`**: Class for visualizing micropatterns with their bounding boxes and indices

- **`cell_filter.core.count`**: Cell counting functionality using Cellpose.
  - **`CellposeCounter`**: Class that uses Cellpose model to detect and count nuclei in images

- **`cell_filter.core.analyze`**: Run cell detection/segmentation and count cells per micropattern view.
  - **`Analyzer`**: Main analysis class that processes time series data and tracks nuclei counts
  - **`Patterns`**: Helper class for tracking pattern states during analysis

- **`cell_filter.core.extract`**: Extract and save time-series frames for patterns that meet target criteria.
  - **`Extractor`**: Main extraction class that refines time series data and saves frame stacks

Helper scripts in `scripts/` wire these components for quick use:

- **`scripts/pattern.py`**: Display all or a selected view for quick inspection.
- **`scripts/analyze.py`**: Analyze all or a range of views and write results (JSON) to an output folder.
- **`scripts/extract.py`**: Extract frames for qualifying patterns based on prior analysis/time-series metadata.

## Workflow

1. **Inspect patterns and cells** with `scripts/pattern.py` to confirm the correct nuclei/cytoplasm channels and view indexing.
2. **Analyze** with `scripts/analyze.py` to compute cell counts per micropattern and save results.
3. **Extract** with `scripts/extract.py` to export frames/time-series for patterns matching your desired cell count.

## Inputs and Configuration

- **Image formats**: `.nd2`, `.tif/.tiff`.
- **Channels**: Configure nuclei channel index in the scripts.
- **Ranges**: Process all views or a specified range (e.g., `"0:10"`).
- **GPU**: Enable/disable GPU in analysis (falls back to CPU if disabled/unavailable).

## Environment and Dependencies

- **Python**: 3.12 (per `pyproject.toml`).
- **Key libraries**: NumPy, PyTorch, tifffile, matplotlib, OpenCV, Cellpose, nd2.

## Installation

See `README.md` for full instructions. Typical editable install:

```bash
git clone https://github.com/keejkrej/cell-filter.git
cd cell-filter
pip install -e .
```

## Quick Start

Adjust paths and parameters in the scripts under `scripts/`, then run:

```bash
python scripts/pattern.py
python scripts/analyze.py
python scripts/extract.py
```

## Repository Layout

- **`src/cell_filter/`**: Library source code
  - `core/`: Core modules for image processing and analysis
    - `generate.py`: Core functionality for loading and processing microscopy image data
    - `pattern.py`: Display patterns with bounding boxes and indices for inspection
    - `analyze.py`: Run cell detection/segmentation using Cellpose and count cells per micropattern
    - `extract.py`: Extract and save time-series frames for patterns that meet target criteria
- **`scripts/`**: Ready-to-edit examples orchestrating the core components
  - `pattern.py`: Display all or a selected view for quick inspection
  - `analyze.py`: Analyze all or a range of views and write results (JSON) to an output folder
  - `extract.py`: Extract frames for qualifying patterns based on prior analysis metadata
- **`README.md`**: Installation and basic usage
- **`pyproject.toml`**: Package metadata and dependencies

## License

MIT (see `pyproject.toml`).
