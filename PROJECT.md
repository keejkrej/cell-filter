# Cell Filter — Project Summary

## Overview

- **Goal**: Filter micropatterned timelapse microscopy image datasets by the number of cells per pattern.
- **Approach**: Segment nuclei/cytoplasm, count cells per micropattern, and extract time-series data for patterns that match the desired cell count.
- **Typical inputs**: ND2 or TIFF stacks for patterns and cells, with configurable channel mapping.
- **Typical outputs**: JSON summaries and extracted image sequences for qualifying patterns.

## Core Components

The core functionality is organized into several modules:

- **`cell_filter.core.crop`**: Core functionality for loading and processing microscopy image data. Handles both pattern and cell data loading, segmentation, and pattern processing.
  - **`Cropper`**: Main class for loading ND2 files and extracting regions of interest
  - **`CropperParameters`**: Configuration parameters for image processing

- **`cell_filter.core.pattern`**: Display patterns with bounding boxes and indices for inspection.
  - **`Patterner`**: Class for visualizing micropatterns with their bounding boxes and indices

- **`cell_filter.core.count`**: Cell counting functionality using Cellpose.
  - **`CellposeCounter`**: Class that uses Cellpose model to detect and count nuclei in images

- **`cell_filter.core.filter`**: Run cell detection/segmentation and count cells per micropattern view.
  - **`Filterer`**: Main filtering class that processes frame data and tracks nuclei counts
  - **`Patterns`**: Helper class for tracking pattern states during filtering

- **`cell_filter.core.extract`**: Extract and save frames for patterns that meet target criteria.
  - **`Extractor`**: Main extraction class that refines filter results and saves frame stacks

Helper scripts in `scripts/` wire these components for quick use:

- **`scripts/pattern.py`**: Display all or a selected view for quick inspection.
- **`scripts/filter.py`**: Filter all or a range of views and write results (JSON) to an output folder.
- **`scripts/extract.py`**: Extract frames for qualifying patterns based on prior filtering results.

CLI entrypoints in `cell_filter.cli` provide command-line access:

- **`cell_filter.cli.pattern`**: Pattern visualization and inspection
- **`cell_filter.cli.filtering`**: Cell counting and pattern filtering
- **`cell_filter.cli.extract`**: Time-series extraction for qualifying patterns

## Workflow

### 1. Data Preparation
Organize your microscopy data in a clear structure:

```
project_directory/
├── data/
│   ├── patterns_reference.nd2      # Static pattern template
│   ├── cells_timelapse.nd2         # Time-lapse microscopy data
│   └── analysis/                   # Output directory (created automatically)
└── scripts/                        # Optional: custom analysis scripts
```

### 2. Pattern Inspection
**Purpose**: Verify pattern detection, channel mapping, and view indexing.

```bash
# Inspect all views to verify pattern detection
python -m cell_filter.cli.pattern --patterns data/patterns_reference.nd2 --cells data/cells_timelapse.nd2 --view-all --nuclei-channel 1

# Inspect specific view for detailed analysis
python -m cell_filter.cli.pattern --patterns data/patterns_reference.nd2 --cells data/cells_timelapse.nd2 --view 0 --output pattern_view_0.png
```

**What to check**:
- Pattern bounding boxes are correctly positioned
- Nuclei channel contains clear cell nuclei
- Pattern indices match expected layout

### 3. Cell Counting and Filtering
**Purpose**: Count cells per micropattern across time-lapse frames and identify patterns meeting criteria.

```bash
# Process all views for patterns with exactly 4 cells
python -m cell_filter.cli.filtering --patterns data/patterns_reference.nd2 --cells data/cells_timelapse.nd2 --n-cells 4 --output data/analysis/ --all --debug

# Process subset of views for testing
python -m cell_filter.cli.filtering --patterns data/patterns_reference.nd2 --cells data/cells_timelapse.nd2 --n-cells 4 --output data/analysis/ --range "0:3"
```

**What happens**:
- Cellpose segments nuclei in each frame
- Cell counts are recorded per pattern per frame
- Patterns consistently meeting cell count criteria are identified
- Results saved as JSON files for each view

### 4. Time-Series Extraction
**Purpose**: Extract and save time-series image sequences for qualifying patterns.

```bash
# Extract sequences with minimum 20 consecutive frames
python -m cell_filter.cli.extract --patterns data/patterns_reference.nd2 --cells data/cells_timelapse.nd2 --time-series data/analysis/ --output data/analysis/ --min-frames 20 --max-gap 6
```

**What happens**:
- Identifies continuous sequences meeting frame count requirements
- Extracts individual frames for each qualifying sequence
- Saves metadata including frame indices, cell counts, and pattern coordinates

## Example Result Structure

### Typical Output Directory Layout

After running the complete workflow, your output directory will contain:

```
data/analysis/
├── view_000/                           # Results for view 0
│   ├── frame_counts.json              # Cell counts per frame per pattern
│   ├── pattern_summary.json           # Statistical summary per pattern
│   ├── qualifying_patterns.json       # Patterns meeting n_cells criteria
│   └── debug_images/                  # (if debug enabled)
│       ├── frame_005_pattern_003.png  # Segmentation overlays
│       └── frame_005_pattern_007.png
├── view_001/                           # Results for view 1
│   ├── frame_counts.json
│   ├── pattern_summary.json
│   └── qualifying_patterns.json
├── view_002/
│   └── ...
├── extracted_sequences/                # Time-series extracts
│   ├── view_000_pattern_003_seq_001/   # Sequence 1 from view 0, pattern 3
│   │   ├── metadata.json              # Sequence metadata
│   │   ├── frame_012.tiff             # Individual frames
│   │   ├── frame_013.tiff
│   │   ├── frame_014.tiff
│   │   └── ... (20+ frames)
│   ├── view_000_pattern_003_seq_002/   # Sequence 2 (after gap)
│   │   ├── metadata.json
│   │   ├── frame_045.tiff
│   │   └── ...
│   ├── view_000_pattern_007_seq_001/   # Different pattern
│   │   ├── metadata.json
│   │   ├── frame_008.tiff
│   │   └── ...
│   └── view_001_pattern_012_seq_001/   # Different view
│       ├── metadata.json
│       └── ...
├── summary.json                        # Overall analysis summary
└── processing_log.txt                  # Detailed processing log
```

### Example File Contents

#### `frame_counts.json` (Cell counts per frame)
```json
{
    "view_index": 0,
    "n_patterns": 96,
    "n_frames": 150,
    "frame_counts": {
        "pattern_003": {
            "0": 4, "1": 4, "2": 3, "3": 4, "4": 4,
            "5": 4, "6": 4, "7": 4, "8": 4, "9": 4,
            "...": "..."
        },
        "pattern_007": {
            "0": 2, "1": 2, "2": 2, "3": 3, "4": 3,
            "...": "..."
        }
    },
    "processing_info": {
        "nuclei_channel": 1,
        "target_cell_count": 4,
        "segmentation_model": "cyto2",
        "processing_time": "2024-01-15T10:30:45Z"
    }
}
```

#### `pattern_summary.json` (Statistical summary)
```json
{
    "view_index": 0,
    "target_cell_count": 4,
    "pattern_statistics": {
        "pattern_003": {
            "mean_cell_count": 3.89,
            "std_cell_count": 0.31,
            "frames_meeting_criteria": 142,
            "total_frames": 150,
            "criteria_percentage": 94.67,
            "longest_qualifying_sequence": 89,
            "sequence_gaps": [8, 12]
        },
        "pattern_007": {
            "mean_cell_count": 2.34,
            "std_cell_count": 0.48,
            "frames_meeting_criteria": 12,
            "total_frames": 150,
            "criteria_percentage": 8.0,
            "longest_qualifying_sequence": 0,
            "sequence_gaps": []
        }
    }
}
```

#### `qualifying_patterns.json` (Patterns meeting criteria)
```json
{
    "view_index": 0,
    "target_cell_count": 4,
    "min_frames_threshold": 20,
    "qualifying_patterns": [
        {
            "pattern_id": "pattern_003",
            "total_qualifying_frames": 142,
            "percentage_qualifying": 94.67,
            "continuous_sequences": [
                {
                    "start_frame": 0,
                    "end_frame": 88,
                    "length": 89,
                    "sequence_id": "seq_001"
                },
                {
                    "start_frame": 97,
                    "end_frame": 149,
                    "length": 53,
                    "sequence_id": "seq_002"
                }
            ]
        },
        {
            "pattern_id": "pattern_015",
            "total_qualifying_frames": 89,
            "percentage_qualifying": 59.33,
            "continuous_sequences": [
                {
                    "start_frame": 23,
                    "end_frame": 111,
                    "length": 89,
                    "sequence_id": "seq_001"
                }
            ]
        }
    ],
    "summary": {
        "total_patterns_analyzed": 96,
        "patterns_meeting_criteria": 12,
        "qualification_rate": 12.5
    }
}
```

#### `extracted_sequences/view_000_pattern_003_seq_001/metadata.json`
```json
{
    "sequence_info": {
        "view_index": 0,
        "pattern_id": "pattern_003",
        "sequence_id": "seq_001",
        "start_frame": 0,
        "end_frame": 88,
        "total_frames": 89
    },
    "pattern_coordinates": {
        "x_min": 234,
        "y_min": 567,
        "x_max": 334,
        "y_max": 667,
        "width": 100,
        "height": 100
    },
    "cell_counts": {
        "0": 4, "1": 4, "2": 4, "3": 4, "4": 4,
        "...": "..."
    },
    "extraction_parameters": {
        "nuclei_channel": 1,
        "target_cell_count": 4,
        "min_frames": 20,
        "max_gap": 6
    },
    "file_info": {
        "original_patterns_file": "data/patterns_reference.nd2",
        "original_cells_file": "data/cells_timelapse.nd2",
        "extraction_date": "2024-01-15T11:45:30Z",
        "frame_format": "TIFF",
        "bit_depth": 16
    }
}
```

#### `summary.json` (Overall analysis summary)
```json
{
    "analysis_summary": {
        "input_files": {
            "patterns": "data/patterns_reference.nd2",
            "cells": "data/cells_timelapse.nd2"
        },
        "processing_parameters": {
            "nuclei_channel": 1,
            "target_cell_count": 4,
            "min_frames": 20,
            "max_gap": 6
        },
        "results_overview": {
            "total_views_processed": 8,
            "total_patterns_analyzed": 768,
            "total_frames_analyzed": 1200,
            "patterns_meeting_criteria": 89,
            "qualification_rate": 11.6,
            "total_sequences_extracted": 156,
            "average_sequence_length": 67.3
        },
        "processing_info": {
            "start_time": "2024-01-15T10:15:22Z",
            "end_time": "2024-01-15T11:52:18Z",
            "total_processing_time": "1h 36m 56s",
            "gpu_used": true,
            "cellpose_model": "cyto2"
        }
    }
}
```

## File Formats and Data Types

### Input Files
- **Pattern files**: ND2 or TIFF format, typically single timepoint
- **Cell files**: ND2 or TIFF format, time-lapse series
- **Supported channels**: Multiple channels with configurable nuclei channel index
- **Bit depths**: 8-bit, 12-bit, 16-bit images supported

### Output Files
- **JSON files**: Human-readable analysis results and metadata
- **TIFF files**: Extracted image sequences in 16-bit format
- **Logs**: Plain text processing logs with timestamps

### Typical File Sizes
- **Small dataset**: ~100MB input → ~50MB output
- **Medium dataset**: ~2GB input → ~500MB output
- **Large dataset**: ~10GB input → ~2-5GB output

*Note: Output size depends on number of qualifying patterns and sequence lengths*

## Inputs and Configuration

### Image Requirements
- **Format**: ND2 (preferred) or TIFF stacks
- **Dimensions**: Minimum 512×512 pixels, maximum limited by available RAM
- **Channels**: Multi-channel supported, specify nuclei channel index
- **Time series**: Variable length supported (tested up to 500+ frames)

### Configuration Parameters
- **Nuclei channel**: Channel index containing nuclei signal (0-based indexing)
- **Target cell count**: Desired number of cells per pattern (typically 1-10)
- **View ranges**: Process all views or specify ranges (e.g., "0:10")
- **Sequence parameters**: Minimum frames (default 20), maximum gap (default 6)
- **GPU settings**: Automatic GPU detection with CPU fallback

### Performance Considerations
- **GPU vs CPU**: GPU provides ~10x speed improvement for segmentation
- **Memory usage**: ~2-4GB RAM per gigapixel of image data
- **Processing time**: ~1-5 minutes per view depending on image size and complexity

## Environment and Dependencies

### System Requirements
- **Python**: ≥3.11 (specified in pyproject.toml)
- **Operating System**: Windows, macOS, Linux
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large datasets
- **GPU**: CUDA-compatible GPU recommended (4GB+ VRAM)

### Key Dependencies
- **Core processing**: NumPy, SciPy, scikit-image
- **Deep learning**: PyTorch, Cellpose (>4.0)
- **Image I/O**: nd2, tifffile, opencv-python
- **Data handling**: xarray, dask
- **Visualization**: matplotlib
- **Utilities**: networkx for graph operations

### Installation Methods
```bash
# Development installation (recommended)
git clone https://github.com/keejkrej/cell-filter.git
cd cell-filter
pip install -e .

# Production installation (when available)
pip install cell-filter
```

## Repository Layout

```
cell-filter/
├── src/cell_filter/              # Main package source
│   ├── __init__.py
│   ├── cli/                      # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── pattern.py           # Pattern visualization CLI
│   │   ├── filtering.py         # Filtering and analysis CLI
│   │   └── extract.py           # Extraction CLI
│   ├── core/                    # Core functionality modules
│   │   ├── __init__.py
│   │   ├── crop.py              # Image loading and cropping
│   │   ├── pattern.py           # Pattern visualization
│   │   ├── count.py             # Cell counting with Cellpose
│   │   ├── filter.py            # Pattern filtering logic
│   │   ├── extract.py           # Time-series extraction
│   │   └── segmentation.py      # Segmentation utilities
│   ├── extract/                 # Extraction-specific modules
│   ├── filter/                  # Filtering-specific modules
│   ├── pattern/                 # Pattern-specific modules
│   └── utils/                   # Utility functions
│       └── gpu_utils.py         # GPU validation and setup
├── scripts/                     # Example analysis scripts
│   ├── pattern.py               # Pattern inspection script
│   ├── filter.py                # Filtering analysis script
│   └── extract.py               # Extraction script
├── README.md                    # Installation and usage guide
├── PROJECT.md                   # This file - project overview
├── pyproject.toml               # Package configuration and dependencies
├── uv.lock                      # Dependency lock file
└── .gitignore                   # Git ignore rules
```

## Performance Optimization

### GPU Utilization
- Cellpose automatically uses available GPU for segmentation
- GPU memory management handled automatically
- Fallback to CPU processing if GPU unavailable

### Memory Management
- Lazy loading of image data using dask arrays
- Chunked processing for large time-series
- Automatic memory cleanup between views

### Processing Strategies
- **Small datasets** (<1GB): Process all views at once (`--all`)
- **Medium datasets** (1-5GB): Process in view ranges (`--range "0:10"`)
- **Large datasets** (>5GB): Process individual views and combine results

## Troubleshooting Common Issues

### GPU Detection Issues
```bash
# Verify CUDA installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Verify Cellpose GPU support
python -c "from cellpose import models; print(models.use_gpu())"
```

### Memory Issues
- Reduce processing range: `--range "0:2"`
- Close other applications to free memory
- Process smaller image crops or reduce resolution

### File Format Issues
- Verify ND2 files: `python -c "import nd2; print(nd2.imread('file.nd2').shape)"`
- Check TIFF compatibility: `python -c "import tifffile; print(tifffile.imread('file.tiff').shape)"`

### Channel Configuration
- Use pattern inspection to verify nuclei channel visibility
- Common nuclei channels: DAPI (0 or 1), Hoechst (varies)
- Check channel order in acquisition software

## Future Development

### Planned Features
- Support for additional image formats (CZI, LSM)
- Interactive pattern selection GUI
- Batch processing automation
- Statistical analysis and visualization tools
- Integration with common microscopy analysis pipelines

### Extension Points
- Custom segmentation models
- Alternative cell counting methods
- Additional filtering criteria
- Export to common analysis formats (ImageJ, CellProfiler)

## License and Citation

**License**: MIT (see pyproject.toml for full details)

**Citation**: If you use Cell Filter in your research, please cite:
```
Cao, T. (2025). Cell Filter: Filtering micropatterned timelapse microscopy images
based on number of cells. GitHub: https://github.com/keejkrej/cell-filter
```

## Support and Contact

- **Issues**: Submit bug reports and feature requests via GitHub Issues
- **Documentation**: See README.md for usage examples
- **Contact**: Tianyi Cao (ctyjackcao@outlook.com)
