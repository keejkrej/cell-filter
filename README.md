# Cell Counter

A Python package for counting cells in microscopy images.

## Installation

```bash
git clone https://github.com/yourusername/cell-counter.git
cd cell-counter
pip install -e .
```

## Usage

### Command Line Interface

#### Info Command

Display information about the image stacks:

```bash
# Show info for patterns and nuclei
python -m cell_counter.cli.info --patterns <patterns_file> --nuclei <nuclei_file>

# Show info for patterns and cytoplasm
python -m cell_counter.cli.info --patterns <patterns_file> --cyto <cyto_file>

# Show info for all
python -m cell_counter.cli.info --patterns <patterns_file> --nuclei <nuclei_file> --cyto <cyto_file>

# Specify custom grid size for pattern centers
python -m cell_counter.cli.info --patterns <patterns_file> --nuclei <nuclei_file> --grid-size 30
```

The info command displays:
- Patterns image dimensions
- Grid size used for pattern center alignment (default: 20x20)
- Number of frames in nuclei/cytoplasm stacks
- Dimensions of each frame
- Total number of contours found
- Number of contours after filtering (contours touching edges are filtered out)
- Number of contours filtered out (if any)

Optional arguments:
- `--grid-size`: Size of the grid for snapping pattern centers (default: 20). A larger value creates a coarser grid, while a smaller value creates a finer grid.

#### Analyze Command

Analyze time series data and track nuclei counts:

```bash
# Basic usage
python -m cell_counter.cli.analyze --patterns <patterns_file> --nuclei <nuclei_file> --output <output_file>

# With custom parameters
python -m cell_counter.cli.analyze --patterns <patterns_file> --nuclei <nuclei_file> --output <output_file> --wanted 3 --no-cellpose --diameter 20

Optional arguments:
--wanted: Number of nuclei to look for (default: 3)
--no-cellpose: Use simple thresholding instead of Cellpose
--no-gpu: Don't use GPU for Cellpose
--diameter: Expected diameter of cells in pixels (default: 15)
--channels: Channel indices for Cellpose (default: "0,0")
--model: Type of Cellpose model to use (default: "cyto3")
--min-intensity: Minimum average intensity for valid regions (default: 10)
--grid-size: Size of the grid for snapping pattern centers (default: 20)
--threshold: Threshold value for nuclei extraction (optional, if not provided uses Otsu's method)
```

#### Extract Command

Extract valid frames for each contour based on analysis results:

```bash
# Extract nuclei frames
python -m cell_counter.cli.extract --patterns <patterns_file> --nuclei <nuclei_file> --time-series <time_series_file> --output <output_folder>

# Extract cytoplasm frames
python -m cell_counter.cli.extract --patterns <patterns_file> --cyto <cyto_file> --time-series <time_series_file> --output <output_folder>

# Extract both nuclei and cytoplasm frames
python -m cell_counter.cli.extract --patterns <patterns_file> --nuclei <nuclei_file> --cyto <cyto_file> --time-series <time_series_file> --output <output_folder>
```

Optional arguments:
- `--min-frames`: Minimum number of valid frames required (default: 20)
- `--grid-size`: Size of the grid for snapping pattern centers (default: 20). A larger value creates a coarser grid, while a smaller value creates a finer grid.

Example:
```bash
# Extract nuclei frames with custom parameters
python -m cell_counter.cli.extract --patterns /path/to/patterns.tif --nuclei /path/to/nuclei.tif --time-series results.json --output extracted_frames --min-frames 20 --grid-size 25

# Extract cytoplasm frames with custom parameters
python -m cell_counter.cli.extract --patterns /path/to/patterns.tif --cyto /path/to/cyto.tif --time-series results.json --output extracted_frames --min-frames 20 --grid-size 25
```

The extract command creates a directory structure containing valid frames that meet the specified criteria:
```
output_folder/
  nuclei/
    contour_000.tif
    contour_001.tif
    ...
  cyto/
    contour_000.tif
    contour_001.tif
    ...
```

Where:
- `nuclei/` and `cyto/` directories contain the extracted frames for each type
- `contour_XXX.tif`: TIFF stack containing valid frames for that contour
- Each TIFF stack contains frames that meet the validity criteria (correct nuclei count, sufficient intensity, etc.)

### Test Cellpose on Single Contour

```bash
python -m cell_counter.cli.test --patterns /path/to/patterns.tif --nuclei /path/to/nuclei.tif --frame 4 --contour 2 --show-plot --threshold 128
```

This command will:
1. Load the specified frame from the nuclei TIFF file
2. Extract the specified contour region
3. Run Cellpose on that region
4. Count the detected nuclei
5. Show a segmentation plot (if --show-plot is specified)

Additional options:
- `--diameter`: Expected cell diameter in pixels (default: 15)
- `--channels`: Channel indices for cellpose (default: "0,0")
- `--model-type`: Type of cellpose model to use (default: "cyto3")
- `--use-gpu`: Whether to use GPU for cellpose
- `--grid-size`: Size of the grid for snapping pattern centers (default: 20)
- `--threshold`: Threshold value for nuclei extraction (optional, if not provided uses Otsu's method)

## Features

- Process both nuclei and cytoplasm data
- Extract regions of interest using contours
- Filter out contours that touch or are too close to image edges (5-pixel margin)
- Support for processing specific ranges of frames and contours
- Normalize image intensity for better visualization
- Time series analysis with Cellpose integration for nuclei counting
- Flexible extraction of valid frames based on analysis results
- Configurable grid-based pattern center alignment for improved accuracy
- Save extracted frames as TIFF stacks for each contour
- Comprehensive metadata tracking in time series analysis

## Output

### Time Series Output
The time series command generates a JSON file containing:
- Analysis results for each contour
- Frame-by-frame nuclei counts
- Validity status for each frame
- Average intensities and other metrics

Example time series output structure:
```
{
  "metadata": {
    "patterns_path": "/path/to/patterns.tif",
    "nuclei_path": "/path/to/nuclei.tif",
    "wanted_nuclei": 3,
    "use_cellpose": true,
    "use_gpu": true,
    "diameter": 15,
    "channels": [0, 0],
    "model_type": "cyto3",
    "total_contours": 2,
    "total_frames": 6
  },
  "time_lapse": {
    "0": [0, 1, 2],
    "1": [3, 4, 5]
  }
}
```

### Extract Output
The extract command creates a directory structure containing valid frames that meet the specified criteria:
```
output_folder/
  nuclei/
    contour_000.tif
    contour_001.tif
    ...
  cyto/
    contour_000.tif
    contour_001.tif
    ...
```

Where:
- `nuclei/` and `cyto/` directories contain the extracted frames for each type
- `contour_XXX.tif`: TIFF stack containing valid frames for that contour
- Each TIFF stack contains frames that meet the validity criteria (correct nuclei count, sufficient intensity, etc.)

## Development

This project uses:
- `black`